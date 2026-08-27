import asyncio
import contextvars
import logging
import time
from dataclasses import dataclass
from functools import lru_cache

import trafilatura
import streamlit as st
from ddgs import DDGS
from langchain_core.tools import tool

from tools.data import generate_doc_hash, save_embeddings
from tools.preprocess import CustomDocumentLoader, split_text
from tools.provenance import evidence_from_mapping, format_evidence, register_evidence


logger = logging.getLogger(__name__)
_INDEXED_HASHES_KEY = "indexed_content_hashes"
_INFLIGHT_INGESTIONS_KEY = "inflight_content_ingestions"
SEARCH_TIMEOUT_SECONDS = 6
PER_URL_TIMEOUT_SECONDS = 10
WEB_ACQUISITION_DEADLINE_SECONDS = 25
MAX_CONCURRENT_FETCHES = 4
_CURRENT_DIAGNOSTICS: contextvars.ContextVar["WebAcquisitionDiagnostics | None"] = contextvars.ContextVar(
    "web_acquisition_diagnostics", default=None
)


class _OperationTimeout(Exception):
    """Internal marker that distinguishes our deadline from a source failure."""


@dataclass
class WebAcquisitionDiagnostics:
    """Request-local WEB acquisition measurements and outcome counts."""

    started_at: float
    search_seconds: float = 0.0
    fetch_extract_seconds: float = 0.0
    embedding_index_seconds: float = 0.0
    total_web_acquisition_seconds: float = 0.0
    search_result_count: int = 0
    fetch_attempt_count: int = 0
    fetch_success_count: int = 0
    fetch_timeout_count: int = 0
    fetch_failure_count: int = 0
    search_timeout: bool = False
    acquisition_deadline_reached: bool = False

    def lines(self) -> list[str]:
        """Return stable, agent-readable diagnostics without exposing internals."""
        return [
            f"search_seconds: {self.search_seconds:.3f}",
            f"fetch_extract_seconds: {self.fetch_extract_seconds:.3f}",
            f"embedding_index_seconds: {self.embedding_index_seconds:.3f}",
            f"total_web_acquisition_seconds: {self.total_web_acquisition_seconds:.3f}",
            f"search_result_count: {self.search_result_count}",
            f"fetch_attempt_count: {self.fetch_attempt_count}",
            f"fetch_success_count: {self.fetch_success_count}",
            f"fetch_timeout_count: {self.fetch_timeout_count}",
            f"fetch_failure_count: {self.fetch_failure_count}",
            f"search_timeout: {str(self.search_timeout).lower()}",
            f"acquisition_deadline_reached: {str(self.acquisition_deadline_reached).lower()}",
        ]


def _error_message(error: Exception) -> str:
    """Return concise error context without exposing a traceback."""
    message = " ".join(str(error).split())
    return message[:200] if message else error.__class__.__name__


def _status(status: str, body: str, warnings: list[str] | None = None) -> str:
    """Serialize tool state into a deterministic, agent-readable response."""
    response = f"STATUS: {status}\n\n{body}"
    if warnings:
        response += "\n\nWarnings:\n" + "\n".join(f"- {warning}" for warning in warnings)
    return response


def _remaining_seconds(deadline: float) -> float:
    """Return the non-negative time remaining in a monotonic deadline."""
    return max(0.0, deadline - time.perf_counter())


def _with_diagnostics(response: str, diagnostics: WebAcquisitionDiagnostics) -> str:
    """Finalize and attach request-local diagnostics to a WEB acquisition result."""
    diagnostics.total_web_acquisition_seconds = time.perf_counter() - diagnostics.started_at
    logger.info("WEB acquisition diagnostics: %s", "; ".join(diagnostics.lines()))
    return response + "\n\nDiagnostics:\n" + "\n".join(diagnostics.lines())


def _trafilatura_timeout_config():
    """Build Trafilatura's normal config with a bounded download timeout."""
    try:
        # Import lazily so offline test doubles need only expose fetch/extract.
        from trafilatura.settings import use_config
    except ModuleNotFoundError:
        return None
    config = use_config()
    config.set("DEFAULT", "DOWNLOAD_TIMEOUT", str(PER_URL_TIMEOUT_SECONDS))
    return config


@lru_cache(maxsize=128)
def cached_fetch_content(url: str) -> str:
    """Fetch and extract usable webpage text, raising when either step fails."""
    # Trafilatura's network downloader observes DOWNLOAD_TIMEOUT.  This is a
    # real transport timeout, not merely cancellation of its worker thread.
    config = _trafilatura_timeout_config()
    content = trafilatura.fetch_url(url, config=config) if config is not None else trafilatura.fetch_url(url)
    if not content:
        raise ValueError("No page content was returned.")

    extracted = trafilatura.extract(content, favor_recall=True)
    if not extracted or not extracted.strip():
        raise ValueError("No readable text could be extracted from the page.")
    return extracted


@lru_cache(maxsize=128)
def cached_search_content(query: str, source: str) -> list[dict[str, str]]:
    """Run one DuckDuckGo source and raise if that source fails."""
    clean_q = str(query).strip().strip("`'\"\n\r\t ")
    results = []
    # DDGS passes this timeout to its HTTP client, so a cancelled worker is not
    # left performing an unbounded network request.
    with DDGS(timeout=SEARCH_TIMEOUT_SECONDS) as ddgs:
        if source == "news":
            items = list(ddgs.news(clean_q, max_results=6))
            link_key = "url"
        else:
            items = list(ddgs.text(clean_q, max_results=6))
            link_key = "href"

    for item in items:
        results.append(
            {
                "title": item.get("title", ""),
                "link": item.get(link_key, ""),
                "snippet": item.get("body", ""),
            }
        )
    return results


async def _search_sources(
    query: str,
    force_fresh: bool = False,
    *,
    deadline: float | None = None,
    diagnostics: WebAcquisitionDiagnostics | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    """Run independent text/news searches while preserving a successful sibling."""
    diagnostics = diagnostics or _CURRENT_DIAGNOSTICS.get()
    timeout = SEARCH_TIMEOUT_SECONDS if deadline is None else min(SEARCH_TIMEOUT_SECONDS, _remaining_seconds(deadline))
    if timeout <= 0:
        if diagnostics:
            diagnostics.search_timeout = True
            diagnostics.acquisition_deadline_reached = True
        return [], [], ["DuckDuckGo text search_timeout", "DuckDuckGo news search_timeout"]

    async def search(source: str):
        operation = cached_search_content.__wrapped__ if force_fresh else cached_search_content

        async def run_in_worker():
            try:
                return True, await asyncio.to_thread(operation, query, source)
            except Exception as error:
                return False, error

        try:
            succeeded, outcome = await asyncio.wait_for(run_in_worker(), timeout=timeout)
            return outcome if succeeded else outcome
        except asyncio.TimeoutError:
            return _OperationTimeout("search_timeout")

    started_at = time.perf_counter()
    outcomes = await asyncio.gather(search("text"), search("news"))
    if diagnostics:
        diagnostics.search_seconds += time.perf_counter() - started_at

    results: list[list[dict[str, str]]] = [[], []]
    warnings = []
    for source, outcome, index in (("text", outcomes[0], 0), ("news", outcomes[1], 1)):
        if isinstance(outcome, Exception):
            logger.warning("DuckDuckGo %s search failed: %s", source, _error_message(outcome))
            if isinstance(outcome, _OperationTimeout):
                warnings.append(f"DuckDuckGo {source} search_timeout")
                if diagnostics:
                    diagnostics.search_timeout = True
            else:
                warnings.append(f"DuckDuckGo {source} search failed: {_error_message(outcome)}")
        else:
            results[index] = outcome
            if diagnostics:
                diagnostics.search_result_count += len(outcome)
    return results[0], results[1], warnings


def _combined_results(
    text_results: list[dict[str, str]], news_results: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Combine result sources without mutating cached result dictionaries."""
    unique_results = {}
    for item in news_results + text_results:
        link = item.get("link")
        if link and link not in unique_results:
            unique_results[link] = dict(item)
    return list(unique_results.values())


async def _visit_url(
    query: str | dict,
    fetch_semaphore: asyncio.Semaphore | None = None,
    force_fresh: bool = False,
    *,
    diagnostics: WebAcquisitionDiagnostics | None = None,
) -> str:
    """Fetch a page, then report extraction and indexing outcomes separately."""
    if isinstance(query, str):
        url = query.strip().strip("`'\"\n\r\t ")
        query_dict = {"link": url}
    elif isinstance(query, dict):
        url = str(query.get("link", "")).strip()
        query_dict = dict(query)
        query_dict["link"] = url
    else:
        return _status("FAILURE", "FETCH_FAILED\nReason: URL input must be a string or mapping.")

    if not url:
        return _status("FAILURE", "FETCH_FAILED\nReason: No URL was provided.")

    diagnostics = diagnostics or _CURRENT_DIAGNOSTICS.get()
    operation = cached_fetch_content.__wrapped__ if force_fresh else cached_fetch_content

    async def fetch() -> str:
        if diagnostics:
            diagnostics.fetch_attempt_count += 1
        started_at = time.perf_counter()
        try:
            return await asyncio.wait_for(asyncio.to_thread(operation, url), timeout=PER_URL_TIMEOUT_SECONDS)
        finally:
            if diagnostics:
                diagnostics.fetch_extract_seconds += time.perf_counter() - started_at

    try:
        if fetch_semaphore is None:
            content = await fetch()
        else:
            async with fetch_semaphore:
                content = await fetch()
    except asyncio.TimeoutError:
        if diagnostics:
            diagnostics.fetch_timeout_count += 1
        logger.warning("Page fetch/extraction timed out for %s", url)
        return _status("FAILURE", f"FETCH_TIMEOUT\nURL: {url}\nReason: per_url_timeout")
    except Exception as error:
        if diagnostics:
            diagnostics.fetch_failure_count += 1
        logger.warning("Page fetch/extraction failed for %s: %s", url, _error_message(error))
        return _status(
            "FAILURE",
            f"FETCH_FAILED\nURL: {url}\nReason: {_error_message(error)}",
        )

    if diagnostics:
        diagnostics.fetch_success_count += 1
    query_dict["content"] = content
    indexing_started_at = time.perf_counter()
    try:
        _, duplicate = await process_and_save(query_dict)
    except Exception as error:
        if diagnostics:
            diagnostics.embedding_index_seconds += time.perf_counter() - indexing_started_at
        logger.exception("Indexing failed after extraction for %s", url)
        return _status(
            "PARTIAL_FAILURE",
            f"FETCH_SUCCESS\nINDEXING_FAILED\nURL: {url}\n\nContent:\n{content}",
            [f"Indexing failed after successful extraction: {_error_message(error)}"],
        )

    if diagnostics:
        diagnostics.embedding_index_seconds += time.perf_counter() - indexing_started_at

    indexing_status = "INDEXING_SKIPPED_DUPLICATE" if duplicate else "INDEXING_SUCCESS"
    sources = register_evidence([evidence_from_mapping(query_dict, "web")])
    return _status("SUCCESS", f"FETCH_SUCCESS\n{indexing_status}\n\nEVIDENCE\n{format_evidence(sources)}")


async def _fetch_sites(query: str, force_fresh: bool = False) -> str:
    """Search sources and visit their URLs without hiding partial failures."""
    clean_q = str(query).strip().strip("`'\"\n\r\t ")
    diagnostics = WebAcquisitionDiagnostics(started_at=time.perf_counter())
    deadline = diagnostics.started_at + WEB_ACQUISITION_DEADLINE_SECONDS
    token = _CURRENT_DIAGNOSTICS.set(diagnostics)
    try:
        text_results, news_results, search_warnings = await _search_sources(clean_q, force_fresh, deadline=deadline)

        if len(search_warnings) == 2:
            return _with_diagnostics(_status(
                "FAILURE",
                "Search could not be completed because both text and news searches failed.",
                search_warnings,
            ), diagnostics)

        results = _combined_results(text_results, news_results)
        if not results:
            if search_warnings:
                return _with_diagnostics(_status(
                    "PARTIAL_FAILURE",
                    "Search completed with zero results from the available source.",
                    search_warnings,
                ), diagnostics)
            return _with_diagnostics(_status("SUCCESS_EMPTY", "Search completed successfully but returned zero results."), diagnostics)

        fetch_semaphore = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)
        rows = results[:8]
        tasks = [
            asyncio.create_task(_visit_url(row, fetch_semaphore) if not force_fresh else _visit_url(row, fetch_semaphore, True))
            for row in rows
        ]
        done, pending = await asyncio.wait(tasks, timeout=_remaining_seconds(deadline))
        timed_out_tasks = set(pending)
        if pending:
            diagnostics.acquisition_deadline_reached = True
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        warnings = list(search_warnings)
        rendered_results = []
        for row, task in zip(rows, tasks):
            if task in timed_out_tasks:
                diagnostics.fetch_timeout_count += 1
                page_result = _status(
                    "FAILURE",
                    f"FETCH_TIMEOUT\nURL: {row['link']}\nReason: web_acquisition_deadline",
                )
                warnings.append(f"URL processing timed out for {row['link']}.")
            else:
                outcome = task.result()
                if isinstance(outcome, Exception):
                    logger.error("Unexpected URL processing failure for %s: %s", row["link"], _error_message(outcome))
                    page_result = _status(
                        "FAILURE",
                        f"FETCH_FAILED\nURL: {row['link']}\nReason: {_error_message(outcome)}",
                    )
                    warnings.append(f"URL processing failed for {row['link']}: {_error_message(outcome)}")
                else:
                    page_result = outcome
                    if not outcome.startswith("STATUS: SUCCESS"):
                        warnings.append(f"URL processing did not fully succeed for {row['link']}.")

            rendered_results.append(
                f"Title: {row.get('title', '')}\nLink: {row['link']}\n"
                f"Snippet: {row.get('snippet', '')}\n\n{page_result}"
            )

        status = "PARTIAL_FAILURE" if warnings else "SUCCESS"
        return _with_diagnostics(_status(status, "Results:\n\n" + "\n\n---\n\n".join(rendered_results), warnings), diagnostics)
    finally:
        _CURRENT_DIAGNOSTICS.reset(token)


@tool("Search", parse_docstring=True)
async def fetch_sites(query: str) -> str:
    """Search DuckDuckGo text and news results, preserving partial failures.

    Args:
        query: The search term or question to query.

    Returns:
        An explicit status, result data, and any source or URL-processing warnings.
    """
    return await _fetch_sites(query)


@tool("OpenLink", parse_docstring=True)
async def visit(query: str) -> str:
    """Fetch a URL and report extraction and FAISS-indexing status separately.

    Args:
        query: URL string or mapping containing a ``link`` key.

    Returns:
        An explicit status and extracted text when available.
    """
    return await _visit_url(query)


async def _claim_ingestion(content_hash: str) -> tuple[bool, asyncio.Future | None]:
    """Claim one content hash or wait for an in-flight owner in this session."""
    state = st.session_state
    indexed_hashes = state.setdefault(_INDEXED_HASHES_KEY, set())
    inflight = state.setdefault(_INFLIGHT_INGESTIONS_KEY, {})
    while True:
        if content_hash in indexed_hashes:
            return False, None
        completion = inflight.get(content_hash)
        if completion is None:
            completion = asyncio.get_running_loop().create_future()
            inflight[content_hash] = completion
            return True, completion
        await asyncio.shield(completion)


def _finish_ingestion(content_hash: str, completion: asyncio.Future, succeeded: bool) -> None:
    """Publish completion after FAISS success and always remove in-flight state."""
    state = st.session_state
    if succeeded:
        state[_INDEXED_HASHES_KEY].add(content_hash)
    if not completion.done():
        completion.set_result(succeeded)
    if state[_INFLIGHT_INGESTIONS_KEY].get(content_hash) is completion:
        del state[_INFLIGHT_INGESTIONS_KEY][content_hash]


async def process_and_save(query: dict) -> tuple[str, bool]:
    """Index new document content once per session and report duplicate reuse."""
    content_hash = generate_doc_hash(query["content"])
    owns_ingestion, completion = await _claim_ingestion(content_hash)
    if not owns_ingestion:
        return content_hash, True

    query = dict(query)
    query["document_content_hash"] = content_hash
    documents = []
    try:
        loader = CustomDocumentLoader(query)
        async for doc in loader.lazy_load():
            documents.append(doc)
        chunks = await split_text(documents, 2048, 512)
        if not chunks:
            raise ValueError("No document chunks were produced for indexing.")
        for index in range(0, len(chunks), 10):
            await save_embeddings(chunks[index : index + 10])
    except BaseException:
        _finish_ingestion(content_hash, completion, succeeded=False)
        raise

    _finish_ingestion(content_hash, completion, succeeded=True)
    return content_hash, False
