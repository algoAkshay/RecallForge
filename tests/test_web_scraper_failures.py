import asyncio
import importlib
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


def _install_import_stubs():
    streamlit = types.ModuleType("streamlit")
    streamlit.session_state = {}
    sys.modules["streamlit"] = streamlit

    ddgs = types.ModuleType("ddgs")
    ddgs.DDGS = object
    sys.modules.setdefault("ddgs", ddgs)

    trafilatura = types.ModuleType("trafilatura")
    trafilatura.fetch_url = lambda url: None
    trafilatura.extract = lambda content, favor_recall=True: None
    sys.modules.setdefault("trafilatura", trafilatura)

    tools_module = types.ModuleType("langchain_core.tools")

    def tool(*_args, **_kwargs):
        return lambda function: function

    tools_module.tool = tool
    core_module = types.ModuleType("langchain_core")
    sys.modules.setdefault("langchain_core", core_module)
    sys.modules.setdefault("langchain_core.tools", tools_module)

    data_module = types.ModuleType("tools.data")

    async def save_embeddings(_chunks):
        return None

    data_module.save_embeddings = save_embeddings

    def generate_doc_hash(content):
        normalized = " ".join(content.replace("\r\n", "\n").replace("\r", "\n").split())
        return __import__("hashlib").sha256(normalized.encode("utf-8")).hexdigest()

    data_module.generate_doc_hash = generate_doc_hash
    sys.modules.setdefault("tools.data", data_module)

    preprocess_module = types.ModuleType("tools.preprocess")
    preprocess_module.CustomDocumentLoader = object

    async def split_text(_documents, *_args):
        return []

    preprocess_module.split_text = split_text
    sys.modules.setdefault("tools.preprocess", preprocess_module)


_install_import_stubs()
web_scraper = importlib.import_module("tools.web_scraper")


class WebScraperFailureTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        web_scraper.cached_search_content.cache_clear()
        web_scraper.cached_fetch_content.cache_clear()
        web_scraper.st.session_state.clear()

    async def test_both_search_channels_succeed(self):
        def search(_query, source):
            return [{"title": source, "link": f"https://{source}.test", "snippet": source}]

        with patch.object(web_scraper, "cached_search_content", side_effect=search), patch.object(
            web_scraper, "_visit_url", new=AsyncMock(return_value="STATUS: SUCCESS\n\nContent")
        ):
            result = await web_scraper._fetch_sites("topic")

        self.assertTrue(result.startswith("STATUS: SUCCESS"))
        self.assertIn("https://text.test", result)
        self.assertIn("https://news.test", result)
        self.assertNotIn("Warnings:", result)

    async def test_text_success_news_failure_preserves_text(self):
        def search(_query, source):
            if source == "news":
                raise TimeoutError("request timed out")
            return [{"title": "text", "link": "https://text.test", "snippet": "usable"}]

        with patch.object(web_scraper, "cached_search_content", side_effect=search), patch.object(
            web_scraper, "_visit_url", new=AsyncMock(return_value="STATUS: SUCCESS\n\nContent")
        ):
            result = await web_scraper._fetch_sites("topic")

        self.assertTrue(result.startswith("STATUS: PARTIAL_FAILURE"))
        self.assertIn("https://text.test", result)
        self.assertIn("DuckDuckGo news search failed", result)

    async def test_text_failure_news_success_preserves_news(self):
        def search(_query, source):
            if source == "text":
                raise TimeoutError("request timed out")
            return [{"title": "news", "link": "https://news.test", "snippet": "usable"}]

        with patch.object(web_scraper, "cached_search_content", side_effect=search), patch.object(
            web_scraper, "_visit_url", new=AsyncMock(return_value="STATUS: SUCCESS\n\nContent")
        ):
            result = await web_scraper._fetch_sites("topic")

        self.assertTrue(result.startswith("STATUS: PARTIAL_FAILURE"))
        self.assertIn("https://news.test", result)
        self.assertIn("DuckDuckGo text search failed", result)

    async def test_successful_zero_result_search_is_explicit(self):
        with patch.object(web_scraper, "cached_search_content", return_value=[]):
            result = await web_scraper._fetch_sites("topic")

        self.assertTrue(result.startswith("STATUS: SUCCESS_EMPTY"))

    async def test_both_search_channels_fail_explicitly(self):
        with patch.object(web_scraper, "cached_search_content", side_effect=ConnectionError("offline")):
            result = await web_scraper._fetch_sites("topic")

        self.assertTrue(result.startswith("STATUS: FAILURE"))
        self.assertIn("both text and news searches failed", result)

    async def test_one_url_failure_preserves_other_results(self):
        def search(_query, source):
            return (
                [{"title": "A", "link": "https://a.test", "snippet": "a"}]
                if source == "text"
                else [
                    {"title": "B", "link": "https://b.test", "snippet": "b"},
                    {"title": "C", "link": "https://c.test", "snippet": "c"},
                ]
            )

        async def visit(url, _semaphore=None):
            link = url["link"] if isinstance(url, dict) else url
            if link == "https://b.test":
                return "STATUS: FAILURE\n\nFETCH_FAILED"
            return f"STATUS: SUCCESS\n\nContent from {link}"

        with patch.object(web_scraper, "cached_search_content", side_effect=search), patch.object(
            web_scraper, "_visit_url", side_effect=visit
        ):
            result = await web_scraper._fetch_sites("topic")

        self.assertIn("Content from https://a.test", result)
        self.assertIn("Content from https://c.test", result)
        self.assertIn("URL processing did not fully succeed for https://b.test", result)

    async def test_scrape_failure_does_not_attempt_indexing(self):
        with patch.object(web_scraper, "cached_fetch_content", side_effect=ValueError("no readable text")), patch.object(
            web_scraper, "process_and_save", new=AsyncMock()
        ) as save:
            result = await web_scraper._visit_url("https://bad.test")

        self.assertTrue(result.startswith("STATUS: FAILURE"))
        self.assertIn("FETCH_FAILED", result)
        save.assert_not_awaited()

    async def test_indexing_failure_preserves_extracted_content(self):
        with patch.object(web_scraper, "cached_fetch_content", return_value="article text"), patch.object(
            web_scraper, "process_and_save", new=AsyncMock(side_effect=RuntimeError("embedding operation failed"))
        ):
            result = await web_scraper._visit_url("https://example.test")

        self.assertTrue(result.startswith("STATUS: PARTIAL_FAILURE"))
        self.assertIn("FETCH_SUCCESS", result)
        self.assertIn("INDEXING_FAILED", result)
        self.assertIn("article text", result)

    async def test_scrape_and_indexing_success(self):
        with patch.object(web_scraper, "cached_fetch_content", return_value="article text"), patch.object(
            web_scraper, "process_and_save", new=AsyncMock(return_value=("hash", False))
        ):
            result = await web_scraper._visit_url("https://example.test")

        self.assertTrue(result.startswith("STATUS: SUCCESS"))
        self.assertIn("FETCH_SUCCESS", result)
        self.assertIn("INDEXING_SUCCESS", result)

    def _loader(self, seen_queries):
        class Loader:
            def __init__(self, query):
                seen_queries.append(query)
                self.query = query

            async def lazy_load(self):
                yield {"metadata": {"source": self.query["link"]}}

        return Loader

    async def test_sequential_duplicate_skips_ingestion_work(self):
        seen_queries = []
        split = AsyncMock(return_value=["chunk"])
        save = AsyncMock()
        with patch.object(web_scraper, "cached_fetch_content", return_value="content X"), patch.object(
            web_scraper, "CustomDocumentLoader", self._loader(seen_queries)
        ), patch.object(web_scraper, "split_text", split), patch.object(web_scraper, "save_embeddings", save):
            first = await web_scraper._visit_url("https://a.test")
            second = await web_scraper._visit_url("https://a.test")

        self.assertIn("INDEXING_SUCCESS", first)
        self.assertIn("INDEXING_SKIPPED_DUPLICATE", second)
        self.assertEqual(split.await_count, 1)
        self.assertEqual(save.await_count, 1)
        self.assertIn("document_content_hash", seen_queries[0])

    async def test_identical_content_at_different_urls_is_indexed_once(self):
        split = AsyncMock(return_value=["chunk"])
        save = AsyncMock()
        with patch.object(web_scraper, "cached_fetch_content", return_value="content X"), patch.object(
            web_scraper, "CustomDocumentLoader", self._loader([])
        ), patch.object(web_scraper, "split_text", split), patch.object(web_scraper, "save_embeddings", save):
            first = await web_scraper._visit_url("https://a.test")
            second = await web_scraper._visit_url("https://b.test")

        self.assertIn("INDEXING_SUCCESS", first)
        self.assertIn("INDEXING_SKIPPED_DUPLICATE", second)
        self.assertEqual(save.await_count, 1)

    async def test_changed_content_at_same_url_is_indexed_twice(self):
        split = AsyncMock(return_value=["chunk"])
        save = AsyncMock()
        with patch.object(web_scraper, "cached_fetch_content", side_effect=["content X", "content Y"]), patch.object(
            web_scraper, "CustomDocumentLoader", self._loader([])
        ), patch.object(web_scraper, "split_text", split), patch.object(web_scraper, "save_embeddings", save):
            first = await web_scraper._visit_url("https://a.test")
            second = await web_scraper._visit_url("https://a.test")

        self.assertIn("INDEXING_SUCCESS", first)
        self.assertIn("INDEXING_SUCCESS", second)
        self.assertEqual(save.await_count, 2)
        self.assertEqual(len(web_scraper.st.session_state[web_scraper._INDEXED_HASHES_KEY]), 2)

    def test_normalization_produces_one_content_identity(self):
        self.assertEqual(
            web_scraper.generate_doc_hash("A\r\n useful   page  "),
            web_scraper.generate_doc_hash("A useful page"),
        )

    async def test_document_hash_and_source_reach_indexed_chunks(self):
        indexed_chunks = []

        class Loader:
            def __init__(self, query):
                self.query = query

            async def lazy_load(self):
                yield types.SimpleNamespace(
                    metadata={
                        "link": self.query["link"],
                        "document_content_hash": self.query["document_content_hash"],
                    }
                )

        async def split(documents, *_args):
            return documents

        async def save(chunks):
            indexed_chunks.extend(chunks)

        with patch.object(web_scraper, "cached_fetch_content", return_value="content X"), patch.object(
            web_scraper, "CustomDocumentLoader", Loader
        ), patch.object(web_scraper, "split_text", split), patch.object(web_scraper, "save_embeddings", save):
            await web_scraper._visit_url("https://source.test")

        self.assertEqual(indexed_chunks[0].metadata["link"], "https://source.test")
        self.assertEqual(
            indexed_chunks[0].metadata["document_content_hash"],
            web_scraper.generate_doc_hash("content X"),
        )

    async def test_failed_indexing_is_retryable_and_clears_inflight_state(self):
        split = AsyncMock(return_value=["chunk"])
        save = AsyncMock(side_effect=[RuntimeError("index failed"), None])
        with patch.object(web_scraper, "cached_fetch_content", return_value="content X"), patch.object(
            web_scraper, "CustomDocumentLoader", self._loader([])
        ), patch.object(web_scraper, "split_text", split), patch.object(web_scraper, "save_embeddings", save):
            first = await web_scraper._visit_url("https://a.test")
            second = await web_scraper._visit_url("https://a.test")

        self.assertIn("INDEXING_FAILED", first)
        self.assertIn("INDEXING_SUCCESS", second)
        self.assertEqual(save.await_count, 2)
        self.assertFalse(web_scraper.st.session_state.get(web_scraper._INFLIGHT_INGESTIONS_KEY, {}))

    async def test_concurrent_duplicate_ingestion_runs_once(self):
        started = asyncio.Event()
        release = asyncio.Event()
        save = AsyncMock()

        async def split(_documents, *_args):
            started.set()
            await release.wait()
            return ["chunk"]

        with patch.object(web_scraper, "cached_fetch_content", return_value="content X"), patch.object(
            web_scraper, "CustomDocumentLoader", self._loader([])
        ), patch.object(web_scraper, "split_text", split), patch.object(web_scraper, "save_embeddings", save):
            first_task = asyncio.create_task(web_scraper._visit_url("https://a.test"))
            await started.wait()
            second_task = asyncio.create_task(web_scraper._visit_url("https://b.test"))
            await asyncio.sleep(0)
            release.set()
            first, second = await asyncio.gather(first_task, second_task)

        self.assertEqual(save.await_count, 1)
        self.assertEqual(sum("INDEXING_SUCCESS" in result for result in (first, second)), 1)
        self.assertEqual(sum("INDEXING_SKIPPED_DUPLICATE" in result for result in (first, second)), 1)

    async def test_fetches_overlap_in_worker_threads(self):
        active = 0
        maximum = 0
        lock = threading.Lock()
        overlapped = threading.Event()
        release = threading.Event()

        def fetch(url):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
                if active >= 2:
                    overlapped.set()
            release.wait()
            with lock:
                active -= 1
            return f"content {url}"

        semaphore = asyncio.Semaphore(2)
        with patch.object(web_scraper, "cached_fetch_content", side_effect=fetch), patch.object(
            web_scraper, "process_and_save", new=AsyncMock(return_value=("hash", False))
        ):
            first = asyncio.create_task(web_scraper._visit_url("https://a.test", semaphore))
            second = asyncio.create_task(web_scraper._visit_url("https://b.test", semaphore))
            await asyncio.wait_for(asyncio.to_thread(overlapped.wait), timeout=1)
            release.set()
            await asyncio.gather(first, second)

        self.assertGreaterEqual(maximum, 2)

    async def test_fetch_limit_is_enforced(self):
        active = 0
        maximum = 0
        lock = threading.Lock()
        reached_limit = threading.Event()
        release = threading.Event()

        def fetch(url):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
                if active == web_scraper.MAX_CONCURRENT_FETCHES:
                    reached_limit.set()
            release.wait()
            with lock:
                active -= 1
            return f"content {url}"

        semaphore = asyncio.Semaphore(web_scraper.MAX_CONCURRENT_FETCHES)
        with patch.object(web_scraper, "cached_fetch_content", side_effect=fetch), patch.object(
            web_scraper, "process_and_save", new=AsyncMock(return_value=("hash", False))
        ):
            tasks = [
                asyncio.create_task(web_scraper._visit_url(f"https://{index}.test", semaphore))
                for index in range(web_scraper.MAX_CONCURRENT_FETCHES + 2)
            ]
            await asyncio.wait_for(asyncio.to_thread(reached_limit.wait), timeout=1)
            self.assertEqual(maximum, web_scraper.MAX_CONCURRENT_FETCHES)
            release.set()
            await asyncio.gather(*tasks)

        self.assertLessEqual(maximum, web_scraper.MAX_CONCURRENT_FETCHES)

    async def test_event_loop_progresses_while_fetch_worker_is_blocked(self):
        started = threading.Event()
        release = threading.Event()
        heartbeat = asyncio.Event()

        def fetch(_url):
            started.set()
            release.wait()
            return "content"

        async def mark_heartbeat():
            heartbeat.set()

        with patch.object(web_scraper, "cached_fetch_content", side_effect=fetch), patch.object(
            web_scraper, "process_and_save", new=AsyncMock(return_value=("hash", False))
        ):
            task = asyncio.create_task(web_scraper._visit_url("https://a.test", asyncio.Semaphore(1)))
            await asyncio.wait_for(asyncio.to_thread(started.wait), timeout=1)
            await asyncio.create_task(mark_heartbeat())
            self.assertTrue(heartbeat.is_set())
            release.set()
            await task

    async def test_worker_fetch_failure_preserves_sibling_results(self):
        def search(_query, source):
            return (
                [{"title": "A", "link": "https://a.test", "snippet": "a"}]
                if source == "text"
                else [
                    {"title": "B", "link": "https://b.test", "snippet": "b"},
                    {"title": "C", "link": "https://c.test", "snippet": "c"},
                ]
            )

        def fetch(url):
            if url == "https://b.test":
                raise ConnectionError("offline")
            return f"content {url}"

        with patch.object(web_scraper, "cached_search_content", side_effect=search), patch.object(
            web_scraper, "cached_fetch_content", side_effect=fetch
        ), patch.object(web_scraper, "process_and_save", new=AsyncMock(return_value=("hash", False))):
            result = await web_scraper._fetch_sites("topic")

        self.assertIn("content https://a.test", result)
        self.assertIn("content https://c.test", result)
        self.assertIn("URL processing did not fully succeed for https://b.test", result)

    async def test_concurrent_identical_fetches_still_index_once(self):
        fetches = 0
        split_started = asyncio.Event()
        release_split = asyncio.Event()
        save = AsyncMock()

        def fetch(_url):
            nonlocal fetches
            fetches += 1
            return "shared content"

        async def split(_documents, *_args):
            split_started.set()
            await release_split.wait()
            return ["chunk"]

        semaphore = asyncio.Semaphore(2)
        with patch.object(web_scraper, "cached_fetch_content", side_effect=fetch), patch.object(
            web_scraper, "CustomDocumentLoader", self._loader([])
        ), patch.object(web_scraper, "split_text", split), patch.object(web_scraper, "save_embeddings", save):
            first_task = asyncio.create_task(web_scraper._visit_url("https://a.test", semaphore))
            await split_started.wait()
            second_task = asyncio.create_task(web_scraper._visit_url("https://b.test", semaphore))
            await asyncio.sleep(0)
            release_split.set()
            first, second = await asyncio.gather(first_task, second_task)

        self.assertEqual(fetches, 2)
        self.assertEqual(save.await_count, 1)
        self.assertEqual(sum("INDEXING_SKIPPED_DUPLICATE" in result for result in (first, second)), 1)

    async def test_sequential_cached_url_executes_fetch_once(self):
        with patch.object(web_scraper.trafilatura, "fetch_url", return_value="html") as fetch, patch.object(
            web_scraper.trafilatura, "extract", return_value="content"
        ), patch.object(web_scraper, "process_and_save", new=AsyncMock(return_value=("hash", False))):
            await web_scraper._visit_url("https://cached.test")
            await web_scraper._visit_url("https://cached.test")

        self.assertEqual(fetch.call_count, 1)

    async def test_search_timeout_is_truthful_and_diagnostic(self):
        release = threading.Event()
        diagnostics = web_scraper.WebAcquisitionDiagnostics(started_at=0.0)

        def search(_query, _source):
            release.wait()
            return []

        with patch.object(web_scraper, "cached_search_content", side_effect=search), patch.object(
            web_scraper, "SEARCH_TIMEOUT_SECONDS", 0.02
        ):
            text, news, warnings = await web_scraper._search_sources("topic", diagnostics=diagnostics)
        release.set()

        self.assertEqual((text, news), ([], []))
        self.assertTrue(diagnostics.search_timeout)
        self.assertTrue(all("search_timeout" in warning for warning in warnings))
        self.assertGreaterEqual(diagnostics.search_seconds, 0.0)

    async def test_per_url_timeout_is_truthful_and_does_not_index_late_worker(self):
        release = threading.Event()
        diagnostics = web_scraper.WebAcquisitionDiagnostics(started_at=0.0)

        def fetch(_url):
            release.wait()
            return "late content"

        with patch.object(web_scraper, "cached_fetch_content", side_effect=fetch), patch.object(
            web_scraper, "PER_URL_TIMEOUT_SECONDS", 0.02
        ), patch.object(web_scraper, "process_and_save", new=AsyncMock()) as save:
            result = await web_scraper._visit_url("https://slow.test", diagnostics=diagnostics)
            release.set()
            await asyncio.sleep(0)

        self.assertIn("FETCH_TIMEOUT", result)
        self.assertEqual(diagnostics.fetch_attempt_count, 1)
        self.assertEqual(diagnostics.fetch_timeout_count, 1)
        save.assert_not_awaited()

    async def test_mixed_fetches_retain_evidence_after_sibling_timeout(self):
        release = threading.Event()

        def search(_query, source):
            return ([{"title": "fast", "link": "https://fast.test", "snippet": ""}] if source == "text" else [
                {"title": "slow", "link": "https://slow.test", "snippet": ""}
            ])

        def fetch(url):
            if url.endswith("slow.test"):
                release.wait()
            return f"content {url}"

        with patch.object(web_scraper, "cached_search_content", side_effect=search), patch.object(
            web_scraper, "cached_fetch_content", side_effect=fetch
        ), patch.object(web_scraper, "PER_URL_TIMEOUT_SECONDS", 0.10), patch.object(
            web_scraper, "process_and_save", new=AsyncMock(return_value=("hash", False))
        ):
            result = await web_scraper._fetch_sites("topic")
        release.set()

        self.assertIn("EVIDENCE", result)
        self.assertIn("https://fast.test", result)
        self.assertIn("FETCH_TIMEOUT", result)
        self.assertIn("fetch_success_count: 1", result)
        self.assertIn("fetch_timeout_count: 1", result)

    async def test_overall_deadline_cancels_pending_tasks_and_keeps_fast_result(self):
        cancelled = asyncio.Event()

        def search(_query, source):
            return ([{"title": "fast", "link": "https://fast.test", "snippet": ""}] if source == "text" else [
                {"title": "slow", "link": "https://slow.test", "snippet": ""}
            ])

        async def visit(row, _semaphore=None):
            if row["link"].endswith("fast.test"):
                return "STATUS: SUCCESS\n\nfast evidence"
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        with patch.object(web_scraper, "cached_search_content", side_effect=search), patch.object(
            web_scraper, "_visit_url", side_effect=visit
        ), patch.object(web_scraper, "WEB_ACQUISITION_DEADLINE_SECONDS", 0.10):
            result = await web_scraper._fetch_sites("topic")

        self.assertTrue(cancelled.is_set())
        self.assertIn("fast evidence", result)
        self.assertIn("web_acquisition_deadline", result)
        self.assertIn("acquisition_deadline_reached: true", result)

    async def test_all_fetch_timeouts_are_reported_without_fabricating_evidence(self):
        release = threading.Event()

        def search(_query, _source):
            return [{"title": "slow", "link": "https://slow.test", "snippet": ""}]

        def fetch(_url):
            release.wait()
            return "late content"

        with patch.object(web_scraper, "cached_search_content", side_effect=search), patch.object(
            web_scraper, "cached_fetch_content", side_effect=fetch
        ), patch.object(web_scraper, "PER_URL_TIMEOUT_SECONDS", 0.02), patch.object(
            web_scraper, "process_and_save", new=AsyncMock()
        ) as save:
            result = await web_scraper._fetch_sites("topic")
        release.set()

        self.assertIn("FETCH_TIMEOUT", result)
        self.assertNotIn("EVIDENCE\n", result)
        self.assertIn("fetch_success_count: 0", result)
        self.assertIn("fetch_timeout_count: 1", result)
        save.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
