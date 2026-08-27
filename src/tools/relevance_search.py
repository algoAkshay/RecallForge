from tools.data import fetch_model
from tools.provenance import evidence_from_document, format_evidence, register_evidence
from features import ENABLE_PDF_FEATURES
from langchain_core.tools import tool
import hashlib
import re
import streamlit as st

MEMORY_CANDIDATE_POOL = 18
ACTIVE_DOCUMENT_BONUS = 0.04
LEXICAL_MATCH_BONUS = 0.015

def generate_content_hash(content: str) -> str:
    """Generate SHA-256 hash for document content"""
    return hashlib.sha256(content.encode()).hexdigest()

def _lexical_tokens(text: str) -> set[str]:
    """Keep acronyms, decimal requirements, and C++-style terms for local ranking."""
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-z]+\+{1,2}|\d+(?:\.\d+)?|[A-Za-z][A-Za-z0-9]*", text)
        if len(token) > 1 or token.isdigit()
    }


def _filename_key(filename: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", filename.casefold().removesuffix(".pdf")))


def _referenced_active_hashes(query: str, active_documents: list[dict]) -> set[str]:
    normalized_query = _filename_key(query)
    explicit_reference = any(phrase in query.casefold() for phrase in (
        "according to", "in this pdf", "in this document", "in the uploaded document", "from this document",
    ))
    filename_matches = set()
    query_terms = set(normalized_query.split())
    for item in active_documents:
        filename_terms = set(_filename_key(str(item.get("filename", ""))).split())
        # Two matching filename terms tolerates harmless punctuation and a
        # one-character spelling variation without hardcoding any document.
        if filename_terms and len(filename_terms & query_terms) >= min(2, len(filename_terms)):
            filename_matches.add(item["document_hash"])
    return filename_matches or ({item["document_hash"] for item in active_documents} if explicit_reference else set())


def is_document_context_query(query: str, active_documents: list[dict] | None = None) -> bool:
    """Recognize generic uploaded-document references and active filenames."""
    normalized = " ".join(query.casefold().split())
    generic_phrases = (
        "uploaded pdf", "uploaded document", "this pdf", "this document", "the pdf", "the document",
        "according to the pdf", "according to the document", "from the pdf", "from the document",
        "in the pdf", "in the document",
    )
    return any(phrase in normalized for phrase in generic_phrases) or bool(_referenced_active_hashes(query, active_documents or []))


def _document_overview_query(query: str) -> bool:
    normalized = query.casefold()
    return any(phrase in normalized for phrase in (
        "what have you read", "what have u read", "summarize", "summary", "what is this document about",
        "what does this pdf contain", "what does the pdf contain", "what is in the uploaded pdf",
    ))


def _representative_document_candidates(db, document_hashes: set[str], limit: int = 5):
    """Return bounded page-spread samples for an explicit document overview."""
    documents = [
        document for document in getattr(getattr(db, "docstore", None), "_dict", {}).values()
        if (getattr(document, "metadata", {}) or {}).get("document_hash") in document_hashes
    ]
    documents.sort(key=lambda document: ((getattr(document, "metadata", {}) or {}).get("page_number", 0), (getattr(document, "metadata", {}) or {}).get("start_index", 0)))
    if len(documents) <= limit:
        return [(document, 0.0) for document in documents]
    indexes = sorted({round(index * (len(documents) - 1) / (limit - 1)) for index in range(limit)})
    return [(documents[index], 0.0) for index in indexes]


async def retrieve_active_document_candidates(query: str, active_documents: list[dict], top_k: int = 6):
    """Search active upload hashes first, never after global historical evidence."""
    active_hashes = {item["document_hash"] for item in active_documents if item.get("document_hash")}
    if not active_hashes:
        return []
    db = await fetch_model()
    referenced_hashes = _referenced_active_hashes(query, active_documents)
    scope_hashes = referenced_hashes or active_hashes
    if _document_overview_query(query) and is_document_context_query(query, active_documents):
        return _representative_document_candidates(db, scope_hashes)
    document_count = len(getattr(getattr(db, "docstore", None), "_dict", {}))
    candidates = await db.asimilarity_search_with_score(
        query,
        k=max(MEMORY_CANDIDATE_POOL, top_k),
        fetch_k=min(max(document_count, MEMORY_CANDIDATE_POOL), 128),
        filter=lambda metadata: metadata.get("document_hash") in scope_hashes,
    )
    return rank_memory_candidates(query, candidates, active_documents)[:top_k]


def rank_memory_candidates(query: str, candidates, active_documents: list[dict] | None = None):
    """Rerank only FAISS-returned candidates; original L2 scores remain intact."""
    active_documents = active_documents or []
    active_hashes = {item["document_hash"] for item in active_documents if item.get("document_hash")}
    referenced_hashes = _referenced_active_hashes(query, active_documents)
    query_tokens = _lexical_tokens(query)
    ranked = []
    for document, score in candidates:
        metadata = getattr(document, "metadata", {}) or {}
        document_hash = metadata.get("document_hash")
        if referenced_hashes and document_hash not in referenced_hashes:
            continue
        overlap = len(query_tokens & _lexical_tokens(getattr(document, "page_content", "")))
        bonus = min(overlap * LEXICAL_MATCH_BONUS, 0.06)
        if document_hash in active_hashes:
            bonus += ACTIVE_DOCUMENT_BONUS
        ranked.append((float(score) - bonus, document, score))
    ranked.sort(key=lambda item: item[0])
    return [(document, score) for _, document, score in ranked]


async def retrieve_memory_candidates(query: str, top_k: int = 6, active_documents: list[dict] | None = None):
    """Return global FAISS candidates for normal MEMORY/WEB fallback behavior."""
    db = await fetch_model()
    options = {} if ENABLE_PDF_FEATURES else {"filter": lambda metadata: metadata.get("origin") != "upload"}
    raw_candidates = await db.asimilarity_search_with_score(query, k=max(MEMORY_CANDIDATE_POOL, top_k), **options)
    return rank_memory_candidates(query, raw_candidates, active_documents)[:top_k]

@tool("DBSearch",parse_docstring=True)
async def fetch_information(query: str, top_k: int=6): 
    """
    Search database for relevant documents. Contents of websearches are cached in the database, so that they could be accessed through this tool

    Args:
        query (str): The search query text to find relevant documents.
        top_k (int): The maximum number of documents to return.

    Returns:
        str: A concatenated string containing the unique relevant documents, up to the specified limit of `top_k`.
    """
    db = await fetch_model()

    raw_results = await db.asimilarity_search_with_relevance_scores(query, k=top_k*3)

    # Filter by score and deduplicate
    seen_hashes = set()
    unique_results = []

    for doc, score in raw_results:
        content_hash = generate_content_hash(doc.page_content)
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_results.append(doc)
            
        if len(unique_results) >= top_k:
            break
            
    if not unique_results:
        return "No relevant information found matching the query\n"

    sources = register_evidence([evidence_from_document(doc, "memory") for doc in unique_results])
    return "EVIDENCE\n\n" + format_evidence(sources)
