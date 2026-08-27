"""Local PDF ingestion into RecallForge's existing persistent FAISS memory."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from documents.chunking import PDF_INGESTION_VERSION, chunk_parsed_pdf
from documents.registry import IndexedDocument, activate_indexed_document
from tools.data import persist_vector_store
from documents.pdf_parser import MAX_PDFS_PER_UPLOAD_BATCH, PDFValidationError, parse_pdf_bytes
from tools.data import fetch_model, save_embeddings


@dataclass(frozen=True)
class UploadDiagnostics:
    parse_seconds: float = 0.0
    chunk_seconds: float = 0.0
    embedding_index_seconds: float = 0.0
    total_seconds: float = 0.0
    page_count: int = 0
    chunk_count: int = 0


@dataclass(frozen=True)
class UploadResult:
    filename: str
    status: str
    message: str
    document_hash: str | None = None
    diagnostics: UploadDiagnostics = UploadDiagnostics()


def activate_uploaded_documents(active_documents: list[dict], results: list[UploadResult]) -> list[dict]:
    """Add successful and already-indexed uploads to this session's active scope."""
    active = [dict(item) for item in active_documents]
    for result in results:
        if result.status in {"SUCCESS", "ALREADY_INDEXED"} and result.document_hash:
            active = activate_indexed_document(active, IndexedDocument(result.document_hash, result.filename, 0, 0, None))
    return active


def _existing_document_hashes(db: Any) -> set[str]:
    documents = getattr(getattr(db, "docstore", None), "_dict", {}).values()
    return {metadata.get("document_hash") for document in documents if isinstance((metadata := getattr(document, "metadata", {})), dict) and metadata.get("document_hash")}


def _existing_chunk_hashes(db: Any) -> set[str]:
    documents = getattr(getattr(db, "docstore", None), "_dict", {}).values()
    return {metadata.get("document_content_hash") for document in documents if isinstance((metadata := getattr(document, "metadata", {})), dict) and metadata.get("document_content_hash")}


def _document_entries(db: Any, document_hash: str) -> list[tuple[str, Any]]:
    return [
        (identifier, document)
        for identifier, document in getattr(getattr(db, "docstore", None), "_dict", {}).items()
        if (metadata := getattr(document, "metadata", {}) or {}).get("origin") == "upload"
        and metadata.get("document_hash") == document_hash
    ]


async def ingest_pdf_bytes(filename: str, pdf_bytes: bytes, *, mime_type: str | None = None, memory_path=None) -> UploadResult:
    """Parse, dedupe, locally embed, and persist one PDF without storing its bytes."""
    started_at = time.perf_counter()
    if mime_type and mime_type not in {"application/pdf", "application/x-pdf", "application/octet-stream"}:
        return UploadResult(filename, "INVALID_PDF", "The uploaded file does not have a PDF MIME type.")
    try:
        parse_started = time.perf_counter()
        parsed = parse_pdf_bytes(filename, pdf_bytes)
        parse_seconds = time.perf_counter() - parse_started
    except PDFValidationError as error:
        return UploadResult(filename, error.status, str(error), diagnostics=UploadDiagnostics(total_seconds=time.perf_counter() - started_at))

    db = await fetch_model(memory_path=memory_path)
    existing_entries = _document_entries(db, parsed.document_hash)
    old_entry_ids = [identifier for identifier, _ in existing_entries]
    if existing_entries and all((getattr(document, "metadata", {}) or {}).get("pdf_ingestion_version") == PDF_INGESTION_VERSION for _, document in existing_entries):
        return UploadResult(parsed.filename, "ALREADY_INDEXED", "Already indexed.", parsed.document_hash, UploadDiagnostics(parse_seconds=parse_seconds, total_seconds=time.perf_counter() - started_at, page_count=parsed.page_count))

    chunk_started = time.perf_counter()
    chunks = chunk_parsed_pdf(parsed)
    chunk_seconds = time.perf_counter() - chunk_started
    existing_hashes = _existing_chunk_hashes(db) if not old_entry_ids else set()
    new_chunks = [chunk for chunk in chunks if chunk.metadata["document_content_hash"] not in existing_hashes]
    diagnostics = UploadDiagnostics(parse_seconds, chunk_seconds, 0.0, time.perf_counter() - started_at, parsed.page_count, len(new_chunks))
    if not new_chunks:
        return UploadResult(parsed.filename, "ALREADY_INDEXED", "Already indexed.", parsed.document_hash, diagnostics)
    try:
        index_started = time.perf_counter()
        entry_ids_before_indexing = set(getattr(getattr(db, "docstore", None), "_dict", {}))
        await save_embeddings(new_chunks, memory_path=memory_path)
        if old_entry_ids:
            try:
                db.delete(ids=old_entry_ids)
                persist_vector_store(db, memory_path)
            except Exception:
                new_entry_ids = list(set(getattr(getattr(db, "docstore", None), "_dict", {})) - entry_ids_before_indexing)
                if new_entry_ids:
                    db.delete(ids=new_entry_ids)
                    persist_vector_store(db, memory_path)
                raise
        diagnostics = UploadDiagnostics(parse_seconds, chunk_seconds, time.perf_counter() - index_started, time.perf_counter() - started_at, parsed.page_count, len(new_chunks))
    except Exception:
        return UploadResult(parsed.filename, "INDEXING_FAILED", "PDF parsing succeeded, but indexing failed.", parsed.document_hash, UploadDiagnostics(parse_seconds, chunk_seconds, 0.0, time.perf_counter() - started_at, parsed.page_count, len(new_chunks)))
    return UploadResult(parsed.filename, "SUCCESS", f"Indexed · {parsed.page_count} pages · {len(new_chunks)} chunks", parsed.document_hash, diagnostics)


async def ingest_pdf_batch(files: list[tuple[str, bytes, str | None]], *, memory_path=None) -> list[UploadResult]:
    """Process each PDF independently so one bad upload cannot roll back siblings."""
    if len(files) > MAX_PDFS_PER_UPLOAD_BATCH:
        return [UploadResult(name, "TOO_MANY_FILES", f"At most {MAX_PDFS_PER_UPLOAD_BATCH} PDFs can be indexed per batch.") for name, _, _ in files]
    return [await ingest_pdf_bytes(name, data, mime_type=mime, memory_path=memory_path) for name, data, mime in files]
