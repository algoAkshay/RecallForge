"""Document-management helpers over persisted uploaded-document FAISS metadata."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tools.data import persist_vector_store


@dataclass(frozen=True)
class IndexedDocument:
    document_hash: str
    filename: str
    page_count: int
    chunk_count: int
    ingestion_version: int | str | None


def _documents(db: Any):
    return getattr(getattr(db, "docstore", None), "_dict", {}).items()


def get_document_chunk_ids(db: Any, document_hash: str) -> list[str]:
    """Return only uploaded chunks with the exact stable document identity."""
    return [
        identifier
        for identifier, document in _documents(db)
        if (metadata := getattr(document, "metadata", {}) or {}).get("origin") == "upload"
        and metadata.get("document_hash") == document_hash
    ]


def list_indexed_documents(db: Any) -> list[IndexedDocument]:
    """Collapse persisted upload chunks into deterministic document rows."""
    grouped: dict[str, list[dict]] = {}
    for _, document in _documents(db):
        metadata = getattr(document, "metadata", {}) or {}
        document_hash = metadata.get("document_hash")
        if metadata.get("origin") != "upload" or not isinstance(document_hash, str) or not document_hash:
            continue
        grouped.setdefault(document_hash, []).append(metadata)

    records = []
    for document_hash, chunks in grouped.items():
        filenames = [metadata.get("filename") for metadata in chunks if isinstance(metadata.get("filename"), str) and metadata["filename"]]
        pages = [metadata.get("page_number") for metadata in chunks if isinstance(metadata.get("page_number"), int)]
        versions = [metadata.get("pdf_ingestion_version") for metadata in chunks if metadata.get("pdf_ingestion_version") is not None]
        records.append(IndexedDocument(
            document_hash=document_hash,
            filename=filenames[0] if filenames else "Uploaded document",
            page_count=max(pages) if pages else 0,
            chunk_count=len(chunks),
            ingestion_version=max(versions) if versions else None,
        ))
    return sorted(records, key=lambda record: (record.filename.casefold(), record.document_hash))


def activate_indexed_document(active_documents: list[dict], document: IndexedDocument) -> list[dict]:
    """Add one registry row to session-scoped retrieval state without duplicates."""
    active = [dict(item) for item in active_documents]
    if document.document_hash not in {item.get("document_hash") for item in active}:
        active.append({"document_hash": document.document_hash, "filename": document.filename})
    return active


def remove_document_from_active_state(active_documents: list[dict], document_hash: str) -> list[dict]:
    """Remove one deleted document while preserving all other active documents."""
    return [dict(item) for item in active_documents if item.get("document_hash") != document_hash]


def delete_indexed_document(db: Any, document_hash: str, *, memory_path=None) -> None:
    """Delete the exact uploaded document via LangChain's supported FAISS API."""
    identifiers = get_document_chunk_ids(db, document_hash)
    if not identifiers:
        raise KeyError("Indexed document was not found.")
    db.delete(ids=identifiers)
    persist_vector_store(db, memory_path)
