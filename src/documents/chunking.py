"""Deterministic page-aware recursive chunking for uploaded PDFs."""
from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from documents.pdf_parser import ParsedPDF
from tools.data import generate_doc_hash

# Character sizing is an approximate 700-1000 token target, not a claim that
# characters and tokens are equivalent.
PDF_CHUNK_SIZE = 1800
PDF_CHUNK_OVERLAP = 250
PDF_INGESTION_VERSION = 2


def chunk_parsed_pdf(parsed: ParsedPDF, *, chunk_size: int = PDF_CHUNK_SIZE, chunk_overlap: int = PDF_CHUNK_OVERLAP) -> list[Document]:
    """Split each page independently so every chunk retains one precise page."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    ingested_at = datetime.now(timezone.utc).isoformat()
    chunks: list[Document] = []
    for page in parsed.pages:
        metadata = {
            "filename": parsed.filename,
            "title": parsed.filename,
            "page_number": page.page_number,
            "page_start": page.page_number,
            "page_end": page.page_number,
            "document_hash": parsed.document_hash,
            "origin": "upload",
            "source_type": "uploaded_document",
            "pdf_ingestion_version": PDF_INGESTION_VERSION,
            "ingested_at": ingested_at,
        }
        for chunk in splitter.create_documents([page.text], metadatas=[metadata]):
            chunk.metadata["document_content_hash"] = generate_doc_hash(chunk.page_content)
            chunks.append(chunk)
    return chunks
