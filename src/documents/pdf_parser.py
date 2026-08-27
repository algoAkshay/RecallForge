"""Safe local parsing for text-based PDF uploads."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

import fitz

MAX_PDF_SIZE_MB = 50
MAX_PDF_PAGES = 300
MAX_PDFS_PER_UPLOAD_BATCH = 10


class PDFValidationError(ValueError):
    status = "INVALID_PDF"


class PDFPasswordProtectedError(PDFValidationError):
    status = "PASSWORD_PROTECTED"


class PDFNoExtractableTextError(PDFValidationError):
    status = "NO_EXTRACTABLE_TEXT"


@dataclass(frozen=True)
class ParsedPage:
    page_number: int
    text: str


@dataclass(frozen=True)
class ParsedPDF:
    filename: str
    page_count: int
    pages: list[ParsedPage]
    document_hash: str


def document_hash(pdf_bytes: bytes) -> str:
    """Use original bytes as the stable uploaded-document identity."""
    return hashlib.sha256(pdf_bytes).hexdigest()


def _normalized_page_text(text: str) -> str:
    paragraphs = [" ".join(part.split()) for part in text.replace("\r\n", "\n").replace("\r", "\n").split("\n\n")]
    return "\n\n".join(part for part in paragraphs if part)


def parse_pdf_bytes(filename: str, pdf_bytes: bytes, *, max_size_mb: int = MAX_PDF_SIZE_MB, max_pages: int = MAX_PDF_PAGES) -> ParsedPDF:
    """Validate and extract 1-based, text-only pages without writing uploads to disk."""
    if not filename.lower().endswith(".pdf"):
        raise PDFValidationError("Only PDF uploads are supported.")
    if len(pdf_bytes) > max_size_mb * 1024 * 1024:
        error = PDFValidationError(f"PDF exceeds the {max_size_mb} MB upload limit.")
        error.status = "FILE_TOO_LARGE"
        raise error
    if not pdf_bytes.startswith(b"%PDF-"):
        raise PDFValidationError("The uploaded file does not have a valid PDF signature.")
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except (fitz.FileDataError, RuntimeError, ValueError) as error:
        raise PDFValidationError("The uploaded PDF could not be parsed.") from error
    try:
        if document.needs_pass:
            raise PDFPasswordProtectedError("This PDF is password-protected and cannot currently be indexed.")
        page_count = document.page_count
        if page_count > max_pages:
            error = PDFValidationError(f"PDF has {page_count} pages; current limit is {max_pages} pages.")
            error.status = "TOO_MANY_PAGES"
            raise error
        pages: list[ParsedPage] = []
        for index in range(page_count):
            text = _normalized_page_text(document.load_page(index).get_text("text"))
            if text:
                pages.append(ParsedPage(index + 1, text))
    except PDFValidationError:
        raise
    except Exception as error:
        raise PDFValidationError("Text could not be extracted from the uploaded PDF.") from error
    finally:
        document.close()
    if not pages or not any(re.search(r"\w", page.text) for page in pages):
        raise PDFNoExtractableTextError("No extractable text was found. OCR is not supported yet.")
    return ParsedPDF(filename=filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1], page_count=page_count, pages=pages, document_hash=document_hash(pdf_bytes))
