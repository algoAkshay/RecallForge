"""Invocation-local evidence grouping and application-owned citation mapping."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import re
from typing import Any, Iterator

@dataclass(frozen=True)
class EvidenceItem:
    content: str
    source_url: str | None = None
    title: str | None = None
    retrieved_at: str | None = None
    document_content_hash: str | None = None
    origin: str = "memory"
    filename: str | None = None
    page_number: int | None = None
    document_hash: str | None = None

@dataclass
class SourceRecord:
    citation_id: str
    source_url: str | None
    title: str | None
    retrieved_at: str | None
    document_content_hash: str | None
    filename: str | None = None
    page_number: int | None = None
    document_hash: str | None = None
    evidence: list[EvidenceItem] = field(default_factory=list)

def evidence_from_document(document: Any, origin: str = "memory") -> EvidenceItem:
    """Normalize legacy and new document metadata at one boundary."""
    metadata = getattr(document, "metadata", None) or (document.get("metadata", {}) if isinstance(document, dict) else {})
    content = getattr(document, "page_content", None) or (document.get("content", "") if isinstance(document, dict) else "")
    item_origin = metadata.get("origin") or origin
    filename = metadata.get("filename")
    source_url = None if item_origin == "upload" else metadata.get("source_url") or metadata.get("link") or metadata.get("source")
    title = filename if item_origin == "upload" else metadata.get("title") or (metadata.get("source") if metadata.get("source") != source_url else None)
    return EvidenceItem(str(content), source_url, title, metadata.get("retrieved_at") or metadata.get("ingested_at"), metadata.get("document_content_hash"), item_origin, filename, metadata.get("page_number"), metadata.get("document_hash"))

def evidence_from_mapping(record: dict[str, Any], origin: str = "web") -> EvidenceItem:
    return EvidenceItem(str(record.get("content", "")), record.get("source_url") or record.get("link") or record.get("url") or record.get("source"), record.get("title"), record.get("retrieved_at"), record.get("document_content_hash"), origin)

def _group_key(item: EvidenceItem) -> tuple[str, str]:
    if item.origin == "upload" and item.filename and item.page_number:
        return "upload_page", f"{item.filename}:{item.page_number}:{item.document_content_hash or item.content}"
    if item.document_content_hash:
        return "hash", item.document_content_hash
    if item.source_url:
        return "url", item.source_url
    return "content", item.content

def group_evidence(items: list[EvidenceItem]) -> list[SourceRecord]:
    sources: list[SourceRecord] = []
    by_key: dict[tuple[str, str], SourceRecord] = {}
    for item in items:
        source = by_key.get(_group_key(item))
        if source is None:
            source = SourceRecord(f"S{len(sources) + 1}", item.source_url, item.title, item.retrieved_at, item.document_content_hash, item.filename, item.page_number, item.document_hash)
            by_key[_group_key(item)] = source
            sources.append(source)
        source.evidence.append(item)
    return sources

def format_evidence(sources: list[SourceRecord]) -> str:
    if not sources:
        return "No retrieved evidence is available."
    blocks = []
    for source in sources:
        lines = [f"[{source.citation_id}]"]
        if source.title: lines.append(f"Title: {source.title}")
        if source.source_url: lines.append(f"URL: {source.source_url}")
        if source.page_number: lines.append(f"Page: {source.page_number}")
        if source.retrieved_at: lines.append(f"Retrieved: {source.retrieved_at}")
        lines.append("Evidence:\n" + "\n\n".join(item.content for item in source.evidence))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)

_collector: ContextVar["CitationCollector | None"] = ContextVar("citation_collector", default=None)

class CitationCollector:
    """Mutable only inside one agent invocation; never stored in Streamlit or FAISS."""
    def __init__(self):
        self.sources: list[SourceRecord] = []
        self._by_key: dict[tuple[str, str], SourceRecord] = {}
    def add(self, items: list[EvidenceItem]) -> list[SourceRecord]:
        used = []
        for item in items:
            source = self._by_key.get(_group_key(item))
            if source is None:
                source = SourceRecord(f"S{len(self.sources) + 1}", item.source_url, item.title, item.retrieved_at, item.document_content_hash, item.filename, item.page_number, item.document_hash)
                self._by_key[_group_key(item)] = source
                self.sources.append(source)
            source.evidence.append(item)
            if source not in used: used.append(source)
        return used
    def clear(self) -> None:
        """Discard request-local evidence before replacing it with a new route."""
        self.sources.clear()
        self._by_key.clear()
    @property
    def source_map(self) -> dict[str, SourceRecord]:
        return {source.citation_id: source for source in self.sources}

@contextmanager
def citation_collection() -> Iterator[CitationCollector]:
    collector = CitationCollector()
    token = _collector.set(collector)
    try: yield collector
    finally: _collector.reset(token)

def register_evidence(items: list[EvidenceItem]) -> list[SourceRecord]:
    collector = _collector.get()
    return collector.add(items) if collector else group_evidence(items)

_CITATION_RE = re.compile(r"\[S([1-9]\d*)\]")
def validate_citations(answer: str, source_map: dict[str, SourceRecord]) -> tuple[set[str], set[str]]:
    referenced = {f"S{number}" for number in _CITATION_RE.findall(answer)}
    return referenced & source_map.keys(), referenced - source_map.keys()
def remove_invalid_citations(answer: str, invalid: set[str]) -> str:
    return re.sub(r"\[(S[1-9]\d*)\]", lambda m: "" if m.group(1) in invalid else m.group(0), answer)
def render_sources(source_map: dict[str, SourceRecord], cited_ids: set[str]) -> str:
    records = [source_map[key] for key in source_map if key in cited_ids]
    if not records: return ""
    return "\n".join(["### Sources"] + [f"[{s.citation_id}] {s.title or 'Retrieved evidence'} — {s.source_url or 'URL unavailable'}" for s in records])
