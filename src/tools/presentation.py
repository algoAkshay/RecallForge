"""Presentation-only cleanup and rendering for RecallForge answers and source records."""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

_TRAILING_BIBLIOGRAPHY = re.compile(r"(?im)^\s{0,3}#{2,6}\s+(?:sources|references|bibliography)\s*$[\s\S]*\Z")
_LOCAL_SVG_ANCHOR = re.compile(r"\[svg\]\(https?://localhost(?::\d+)?/[^)]*\)", re.IGNORECASE)
_DATA_SVG_IMAGE = re.compile(r"!?\[[^\]]*\]\(data:image/svg\+xml,[^\n)]*\)", re.IGNORECASE)
_STANDALONE_SVG = re.compile(r"(?im)^\s*svg\s*$\n?")
_FENCED_BLOCK = re.compile(r"(```[\s\S]*?```)")
_TABLE_SEPARATOR_CELL = re.compile(r"^\s*:?-{3,}:?\s*$")


def _sanitize_non_code(text: str) -> str:
    text = _TRAILING_BIBLIOGRAPHY.sub("", text)
    text = _LOCAL_SVG_ANCHOR.sub("", text)
    text = _DATA_SVG_IMAGE.sub("", text)
    return repair_markdown_tables(_STANDALONE_SVG.sub("", text))


def _table_cells(line: str) -> list[str] | None:
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return None
    return [cell.strip() for cell in stripped[1:-1].split("|")]


def repair_markdown_tables(text: str) -> str:
    """Repair only a provably concatenated header with an inferable column count."""
    lines = text.splitlines(keepends=True)
    for index in range(len(lines) - 2):
        header, separator, body = (_table_cells(lines[index]), _table_cells(lines[index + 1]), _table_cells(lines[index + 2]))
        if header is None or separator is None or body is None or len(header) != 1:
            continue
        column_count = len(separator)
        if column_count < 2 or len(body) != column_count or not all(_TABLE_SEPARATOR_CELL.fullmatch(cell) for cell in separator):
            continue
        # A lowercase/digit/closing-paren followed by an uppercase acronym is a structural boundary,
        # e.g. "Feature / AspectRDB (...)AOF (...)". Refuse ambiguous headers.
        cells = [part.strip() for part in re.split(r"(?<=[a-z0-9)])(?=[A-Z]{2,}\b)", header[0])]
        if len(cells) == column_count and all(cells):
            ending = "\n" if lines[index].endswith("\n") else ""
            lines[index] = "| " + " | ".join(cells) + " |" + ending
    return "".join(lines)


def sanitize_answer_markdown(text: str) -> str:
    """Remove known model/render artifacts while preserving ordinary Markdown."""
    parts = _FENCED_BLOCK.split(text)
    return "".join(part if part.startswith("```") else _sanitize_non_code(part) for part in parts).rstrip()


def normalize_source_title(title: str | None) -> str:
    """Normalize only whitespace and exact adjacent duplicate title strings."""
    normalized = " ".join((title or "").split())
    duplicate = re.fullmatch(r"(.+?)\s+\1", normalized, flags=re.IGNORECASE)
    return duplicate.group(1) if duplicate else normalized


def source_title(source: Any) -> str:
    title = normalize_source_title(getattr(source, "title", None))
    if title:
        return title
    hostname = urlparse(getattr(source, "source_url", "") or "").hostname
    return hostname or "Untitled source"


def render_authoritative_sources(source_map: dict[str, Any], cited_ids: set[str]) -> str:
    """Return one application-owned, stable Markdown block for cited provenance records."""
    records = [source_map[key] for key in source_map if key in cited_ids]
    if not records:
        return ""
    entries, seen = [], set()
    for source in records:
        if source.citation_id in seen:
            continue
        seen.add(source.citation_id)
        url = source.source_url
        if getattr(source, "page_number", None):
            destination = f"Page {source.page_number} · Uploaded document"
        else:
            destination = f"[{url}]({url})" if url else "URL unavailable"
        entries.append(f"[{source.citation_id}] **{source_title(source)}**  \n{destination}")
    return "### Sources\n\n" + "\n\n".join(entries)
