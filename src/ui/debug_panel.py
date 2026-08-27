"""Safe, user-facing response diagnostics with no hidden model internals."""
from __future__ import annotations

from collections.abc import Iterable


_WEB_FLOAT_FIELDS = {
    "search_seconds": "Search",
    "fetch_extract_seconds": "Fetch/extract",
    "embedding_index_seconds": "Embedding/indexing",
    "total_web_acquisition_seconds": "WEB acquisition",
}
_WEB_COUNT_FIELDS = {
    "search_result_count": "Search results",
    "fetch_attempt_count": "Fetch attempts",
    "fetch_success_count": "Fetch successes",
    "fetch_timeout_count": "Fetch timeouts",
    "fetch_failure_count": "Fetch failures",
}


def parse_web_diagnostics(evidence: str) -> dict[str, float | int | bool]:
    """Extract only the scraper's fixed public metrics from its response text."""
    marker = "\n\nDiagnostics:\n"
    if marker not in evidence:
        return {}
    values: dict[str, float | int | bool] = {}
    allowed = set(_WEB_FLOAT_FIELDS) | set(_WEB_COUNT_FIELDS) | {"search_timeout", "acquisition_deadline_reached"}
    for line in evidence.rsplit(marker, 1)[1].splitlines():
        key, separator, raw_value = line.partition(":")
        if not separator or key not in allowed:
            continue
        raw_value = raw_value.strip()
        try:
            if key in _WEB_FLOAT_FIELDS:
                values[key] = float(raw_value)
            elif key in _WEB_COUNT_FIELDS:
                values[key] = int(raw_value)
            elif raw_value in {"true", "false"}:
                values[key] = raw_value == "true"
        except ValueError:
            continue
    return values


def format_duration(value: object) -> str:
    """Format measured durations, including a genuinely measured zero."""
    return f"{float(value):.1f}s"


def build_debug_rows(message: dict) -> list[tuple[str, str]]:
    """Return an allow-listed diagnostic view suitable for current or old messages."""
    debug = message.get("debug") if isinstance(message.get("debug"), dict) else {}
    rows: list[tuple[str, str]] = []
    for label, value in (("Route", message.get("route")), ("Mode", debug.get("routing_mode")), ("Initial route", debug.get("initial_route")), ("Final route", debug.get("final_route")), ("Reason", message.get("reason"))):
        if value:
            rows.append((label, str(value)))
    if "fallback_attempted" in debug:
        rows.append(("Fallback", "Yes" if debug["fallback_attempted"] else "No"))
    if "total_seconds" in debug:
        rows.append(("Total", format_duration(debug["total_seconds"])))
    elif message.get("elapsed"):
        rows.append(("Total", str(message["elapsed"])))
    for key, label in (("memory_retrieval_seconds", "Memory retrieval"), ("synthesis_seconds", "Synthesis")):
        if key in debug:
            rows.append((label, format_duration(debug[key])))
    for key, label in _WEB_FLOAT_FIELDS.items():
        if key in debug:
            rows.append((label, format_duration(debug[key])))
    for key, label in _WEB_COUNT_FIELDS.items():
        if key in debug:
            rows.append((label, str(debug[key])))
    for key, label in (("search_timeout", "Search timeout"), ("acquisition_deadline_reached", "Acquisition deadline reached")):
        if key in debug:
            rows.append((label, "Yes" if debug[key] else "No"))
    if "source_count" in debug:
        rows.append(("Sources", str(debug["source_count"])))
    return rows


def debug_markdown(rows: Iterable[tuple[str, str]]) -> str:
    """Render compact text without exposing arbitrary diagnostic dictionary keys."""
    return "\n\n".join(f"**{label}:** {value}" for label, value in rows)
