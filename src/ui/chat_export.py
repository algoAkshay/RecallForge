"""Pure user-visible chat export and sidebar timestamp helpers."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import re


def safe_export_filename(title: str) -> str:
    """Create a conservative, download-safe Markdown filename."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-")[:72].strip("-")
    return f"{slug or 'recallforge-chat'}.md"


def format_thread_timestamp(value: str, now: datetime | None = None) -> str:
    """Render UTC thread metadata in compact local-relative form."""
    try:
        updated = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return ""
    updated = updated.astimezone()
    now = (now or datetime.now(timezone.utc)).astimezone(updated.tzinfo)
    elapsed = max(timedelta(), now - updated)
    minutes = int(elapsed.total_seconds() // 60)
    if minutes < 1: return "Just now"
    if minutes < 60: return f"{minutes}m ago"
    if minutes < 24 * 60: return f"{minutes // 60}h ago"
    if updated.date() == (now.date() - timedelta(days=1)): return "Yesterday"
    return updated.strftime("%b %d")


def export_chat_markdown(title: str, messages: list[dict]) -> str:
    """Export only already-visible conversation fields; never regenerate content."""
    blocks = [f"# {title}", "", "Exported from RecallForge"]
    for message in messages:
        if message.get("role") == "system":
            continue
        heading = "User" if message.get("role") == "user" else "RecallForge"
        blocks.extend(["", f"## {heading}", "", str(message.get("content", "")).strip()])
        if message.get("role") == "assistant":
            for label, key in (("Route", "route"), ("Reason", "reason"), ("Duration", "elapsed")):
                if value := str(message.get(key, "")).strip():
                    blocks.extend(["", f"**{label}:** {value}"])
            if sources := str(message.get("sources", "")).strip():
                blocks.extend(["", sources])
    return "\n".join(blocks).strip() + "\n"
