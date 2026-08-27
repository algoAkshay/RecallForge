"""One-shot user controls that do not alter RecallForge's AUTO policy."""
from __future__ import annotations

from tools.routing import RoutingDecision


FORCE_WEB_REASON = "Fresh web research was requested."


def consume_search_fresh(state: dict, selected: bool) -> bool:
    """Consume a checkbox value and advance its key for the next query."""
    state["search_fresh_control_version"] = int(state.get("search_fresh_control_version", 0)) + 1
    return bool(selected)


def forced_web_decision() -> RoutingDecision:
    """Represent an explicit user request without changing AUTO routing rules."""
    return RoutingDecision("web", "force_web_requested", FORCE_WEB_REASON, freshness_sensitive=True)
