"""One-shot MEMORY-to-WEB recovery orchestration, independent of the UI."""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from collections.abc import Awaitable, Callable


INSUFFICIENT_EVIDENCE_SENTINEL = "I couldn't find enough relevant evidence to answer this confidently."
FALLBACK_REASON = "Stored evidence was relevant but insufficient to answer the query, so fresh web research was used."
FALLBACK_FAILURE = "Stored evidence was insufficient, and fresh web research did not return enough usable evidence."


@dataclass(frozen=True)
class SynthesisResult:
    """Structured interpretation of one synthesis response."""

    answer: str
    sufficient: bool

    @classmethod
    def from_answer(cls, answer: str) -> "SynthesisResult":
        """Recognize only the explicit insufficiency protocol, never loose phrases."""
        normalized = " ".join(unicodedata.normalize("NFKC", answer).replace("’", "'").split()).casefold()
        sentinel = " ".join(INSUFFICIENT_EVIDENCE_SENTINEL.split()).casefold()
        return cls(answer=answer, sufficient=normalized != sentinel)


@dataclass(frozen=True)
class FallbackDiagnostics:
    initial_route: str
    fallback_attempted: bool
    fallback_reason: str
    final_route: str
    synthesis_count: int


@dataclass(frozen=True)
class FallbackOutcome:
    """Final response state after zero or one bounded fallback."""

    answer: str
    reason: str
    diagnostics: FallbackDiagnostics


def has_usable_web_evidence(evidence: str) -> bool:
    """Accept only the existing scraper's explicit evidence section."""
    return bool(re.search(r"(?:^|\n)EVIDENCE(?:\n|$)", evidence))


async def synthesize_with_single_fallback(
    *,
    initial_route: str,
    initial_reason: str,
    initial_evidence: str,
    synthesize: Callable[[str, str], Awaitable[SynthesisResult]],
    acquire_web: Callable[[], Awaitable[str]],
    replace_request_evidence: Callable[[], None],
) -> FallbackOutcome:
    """Synthesize once, recovering only insufficient MEMORY with one WEB pass."""
    route = initial_route.lower()
    first = await synthesize(route, initial_evidence)
    if route != "memory" or first.sufficient:
        return FallbackOutcome(
            first.answer,
            initial_reason,
            FallbackDiagnostics(route, False, "", route, 1),
        )

    # Citation state belongs to the final answer.  Clearing it before acquisition
    # ensures WEB receives a fresh S1... mapping rather than a mixed source map.
    replace_request_evidence()
    web_evidence = await acquire_web()
    if not has_usable_web_evidence(web_evidence):
        replace_request_evidence()
        return FallbackOutcome(
            FALLBACK_FAILURE,
            FALLBACK_REASON,
            FallbackDiagnostics(route, True, "memory_synthesis_insufficient", "web", 1),
        )

    second = await synthesize("web", web_evidence)
    if not second.sufficient:
        replace_request_evidence()
        return FallbackOutcome(
            FALLBACK_FAILURE,
            FALLBACK_REASON,
            FallbackDiagnostics(route, True, "memory_synthesis_insufficient", "web", 2),
        )
    return FallbackOutcome(
        second.answer,
        FALLBACK_REASON,
        FallbackDiagnostics(route, True, "memory_synthesis_insufficient", "web", 2),
    )
