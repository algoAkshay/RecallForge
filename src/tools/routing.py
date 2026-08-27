"""Small, deterministic policy for choosing durable memory or fresh web evidence."""
from dataclasses import dataclass
import re
import unicodedata
from typing import Any

# FAISS IndexFlatL2 returns distances: smaller values are closer matches.
# Provisional heuristics; empirical calibration is intentionally deferred.
MAX_MEMORY_L2_DISTANCE = 1.0
STRONG_MEMORY_L2_DISTANCE = 0.5
MIN_ACCEPTABLE_MEMORY_RESULTS = 2
MIN_DISTINCT_MEMORY_DOCUMENTS = 2

# Initial deterministic heuristic; real evaluation belongs in Task 8.
MIN_EVIDENCE_TERM_COVERAGE = 0.6
MIN_QUERY_CONCEPTS_FOR_COVERAGE_CHECK = 2
MAX_MISSING_REQUIRED_CONCEPTS = 0
MIN_DEFINITION_CONTEXT_TOKENS = 2

STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "by", "compare", "describe", "do",
    "define", "does", "explain", "for", "from", "how", "in", "is", "it", "of", "on",
    "or", "that", "the", "this", "to", "using", "what", "with", "work",
    "who",
})

CURRENTNESS_PHRASES = ("as of now", "right now", "currently", "current pricing", "current ceo", "currently available")
RECENCY_PHRASES = ("latest ", "newest ", "recent ", "recently updated", "what changed", "this week", "today")
NEWS_PHRASES = ("news today", "latest news", "breaking news")
RECENCY_TARGETS = ("change", "changes", "release", "version", "update", "updates", "information", "pricing", "model")
SHORT_BROAD_QUERY_PATTERNS = (
    re.compile(r"^what\s+is\s+(?P<subject>[\w\s-]+)$"),
    re.compile(r"^what's\s+(?P<subject>[\w\s-]+)$"),
    re.compile(r"^define\s+(?P<subject>[\w\s-]+)$"),
    re.compile(r"^explain\s+(?P<subject>[\w\s-]+)$"),
    re.compile(r"^who\s+is\s+(?P<subject>[\w\s-]+)$"),
)
DESCRIPTIVE_PREDICATES = (
    "is", "are", "was", "were", "refers to", "means", "denotes", "describes",
    "provides", "manages", "coordinates", "stores", "retrieves", "enables", "retains",
)
DEFINITION_CLASS_TERMS = frozenset({
    "database", "element", "framework", "language", "organization", "person", "platform",
    "protocol", "service", "system", "technology", "tool", "type",
})

@dataclass(frozen=True)
class RoutingDecision:
    route: str
    reason_code: str
    reason: str
    freshness_sensitive: bool = False
    best_score: float | None = None
    acceptable_candidates: int = 0
    distinct_sources: int = 0
    coverage_ratio: float | None = None
    required_concepts: tuple[str, ...] = ()
    missing_required_concepts: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceCoverage:
    """Deterministic, token-boundary-aware evidence coverage diagnostics."""

    applies: bool
    coverage_ratio: float | None
    required_concepts: tuple[str, ...]
    missing_required_concepts: tuple[str, ...]

    @property
    def sufficient(self) -> bool:
        return (
            not self.applies
            or (
                not self.missing_required_concepts
                and self.coverage_ratio is not None
                and self.coverage_ratio >= MIN_EVIDENCE_TERM_COVERAGE
            )
        )


@dataclass(frozen=True)
class ShortBroadQuery:
    """A normalized, one-subject definition or identity request."""

    subject: tuple[str, ...]


def _normalize_query(query: str) -> str:
    """Normalize case, Unicode apostrophes, and terminal punctuation for matching."""
    normalized = unicodedata.normalize("NFKC", query).lower().replace("’", "'")
    return " ".join(normalized.strip().rstrip("?!.").split())


def detect_short_broad_query(query: str) -> ShortBroadQuery | None:
    """Recognize short single-subject definition requests without entity names."""
    normalized = _normalize_query(query)
    for pattern in SHORT_BROAD_QUERY_PATTERNS:
        match = pattern.fullmatch(normalized)
        if not match:
            continue
        subject = tuple(token.lower() for token in _tokens(match.group("subject")) if token.lower() not in STOPWORDS)
        # A single named/concept subject is the narrow case at risk of broad-token
        # coverage.  Identity requests also naturally include short full names.
        if 1 <= len(subject) <= 3 and (len(subject) == 1 or normalized.startswith("who is ")):
            return ShortBroadQuery(subject)
    return None


def _has_answerable_definition(subject: tuple[str, ...], content: str) -> bool:
    """Check whether a document describes the subject itself, not a specialization."""
    subject_pattern = r"\s+".join(re.escape(token) for token in subject)
    predicate_pattern = "|".join(re.escape(predicate).replace(r"\ ", r"\s+") for predicate in DESCRIPTIVE_PREDICATES)
    # Requiring the exact subject at a sentence boundary prevents a broad entity
    # token in phrases such as "Redis persistence uses ..." from qualifying.
    match = re.search(
        rf"(?:^|[.!?]\s+)\s*{subject_pattern}\s+(?:{predicate_pattern})\s+(?P<context>[^.!?]+)",
        _normalize_query(content),
    )
    if not match:
        return False
    context_tokens = [token.lower() for token in _tokens(match.group("context")) if token.lower() not in STOPWORDS]
    return (
        len(context_tokens) >= MIN_DEFINITION_CONTEXT_TOKENS
        or bool(set(context_tokens) & DEFINITION_CLASS_TERMS)
    )


def has_short_query_answerability(query: str, candidates: list[tuple[Any, float]]) -> bool:
    """Require actual descriptive evidence for a narrow broad-definition query."""
    broad_query = detect_short_broad_query(query)
    if broad_query is None:
        return True
    return any(_has_answerable_definition(broad_query.subject, getattr(document, "page_content", "") or "") for document, _ in candidates)

def detect_freshness(query: str) -> bool:
    normalized = " " + " ".join(query.lower().split()) + " "
    if any(phrase in normalized for phrase in CURRENTNESS_PHRASES + NEWS_PHRASES):
        return True
    # These words only count with a time-sensitive target, avoiding e.g. "new student".
    recency_target = "|".join(RECENCY_TARGETS)
    return bool(
        re.search(rf"\b(latest|newest|recent)\b(?:\s+\w+){{0,3}}\s+({recency_target})\b", normalized)
        or re.search(r"\b(today|yesterday)\b(?:'s)?\s+news\b", normalized)
        or re.search(r"\bwhat changed\b.*\b(today|this week|recently)\b", normalized)
        or re.search(r"\bwho is the current\b", normalized)
    )

def _document_key(document: Any) -> str:
    metadata = getattr(document, "metadata", {}) or {}
    return metadata.get("document_content_hash") or metadata.get("source_url") or metadata.get("link") or metadata.get("source") or str(id(document))

def _tokens(text: str) -> list[str]:
    """Normalize text into comparison tokens without substring matching."""
    normalized = unicodedata.normalize("NFKC", text).replace("-", " ")
    return re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)


def _is_required_technical_token(token: str) -> bool:
    """Recognize explicit acronyms and code-like tokens before lowercasing."""
    return bool(
        re.fullmatch(r"[A-Z]{2,8}", token)
        or (any(character.isdigit() for character in token) and len(token) >= 2)
        or (any(character.isupper() for character in token[1:]) and any(character.islower() for character in token))
    )


def extract_query_concepts(query: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return informative concepts and explicit technical concepts, in query order."""
    # Broad definition commands are not concepts, and their casing should not
    # accidentally turn ordinary entity names into required acronyms.
    broad_query = detect_short_broad_query(query)
    original_tokens = _tokens(query)
    short_acronym = (
        broad_query is not None
        and not _normalize_query(query).startswith("who is ")
        and any(token.isupper() and 2 <= len(token) <= 4 for token in original_tokens[-len(broad_query.subject) :])
    )
    raw_tokens = original_tokens if short_acronym or broad_query is None else _tokens(_normalize_query(query))
    informative = tuple(dict.fromkeys(
        token.lower() for token in raw_tokens if len(token) > 1 and token.lower() not in STOPWORDS
    ))
    required = tuple(dict.fromkeys(
        token.lower() for token in raw_tokens if _is_required_technical_token(token)
    ))
    return informative, required


def evaluate_evidence_coverage(query: str, candidates: list[tuple[Any, float]]) -> EvidenceCoverage:
    """Measure concept coverage across the semantically acceptable evidence set."""
    informative, required = extract_query_concepts(query)
    # A short, ordinary follow-up has insufficient standalone context to judge.
    applies = len(informative) >= MIN_QUERY_CONCEPTS_FOR_COVERAGE_CHECK or bool(required)
    if not applies:
        return EvidenceCoverage(False, None, required, ())

    evidence_tokens: set[str] = set()
    for document, _ in candidates:
        evidence_tokens.update(token.lower() for token in _tokens(getattr(document, "page_content", "") or ""))

    matched = sum(concept in evidence_tokens for concept in informative)
    coverage_ratio = matched / len(informative) if informative else 1.0
    missing_required = tuple(concept for concept in required if concept not in evidence_tokens)
    return EvidenceCoverage(True, coverage_ratio, required, missing_required)


def evaluate_memory_evidence(query: str, candidates: list[tuple[Any, float]]) -> RoutingDecision:
    if not candidates:
        return RoutingDecision("web", "memory_empty", "No stored research evidence was found.")
    acceptable = [(doc, score) for doc, score in candidates if score <= MAX_MEMORY_L2_DISTANCE]
    best = min(score for _, score in candidates)
    sources = {_document_key(doc) for doc, _ in acceptable}
    strong = best <= STRONG_MEMORY_L2_DISTANCE
    sufficient = len(acceptable) >= MIN_ACCEPTABLE_MEMORY_RESULTS and len(sources) >= MIN_DISTINCT_MEMORY_DOCUMENTS
    if not (strong or sufficient):
        return RoutingDecision("web", "memory_insufficient", "Stored evidence did not meet the minimum relevance requirement.", best_score=best, acceptable_candidates=len(acceptable), distinct_sources=len(sources))

    coverage = evaluate_evidence_coverage(query, acceptable)
    if not coverage.sufficient:
        return RoutingDecision(
            "web", "memory_insufficient_coverage",
            "Stored evidence was relevant to the topic but did not cover enough of the query's key concepts.",
            best_score=best, acceptable_candidates=len(acceptable), distinct_sources=len(sources),
            coverage_ratio=coverage.coverage_ratio, required_concepts=coverage.required_concepts,
            missing_required_concepts=coverage.missing_required_concepts,
        )
    if not has_short_query_answerability(query, acceptable):
        return RoutingDecision(
            "web", "memory_insufficient_answerability",
            "Stored evidence mentions the subject but does not provide enough descriptive context to answer this broad query.",
            best_score=best, acceptable_candidates=len(acceptable), distinct_sources=len(sources),
            coverage_ratio=coverage.coverage_ratio, required_concepts=coverage.required_concepts,
            missing_required_concepts=coverage.missing_required_concepts,
        )
    return RoutingDecision(
        "memory", "memory_strong_match" if strong else "memory_sufficient",
        "Stored research contains a strong relevant match." if strong else "Stored research contains sufficient relevant evidence.",
        best_score=best, acceptable_candidates=len(acceptable), distinct_sources=len(sources),
        coverage_ratio=coverage.coverage_ratio, required_concepts=coverage.required_concepts,
        missing_required_concepts=coverage.missing_required_concepts,
    )

def choose_route(query: str, candidates: list[tuple[Any, float]] | None = None) -> RoutingDecision:
    if detect_freshness(query):
        return RoutingDecision("web", "freshness_required", "Query requests current information.", freshness_sensitive=True)
    return evaluate_memory_evidence(query, candidates or [])
