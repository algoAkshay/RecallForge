"""Fixed authored datasets. Synthetic distances are policy tests, not calibration data."""
from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateSpec:
    key: str
    content: str
    distance: float


@dataclass(frozen=True)
class RoutingCase:
    name: str
    split: str
    query: str
    expected_route: str
    freshness_label: bool
    candidates: tuple[CandidateSpec, ...] = ()


GOOD_RAG = "RAG retrieves relevant context before generating an answer."
GOOD_REDIS = "Redis persistence uses RDB snapshots and AOF append-only logging."
REDIS_RELEASE = "Redis release support schedule and version lifecycle information."


def _case(name, split, query, expected, fresh=False, candidates=()):
    return RoutingCase(name, split, query, expected, fresh, tuple(candidates))


def controlled_cases() -> tuple[RoutingCase, ...]:
    """46 fixed cases: 30 calibration and 16 held-out evaluation."""
    strong_rag = (CandidateSpec("rag", GOOD_RAG, 0.2),)
    strong_redis = (CandidateSpec("redis", GOOD_REDIS, 0.2),)
    multi_rag = (CandidateSpec("rag-a", GOOD_RAG, 0.8), CandidateSpec("rag-b", GOOD_RAG, 0.9))
    split_redis = (CandidateSpec("rdb", "RDB snapshots persist Redis data.", 0.8), CandidateSpec("aof", "AOF append-only logging records Redis writes.", 0.9))
    cases = [
        _case("fresh-latest-release", "calibration", "latest Redis release", "web", True, strong_redis),
        _case("fresh-current-ceo", "calibration", "Who is the current CEO of Example Corp?", "web", True, strong_rag),
        _case("fresh-news-today", "calibration", "AI news today", "web", True, strong_rag),
        _case("fresh-newest-version", "calibration", "newest stable package version", "web", True, strong_rag),
        _case("trap-current-flow", "calibration", "Explain current flow in an electrical circuit", "memory", False, (CandidateSpec("flow", "Current flow in an electrical circuit describes charge movement.", .2),)),
        _case("trap-news-system", "calibration", "How does a news recommendation system work?", "memory", False, (CandidateSpec("news", "A news recommendation system ranks articles for readers.", .2),)),
        _case("strong-rag", "calibration", "How does RAG retrieve relevant context?", "memory", False, strong_rag),
        _case("acceptable-multi", "calibration", "Explain RAG retrieval context", "memory", False, multi_rag),
        _case("weak-memory", "calibration", "Explain RAG retrieval context", "web", False, (CandidateSpec("weak", GOOD_RAG, 1.1),)),
        _case("empty-memory", "calibration", "Explain database indexing", "web"),
        _case("coverage-rdb-aof", "calibration", "Compare RDB and AOF persistence", "memory", False, strong_redis),
        _case("coverage-rdb-only", "calibration", "Compare RDB and AOF persistence", "web", False, (CandidateSpec("rdb", "RDB snapshots persist Redis data.", .2),)),
        _case("coverage-split", "calibration", "Compare RDB and AOF persistence", "memory", False, split_redis),
        _case("follow-up", "calibration", "How does it work?", "memory", False, (CandidateSpec("follow", "Prior discussion evidence.", .2),)),
        _case("single-acceptable", "calibration", "Explain FAISS vector database", "web", False, (CandidateSpec("faiss", "FAISS is a vector database index.", .8),)),
        _case("duplicate-source", "calibration", "Explain FAISS vector database", "web", False, (CandidateSpec("same", "FAISS vector database index.", .8), CandidateSpec("same", "FAISS vector database index.", .9))),
        _case("two-source", "calibration", "Explain FAISS vector database", "memory", False, (CandidateSpec("one", "FAISS vector database index.", .8), CandidateSpec("two", "FAISS vector database searches embeddings.", .9))),
        _case("fresh-pricing", "calibration", "current pricing for Example", "web", True, strong_rag),
        _case("trap-new-student", "calibration", "new student onboarding", "memory", False, (CandidateSpec("student", "New student onboarding introduces the program.", .2),)),
        _case("strong-crdt", "calibration", "Explain CRDT tombstones", "memory", False, (CandidateSpec("crdt", "CRDT tombstones retain deletion information for replicas.", .2),)),
        _case("weak-unrelated", "calibration", "Explain HTTP caching", "web", False, (CandidateSpec("other", "Database indexing organizes records.", 1.2),)),
        _case("lowercase-coverage", "calibration", "Compare snapshot persistence and append-only logging", "memory", False, strong_redis),
        _case("fresh-recent-changes", "calibration", "What changed recently in LangChain?", "web", True, strong_rag),
        _case("trap-latest-element", "calibration", "Explain the latest-element pointer", "memory", False, (CandidateSpec("pointer", "A latest element pointer refers to the newest node.", .2),)),
        _case("answerability-specialized-redis", "calibration", "What is Redis?", "web", False, (CandidateSpec("redis-persistence", "Redis persistence uses RDB snapshots and AOF append-only logging.", .2),)),
        _case("answerability-definition-redis", "calibration", "What is Redis?", "memory", False, (CandidateSpec("redis-definition", "Redis is an in-memory data store commonly used as a cache, database, and message broker.", .2),)),
        _case("answerability-person", "calibration", "Who is Ada Lovelace?", "memory", False, (CandidateSpec("ada", "Ada Lovelace was a mathematician who wrote an early algorithm for a computer.", .2),)),
        _case("answerability-technical-concept", "calibration", "Explain Kubernetes", "memory", False, (CandidateSpec("kubernetes", "Kubernetes is an orchestration system that manages container workloads.", .2),)),
        _case("answerability-freshness", "calibration", "What is the latest Redis version?", "web", True, (CandidateSpec("redis-definition", "Redis is an in-memory data store.", .2),)),
        _case("answerability-multi-concept", "calibration", "Explain Redis persistence using RDB and AOF.", "memory", False, strong_redis),
        _case("eval-redis-insufficient", "evaluation", "Explain Redis persistence using RDB and AOF.", "web", False, (CandidateSpec("release", REDIS_RELEASE, .2),)),
        _case("eval-redis-good", "evaluation", "Explain Redis persistence using RDB and AOF.", "memory", False, strong_redis),
        _case("eval-fresh-latest", "evaluation", "latest stable Redis version", "web", True, strong_redis),
        _case("eval-fresh-news", "evaluation", "breaking news", "web", True, strong_rag),
        _case("eval-trap-architecture", "evaluation", "Explain current architecture", "memory", False, (CandidateSpec("architecture", "Current architecture explains component design.", .2),)),
        _case("eval-rag", "evaluation", "How does RAG retrieve relevant context?", "memory", False, strong_rag),
        _case("eval-split", "evaluation", "Compare RDB and AOF persistence", "memory", False, split_redis),
        _case("eval-rdb-only", "evaluation", "Compare RDB and AOF persistence", "web", False, (CandidateSpec("rdb", "RDB snapshot persistence.", .2),)),
        _case("eval-weak", "evaluation", "Explain FAISS", "web", False, (CandidateSpec("weak", "FAISS index", 1.1),)),
        _case("eval-empty", "evaluation", "Explain database indexing", "web"),
        _case("eval-two-source", "evaluation", "Explain HTTP caching", "memory", False, (CandidateSpec("cache-a", "HTTP caching stores responses.", .8), CandidateSpec("cache-b", "HTTP caching validates cached responses.", .9))),
        _case("eval-duplicate-source", "evaluation", "Explain HTTP caching", "web", False, (CandidateSpec("cache", "HTTP caching stores responses.", .8), CandidateSpec("cache", "HTTP caching stores responses.", .9))),
        _case("eval-followup", "evaluation", "Can you explain that further?", "memory", False, (CandidateSpec("follow", "Prior evidence.", .2),)),
        _case("eval-trap-flow", "evaluation", "current flow in electrical circuits", "memory", False, (CandidateSpec("flow", "Current flow in electrical circuits.", .2),)),
        _case("eval-crdt", "evaluation", "Explain CRDT tombstones", "memory", False, (CandidateSpec("crdt", "CRDT tombstones preserve delete state.", .2),)),
        _case("eval-fresh-pricing", "evaluation", "current pricing", "web", True, strong_rag),
    ]
    return tuple(cases)


FRESHNESS_DATASET = (
    ("latest Redis release", True), ("current CEO of a company", True), ("today's news", True),
    ("recent changes in LangChain", True), ("newest stable version", True), ("current pricing", True),
    ("current flow in an electrical circuit", False), ("news recommendation system", False),
    ("new student onboarding", False), ("current architecture", False), ("latest-element pointer", False),
    ("Explain FAISS", False),
)
