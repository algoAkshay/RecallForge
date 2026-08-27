import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tools import routing
from tools.routing import (
    MAX_MEMORY_L2_DISTANCE, STRONG_MEMORY_L2_DISTANCE, choose_route, detect_freshness,
    detect_short_broad_query, evaluate_evidence_coverage,
)

class Doc:
    def __init__(self, key, content=""): 
        self.metadata = {"document_content_hash": key}
        self.page_content = content

class RoutingTests(unittest.TestCase):
    def test_freshness_positives_and_false_positive_protection(self):
        positives = ["What is the latest Gemini release?", "What is OpenAI's current pricing?", "Who is the current CEO of Company X?", "What happened in AI news today?", "What changed recently in LangChain?", "What is the newest stable release of package X?", "today's news", "recent changes in LangChain"]
        negatives = ["Explain FAISS", "Explain Newton's laws to a new student.", "How does a news recommendation engine work?", "What is current flow in electrical circuits?", "Explain the latest-element pointer."]
        self.assertTrue(all(detect_freshness(query) for query in positives))
        self.assertTrue(all(not detect_freshness(query) for query in negatives))

    def test_routes_are_deterministic_and_distance_direction_is_correct(self):
        strong = [(Doc("a"), 0.4)]
        weak = [(Doc("a"), MAX_MEMORY_L2_DISTANCE + 0.01)]
        self.assertEqual(choose_route("Explain architecture X", strong).route, "memory")
        self.assertEqual(choose_route("Explain topic Y", weak).reason_code, "memory_insufficient")
        self.assertEqual(choose_route("Explain topic Y", []).reason_code, "memory_empty")
        self.assertEqual(choose_route("Explain architecture X", strong), choose_route("Explain architecture X", strong))

    def test_freshness_overrides_excellent_memory_without_needing_candidates(self):
        decision = choose_route("What is the latest Gemini model?", [(Doc("a"), 0.0)])
        self.assertEqual((decision.route, decision.reason_code, decision.freshness_sensitive), ("web", "freshness_required", True))

    def test_multi_source_acceptable_memory_is_sufficient(self):
        decision = choose_route("Explain architecture", [
            (Doc("a", "Architecture is a system that organizes components."), 0.8),
            (Doc("b", "Architecture provides a structure for component interaction."), 0.9),
        ])
        self.assertEqual((decision.route, decision.reason_code), ("memory", "memory_sufficient"))

    def test_redis_release_evidence_cannot_answer_rdb_and_aof_question(self):
        decision = choose_route("Explain Redis persistence using RDB and AOF.", [
            (Doc("release", "Redis release support and lifecycle information."), 0.2),
        ])
        self.assertEqual((decision.route, decision.reason_code), ("web", "memory_insufficient_coverage"))
        self.assertEqual(decision.missing_required_concepts, ("rdb", "aof"))

    def test_both_required_concepts_allow_memory(self):
        evidence = "RDB creates snapshots for persistence. AOF records append only commands for persistence."
        self.assertEqual(choose_route("Compare RDB and AOF persistence.", [(Doc("both", evidence), 0.2)]).route, "memory")

    def test_one_missing_required_concept_forces_web(self):
        decision = choose_route("Compare RDB and AOF persistence.", [(Doc("rdb", "RDB snapshot persistence."), 0.2)])
        self.assertEqual((decision.route, decision.reason_code), ("web", "memory_insufficient_coverage"))
        self.assertEqual(decision.missing_required_concepts, ("aof",))

    def test_concepts_across_acceptable_chunks_jointly_satisfy_coverage(self):
        candidates = [(Doc("rdb", "RDB creates persistence snapshots."), 0.8), (Doc("aof", "AOF appends log commands."), 0.9)]
        self.assertEqual(choose_route("Compare RDB and AOF persistence.", candidates).route, "memory")

    def test_acronym_matching_is_boundary_aware_and_case_insensitive(self):
        wrong = evaluate_evidence_coverage("Explain RAG", [(Doc("wrong", "Storage is durable."), 0.2)])
        right = evaluate_evidence_coverage("Explain RDB", [(Doc("right", "rdb snapshots are durable."), 0.2)])
        self.assertEqual(wrong.missing_required_concepts, ("rag",))
        self.assertEqual(right.missing_required_concepts, ())

    def test_lowercase_technical_concepts_can_pass(self):
        evidence = "Snapshot persistence saves state. Append-only logging records each command."
        self.assertEqual(choose_route("Compare snapshot persistence and append-only logging.", [(Doc("lowercase", evidence), 0.2)]).route, "memory")

    def test_generic_follow_up_is_neutral(self):
        self.assertEqual(choose_route("How does it work?", [(Doc("followup", "Unrelated evidence."), 0.2)]).route, "memory")

    def test_semantic_failure_remains_semantic_failure(self):
        decision = choose_route("Explain Redis persistence using RDB and AOF.", [(Doc("weak", "RDB and AOF persistence."), 1.1)])
        self.assertEqual(decision.reason_code, "memory_insufficient")

    def test_freshness_does_not_invoke_coverage(self):
        original = routing.evaluate_evidence_coverage
        calls = 0
        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)
        routing.evaluate_evidence_coverage = counted
        try:
            decision = choose_route("latest Redis release", [(Doc("current", "Redis RDB AOF"), 0.0)])
        finally:
            routing.evaluate_evidence_coverage = original
        self.assertEqual((decision.route, decision.reason_code, calls), ("web", "freshness_required", 0))

    def test_closed_loop_redis_memory_improves_after_ingestion(self):
        query = "Explain Redis persistence using RDB and AOF."
        before = [(Doc("release", "Redis release support and lifecycle information."), 0.2)]
        after = before + [(Doc("persistence", "Redis RDB snapshots persist data and AOF append-only logging records writes."), 0.2)]
        self.assertEqual(choose_route(query, before).route, "web")
        self.assertEqual(choose_route(query, after).route, "memory")

    def test_short_definition_entity_overlap_alone_is_insufficient(self):
        decision = choose_route("What is Redis?", [
            (Doc("persistence", "Redis persistence uses RDB snapshots and AOF logs for durability."), .2),
        ])
        self.assertEqual((decision.route, decision.reason_code), ("web", "memory_insufficient_answerability"))

    def test_short_definition_accepts_actual_definition_memory(self):
        decision = choose_route("What is Redis?", [
            (Doc("definition", "Redis is an in-memory data store commonly used as a cache and message broker."), .2),
        ])
        self.assertEqual((decision.route, decision.reason_code), ("memory", "memory_strong_match"))

    def test_short_definition_patterns_normalize_generically(self):
        cases = (
            ("what's kafka", "Kafka is a distributed event streaming platform for reliable data pipelines."),
            ("DEFINE DOCKER", "Docker is a platform for building and running containerized applications."),
            ("Explain Kubernetes", "Kubernetes is an orchestration system that manages container workloads."),
            ("Who is Ada Lovelace?", "Ada Lovelace was a mathematician who wrote an early algorithm for a computer."),
        )
        for query, definition in cases:
            with self.subTest(query=query):
                self.assertIsNotNone(detect_short_broad_query(query))
                self.assertEqual(choose_route(query, [(Doc(query, definition), .2)]).route, "memory")

    def test_short_query_rule_has_no_entity_specific_exception(self):
        decision = choose_route("What is Kafka?", [
            (Doc("kafka-specialized", "Kafka replication uses partitions and leader brokers for durability."), .2),
        ])
        self.assertEqual((decision.route, decision.reason_code), ("web", "memory_insufficient_answerability"))

    def test_multi_concept_technical_query_and_coverage_are_unchanged(self):
        decision = choose_route("Explain Redis persistence using RDB and AOF.", [
            (Doc("redis", "Redis persistence uses RDB snapshots and AOF append-only logs for durability."), .2),
        ])
        self.assertEqual(decision.route, "memory")

    def test_freshness_precedes_short_query_answerability(self):
        decision = choose_route("What is the latest Redis version?", [(Doc("definition", "Redis is a data store."), .0)])
        self.assertEqual((decision.route, decision.reason_code), ("web", "freshness_required"))

    def test_faiss_thresholds_remain_fixed(self):
        self.assertEqual((STRONG_MEMORY_L2_DISTANCE, MAX_MEMORY_L2_DISTANCE), (.5, 1.0))

if __name__ == "__main__": unittest.main()
