import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.datasets import controlled_cases
from src.eval.metrics import ConfusionMetrics, RoutingMetrics
from src.eval.routing_benchmark import (
    CURRENT_PAIR, coverage_result, evaluate, freshness_result, route_case, routing_metrics,
    select_controlled_policy_optimum, split_cases, valid_threshold_pairs,
)


class EvaluationTests(unittest.TestCase):
    def test_controlled_dataset_has_fixed_complete_split(self):
        cases = controlled_cases()
        self.assertEqual(len(cases), 46)
        self.assertEqual((len(split_cases("calibration")), len(split_cases("evaluation"))), (30, 16))
        self.assertEqual({case.split for case in cases}, {"calibration", "evaluation"})

    def test_calibration_isolation_rejects_held_out_cases(self):
        with self.assertRaises(ValueError):
            select_controlled_policy_optimum(controlled_cases())

    def test_threshold_grid_is_valid_and_selection_is_deterministic(self):
        self.assertTrue(all(strong <= acceptable for strong, acceptable in valid_threshold_pairs()))
        first = select_controlled_policy_optimum(split_cases("calibration"))
        second = select_controlled_policy_optimum(split_cases("calibration"))
        self.assertEqual(first, second)
        self.assertEqual((first["strong"], first["acceptable"]), CURRENT_PAIR)

    def test_weighted_and_raw_count_metrics(self):
        metrics = RoutingMetrics(10, 7, 4, 6, 1, 2)
        self.assertEqual((metrics.weighted_error, metrics.accuracy), (4, .7))

    def test_required_baselines_and_current_policy(self):
        cases = split_cases("evaluation")
        always_web = routing_metrics(cases, policy="always_web")
        always_memory = routing_metrics(cases, policy="always_memory")
        semantic_only = routing_metrics(cases, policy="semantic_only")
        current = routing_metrics(cases, policy="current")
        self.assertEqual(always_web.web_decisions, len(cases))
        self.assertEqual(always_memory.web_decisions, 3)
        self.assertGreater(semantic_only.false_memory, current.false_memory)
        self.assertEqual(current.false_memory, 0)

    def test_redis_ablation_and_source_diversity(self):
        redis = next(case for case in split_cases("evaluation") if case.name == "eval-redis-insufficient")
        self.assertEqual(route_case(redis, policy="semantic_only"), "memory")
        self.assertEqual(route_case(redis, policy="current"), "web")
        result = evaluate()["controlled_policy"]["diversity_ablation"]
        self.assertGreater(result["diversity_disabled"]["false_memory"], result["diversity_aware"]["false_memory"])

    def test_freshness_and_coverage_confusion_metrics(self):
        freshness = freshness_result()
        coverage = coverage_result()
        self.assertEqual((freshness.tp, freshness.tn, freshness.fp, freshness.fn), (6, 6, 0, 0))
        self.assertEqual((coverage.tp, coverage.tn, coverage.fp, coverage.fn, coverage.neutral), (3, 2, 0, 0, 1))
        self.assertIsInstance(ConfusionMetrics(1, 1, 0, 0).as_dict(), dict)

    def test_integrity_closed_loop_safe_real_skip_and_json(self):
        result = evaluate()
        integrity = result["system_integrity"]
        self.assertTrue(integrity["persistence"]["post_restart_memory_reuse"])
        self.assertTrue(integrity["persistence"]["hash_reconstruction"])
        self.assertEqual(integrity["deduplication"], {"ingestion_attempts": 5, "semantic_insertions": 3, "duplicate_skips": 2, "same_content_different_url": True, "changed_content_same_url": True})
        self.assertTrue(all(integrity["citation_provenance"].values()))
        self.assertTrue(integrity["closed_loop"]["web_ingest_persist_restart_memory"])
        self.assertEqual(result["real_retrieval"]["status"], "not_executed")
        self.assertIsInstance(json.loads(json.dumps(result)), dict)
