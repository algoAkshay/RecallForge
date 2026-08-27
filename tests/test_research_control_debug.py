import sys
import asyncio
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ui.debug_panel import build_debug_rows, format_duration, parse_web_diagnostics
from ui.research_control import FORCE_WEB_REASON, consume_search_fresh, forced_web_decision
from agents.fallback import SynthesisResult, synthesize_with_single_fallback
from storage.chat_history import add_message, create_thread, load_messages


class SearchFreshControlTests(unittest.TestCase):
    def test_auto_control_is_not_forced_and_next_query_uses_new_checkbox_key(self):
        state = {"search_fresh_control_version": 3}
        self.assertFalse(consume_search_fresh(state, False))
        self.assertEqual(state["search_fresh_control_version"], 4)
        self.assertTrue(consume_search_fresh(state, True))
        self.assertEqual(state["search_fresh_control_version"], 5)

    def test_forced_web_decision_is_distinct_and_fresh(self):
        decision = forced_web_decision()
        self.assertEqual((decision.route, decision.reason, decision.freshness_sensitive), ("web", FORCE_WEB_REASON, True))
        self.assertEqual(decision.reason_code, "force_web_requested")

    def test_forced_web_stays_on_existing_web_path_without_memory_fallback(self):
        calls = []

        async def synthesize(route, _evidence):
            calls.append(route)
            return SynthesisResult.from_answer("I couldn't find enough relevant evidence to answer this confidently.")

        async def acquire_web():
            raise AssertionError("A WEB request must not fall back to another acquisition.")

        outcome = asyncio.run(synthesize_with_single_fallback(
            initial_route=forced_web_decision().route,
            initial_reason=FORCE_WEB_REASON,
            initial_evidence="WEB evidence",
            synthesize=synthesize,
            acquire_web=acquire_web,
            replace_request_evidence=lambda: None,
        ))
        self.assertEqual((calls, outcome.diagnostics.final_route, outcome.diagnostics.fallback_attempted), (["web"], "web", False))

    def test_forced_route_and_reason_use_existing_chat_persistence_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chat.db"
            thread = create_thread("question", path)
            add_message(thread, {"role": "assistant", "content": "answer", "route": "WEB", "reason": FORCE_WEB_REASON}, path)
            restored = load_messages(thread, path)[0]
        self.assertEqual((restored["route"], restored["reason"]), ("WEB", FORCE_WEB_REASON))


class DebugPanelTests(unittest.TestCase):
    def test_memory_payload_contains_only_measured_memory_fields(self):
        rows = dict(build_debug_rows({
            "route": "MEMORY", "reason": "Stored evidence", "elapsed": "3s",
            "debug": {"routing_mode": "AUTO", "initial_route": "MEMORY", "final_route": "MEMORY", "fallback_attempted": False, "total_seconds": 3.1, "memory_retrieval_seconds": 0.2, "synthesis_seconds": 2.7, "source_count": 2},
        }))
        self.assertEqual(rows["Mode"], "AUTO")
        self.assertEqual(rows["Fallback"], "No")
        self.assertEqual(rows["Memory retrieval"], "0.2s")
        self.assertEqual(rows["Sources"], "2")
        self.assertNotIn("Search", rows)

    def test_web_diagnostics_preserve_zero_and_safe_failure_counts(self):
        evidence = "EVIDENCE\n\nDiagnostics:\nsearch_seconds: 0.000\nfetch_extract_seconds: 5.600\nembedding_index_seconds: 1.200\ntotal_web_acquisition_seconds: 6.800\nsearch_result_count: 4\nfetch_attempt_count: 4\nfetch_success_count: 3\nfetch_timeout_count: 1\nfetch_failure_count: 0\nsearch_timeout: false\nacquisition_deadline_reached: true\napi_key: secret"
        parsed = parse_web_diagnostics(evidence)
        self.assertEqual(parsed["search_seconds"], 0.0)
        self.assertEqual(parsed["fetch_timeout_count"], 1)
        self.assertNotIn("api_key", parsed)
        rows = dict(build_debug_rows({"route": "WEB", "debug": {"routing_mode": "FORCE_WEB", "total_seconds": 10.8, "source_count": 4, **parsed}}))
        self.assertEqual(rows["Mode"], "FORCE_WEB")
        self.assertEqual(rows["Search"], "0.0s")
        self.assertEqual(rows["Fetch timeouts"], "1")
        self.assertEqual(rows["Acquisition deadline reached"], "Yes")

    def test_fallback_and_historical_minimal_payloads_render_safely(self):
        fallback = dict(build_debug_rows({"route": "WEB", "reason": "Fresh fallback", "debug": {"fallback_attempted": True, "initial_route": "MEMORY", "final_route": "WEB"}}))
        self.assertEqual((fallback["Fallback"], fallback["Initial route"], fallback["Final route"]), ("Yes", "MEMORY", "WEB"))
        historical = dict(build_debug_rows({"route": "MEMORY", "reason": "Stored evidence", "elapsed": "3s", "sources": "### Sources"}))
        self.assertEqual(historical, {"Route": "MEMORY", "Reason": "Stored evidence", "Total": "3s"})

    def test_unknown_internal_debug_fields_are_never_rendered(self):
        rows = dict(build_debug_rows({"route": "WEB", "debug": {"routing_mode": "AUTO", "prompt": "hidden", "embedding": [1, 2], "api_key": "secret", "traceback": "hidden"}}))
        self.assertEqual(rows, {"Route": "WEB", "Mode": "AUTO"})
        self.assertEqual(format_duration(0), "0.0s")
