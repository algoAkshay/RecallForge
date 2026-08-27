import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.fallback import (
    FALLBACK_FAILURE,
    FALLBACK_REASON,
    INSUFFICIENT_EVIDENCE_SENTINEL,
    SynthesisResult,
    synthesize_with_single_fallback,
)
from tools.provenance import EvidenceItem, citation_collection, register_evidence, validate_citations
from ui.components import append_assistant_message, complete_assistant_message
from storage.chat_history import add_message, create_thread, load_messages


class MemoryWebFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_insufficient_memory_replaces_provenance_and_returns_web_answer_once(self):
        syntheses, acquisitions = [], []
        with citation_collection() as collector:
            register_evidence([
                EvidenceItem("old one", "https://memory-one.test", document_content_hash="m1"),
                EvidenceItem("old two", "https://memory-two.test", document_content_hash="m2"),
            ])

            async def synthesize(route, _evidence):
                syntheses.append(route)
                if route == "memory":
                    self.assertEqual(set(collector.source_map), {"S1", "S2"})
                    return SynthesisResult(INSUFFICIENT_EVIDENCE_SENTINEL, False)
                self.assertEqual(set(collector.source_map), {"S1"})
                self.assertEqual(collector.source_map["S1"].source_url, "https://web.test")
                return SynthesisResult("Fresh answer [S1]", True)

            async def acquire_web():
                acquisitions.append("web")
                register_evidence([EvidenceItem("fresh", "https://web.test", document_content_hash="w1", origin="web")])
                return "STATUS: SUCCESS\n\nEVIDENCE\n[S1]\nURL: https://web.test"

            outcome = await synthesize_with_single_fallback(
                initial_route="memory", initial_reason="Stored evidence", initial_evidence="memory evidence",
                synthesize=synthesize, acquire_web=acquire_web, replace_request_evidence=collector.clear,
            )
            valid, invalid = validate_citations("Fresh answer [S1] stale [S2]", collector.source_map)

            self.assertEqual((outcome.answer, outcome.reason), ("Fresh answer [S1]", FALLBACK_REASON))
            self.assertEqual((outcome.diagnostics.final_route, outcome.diagnostics.synthesis_count), ("web", 2))
            self.assertTrue(outcome.diagnostics.fallback_attempted)
            self.assertEqual((syntheses, acquisitions), (["memory", "web"], ["web"]))
            self.assertEqual((valid, invalid), ({"S1"}, {"S2"}))
            self.assertEqual(collector.source_map["S1"].source_url, "https://web.test")

    async def test_sufficient_memory_never_calls_web(self):
        calls = []

        async def synthesize(route, _evidence):
            calls.append(route)
            return SynthesisResult("Memory answer [S1]", True)

        async def acquire_web():
            self.fail("WEB must not be called for sufficient MEMORY")

        outcome = await synthesize_with_single_fallback(
            initial_route="memory", initial_reason="Stored evidence", initial_evidence="memory",
            synthesize=synthesize, acquire_web=acquire_web, replace_request_evidence=lambda: None,
        )
        self.assertEqual((outcome.diagnostics.fallback_attempted, outcome.diagnostics.synthesis_count, calls), (False, 1, ["memory"]))

    async def test_original_web_route_never_enters_fallback(self):
        calls = []

        async def synthesize(route, _evidence):
            calls.append(route)
            return SynthesisResult("Web answer", True)

        async def acquire_web():
            self.fail("Original WEB routes already acquired evidence")

        outcome = await synthesize_with_single_fallback(
            initial_route="web", initial_reason="Freshness", initial_evidence="web",
            synthesize=synthesize, acquire_web=acquire_web, replace_request_evidence=lambda: None,
        )
        self.assertEqual((outcome.diagnostics.fallback_attempted, outcome.diagnostics.synthesis_count, calls), (False, 1, ["web"]))

    async def test_failed_web_acquisition_returns_truthful_failure_without_retry(self):
        for evidence in (
            "STATUS: FAILURE\n\nSearch could not be completed.",
            "STATUS: PARTIAL_FAILURE\n\nFETCH_TIMEOUT\nURL: https://slow.test",
        ):
            with self.subTest(evidence=evidence):
                calls, resets = [], []

                async def synthesize(route, _evidence):
                    calls.append(route)
                    return SynthesisResult(INSUFFICIENT_EVIDENCE_SENTINEL, False)

                async def acquire_web():
                    return evidence

                outcome = await synthesize_with_single_fallback(
                    initial_route="memory", initial_reason="Stored evidence", initial_evidence="memory",
                    synthesize=synthesize, acquire_web=acquire_web, replace_request_evidence=lambda: resets.append("clear"),
                )
                self.assertEqual((outcome.answer, outcome.diagnostics.synthesis_count, calls), (FALLBACK_FAILURE, 1, ["memory"]))
                self.assertTrue(outcome.diagnostics.fallback_attempted)
                self.assertEqual(resets, ["clear", "clear"])

    async def test_insufficient_web_synthesis_stops_after_two_syntheses(self):
        calls = []

        async def synthesize(route, _evidence):
            calls.append(route)
            return SynthesisResult(INSUFFICIENT_EVIDENCE_SENTINEL, False)

        async def acquire_web():
            return "STATUS: SUCCESS\n\nEVIDENCE\n[S1]\nURL: https://web.test"

        outcome = await synthesize_with_single_fallback(
            initial_route="memory", initial_reason="Stored evidence", initial_evidence="memory",
            synthesize=synthesize, acquire_web=acquire_web, replace_request_evidence=lambda: None,
        )
        self.assertEqual((outcome.answer, outcome.diagnostics.synthesis_count, calls), (FALLBACK_FAILURE, 2, ["memory", "web"]))

    async def test_only_final_fallback_message_is_constructed_for_history_and_rendering(self):
        messages = [{"role": "user", "content": "question"}]
        final = complete_assistant_message("Fresh answer [S1]", "WEB", FALLBACK_REASON, "4s", "### Sources\n[S1] web")
        append_assistant_message(messages, final)
        assistant_messages = [message for message in messages if message["role"] == "assistant"]
        self.assertEqual(len(assistant_messages), 1)
        self.assertEqual((assistant_messages[0]["route"], assistant_messages[0]["sources"]), ("WEB", "### Sources\n[S1] web"))

    async def test_only_final_web_message_is_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "history.db"
            thread = create_thread("question", database)
            add_message(thread, {"role": "user", "content": "question"}, database)
            final = complete_assistant_message("Fresh answer [S1]", "WEB", FALLBACK_REASON, "4s", "### Sources\n[S1] web")
            add_message(thread, final, database)
            messages = load_messages(thread, database)

        self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
        self.assertEqual((messages[-1]["route"], messages[-1]["sources"]), ("WEB", "### Sources\n[S1] web"))


class SynthesisResultTests(unittest.TestCase):
    def test_only_exact_sentinel_marks_insufficient(self):
        self.assertFalse(SynthesisResult.from_answer("  I couldn’t find enough relevant evidence to answer this confidently. ").sufficient)
        self.assertTrue(SynthesisResult.from_answer("I couldn't find enough relevant evidence, but here is a guess.").sufficient)
