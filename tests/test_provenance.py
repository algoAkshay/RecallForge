import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tools.provenance import (
    EvidenceItem, citation_collection, evidence_from_document, format_evidence,
    group_evidence, register_evidence, remove_invalid_citations, render_sources,
    validate_citations,
)


class Document:
    def __init__(self, content, metadata):
        self.page_content, self.metadata = content, metadata


class ProvenanceTests(unittest.TestCase):
    def test_canonical_metadata_variants(self):
        for key in ("source_url", "source", "link"):
            item = evidence_from_document(Document("text", {key: "https://a.test", "title": "A", "document_content_hash": "h", "retrieved_at": "now"}))
            self.assertEqual(item.source_url, "https://a.test")
            self.assertEqual((item.title, item.document_content_hash, item.retrieved_at), ("A", "h", "now"))

    def test_grouping_is_deterministic_and_version_aware(self):
        items = [EvidenceItem("A1", "https://a", document_content_hash="h1"), EvidenceItem("A2", "https://a", document_content_hash="h1"), EvidenceItem("B", "https://a", document_content_hash="h2")]
        first, second = group_evidence(items), group_evidence(items)
        self.assertEqual([s.citation_id for s in first], ["S1", "S2"])
        self.assertEqual([s.citation_id for s in second], ["S1", "S2"])
        self.assertEqual([len(s.evidence) for s in first], [2, 1])

    def test_legacy_url_fallback_groups_chunks(self):
        sources = group_evidence([EvidenceItem("one", "https://a"), EvidenceItem("two", "https://a")])
        self.assertEqual(len(sources), 1)

    def test_live_and_memory_use_identical_format(self):
        web = group_evidence([EvidenceItem("web", "https://a", "A", "now", "h", "web")])
        memory = group_evidence([EvidenceItem("memory", "https://b", "B", "now", "x", "memory")])
        for rendered in (format_evidence(web), format_evidence(memory)):
            self.assertIn("[S1]", rendered); self.assertIn("URL:", rendered); self.assertIn("Evidence:", rendered)

    def test_validation_ignores_non_source_brackets_and_renders_authoritatively(self):
        sources = group_evidence([EvidenceItem("a", "https://a", "A", document_content_hash="h1"), EvidenceItem("b", "https://b", "B", document_content_hash="h2")])
        source_map = {source.citation_id: source for source in sources}
        valid, invalid = validate_citations("Revenue [2026]; A [S1]; unknown [S8].", source_map)
        self.assertEqual(valid, {"S1"}); self.assertEqual(invalid, {"S8"})
        self.assertNotIn("[S8]", remove_invalid_citations("A [S1] B [S8]", invalid))
        rendered = render_sources(source_map, valid)
        self.assertIn("https://a", rendered); self.assertNotIn("https://b", rendered)

    def test_no_evidence_and_turns_are_isolated(self):
        self.assertEqual(render_sources({}, set()), "")
        self.assertEqual(validate_citations("plain [2026]", {}), (set(), set()))
        with citation_collection() as first:
            register_evidence([EvidenceItem("a", "https://a", document_content_hash="h1")])
        with citation_collection() as second:
            register_evidence([EvidenceItem("b", "https://b", document_content_hash="h2")])
        self.assertEqual(first.source_map["S1"].source_url, "https://a")
        self.assertEqual(second.source_map["S1"].source_url, "https://b")


if __name__ == "__main__": unittest.main()
