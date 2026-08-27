import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tools.presentation import normalize_source_title, render_authoritative_sources, sanitize_answer_markdown
from tools.provenance import EvidenceItem, group_evidence


class PresentationTests(unittest.TestCase):
    def test_trailing_bibliography_and_svg_artifacts_are_removed(self):
        text = "### Explanation[svg](http://localhost:8501/#explanation)\nUseful [S1].\nsvg\n![x](data:image/svg+xml,abc)\n### Sources\n[S1] model list"
        clean = sanitize_answer_markdown(text)
        self.assertEqual(clean, "### Explanation\nUseful [S1].")

    def test_markdown_and_false_positive_content_survive(self):
        text = "[Google](https://google.com)\n![Architecture](https://example.com/a.png)\n[S1]\n### Sources of electrical noise\nThe SVG format is useful.\n[SVG specification](https://www.w3.org/TR/SVG2/)\n\n| A | B |\n|---|---|\n| x [S1] | y |\n\n```\nsvg\n```"
        self.assertEqual(sanitize_answer_markdown(text), text)

    def test_titles_fallback_and_independent_authoritative_entries(self):
        sources = group_evidence([
            EvidenceItem("a", "https://redis.io/docs", " Redis Docs   Redis Docs ", document_content_hash="a"),
            EvidenceItem("b", "https://example.com/path", "", document_content_hash="b"),
        ])
        rendered = render_authoritative_sources({s.citation_id: s for s in sources}, {"S1", "S2"})
        self.assertEqual(normalize_source_title(" A\n A "), "A")
        self.assertEqual(rendered.count("### Sources"), 1)
        self.assertIn("**Redis Docs**", rendered)
        self.assertIn("**example.com**", rendered)
        self.assertIn("[https://redis.io/docs](https://redis.io/docs)", rendered)

    def test_synthesis_instruction_owns_no_bibliography(self):
        agent = (Path(__file__).resolve().parents[1] / "src" / "agents" / "agent.py").read_text(encoding="utf-8")
        self.assertIn("Do not generate Sources, References", agent)

    def test_observed_concatenated_three_column_header_is_repaired(self):
        text = "| Feature / AspectRDB (Redis Database)AOF (Append Only File) |\n| --- | --- | --- |\n| Durability | Lower [S1] | Higher [S2] |"
        expected = "| Feature / Aspect | RDB (Redis Database) | AOF (Append Only File) |\n| --- | --- | --- |\n| Durability | Lower [S1] | Higher [S2] |"
        self.assertEqual(sanitize_answer_markdown(text), expected)

    def test_tables_and_pipe_text_are_safe_when_not_high_confidence_repairs(self):
        valid = "| Feature | RDB | AOF |\n| --- | --- | --- |\n| **Durability** | `low | value` [S1] | Higher [S2] |"
        ambiguous = "| Feature RDB AOF |\n| --- | --- | --- |\n| x | y | z |"
        fenced = "```text\n| FeatureRDB |\n| --- | --- |\n| x | y |\n```"
        self.assertEqual(sanitize_answer_markdown(valid), valid)
        self.assertEqual(sanitize_answer_markdown(ambiguous), ambiguous)
        self.assertEqual(sanitize_answer_markdown("Use A | B to describe alternatives."), "Use A | B to describe alternatives.")
        self.assertEqual(sanitize_answer_markdown(fenced), fenced)


if __name__ == "__main__": unittest.main()
