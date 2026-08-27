import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BrandingTests(unittest.TestCase):
    def test_user_facing_branding_uses_recallforge(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        main = (ROOT / "src" / "main.py").read_text(encoding="utf-8")
        agent = (ROOT / "src" / "agents" / "agent.py").read_text(encoding="utf-8")
        self.assertIn("# RecallForge", readme)
        self.assertIn('page_title="RecallForge"', main)
        self.assertIn("You are RecallForge", agent)
        self.assertIn("Akshay Kumar", agent)
        self.assertNotIn("Link" + "Mind", readme + main + agent)


if __name__ == "__main__":
    unittest.main()
