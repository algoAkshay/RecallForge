import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ui.components import append_assistant_message, build_copy_payload, complete_assistant_message, copy_payload, source_action_label
from ui.styles import workspace_css


class WorkspaceUiTests(unittest.TestCase):
    def test_workspace_css_follows_streamlit_native_theme_variables(self):
        css = workspace_css()
        for token in ("--background-color", "--secondary-background-color", "--text-color", "--border-color", "--primary-color"):
            self.assertIn(token, css)
        self.assertNotIn("prefers-color-scheme", css)
        self.assertNotIn("--rf-bg:#", css)

    def test_native_theme_has_readable_tokens_and_targeted_native_overrides(self):
        css = workspace_css()
        self.assertIn("textarea::placeholder", css)
        self.assertIn("[data-testid=\"stSidebar\"] p", css)
        self.assertIn("th,td{color:var(--rf-text)", css)
        self.assertIn("--rf-focus:var(--primary-color)", css)
        self.assertIn(".stButton>button:hover", css)
        self.assertNotIn("color:white", css.lower())

    def test_theme_css_adapts_portaled_native_surfaces_without_independent_theme_state(self):
        css = workspace_css()
        for token in ("--rf-surface-secondary", "--rf-sidebar", "--rf-input-bg", "--rf-code-bg", "--rf-shadow"):
            self.assertIn(token, css)
        for selector in ('[data-testid="stPopoverBody"]', 'ul[role="listbox"]', '[data-testid="stChatInput"]', '[data-testid="stExpander"]'):
            self.assertIn(selector, css)
        self.assertIn("transition:background-color 160ms ease", css)
        self.assertNotIn("transition:all", css.replace(" ", "").lower())

    def test_thread_actions_use_a_native_dialog_not_a_sidebar_popover(self):
        css = workspace_css()
        self.assertIn('[data-testid="stDialog"]{opacity:1!important', css)
        self.assertIn('[class*="st-key-recent_thread_row_"]', css)
        self.assertIn('text-overflow:ellipsis', css)
        self.assertIn("opacity:1!important", css)
        self.assertIn("background-color:var(--secondary-background-color)!important", css)
        self.assertNotIn("#stFloatingOverlayPortal:has", css)
        main_source = (Path(__file__).resolve().parents[1] / "src" / "main.py").read_text(encoding="utf-8")
        for label in ('@st_dialog("Chat options", width="small")', '"⋯"', '"Save name"', '"Export Markdown"', '"Delete chat"'):
            self.assertIn(label, main_source)
        self.assertNotIn('st.popover("Actions"', main_source)
        self.assertIn('key=f"thread_menu_{thread_id}"', main_source)
        self.assertIn('row_key = f"recent_thread_row_', main_source)

    def test_sidebar_has_no_custom_theme_control_or_theme_state(self):
        main_source = (Path(__file__).resolve().parents[1] / "src" / "main.py").read_text(encoding="utf-8")
        component_source = (Path(__file__).resolve().parents[1] / "src" / "ui" / "components.py").read_text(encoding="utf-8")
        self.assertNotIn("appearance", main_source.lower())
        self.assertNotIn("appearance", component_source.lower())
        self.assertNotIn("THEME_MODES", component_source)

    def test_response_actions_are_presentation_only(self):
        answer = "Answer body [S1]"
        self.assertEqual(copy_payload(answer), answer)
        self.assertNotIn("MEMORY", copy_payload(answer))
        self.assertEqual((source_action_label(False), source_action_label(True)), ("Open sources", "Hide sources"))

    def test_copy_payload_excludes_all_response_metadata(self):
        first = {"content": "Answer with `code` and \"quotes\"\n[S1]", "route": "MEMORY", "reason": "Stored evidence", "elapsed": "3s", "sources": "### Sources\nhttps://example.test"}
        second = {"content": "Second answer", "route": "WEB", "reason": "Freshness", "elapsed": "5s", "sources": "https://other.test"}
        payload = build_copy_payload(first)
        self.assertEqual(payload, "Answer with `code` and \"quotes\"\n[S1]")
        for excluded in ("MEMORY", "Stored evidence", "3s", "### Sources", "https://example.test"):
            self.assertNotIn(excluded, payload)
        self.assertNotEqual(payload, build_copy_payload(second))

    def test_copy_control_uses_safe_promise_aware_browser_clipboard(self):
        source = (Path(__file__).resolve().parents[1] / "src" / "ui" / "components.py").read_text(encoding="utf-8")
        self.assertIn("await navigator.clipboard.writeText(payload)", source)
        self.assertIn("Copy failed", source)
        self.assertIn('replace("<", "\\\\u003c")', source)

    def test_new_research_is_scoped_to_session_ui(self):
        source = (Path(__file__).resolve().parents[1] / "src" / "ui" / "components.py").read_text(encoding="utf-8")
        self.assertIn("st.session_state.messages = []", source)
        self.assertNotIn("research_memory", source)

    def test_sidebar_has_no_placeholder_navigation_and_preserves_its_core_controls(self):
        source = (Path(__file__).resolve().parents[1] / "src" / "ui" / "components.py").read_text(encoding="utf-8")
        for absent in ('"WORKSPACE"', '"Research", icon=', 'Memory · Coming soon', 'Evaluation · Coming soon', 'Settings · Coming soon'):
            self.assertNotIn(absent, source)
        for present in ('"New research"', '"RecallForge · Akshay Kumar"'):
            self.assertIn(present, source)
        self.assertNotIn('"Appearance"', source)

    def test_submission_order_and_stable_assistant_indices(self):
        messages = [{"role": "user", "content": "Q1"}]
        first = append_assistant_message(messages, {"role": "assistant", "content": "A1", "route": "MEMORY"})
        messages.append({"role": "user", "content": "Q2"})
        second = append_assistant_message(messages, {"role": "assistant", "content": "A2", "route": "WEB"})
        self.assertEqual([message["role"] for message in messages], ["user", "assistant", "user", "assistant"])
        self.assertEqual((first, second), (1, 3))
        self.assertEqual((messages[first]["route"], messages[second]["route"]), ("MEMORY", "WEB"))

    def test_complete_message_precedes_append_and_renderer_paths(self):
        message = complete_assistant_message("A1", "MEMORY", "Stored evidence", "3s", "### Sources")
        self.assertEqual(message, {"role": "assistant", "content": "A1", "route": "MEMORY", "reason": "Stored evidence", "elapsed": "3s", "sources": "### Sources"})
        source = (Path(__file__).resolve().parents[1] / "src" / "main.py").read_text(encoding="utf-8")
        self.assertIn("response_index = append_assistant_message", source)
        self.assertIn("render_response(assistant_message, response_index)", source)
        self.assertNotIn("render_response(message, index) if", source)


if __name__ == "__main__": unittest.main()
