import tempfile
import unittest
from pathlib import Path
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from storage.chat_history import add_message, create_thread, delete_thread, initialize_database, list_threads, load_messages, rename_thread, title_from_prompt
from ui.chat_export import export_chat_markdown, format_thread_timestamp, safe_export_filename
from ui.components import clear_deleted_active_thread

class ChatHistoryTests(unittest.TestCase):
    def setUp(self): self.temp=tempfile.TemporaryDirectory(); self.path=Path(self.temp.name)/"chat.db"
    def tearDown(self): self.temp.cleanup()
    def test_persistence_order_metadata_and_restart(self):
        initialize_database(self.path); thread=create_thread("What is RAG?",self.path)
        add_message(thread,{"role":"user","content":"What is RAG?"},self.path)
        add_message(thread,{"role":"assistant","content":"Answer [S1]","route":"MEMORY","reason":"Stored","elapsed":"3s","sources":"### Sources"},self.path)
        self.assertEqual([m["role"] for m in load_messages(thread,self.path)],["user","assistant"])
        answer=load_messages(thread,self.path)[1]; self.assertEqual((answer["route"],answer["reason"],answer["elapsed"],answer["sources"]),("MEMORY","Stored","3s","### Sources"))
        self.assertEqual(list_threads(self.path)[0]["id"],thread)
    def test_titles_quotes_isolation_and_delete(self):
        first=create_thread("Apostrophe's prompt",self.path); second=create_thread("Second",self.path)
        add_message(first,{"role":"user","content":"one"},self.path); add_message(second,{"role":"user","content":"two"},self.path)
        self.assertEqual(load_messages(first,self.path)[0]["content"],"one")
        delete_thread(first,self.path); self.assertEqual(load_messages(first,self.path),[])
        self.assertEqual(title_from_prompt("x "*40),("x "*40)[:53].rstrip()+"...")

    def test_delete_uses_thread_uuid_and_cascades_only_its_messages(self):
        first=create_thread("Same title",self.path); second=create_thread("Same title",self.path)
        add_message(first,{"role":"user","content":"first"},self.path)
        add_message(second,{"role":"user","content":"second"},self.path)
        delete_thread(first,self.path)
        self.assertEqual(load_messages(first,self.path),[])
        self.assertEqual(load_messages(second,self.path)[0]["content"],"second")
        self.assertEqual([thread["id"] for thread in list_threads(self.path)],[second])
        with self.assertRaises(KeyError): delete_thread(first,self.path)

    def test_rename_normalizes_title_and_preserves_messages(self):
        thread=create_thread("Original",self.path)
        add_message(thread,{"role":"user","content":"Keep this"},self.path)
        with patch("storage.chat_history._now", side_effect=["2026-01-01T00:00:00+00:00"]):
            self.assertEqual(rename_thread(thread,"  New\n  research   title  ",self.path),"New research title")
        self.assertEqual(list_threads(self.path)[0]["title"],"New research title")
        self.assertEqual(load_messages(thread,self.path)[0]["content"],"Keep this")
        with self.assertRaises(ValueError): rename_thread(thread," \n ",self.path)
        self.assertEqual(rename_thread(thread,"x" * 100,self.path),"x" * 80)

    def test_export_contains_visible_history_and_existing_sources_only(self):
        thread=create_thread("Source question",self.path)
        add_message(thread,{"role":"user","content":"What changed?"},self.path)
        add_message(thread,{"role":"assistant","content":"A change happened [S1].","route":"WEB","reason":"Freshness-sensitive","elapsed":"3s","sources":"### Sources\n\n[S1] **Official update** — https://example.test/update"},self.path)
        output=export_chat_markdown("Source question",load_messages(thread,self.path))
        self.assertIn("# Source question",output)
        self.assertIn("## User\n\nWhat changed?",output)
        self.assertIn("## RecallForge\n\nA change happened [S1].",output)
        self.assertIn("**Route:** WEB",output)
        self.assertIn("**Reason:** Freshness-sensitive",output)
        self.assertIn("**Duration:** 3s",output)
        self.assertIn("[S1] **Official update**",output)
        self.assertNotIn("URL unavailable",output)
        self.assertEqual(safe_export_filename("A / tricky: title!"),"a-tricky-title.md")

    def test_human_readable_timestamps(self):
        now=datetime(2026,8,28,12,0,tzinfo=timezone.utc)
        self.assertEqual(format_thread_timestamp((now-timedelta(seconds=20)).isoformat(),now),"Just now")
        self.assertEqual(format_thread_timestamp((now-timedelta(minutes=7)).isoformat(),now),"7m ago")
        self.assertEqual(format_thread_timestamp((now-timedelta(hours=3)).isoformat(),now),"3h ago")
        self.assertEqual(format_thread_timestamp((now-timedelta(days=1)).isoformat(),now),"Yesterday")
        self.assertEqual(format_thread_timestamp("invalid",now),"")

    def test_active_deleted_thread_clears_only_visible_thread_state(self):
        active={"active_thread_id":"a","thread_id":"a","messages":[{"role":"user","content":"x"}]}
        self.assertTrue(clear_deleted_active_thread(active,"a"))
        self.assertEqual(active,{"active_thread_id":None,"thread_id":None,"messages":[]})
        inactive={"active_thread_id":"a","thread_id":"a","messages":[{"role":"user","content":"x"}]}
        self.assertFalse(clear_deleted_active_thread(inactive,"b"))
        self.assertEqual(inactive["messages"][0]["content"],"x")
