import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from storage.chat_history import add_message, chat_history_path, create_thread, load_messages
from storage.paths import PROJECT_ROOT, research_memory_path, storage_root


class StoragePathTests(unittest.TestCase):
    def test_local_default_uses_project_storage_directory(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(storage_root(), PROJECT_ROOT / ".storage")
            self.assertEqual(research_memory_path(), PROJECT_ROOT / ".storage" / "research_memory")
            self.assertEqual(chat_history_path(), PROJECT_ROOT / ".storage" / "chat_history.db")

    def test_custom_storage_root_keeps_chat_history_across_calls(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"RECALLFORGE_STORAGE_DIR": temporary}, clear=True
        ):
            thread = create_thread("What is RAG?")
            add_message(thread, {"role": "user", "content": "What is RAG?"})
            self.assertEqual(chat_history_path(), Path(temporary) / "chat_history.db")
            self.assertTrue(chat_history_path().exists())
            self.assertEqual(load_messages(thread)[0]["content"], "What is RAG?")

    def test_legacy_memory_override_still_wins(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ,
            {"RECALLFORGE_STORAGE_DIR": "ignored", "LINKMIND_MEMORY_PATH": temporary},
            clear=True,
        ):
            self.assertEqual(research_memory_path(), Path(temporary))
