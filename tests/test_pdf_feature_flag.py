import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import ENABLE_PDF_FEATURES
from tools import relevance_search
from ui.debug_panel import build_debug_rows


class PdfFeatureFlagTests(unittest.IsolatedAsyncioTestCase):
    class DB:
        def __init__(self):
            self.documents = [
                types.SimpleNamespace(page_content="Uploaded PDF content", metadata={"origin": "upload", "document_hash": "pdf"}),
                types.SimpleNamespace(page_content="Web research content", metadata={"origin": "web", "document_content_hash": "web"}),
            ]
            self.filters = []

        async def asimilarity_search_with_score(self, _query, k, filter=None):
            self.filters.append(filter)
            rows = [(document, 0.2) for document in self.documents]
            return [row for row in rows if filter is None or filter(row[0].metadata)][:k]

    async def test_disabled_mode_excludes_upload_origin_from_global_memory(self):
        db = self.DB()
        with patch.object(relevance_search, "fetch_model", new=AsyncMock(return_value=db)):
            candidates = await relevance_search.retrieve_memory_candidates("research")
        self.assertFalse(ENABLE_PDF_FEATURES)
        self.assertEqual([document.metadata.get("origin") for document, _ in candidates], ["web"])
        self.assertTrue(callable(db.filters[0]))

    async def test_web_origin_memory_remains_usable_when_pdf_is_disabled(self):
        db = self.DB()
        with patch.object(relevance_search, "fetch_model", new=AsyncMock(return_value=db)):
            candidates = await relevance_search.retrieve_memory_candidates("web")
        self.assertEqual(candidates[0][0].page_content, "Web research content")

    async def test_pdf_modules_remain_importable_and_historical_pdf_message_is_safe(self):
        for name in ("documents.pdf_parser", "documents.chunking", "documents.ingestion", "documents.registry"):
            self.assertIsNotNone(importlib.import_module(name))
        rows = dict(build_debug_rows({"route": "MEMORY", "reason": "Stored PDF evidence", "elapsed": "2s", "sources": "**file.pdf** — Page 1"}))
        self.assertEqual(rows["Route"], "MEMORY")


class PdfUiGateTests(unittest.TestCase):
    def test_pdf_ui_and_active_document_path_are_guarded_by_the_default_off_flag(self):
        source = (Path(__file__).resolve().parents[1] / "src" / "main.py").read_text(encoding="utf-8")
        self.assertIn("if ENABLE_PDF_FEATURES and active_documents:", source)
        for leaked in ("Legacy inline PDF sidebar", "st.file_uploader", "uploaded_pdfs", "pdf_upload_statuses"):
            self.assertNotIn(leaked, source)
        self.assertIn("ENABLE_PDF_FEATURES = False", (Path(__file__).resolve().parents[1] / "src" / "features.py").read_text(encoding="utf-8"))
