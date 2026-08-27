import asyncio
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from documents.chunking import chunk_parsed_pdf
from documents.ingestion import UploadResult, activate_uploaded_documents, ingest_pdf_batch, ingest_pdf_bytes
from documents.registry import IndexedDocument, activate_indexed_document, delete_indexed_document, get_document_chunk_ids, list_indexed_documents, remove_document_from_active_state
from documents.pdf_parser import ParsedPDF, ParsedPage, PDFNoExtractableTextError, PDFPasswordProtectedError, PDFValidationError, document_hash, parse_pdf_bytes
from tools.provenance import EvidenceItem, citation_collection, evidence_from_document, register_evidence, validate_citations
from tools.presentation import render_authoritative_sources
from tools import relevance_search
from tools.relevance_search import _filename_key, _lexical_tokens, is_document_context_query, rank_memory_candidates, retrieve_active_document_candidates


def pdf_bytes(*pages: str, password: bool = False) -> bytes:
    document = fitz.open()
    for text in pages:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    options = {"encryption": fitz.PDF_ENCRYPT_AES_256, "owner_pw": "owner", "user_pw": "secret"} if password else {}
    data = document.tobytes(**options)
    document.close()
    return data


class PDFParserTests(unittest.TestCase):
    def test_extracts_multiple_one_based_pages_and_stable_identity(self):
        data = pdf_bytes("First page", "Second page")
        parsed = parse_pdf_bytes("paper.pdf", data)
        self.assertEqual((parsed.page_count, [page.page_number for page in parsed.pages]), (2, [1, 2]))
        self.assertEqual([page.text for page in parsed.pages], ["First page", "Second page"])
        self.assertEqual(parsed.document_hash, document_hash(data))
        self.assertEqual(parse_pdf_bytes("renamed.pdf", data).document_hash, parsed.document_hash)

    def test_rejects_invalid_size_page_limit_password_and_empty_pdf(self):
        with self.assertRaises(PDFValidationError):
            parse_pdf_bytes("not-a-pdf.pdf", b"not pdf")
        with self.assertRaises(PDFValidationError) as size:
            parse_pdf_bytes("large.pdf", pdf_bytes("x"), max_size_mb=0)
        self.assertEqual(size.exception.status, "FILE_TOO_LARGE")
        self.assertEqual(parse_pdf_bytes("one.pdf", pdf_bytes("one"), max_pages=1).page_count, 1)
        with self.assertRaises(PDFValidationError) as pages:
            parse_pdf_bytes("two.pdf", pdf_bytes("one", "two"), max_pages=1)
        self.assertEqual(pages.exception.status, "TOO_MANY_PAGES")
        with self.assertRaises(PDFPasswordProtectedError):
            parse_pdf_bytes("locked.pdf", pdf_bytes("secret", password=True))
        with self.assertRaises(PDFNoExtractableTextError):
            parse_pdf_bytes("empty.pdf", pdf_bytes(""))


class PDFChunkingTests(unittest.TestCase):
    def test_chunks_are_page_aware_deterministic_and_keep_metadata(self):
        parsed = ParsedPDF("paper.pdf", 2, [ParsedPage(1, "word " * 3000), ParsedPage(2, "Second page text")], "document-hash")
        first = chunk_parsed_pdf(parsed, chunk_size=500, chunk_overlap=100)
        second = chunk_parsed_pdf(parsed, chunk_size=500, chunk_overlap=100)
        self.assertGreater(len(first), 2)
        self.assertEqual([chunk.page_content for chunk in first], [chunk.page_content for chunk in second])
        self.assertTrue(all(chunk.metadata["filename"] == "paper.pdf" for chunk in first))
        self.assertTrue(all(chunk.metadata["origin"] == "upload" for chunk in first))
        self.assertTrue(all(chunk.metadata["page_start"] == chunk.metadata["page_end"] == chunk.metadata["page_number"] for chunk in first))
        self.assertEqual({chunk.metadata["page_number"] for chunk in first if "Second page" in chunk.page_content}, {2})
        self.assertTrue(all(chunk.metadata["document_content_hash"] for chunk in first))


class _FakeDB:
    def __init__(self):
        self.docstore = types.SimpleNamespace(_dict={})

    def delete(self, ids):
        for identifier in ids:
            del self.docstore._dict[identifier]


class PDFIngestionTests(unittest.IsolatedAsyncioTestCase):
    async def test_ingestion_dedupe_and_different_same_filename(self):
        db = _FakeDB()

        async def save(chunks, **_kwargs):
            for index, chunk in enumerate(chunks):
                db.docstore._dict[f"doc-{len(db.docstore._dict) + index}"] = chunk

        with patch("documents.ingestion.fetch_model", new=AsyncMock(return_value=db)), patch("documents.ingestion.save_embeddings", new=AsyncMock(side_effect=save)) as embed:
            first = await ingest_pdf_bytes("paper.pdf", pdf_bytes("Unique upload evidence."))
            duplicate = await ingest_pdf_bytes("paper-copy.pdf", pdf_bytes("Unique upload evidence."))
            different = await ingest_pdf_bytes("paper.pdf", pdf_bytes("Different document evidence."))

        self.assertEqual((first.status, duplicate.status, different.status), ("SUCCESS", "ALREADY_INDEXED", "SUCCESS"))
        self.assertEqual(embed.await_count, 2)
        self.assertNotEqual(first.document_hash, different.document_hash)

    async def test_indexing_failure_is_not_marked_success_and_batch_is_independent(self):
        db = _FakeDB()
        with patch("documents.ingestion.fetch_model", new=AsyncMock(return_value=db)), patch(
            "documents.ingestion.save_embeddings", new=AsyncMock(side_effect=RuntimeError("index failed"))
        ):
            failed = await ingest_pdf_bytes("paper.pdf", pdf_bytes("Index me"))
        self.assertEqual((failed.status, len(db.docstore._dict)), ("INDEXING_FAILED", 0))

        async def save(chunks, **_kwargs):
            for index, chunk in enumerate(chunks):
                db.docstore._dict[str(index)] = chunk

        with patch("documents.ingestion.fetch_model", new=AsyncMock(return_value=db)), patch("documents.ingestion.save_embeddings", new=AsyncMock(side_effect=save)):
            batch = await ingest_pdf_batch([
                ("good.pdf", pdf_bytes("good evidence"), "application/pdf"),
                ("bad.pdf", b"bad", "application/pdf"),
            ])
        self.assertEqual([result.status for result in batch], ["SUCCESS", "INVALID_PDF"])

    async def test_batch_limit_and_mime_validation(self):
        too_many = await ingest_pdf_batch([("a.pdf", pdf_bytes("a"), "application/pdf")] * 11)
        self.assertTrue(all(result.status == "TOO_MANY_FILES" for result in too_many))
        invalid = await ingest_pdf_bytes("a.pdf", pdf_bytes("a"), mime_type="text/plain")
        self.assertEqual(invalid.status, "INVALID_PDF")

    async def test_reindex_replaces_older_pdf_chunks_without_duplicates(self):
        db = _FakeDB()
        data = pdf_bytes("Candidates must have a minimum CGPA of 7.5 to apply.")
        old_hash = document_hash(data)
        db.docstore._dict["old"] = types.SimpleNamespace(
            page_content="old broad page",
            metadata={"origin": "upload", "document_hash": old_hash, "document_content_hash": "old", "pdf_ingestion_version": 1},
        )

        async def save(chunks, **_kwargs):
            for index, chunk in enumerate(chunks):
                db.docstore._dict[f"new-{index}"] = chunk

        with patch("documents.ingestion.fetch_model", new=AsyncMock(return_value=db)), patch(
            "documents.ingestion.save_embeddings", new=AsyncMock(side_effect=save)
        ), patch("documents.ingestion.persist_vector_store") as persist:
            result = await ingest_pdf_bytes("criteria.pdf", data)

        self.assertEqual(result.status, "SUCCESS")
        self.assertNotIn("old", db.docstore._dict)
        self.assertTrue(all(document.metadata["pdf_ingestion_version"] == 2 for document in db.docstore._dict.values()))
        persist.assert_called_once()

    async def test_reindex_replacement_preserves_web_evidence_with_same_hash(self):
        db = _FakeDB()
        data = pdf_bytes("Candidates must have a minimum CGPA of 7.5 to apply.")
        old_hash = document_hash(data)
        db.docstore._dict = {
            "old": types.SimpleNamespace(page_content="old", metadata={"origin": "upload", "document_hash": old_hash, "document_content_hash": "old", "pdf_ingestion_version": 1}),
            "web": types.SimpleNamespace(page_content="web", metadata={"origin": "web", "document_hash": old_hash, "document_content_hash": "web", "pdf_ingestion_version": 1}),
        }

        async def save(chunks, **_kwargs):
            for index, chunk in enumerate(chunks):
                db.docstore._dict[f"new-{index}"] = chunk

        with patch("documents.ingestion.fetch_model", new=AsyncMock(return_value=db)), patch("documents.ingestion.save_embeddings", new=AsyncMock(side_effect=save)), patch("documents.ingestion.persist_vector_store"):
            result = await ingest_pdf_bytes("criteria.pdf", data)

        self.assertEqual(result.status, "SUCCESS")
        self.assertIn("web", db.docstore._dict)
        self.assertNotIn("old", db.docstore._dict)


class IndexedDocumentRegistryTests(unittest.TestCase):
    class DB(_FakeDB):
        def __init__(self):
            super().__init__()
            self.deleted_ids = []

        def delete(self, ids):
            self.deleted_ids.extend(ids)
            for identifier in ids:
                del self.docstore._dict[identifier]

    @staticmethod
    def document(**metadata):
        return types.SimpleNamespace(page_content="evidence", metadata=metadata)

    def test_registry_collapses_uploaded_chunks_and_excludes_web_evidence(self):
        db = self.DB()
        db.docstore._dict = {
            "a1": self.document(origin="upload", document_hash="a", filename="Resume.pdf", page_number=1, pdf_ingestion_version=2),
            "a2": self.document(origin="upload", document_hash="a", filename="Resume.pdf", page_number=3, pdf_ingestion_version=2),
            "web": self.document(origin="web", document_hash="a", filename="Resume.pdf", page_number=99, pdf_ingestion_version=9),
            "b": self.document(origin="upload", document_hash="b", filename="Resume.pdf", page_number=2, pdf_ingestion_version=1),
        }
        records = list_indexed_documents(db)
        self.assertEqual(records, [
            IndexedDocument("a", "Resume.pdf", 3, 2, 2),
            IndexedDocument("b", "Resume.pdf", 2, 1, 1),
        ])
        self.assertEqual(get_document_chunk_ids(db, "a"), ["a1", "a2"])

    def test_registry_reconstructs_from_persisted_metadata(self):
        persisted = {
            "one": self.document(origin="upload", document_hash="restart", filename="Restart.pdf", page_number=4, pdf_ingestion_version=2),
        }
        restarted = self.DB()
        restarted.docstore._dict = dict(persisted)
        self.assertEqual(list_indexed_documents(restarted), [IndexedDocument("restart", "Restart.pdf", 4, 1, 2)])

    def test_activation_uses_hash_identity_without_embedding_or_duplicate_state(self):
        first = IndexedDocument("a", "A.pdf", 1, 1, 2)
        second = IndexedDocument("b", "B.pdf", 1, 1, 2)
        active = activate_indexed_document([], first)
        self.assertEqual(activate_indexed_document(active, first), active)
        self.assertEqual(activate_indexed_document(active, second), [{"document_hash": "a", "filename": "A.pdf"}, {"document_hash": "b", "filename": "B.pdf"}])

    def test_delete_removes_only_target_uploaded_chunks_and_persists(self):
        db = self.DB()
        db.docstore._dict = {
            "target": self.document(origin="upload", document_hash="target", filename="Same.pdf", page_number=1),
            "other": self.document(origin="upload", document_hash="other", filename="Same.pdf", page_number=1),
            "web": self.document(origin="web", document_hash="target", filename="Same.pdf", page_number=1),
        }
        persisted = {}
        def persist(db, memory_path):
            persisted.update(db.docstore._dict)
        with patch("documents.registry.persist_vector_store", side_effect=persist) as persist_vector:
            delete_indexed_document(db, "target")
        self.assertEqual(db.deleted_ids, ["target"])
        self.assertEqual(set(db.docstore._dict), {"other", "web"})
        persist_vector.assert_called_once_with(db, None)
        restarted = self.DB()
        restarted.docstore._dict = dict(persisted)
        self.assertEqual([record.document_hash for record in list_indexed_documents(restarted)], ["other"])
        active = remove_document_from_active_state(
            [{"document_hash": "target", "filename": "Same.pdf"}, {"document_hash": "other", "filename": "Same.pdf"}], "target"
        )
        self.assertEqual(active, [{"document_hash": "other", "filename": "Same.pdf"}])

    def test_delete_failure_does_not_claim_success_or_change_active_state(self):
        db = self.DB()
        db.docstore._dict = {"target": self.document(origin="upload", document_hash="target", filename="A.pdf", page_number=1)}
        db.delete = lambda *, ids: (_ for _ in ()).throw(RuntimeError("delete failed"))
        active = [{"document_hash": "target", "filename": "A.pdf"}]
        with self.assertRaises(RuntimeError):
            delete_indexed_document(db, "target")
        self.assertEqual(list(db.docstore._dict), ["target"])
        self.assertEqual(active, [{"document_hash": "target", "filename": "A.pdf"}])


class PDFProvenanceTests(unittest.TestCase):
    def test_uploaded_provenance_has_filename_page_no_url_and_deterministic_ids(self):
        document = types.SimpleNamespace(
            page_content="Uploaded evidence",
            metadata={"filename": "paper.pdf", "title": "paper.pdf", "page_number": 4, "document_hash": "d", "document_content_hash": "c", "origin": "upload"},
        )
        item = evidence_from_document(document)
        self.assertEqual((item.origin, item.filename, item.page_number, item.document_hash, item.source_url), ("upload", "paper.pdf", 4, "d", None))
        with citation_collection() as collector:
            register_evidence([item, EvidenceItem("web", "https://example.test", title="Example", document_content_hash="w", origin="web")])
            self.assertEqual(list(collector.source_map), ["S1", "S2"])
            rendered = render_authoritative_sources(collector.source_map, {"S1", "S2"})
            valid, invalid = validate_citations("PDF [S1] Web [S2] Unknown [S9]", collector.source_map)
        self.assertIn("**paper.pdf**", rendered)
        self.assertIn("Page 4 · Uploaded document", rendered)
        self.assertNotIn("URL unavailable", rendered)
        self.assertEqual((valid, invalid), ({"S1", "S2"}, {"S9"}))


class PDFRetrievalRankingTests(unittest.TestCase):
    class Doc:
        def __init__(self, content, **metadata):
            self.page_content = content
            self.metadata = metadata

    def test_acronyms_numbers_and_cpp_are_preserved_for_lexical_matching(self):
        self.assertTrue({"cgpa", "7.5", "c++"}.issubset(_lexical_tokens("Minimum CGPA is 7.5 for C++ experience.")))

    def test_active_pdf_and_exact_terms_outrank_unrelated_web_candidate(self):
        web = self.Doc("General admissions information.", origin="web")
        pdf = self.Doc("Candidates must have a minimum CGPA of 7.5 to apply.", origin="upload", document_hash="fischer")
        ranked = rank_memory_candidates("What is the minimum CGPA criteria?", [(web, .31), (pdf, .34)], [{"document_hash": "fischer", "filename": "Fischer Jordan - JD.pdf"}])
        self.assertEqual(ranked[0][0], pdf)

    def test_explicit_normalized_filename_selects_matching_active_pdf(self):
        a = self.Doc("CGPA is 6.0", origin="upload", document_hash="a")
        b = self.Doc("CGPA is 7.5", origin="upload", document_hash="b")
        active = [
            {"document_hash": "a", "filename": "Other Program.pdf"},
            {"document_hash": "b", "filename": "Fischer Jordan - JD.pdf"},
        ]
        ranked = rank_memory_candidates("What is the CGPA according to Fischer Jordan JD?", [(a, .2), (b, .4)], active)
        self.assertEqual(ranked, [(b, .4)])
        self.assertEqual(_filename_key("Fischer Jordan - JD.pdf"), _filename_key("fischer jordan jd"))

    def test_multiple_active_pdfs_do_not_force_irrelevant_document(self):
        a = self.Doc("CGPA is 6.0", origin="upload", document_hash="a")
        web = self.Doc("Quantum entanglement describes correlations between particles.", origin="web")
        ranked = rank_memory_candidates("Explain quantum entanglement", [(a, .9), (web, .2)], [{"document_hash": "a", "filename": "Admissions.pdf"}])
        self.assertEqual(ranked[0][0], web)

    def test_upload_results_activate_success_and_already_indexed_documents(self):
        active = activate_uploaded_documents([], [
            UploadResult("new.pdf", "SUCCESS", "Indexed", "new"),
            UploadResult("old.pdf", "ALREADY_INDEXED", "Already indexed", "old"),
        ])
        self.assertEqual(active, [{"document_hash": "new", "filename": "new.pdf"}, {"document_hash": "old", "filename": "old.pdf"}])

    def test_document_context_intent_recognizes_generic_and_normalized_names(self):
        active = [{"document_hash": "fischer", "filename": "Fischer Jordan - JD.pdf"}]
        self.assertTrue(is_document_context_query("What have u read in the uploaded PDF?", active))
        self.assertTrue(is_document_context_query("minimum CGPA in fisher jordan jd", active))
        self.assertTrue(is_document_context_query("What is this document about?", active))
        self.assertFalse(is_document_context_query("Explain quantum entanglement", active))


class ActiveDocumentScopedRetrievalTests(unittest.IsolatedAsyncioTestCase):
    class Doc:
        def __init__(self, content, **metadata):
            self.page_content = content
            self.metadata = metadata

    class DB:
        def __init__(self, documents):
            self.docstore = types.SimpleNamespace(_dict={str(index): document for index, document in enumerate(documents)})

        async def asimilarity_search_with_score(self, _query, k=4, filter=None, **_kwargs):
            rows = [(document, .2 if document.metadata.get("document_hash") else .1) for document in self.docstore._dict.values()]
            if filter:
                rows = [(document, score) for document, score in rows if filter(document.metadata)]
            return rows[:k]

    async def test_generic_query_scopes_active_pdf_before_historical_web(self):
        web = self.Doc("Historical generic CGPA web discussion", source_url="https://web.test")
        pdf = self.Doc("Candidates must have a minimum CGPA of 7.5 to apply.", origin="upload", document_hash="fischer", filename="Fischer Jordan - JD.pdf", page_number=2)
        db = self.DB([web, pdf])
        active = [{"document_hash": "fischer", "filename": "Fischer Jordan - JD.pdf"}]
        with patch.object(relevance_search, "fetch_model", new=AsyncMock(return_value=db)):
            candidates = await retrieve_active_document_candidates("What is the minimum CGPA criteria?", active)
        self.assertEqual([document for document, _ in candidates], [pdf])

    async def test_explicit_filename_hard_scopes_one_of_multiple_active_documents(self):
        a = self.Doc("CGPA is 6.0", origin="upload", document_hash="a", filename="A.pdf", page_number=1)
        b = self.Doc("CGPA is 7.5", origin="upload", document_hash="b", filename="Fischer Jordan - JD.pdf", page_number=2)
        db = self.DB([a, b])
        active = [{"document_hash": "a", "filename": "Admissions.pdf"}, {"document_hash": "b", "filename": "Fischer Jordan - JD.pdf"}]
        with patch.object(relevance_search, "fetch_model", new=AsyncMock(return_value=db)):
            candidates = await retrieve_active_document_candidates("What is the CGPA in fisher jordan jd?", active)
        self.assertEqual([document for document, _ in candidates], [b])

    async def test_document_overview_uses_bounded_representative_upload_chunks(self):
        documents = [self.Doc(f"Page {page}", origin="upload", document_hash="fischer", filename="Fischer Jordan - JD.pdf", page_number=page) for page in range(1, 10)]
        db = self.DB(documents)
        active = [{"document_hash": "fischer", "filename": "Fischer Jordan - JD.pdf"}]
        with patch.object(relevance_search, "fetch_model", new=AsyncMock(return_value=db)):
            candidates = await retrieve_active_document_candidates("What have you read in the uploaded PDF?", active)
        self.assertLessEqual(len(candidates), 5)
        self.assertEqual(candidates[0][0].metadata["page_number"], 1)
        self.assertEqual(candidates[-1][0].metadata["page_number"], 9)
