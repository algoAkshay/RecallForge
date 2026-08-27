import asyncio
import importlib.util
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
DATA_SOURCE = SRC / "tools" / "data.py"


class FakeDocument:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


class FakeEmbeddings:
    def embed_query(self, _query):
        return [0.0, 0.0, 0.0]


class FakeIndex:
    def __init__(self, dimensions):
        self.dimensions = dimensions


class FakeDocstore:
    def __init__(self, documents=None):
        self._dict = documents or {}


class FakeFAISS:
    add_calls = 0

    def __init__(self, embedding_function, index, docstore, index_to_docstore_id):
        self.embedding_function = embedding_function
        self.index = index
        self.docstore = docstore
        self.index_to_docstore_id = index_to_docstore_id

    async def aadd_documents(self, documents):
        type(self).add_calls += 1
        for document in documents:
            key = str(len(self.docstore._dict))
            self.docstore._dict[key] = document
            self.index_to_docstore_id[len(self.index_to_docstore_id)] = key

    async def asimilarity_search_with_relevance_scores(self, query, k=3):
        matches = [
            document for document in self.docstore._dict.values()
            if query.lower() in document.page_content.lower()
        ]
        return [(document, 1.0) for document in matches[:k]]

    def save_local(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "index.faiss").write_bytes(b"fake-index")
        with (folder / "index.pkl").open("wb") as handle:
            pickle.dump((self.docstore._dict, self.index_to_docstore_id), handle)

    @classmethod
    def load_local(cls, folder, embeddings, allow_dangerous_deserialization=False):
        if not allow_dangerous_deserialization:
            raise ValueError("explicit deserialization opt-in required")
        with (Path(folder) / "index.pkl").open("rb") as handle:
            documents, mapping = pickle.load(handle)
        return cls(embeddings, FakeIndex(3), FakeDocstore(documents), mapping)


class PersistentMemoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.original_modules = {name: sys.modules.get(name) for name in (
            "streamlit", "faiss", "langchain_community.vectorstores",
            "langchain_community.docstore.in_memory", "langchain_huggingface",
        )}
        streamlit = types.ModuleType("streamlit")
        streamlit.session_state = {}
        faiss = types.ModuleType("faiss")
        faiss.IndexFlatL2 = FakeIndex
        vectorstores = types.ModuleType("langchain_community.vectorstores")
        vectorstores.FAISS = FakeFAISS
        docstore = types.ModuleType("langchain_community.docstore.in_memory")
        docstore.InMemoryDocstore = FakeDocstore
        huggingface = types.ModuleType("langchain_huggingface")
        huggingface.HuggingFaceEmbeddings = FakeEmbeddings
        sys.modules.update({
            "streamlit": streamlit,
            "faiss": faiss,
            "langchain_community.vectorstores": vectorstores,
            "langchain_community.docstore.in_memory": docstore,
            "langchain_huggingface": huggingface,
        })
        spec = importlib.util.spec_from_file_location("persistent_memory_data", DATA_SOURCE)
        self.data = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.data)
        FakeFAISS.add_calls = 0

    def tearDown(self):
        for name, module in self.original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    async def _insert(self, path, content="RecallForge persistent research memory test marker", source="https://a.test"):
        db = await self.data.fetch_model(memory_path=path, embeddings=FakeEmbeddings())
        document = FakeDocument(content, {"source": source, "document_content_hash": "hash-x", "retrieved_at": "2026-01-01T00:00:00+00:00"})
        await self.data.save_embeddings([document], memory_path=path)
        return db

    async def test_fresh_startup_creates_empty_memory(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "memory"
            db = await self.data.fetch_model(memory_path=path, embeddings=FakeEmbeddings())
            self.assertEqual(db.docstore._dict, {})
            self.assertFalse(path.exists())

    async def test_successful_ingestion_writes_durable_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "memory"
            await self._insert(path)
            self.assertTrue((path / "index.faiss").exists())
            self.assertTrue((path / "index.pkl").exists())

    async def test_restart_reloads_retrievable_vectors_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "memory"
            await self._insert(path)
            self.data.st.session_state.clear()  # discard the old session/object
            db = await self.data.fetch_model(memory_path=path, embeddings=FakeEmbeddings())
            results = await db.asimilarity_search_with_relevance_scores("persistent research memory")
            document = results[0][0]
            self.assertEqual(document.metadata["source"], "https://a.test")
            self.assertEqual(document.metadata["document_content_hash"], "hash-x")
            self.assertEqual(document.metadata["retrieved_at"], "2026-01-01T00:00:00+00:00")

    async def test_restart_reconstructs_dedupe_hashes_without_second_insertion(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "memory"
            await self._insert(path)
            self.assertEqual(FakeFAISS.add_calls, 1)
            self.data.st.session_state.clear()
            await self.data.fetch_model(memory_path=path, embeddings=FakeEmbeddings())
            self.assertIn("hash-x", self.data.st.session_state["indexed_content_hashes"])
            if "hash-x" not in self.data.st.session_state["indexed_content_hashes"]:
                await self._insert(path)
            self.assertEqual(FakeFAISS.add_calls, 1)

    async def test_changed_content_after_restart_can_be_saved(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "memory"
            await self._insert(path)
            self.data.st.session_state.clear()
            await self.data.fetch_model(memory_path=path, embeddings=FakeEmbeddings())
            await self.data.save_embeddings([
                FakeDocument("changed content", {"source": "https://a.test", "document_content_hash": "hash-y"})
            ], memory_path=path)
            self.assertEqual(FakeFAISS.add_calls, 2)

    async def test_failed_ingestion_is_not_persisted_as_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "memory"
            db = await self.data.fetch_model(memory_path=path, embeddings=FakeEmbeddings())
            original = db.aadd_documents

            async def fail(_documents):
                raise RuntimeError("embedding failed")

            db.aadd_documents = fail
            with self.assertRaises(RuntimeError):
                await self.data.save_embeddings([FakeDocument("x", {"document_content_hash": "failed"})], path)
            self.assertFalse(path.exists())
            db.aadd_documents = original

    async def test_corrupt_persistence_is_explicit_and_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "memory"
            path.mkdir()
            (path / "index.pkl").write_bytes(b"not a pickle")
            with self.assertRaises(self.data.MemoryLoadError):
                await self.data.fetch_model(memory_path=path, embeddings=FakeEmbeddings())
            self.assertTrue((path / "index.pkl").exists())


if __name__ == "__main__":
    unittest.main()
