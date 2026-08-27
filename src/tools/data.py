import asyncio
import hashlib
import logging
import os
from pathlib import Path

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st


logger = logging.getLogger(__name__)
DEFAULT_MEMORY_PATH = Path(
    # Preserve the legacy environment key so existing deployments keep their configured memory path.
    os.environ.get("LINKMIND_MEMORY_PATH", Path(__file__).resolve().parents[2] / ".storage" / "research_memory")
)
_INDEXED_HASHES_KEY = "indexed_content_hashes"
_INFLIGHT_INGESTIONS_KEY = "inflight_content_ingestions"


class MemoryLoadError(RuntimeError):
    """Raised when application-owned persisted semantic memory cannot be loaded."""


def _memory_path(memory_path: str | Path | None = None) -> Path:
    return Path(memory_path) if memory_path is not None else DEFAULT_MEMORY_PATH


def _create_vector_store(embeddings):
    """Create the empty FAISS store used on a fresh installation."""
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )


def _reconstruct_indexed_hashes(db) -> set[str]:
    """Recover Task 3 document identities from persisted chunk metadata."""
    hashes: set[str] = set()
    docstore = getattr(db, "docstore", None)
    documents = getattr(docstore, "_dict", {})
    for document in documents.values():
        metadata = getattr(document, "metadata", {}) or {}
        content_hash = metadata.get("document_content_hash")
        if isinstance(content_hash, str) and content_hash:
            hashes.add(content_hash)
        elif content_hash is not None:
            logger.warning("Ignoring malformed document_content_hash in persisted memory.")
    return hashes


def _restore_semantic_memory_state(db) -> None:
    """Restore durable dedupe state and always reset transient coordination."""
    st.session_state[_INDEXED_HASHES_KEY] = _reconstruct_indexed_hashes(db)
    st.session_state[_INFLIGHT_INGESTIONS_KEY] = {}


def load_or_create_vector_store(embeddings, memory_path: str | Path | None = None):
    """Load application-owned FAISS memory, or create an empty store when absent.

    FAISS persists its docstore with Python pickle.  The path is application-controlled
    local storage, never a user-supplied upload; do not point it at untrusted files.
    """
    path = _memory_path(memory_path)
    if not path.exists():
        return _create_vector_store(embeddings)

    try:
        # LangChain requires this explicit opt-in because FAISS docstores use pickle.
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    except Exception as error:
        logger.error("Persistent research memory could not be loaded from %s: %s", path, error)
        raise MemoryLoadError(
            f"Persistent research memory could not be loaded from {path}. "
            "The files were preserved; rebuild or remove them only after inspection."
        ) from error


def persist_vector_store(db, memory_path: str | Path | None = None) -> None:
    """Persist semantic memory only after successful new FAISS insertion."""
    path = _memory_path(memory_path)
    path.mkdir(parents=True, exist_ok=True)
    db.save_local(path)

# Normalize formatting-only differences before computing document ingestion identity.
def generate_doc_hash(content: str) -> str:
    """Create a SHA-256 hash for normalized document content."""
    normalized = " ".join(content.replace("\r\n", "\n").replace("\r", "\n").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

# Asynchronously save document chunks to the FAISS vector store in memory.
async def save_embeddings(chunks, memory_path: str | Path | None = None):
    if not chunks:
        raise ValueError("Chunks not provided!")
    
    # Check if the FAISS database exists in session; if not, initialize it.
    if "db" not in st.session_state or st.session_state["db"] is None:
        db = await fetch_model(memory_path=memory_path)
    else:
        db = st.session_state["db"]

    await db.aadd_documents(chunks)  # Add document chunks to FAISS index asynchronously
    persist_vector_store(db, memory_path)
    logger.info("Saved %d chunks to persistent FAISS memory", len(chunks))
    st.session_state["db"] = db  # Update session state with the latest database
    return st.session_state["db"]

# Asynchronously initialize and load the FAISS vector store with Hugging Face embeddings.
async def fetch_model(memory_path: str | Path | None = None, embeddings=None):
    """Initialize this session's handle to the durable, local semantic memory."""
    if "db" not in st.session_state or st.session_state["db"] is None:
        embeddings = embeddings or HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = load_or_create_vector_store(embeddings, memory_path)
        st.session_state["db"] = db
        _restore_semantic_memory_state(db)
    return st.session_state["db"]
