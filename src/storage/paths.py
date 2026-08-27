"""Application-owned paths for durable local state.

The storage root is configurable so a single Render persistent disk can retain
both SQLite chat history and FAISS research memory.  The legacy FAISS override
remains available for existing local installations.
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def storage_root() -> Path:
    """Return the configured durable-state root, defaulting to local `.storage`."""
    return Path(os.environ.get("RECALLFORGE_STORAGE_DIR", PROJECT_ROOT / ".storage")).expanduser()


def research_memory_path() -> Path:
    """Return the FAISS location, honoring the legacy direct-path override."""
    legacy_path = os.environ.get("LINKMIND_MEMORY_PATH")
    return Path(legacy_path).expanduser() if legacy_path else storage_root() / "research_memory"


def chat_history_path() -> Path:
    """Return the SQLite chat-history location under the shared storage root."""
    return storage_root() / "chat_history.db"
