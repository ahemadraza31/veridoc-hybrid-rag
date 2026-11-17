import os
import re
import json
import time
import shutil
import tempfile
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

BASE_STORAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".veridoc_storage")
SESSION_ID_RE = re.compile(r"^[0-9a-f]{8,32}$")


def validate_session_id(session_id: str) -> str:
    if not isinstance(session_id, str) or not SESSION_ID_RE.fullmatch(session_id):
        raise ValueError(f"Invalid session_id: {session_id!r}. Expected 8-32 lowercase hex chars.")
    return session_id


def get_session_dir(session_id: str) -> str:
    valid_id = validate_session_id(session_id)
    return os.path.join(BASE_STORAGE_DIR, valid_id)


def _atomic_write_json(target_path: str, payload: Any) -> None:
    parent_dir = os.path.dirname(target_path)
    os.makedirs(parent_dir, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=parent_dir, delete=False, suffix=".tmp"
        ) as tmp_file:
            json.dump(payload, tmp_file, ensure_ascii=False, indent=2)
            tmp_path = tmp_file.name
        os.replace(tmp_path, target_path)
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def save_persisted_state(
    session_id: str,
    faiss_db: FAISS,
    chunks: List[Document],
    indexed_files: List[str],
    dense_weight: float,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> None:
    s_dir = get_session_dir(session_id)
    os.makedirs(s_dir, exist_ok=True)
    faiss_db.save_local(os.path.join(s_dir, "faiss_index"))

    serialized_docs = [
        {"page_content": d.page_content, "metadata": d.metadata}
        for d in chunks
    ]
    _atomic_write_json(os.path.join(s_dir, "documents.json"), serialized_docs)
    _atomic_write_json(
        os.path.join(s_dir, "metadata.json"),
        {
            "indexed_files": indexed_files,
            "dense_weight": dense_weight,
            "chunk_count": len(chunks),
            "embedding_model": embedding_model,
        },
    )


def save_chat_history(session_id: str, chat_history: List[Dict[str, Any]]) -> None:
    s_dir = get_session_dir(session_id)
    os.makedirs(s_dir, exist_ok=True)
    _atomic_write_json(os.path.join(s_dir, "chat_history.json"), chat_history)


def load_persisted_state(
    session_id: str,
    embed_model,
    expected_embedding_model: str = "all-MiniLM-L6-v2",
) -> Tuple[Optional[FAISS], List[Document], List[str], float, List[Dict[str, Any]], Optional[str]]:
    """
    Returns: (faiss_db, chunks, indexed_files, dense_weight, chat_history, warning_msg)
    """
    s_dir = get_session_dir(session_id)
    chat_history: List[Dict[str, Any]] = []
    warning_msg: Optional[str] = None

    chat_file = os.path.join(s_dir, "chat_history.json")
    if os.path.exists(chat_file):
        try:
            with open(chat_file, "r", encoding="utf-8") as f:
                chat_history = json.load(f)
        except Exception as e:
            warning_msg = f"Could not restore chat history: {e}"

    faiss_dir = os.path.join(s_dir, "faiss_index")
    docs_file = os.path.join(s_dir, "documents.json")
    meta_file = os.path.join(s_dir, "metadata.json")

    if os.path.exists(faiss_dir) and os.path.exists(docs_file) and os.path.exists(meta_file):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

            if "embedding_model" in meta and meta["embedding_model"] is not None:
                saved_model = meta["embedding_model"]
                if saved_model != expected_embedding_model:
                    return (
                        None,
                        [],
                        [],
                        0.5,
                        chat_history,
                        f"Saved index embedding model ({saved_model!r}) mismatches expected ({expected_embedding_model!r}). Please re-index.",
                    )

            with open(docs_file, "r", encoding="utf-8") as f:
                raw_docs = json.load(f)
            chunks = [
                Document(page_content=d["page_content"], metadata=d["metadata"])
                for d in raw_docs
            ]

            if "chunk_count" in meta and meta["chunk_count"] is not None:
                if meta["chunk_count"] != len(chunks):
                    return (
                        None,
                        [],
                        [],
                        0.5,
                        chat_history,
                        f"Saved metadata chunk_count ({meta['chunk_count']}) mismatches chunks ({len(chunks)}). Please re-index.",
                    )

            faiss_db = FAISS.load_local(
                faiss_dir, embed_model, allow_dangerous_deserialization=True
            )
            if faiss_db.index.ntotal != len(chunks):
                return (
                    None,
                    [],
                    [],
                    0.5,
                    chat_history,
                    f"Saved FAISS index count ({faiss_db.index.ntotal}) mismatches chunks ({len(chunks)}). Please re-index.",
                )

            return (
                faiss_db,
                chunks,
                meta.get("indexed_files", []),
                float(meta.get("dense_weight", 0.5)),
                chat_history,
                warning_msg,
            )
        except Exception as e:
            return None, [], [], 0.5, chat_history, f"Saved index could not be loaded ({e}). Please re-index."

    return None, [], [], 0.5, chat_history, warning_msg


def clear_persisted_state(session_id: str) -> None:
    s_dir = get_session_dir(session_id)
    if os.path.exists(s_dir):
        shutil.rmtree(s_dir, ignore_errors=True)


def purge_old_sessions(max_age_days: int = 7) -> int:
    """Delete session directories older than max_age_days based on latest file/dir mtime."""
    if not os.path.exists(BASE_STORAGE_DIR):
        return 0
    cutoff = time.time() - (max_age_days * 86400)
    removed = 0
    for entry in os.scandir(BASE_STORAGE_DIR):
        if not entry.is_dir():
            continue
        latest_mtime = entry.stat().st_mtime
        for root, _, files in os.walk(entry.path):
            for fname in files:
                try:
                    fpath = os.path.join(root, fname)
                    latest_mtime = max(latest_mtime, os.stat(fpath).st_mtime)
                except OSError:
                    pass
        if latest_mtime < cutoff:
            shutil.rmtree(entry.path, ignore_errors=True)
            removed += 1
    return removed
