import os
import json
import shutil
from typing import List, Dict, Any, Tuple, Optional
import streamlit as st
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

BASE_STORAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".veridoc_storage")


def get_session_dir(session_id: str) -> str:
    safe_id = "".join(c for c in session_id if c.isalnum() or c in ("-", "_")) or "default"
    return os.path.join(BASE_STORAGE_DIR, safe_id)


def save_persisted_state(
    session_id: str,
    faiss_db: FAISS,
    chunks: List[Document],
    indexed_files: List[str],
    dense_weight: float,
) -> None:
    s_dir = get_session_dir(session_id)
    os.makedirs(s_dir, exist_ok=True)
    faiss_db.save_local(os.path.join(s_dir, "faiss_index"))

    serialized_docs = [
        {"page_content": d.page_content, "metadata": d.metadata}
        for d in chunks
    ]
    with open(os.path.join(s_dir, "documents.json"), "w", encoding="utf-8") as f:
        json.dump(serialized_docs, f, ensure_ascii=False, indent=2)

    with open(os.path.join(s_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"indexed_files": indexed_files, "dense_weight": dense_weight},
            f,
            ensure_ascii=False,
            indent=2,
        )


def save_chat_history(session_id: str, chat_history: List[Dict[str, Any]]) -> None:
    s_dir = get_session_dir(session_id)
    os.makedirs(s_dir, exist_ok=True)
    with open(os.path.join(s_dir, "chat_history.json"), "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=2)


def load_persisted_state(
    session_id: str, embed_model
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
            faiss_db = FAISS.load_local(
                faiss_dir, embed_model, allow_dangerous_deserialization=True
            )
            with open(docs_file, "r", encoding="utf-8") as f:
                raw_docs = json.load(f)
            chunks = [
                Document(page_content=d["page_content"], metadata=d["metadata"])
                for d in raw_docs
            ]
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
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
