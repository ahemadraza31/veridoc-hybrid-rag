import html
import os
import re
import time
import uuid
from typing import List, Dict, Any, Tuple
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder

from ingest import process_uploaded_files, get_sample_document
from retrieval import (
    build_or_update_index,
    retrieve_and_rerank,
    validate_citations,
    strip_think_tags,
)
from storage import (
    SESSION_ID_RE,
    save_persisted_state,
    save_chat_history,
    load_persisted_state,
    clear_persisted_state,
    purge_old_sessions,
)
from eval_suite import run_evaluation

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Load local .env if present
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Streamlit Page Config
st.set_page_config(
    page_title="VeriDoc AI — Enterprise Hybrid RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _resolve_groq_key() -> str:
    env_key = os.getenv("GROQ_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        if "GROQ_API_KEY" in st.secrets:
            return str(st.secrets["GROQ_API_KEY"]).strip()
        for val in st.secrets.values():
            if isinstance(val, dict) and "GROQ_API_KEY" in val:
                return str(val["GROQ_API_KEY"]).strip()
    except Exception:
        pass
    return st.session_state.get("user_groq_key", "").strip()


groq_api_key = _resolve_groq_key()

# Theme-safe Custom Styling (works in both Light & Dark themes)
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        opacity: 0.78;
        margin-bottom: 1.2rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        background-color: rgba(100, 116, 139, 0.16);
        border: 1px solid rgba(100, 116, 139, 0.3);
        font-size: 0.84rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .chunk-card {
        padding: 12px 14px;
        border-radius: 8px;
        border: 1px solid rgba(100, 116, 139, 0.35);
        background-color: rgba(100, 116, 139, 0.08);
        margin-bottom: 10px;
        font-size: 0.9rem;
        line-height: 1.45;
    }
    .chunk-meta {
        font-size: 0.84rem;
        margin-bottom: 6px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------- CACHED MODELS -----------------
@st.cache_resource(show_spinner="Loading MiniLM Embeddings (normalized)...")
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner="Loading Cross-Encoder Reranker...")
def get_cross_encoder():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@st.cache_resource
def _run_startup_purge() -> int:
    try:
        return purge_old_sessions(max_age_days=7)
    except Exception:
        return 0


# ----------------- SESSION INITIALIZATION -----------------
if "session_id" not in st.session_state:
    raw_sid = str(st.query_params.get("sid", "")).strip()
    if SESSION_ID_RE.fullmatch(raw_sid):
        valid_sid = raw_sid
    else:
        valid_sid = uuid.uuid4().hex[:16]
    st.session_state.session_id = valid_sid
    st.query_params["sid"] = valid_sid

_run_startup_purge()

if "initialized" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.faiss_store = None
    st.session_state.bm25_retriever = None
    st.session_state.all_documents = []
    st.session_state.indexed_files = []
    st.session_state.dense_weight = 0.5
    st.session_state.last_inspection = None
    st.session_state.eval_results = None

    embed_model = get_embedding_model()
    faiss_db, chunks, files, d_weight, chat_hist, warn_msg = load_persisted_state(
        st.session_state.session_id,
        embed_model,
        expected_embedding_model=EMBEDDING_MODEL_NAME,
    )
    st.session_state.startup_warning = warn_msg

    if faiss_db is not None and chunks:
        _, bm25_ret = build_or_update_index(
            chunks, embed_model, existing_faiss=faiss_db
        )
        st.session_state.faiss_store = faiss_db
        st.session_state.bm25_retriever = bm25_ret
        st.session_state.all_documents = chunks
        st.session_state.indexed_files = files
        st.session_state.dense_weight = d_weight

    st.session_state.chat_history = chat_hist
    st.session_state.initialized = True


def render_source_cards(sources: List[Dict[str, Any]]):
    for src in sources:
        safe_source = html.escape(str(src["source"]))
        safe_id = html.escape(str(src["chunk_id"]))
        safe_score = html.escape(str(src["score"]))
        clean_snip = re.sub(r"^#+\s*", "", str(src["snippet"]), flags=re.MULTILINE)
        safe_snippet = html.escape(clean_snip).replace("\n", " ")
        st.markdown(
            f"""
            <div class="chunk-card">
                <div class="chunk-meta">
                    <strong>Source:</strong> {safe_source} &nbsp;|&nbsp;
                    <strong>ID:</strong> <code>{safe_id}</code> &nbsp;|&nbsp;
                    <strong>Relevance:</strong> <code>{safe_score}</code>
                </div>
                <div>"{safe_snippet}"</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _prepare_recent_history(history: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Return last 6 messages as ('human'|'assistant', clipped_text), stripping fallback notices."""
    turns: List[Tuple[str, str]] = []
    for m in history[-6:]:
        role = m.get("role", "user")
        content = str(m.get("content", ""))
        if role == "assistant":
            content = re.sub(
                r"^(?:>\s*)?(?:ℹ️|⚠️)?\s*\*{0,2}Note:\*{0,2}[^\n]*(?:\n+|$)",
                "",
                content.lstrip(),
            ).lstrip()
            turns.append(("assistant", content[:700]))
        else:
            turns.append(("human", content[:300]))
    return turns


def rewrite_followup_query(llm: ChatGroq, query: str, history: List[Dict[str, Any]]) -> str:
    """Rewrite follow-up questions into standalone queries using recent conversation turns."""
    turns = _prepare_recent_history(history)
    if not turns:
        return query
    conv_text = "\n".join(
        f"{'USER' if r == 'human' else 'ASSISTANT'}: {c}" for r, c in turns
    )
    prompt = (
        "Given the recent conversation and a follow-up question, rewrite the follow-up into a clear, "
        "standalone search query. If it is already standalone, return it unchanged. Output ONLY the query.\n\n"
        f"Conversation:\n{conv_text}\n\nFollow-up: {query}"
    )
    try:
        rewritten = strip_think_tags(llm.invoke(prompt).content)
        if not rewritten:
            raise ValueError("Empty rewrite")
        return rewritten
    except Exception:
        st.caption("ℹ️ Follow-up could not be rewritten; using the raw question.")
        return query


def export_chat_markdown(history: List[Dict[str, Any]]) -> str:
    lines = ["# VeriDoc AI — Session Transcript\n"]
    for m in history:
        role = "User" if m["role"] == "user" else "VeriDoc AI"
        lines.append(f"### {role}\n{m['content']}\n")
        if m.get("sources"):
            lines.append("**Verified Sources:**")
            for s in m["sources"]:
                lines.append(f"- `{s['chunk_id']}` ({s['score']}): {s['snippet']}")
        lines.append("\n---\n")
    return "\n".join(lines)


# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Configuration")

    if not groq_api_key:
        entered_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Set GROQ_API_KEY in Streamlit Secrets or paste one here for this session.",
        )
        if entered_key:
            st.session_state.user_groq_key = entered_key.strip()
            groq_api_key = entered_key.strip()
        else:
            st.warning("Add `GROQ_API_KEY` in Streamlit Secrets (or paste above) to enable LLM answers.")

    selected_model = st.selectbox(
        "Inference LLM (Groq)",
        ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"],
        index=0,
        help="Groq LPU models for low-latency synthesis.",
    )
    temperature = st.slider("Temperature", 0.0, 0.7, 0.1, 0.05)

    st.divider()
    st.markdown("#### 🔍 Retrieval Tuning")
    dense_weight = st.slider(
        "Hybrid Ratio (0 = BM25 | 1 = FAISS)",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.dense_weight),
        step=0.1,
    )
    st.session_state.dense_weight = dense_weight

    top_k = st.slider("Candidates to Retrieve (Top-K)", 2, 12, 6)
    context_k = st.slider("Context Chunks Sent to LLM", 1, 8, 4)
    use_reranker = st.toggle("Cross-Encoder Re-Ranking", value=True)
    relevance_threshold = st.slider(
        "Min Relevance Score (%)",
        min_value=5,
        max_value=80,
        value=25,
        step=5,
        help="Chunks below this Cross-Encoder relevance score are treated as irrelevant.",
    ) / 100.0
    use_hyde = st.toggle("HyDE (Query Expansion)", value=False)
    strict_mode = st.toggle(
        "Strict Document-Only Mode",
        value=False,
        help="When ON, off-topic questions are refused instead of falling back to general AI.",
    )

    st.divider()
    st.markdown("#### 📂 Document Management")
    uploaded_files = st.file_uploader(
        "Upload Enterprise Documents",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True,
    )

    if st.button("🚀 Index Uploaded Files", use_container_width=True):
        if uploaded_files:
            with st.spinner("Parsing, embedding & persisting..."):
                existing_set = set(st.session_state.indexed_files)
                new_chunks, newly_indexed, msgs = process_uploaded_files(
                    uploaded_files, existing_files=existing_set
                )
                for msg in msgs:
                    st.warning(msg)

                if new_chunks:
                    embed_model = get_embedding_model()
                    combined_chunks = st.session_state.all_documents + new_chunks
                    combined_files = st.session_state.indexed_files + newly_indexed
                    faiss_db, bm25_ret = build_or_update_index(
                        chunks=combined_chunks,
                        embed_model=embed_model,
                        existing_faiss=st.session_state.faiss_store,
                        new_chunks=new_chunks,
                    )
                    st.session_state.all_documents = combined_chunks
                    st.session_state.indexed_files = combined_files
                    st.session_state.faiss_store = faiss_db
                    st.session_state.bm25_retriever = bm25_ret
                    save_persisted_state(
                        st.session_state.session_id,
                        faiss_db,
                        combined_chunks,
                        combined_files,
                        dense_weight,
                        embedding_model=EMBEDDING_MODEL_NAME,
                    )
                    st.success(f"Added {len(new_chunks)} chunks from {len(newly_indexed)} file(s).")
        else:
            st.warning("Please upload at least one document first.")

    if st.button("📋 Load Demo Enterprise Doc", use_container_width=True):
        demo_name = "ACME_Enterprise_Arch_2026.pdf"
        if demo_name in st.session_state.indexed_files:
            st.info(f"`{demo_name}` is already indexed.")
        else:
            with st.spinner("Indexing demo document..."):
                sample_chunks = get_sample_document()
                embed_model = get_embedding_model()
                combined_chunks = st.session_state.all_documents + sample_chunks
                combined_files = st.session_state.indexed_files + [demo_name]
                faiss_db, bm25_ret = build_or_update_index(
                    chunks=combined_chunks,
                    embed_model=embed_model,
                    existing_faiss=st.session_state.faiss_store,
                    new_chunks=sample_chunks,
                )
                st.session_state.all_documents = combined_chunks
                st.session_state.indexed_files = combined_files
                st.session_state.faiss_store = faiss_db
                st.session_state.bm25_retriever = bm25_ret
                save_persisted_state(
                    st.session_state.session_id,
                    faiss_db,
                    combined_chunks,
                    combined_files,
                    dense_weight,
                    embedding_model=EMBEDDING_MODEL_NAME,
                )
                st.success(f"Loaded `{demo_name}` ({len(sample_chunks)} chunks).")

    # Granular Per-Document Delete
    if st.session_state.indexed_files:
        st.markdown("**Indexed Corpus (Persisted):**")
        for fname in list(st.session_state.indexed_files):
            c1, c2 = st.columns([4, 1])
            c1.caption(f"📄 {fname}")
            if c2.button("✖", key=f"del_{fname}", help=f"Remove {fname}"):
                remaining_chunks = [
                    d for d in st.session_state.all_documents
                    if d.metadata.get("source_name") != fname
                ]
                remaining_files = [f for f in st.session_state.indexed_files if f != fname]
                st.session_state.all_documents = remaining_chunks
                st.session_state.indexed_files = remaining_files
                if remaining_chunks:
                    embed_model = get_embedding_model()
                    faiss_db, bm25_ret = build_or_update_index(
                        remaining_chunks, embed_model
                    )
                    st.session_state.faiss_store = faiss_db
                    st.session_state.bm25_retriever = bm25_ret
                    save_persisted_state(
                        st.session_state.session_id,
                        faiss_db,
                        remaining_chunks,
                        remaining_files,
                        dense_weight,
                        embedding_model=EMBEDDING_MODEL_NAME,
                    )
                else:
                    st.session_state.faiss_store = None
                    st.session_state.bm25_retriever = None
                    clear_persisted_state(st.session_state.session_id)
                st.rerun()

    st.divider()
    if st.session_state.chat_history:
        st.download_button(
            "⬇️ Export Chat (Markdown)",
            data=export_chat_markdown(st.session_state.chat_history),
            file_name="veridoc_transcript.md",
            mime="text/markdown",
            use_container_width=True,
        )

    if st.button("🗑️ Clear All & Reset", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.faiss_store = None
        st.session_state.bm25_retriever = None
        st.session_state.all_documents = []
        st.session_state.indexed_files = []
        st.session_state.last_inspection = None
        clear_persisted_state(st.session_state.session_id)
        st.rerun()


# ----------------- MAIN UI -----------------
st.markdown('<div class="main-header">⚡ VeriDoc AI — Enterprise Hybrid RAG Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Dense FAISS + Sparse BM25 · MS-MARCO Cross-Encoder Reranking · Grounded Citations & Evaluation</div>',
    unsafe_allow_html=True,
)

if st.session_state.get("startup_warning"):
    st.warning(st.session_state.startup_warning)

has_index = st.session_state.faiss_store is not None and st.session_state.bm25_retriever is not None

if not has_index:
    st.info(
        "👈 **No documents indexed yet:** Upload PDF/DOCX/TXT/CSV files or click **'📋 Load Demo Enterprise Doc'** in the sidebar."
    )
else:
    dense_pct = int(dense_weight * 100)
    sparse_pct = 100 - dense_pct
    st.markdown(
        f'<span class="metric-badge">🟢 Ready ({len(st.session_state.indexed_files)} file(s), {len(st.session_state.all_documents)} chunks)</span>'
        f'<span class="metric-badge">⚖️ {dense_pct}% FAISS / {sparse_pct}% BM25</span>'
        f'<span class="metric-badge">🎯 Reranker: {"ON" if use_reranker else "OFF"}</span>'
        f'<span class="metric-badge">🔒 Strict: {"ON" if strict_mode else "OFF"}</span>',
        unsafe_allow_html=True,
    )

active_view = st.radio(
    "Navigation",
    ["💬 Chat & Q&A", "🔍 Retrieval Inspector", "📊 Accuracy & Eval"],
    horizontal=True,
    label_visibility="collapsed",
)
st.divider()

# ================= TAB 1: CHAT & Q&A =================
if active_view == "💬 Chat & Q&A":
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "metrics" in message:
                    m = message["metrics"]
                    cit_info = message.get("citation_check", "")
                    st.caption(
                        f"⏱️ Retrieval: {m.get('retrieval_ms', 0)}ms | Rerank: {m.get('rerank_ms', 0)}ms | "
                        f"Gen: {m.get('gen_ms', 0)}ms | Total: {m.get('total_ms', 0)}ms"
                        + (f" | {cit_info}" if cit_info else "")
                    )
                if message.get("sources"):
                    with st.expander(f"🔎 Grounding Sources ({len(message['sources'])} relevant chunk(s))"):
                        render_source_cards(message["sources"])

    # Root-level chat_input stays pinned at the bottom below all chat messages
    user_query = st.chat_input("Ask a question about your documents...")

    if user_query:
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_query)
            assistant_slot = st.chat_message("assistant")

        with assistant_slot:
            try:
                if not groq_api_key:
                    raise ValueError("Groq API key is missing. Add it in Streamlit Secrets or the sidebar.")

                t_start = time.time()
                groq_llm = ChatGroq(model_name=selected_model, temperature=temperature, api_key=groq_api_key)

                # Case A: No documents indexed
                if not has_index:
                    retrieval_ms = 0
                    rerank_ms = 0
                    sources_data = []
                    cit_badge = ""

                    if strict_mode:
                        answer_content = "🔒 **Strict Mode:** No documents are currently indexed. Please upload a document or load the demo document first."
                        st.markdown(answer_content)
                        gen_ms = 0
                    else:
                        notice = "ℹ️ **Note:** No documents are currently indexed, so here is a general AI answer:\n\n"
                        st.markdown(notice)
                        t_gen_start = time.time()

                        fallback_msgs = _prepare_recent_history(st.session_state.chat_history) + [
                            ("human", user_query)
                        ]

                        def stream_gen():
                            for chunk in groq_llm.stream(fallback_msgs):
                                yield chunk.content

                        streamed_body = strip_think_tags(st.write_stream(stream_gen()))
                        answer_content = notice + streamed_body
                        gen_ms = int((time.time() - t_gen_start) * 1000)

                # Case B: Documents indexed -> Conversational Rewrite + Hybrid Search + Rerank
                else:
                    standalone_query = rewrite_followup_query(
                        groq_llm, user_query, st.session_state.chat_history
                    )
                    dense_query = None
                    if use_hyde:
                        hyde_resp = groq_llm.invoke(
                            f"Write a concise technical passage answering: '{standalone_query}'"
                        )
                        dense_query = strip_think_tags(hyde_resp.content)

                    t_ret_start = time.time()
                    ret_out = retrieve_and_rerank(
                        query=standalone_query,
                        bm25_ret=st.session_state.bm25_retriever,
                        faiss_db=st.session_state.faiss_store,
                        dense_weight=dense_weight,
                        top_k=top_k,
                        context_k=context_k,
                        use_reranker=use_reranker,
                        cross_encoder=get_cross_encoder() if use_reranker else None,
                        dense_query=dense_query,
                    )
                    retrieval_ms = int((time.time() - t_ret_start) * 1000)

                    # Store for Retrieval Inspector tab
                    st.session_state.last_inspection = {
                        "query": standalone_query,
                        "bm25_docs": ret_out["bm25_docs"],
                        "faiss_docs_with_scores": ret_out["faiss_docs_with_scores"],
                        "reranked": ret_out["scored_chunks"],
                    }

                    t_rerank_start = time.time()
                    scored_chunks = ret_out["scored_chunks"]
                    if use_reranker:
                        relevant_chunks = [
                            s for s in scored_chunks if s["score"] >= relevance_threshold
                        ]
                    else:
                        relevant_chunks = scored_chunks
                    rerank_ms = int((time.time() - t_rerank_start) * 1000)

                    t_gen_start = time.time()
                    if not relevant_chunks:
                        sources_data = []
                        cit_badge = ""
                        if strict_mode:
                            answer_content = (
                                "🔒 **Strict Mode:** Your uploaded documents do not contain relevant information "
                                "for this query (no chunks met the relevance threshold)."
                            )
                            st.markdown(answer_content)
                        else:
                            notice = (
                                "> ⚠️ **Note:** Your uploaded documents do not contain information about this query. "
                                "Here is a general AI answer:\n\n"
                            )
                            st.markdown(notice)

                            fallback_msgs = _prepare_recent_history(st.session_state.chat_history) + [
                                ("human", standalone_query)
                            ]

                            def stream_fallback():
                                for chunk in groq_llm.stream(fallback_msgs):
                                    yield chunk.content

                            streamed_body = strip_think_tags(st.write_stream(stream_fallback()))
                            answer_content = notice + streamed_body
                    else:
                        context_str = ""
                        retrieved_ids = []
                        sources_data = []
                        for rank_idx, item in enumerate(relevant_chunks, start=1):
                            doc = item["doc"]
                            prob = item["score"]
                            c_id = doc.metadata.get("chunk_id", "Unknown")
                            src_name = doc.metadata.get("source_name", "Document")
                            retrieved_ids.append(c_id)
                            context_str += f"\n--- [Source: {src_name} | ID: {c_id}] ---\n{doc.page_content}\n"
                            sources_data.append(
                                {
                                    "source": src_name,
                                    "chunk_id": c_id,
                                    "score": f"{prob * 100:.1f}%" if item["logit"] is not None else f"Rank #{rank_idx}",
                                    "snippet": doc.page_content[:220]
                                    + ("..." if len(doc.page_content) > 220 else ""),
                                }
                            )

                        grounded_prompt = f"""You are VeriDoc AI, an enterprise intelligence assistant.
Answer the USER QUESTION using ONLY the RETRIEVED CONTEXT below.

Rules:
1. If the RETRIEVED CONTEXT does not contain enough information to answer the question, output ONLY the exact token: NOT_FOUND
2. Otherwise, answer clearly using bullet points and bold highlights, and cite facts inline using `[Source: <filename>, ID: <chunk_id>]`.

RETRIEVED CONTEXT:
{context_str}

USER QUESTION:
{standalone_query}
"""
                        raw_resp = strip_think_tags(groq_llm.invoke(grounded_prompt).content)
                        if "NOT_FOUND" in raw_resp[:40]:
                            sources_data = []
                            cit_badge = ""
                            if strict_mode:
                                answer_content = "🔒 **Strict Mode:** Your uploaded documents do not contain enough information to answer this query."
                                st.markdown(answer_content)
                            else:
                                notice = (
                                    "> ⚠️ **Note:** Your uploaded documents do not contain information about this query. "
                                    "Here is a general AI answer:\n\n"
                                )
                                st.markdown(notice)

                                fallback_msgs = _prepare_recent_history(st.session_state.chat_history) + [
                                    ("human", standalone_query)
                                ]

                                def stream_fb2():
                                    for chunk in groq_llm.stream(fallback_msgs):
                                        yield chunk.content

                                streamed_body = strip_think_tags(st.write_stream(stream_fb2()))
                                answer_content = notice + streamed_body
                        else:
                            answer_content = raw_resp
                            st.markdown(answer_content)
                            valid_cits, invalid_cits = validate_citations(answer_content, retrieved_ids)
                            if invalid_cits:
                                cit_badge = f"⚠️ Unverified citations: {', '.join(invalid_cits)}"
                            elif valid_cits:
                                cit_badge = f"✅ {len(valid_cits)} citation(s) verified"
                            else:
                                cit_badge = "ℹ️ No citations found in answer"

                    gen_ms = int((time.time() - t_gen_start) * 1000)

                total_ms = int((time.time() - t_start) * 1000)
                metrics = {
                    "retrieval_ms": retrieval_ms,
                    "rerank_ms": rerank_ms,
                    "gen_ms": gen_ms,
                    "total_ms": total_ms,
                }
                st.caption(
                    f"⏱️ Retrieval: {retrieval_ms}ms | Rerank: {rerank_ms}ms | "
                    f"Gen: {gen_ms}ms | Total: {total_ms}ms"
                    + (f" | {cit_badge}" if cit_badge else "")
                )

                if sources_data:
                    with st.expander(f"🔎 Grounding Sources ({len(sources_data)} relevant chunk(s))"):
                        render_source_cards(sources_data)

                st.session_state.chat_history.append({"role": "user", "content": user_query})
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer_content,
                        "metrics": metrics,
                        "sources": sources_data,
                        "citation_check": cit_badge,
                    }
                )
                save_chat_history(st.session_state.session_id, st.session_state.chat_history)

            except Exception as e:
                st.error(f"Error during query execution: {e}")

# ================= TAB 2: RETRIEVAL INSPECTOR =================
elif active_view == "🔍 Retrieval Inspector":
    st.markdown("### 🔍 Hybrid Retrieval Inspector (BM25 vs FAISS vs Cross-Encoder)")
    insp = st.session_state.get("last_inspection")
    if not insp:
        st.info("Run a query against your indexed documents in the **Chat & Q&A** tab to inspect BM25 vs FAISS vs Cross-Encoder rankings side by side.")
    else:
        st.caption(f"**Last Standalone Query:** `{insp['query']}`")
        col_b, col_f, col_r = st.columns(3)
        with col_b:
            st.markdown("#### 🔤 BM25 (Sparse Lexical)")
            for i, d in enumerate(insp["bm25_docs"][:5], 1):
                st.markdown(f"**#{i}** `{d.metadata.get('chunk_id')}`")
                st.caption(d.page_content[:140] + "...")
        with col_f:
            st.markdown("#### 🧠 FAISS (Dense Semantic)")
            for i, (d, dist) in enumerate(insp["faiss_docs_with_scores"][:5], 1):
                st.markdown(f"**#{i}** `{d.metadata.get('chunk_id')}` (L2: `{dist:.3f}`)")
                st.caption(d.page_content[:140] + "...")
        with col_r:
            st.markdown("#### 🎯 Final Reranked")
            for i, item in enumerate(insp["reranked"][:5], 1):
                d = item["doc"]
                if item["logit"] is None:
                    st.markdown(f"**#{i}** `{d.metadata.get('chunk_id')}` (`Rank #{i}`)")
                else:
                    st.markdown(f"**#{i}** `{d.metadata.get('chunk_id')}` (`{item['score']*100:.1f}%`)")
                st.caption(d.page_content[:140] + "...")

# ================= TAB 3: ACCURACY & EVAL =================
elif active_view == "📊 Accuracy & Eval":
    st.markdown("### 📊 Automated RAG Accuracy Benchmark")
    st.write(
        "Evaluates the pipeline across **10 curated test cases** (5 in-domain technical questions + 5 off-topic rejection checks: 2 general + 3 near-domain) "
        "measuring **Retrieval Hit Rate@K**, **MRR**, **Faithfulness**, **Answer Correctness**, and **Off-Topic Rejection**."
    )
    eval_enabled = os.getenv("VERIDOC_ENABLE_EVAL", "").strip() == "1"
    if not eval_enabled:
        st.info("Evaluation runner is disabled in this environment (set `VERIDOC_ENABLE_EVAL=1` to enable).")
    elif st.button("▶️ Run Accuracy Benchmark Now"):
        if not groq_api_key:
            st.error("Groq API key is missing. Add it in `.env`, Streamlit Secrets, or the sidebar.")
        else:
            with st.spinner("Running 10-case RAG evaluation suite..."):
                results = run_evaluation(
                    embed_model=get_embedding_model(),
                    cross_encoder=get_cross_encoder(),
                    model_name=selected_model,
                    dense_weight=dense_weight,
                    top_k=top_k,
                    context_k=context_k,
                    relevance_threshold=relevance_threshold,
                    api_key=groq_api_key,
                )
                st.session_state.eval_results = results

    if st.session_state.eval_results:
        res = st.session_state.eval_results
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Hit Rate@K", f"{res['hit_rate']:.1f}%")
        c2.metric("MRR", f"{res['mrr']:.2f}")
        c3.metric("Faithfulness", f"{res['avg_faithfulness']:.1f}%")
        c4.metric("Correctness", f"{res['avg_correctness']:.1f}%")
        c5.metric("Off-Topic Rejection", f"{res['off_topic_rejection']:.1f}%")
        st.dataframe(res["details"], use_container_width=True)
