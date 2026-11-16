import math
import re
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

try:
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain.retrievers import EnsembleRetriever
    except ImportError:
        from langchain_community.retrievers.ensemble import EnsembleRetriever


def bm25_tokenize(text: str) -> List[str]:
    """Case- and punctuation-insensitive tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


def sigmoid(logit: float) -> float:
    """Convert raw Cross-Encoder logit into a [0, 1] confidence probability."""
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    z = math.exp(logit)
    return z / (1.0 + z)


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks emitted by some LLMs."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def build_or_update_index(
    chunks: List[Document],
    embed_model,
    dense_weight: float = 0.5,
    existing_faiss: FAISS = None,
    new_chunks: List[Document] = None,
) -> Tuple[FAISS, BM25Retriever, EnsembleRetriever]:
    """
    Incrementally add new_chunks to existing_faiss if provided, or build fresh.
    Always rebuilds BM25 (fast in-memory) over the full chunks list.
    """
    if existing_faiss is not None and new_chunks:
        existing_faiss.add_documents(new_chunks)
        faiss_db = existing_faiss
    else:
        faiss_db = FAISS.from_documents(chunks, embed_model)

    faiss_retriever = faiss_db.as_retriever(search_kwargs={"k": 8})
    bm25_ret = BM25Retriever.from_documents(chunks, preprocess_func=bm25_tokenize)
    bm25_ret.k = 8

    ensemble = EnsembleRetriever(
        retrievers=[bm25_ret, faiss_retriever],
        weights=[round(1.0 - dense_weight, 2), round(dense_weight, 2)],
    )
    return faiss_db, bm25_ret, ensemble


def retrieve_and_rerank(
    query: str,
    bm25_ret: BM25Retriever,
    faiss_db: FAISS,
    ensemble: EnsembleRetriever,
    dense_weight: float,
    top_k: int,
    context_k: int,
    use_reranker: bool,
    cross_encoder=None,
) -> Dict[str, Any]:
    """
    Execute BM25, FAISS, Hybrid Ensemble, and Cross-Encoder reranking.
    Returns ranked chunks plus separate BM25 & FAISS lists for the Retrieval Inspector.
    """
    bm25_ret.k = top_k
    faiss_ret = faiss_db.as_retriever(search_kwargs={"k": top_k})
    ensemble.retrievers = [bm25_ret, faiss_ret]
    ensemble.weights = [round(1.0 - dense_weight, 2), round(dense_weight, 2)]

    bm25_docs = bm25_ret.invoke(query)
    faiss_docs_with_scores = faiss_db.similarity_search_with_score(query, k=top_k)
    faiss_docs = [doc for doc, _ in faiss_docs_with_scores]
    hybrid_docs = ensemble.invoke(query)

    scored_chunks: List[Dict[str, Any]] = []
    if use_reranker and cross_encoder is not None and hybrid_docs:
        pairs = [[query, doc.page_content] for doc in hybrid_docs]
        logits = cross_encoder.predict(pairs)
        for doc, logit in zip(hybrid_docs, logits):
            prob = sigmoid(float(logit))
            scored_chunks.append({"doc": doc, "score": prob, "logit": float(logit)})
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    else:
        for rank, doc in enumerate(hybrid_docs):
            # Reciprocal rank approximation when Cross-Encoder is OFF
            approx_score = round(1.0 / (rank + 1), 4)
            scored_chunks.append({"doc": doc, "score": approx_score, "logit": None})

    return {
        "scored_chunks": scored_chunks[:context_k],
        "all_scored": scored_chunks,
        "bm25_docs": bm25_docs,
        "faiss_docs_with_scores": faiss_docs_with_scores,
    }


def validate_citations(answer_text: str, retrieved_chunk_ids: List[str]) -> Tuple[List[str], List[str]]:
    """
    Parse chunk IDs cited in `[Source: ..., ID: <chunk_id>]` and verify they exist in retrieved_chunk_ids.
    Returns: (valid_ids, invalid_ids)
    """
    found_ids = re.findall(r"ID:\s*([^\],\n]+)", answer_text)
    found_ids = [cid.strip() for cid in found_ids]
    valid = [cid for cid in found_ids if cid in retrieved_chunk_ids]
    invalid = [cid for cid in found_ids if cid not in retrieved_chunk_ids]
    return list(dict.fromkeys(valid)), list(dict.fromkeys(invalid))
