import math
import re
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever


def bm25_tokenize(text: str) -> List[str]:
    """Case- and punctuation-insensitive tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


def sigmoid(logit: float) -> float:
    """Convert raw Cross-Encoder logit into a [0, 1] relevance score (not a calibrated probability)."""
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    z = math.exp(logit)
    return z / (1.0 + z)


def strip_think_tags(text: str) -> str:
    """Remove closed and unclosed <think> reasoning blocks; raise if empty afterward."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL).strip()
    if not cleaned:
        raise ValueError("Model returned an empty response after stripping reasoning tags.")
    return cleaned


def weighted_rrf(
    bm25_docs: List[Document],
    faiss_docs: List[Document],
    dense_weight: float = 0.5,
    c: int = 60,
) -> List[Tuple[Document, float]]:
    """Fuse BM25 and FAISS ranked lists using weighted Reciprocal Rank Fusion (RRF)."""
    sparse_weight = round(1.0 - dense_weight, 4)
    scores: Dict[Tuple[str, str], float] = {}
    doc_map: Dict[Tuple[str, str], Document] = {}

    for rank, doc in enumerate(bm25_docs, start=1):
        key = (str(doc.metadata.get("chunk_id", "")), doc.page_content)
        doc_map[key] = doc
        scores[key] = scores.get(key, 0.0) + sparse_weight / (c + rank)

    for rank, doc in enumerate(faiss_docs, start=1):
        key = (str(doc.metadata.get("chunk_id", "")), doc.page_content)
        doc_map[key] = doc
        scores[key] = scores.get(key, 0.0) + dense_weight / (c + rank)

    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(doc_map[key], rrf_score) for key, rrf_score in fused]


def build_or_update_index(
    chunks: List[Document],
    embed_model,
    existing_faiss: Optional[FAISS] = None,
    new_chunks: Optional[List[Document]] = None,
) -> Tuple[FAISS, BM25Retriever]:
    """
    Reuse existing_faiss when provided (only embedding new_chunks if non-empty),
    or build a fresh FAISS index. Always rebuilds BM25 over the full chunks list.
    """
    if existing_faiss is not None:
        faiss_db = existing_faiss
        if new_chunks:
            faiss_db.add_documents(new_chunks)
    else:
        faiss_db = FAISS.from_documents(chunks, embed_model)

    bm25_ret = BM25Retriever.from_documents(chunks, preprocess_func=bm25_tokenize)
    bm25_ret.k = 8
    return faiss_db, bm25_ret


def retrieve_and_rerank(
    query: str,
    bm25_ret: BM25Retriever,
    faiss_db: FAISS,
    dense_weight: float,
    top_k: int,
    context_k: int,
    use_reranker: bool,
    cross_encoder=None,
    dense_query: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute BM25 (on query) and FAISS (on dense_query or query) once, fuse with weighted RRF (c=60),
    and optionally rerank with Cross-Encoder on query.
    """
    bm25_ret.k = top_k
    bm25_docs = bm25_ret.invoke(query)

    faiss_q = dense_query if dense_query else query
    faiss_docs_with_scores = faiss_db.similarity_search_with_score(faiss_q, k=top_k)
    faiss_docs = [doc for doc, _ in faiss_docs_with_scores]

    fused_docs = weighted_rrf(bm25_docs, faiss_docs, dense_weight=dense_weight, c=60)

    scored_chunks: List[Dict[str, Any]] = []
    if use_reranker and cross_encoder is not None and fused_docs:
        candidates = [doc for doc, _ in fused_docs]
        pairs = [[query, doc.page_content] for doc in candidates]
        logits = cross_encoder.predict(pairs)
        for doc, logit in zip(candidates, logits):
            prob = sigmoid(float(logit))
            scored_chunks.append({"doc": doc, "score": prob, "logit": float(logit)})
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    else:
        for doc, rrf_score in fused_docs:
            scored_chunks.append({"doc": doc, "score": float(rrf_score), "logit": None})

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
