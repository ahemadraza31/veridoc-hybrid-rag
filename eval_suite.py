import json
import re
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from ingest import get_sample_document
from retrieval import build_or_update_index, retrieve_and_rerank, strip_think_tags

# Curated ground-truth evaluation dataset for the Enterprise Architecture doc
EVAL_DATASET: List[Dict[str, Any]] = [
    {
        "question": "What OAuth 2.0 flow and token expiration TTLs does ACME Corp mandate?",
        "expected_answer": "OAuth 2.0 with PKCE, 15-minute access token TTL, and 24-hour rotating refresh tokens.",
        "expected_keyword": "PKCE",
        "is_off_topic": False,
    },
    {
        "question": "What encryption standards are required for data at rest and in transit?",
        "expected_answer": "AES-256 encryption at rest and TLS 1.3 minimum in transit.",
        "expected_keyword": "AES-256",
        "is_off_topic": False,
    },
    {
        "question": "How must developers resolve N+1 query spikes in Spring Boot & Hibernate?",
        "expected_answer": "Use explicit JOIN FETCH or batch fetching with a maximum batch size of 50; queries over 200ms trigger APM alerts.",
        "expected_keyword": "JOIN FETCH",
        "is_off_topic": False,
    },
    {
        "question": "What is ACME Corp's SLA availability target and failover time?",
        "expected_answer": "99.99% availability (less than 52.6 minutes annual downtime), active-passive replicas in Mumbai and Pune, and 30-second automated failover.",
        "expected_keyword": "99.99%",
        "is_off_topic": False,
    },
    {
        "question": "What is the minimum cross-encoder confidence score required for AI RAG guardrails?",
        "expected_answer": "Minimum cross-encoder confidence score of 0.65 (65%), hallucination checks cross-referencing citation IDs, and zero data retention.",
        "expected_keyword": "0.65",
        "is_off_topic": False,
    },
    {
        "question": "What is the capital of France and who won the 2022 World Cup?",
        "expected_answer": "NOT_FOUND",
        "expected_keyword": "",
        "is_off_topic": True,
    },
    {
        "question": "Explain how a Java HashMap handles hash collisions internally.",
        "expected_answer": "NOT_FOUND",
        "expected_keyword": "",
        "is_off_topic": True,
    },
]


def run_evaluation(
    embed_model,
    cross_encoder,
    model_name: str = "openai/gpt-oss-20b",
    dense_weight: float = 0.5,
    top_k: int = 6,
    context_k: int = 4,
    confidence_threshold: float = 0.25,
) -> Dict[str, Any]:
    """
    Runs the full RAG accuracy benchmark on the sample enterprise document.
    Returns summary metrics and per-question results.
    """
    chunks = get_sample_document()
    faiss_db, bm25_ret, ensemble = build_or_update_index(chunks, embed_model, dense_weight=dense_weight)
    llm = ChatGroq(model_name=model_name, temperature=0.0)

    in_domain_total = 0
    hits = 0
    mrr_sum = 0.0
    faithfulness_scores: List[float] = []
    correctness_scores: List[float] = []

    off_topic_total = 0
    off_topic_rejected = 0

    details: List[Dict[str, Any]] = []

    for item in EVAL_DATASET:
        q = item["question"]
        is_off = item["is_off_topic"]
        expected_kw = item["expected_keyword"]

        res = retrieve_and_rerank(
            query=q,
            bm25_ret=bm25_ret,
            faiss_db=faiss_db,
            ensemble=ensemble,
            dense_weight=dense_weight,
            top_k=top_k,
            context_k=context_k,
            use_reranker=True,
            cross_encoder=cross_encoder,
        )
        scored = res["scored_chunks"]
        top_prob = scored[0]["score"] if scored else 0.0

        if is_off:
            off_topic_total += 1
            rejected = top_prob < confidence_threshold
            if not rejected:
                # Check if LLM returns NOT_FOUND sentinel
                ctx = "\n".join([s["doc"].page_content for s in scored])
                ans = strip_think_tags(
                    llm.invoke(
                        f"If the context does not answer the question, output ONLY 'NOT_FOUND'.\nContext:\n{ctx}\nQuestion: {q}"
                    ).content
                )
                rejected = "NOT_FOUND" in ans
            if rejected:
                off_topic_rejected += 1
            details.append({
                "question": q,
                "type": "Off-Topic",
                "hit": "N/A",
                "top_confidence": f"{top_prob * 100:.1f}%",
                "faithfulness": "N/A",
                "correctness": "100%" if rejected else "0% (False Positive)",
            })
            continue

        # In-domain evaluation
        in_domain_total += 1
        hit_rank = 0
        for idx, s in enumerate(scored, start=1):
            if expected_kw.lower() in s["doc"].page_content.lower():
                hit_rank = idx
                break
        if hit_rank > 0:
            hits += 1
            mrr_sum += 1.0 / hit_rank

        context_str = "\n".join(
            [f"[{s['doc'].metadata.get('chunk_id')}] {s['doc'].page_content}" for s in scored]
        )
        gen_ans = strip_think_tags(
            llm.invoke(
                f"Answer strictly from the context below. Be concise.\n\nContext:\n{context_str}\n\nQuestion: {q}"
            ).content
        )

        # LLM-as-a-Judge for Faithfulness & Correctness
        judge_prompt = f"""Score the generated answer on two 0-100 scales and return ONLY valid JSON:
{{"faithfulness": <int 0-100, 100 if every claim is supported by Context>, "correctness": <int 0-100, factual match with Expected Answer>}}

Context:
{context_str}

Expected Answer:
{item['expected_answer']}

Generated Answer:
{gen_ans}
"""
        judge_raw = strip_think_tags(llm.invoke(judge_prompt).content)
        f_score, c_score = 80.0, 80.0
        match = re.search(r"\{.*\}", judge_raw, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                f_score = float(parsed.get("faithfulness", 80))
                c_score = float(parsed.get("correctness", 80))
            except Exception:
                pass

        faithfulness_scores.append(f_score)
        correctness_scores.append(c_score)

        details.append({
            "question": q,
            "type": "In-Domain",
            "hit": f"Rank #{hit_rank}" if hit_rank > 0 else "Miss",
            "top_confidence": f"{top_prob * 100:.1f}%",
            "faithfulness": f"{f_score:.0f}%",
            "correctness": f"{c_score:.0f}%",
        })

    return {
        "hit_rate": (hits / in_domain_total * 100) if in_domain_total else 0.0,
        "mrr": (mrr_sum / in_domain_total) if in_domain_total else 0.0,
        "avg_faithfulness": (sum(faithfulness_scores) / len(faithfulness_scores)) if faithfulness_scores else 0.0,
        "avg_correctness": (sum(correctness_scores) / len(correctness_scores)) if correctness_scores else 0.0,
        "off_topic_rejection": (off_topic_rejected / off_topic_total * 100) if off_topic_total else 0.0,
        "details": details,
    }
