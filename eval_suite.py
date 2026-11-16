import json
import os
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from ingest import get_sample_document
from retrieval import build_or_update_index, retrieve_and_rerank, strip_think_tags

GENERATOR_MODEL = "qwen/qwen3.8-27b"
JUDGE_MODEL = "openai/gpt-oss-120b"

# Curated ground-truth evaluation dataset for the Enterprise Architecture doc (5 In-Domain + 5 Off-Topic)
EVAL_DATASET: List[Dict[str, Any]] = [
    {
        "question": "What OAuth 2.0 flow and token expiration TTLs does ACME Corp mandate?",
        "expected_answer": "OAuth 2.0 with PKCE, 15-minute access token TTL, and 24-hour rotating refresh tokens.",
        "expected_keyword": "PKCE",
        "is_off_topic": False,
        "category": "In-Domain",
    },
    {
        "question": "What encryption standards are required for data at rest and in transit?",
        "expected_answer": "AES-256 encryption at rest and TLS 1.3 minimum in transit.",
        "expected_keyword": "AES-256",
        "is_off_topic": False,
        "category": "In-Domain",
    },
    {
        "question": "How must developers resolve N+1 query spikes in Spring Boot & Hibernate?",
        "expected_answer": "Use explicit JOIN FETCH or batch fetching with a maximum batch size of 50; queries over 200ms trigger APM alerts.",
        "expected_keyword": "JOIN FETCH",
        "is_off_topic": False,
        "category": "In-Domain",
    },
    {
        "question": "What is ACME Corp's SLA availability target and failover time?",
        "expected_answer": "99.99% availability (less than 52.6 minutes annual downtime), active-passive replicas in Mumbai and Pune, and 30-second automated failover.",
        "expected_keyword": "99.99%",
        "is_off_topic": False,
        "category": "In-Domain",
    },
    {
        "question": "What is the minimum cross-encoder confidence score required for AI RAG guardrails?",
        "expected_answer": "Minimum cross-encoder confidence score of 0.65 (65%), hallucination checks cross-referencing citation IDs, and zero data retention.",
        "expected_keyword": "0.65",
        "is_off_topic": False,
        "category": "In-Domain",
    },
    {
        "question": "What is the capital of France and who won the 2022 World Cup?",
        "expected_answer": "NOT_FOUND",
        "expected_keyword": "",
        "is_off_topic": True,
        "category": "Off-Topic (General)",
    },
    {
        "question": "Explain how a Java HashMap handles hash collisions internally.",
        "expected_answer": "NOT_FOUND",
        "expected_keyword": "",
        "is_off_topic": True,
        "category": "Off-Topic (General)",
    },
    {
        "question": "What was ACME Corp's 2025 annual revenue and EBITDA margin?",
        "expected_answer": "NOT_FOUND",
        "expected_keyword": "",
        "is_off_topic": True,
        "category": "Off-Topic (Near-Domain)",
    },
    {
        "question": "What is ACME Corp's enterprise SLA breach refund and credit policy?",
        "expected_answer": "NOT_FOUND",
        "expected_keyword": "",
        "is_off_topic": True,
        "category": "Off-Topic (Near-Domain)",
    },
    {
        "question": "What is ACME Corp's cloud migration budget and target completion quarter?",
        "expected_answer": "NOT_FOUND",
        "expected_keyword": "",
        "is_off_topic": True,
        "category": "Off-Topic (Near-Domain)",
    },
]


def run_evaluation(
    embed_model,
    cross_encoder,
    model_name: str = GENERATOR_MODEL,
    judge_model: str = JUDGE_MODEL,
    dense_weight: float = 0.5,
    top_k: int = 6,
    context_k: int = 4,
    relevance_threshold: float = 0.25,
    api_key: str = "",
) -> Dict[str, Any]:
    """
    Runs the full RAG accuracy benchmark on the sample enterprise document.
    Returns summary metrics and per-question results.
    """
    chunks = get_sample_document()
    faiss_db, bm25_ret = build_or_update_index(chunks, embed_model)
    resolved_key = api_key or os.getenv("GROQ_API_KEY", "")
    llm = ChatGroq(model_name=model_name, temperature=0.0, api_key=resolved_key)
    judge_llm = ChatGroq(model_name=judge_model, temperature=0.0, api_key=resolved_key)

    in_domain_total = 0
    hits = 0
    mrr_sum = 0.0
    faithfulness_scores: List[float] = []
    correctness_scores: List[float] = []
    judge_fallbacks = 0

    off_topic_total = 0
    off_topic_rejected = 0
    threshold_gate_rejected = 0

    details: List[Dict[str, Any]] = []

    for item in EVAL_DATASET:
        q = item["question"]
        is_off = item["is_off_topic"]
        category = item.get("category", "Off-Topic" if is_off else "In-Domain")
        expected_kw = item["expected_keyword"]

        res = retrieve_and_rerank(
            query=q,
            bm25_ret=bm25_ret,
            faiss_db=faiss_db,
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
            passed_gate = top_prob < relevance_threshold
            if passed_gate:
                threshold_gate_rejected += 1
                rejected = True
                rejection_stage = "Threshold Gate"
            else:
                ctx = "\n".join([s["doc"].page_content for s in scored])
                ans = strip_think_tags(
                    llm.invoke(
                        f"If the context does not answer the question, output ONLY 'NOT_FOUND'.\nContext:\n{ctx}\nQuestion: {q}"
                    ).content
                )
                rejected = "NOT_FOUND" in ans[:40]
                rejection_stage = "NOT_FOUND Sentinel (Gate Missed)" if rejected else "FAILED"

            if rejected:
                off_topic_rejected += 1
            details.append({
                "question": q,
                "type": category,
                "hit": "N/A",
                "top_relevance": f"{top_prob * 100:.1f}%",
                "faithfulness": "N/A",
                "correctness": f"100% ({rejection_stage})" if rejected else "0% (False Positive)",
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

        # Separate Judge LLM for Faithfulness & Correctness
        judge_prompt = f"""Score the generated answer on two 0-100 scales and return ONLY valid JSON:
{{"faithfulness": <int 0-100, 100 if every claim is supported by Context>, "correctness": <int 0-100, factual match with Expected Answer>}}

Context:
{context_str}

Expected Answer:
{item['expected_answer']}

Generated Answer:
{gen_ans}
"""
        judge_raw = strip_think_tags(judge_llm.invoke(judge_prompt).content)
        f_score, c_score = 80.0, 80.0
        parsed_ok = False
        match = re.search(r"\{.*\}", judge_raw, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                f_score = float(parsed.get("faithfulness", 80))
                c_score = float(parsed.get("correctness", 80))
                parsed_ok = True
            except Exception:
                pass
        if not parsed_ok:
            judge_fallbacks += 1

        faithfulness_scores.append(f_score)
        correctness_scores.append(c_score)

        details.append({
            "question": q,
            "type": category,
            "hit": f"Rank #{hit_rank}" if hit_rank > 0 else "Miss",
            "top_relevance": f"{top_prob * 100:.1f}%",
            "faithfulness": f"{f_score:.0f}%",
            "correctness": f"{c_score:.0f}%",
        })

    return {
        "generator_model": model_name,
        "judge_model": judge_model,
        "hit_rate": (hits / in_domain_total * 100) if in_domain_total else 0.0,
        "mrr": (mrr_sum / in_domain_total) if in_domain_total else 0.0,
        "avg_faithfulness": (sum(faithfulness_scores) / len(faithfulness_scores)) if faithfulness_scores else 0.0,
        "avg_correctness": (sum(correctness_scores) / len(correctness_scores)) if correctness_scores else 0.0,
        "off_topic_rejection": (off_topic_rejected / off_topic_total * 100) if off_topic_total else 0.0,
        "threshold_gate_rejection": (threshold_gate_rejected / off_topic_total * 100) if off_topic_total else 0.0,
        "judge_calls": in_domain_total,
        "judge_fallbacks": judge_fallbacks,
        "details": details,
    }


if __name__ == "__main__":
    from langchain_huggingface import HuggingFaceEmbeddings
    from sentence_transformers import CrossEncoder

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    emb = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    runs = []
    for run_idx in range(1, 4):
        res = run_evaluation(emb, ce, model_name=GENERATOR_MODEL, judge_model=JUDGE_MODEL)
        runs.append(res)
        print(
            f"Run {run_idx} | Gen={res['generator_model']} | Judge={res['judge_model']} | "
            f"Hit={res['hit_rate']:.1f}% | MRR={res['mrr']:.2f} | "
            f"Faith={res['avg_faithfulness']:.1f}% | Corr={res['avg_correctness']:.1f}% | "
            f"OffTopic={res['off_topic_rejection']:.1f}% (Gate={res['threshold_gate_rejection']:.1f}%) | "
            f"Fallbacks={res['judge_fallbacks']} of {res['judge_calls']}"
        )

    print("\n--- 3-Run Summary (Mean & Range) ---")
    for key, label, suffix in [
        ("hit_rate", "Hit Rate@K", "%"),
        ("mrr", "MRR", ""),
        ("avg_faithfulness", "Faithfulness", "%"),
        ("avg_correctness", "Correctness", "%"),
        ("off_topic_rejection", "Off-Topic Rejection (Total)", "%"),
        ("threshold_gate_rejection", "Off-Topic Gate Only", "%"),
    ]:
        vals = [r[key] for r in runs]
        print(f"{label}: mean={sum(vals)/len(vals):.1f}{suffix} | range={min(vals):.1f}..{max(vals):.1f}{suffix}")
    total_fb = sum(r["judge_fallbacks"] for r in runs)
    total_jc = sum(r["judge_calls"] for r in runs)
    print(f"Total Judge Fallbacks: {total_fb} of {total_jc} judge calls")
    print("\n--- Per-Case Breakdown (Run 3) ---")
    for d in runs[-1]["details"]:
        print(d)
