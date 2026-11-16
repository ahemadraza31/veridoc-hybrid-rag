# ⚡ VeriDoc AI — Hybrid RAG Document Q&A Engine

Upload PDF, Word, text, or CSV documents and get fast, source-cited answers backed by hybrid vector + keyword retrieval, Cross-Encoder re-ranking, and relevance-score gating.

[![Python](https://img.shields.io/badge/Python-3.13.9-blue.svg)](#run-it-yourself)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38%2B-FF4B4B.svg)](https://streamlit.io)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-brightgreen.svg)](https://veridoc-rag-ahemad.streamlit.app/)

---

## Live Demo

🚀 **[veridoc-rag-ahemad.streamlit.app](https://veridoc-rag-ahemad.streamlit.app/)**

> *Free Streamlit apps sleep when idle. If you see a "wake up" button, click it and wait ~30 seconds.*

<!-- Record docs/demo.gif — ask a question, show the answer, source cards, and relevance score -->
![VeriDoc AI demo showing a user question, grounded answer, source cards, and relevance score](docs/demo.gif)

---

## Why I Built This

Reading through 50–100 page technical specs, legal notes, or company policies to verify a single rule is slow, while pasting large files straight into a chat assistant often misses exact terms, mixes unrelated sections, or invents references.

VeriDoc AI combines meaning-based search with exact keyword search, re-scores candidate passages using a cross-encoder relevance score, blocks low-relevance matches from polluting the answer, and checks cited chunk identifiers against what was actually retrieved before showing sources.

---

## Architecture

### Ingestion (runs once per upload)

```mermaid
flowchart TB
    A["Upload\n(PDF, DOCX, TXT, CSV)"] --> B["Parse & Clean\n(ingest.py)"]
    B --> C["Chunk Text\n(Recursive Splitter)"]
    C --> D["Embed\n(all-MiniLM-L6-v2)"]
    subgraph Indexing ["Parallel Indexing"]
        direction LR
        E[("FAISS Vector\nIndex")]
        F[("BM25 Keyword\nIndex")]
    end
    D --> E
    C --> F
    E --> G["Persist FAISS + chunks\n(BM25 rebuilt on load)"]
    F --> G
```

Uploads are parsed per file in `ingest.py`, scrubbed of temporary paths, and split into overlapping text chunks. Already-indexed filenames are skipped to avoid duplicate chunks, and new chunks are appended incrementally to the `FAISS` vector store while the `BM25` keyword retriever is rebuilt in memory. Both the `FAISS` index and serialized document chunks are persisted per session in `storage.py`, and `BM25` is rebuilt from those chunks on reload without re-embedding.

### Query (runs on every question)

```mermaid
flowchart TB
    Q["User Question\n(+ Rewrite / Optional HyDE)"] --> S
    Q --> K
    subgraph Retrieval ["Parallel Retrieval"]
        direction LR
        S[("FAISS Vector\nSearch")]
        K[("BM25 Keyword\nSearch")]
    end
    S --> F["Weighted RRF Fusion\n(c=60, retrieval.py)"]
    K --> F
    F --> R["Cross-Encoder Rerank\n(ms-marco-MiniLM-L-6-v2)"]
    R --> P["Relevance Score\n(0-100%)"]
    P --> G{"Above Relevance\nThreshold?"}
    G -- "Yes" --> L["Groq LLM\n(Context Only)"]
    L -- "Grounded" --> V["Citation Check\n(validate_citations)"]
    V --> O["Answer + Source Cards\n+ Relevance Score"]
    G -- "No" --> N["Off-Topic Path\n(Warn + General AI or Strict Refuse)"]
    L -- "NOT_FOUND" --> N
```

Every question can be rewritten into a standalone query from recent conversation history; if `HyDE` is enabled, the hypothetical passage is sent only to `FAISS`, while `BM25` and the Cross-Encoder use the standalone question. Results are merged once through a local weighted RRF function in `retrieval.py` and re-scored by the MS-MARCO Cross-Encoder into a `0–100%` relevance score. Queries whose chunks fall below the threshold—or whose grounded LLM pass returns `NOT_FOUND`—route to the off-topic path (refuse in Strict Mode, or warn and stream a general AI reply), while grounded answers pass through `validate_citations`.

---

## Key Features

- **Supported Document Formats:** Parses `.pdf` (`PyPDFLoader`), `.docx` (`Docx2txtLoader`), `.txt` (`TextLoader`, UTF-8), and `.csv` (`CSVLoader`), plus a 1-click **Load Demo Enterprise Doc** (`ACME_Enterprise_Arch_2026.pdf`). *(Excel `.xlsx`/`.xls` and legacy `.doc` files are not supported.)*
- **Hybrid Search + Custom BM25:** Pairs L2-normalized `FAISS` vectors with a case- and punctuation-insensitive `BM25` word tokenizer.
- **Cross-Encoder Relevance Gate & Source Cards:** Converts raw MS-MARCO logits to a `0–100%` relevance score, filters out low-scoring distractors, HTML-escapes snippets, and verifies inline citations.
- **Strict Mode & Off-Topic Fallback:** Refuses off-topic questions when **Strict Document-Only Mode** is enabled, or explicitly warns the user before streaming a general AI answer.
- **Session Persistence & Management:** Validates `?sid=` session IDs, verifies chunk count and embedding model metadata on reload, supports per-document deletion, and exports transcripts to Markdown.
- **Inspection & Eval Views:** Includes a **Retrieval Inspector** (side-by-side BM25 vs. FAISS vs. Cross-Encoder rankings) and an env-gated **Accuracy & Eval** smoke-test runner.

---

## Tech Stack

| Layer | Choice | Why |
| :--- | :--- | :--- |
| **UI & Hosting** | Streamlit (`>=1.38.0`) | Interactive sidebar controls, streaming fallback, and free cloud hosting |
| **Orchestration** | LangChain (`core`, `community`, `text-splitters`) | Standard document loaders, text splitters, and LLM/vector integrations |
| **Document Parsing** | `pypdf`, `docx2txt`, `CSVLoader`, `TextLoader` | Lightweight text extraction for `.pdf`, `.docx`, `.csv`, and `.txt` |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Fast local embeddings with unit-norm cosine similarity behavior |
| **Vector & Sparse Stores** | `faiss-cpu` + `rank-bm25` | Local semantic similarity search paired with exact keyword matching |
| **Fusion & Re-Ranker** | Weighted RRF (`retrieval.py`) + `cross-encoder/ms-marco-MiniLM-L-6-v2` | Single-pass rank fusion followed by joint `(query, chunk)` scoring |
| **LLM Inference** | Groq (`ChatGroq`) | Low-latency LPU inference with selectable models and `<think>` tag stripping |
| **Persistence** | `FAISS.save_local` + atomic JSON | Zero-infrastructure session-namespaced storage on local disk |

---

## Design Decisions

- **Why hybrid search (`FAISS` + `BM25`):** Dense embeddings (`all-MiniLM-L6-v2`, `normalize_embeddings=True`) match paraphrases and synonyms, while `BM25` tokenized via `re.findall(r"\w+", text.lower())` catches exact tokens like `PKCE`, `TLS 1.3`, or `99.99%` regardless of casing or punctuation.
- **Why a two-stage re-ranker:** First-stage retrieval pulls `top_k` candidates per retriever (`2–12`, default `6`), then `cross-encoder/ms-marco-MiniLM-L-6-v2` jointly scores those fused candidates on the standalone query and keeps the top `context_k` chunks (`1–8`, default `4`); optional `HyDE` alters only the dense `FAISS` query.
- **How fusion works:** A local weighted Reciprocal Rank Fusion function (`weighted_rrf` in `retrieval.py`, `c=60`) merges the `BM25` and `FAISS` lists already retrieved in one pass, weighted by `dense_weight` (`0.0–1.0`, default `0.5`).
- **How the relevance score is computed:** The `0–100%` relevance score is a `sigmoid` of raw Cross-Encoder logits (`1 / (1 + exp(-logit))`), not a calibrated probability; the default cutoff (`25%` / `0.25`, tunable `5%–80%`) was tuned empirically in `app.py` and `eval_suite.py` rather than via a statistical calibration script.
- **Chunking & history choices:** Uploaded files use `RecursiveCharacterTextSplitter` with `chunk_size=750`, `chunk_overlap=150`, and `["\n\n", "\n", " ", ""]` separators (`420` / `50` for the shorter built-in demo doc), while follow-up rewrites use the last `4` chat turns (capped at `250` chars each).
- **Why Groq & tag handling:** `ChatGroq` runs `qwen/qwen3.8-27b` (default), `openai/gpt-oss-120b`, or `openai/gpt-oss-20b` at default `temperature=0.1` (`0.0–0.7`) using explicit `api_key=...` (never mutating `os.environ`); `strip_think_tags` strips both closed and unclosed `<think>` blocks and raises an error if empty afterward, while `NOT_FOUND` is checked within the first `40` characters.
- **How citations & sessions are checked:** `validate_citations` extracts chunk IDs with `r"ID:\s*([^\],\n]+)"` from `[Source: ..., ID: ...]` tags; session IDs are restricted to `[0-9a-f]{8,32}` in `?sid=`, persisted atomically via temp file + `os.replace`, verified against `index.ntotal` and `embedding_model` on load, and purged after `7` days.
- **Gated evaluation:** The UI benchmark button in `app.py` is hidden unless `VERIDOC_ENABLE_EVAL=1` is set in the environment.

---

## 7-Case Smoke Test

Evaluated on `ACME_Enterprise_Arch_2026.pdf` via `eval_suite.py` using `qwen/qwen3.8-27b` (`temperature=0.0` for evaluation and judging):

- **Dataset split:** **5 In-Domain** technical questions + **2 Off-Topic** questions.
- **Judge & fallback behavior:** The same selected `ChatGroq` model (`qwen/qwen3.8-27b`) acts as LLM-as-a-Judge for faithfulness and correctness. `eval_suite.py` tracks JSON parse failures in `judge_fallbacks` (falling back to `80.0` if parsing fails); on this run **`judge_fallbacks` was `0`** (all 5 in-domain judge calls parsed cleanly: faithfulness `100%` × 5; correctness `100%, 100%, 50%, 50%, 33%`).
- **How each metric is scored:**
  - **Hit Rate@K & MRR:** Case-insensitive **string match** checking whether `expected_keyword` appears in `doc.page_content` of the top reranked chunks.
  - **Faithfulness & Answer Correctness:** **LLM-as-a-Judge** scoring `0–100` in JSON (`faithfulness` vs. retrieved context; `correctness` vs. `expected_answer`).
  - **Off-Topic Rejection:** Passes if top Cross-Encoder relevance score is below threshold, or if the LLM outputs the `NOT_FOUND` sentinel.

| Metric | Score | Notes |
| :--- | :---: | :--- |
| **Retrieval Hit Rate@K** | **100.0%** | 5/5 in-domain queries matched `expected_keyword` at **Rank #1** |
| **Mean Reciprocal Rank (MRR)** | **1.00** | Top-ranked chunk hit on every in-domain case |
| **Faithfulness** | **100.0%** | Judge found zero unsupported claims across all 5 in-domain answers |
| **Off-Topic Rejection** | **100.0%** | 2/2 off-topic queries scored `0.0%` relevance and triggered fallback |
| **Answer Correctness** | **66.6%** | Lower because `expected_answer` bundles extra section facts while the generator prompt asks to be concise |

> *Note: This 7-case suite is a lightweight smoke test, not a full benchmark; a larger multi-document evaluation set is on the roadmap.*

---

## Project Structure

- `app.py` — Streamlit UI, sidebar controls, session initialization, query routing, and views
- `ingest.py` — Multi-format parsing (`.pdf`, `.docx`, `.txt`, `.csv`), chunking, deduplication, and demo doc
- `retrieval.py` — Normalized `FAISS` + regex `BM25`, weighted RRF fusion (`c=60`), sigmoid Cross-Encoder reranking, and citation checks
- `storage.py` — Validated session-namespaced atomic persistence (`.veridoc_storage/<sid>/`), integrity checks, and 7-day purge
- `eval_suite.py` — 7-case retrieval, LLM-as-a-Judge faithfulness/correctness (with `judge_fallbacks` tracking), and off-topic rejection smoke test
- `requirements.txt` — Python package dependencies
- `.env.example` — Environment variable template (`GROQ_API_KEY`)

---

## Run It Yourself

Tested locally on **Python 3.13.9** (and **Python 3.14.10** on Streamlit Community Cloud).

```bash
git clone https://github.com/ahemadraza31/veridoc-hybrid-rag.git
cd veridoc-hybrid-rag
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add your Groq API key (optionally set `VERIDOC_ENABLE_EVAL=1` to show the in-app benchmark button), then start the app:

```bash
streamlit run app.py
```

---

## Limitations

- **Session links are not private:** anyone with the `?sid=` URL parameter can view that session's stored index and chat history.
- **7-day session retention:** persisted session folders under `.veridoc_storage/` are automatically deleted after 7 days.
- **No OCR for scanned PDFs:** `PyPDFLoader` only extracts embedded text; image-only PDFs return a warning and are skipped.
- **Unsupported formats:** `.xlsx`/`.xls` spreadsheets and legacy `.doc` files are not accepted by the uploader.
- **Grounded answers are non-streaming:** General AI fallback replies stream via `st.write_stream`, whereas grounded RAG answers currently wait for a full `.invoke()` call so `NOT_FOUND` and citations can be checked first.
- **Local disk persistence & API bounds:** `.veridoc_storage/<sid>/` assumes trusted local disk (`allow_dangerous_deserialization=True`) and resets when free-tier cloud containers reboot; queries are also subject to Groq rate limits and send only the top `context_k` chunks to the LLM.

---

## Roadmap

- [ ] Expand `eval_suite.py` into a larger multi-document, multi-hop evaluation dataset
- [ ] Add OCR / vision-model ingestion for scanned PDFs and complex tables
- [ ] Add `.xlsx` / `.xls` spreadsheet ingestion support
- [ ] Stream grounded RAG answers while preserving post-response citation validation
- [ ] Add optional managed vector database backend (e.g., Qdrant / pgvector) for multi-user cloud persistence

---

## License

MIT License — see [LICENSE](LICENSE).
