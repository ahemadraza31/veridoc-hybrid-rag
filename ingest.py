import os
import tempfile
from typing import List, Tuple, Set
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader


def process_uploaded_files(
    uploaded_files,
    existing_files: Set[str],
    chunk_size: int = 750,
    chunk_overlap: int = 150,
) -> Tuple[List[Document], List[str], List[str]]:
    """
    Parse uploaded files into chunked Documents, skipping already-indexed files.
    Returns: (new_chunks, newly_indexed_names, warnings_or_errors)
    """
    all_chunks: List[Document] = []
    newly_indexed: List[str] = []
    messages: List[str] = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    for uploaded_file in uploaded_files:
        fname = uploaded_file.name
        if fname in existing_files:
            messages.append(f"Skipped `{fname}` (already indexed).")
            continue

        file_ext = os.path.splitext(fname)[1].lower()
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            if file_ext == ".pdf":
                docs = PyPDFLoader(tmp_path).load()
            elif file_ext == ".docx":
                docs = Docx2txtLoader(tmp_path).load()
            elif file_ext == ".csv":
                docs = CSVLoader(tmp_path).load()
            else:
                docs = TextLoader(tmp_path, encoding="utf-8").load()

            # Check for empty/scanned documents
            non_empty_docs = [d for d in docs if d.page_content and d.page_content.strip()]
            if not non_empty_docs:
                messages.append(
                    f"`{fname}` contained no extractable text (scanned/image PDFs require OCR)."
                )
                continue

            for doc in non_empty_docs:
                # Replace temp path with clean filename
                doc.metadata["source"] = fname
                doc.metadata["source_name"] = fname
                doc.metadata["doc_type"] = file_ext.replace(".", "").upper()
                if "page" in doc.metadata and isinstance(doc.metadata["page"], int):
                    doc.metadata["page_num"] = doc.metadata["page"] + 1

            chunks = text_splitter.split_documents(non_empty_docs)
            for idx, chunk in enumerate(chunks):
                page_suffix = f"-p{chunk.metadata['page_num']}" if "page_num" in chunk.metadata else ""
                chunk.metadata["chunk_id"] = f"{fname}{page_suffix}#chunk-{idx + 1}"

            all_chunks.extend(chunks)
            newly_indexed.append(fname)

        except Exception as e:
            messages.append(f"Failed to parse `{fname}`: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    return all_chunks, newly_indexed, messages


def get_sample_document() -> List[Document]:
    sample_text = """
# ACME Corp Enterprise Architecture & Cloud Migration Standard 2026

## Section 1: Security and OAuth 2.0 Identity Protocol
ACME Corp mandates OAuth 2.0 with PKCE (Proof Key for Code Exchange) for all customer-facing and internal microservices.
Access tokens must have an expiration TTL of 15 minutes, with rotating refresh tokens valid up to 24 hours.
All data at rest must use AES-256 encryption, and transit must enforce TLS 1.3 minimum.

## Section 2: Database Scalability and N+1 Query Resolution
Internal benchmarks revealed that unindexed ORM joins created critical N+1 latency spikes during peak CRUD operations.
Standard requirement: Developers must use explicit JOIN FETCH or batch fetching with maximum batch size of 50.
In Spring Boot & Hibernate architectures, queries exceeding 200ms trigger automated APM alerts.

## Section 3: SLA and High-Availability Guarantee
Core banking transaction endpoints require 99.99% availability (less than 52.6 minutes annual downtime).
Multi-region database replicas are configured in active-passive pairs across Mumbai (Primary) and Pune (Secondary).
Automated failover must initiate within 30 seconds of persistent heartbeat failure.

## Section 4: AI & RAG Deployment Guardrails
Generative AI integrations must adhere to enterprise grounding benchmarks:
- Minimum cross-encoder confidence score of 0.65 (65%) required before displaying context to end users.
- Hallucination checks must cross-reference exact citation IDs.
- Zero data retention policy strictly enforced for third-party inference APIs.
"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=420, chunk_overlap=50)
    docs = [
        Document(
            page_content=sample_text.strip(),
            metadata={
                "source": "ACME_Enterprise_Arch_2026.pdf",
                "source_name": "ACME_Enterprise_Arch_2026.pdf",
                "doc_type": "PDF",
                "page_num": 1,
            },
        )
    ]
    chunks = splitter.split_documents(docs)
    for idx, c in enumerate(chunks):
        c.metadata["chunk_id"] = f"ACME_Arch-p1#chunk-{idx + 1}"
    return chunks
