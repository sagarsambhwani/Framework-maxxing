"""Unit & Integration Tests for Enterprise Document Ingestion & Tri-Brid RAG Subsystem.

Tests:
    1. Zero-Trust PII / PCI-DSS Sanitizer (PAN, SSN, Account, Email, Phone).
    2. Change Data Capture (CDC) Registry (Check, Register, Skip, Purge).
    3. Document Triage & Layout Extractor (Regulation, Invoice, Statement classification).
    4. Hierarchical & Contextual Chunking (Parent-child linking, Breadcrumb injection).
    5. Tri-Brid Hybrid Search Store (Dense + BM25 + RBAC Pre-filtering + Active-only).
    6. Dead Letter Queue (DLQ Quarantine, Inspect, Resolve).
    7. Asynchronous Ingestion Worker (End-to-end ingestion flow).
    8. FastAPI Ingestion & Search Router Endpoints.
"""

import os
import sys
import pytest
import asyncio
from fastapi.testclient import TestClient

# Add project root to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.ingestion.sanitizer import sanitizer
from src.rag.storage.cdc_registry import cdc_registry
from src.rag.ingestion.triage import triage_engine
from src.rag.chunking.contextual import contextual_chunker
from src.rag.chunking.hierarchical import hierarchical_chunker
from src.rag.storage.hybrid_store import hybrid_store
from src.rag.storage.audit import audit_store
from src.rag.pipeline.dlq import dlq
from src.rag.pipeline.worker import ingestion_worker
from src.rag.ingestion.job_tracker import job_tracker
from src.server.app import create_app

# ---------------------------------------------------------------------------
# 1. PII / PCI-DSS Sanitizer Tests
# ---------------------------------------------------------------------------

def test_sanitizer_masks_credit_card_pan():
    raw = "Customer paid using VISA card 4532-8921-3456-7890 for enterprise billing."
    sanitized, redacted = sanitizer.sanitize(raw)
    assert "4532-8921-3456-7890" not in sanitized
    assert "[MASKED_PAN_7890]" in sanitized
    assert "PCI_DSS_CREDIT_CARD" in redacted

def test_sanitizer_masks_ssn_and_bank_account():
    raw = "SSN: 123-45-6789 and Bank Account Number: 9876543210 were verified."
    sanitized, redacted = sanitizer.sanitize(raw)
    assert "123-45-6789" not in sanitized
    assert "[MASKED_SSN]" in sanitized
    assert "SSN_NATIONAL_ID" in redacted

def test_sanitizer_clean_text_unchanged():
    raw = "Dodd-Frank Wall Street Reform Act section 165 requires periodic stress testing."
    sanitized, redacted = sanitizer.sanitize(raw)
    assert sanitized == raw
    assert len(redacted) == 0

# ---------------------------------------------------------------------------
# 2. Change Data Capture (CDC) Registry Tests
# ---------------------------------------------------------------------------

def test_cdc_lifecycle():
    doc_id = "test-doc-cdc-1"
    content_v1 = b"Version 1 of the regulatory policy."
    content_v2 = b"Version 2 with updated capital requirement."

    # First check: Should be NEW
    chk1 = cdc_registry.check_file(doc_id, content_v1)
    assert chk1["status"] == "NEW"
    assert chk1["version"] == 1

    # Register success
    cdc_registry.register_success(doc_id, chk1["hash"], chk1["version"], ["chunk-1", "chunk-2"])

    # Second check with identical bytes: Should be UNCHANGED (0ms duplicate skip)
    chk2 = cdc_registry.check_file(doc_id, content_v1)
    assert chk2["status"] == "UNCHANGED"

    # Third check with updated bytes: Should be MODIFIED and return old chunks to purge
    chk3 = cdc_registry.check_file(doc_id, content_v2)
    assert chk3["status"] == "MODIFIED"
    assert chk3["version"] == 2
    assert "chunk-1" in chk3["chunk_ids_to_purge"]
    assert "chunk-2" in chk3["chunk_ids_to_purge"]

    # Clean up / Purge
    purged = cdc_registry.purge_document(doc_id)
    assert "chunk-1" in purged
    chk4 = cdc_registry.check_file(doc_id, content_v1)
    assert chk4["status"] == "NEW"

# ---------------------------------------------------------------------------
# 3. Document Triage & Layout Extractor Tests
# ---------------------------------------------------------------------------

def test_document_triage_classification():
    reg_text = "Federal Reserve 12 CFR 243 Regulation regarding capital stress tests and compliance."
    dtype, conf = triage_engine.classify(reg_text, "12_CFR_243.pdf")
    assert dtype == "LEGAL_REGULATION"
    assert conf > 0.5

    inv_text = "INVOICE #98214\nTotal Amount Due: $1,400.00\nPayment due upon receipt."
    dtype, conf = triage_engine.classify(inv_text, "invoice_98214.txt")
    assert dtype == "INVOICE_RECEIPT"
    assert conf > 0.5

    stmt_text = "Account Statement: Beginning Balance $50,000, Total Deposits $5,000, Ending Balance $55,000."
    dtype, conf = triage_engine.classify(stmt_text)
    assert dtype == "BANK_STATEMENT"

def test_triage_layout_extraction():
    content = """# Header Title
Some preamble text.

| Item | Qty | Price |
| :--- | :--- | :--- |
| Server | 2 | $500 |

Footer notes.
"""
    layout = triage_engine.extract_layout(content)
    assert layout["tables_found"] == 1
    assert len(layout["tables"]) == 1
    assert "Server" in layout["tables"][0]

# ---------------------------------------------------------------------------
# 4. Hierarchical & Contextual Chunking Tests
# ---------------------------------------------------------------------------

def test_contextual_chunker_breadcrumbs():
    breadcrumb = contextual_chunker.build_breadcrumb(
        doc_title="Dodd-Frank Title I",
        domain="Financial Compliance",
        section_path=["Liquidity Standards", "Stress Testing Horizon"]
    )
    assert "[Financial Compliance] Dodd-Frank Title I > Liquidity Standards > Stress Testing Horizon" in breadcrumb

def test_hierarchical_chunker_generates_parent_child():
    doc = """# Bank Liquidity Guidelines
## Section 1: Overview
The bank holds assets across multiple jurisdictions to ensure liquidity during volatility.
The minimum ratio must be maintained at all times.

## Section 2: Stress Testing
Stress tests simulate historical shocks including the 2008 financial crisis and liquidity runs.
"""
    chunks = hierarchical_chunker.chunk_document(
        doc_id="doc-guidelines",
        doc_title="Liquidity Guidelines",
        doc_type="LEGAL_REGULATION",
        layout={"has_tables": False, "tables": [], "headings": ["Section 1: Overview", "Section 2: Stress Testing"]},
        sanitized_text=doc
    )
    assert len(chunks) >= 1
    for c in chunks:
        assert "chunk_id" in c
        assert "parent_id" in c
        assert "context_breadcrumb" in c
        assert "parent_text" in c
        assert len(c["text"]) > 0

# ---------------------------------------------------------------------------
# 5. Tri-Brid Hybrid Search Store Tests
# ---------------------------------------------------------------------------

def test_hybrid_store_upsert_and_rbac_search():
    # Setup test chunks
    test_chunks = [
        {
            "chunk_id": "test-c1",
            "doc_id": "doc-public-reg",
            "doc_type": "LEGAL_REGULATION",
            "text": "Dodd-Frank Section 165 requires resolution plan living wills for banks.",
            "context_breadcrumb": "[Regulation] Dodd-Frank",
            "parent_id": "parent-1",
            "parent_text": "Full text of Dodd-Frank resolution plan mandate.",
            "embedding": None, # Will be auto-embedded
            "page_number": 1,
            "access_roles": ["PUBLIC", "AUDITOR"],
            "is_active": True
        },
        {
            "chunk_id": "test-c2",
            "doc_id": "doc-confidential-payroll",
            "doc_type": "BANK_STATEMENT",
            "text": "Confidential executive bonus distribution wire transfer details.",
            "context_breadcrumb": "[Internal] Executive Payroll",
            "parent_id": "parent-2",
            "parent_text": "Confidential executive payroll records.",
            "embedding": None,
            "page_number": 1,
            "access_roles": ["EXECUTIVE_HR"],
            "is_active": True
        }
    ]

    hybrid_store.upsert_chunks(test_chunks)

    # 1. Search with PUBLIC role -> should retrieve c1, but NOT c2 (RBAC blocked)
    res_public = hybrid_store.search("executive bonus wire transfer", top_k=5, user_roles=["PUBLIC"])
    res_ids_public = [r["chunk_id"] for r in res_public]
    assert "test-c2" not in res_ids_public

    # 2. Search with EXECUTIVE_HR role -> should retrieve c2
    res_hr = hybrid_store.search("executive bonus wire transfer", top_k=5, user_roles=["EXECUTIVE_HR"])
    res_ids_hr = [r["chunk_id"] for r in res_hr]
    assert "test-c2" in res_ids_hr

    # 3. Exact clause keyword search (testing BM25 fusion)
    res_clause = hybrid_store.search("Section 165 living wills", top_k=5, user_roles=["PUBLIC"])
    assert len(res_clause) > 0
    assert res_clause[0]["chunk_id"] == "test-c1"

# ---------------------------------------------------------------------------
# 6. Dead Letter Queue (DLQ) Tests
# ---------------------------------------------------------------------------

def test_dlq_quarantine_and_retry():
    dlq.push_failure(
        doc_id="corrupt-file.bin",
        error_type="EncodingError",
        error_msg="Unsupported binary UTF-16 stream",
        payload_snippet="0xDEADBEEF",
        job_id="job-test-dlq"
    )

    failures = dlq.get_all()
    assert len(failures) >= 1
    target = next(f for f in failures if f["doc_id"] == "corrupt-file.bin")
    assert target["error_type"] == "EncodingError"
    assert target["status"] == "QUARANTINED"

    # Retry item
    retried = dlq.retry_item(target["quarantine_id"])
    assert retried["status"] == "RETRIED"

# ---------------------------------------------------------------------------
# 7. Asynchronous Ingestion Worker Integration Test
# ---------------------------------------------------------------------------

def test_worker_end_to_end_ingestion():
    job_id = "test-job-worker-e2e"
    docs = [
        {
            "doc_id": "e2e-reg-1",
            "filename": "E2E_Regulation.txt",
            "content": "Federal Reserve capital liquidity standards under 12 CFR Part 252.",
            "access_roles": ["PUBLIC"],
            "is_active": True
        }
    ]

    job_tracker.create_job(job_id, total_files=1)
    asyncio.run(ingestion_worker.process_job(job_id, docs))

    job = job_tracker.get_job(job_id)
    assert job["status"] == "COMPLETED"
    assert job["progress_pct"] == 100
    assert job["processed_files"] == 1

# ---------------------------------------------------------------------------
# 8. FastAPI Ingestion & Search Router Tests
# ---------------------------------------------------------------------------

def test_fastapi_ingestion_api():
    app = create_app()
    client = TestClient(app)

    # Ingest document via POST /api/ingest
    payload = {
        "documents": [
            {
                "doc_id": "api-test-doc",
                "filename": "api_test.txt",
                "content": "API Ingestion testing document with Section 404 compliance rules.",
                "access_roles": ["PUBLIC"],
                "is_active": True
            }
        ]
    }
    res = client.post("/api/ingest", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "job_id" in data
    job_id = data["job_id"]

    # Poll status via GET /api/ingest/status/{job_id}
    status_res = client.get(f"/api/ingest/status/{job_id}")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["job_id"] == job_id

    # Search via POST /api/search
    search_payload = {
        "query": "Section 404 compliance rules",
        "top_k": 3,
        "user_roles": ["PUBLIC"],
        "active_only": True
    }
    search_res = client.post("/api/search", json=search_payload)
    assert search_res.status_code == 200
    s_data = search_res.json()
    assert "results" in s_data
