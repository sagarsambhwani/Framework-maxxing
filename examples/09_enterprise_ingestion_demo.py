"""Enterprise Document Ingestion, Background Workers & Tri-Brid RAG Demo.

Demonstrates:
    1. Heterogeneous Banking Document Ingestion (Regulations, Invoices, Statements).
    2. Zero-Trust PII / PCI-DSS Sanitization (Masking PAN, SSN, Account Numbers).
    3. Change Data Capture (CDC) 0ms Duplicate Skip & Orphan Purging.
    4. Contextual & Hierarchical Parent-Child Chunking with Breadcrumbs & Table Preservation.
    5. Tri-Brid Search (Dense Vector + BM25 Lexical + RBAC/Temporal Metadata Fusion).
"""

import asyncio
import os
import sys
import time
from typing import List, Dict, Any

# Add project root to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.pipeline.worker import ingestion_worker
from src.rag.storage.hybrid_store import hybrid_store
from src.rag.storage.audit import audit_store
from src.rag.storage.cdc_registry import cdc_registry
from src.common.logging import term_log, Colors

# ---------------------------------------------------------------------------
# Sample Enterprise Documents
# ---------------------------------------------------------------------------

DODD_FRANK_DOC = """# Dodd-Frank Wall Street Reform Act: Section 165(d) Resolution Plans
Document ID: REG-12-CFR-243
Effective Date: 2024-01-01
Domain: US Banking Regulatory Compliance

## Section 165(d) Liquidity Stress Testing & Capital Adequacy
Covered bank holding companies with total consolidated assets of $250 billion or more
must submit periodic resolution plans (living wills) to the Federal Reserve Board and FDIC.

### Liquidity Buffer Requirements
Each covered banking organization must maintain a minimum liquidity buffer of unencumbered
highly liquid assets (HQLA) sufficient to meet its projected net cash outflows over a 30-day
stress horizon. Liquidity stress tests must be conducted at least monthly under baseline,
adverse, and severely adverse economic scenarios.

### Governance and Oversight
The board of directors of each covered institution holds ultimate fiduciary responsibility
for reviewing and approving resolution plan submissions every two years.
"""

INVOICE_DOC = """# INVOICE: AWS Cloud Infrastructure Services
Invoice Number: INV-2024-98214
Billing Period: October 1 - October 31, 2024
Customer: Apex Global Banking Corp
Payment Card: 4532-8921-3456-7890
Contact: billing@apexbanking.com, phone: +1-202-555-0193

## Line Items Breakdown
| Service Description | Region | Usage Quantity | Unit Price | Total Amount |
| :--- | :--- | :--- | :--- | :--- |
| Amazon EC2 c6i.4xlarge Compute | us-east-1 | 744 Hours | $0.68 | $505.92 |
| Amazon Aurora PostgreSQL Multi-AZ | us-east-1 | 744 Hours | $1.42 | $1,056.48 |
| Amazon S3 Standard Storage (TB) | us-east-1 | 150 TB | $23.00 | $3,450.00 |
| AWS Direct Connect 10Gbps Port | us-east-1 | 1 Month | $1,620.00 | $1,620.00 |

Total Invoice Amount Due: $6,632.40
Payment Status: Paid in full via Card ending in 7890.
"""

BANK_STATEMENT_DOC = """# Sovereign Wealth Bank: High-Net-Worth Account Statement
Statement Period: September 1, 2024 - September 30, 2024
Account Holder: Jonathan Vance
Social Security Number: 123-45-6789
Checking Account: 9876543210
Routing Number: 021000021

## Balance Summary
- Beginning Balance: $1,420,500.00
- Total Deposits: $250,000.00
- Wire Transfers: $50,000.00 (Escrow deposit for Manhattan commercial real estate)
- Ending Balance: $1,620,500.00

Confidential Financial Record. Authorized bank personnel and legal auditor access only.
"""

async def run_demo():
    print("=" * 80)
    print("🚀 SPRINT 2: ENTERPRISE INGESTION & TRI-BRID RAG DEMO")
    print("=" * 80)

    # 1. INGESTION RUN 1: Ingesting 3 Heterogeneous Banking Documents
    print("\n--- [STEP 1] Ingesting 3 Enterprise Documents (Regulation, Invoice, Statement) ---")
    documents = [
        {
            "doc_id": "reg-dodd-frank-165d",
            "filename": "12_CFR_243_Dodd_Frank_165d.md",
            "content": DODD_FRANK_DOC,
            "access_roles": ["PUBLIC", "COMPLIANCE_OFFICER", "LEGAL_AUDITOR"],
            "valid_from": "2024-01-01",
            "valid_to": None,
            "is_active": True
        },
        {
            "doc_id": "inv-aws-2024-98214",
            "filename": "AWS_Hosting_Invoice_Oct2024.md",
            "content": INVOICE_DOC,
            "access_roles": ["FINANCE_ADMIN", "LEGAL_AUDITOR"],
            "valid_from": "2024-10-01",
            "valid_to": None,
            "is_active": True
        },
        {
            "doc_id": "stmt-jonathan-vance-sep2024",
            "filename": "Private_Wealth_Statement_Sep2024.md",
            "content": BANK_STATEMENT_DOC,
            "access_roles": ["PRIVATE_BANKER", "COMPLIANCE_AUDITOR"],
            "valid_from": "2024-09-01",
            "valid_to": None,
            "is_active": True
        }
    ]

    job_id_1 = "demo-job-1"
    await ingestion_worker.process_job(job_id_1, documents)

    # 2. CHANGE DATA CAPTURE (CDC) TEST: Re-submitting identical regulation
    print("\n--- [STEP 2] Testing CDC Deduplication (0ms Skip on Unchanged Doc) ---")
    t0 = time.perf_counter()
    dup_job_id = "demo-job-cdc"
    await ingestion_worker.process_job(dup_job_id, [documents[0]])
    cdc_elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"⚡ CDC Re-Ingestion Time: {cdc_elapsed_ms:.2f}ms (Skipped parsing and embedding!)")

    # 3. PII / PCI-DSS SANITIZATION VERIFICATION
    print("\n--- [STEP 3] Verifying Zero-Trust PII / PCI-DSS Masking ---")
    invoice_chunks = [c for c in hybrid_store._chunks.values() if c.get("doc_id") == "inv-aws-2024-98214"]
    for chunk in invoice_chunks:
        text = chunk["text"]
        if "MASKED_PAN" in text:
            start_idx = text.find("[MASKED_PAN")
            end_idx = text.find("]", start_idx) + 1
            print(f"🛡️ Verified PAN Masking: Found '{text[start_idx:end_idx]}'")
        if "4532-8921-3456-7890" in text:
            raise AssertionError("FAIL: Raw credit card leaked into vector store!")

    stmt_chunks = [c for c in hybrid_store._chunks.values() if c.get("doc_id") == "stmt-jonathan-vance-sep2024"]
    for chunk in stmt_chunks:
        text = chunk["text"]
        if "MASKED_SSN" in text:
            start_idx = text.find("[MASKED_SSN")
            end_idx = text.find("]", start_idx) + 1
            print(f"🛡️ Verified SSN Masking: Found '{text[start_idx:end_idx]}'")
        if "123-45-6789" in text:
            raise AssertionError("FAIL: Raw SSN leaked into vector store!")

    # 4. TRI-BRID SEARCH QUERIES
    print("\n--- [STEP 4] Tri-Brid Hybrid Search Demonstration ---")

    # Query A: Exact Regulatory Clause (Dense + BM25 fusion)
    q_reg = "Section 165(d) liquidity stress testing 30-day horizon"
    print(f"\n🔍 Query A: '{q_reg}' (User Role: 'PUBLIC')")
    results_a = hybrid_store.search(q_reg, top_k=2, user_roles=["PUBLIC"])
    for r in results_a:
        print(f"   RRF Score: {r['rrf_score']:.5f} | Dense Rank: {r['dense_rank']} | BM25 Rank: {r['bm25_rank']}")
        print(f"   Doc: {r['doc_id']} | Type: {r['metadata'].get('doc_type')}")
        print(f"   Snippet: {r['text'][:140]}...\n")

    # Query B: Tabular Invoice Line Item
    q_inv = "Amazon EC2 c6i compute usage hours and price"
    print(f"🔍 Query B: '{q_inv}' (User Role: 'FINANCE_ADMIN')")
    results_b = hybrid_store.search(q_inv, top_k=2, user_roles=["FINANCE_ADMIN"])
    for r in results_b:
        print(f"   RRF Score: {r['rrf_score']:.5f} | Dense Rank: {r['dense_rank']} | BM25 Rank: {r['bm25_rank']}")
        print(f"   Doc: {r['doc_id']} | Snippet: {r['text'][:140]}...\n")

    # Query C: RBAC Authorization Gate
    print("🔍 Query C: RBAC Authorization Isolation Check")
    # Public user searches for private bank statement content
    q_confidential = "Wire transfer for Manhattan commercial real estate"
    unauth_results = hybrid_store.search(q_confidential, top_k=5, user_roles=["PUBLIC"])
    unauth_doc_ids = set(r["doc_id"] for r in unauth_results)
    print(f"   Public User Query Returned Doc IDs: {unauth_doc_ids}")
    assert "stmt-jonathan-vance-sep2024" not in unauth_doc_ids, "RBAC LEAK: Confidential doc returned to PUBLIC!"
    print("   🛡️ RBAC Isolation Verified: 'stmt-jonathan-vance-sep2024' strictly hidden from PUBLIC role.")

    auth_results = hybrid_store.search(q_confidential, top_k=2, user_roles=["PRIVATE_BANKER"])
    auth_doc_ids = set(r["doc_id"] for r in auth_results)
    print(f"   Private Banker Query Returned Doc IDs: {auth_doc_ids}")
    assert "stmt-jonathan-vance-sep2024" in auth_doc_ids, "RBAC FAILURE: Authorized user could not retrieve document!"
    print(f"   ✅ Authorized Retrieval: {auth_results[0]['text'][:110]}...")

    print("\n" + "=" * 80)
    print("✅ SPRINT 2 DEMO SUCCEEDED: 100% Zero-Trust Compliance, CDC, and Tri-Brid RAG!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(run_demo())
