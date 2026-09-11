"""Asynchronous Ingestion Worker & Pipeline Orchestrator.

Executes the full enterprise document processing lifecycle:
    CDC Check -> Triage -> PII Sanitization -> Contextual & Hierarchical Chunking ->
    Dynamic Micro-Batching -> Tri-Brid Indexing -> Audit Logging -> SSE Progress Emitter.
"""

import time
import asyncio
from typing import List, Dict, Any

from src.rag.storage.cdc_registry import cdc_registry
from src.rag.storage.audit import audit_store
from src.rag.storage.hybrid_store import hybrid_store
from src.rag.ingestion.triage import triage_engine
from src.rag.ingestion.sanitizer import sanitizer
from src.rag.chunking.hierarchical import hierarchical_chunker
from src.rag.pipeline.batcher import batcher
from src.rag.pipeline.dlq import dlq
from src.rag.ingestion.job_tracker import job_tracker
from src.common.logging import term_log, debug_log, Colors

class AsyncIngestionWorker:
    """Orchestrates end-to-end background ingestion for batches of documents."""

    async def process_job(self, job_id: str, documents: List[Dict[str, Any]]):
        """Processes a list of raw documents asynchronously.

        Args:
            job_id: Correlated job ID.
            documents: List of dicts containing:
                - 'doc_id': str
                - 'filename': str
                - 'content': bytes or str
                - 'access_roles': Optional[List[str]]
                - 'valid_from': Optional[str]
                - 'valid_to': Optional[str]
        """
        total_files = len(documents)
        job_tracker.update_status(job_id, "PROCESSING", 5, f"Starting ingestion for {total_files} document(s)...")

        for idx, doc in enumerate(documents):
            doc_id = doc.get("doc_id", f"doc-{idx}")
            filename = doc.get("filename", doc_id)
            raw_content = doc.get("content", "")

            # Ensure bytes for hash calculation
            if isinstance(raw_content, str):
                raw_bytes = raw_content.encode("utf-8")
                raw_text = raw_content
            else:
                raw_bytes = raw_content
                raw_text = raw_content.decode("utf-8", errors="replace")

            pct = int(((idx + 1) / total_files) * 90)

            try:
                # -------------------------------------------------------------
                # 1. CHANGE DATA CAPTURE (CDC) CHECKSUM GATE
                # -------------------------------------------------------------
                cdc_status = cdc_registry.check_file(doc_id, raw_bytes)
                if cdc_status["status"] == "UNCHANGED":
                    term_log("⚡ [INGESTION:CDC]", f"Skipping unchanged '{filename}' (0ms skip)", Colors.GREEN)
                    if job_id in job_tracker._jobs:
                        job_tracker._jobs[job_id]["unchanged_files_skipped"] += 1
                    job_tracker.update_status(job_id, "PROCESSING", pct, f"Skipped unchanged '{filename}' (CDC 0ms)")
                    continue

                if cdc_status["status"] == "MODIFIED":
                    old_chunks = cdc_status["chunk_ids_to_purge"]
                    hybrid_store.purge_chunks(old_chunks)
                    term_log("🔄 [INGESTION:CDC]", f"Purged {len(old_chunks)} stale chunks for modified '{filename}'", Colors.YELLOW)

                # -------------------------------------------------------------
                # 2. ZERO-TRUST PII / PCI-DSS SANITIZATION GATE
                # -------------------------------------------------------------
                sanitized_text, redacted_fields = sanitizer.sanitize(raw_text)
                if redacted_fields:
                    term_log("🛡️ [INGESTION:PII]", f"Sanitized {len(redacted_fields)} sensitive field(s) in '{filename}': {redacted_fields}", Colors.YELLOW)

                # -------------------------------------------------------------
                # 3. DOCUMENT TRIAGE & CLASSIFICATION
                # -------------------------------------------------------------
                job_tracker.update_status(job_id, "TRIAGING", pct, f"Classifying layout for '{filename}'...")
                doc_type, confidence = triage_engine.classify(sanitized_text, filename=filename)
                layout = triage_engine.extract_layout(sanitized_text)

                # -------------------------------------------------------------
                # 4. CONTEXTUAL & HIERARCHICAL CHUNKING
                # -------------------------------------------------------------
                job_tracker.update_status(job_id, "CHUNKING", pct, f"Generating hierarchical chunks for '{filename}'...")
                chunks = hierarchical_chunker.chunk_document(
                    doc_id=doc_id,
                    doc_title=filename,
                    doc_type=doc_type,
                    layout=layout,
                    sanitized_text=sanitized_text
                )

                # -------------------------------------------------------------
                # 5. DYNAMIC MICRO-BATCH VECTORIZATION
                # -------------------------------------------------------------
                job_tracker.update_status(job_id, "EMBEDDING", pct, f"Vectorizing {len(chunks)} chunks in micro-batches...")
                embedded_chunks = await batcher.process_chunks_in_batches(chunks)

                # -------------------------------------------------------------
                # 6. TRI-BRID INDEXING & AUDIT PROVENANCE LOGGING
                # -------------------------------------------------------------
                hybrid_store.upsert_chunks(embedded_chunks)

                chunk_ids = []
                for c in embedded_chunks:
                    cid = c["chunk_id"]
                    chunk_ids.append(cid)
                    audit_store.log_chunk_provenance(
                        chunk_id=cid,
                        doc_id=doc_id,
                        doc_type=doc_type,
                        sha256=cdc_status["hash"],
                        page_number=c.get("page_number", 1),
                        redacted_fields=redacted_fields,
                        access_roles=doc.get("access_roles", ["PUBLIC"]),
                        valid_from=doc.get("valid_from"),
                        valid_to=doc.get("valid_to"),
                        is_active=doc.get("is_active", True)
                    )

                # Commit CDC success
                cdc_registry.register_success(doc_id, cdc_status["hash"], cdc_status["version"], chunk_ids)

                term_log(
                    "✅ [INGESTION:INDEXED]",
                    f"Indexed '{filename}' ({doc_type}) -> {len(chunk_ids)} chunks (Dense + BM25)",
                    Colors.GREEN
                )

                if job_id in job_tracker._jobs:
                    job_tracker._jobs[job_id]["processed_files"] += 1

                job_tracker.update_status(job_id, "PROCESSING", pct, f"Indexed '{filename}' ({len(chunks)} chunks)", chunks_added=len(chunks))

            except Exception as e:
                dlq.push_failure(doc_id, type(e).__name__, str(e), payload_snippet=raw_text[:200], job_id=job_id)
                if job_id in job_tracker._jobs:
                    job_tracker._jobs[job_id]["quarantined_files"] += 1
                job_tracker.update_status(job_id, "PROCESSING", pct, f"Quarantined '{filename}' into DLQ: {e}")

        # Finalize job
        job_tracker.complete_job(job_id, success=True)
        term_log("🎉 [INGESTION:COMPLETE]", f"Job '{job_id}' successfully finalized", Colors.GREEN)

# Singleton worker instance
ingestion_worker = AsyncIngestionWorker()
