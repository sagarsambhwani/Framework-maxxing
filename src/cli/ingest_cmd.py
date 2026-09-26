"""Enterprise Ingestion CLI Command Handler."""

import os
import uuid
import asyncio
import importlib
from src.common.logging import print_banner

def run_ingest(file_path=None):
    """Runs enterprise document ingestion on a file or launches the benchmark demo."""
    if file_path and os.path.exists(file_path):
        from src.rag.pipeline.worker import ingestion_worker
        from src.rag.ingestion.job_tracker import job_tracker
        with open(file_path, "rb") as f:
            content = f.read()
        filename = os.path.basename(file_path)
        job_id = f"cli-job-{uuid.uuid4().hex[:6]}"
        job_tracker.create_job(job_id, total_files=1)
        print_banner("ENTERPRISE DOCUMENT INGESTION", f"Ingesting '{filename}' | Job: {job_id}")
        asyncio.run(ingestion_worker.process_job(job_id, [{
            "doc_id": f"doc-{filename}",
            "filename": filename,
            "content": content,
            "access_roles": ["PUBLIC"]
        }]))
        job = job_tracker.get_job(job_id)
        print(f"✓ Ingestion complete! Chunks indexed: {job['total_chunks_indexed']}")
    else:
        demo_mod = importlib.import_module("examples.09_enterprise_ingestion_demo")
        asyncio.run(demo_mod.run_demo())
