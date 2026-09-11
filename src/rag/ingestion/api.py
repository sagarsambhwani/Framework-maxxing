"""FastAPI Router for Enterprise Ingestion, Tri-Brid Search & SSE Progress Streaming.

Exposes endpoints:
    - POST /api/ingest: Background document ingestion with metadata.
    - GET  /api/ingest/status/{job_id}: Query ingestion job status.
    - GET  /api/ingest/events/{job_id}: Server-Sent Events (SSE) live progress stream.
    - POST /api/search: Tri-Brid RAG retrieval with RBAC & temporal filtering.
"""

import uuid
import json
import asyncio
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from src.rag.ingestion.job_tracker import job_tracker
from src.rag.storage.hybrid_store import hybrid_store
from src.rag.storage.audit import audit_store
from src.common.logging import term_log, Colors

router = APIRouter(prefix="/api", tags=["Enterprise RAG & Ingestion"])

class SearchRequest(BaseModel):
    query: str = Field(..., description="User query to search")
    top_k: int = Field(default=5, description="Number of results to return")
    user_roles: Optional[List[str]] = Field(default=None, description="User RBAC roles (e.g. ['LEGAL_AUDITOR'])")
    doc_type: Optional[str] = Field(default=None, description="Filter by doc type (e.g. 'LEGAL_REGULATION')")
    active_only: bool = Field(default=True, description="Filter only currently active documents")

class IngestJsonRequest(BaseModel):
    documents: List[Dict[str, Any]] = Field(..., description="List of documents to ingest")

@router.post("/ingest")
async def ingest_documents_endpoint(payload: IngestJsonRequest):
    """Enqueues document batch for asynchronous background ingestion."""
    if not payload.documents:
        raise HTTPException(status_code=400, detail="No documents provided")

    job_id = f"job-{uuid.uuid4().hex[:8]}"
    total_files = len(payload.documents)

    job_tracker.create_job(job_id=job_id, total_files=total_files)
    term_log("🚀 [API:INGEST]", f"Queued background job '{job_id}' with {total_files} doc(s)", Colors.BLUE)

    # Spawn worker asynchronously in background
    from src.rag.pipeline.worker import ingestion_worker
    asyncio.create_task(ingestion_worker.process_job(job_id, payload.documents))

    return {
        "job_id": job_id,
        "status": "QUEUED",
        "total_files": total_files,
        "events_url": f"/api/ingest/events/{job_id}",
        "status_url": f"/api/ingest/status/{job_id}"
    }

@router.post("/ingest/upload")
async def ingest_files_endpoint(
    files: List[UploadFile] = File(...),
    access_roles: Optional[str] = Form(None),
    valid_from: Optional[str] = Form(None),
    valid_to: Optional[str] = Form(None)
):
    """Uploads physical files (txt, md, json, csv) for ingestion."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    parsed_roles = [r.strip() for r in access_roles.split(",")] if access_roles else ["PUBLIC"]
    documents = []

    for file in files:
        content_bytes = await file.read()
        documents.append({
            "doc_id": f"doc-{file.filename}",
            "filename": file.filename,
            "content": content_bytes,
            "access_roles": parsed_roles,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "is_active": True
        })

    job_id = f"job-{uuid.uuid4().hex[:8]}"
    job_tracker.create_job(job_id=job_id, total_files=len(documents))
    term_log("🚀 [API:UPLOAD]", f"Queued file upload job '{job_id}' with {len(documents)} file(s)", Colors.BLUE)

    from src.rag.pipeline.worker import ingestion_worker
    asyncio.create_task(ingestion_worker.process_job(job_id, documents))

    return {
        "job_id": job_id,
        "status": "QUEUED",
        "total_files": len(documents),
        "events_url": f"/api/ingest/events/{job_id}",
        "status_url": f"/api/ingest/status/{job_id}"
    }

@router.get("/ingest/status/{job_id}")
async def get_job_status(job_id: str):
    """Returns current status of an ingestion job."""
    job = job_tracker.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job

@router.get("/ingest/events/{job_id}")
async def stream_job_events(job_id: str):
    """Streams live Server-Sent Events (SSE) for ingestion progress."""
    job = job_tracker.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    async def event_generator():
        async for event in job_tracker.stream_events(job_id):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/search")
async def tri_brid_search(request: SearchRequest):
    """Executes Tri-Brid hybrid search fusing Dense + BM25 + RBAC/Temporal filters."""
    results = hybrid_store.search(
        query=request.query,
        top_k=request.top_k,
        user_roles=request.user_roles,
        doc_type=request.doc_type,
        active_only=request.active_only
    )
    return {
        "query": request.query,
        "total_results": len(results),
        "results": results
    }

@router.get("/audit/provenance/{chunk_id}")
async def get_chunk_provenance(chunk_id: str):
    """Returns compliance and provenance lineage for a specific chunk."""
    record = audit_store.get_provenance(chunk_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Provenance for chunk '{chunk_id}' not found")
    return record
