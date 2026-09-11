"""Job State Tracker & Server-Sent Events (SSE) Progress Emitter.

Tracks lifecycle states of asynchronous background ingestion jobs:
    QUEUED -> TRIAGING -> CHUNKING -> EMBEDDING -> COMPLETED / FAILED
"""

import time
import asyncio
from typing import Dict, Any, Optional, List, AsyncGenerator

class JobStateTracker:
    """Manages asynchronous job lifecycle states and real-time SSE progress events."""

    def __init__(self):
        # Map: job_id -> job status dictionary
        self._jobs: Dict[str, Dict[str, Any]] = {}
        # Map: job_id -> asyncio.Queue of event strings
        self._event_queues: Dict[str, asyncio.Queue] = {}

    def create_job(self, job_id: str, total_files: int) -> Dict[str, Any]:
        """Initializes a new background ingestion job."""
        job = {
            "job_id": job_id,
            "status": "QUEUED",
            "progress_pct": 0,
            "total_files": total_files,
            "processed_files": 0,
            "total_chunks_indexed": 0,
            "unchanged_files_skipped": 0,
            "quarantined_files": 0,
            "started_at": time.time(),
            "completed_at": None,
            "error": None
        }
        self._jobs[job_id] = job
        self._event_queues[job_id] = asyncio.Queue()
        return job

    def update_status(
        self,
        job_id: str,
        status: str,
        progress_pct: int,
        details: str = "",
        chunks_added: int = 0
    ):
        """Updates job state and broadcasts an SSE progress event."""
        if job_id in self._jobs:
            job = self._jobs[job_id]
            job["status"] = status
            job["progress_pct"] = min(100, max(0, progress_pct))
            job["total_chunks_indexed"] += chunks_added

            event_payload = {
                "job_id": job_id,
                "status": status,
                "progress": job["progress_pct"],
                "message": details,
                "chunks_indexed": job["total_chunks_indexed"]
            }

            # Push to SSE event queue if active
            if job_id in self._event_queues:
                self._event_queues[job_id].put_nowait(event_payload)

    def complete_job(self, job_id: str, success: bool = True, error: Optional[str] = None):
        """Finalizes an ingestion job."""
        if job_id in self._jobs:
            job = self._jobs[job_id]
            job["status"] = "COMPLETED" if success else "FAILED"
            job["progress_pct"] = 100 if success else job["progress_pct"]
            job["completed_at"] = time.time()
            job["error"] = error
            self.update_status(
                job_id,
                job["status"],
                job["progress_pct"],
                details="Ingestion finished successfully" if success else f"Error: {error}"
            )

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Returns current job snapshot."""
        return self._jobs.get(job_id)

    async def stream_events(self, job_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Yields SSE events asynchronously until job completion."""
        if job_id not in self._event_queues:
            return

        q = self._event_queues[job_id]
        while True:
            event = await q.get()
            yield event
            if event["status"] in ["COMPLETED", "FAILED"]:
                break

# Singleton instance
job_tracker = JobStateTracker()
