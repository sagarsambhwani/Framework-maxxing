"""Dead-Letter Queue (DLQ) & Quarantine Service.

Catches corrupt, unparseable, or password-protected files during ingestion,
ensuring a 1% file error rate never halts or crashes the background worker cluster.
"""

import time
from typing import Dict, Any, List, Optional
from src.common.logging import term_log, Colors

class DeadLetterQueue:
    """In-memory Dead-Letter Queue for failed document ingestion jobs."""

    def __init__(self):
        # Map: doc_id -> failure details
        self._quarantine: Dict[str, Dict[str, Any]] = {}

    def push_failure(
        self,
        doc_id: str,
        error_type: str,
        error_message: str = "",
        error_msg: str = "",
        payload_snippet: str = "",
        job_id: str = "unknown"
    ) -> Dict[str, Any]:
        """Quarantines a failed document and records diagnostics."""
        msg = error_message or error_msg
        entry = {
            "quarantine_id": f"dlq-{doc_id}",
            "doc_id": doc_id,
            "job_id": job_id,
            "error_type": error_type,
            "error_message": msg,
            "status": "QUARANTINED",
            "payload_snippet": payload_snippet[:200],
            "quarantined_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "retry_count": self._quarantine.get(doc_id, {}).get("retry_count", 0) + 1
        }
        self._quarantine[doc_id] = entry
        term_log("🚨 [DLQ:QUARANTINE]", f"Document '{doc_id}' moved to DLQ: {error_type} - {msg}", Colors.RED)
        return entry

    def get_quarantined_items(self) -> List[Dict[str, Any]]:
        """Returns all currently quarantined documents."""
        return list(self._quarantine.values())

    def get_all(self) -> List[Dict[str, Any]]:
        """Alias for get_quarantined_items."""
        return self.get_quarantined_items()

    def retry_item(self, quarantine_id: str) -> Optional[Dict[str, Any]]:
        """Marks a quarantined document for retry and returns its details."""
        for doc_id, item in self._quarantine.items():
            if item.get("quarantine_id") == quarantine_id or doc_id == quarantine_id:
                item["status"] = "RETRIED"
                return item
        return None

    def clear(self):
        """Clears the DLQ."""
        self._quarantine.clear()

    def size(self) -> int:
        """Returns count of quarantined files."""
        return len(self._quarantine)

# Singleton instance
dlq = DeadLetterQueue()
