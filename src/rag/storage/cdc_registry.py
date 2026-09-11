"""Change Data Capture (CDC) & SHA-256 Checksum Registry.

Provides idempotent file tracking, avoiding redundant embedding generation for unchanged
documents (0ms skip) and maintaining chunk lifecycle tombstones when documents are modified.
"""

import hashlib
import time
from typing import Dict, Any, Optional, Set, List
from src.common.logging import debug_log

class CDCRegistry:
    """In-memory thread-safe Change Data Capture registry."""

    def __init__(self):
        # Map: doc_id -> {"hash": str, "version": int, "chunk_ids": List[str], "timestamp": float}
        self._registry: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def compute_hash(content: bytes) -> str:
        """Computes SHA-256 hex digest of file raw bytes."""
        return hashlib.sha256(content).hexdigest()

    def check_file(self, doc_id: str, content: bytes) -> Dict[str, Any]:
        """Evaluates whether document is NEW, MODIFIED, or UNCHANGED.

        Args:
            doc_id: Unique document identifier (e.g. filename or URI).
            content: Raw byte content.

        Returns:
            Dict containing 'status' ('NEW', 'MODIFIED', 'UNCHANGED'),
            'hash', and previous 'chunk_ids_to_purge' if modified.
        """
        new_hash = self.compute_hash(content)
        
        if doc_id not in self._registry:
            return {
                "status": "NEW",
                "doc_id": doc_id,
                "hash": new_hash,
                "version": 1,
                "chunk_ids_to_purge": []
            }

        prev = self._registry[doc_id]
        if prev["hash"] == new_hash:
            debug_log("⚡ [CDC:SKIP]", f"Document '{doc_id}' unchanged (SHA-256: {new_hash[:10]}...) -> 0ms skip")
            return {
                "status": "UNCHANGED",
                "doc_id": doc_id,
                "hash": new_hash,
                "version": prev["version"],
                "chunk_ids_to_purge": []
            }

        # Document changed -> increment version and signal old chunks for eviction
        new_version = prev["version"] + 1
        debug_log("🔄 [CDC:MODIFIED]", f"Document '{doc_id}' changed to version {new_version}")
        return {
            "status": "MODIFIED",
            "doc_id": doc_id,
            "hash": new_hash,
            "version": new_version,
            "chunk_ids_to_purge": list(prev.get("chunk_ids", []))
        }

    def register_success(self, doc_id: str, doc_hash: str, version: int, chunk_ids: List[str]):
        """Commits successfully indexed document into the CDC registry."""
        self._registry[doc_id] = {
            "hash": doc_hash,
            "version": version,
            "chunk_ids": chunk_ids,
            "timestamp": time.time()
        }

    def purge_document(self, doc_id: str) -> List[str]:
        """Removes document from registry and returns orphan chunk IDs."""
        if doc_id in self._registry:
            chunks = list(self._registry[doc_id].get("chunk_ids", []))
            del self._registry[doc_id]
            return chunks
        return []

    def get_stats(self) -> Dict[str, Any]:
        """Returns registry summary statistics."""
        return {
            "total_tracked_documents": len(self._registry),
            "total_active_chunks": sum(len(v["chunk_ids"]) for v in self._registry.values())
        }

# Global singleton instance
cdc_registry = CDCRegistry()
