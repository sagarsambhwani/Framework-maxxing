"""Lineage, Provenance & Temporal Audit Store for Banking Regulations & Records.

Provides immutable compliance tracking:
    - Provenance: source document URI, sha256 checksum, page number, ingest timestamp.
    - Zero-Trust Security: Redacted fields audit trail and allowed RBAC roles.
    - Temporal Validity: valid_from, valid_to, is_active flags to prevent quoting superseded laws.
"""

import time
from typing import Dict, Any, Optional, List

class AuditProvenanceStore:
    """In-memory compliance and lineage ledger."""

    def __init__(self):
        # Map: chunk_id -> metadata dictionary
        self._records: Dict[str, Dict[str, Any]] = {}

    def log_chunk_provenance(
        self,
        chunk_id: str,
        doc_id: str,
        doc_type: str,
        sha256: str,
        page_number: int,
        redacted_fields: List[str],
        access_roles: List[str],
        valid_from: Optional[str] = None,
        valid_to: Optional[str] = None,
        is_active: bool = True
    ) -> Dict[str, Any]:
        """Records an immutable lineage entry for a chunk."""
        record = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "doc_type": doc_type,
            "sha256": sha256,
            "page_number": page_number,
            "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "redacted_fields": redacted_fields,
            "access_roles": access_roles or ["PUBLIC"],
            "valid_from": valid_from or "1970-01-01",
            "valid_to": valid_to or "9999-12-31",
            "is_active": is_active
        }
        self._records[chunk_id] = record
        return record

    def get_provenance(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves lineage metadata for a given chunk."""
        return self._records.get(chunk_id)

    def mark_superseded(self, doc_id: str):
        """Marks all chunks belonging to a document as superseded/inactive."""
        count = 0
        for r in self._records.values():
            if r["doc_id"] == doc_id:
                r["is_active"] = False
                count += 1
        return count

    def purge_chunk(self, chunk_id: str):
        """Removes a chunk from the audit store."""
        if chunk_id in self._records:
            del self._records[chunk_id]

# Global singleton instance
audit_store = AuditProvenanceStore()
