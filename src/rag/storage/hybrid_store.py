"""Tri-Brid Enterprise Storage Engine (Dense Vector + BM25 Lexical + Temporal RBAC Metadata).

Combines dense semantic vector retrieval (FastEmbed 384-dim) with sparse lexical search (BM25Okapi)
using Reciprocal Rank Fusion (RRF) and security pre-filtering (RBAC & Temporal validity).
"""

import threading
import numpy as np
from typing import List, Dict, Any, Optional, Set
from rank_bm25 import BM25Okapi

from src.gateway.cache.embeddings import embedding_engine
from src.rag.storage.audit import audit_store
from src.common.logging import debug_log

class TriBridHybridStore:
    """Enterprise Tri-Brid Hybrid Search Store with RRF & Pre-Filtering."""

    def __init__(self):
        self._lock = threading.Lock()
        # chunk_id -> {"chunk_id": str, "text": str, "parent_text": str, "embedding": np.ndarray, "metadata": dict}
        self._chunks: Dict[str, Dict[str, Any]] = {}
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_chunk_ids: List[str] = []
        self._is_bm25_dirty = True

    def _rebuild_bm25_index(self):
        """Rebuilds the BM25 lexical index when chunks are added or purged."""
        if not self._is_bm25_dirty:
            return

        corpus = []
        self._bm25_chunk_ids = []
        for cid, item in self._chunks.items():
            tokens = item["text"].lower().split()
            if tokens:
                corpus.append(tokens)
                self._bm25_chunk_ids.append(cid)

        if corpus:
            self._bm25 = BM25Okapi(corpus)
        else:
            self._bm25 = None
            self._bm25_chunk_ids = []

        self._is_bm25_dirty = False

    def upsert_chunks(self, chunks: List[Dict[str, Any]]):
        """Bulk upserts pre-embedded chunks into dense & sparse indices."""
        with self._lock:
            for c in chunks:
                cid = c["chunk_id"]
                emb = c.get("embedding")
                if emb is None or len(emb) == 0:
                    emb = embedding_engine.embed_query(c["text"])
                else:
                    emb = np.array(emb, dtype=np.float32)

                self._chunks[cid] = {
                    "chunk_id": cid,
                    "doc_id": c.get("doc_id", "unknown"),
                    "text": c["text"],
                    "parent_text": c.get("parent_text", c["text"]),
                    "embedding": emb,
                    "access_roles": c.get("access_roles", ["PUBLIC"]),
                    "is_active": c.get("is_active", True),
                    "doc_type": c.get("doc_type", c.get("metadata", {}).get("doc_type", "GENERAL_DOC")),
                    "metadata": c.get("metadata", {})
                }
            self._is_bm25_dirty = True

    def purge_chunks(self, chunk_ids: List[str]):
        """Purges specific chunks (e.g. when updating a modified file)."""
        with self._lock:
            for cid in chunk_ids:
                if cid in self._chunks:
                    del self._chunks[cid]
                    audit_store.purge_chunk(cid)
            self._is_bm25_dirty = True

    def search(
        self,
        query: str,
        top_k: int = 5,
        user_roles: Optional[List[str]] = None,
        doc_type: Optional[str] = None,
        active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Executes Tri-Brid Search combining Dense Vectors & BM25 with Reciprocal Rank Fusion.

        Args:
            query: User search query.
            top_k: Number of ranked chunks to return.
            user_roles: Optional list of user RBAC roles for pre-filtering.
            doc_type: Optional document type filter ('LEGAL_REGULATION', 'INVOICE_RECEIPT', etc.).
            active_only: If True, filters out superseded regulatory chunks.

        Returns:
            List of ranked result dictionaries with RRF scores, child texts, and parent contexts.
        """
        with self._lock:
            if not self._chunks:
                return []

            self._rebuild_bm25_index()

            # ---------------------------------------------------------
            # 1. HARD PRE-FILTERING (RBAC, Temporal Validity & Doc Type)
            # ---------------------------------------------------------
            allowed_cids: Set[str] = set()
            user_role_set = set(user_roles or ["PUBLIC"])

            for cid, chunk in self._chunks.items():
                prov = audit_store.get_provenance(cid)
                
                # Check active status
                chunk_is_active = chunk.get("is_active", True)
                if prov and "is_active" in prov:
                    chunk_is_active = prov["is_active"]
                if active_only and not chunk_is_active:
                    continue

                # Check doc_type
                chunk_doc_type = chunk.get("doc_type")
                if prov and "doc_type" in prov:
                    chunk_doc_type = prov["doc_type"]
                if doc_type and chunk_doc_type != doc_type:
                    continue

                # Check RBAC access roles
                chunk_roles = chunk.get("access_roles", ["PUBLIC"])
                if prov and "access_roles" in prov:
                    chunk_roles = prov["access_roles"]
                chunk_roles_set = set(chunk_roles)

                # Public chunks are accessible; otherwise must match user role
                if "PUBLIC" not in chunk_roles_set and not (chunk_roles_set & user_role_set):
                    continue

                allowed_cids.add(cid)

            if not allowed_cids:
                return []

            # ---------------------------------------------------------
            # 2. DENSE VECTOR RETRIEVAL
            # ---------------------------------------------------------
            filtered_items = [c for cid, c in self._chunks.items() if cid in allowed_cids]
            query_vec = embedding_engine.embed_query(query)

            dense_matrix = np.array([c["embedding"] for c in filtered_items], dtype=np.float32)
            dense_sims = np.dot(dense_matrix, query_vec)
            dense_ranked_indices = np.argsort(dense_sims)[::-1]

            dense_ranks: Dict[str, int] = {}
            for rank, idx in enumerate(dense_ranked_indices):
                dense_ranks[filtered_items[idx]["chunk_id"]] = rank + 1

            # ---------------------------------------------------------
            # 3. SPARSE BM25 RETRIEVAL (Exact keyword/clause matching)
            # ---------------------------------------------------------
            bm25_ranks: Dict[str, int] = {}
            if self._bm25 is not None and self._bm25_chunk_ids:
                query_tokens = query.lower().split()
                bm25_scores = self._bm25.get_scores(query_tokens)
                bm25_ranked_indices = np.argsort(bm25_scores)[::-1]

                current_rank = 1
                for idx in bm25_ranked_indices:
                    cid = self._bm25_chunk_ids[idx]
                    if cid in allowed_cids and bm25_scores[idx] > 0:
                        bm25_ranks[cid] = current_rank
                        current_rank += 1

            # ---------------------------------------------------------
            # 4. RECIPROCAL RANK FUSION (RRF: k=60)
            # ---------------------------------------------------------
            rrf_k = 60
            fused_scores: Dict[str, float] = {}

            for cid in allowed_cids:
                d_rank = dense_ranks.get(cid, 1000)
                b_rank = bm25_ranks.get(cid, 1000)
                
                score = (1.0 / (rrf_k + d_rank)) + (1.0 / (rrf_k + b_rank))
                fused_scores[cid] = score

            # Sort by fused RRF score
            sorted_cids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)[:top_k]

            results = []
            for cid in sorted_cids:
                chunk = self._chunks[cid]
                prov = audit_store.get_provenance(cid)
                results.append({
                    "chunk_id": cid,
                    "doc_id": chunk["doc_id"],
                    "text": chunk["text"],
                    "parent_context": chunk["parent_text"],
                    "rrf_score": round(fused_scores[cid], 5),
                    "dense_rank": dense_ranks.get(cid, None),
                    "bm25_rank": bm25_ranks.get(cid, None),
                    "provenance": prov,
                    "metadata": chunk["metadata"]
                })

            return results

    def size(self) -> int:
        """Returns total active chunks."""
        with self._lock:
            return len(self._chunks)

# Global singleton instance
hybrid_store = TriBridHybridStore()
