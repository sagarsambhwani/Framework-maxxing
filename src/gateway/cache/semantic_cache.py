"""In-Memory Semantic Vector Cache Store.

Provides cosine-similarity based vector caching with LRU eviction and TTL expiration.
Queries with semantic similarity >= threshold return cached responses in <2ms at $0 cost.

Usage:
    from src.gateway.cache.semantic_cache import semantic_cache
    match = semantic_cache.lookup("Explain AI gateway")
    if match:
        print("Cached response:", match["response"], "Similarity:", match["similarity"])
    else:
        semantic_cache.store("Explain AI gateway", "An AI Gateway...", model="qwen3.8")
"""

import time
import threading
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from src.gateway.cache.embeddings import embedding_engine
from src.common.logging import term_log, debug_log, Colors

class SemanticVectorCache:
    """Thread-safe in-memory semantic vector cache."""

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        default_ttl_seconds: int = 3600,
        max_entries: int = 1000
    ):
        self.similarity_threshold = similarity_threshold
        self.default_ttl = default_ttl_seconds
        self.max_entries = max_entries
        self._lock = threading.Lock()

        # Cache entries stored by unique key
        # entry = {
        #   "id": str, "query": str, "vector": np.ndarray,
        #   "response": str, "model": str, "timestamp": float,
        #   "ttl": int, "hit_count": int, "last_accessed": float
        # }
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._total_lookups = 0
        self._total_hits = 0
        self._total_misses = 0

    @property
    def size(self) -> int:
        """Returns the number of active cached entries."""
        with self._lock:
            return len(self._entries)

    def lookup(
        self,
        query: str,
        threshold: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Searches for a semantically matching cached response.

        Args:
            query: Input prompt text.
            threshold: Optional custom similarity threshold (defaults to instance threshold).

        Returns:
            Matching cache dictionary or None if miss.
        """
        min_sim = threshold if threshold is not None else self.similarity_threshold
        now = time.time()

        with self._lock:
            self._total_lookups += 1
            if not self._entries:
                self._total_misses += 1
                return None

            # 1. Clean expired entries
            expired_keys = [
                k for k, v in self._entries.items()
                if now - v["timestamp"] > v["ttl"]
            ]
            for k in expired_keys:
                del self._entries[k]

            if not self._entries:
                self._total_misses += 1
                return None

            # 2. Vectorize query
            query_vec = embedding_engine.embed_query(query)

            # 3. Fast Vectorized Cosine Similarity Search
            entry_list = list(self._entries.values())
            vectors = np.array([e["vector"] for e in entry_list], dtype=np.float32)

            # Compute dot products (vectors are normalized to unit length)
            similarities = np.dot(vectors, query_vec)
            best_idx = int(np.argmax(similarities))
            best_similarity = float(similarities[best_idx])

            if best_similarity >= min_sim:
                best_entry = entry_list[best_idx]
                best_entry["hit_count"] += 1
                best_entry["last_accessed"] = now
                self._total_hits += 1

                debug_log(
                    "⚡ [SEMANTIC CACHE:HIT]",
                    f"Matched query '{best_entry['query'][:40]}...' (Similarity: {best_similarity:.4f})"
                )

                return {
                    "matched_query": best_entry["query"],
                    "response": best_entry["response"],
                    "model": best_entry["model"],
                    "similarity": best_similarity,
                    "hit_count": best_entry["hit_count"],
                    "cached_at": best_entry["timestamp"],
                    "latency_s": 0.001
                }

            self._total_misses += 1
            debug_log(
                "🔍 [SEMANTIC CACHE:MISS]",
                f"Best similarity {best_similarity:.4f} < threshold {min_sim:.4f}"
            )
            return None

    def store(
        self,
        query: str,
        response: str,
        model: str = "default",
        ttl_seconds: Optional[int] = None
    ) -> str:
        """Stores a new query-response pair in the semantic vector cache.

        Args:
            query: Input prompt text.
            response: Model output text.
            model: Model name.
            ttl_seconds: Optional custom TTL in seconds.

        Returns:
            Entry ID.
        """
        now = time.time()
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        query_vec = embedding_engine.embed_query(query)
        entry_id = f"entry-{hash(query) & 0xffffffff:x}"

        with self._lock:
            # Check capacity and evict least recently used (LRU) entry if full
            if len(self._entries) >= self.max_entries:
                lru_key = min(self._entries.keys(), key=lambda k: self._entries[k]["last_accessed"])
                del self._entries[lru_key]

            self._entries[entry_id] = {
                "id": entry_id,
                "query": query,
                "vector": query_vec,
                "response": response,
                "model": model,
                "timestamp": now,
                "last_accessed": now,
                "ttl": ttl,
                "hit_count": 0
            }

            debug_log(
                "💾 [SEMANTIC CACHE:STORED]",
                f"Cached '{query[:40]}...' (TTL: {ttl}s, Total Cached: {len(self._entries)})"
            )

        return entry_id

    def clear(self):
        """Clears all entries from cache."""
        with self._lock:
            self._entries.clear()
            self._total_lookups = 0
            self._total_hits = 0
            self._total_misses = 0

    def stats(self) -> Dict[str, Any]:
        """Returns operational cache performance metrics."""
        with self._lock:
            hit_rate = (self._total_hits / self._total_lookups * 100) if self._total_lookups > 0 else 0.0
            return {
                "total_entries": len(self._entries),
                "total_lookups": self._total_lookups,
                "total_hits": self._total_hits,
                "total_misses": self._total_misses,
                "hit_rate_pct": round(hit_rate, 2),
                "similarity_threshold": self.similarity_threshold,
                "max_entries": self.max_entries
            }


# Global singleton instance
semantic_cache = SemanticVectorCache()
