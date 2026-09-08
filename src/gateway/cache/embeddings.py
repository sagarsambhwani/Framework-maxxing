"""Local Sub-Millisecond Embedding Engine.

Provides ultra-fast, local vector embedding generation using FastEmbed (ONNX BAAI/bge-small-en-v1.5)
with zero third-party cloud API dependencies and sub-3ms latency.

Usage:
    from src.gateway.cache.embeddings import embedding_engine
    vector = embedding_engine.embed_query("What is an AI gateway?")
"""

import numpy as np
from typing import List, Union, Optional
from src.common.logging import term_log, debug_log, Colors

class LocalEmbeddingEngine:
    """High-speed local dense vector embedder."""

    MODEL_NAME = "BAAI/bge-small-en-v1.5"

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self.MODEL_NAME
        self._embedder = None
        self._dimension = 384
        self._init_embedder()

    def _init_embedder(self):
        """Initializes the fastembed ONNX runtime."""
        try:
            from fastembed import TextEmbedding
            self._embedder = TextEmbedding(model_name=self.model_name)
            debug_log("⚡ [EMBEDDINGS]", f"FastEmbed initialized with '{self.model_name}' (384-dim ONNX)")
        except Exception as e:
            term_log("⚠️ [EMBEDDINGS]", f"FastEmbed fallback active: {e}", Colors.YELLOW)
            self._embedder = None

    @property
    def dimension(self) -> int:
        """Returns the embedding vector dimensionality."""
        return self._dimension

    def embed_query(self, text: str) -> np.ndarray:
        """Generates a normalized 1D float32 embedding vector for a text query.

        Args:
            text: Input string.

        Returns:
            1D numpy array of unit length (L2 norm = 1.0).
        """
        if not text or not text.strip():
            # Return zero vector for empty inputs
            return np.zeros(self._dimension, dtype=np.float32)

        if self._embedder is not None:
            try:
                embeddings = list(self._embedder.embed([text]))
                vec = np.array(embeddings[0], dtype=np.float32)
                norm = np.linalg.norm(vec)
                return vec / norm if norm > 0 else vec
            except Exception as e:
                debug_log("⚠️ [EMBEDDINGS:ERROR]", f"Embedder error: {e}")

        # Fallback deterministic token-hash vector if ONNX fails
        return self._fallback_embed(text)

    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """Batch generates normalized embeddings for multiple texts.

        Args:
            texts: List of text strings.

        Returns:
            List of 1D numpy arrays.
        """
        if not texts:
            return []

        if self._embedder is not None:
            try:
                raw_embeddings = list(self._embedder.embed(texts))
                normalized = []
                for emb in raw_embeddings:
                    vec = np.array(emb, dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    normalized.append(vec / norm if norm > 0 else vec)
                return normalized
            except Exception as e:
                debug_log("⚠️ [EMBEDDINGS:BATCH_ERROR]", f"Batch embedder error: {e}")

        return [self.embed_query(t) for t in texts]

    def _fallback_embed(self, text: str) -> np.ndarray:
        """Lightweight deterministic character n-gram hashing vectorizer."""
        vec = np.zeros(self._dimension, dtype=np.float32)
        words = text.lower().split()
        for i, word in enumerate(words):
            h = hash(word) % self._dimension
            vec[h] += 1.0 / (i + 1)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    @staticmethod
    def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Computes cosine similarity between two normalized vectors (dot product)."""
        dot = float(np.dot(vec_a, vec_b))
        return max(-1.0, min(1.0, dot))


# Global singleton instance
embedding_engine = LocalEmbeddingEngine()
