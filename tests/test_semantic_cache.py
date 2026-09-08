"""Unit Tests for Semantic Vector Caching Engine.

Tests:
    1. Local embedding generation and normalization.
    2. Cosine similarity accuracy.
    3. Exact string & vector cache hits.
    4. Semantically rephrased cache hits (threshold >= 0.88).
    5. Unrelated query cache misses.
    6. TTL expiration handling.
    7. Max entries LRU eviction.
    8. Gateway integration with semantic cache.
"""

import time
import numpy as np
import pytest

from src.gateway.cache.embeddings import LocalEmbeddingEngine, embedding_engine
from src.gateway.cache.semantic_cache import SemanticVectorCache, semantic_cache
from src.gateway.router import gateway


def test_embedding_engine_dimension_and_norm():
    """Verifies that the embedding engine generates 384-dim unit length vectors."""
    vec = embedding_engine.embed_query("What is an AI gateway?")
    assert isinstance(vec, np.ndarray)
    assert len(vec) == 384
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, rel=1e-3) == 1.0


def test_embedding_engine_cosine_similarity():
    """Verifies that cosine similarity between identical vectors is 1.0."""
    vec_a = embedding_engine.embed_query("Multi-cloud AI Gateway routing")
    vec_b = embedding_engine.embed_query("Multi-cloud AI Gateway routing")
    sim = LocalEmbeddingEngine.compute_cosine_similarity(vec_a, vec_b)
    assert pytest.approx(sim, rel=1e-3) == 1.0


def test_semantic_cache_exact_hit():
    """Verifies that an identical query hits the semantic cache with 1.0 similarity."""
    test_cache = SemanticVectorCache(similarity_threshold=0.88)
    test_cache.store("What is Groq LPU?", "Groq LPU is a Language Processing Unit.", model="qwen3.8")

    match = test_cache.lookup("What is Groq LPU?")
    assert match is not None
    assert match["response"] == "Groq LPU is a Language Processing Unit."
    assert match["model"] == "qwen3.8"
    assert match["similarity"] >= 0.99


def test_semantic_cache_paraphrase_hit():
    """Verifies that a semantically rephrased query hits the cache (>= 0.88)."""
    test_cache = SemanticVectorCache(similarity_threshold=0.88)
    test_cache.store(
        "What is an AI Gateway and why is it useful?",
        "An AI Gateway provides routing, failovers, and caching.",
        model="groq/qwen/qwen3.8-27b"
    )

    # Semantically rephrased query
    rephrased = "Explain what an AI gateway does and its main benefits"
    match = test_cache.lookup(rephrased)

    assert match is not None
    assert "AI Gateway provides routing" in match["response"]
    assert match["similarity"] >= 0.88


def test_semantic_cache_unrelated_miss():
    """Verifies that a completely unrelated query results in a cache miss."""
    test_cache = SemanticVectorCache(similarity_threshold=0.88)
    test_cache.store("How do neural networks work?", "Neural networks use layered weights.", model="test")

    unrelated = "How do I bake a chocolate cake?"
    match = test_cache.lookup(unrelated)
    assert match is None


def test_semantic_cache_ttl_expiration():
    """Verifies that expired entries are evicted and return None."""
    test_cache = SemanticVectorCache(similarity_threshold=0.88, default_ttl_seconds=1)
    test_cache.store("Quick query", "Quick answer", ttl_seconds=1)

    # Immediately after store: should hit
    assert test_cache.lookup("Quick query") is not None

    # Wait for TTL expiration
    time.sleep(1.1)
    assert test_cache.lookup("Quick query") is None


def test_semantic_cache_lru_eviction():
    """Verifies that the cache evicts the least recently used entry when max_entries is exceeded."""
    test_cache = SemanticVectorCache(similarity_threshold=0.88, max_entries=2)
    test_cache.store("Query 1", "Answer 1")
    test_cache.store("Query 2", "Answer 2")

    assert test_cache.size == 2

    # Add 3rd query -> should evict Query 1 (the oldest / least accessed)
    test_cache.store("Query 3", "Answer 3")
    assert test_cache.size == 2

    assert test_cache.lookup("Query 3") is not None
    assert test_cache.lookup("Query 2") is not None


def test_gateway_semantic_cache_integration():
    """Verifies that the Gateway Router seamlessly utilizes semantic caching."""
    semantic_cache.clear()

    q1 = "What is LiteLLM and how does it route requests?"
    
    # 1. First call (cold)
    res1 = gateway.complete(
        model="groq/qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": q1}],
        max_tokens=50,
        use_cache=True
    )
    assert res1["content"] != ""
    assert res1.get("cache_hit") is False

    # 2. Second call (rephrased query -> semantic cache hit)
    q2 = "Explain how LiteLLM works for routing model requests"
    res2 = gateway.complete(
        model="groq/qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": q2}],
        max_tokens=50,
        use_cache=True
    )
    assert res2.get("cache_hit") is True
    assert res2["provider"] == "Semantic Vector Cache"
    assert res2["latency_s"] <= 0.05
    assert res2["content"] == res1["content"]
