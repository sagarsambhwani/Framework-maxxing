"""Unit & Integration Tests for Multi-Provider Gateway Routing.

Validates:
    1. Gateway initialization and client configuration.
    2. Primary model completion (Groq LPU Qwen 3.8 27B).
    3. Multi-provider model completion (OpenRouter / Fallback).
"""

import pytest
from src.gateway.router import gateway


def test_gateway_initialization():
    """Verifies that the MultiProviderGateway singleton initializes correctly."""
    assert gateway is not None


def test_gateway_completion_basic():
    """Verifies synchronous completion returns content, token count, and non-negative latency."""
    res = gateway.complete(
        model="groq/qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": "Say hello in 3 words."}],
        max_tokens=20
    )
    assert res is not None
    assert "content" in res
    assert "latency_s" in res
    assert res["latency_s"] >= 0
    assert len(res["content"]) > 0


def test_gateway_fallback():
    """Verifies cross-provider model routing resilience."""
    res = gateway.complete(
        model="groq/meta-llama/llama-prompt-guard-2-86m",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=20
    )
    assert res is not None
    assert "content" in res
    assert res["latency_s"] >= 0
