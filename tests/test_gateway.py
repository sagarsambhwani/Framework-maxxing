"""Tests for Multi-Provider Gateway Routing."""
import pytest
from src.gateway.router import gateway


def test_gateway_initialization():
    assert gateway is not None


def test_gateway_completion_basic():
    res = gateway.complete(
        model="groq/qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": "Say hello in 3 words."}],
        max_tokens=20
    )
    assert res is not None
    assert "content" in res
    assert "latency_s" in res
    assert res["latency_s"] >= 0


def test_gateway_fallback():
    res = gateway.complete(
        model="openrouter/inclusionai/ling-3.0-flash-fin:free",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=20
    )
    assert res is not None
    assert "content" in res
