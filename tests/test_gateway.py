"""Tests for LiteLLM Gateway and OpenRouter Routing."""
import pytest
from src.gateway.router import get_gateway


def test_gateway_initialization():
    gateway = get_gateway()
    assert gateway is not None


def test_gateway_completion_basic():
    gateway = get_gateway()
    res = gateway.completion(
        model="fast-researcher",
        messages=[{"role": "user", "content": "Respond with 'pong'"}],
        max_tokens=20
    )
    assert res is not None
    assert "content" in res
    assert "usage" in res
    assert res.get("latency_seconds") >= 0


def test_gateway_fallback():
    gateway = get_gateway()
    res = gateway.completion(
        model="openrouter-claude",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=20
    )
    assert res is not None
    assert "content" in res
