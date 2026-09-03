"""Integration Tests for FastAPI Web Server & Endpoints.

Validates:
    1. `/api/health` reports status 'healthy' and listed cloud providers.
    2. `/` serves the ChatGPT Pro dark-themed web interface HTML.
"""

import pytest
from fastapi.testclient import TestClient
from src.server.app import create_app

# Instantiate test client for FastAPI application
app = create_app()
client = TestClient(app)


def test_server_health():
    """Verifies that the /api/health endpoint returns 200 OK and expected status keys."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Groq LPU" in data["providers"]
    assert "Google Gemini" in data["providers"]
    assert "OpenRouter" in data["providers"]


def test_server_ui_index():
    """Verifies that the root / route serves the ChatGPT Pro HTML frontend."""
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>ChatGPT Pro" in response.text
    assert "chat-messages" in response.text or "ChatGPT Pro" in response.text
