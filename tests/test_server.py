"""Tests for FastAPI Web Server Endpoints."""
import pytest
from fastapi.testclient import TestClient
from src.server.app import create_app

app = create_app()
client = TestClient(app)


def test_server_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Groq LPU" in data["providers"]


def test_server_ui_index():
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>ChatGPT Pro" in response.text
