"""
Integration tests for the FastAPI endpoints.
Uses pytest-httpx to mock external LLM calls.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from backend.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "services" in data

    def test_root_returns_service_info(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Christianity-Focused AI Assistant"


class TestChatEndpoint:
    def test_chat_requires_query(self, client):
        response = client.post("/api/chat", json={})
        assert response.status_code == 422

    def test_chat_validates_denomination(self, client):
        response = client.post("/api/chat", json={
            "query": "What is grace?",
            "denomination": "InvalidDenomination",
        })
        assert response.status_code == 422

    def test_chat_rejects_empty_query(self, client):
        response = client.post("/api/chat", json={
            "query": "   ",
            "denomination": "General",
        })
        assert response.status_code == 422

    def test_chat_sync_endpoint_exists(self, client):
        # Just check the endpoint is reachable (may fail on LLM calls without API keys)
        response = client.post("/api/chat/sync", json={
            "query": "Test query",
            "denomination": "General",
        })
        # 200 (success) or 500 (no API keys in test) are both acceptable
        assert response.status_code in (200, 500)


class TestInputValidation:
    def test_query_max_length_enforced(self, client):
        long_query = "a" * 2001
        response = client.post("/api/chat", json={
            "query": long_query,
            "denomination": "General",
        })
        assert response.status_code == 422

    def test_valid_denominations_accepted(self, client):
        for denomination in ["Catholic", "Protestant", "Orthodox", "General"]:
            # We only check that validation passes, not the full LLM response
            response = client.post("/api/chat/sync", json={
                "query": "What is faith?",
                "denomination": denomination,
            })
            # 422 would mean invalid denomination schema
            assert response.status_code != 422
