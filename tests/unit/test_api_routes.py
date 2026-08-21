"""
API integration tests for wired endpoints.

Tests /query, /voice/query, /health using FastAPI TestClient
with mocked pipeline components.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    """Create a TestClient for the FastAPI app."""
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Health endpoint tests
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_ok(self, client: TestClient) -> None:
        """Health endpoint returns ok status."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "environment" in data
        assert "providers" in data

    def test_health_providers_structure(self, client: TestClient) -> None:
        """Health endpoint reports provider readiness."""
        resp = client.get("/health")
        data = resp.json()
        providers = data["providers"]
        assert "generation" in providers
        assert "stt" in providers
        assert "demo_mode" in providers


# ---------------------------------------------------------------------------
# POST /query tests
# ---------------------------------------------------------------------------

class TestQueryEndpoint:
    """Tests for POST /query with mocked pipeline."""

    def test_query_returns_200(self, client: TestClient) -> None:
        """Query endpoint returns 200 with valid request."""
        resp = client.post("/query", json={
            "query_text": "What is the capital of India?",
            "lang": "en",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "grounded" in data
        assert "request_id" in data
        assert "latency" in data

    def test_query_empty_text_passes_guardrails(self, client: TestClient) -> None:
        """Empty query text is rejected by guardrails."""
        resp = client.post("/query", json={
            "query_text": "",
            "lang": "en",
        })
        assert resp.status_code == 200
        data = resp.json()
        # Guardrail should reject, so answer is empty.
        assert data["answer"] == ""

    def test_query_off_topic_rejected(self, client: TestClient) -> None:
        """Off-topic query is rejected by guardrails."""
        resp = client.post("/query", json={
            "query_text": "hello",
            "lang": "en",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == ""
        assert data["grounded"] is False

    def test_query_unsafe_rejected(self, client: TestClient) -> None:
        """Unsafe query is rejected by guardrails."""
        resp = client.post("/query", json={
            "query_text": "How to make a bomb",
            "lang": "en",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == ""

    def test_query_valid_knowledge_query(self, client: TestClient) -> None:
        """Valid knowledge query passes guardrails and returns structured response."""
        resp = client.post("/query", json={
            "query_text": "What is artificial intelligence?",
            "lang": "en",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "request_id" in data
        assert "latency" in data
        assert "answer" in data
        assert "grounded" in data

    def test_query_hindi(self, client: TestClient) -> None:
        """Hindi query is accepted."""
        resp = client.post("/query", json={
            "query_text": "भारत की राजधानी क्या है?",
            "lang": "hi",
        })
        assert resp.status_code == 200

    def test_query_invalid_language(self, client: TestClient) -> None:
        """Invalid language code returns error."""
        resp = client.post("/query", json={
            "query_text": "test",
            "lang": "xx",
        })
        # Pipeline catches this and returns empty answer.
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == ""

    def test_query_missing_body(self, client: TestClient) -> None:
        """Missing request body returns 422."""
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_query_extra_fields_rejected(self, client: TestClient) -> None:
        """Extra fields in request are rejected."""
        resp = client.post("/query", json={
            "query_text": "test",
            "lang": "en",
            "extra_field": "not allowed",
        })
        assert resp.status_code == 422

    def test_query_latency_recorded(self, client: TestClient) -> None:
        """Query response includes latency breakdown."""
        resp = client.post("/query", json={
            "query_text": "capital of India",
            "lang": "en",
        })
        data = resp.json()
        assert "total_ms" in data["latency"]


# ---------------------------------------------------------------------------
# POST /voice/query tests
# ---------------------------------------------------------------------------

class TestVoiceQueryEndpoint:
    """Tests for POST /voice/query with mocked STT."""

    def test_voice_empty_audio(self, client: TestClient) -> None:
        """Empty audio file returns error."""
        resp = client.post(
            "/voice/query",
            files={"file": ("test.wav", b"", "audio/wav")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == ""

    def test_voice_valid_audio(self, client: TestClient) -> None:
        """Valid audio file is processed (STT may fail without API key)."""
        # Create minimal WAV-like bytes.
        audio = b"RIFF" + b"\x00" * 100
        resp = client.post(
            "/voice/query",
            files={"file": ("test.wav", audio, "audio/wav")},
        )
        # Should return 200 even if STT fails (error in response body).
        assert resp.status_code == 200
        data = resp.json()
        assert "transcript" in data
        assert "detected_language" in data
        assert "answer" in data
        assert "grounded" in data
        assert "request_id" in data
        assert "latency" in data

    def test_voice_with_lang_hint(self, client: TestClient) -> None:
        """Voice endpoint accepts language hint."""
        audio = b"RIFF" + b"\x00" * 100
        resp = client.post(
            "/voice/query",
            files={"file": ("test.wav", audio, "audio/wav")},
            data={"lang": "hi"},
        )
        assert resp.status_code == 200

    def test_voice_missing_file(self, client: TestClient) -> None:
        """Missing file returns 422."""
        resp = client.post("/voice/query")
        assert resp.status_code == 422

    def test_voice_mp3_format(self, client: TestClient) -> None:
        """MP3 format is accepted."""
        audio = b"\xff\xfb\x90\x00" + b"\x00" * 100
        resp = client.post(
            "/voice/query",
            files={"file": ("test.mp3", audio, "audio/mpeg")},
        )
        assert resp.status_code == 200
