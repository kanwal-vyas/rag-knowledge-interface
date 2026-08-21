from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_app_starts() -> None:
    assert app.title.startswith("HH Goa 2026")


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


def test_query_endpoint_wired() -> None:
    """
    The pipeline is now wired. A query with a simple test query should
    return a 200 response with an answer (possibly empty if guardrails
    reject the query or no relevant passages are found).
    """
    response = client.post(
        "/query", json={"query_text": "test query", "lang": "hi", "retrieval_mode": "monolingual"}
    )
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "grounded" in body
    assert "request_id" in body
    assert "latency" in body


def test_voice_query_endpoint_wired() -> None:
    """Voice endpoint is wired but requires audio file upload."""
    response = client.post("/voice/query")
    # Should return 422 (missing required file) not 501 (not implemented)
    assert response.status_code == 422
