from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.errors import InsufficientEvidenceError, InvalidRequestError
from app.main import app

client = TestClient(app)


def test_invalid_request_error_maps_to_400() -> None:
    err = InvalidRequestError("bad input")
    assert err.http_status == 400
    assert err.error_code == "invalid_request"
    assert err.detail == "bad input"


def test_insufficient_evidence_is_not_a_500() -> None:
    """A correct refusal to answer is not a server failure."""
    err = InsufficientEvidenceError("not enough grounded evidence")
    assert err.http_status == 200
    assert err.error_code == "insufficient_evidence"


def test_query_endpoint_rejects_empty_body_with_typed_error_not_stacktrace() -> None:
    response = client.post("/query", json={"query_text": "", "lang": "hi"})
    # Empty query is handled by guardrails and returns 200 with empty answer.
    # No raw traceback should be exposed.
    assert response.status_code == 200
    assert "Traceback" not in response.text
    data = response.json()
    assert data["answer"] == ""
