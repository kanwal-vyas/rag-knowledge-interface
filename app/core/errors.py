"""
Typed error hierarchy for the query pipeline.

Rule: no exception here may carry a raw stack trace or secret value in its
public `detail` — the API layer serializes `detail` directly to clients.
Internal-only diagnostic info should be logged via structlog, not attached
to the exception's public fields.
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for all typed pipeline errors."""

    http_status: int = 500
    error_code: str = "internal_error"

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class InvalidRequestError(PipelineError):
    http_status = 400
    error_code = "invalid_request"


class InvalidAudioError(PipelineError):
    http_status = 400
    error_code = "invalid_audio"


class STTFailureError(PipelineError):
    http_status = 502
    error_code = "stt_failure"


class EmbeddingFailureError(PipelineError):
    http_status = 502
    error_code = "embedding_failure"


class RetrievalFailureError(PipelineError):
    http_status = 502
    error_code = "retrieval_failure"


class VectorDBFailureError(PipelineError):
    http_status = 502
    error_code = "vector_db_failure"


class GenerationTimeoutError(PipelineError):
    http_status = 504
    error_code = "generation_timeout"


class ProviderTimeoutError(PipelineError):
    http_status = 504
    error_code = "provider_timeout"


class RateLimitError(PipelineError):
    http_status = 429
    error_code = "rate_limited"


class InsufficientEvidenceError(PipelineError):
    """
    Raised when retrieval/grounding cannot support a confident answer.
    Not a system failure — a correct refusal to answer. Kept distinct from
    GuardrailRefusalError because the trigger condition differs (evidence
    quality vs. policy/safety refusal), even though both result in a
    non-answer to the user.
    """

    http_status = 200
    error_code = "insufficient_evidence"


class GuardrailRefusalError(PipelineError):
    http_status = 200
    error_code = "guardrail_refusal"
