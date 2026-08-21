"""
Harness skeleton — orchestrates the request pipeline.

ARCHITECTURE DETAIL MISSING — REQUIRES CONFIRMATION
The actual harness structure is inherited from V2 (Audit §10.7: "hand-
written orchestration") but not reproduced in the available source. This
skeleton implements only the stage sequence that this task's own directory
brief and the Audit's supporting prose require to exist:

    request -> validation -> STT -> query processing -> retrieval
            -> context assembly -> generation -> guardrails -> response

Do not treat this stage list as confirmed architecture — it is the
minimum scaffold needed for the app to import and for latency
instrumentation to have somewhere to attach. Replace/extend once the real
V2 harness spec is available.

No model calls are implemented. Every stage raises NotImplementedError
until DeepSeek implements the pipeline.
"""

from __future__ import annotations

import time

import structlog

from app.core.errors import PipelineError
from app.models.generation import GenerationResponse, GuardrailResult
from app.models.latency import LatencyBreakdown
from app.models.retrieval import RetrievalMode

logger = structlog.get_logger(__name__)


class HarnessResult:
    """Container for a completed (or failed) pipeline run."""

    def __init__(
        self,
        response: GenerationResponse | None,
        guardrail: GuardrailResult | None,
        latency: LatencyBreakdown,
        error: PipelineError | None = None,
        transcript: str | None = None,
        detected_language: str | None = None,
    ) -> None:
        self.response = response
        self.guardrail = guardrail
        self.latency = latency
        self.error = error
        self.transcript = transcript
        self.detected_language = detected_language


class Harness:
    """
    Orchestrates one query end-to-end. Stage implementations are injected
    (constructor-provided) rather than hardcoded, so the harness stays
    swappable per the interfaces-not-implementations principle.
    """

    def __init__(self) -> None:
        # Concrete stage implementations are not wired yet — this is
        # intentional. See docs/DEEPSEEK_IMPLEMENTATION.md for the
        # implementation order.
        pass

    def run(self, query_text: str, lang: str, mode: RetrievalMode) -> HarnessResult:
        _start = time.perf_counter()  # noqa: F841 — wired to LatencyBreakdown once stages exist

        # Stage: validation
        if not query_text.strip():
            from app.core.errors import InvalidRequestError

            raise InvalidRequestError("Query text must not be empty.")

        # Stages: query processing -> retrieval -> context assembly ->
        # generation -> guardrails are all pending implementation.
        raise NotImplementedError(
            "Harness pipeline stages are not implemented in this bootstrap. "
            "See docs/DEEPSEEK_IMPLEMENTATION.md."
        )

    def run_voice(self, audio_bytes: bytes, audio_format: str) -> HarnessResult:
        raise NotImplementedError(
            "Voice pipeline (STT stage) is not implemented in this bootstrap. "
            "STT provider is unconfirmed — see docs/ARCHITECTURE.md §1."
        )
