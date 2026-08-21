"""
Latency instrumentation models.

Project principle (see project instructions, not the Audit): latency claims
are experimentally verifiable claims, never assumptions, and offline
ingestion work is kept separate from online query latency. This module
covers ONLINE query-path latency only. Offline ingestion timing (corpus
build, embedding of the corpus, indexing) is a separate concern and must
not be mixed into this model — see benchmark/ for offline metrics.

ARCHITECTURE DETAIL MISSING — REQUIRES CONFIRMATION
The full harness stage list is inherited from V2 (Audit §10.7) but not
reproduced in the available source. The stage list below is the minimal
set the Audit's own text *requires* to exist (because it discusses their
latency explicitly), not a complete harness stage list. Do not treat this
as exhaustive — extend it once the actual harness spec is available, and
do not silently rename or reorder these fields without updating
ARCHITECTURE.md.

All durations are milliseconds, measured with a monotonic clock
(time.perf_counter() or equivalent) — never wall-clock timestamps, which
are unsuitable for duration measurement (can go backwards on clock sync,
DST, NTP adjustment).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LatencyBreakdown(BaseModel):
    """
    Structured per-request latency. Every field is optional except
    total_ms, because not every request exercises every stage (e.g. a
    text-only query skips STT; a guardrail rejection skips generation).
    Emit `None` rather than 0 for a stage that did not run — 0 would be
    indistinguishable from "ran in under a millisecond."
    """

    model_config = ConfigDict(extra="forbid")

    stt_ms: float | None = Field(None, description="Speech-to-text, voice queries only.")
    preprocessing_ms: float | None = None
    embedding_ms: float | None = Field(None, description="Query embedding via bge-m3.")
    sparse_retrieval_ms: float | None = Field(None, description="BM25 retrieval.")
    dense_retrieval_ms: float | None = Field(
        None,
        description=(
            "Qdrant search. Audit §7 notes this changes measurably between "
            "filtered (Config 1) and unfiltered (Config 2) search."
        ),
    )
    fusion_ms: float | None = None
    reranking_ms: float | None = None
    context_assembly_ms: float | None = None
    generation_ms: float | None = None
    guardrail_ms: float | None = None

    total_ms: float = Field(
        ..., description="Wall time for the whole request, monotonic-clock measured."
    )

    is_estimated: bool = Field(
        ...,
        description=(
            "True if any field in this breakdown is a projection/estimate rather than a "
            "directly measured value. Per project principle: estimated and measured "
            "performance must never be conflated. A benchmark report must never present "
            "an estimated LatencyBreakdown as if it were measured."
        ),
    )
