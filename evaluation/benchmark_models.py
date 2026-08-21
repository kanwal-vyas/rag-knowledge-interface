"""
Benchmark result schemas. Structures only — no benchmark has been executed
yet, and no numbers here are real. See project principle: never fabricate
benchmark data. Do not populate these with placeholder numbers "for demo
purposes" anywhere in the codebase.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.models.retrieval import RetrievalMode


class RetrievalMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recall_at_1: float | None = None
    recall_at_5: float | None = None
    recall_at_10: float | None = None
    mrr: float | None = None
    ndcg: float | None = None


class LatencyPercentiles(BaseModel):
    model_config = ConfigDict(extra="forbid")

    p50_ms: float | None = None
    p70_ms: float | None = None
    p95_ms: float | None = None
    p100_ms: float | None = None


class BenchmarkResult(BaseModel):
    """
    One benchmark run's result. `query_pool` must be BENCHMARK_POOL
    (evaluation.models.BENCHMARK_POOL) for any result reported as final —
    a result computed against the tuning pool is a tuning artifact, not a
    reportable benchmark, per Audit §3/§10 freeze decision 2.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    query_pool: str = Field(
        ...,
        description=(
            'Must be "benchmark" for reportable results, "tuning" for calibration runs.'
        ),
    )
    retrieval_mode: RetrievalMode
    corpus_tier: str = Field(
        ..., description="e.g. T1..T5 — exact tier definitions: ARCHITECTURE DETAIL MISSING"
    )

    retrieval_metrics: RetrievalMetrics
    latency: LatencyPercentiles
    throughput_qps: float | None = None
    error_rate: float | None = None

    is_final_reported: bool = Field(
        ...,
        description=(
            "True only if this run touched the frozen validation benchmark set, and only "
            "after every tunable parameter was already locked from the tuning pool. "
            "False for any tuning-pool calibration run."
        ),
    )
