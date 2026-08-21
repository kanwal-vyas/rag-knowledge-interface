from __future__ import annotations

from app.models.retrieval import RetrievalMode
from evaluation.benchmark_models import BenchmarkResult, LatencyPercentiles, RetrievalMetrics
from evaluation.models import BENCHMARK_POOL, TUNING_POOL


def test_benchmark_result_schema_accepts_null_metrics() -> None:
    """No benchmark has run yet — schema must allow all-null metrics rather
    than forcing fabricated placeholder numbers."""
    result = BenchmarkResult(
        run_id="scaffold-test",
        query_pool=TUNING_POOL,
        retrieval_mode=RetrievalMode.MONOLINGUAL,
        corpus_tier="T1",
        retrieval_metrics=RetrievalMetrics(),
        latency=LatencyPercentiles(),
        is_final_reported=False,
    )
    assert result.retrieval_metrics.recall_at_10 is None
    assert result.is_final_reported is False


def test_final_reported_result_must_use_benchmark_pool_by_convention() -> None:
    """
    Schema-level: this is a documentation/contract test, not an enforced
    constraint (Pydantic won't refuse query_pool='tuning' with
    is_final_reported=True) — real enforcement belongs in the evaluation
    harness once implemented. This test exists so the constraint isn't
    forgotten.
    """
    result = BenchmarkResult(
        run_id="scaffold-test-2",
        query_pool=BENCHMARK_POOL,
        retrieval_mode=RetrievalMode.CROSS_LINGUAL,
        corpus_tier="T3",
        retrieval_metrics=RetrievalMetrics(recall_at_10=None),
        latency=LatencyPercentiles(),
        is_final_reported=True,
    )
    assert result.query_pool == BENCHMARK_POOL
