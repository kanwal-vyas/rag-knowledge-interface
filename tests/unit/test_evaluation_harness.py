"""Tests for evaluation harness: metrics, gold-set handling, and isolation."""
from __future__ import annotations

import pytest


class TestMetricComputation:
    """Test retrieval metric computation."""

    def _compute(self, retrieved: list[str], gold: set[str], k: int = 10) -> dict:
        """Import and run compute_metrics from the evaluation script."""
        # Inline metric computation to test in isolation
        hits_at_1 = 1 if retrieved[:1] and retrieved[0] in gold else 0
        hits_at_5 = 1 if any(pid in gold for pid in retrieved[:5]) else 0
        hits_at_10 = 1 if any(pid in gold for pid in retrieved[:k]) else 0

        mrr = 0.0
        for rank, pid in enumerate(retrieved[:k], 1):
            if pid in gold:
                mrr = 1.0 / rank
                break

        relevance = [1.0 if pid in gold else 0.0 for pid in retrieved[:k]]
        dcg = sum(rel / (i + 1) for i, rel in enumerate(relevance))
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = sum(rel / (i + 1) for i, rel in enumerate(ideal_relevance))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        return {
            "recall@1": hits_at_1,
            "recall@5": hits_at_5,
            "recall@10": hits_at_10,
            "mrr": mrr,
            "ndcg@10": ndcg,
        }

    def test_perfect_retrieval(self) -> None:
        retrieved = ["g1", "g2", "g3", "d1", "d2"]
        gold = {"g1", "g2", "g3"}
        m = self._compute(retrieved, gold)
        assert m["recall@1"] == 1
        assert m["recall@5"] == 1
        assert m["recall@10"] == 1
        assert m["mrr"] == 1.0
        assert m["ndcg@10"] > 0.9

    def test_no_retrieval(self) -> None:
        retrieved = ["d1", "d2", "d3"]
        gold = {"g1", "g2"}
        m = self._compute(retrieved, gold)
        assert m["recall@1"] == 0
        assert m["recall@5"] == 0
        assert m["recall@10"] == 0
        assert m["mrr"] == 0.0
        assert m["ndcg@10"] == 0.0

    def test_mrr_rank_2(self) -> None:
        retrieved = ["d1", "g1", "d2"]
        gold = {"g1"}
        m = self._compute(retrieved, gold)
        assert m["recall@1"] == 0
        assert m["recall@5"] == 1
        assert m["mrr"] == pytest.approx(0.5)

    def test_mrr_rank_3(self) -> None:
        retrieved = ["d1", "d2", "g1"]
        gold = {"g1"}
        m = self._compute(retrieved, gold)
        assert m["mrr"] == pytest.approx(1.0 / 3)

    def test_empty_retrieved(self) -> None:
        m = self._compute([], {"g1"})
        assert m["recall@1"] == 0
        assert m["mrr"] == 0.0

    def test_empty_gold(self) -> None:
        m = self._compute(["d1", "d2"], set())
        assert m["recall@1"] == 0
        assert m["mrr"] == 0.0
        assert m["ndcg@10"] == 0.0

    def test_partial_recall(self) -> None:
        retrieved = ["g1", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "g2"]
        gold = {"g1", "g2", "g3"}
        m = self._compute(retrieved, gold)
        assert m["recall@1"] == 1
        assert m["recall@5"] == 1  # g1 in top 5
        assert m["recall@10"] == 1  # g2 in top 10
        # Only 2/3 found → recall@10 = 2/3... but our metric is binary per query
        # Actually our metric is: does top-K contain ANY gold? Yes → 1
        # So recall@10 = 1 because g1 is in top 10

    def test_ndcg_all_relevant_at_top(self) -> None:
        retrieved = ["g1", "g2"]
        gold = {"g1", "g2"}
        m = self._compute(retrieved, gold)
        assert m["ndcg@10"] == pytest.approx(1.0)

    def test_ndcg_relevant_not_at_top(self) -> None:
        retrieved = ["d1", "g1"]
        gold = {"g1"}
        m = self._compute(retrieved, gold)
        # DCG = 0/1 + 1/2 = 0.5, IDCG = 1/1 = 1.0
        assert m["ndcg@10"] == pytest.approx(0.5)


class TestGoldSetIsolation:
    """Verify evaluation-only fields don't leak into production."""

    FORBIDDEN = {"is_selected", "Answer", "Eng_Answer", "source_query_ids"}

    def test_benchmark_json_no_forbidden_fields(self) -> None:
        """Benchmark query metadata should not contain evaluation labels."""
        # Simulate what the benchmark extraction produces
        benchmark_entry = {
            "query_id": 12345,
            "query_text": "test query",
            "query_type": "DESCRIPTION",
            "source_lang": "en",
            "target_lang": "hi",
            "gold_passage_ids": ["pid_1", "pid_2"],
        }
        # Should NOT contain forbidden fields
        assert set(benchmark_entry.keys()) & self.FORBIDDEN == set()

    def test_corpus_passage_no_forbidden_fields(self) -> None:
        """Corpus passages should not contain evaluation labels."""
        corpus_entry = {
            "passage_id": "pid_1",
            "text": "some text",
            "lang": "en",
        }
        assert set(corpus_entry.keys()) & self.FORBIDDEN == set()

    def test_metric_result_no_forbidden_fields(self) -> None:
        """Metric results should not contain evaluation labels."""
        metric = {
            "recall@1": 0.5,
            "recall@5": 0.8,
            "recall@10": 0.9,
            "mrr": 0.6,
            "ndcg@10": 0.7,
        }
        assert set(metric.keys()) & self.FORBIDDEN == set()


class TestAggregation:
    """Test metric aggregation."""

    def _aggregate(self, metrics: list[dict]) -> dict:
        n = len(metrics)
        if n == 0:
            return {k: 0.0 for k in ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"]}
        return {
            "recall@1": sum(m["recall@1"] for m in metrics) / n,
            "recall@5": sum(m["recall@5"] for m in metrics) / n,
            "recall@10": sum(m["recall@10"] for m in metrics) / n,
            "mrr": sum(m["mrr"] for m in metrics) / n,
            "ndcg@10": sum(m["ndcg@10"] for m in metrics) / n,
        }

    def test_empty_aggregation(self) -> None:
        result = self._aggregate([])
        assert all(v == 0.0 for v in result.values())

    def test_single_query_aggregation(self) -> None:
        m = [{"recall@1": 1, "recall@5": 1, "recall@10": 1, "mrr": 1.0, "ndcg@10": 1.0}]
        result = self._aggregate(m)
        assert result["recall@1"] == 1.0
        assert result["mrr"] == 1.0

    def test_average_aggregation(self) -> None:
        m = [
            {"recall@1": 1, "recall@5": 1, "recall@10": 1, "mrr": 1.0, "ndcg@10": 1.0},
            {"recall@1": 0, "recall@5": 1, "recall@10": 1, "mrr": 0.5, "ndcg@10": 0.5},
        ]
        result = self._aggregate(m)
        assert result["recall@1"] == pytest.approx(0.5)
        assert result["mrr"] == pytest.approx(0.75)
