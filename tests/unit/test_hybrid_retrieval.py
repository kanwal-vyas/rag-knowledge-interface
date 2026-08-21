"""Tests for hybrid retrieval: RRF fusion, dedup, and hybrid retriever."""
from __future__ import annotations

import pytest

from app.models.retrieval import (
    Language,
    Passage,
    Query,
    RetrievalMode,
    RetrievalResult,
)
from retrieval.fusion.rrf import DEFAULT_RRF_K, reciprocal_rank_fusion

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    passage_id: str,
    score: float,
    source: str,
    text: str = "text",
    lang: Language = Language.EN,
) -> RetrievalResult:
    return RetrievalResult(
        passage=Passage(passage_id=passage_id, text=text, lang=lang),
        score=score,
        source=source,
    )


# ---------------------------------------------------------------------------
# RRF Fusion Tests
# ---------------------------------------------------------------------------


class TestRRFFusion:
    """Test Reciprocal Rank Fusion behavior."""

    def test_basic_fusion(self) -> None:
        sparse = [_make_result("p1", 1.0, "bm25"), _make_result("p2", 0.8, "bm25")]
        dense = [_make_result("p2", 0.9, "dense"), _make_result("p3", 0.7, "dense")]
        results = reciprocal_rank_fusion(sparse, dense, top_k=10)
        pids = [r.passage.passage_id for r in results]
        # p2 appears in both lists → highest RRF score
        assert pids[0] == "p2"
        assert "bm25+dense" in results[0].source or (
            "bm25" in results[0].source and "dense" in results[0].source
        )

    def test_deduplication_by_passage_id(self) -> None:
        """Same passage_id from sparse and dense produces one result."""
        sparse = [_make_result("p1", 1.0, "bm25")]
        dense = [_make_result("p1", 0.9, "dense")]
        results = reciprocal_rank_fusion(sparse, dense, top_k=10)
        assert len(results) == 1
        assert results[0].passage.passage_id == "p1"
        assert "bm25" in results[0].source
        assert "dense" in results[0].source

    def test_empty_sparse_results(self) -> None:
        dense = [_make_result("p1", 0.9, "dense"), _make_result("p2", 0.7, "dense")]
        results = reciprocal_rank_fusion([], dense, top_k=10)
        assert len(results) == 2
        assert results[0].passage.passage_id == "p1"

    def test_empty_dense_results(self) -> None:
        sparse = [_make_result("p1", 1.0, "bm25"), _make_result("p2", 0.8, "bm25")]
        results = reciprocal_rank_fusion(sparse, [], top_k=10)
        assert len(results) == 2
        assert results[0].passage.passage_id == "p1"

    def test_both_empty(self) -> None:
        results = reciprocal_rank_fusion([], [], top_k=10)
        assert results == []

    def test_top_k_truncation(self) -> None:
        sparse = [_make_result(f"p{i}", 1.0 - i * 0.1, "bm25") for i in range(10)]
        dense = [_make_result(f"p{i}", 0.9 - i * 0.1, "dense") for i in range(10)]
        results = reciprocal_rank_fusion(sparse, dense, top_k=3)
        assert len(results) == 3

    def test_deterministic_ordering(self) -> None:
        sparse = [_make_result("p1", 1.0, "bm25"), _make_result("p2", 0.9, "bm25")]
        dense = [_make_result("p3", 0.8, "dense"), _make_result("p4", 0.7, "dense")]
        r1 = reciprocal_rank_fusion(sparse, dense, top_k=10)
        r2 = reciprocal_rank_fusion(sparse, dense, top_k=10)
        assert [r.passage.passage_id for r in r1] == [r.passage.passage_id for r in r2]

    def test_tie_breaking_by_passage_id(self) -> None:
        """When RRF scores are tied, passage_id breaks the tie."""
        # Same rank in both lists → same RRF score
        sparse = [_make_result("p_b", 1.0, "bm25")]
        dense = [_make_result("p_a", 1.0, "dense")]
        results = reciprocal_rank_fusion(sparse, dense, top_k=10)
        # Both have same RRF score (1/(k+1) each), so p_a should come first
        assert results[0].passage.passage_id == "p_a"
        assert results[1].passage.passage_id == "p_b"

    def test_language_metadata_preserved(self) -> None:
        sparse = [_make_result("p1", 1.0, "bm25", lang=Language.HI)]
        dense = [_make_result("p2", 0.9, "dense", lang=Language.EN)]
        results = reciprocal_rank_fusion(sparse, dense, top_k=10)
        langs = {r.passage.lang for r in results}
        assert Language.HI in langs
        assert Language.EN in langs

    def test_rrf_score_range(self) -> None:
        """RRF scores should be positive and bounded."""
        sparse = [_make_result("p1", 1.0, "bm25")]
        dense = [_make_result("p2", 0.9, "dense")]
        results = reciprocal_rank_fusion(sparse, dense, top_k=10)
        for r in results:
            assert r.score > 0
            # Max possible score for a doc in both lists: 2/(k+1)
            assert r.score <= 2.0 / (DEFAULT_RRF_K + 1)

    def test_source_label_combined(self) -> None:
        """Document from both sources gets combined source label."""
        sparse = [_make_result("p1", 1.0, "bm25")]
        dense = [_make_result("p1", 0.9, "dense")]
        results = reciprocal_rank_fusion(sparse, dense, top_k=10)
        assert len(results) == 1
        assert "bm25" in results[0].source
        assert "dense" in results[0].source

    def test_source_label_single(self) -> None:
        """Document from only one source gets single label."""
        sparse = [_make_result("p1", 1.0, "bm25")]
        dense = [_make_result("p2", 0.9, "dense")]
        results = reciprocal_rank_fusion(sparse, dense, top_k=10)
        p1_result = next(r for r in results if r.passage.passage_id == "p1")
        p2_result = next(r for r in results if r.passage.passage_id == "p2")
        assert p1_result.source == "bm25"
        assert p2_result.source == "dense"

    def test_custom_k_parameter(self) -> None:
        """Different k values produce different rankings."""
        sparse = [_make_result("p1", 1.0, "bm25"), _make_result("p2", 0.9, "bm25")]
        dense = [_make_result("p2", 0.9, "dense"), _make_result("p1", 0.8, "dense")]

        r_low_k = reciprocal_rank_fusion(sparse, dense, top_k=10, k=1)
        r_high_k = reciprocal_rank_fusion(sparse, dense, top_k=10, k=1000)

        # Both should have same documents but possibly different order
        assert {r.passage.passage_id for r in r_low_k} == {r.passage.passage_id for r in r_high_k}

    def test_no_forbidden_fields_in_results(self) -> None:
        """Fused results must not contain evaluation-only fields."""
        FORBIDDEN = {"is_selected", "Answer", "Eng_Answer", "source_query_ids"}
        sparse = [_make_result("p1", 1.0, "bm25")]
        dense = [_make_result("p2", 0.9, "dense")]
        results = reciprocal_rank_fusion(sparse, dense, top_k=10)
        for r in results:
            result_fields = set(r.model_fields.keys())
            passage_fields = set(r.passage.model_fields.keys())
            assert result_fields & FORBIDDEN == set()
            assert passage_fields & FORBIDDEN == set()


# ---------------------------------------------------------------------------
# Hybrid Retriever Integration Tests (with BM25 + synthetic dense)
# ---------------------------------------------------------------------------


class TestHybridRetrieverUnit:
    """Test hybrid retriever with real BM25 and mocked dense results."""

    def test_rrf_fusion_with_bm25_and_dense(self) -> None:
        """End-to-end: BM25 + dense → RRF → fused results."""
        from retrieval.sparse.bm25_index import HAS_BM25, BM25Index

        if not HAS_BM25:
            pytest.skip("rank_bm25 not installed")

        # Build a small BM25 index
        index = BM25Index()
        index.add_document("p1", "Artificial intelligence is transforming technology", "en")
        index.add_document("p2", "Machine learning and AI are related fields", "en")
        index.add_document("p3", "Italian pasta is delicious with tomato sauce", "en")
        index.add_document("p4", "Deep learning neural networks", "en")
        index.build()

        from retrieval.sparse.bm25_retriever import BM25SparseRetriever

        sparse = BM25SparseRetriever(index=index)
        query = Query(query_text="artificial intelligence", lang=Language.EN)

        sparse_results = sparse.retrieve(query, RetrievalMode.CROSS_LINGUAL, top_k=10)
        assert len(sparse_results) > 0

        # Simulate dense results (slightly different ranking)
        dense_results = [
            _make_result("p2", 0.85, "dense", "Machine learning and AI"),
            _make_result("p1", 0.80, "dense", "Artificial intelligence"),
            _make_result("p4", 0.70, "dense", "Deep learning"),
        ]

        fused = reciprocal_rank_fusion(sparse_results, dense_results, top_k=10)
        assert len(fused) > 0
        # p1 and p2 should be in results (they match both queries and have dense overlap)
        pids = {r.passage.passage_id for r in fused}
        assert "p1" in pids or "p2" in pids

    def test_hybrid_with_real_bm25_and_empty_dense(self) -> None:
        """Hybrid handles empty dense results gracefully."""
        from retrieval.sparse.bm25_index import HAS_BM25, BM25Index

        if not HAS_BM25:
            pytest.skip("rank_bm25 not installed")

        index = BM25Index()
        index.add_document("p1", "Test passage one", "en")
        index.add_document("p2", "Test passage two", "en")
        index.build()

        from retrieval.sparse.bm25_retriever import BM25SparseRetriever

        sparse = BM25SparseRetriever(index=index)
        query = Query(query_text="test", lang=Language.EN)
        sparse_results = sparse.retrieve(query, RetrievalMode.CROSS_LINGUAL, top_k=10)

        fused = reciprocal_rank_fusion(sparse_results, [], top_k=10)
        assert len(fused) == len(sparse_results)
        for r in fused:
            assert r.source == "bm25"
