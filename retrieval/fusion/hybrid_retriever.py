"""
Hybrid retriever — combines BM25 sparse + bge-m3 dense via RRF.

Implements the Retriever ABC so it can be benchmarked interchangeably
with BM25-only and dense-only.

Architecture alignment:
- ADR-0003: sparse=BM25, dense=bge-m3, fusion=experimental
- Audit §7: Config 1 (MONOLINGUAL) / Config 2 (CROSS_LINGUAL)
"""
from __future__ import annotations

import time

import structlog

from app.models.retrieval import (
    Query,
    RetrievalMode,
    RetrievalResult,
)
from retrieval.dense.dense_retriever import BgeM3DenseRetriever
from retrieval.fusion.rrf import DEFAULT_RRF_K, reciprocal_rank_fusion
from retrieval.sparse.bm25_retriever import BM25SparseRetriever

logger = structlog.get_logger(__name__)


class HybridRetriever:
    """
    Combines BM25 sparse retrieval with bge-m3 dense retrieval via RRF.

    Usage:
        hybrid = HybridRetriever(
            sparse_retriever=bm25_retriever,
            dense_retriever=dense_retriever,
        )
        results = hybrid.retrieve(query, mode, top_k=10)
    """

    def __init__(
        self,
        sparse_retriever: BM25SparseRetriever,
        dense_retriever: BgeM3DenseRetriever,
        rrf_k: float = DEFAULT_RRF_K,
    ) -> None:
        self._sparse = sparse_retriever
        self._dense = dense_retriever
        self._rrf_k = rrf_k

    def retrieve(
        self,
        query: Query,
        mode: RetrievalMode,
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        Retrieve using both sparse and dense, fuse via RRF.

        Both retrievers are called with the same mode and top_k.
        Results are fused and deduplicated by canonical passage_id.
        """
        # Retrieve from both sources
        sparse_results = self._sparse.retrieve(query, mode, top_k)
        dense_results = self._dense.retrieve(query, mode, top_k)

        # Fuse via RRF
        fused = reciprocal_rank_fusion(
            sparse_results=sparse_results,
            dense_results=dense_results,
            top_k=top_k,
            k=self._rrf_k,
        )

        logger.debug(
            "hybrid_retrieval",
            query_lang=query.lang.value,
            mode=mode.value,
            sparse_count=len(sparse_results),
            dense_count=len(dense_results),
            fused_count=len(fused),
        )

        return fused

    def retrieve_with_latency(
        self,
        query: Query,
        mode: RetrievalMode,
        top_k: int,
    ) -> tuple[list[RetrievalResult], float]:
        """Retrieve with explicit latency measurement."""
        start_time = time.perf_counter()
        results = self.retrieve(query, mode, top_k)
        latency_ms = (time.perf_counter() - start_time) * 1000
        return results, latency_ms
