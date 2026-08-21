"""
Shared retrieval contracts.

BM25Retriever, DenseRetriever (Qdrant-backed), and any HybridRetriever must
all implement `Retriever` so they can be benchmarked interchangeably
(project principle: interfaces, not implementations, for anything the
architecture treats as swappable). BM25 and bge-m3/Qdrant are themselves
confirmed choices (Audit §4/§5/§7) — the interchangeability here is about
being able to run retrieval with/without fusion, with/without reranking,
and Config 1 vs Config 2 (Audit §7), not about swapping BM25 or bge-m3 out
for something else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.retrieval import Query, RetrievalMode, RetrievalResult


class Retriever(ABC):
    """Common contract for sparse, dense, and fused retrievers."""

    @abstractmethod
    def retrieve(self, query: Query, mode: RetrievalMode, top_k: int) -> list[RetrievalResult]:
        """
        Retrieve top_k results for `query` under `mode`.

        `mode` implements Audit §7's Config 1 (MONOLINGUAL, lang filter
        applied) vs Config 2 (CROSS_LINGUAL, no lang filter) distinction.
        Every concrete Retriever must honor `mode` identically so the
        experiment in Audit §7 is a fair comparison.
        """
        raise NotImplementedError


class SparseRetriever(Retriever):
    """BM25-backed retriever (confirmed, Audit §4)."""


class DenseRetriever(Retriever):
    """bge-m3 + Qdrant-backed retriever (confirmed, Audit §4/§5)."""


class RankFusion(ABC):
    """
    Combines sparse + dense result lists into one ranked list.

    ARCHITECTURE DETAIL MISSING — REQUIRES CONFIRMATION
    The exact fusion algorithm (e.g. RRF vs weighted-score fusion vs
    learned fusion) is not specified in the available architecture source.
    Do not pick one and call it frozen — implement this interface, leave
    the concrete strategy as a to-be-benchmarked alternative per project
    principle ("create an appropriate interface/abstraction... allow the
    implementation phase to benchmark the alternatives").
    """

    @abstractmethod
    def fuse(
        self,
        sparse_results: list[RetrievalResult],
        dense_results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        raise NotImplementedError


class Reranker(ABC):
    """
    ARCHITECTURE DETAIL MISSING — REQUIRES CONFIRMATION
    Exactly 3 off-the-shelf reranker candidates (A/B/C) are confirmed to
    exist (Audit §4, §10.7), and "no reranker" is explicitly a legitimate
    winner of that experiment. Which 3 rerankers is not specified. This
    interface must support a no-op/passthrough implementation as a first-
    class citizen, not a special case, since it may win the A/B/C
    experiment.
    """

    @abstractmethod
    def rerank(
        self, query: Query, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        raise NotImplementedError


class NoOpReranker(Reranker):
    """Passthrough reranker — the null hypothesis for the A/B/C experiment."""

    def rerank(
        self, query: Query, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        return results[:top_k]
