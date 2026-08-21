"""
BM25 sparse retriever — implements the SparseRetriever interface.

Maps between the repository's retrieval models (Query, RetrievalResult,
RetrievalMode) and the BM25Index implementation.

Architecture alignment:
- ADR-0003: sparse = BM25, NOT bge-m3 sparse output
- Audit §7: Config 1 (MONOLINGUAL) filters to query.lang
             Config 2 (CROSS_LINGUAL) retrieves across all indexed languages
"""
from __future__ import annotations

import time

import structlog

from app.models.retrieval import (
    Language,
    Passage,
    Query,
    RetrievalMode,
    RetrievalResult,
)
from retrieval.sparse.bm25_index import BM25Index

logger = structlog.get_logger(__name__)


class BM25SparseRetriever:
    """
    SparseRetriever backed by a BM25Index.

    Usage:
        retriever = BM25SparseRetriever(index=bm25_index)
        results = retriever.retrieve(query, mode=RetrievalMode.MONOLINGUAL, top_k=10)

    Config 1 (MONOLINGUAL): query.lang → filter results to query.lang
    Config 2 (CROSS_LINGUAL): no language filter — retrieve from all languages
    """

    def __init__(self, index: BM25Index) -> None:
        """
        Args:
            index: A built BM25Index instance.
        """
        if not index.is_built:
            raise ValueError("BM25Index must be built before creating a retriever.")
        self._index = index

    def retrieve(
        self,
        query: Query,
        mode: RetrievalMode,
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        Retrieve top_k results for a query using BM25.

        Args:
            query: The query object (query_text + lang).
            mode: MONOLINGUAL (filter to query.lang) or CROSS_LINGUAL (no filter).
            top_k: Number of results to return.

        Returns:
            Ranked list of RetrievalResult objects.
        """
        # Determine language filter based on mode
        lang_filter: str | None = None
        if mode == RetrievalMode.MONOLINGUAL:
            lang_filter = query.lang.value

        start_time = time.perf_counter()

        bm25_results = self._index.search(
            query=query.query_text,
            top_k=top_k,
            lang_filter=lang_filter,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Convert BM25 results to RetrievalResult
        retrieval_results: list[RetrievalResult] = []
        for bm25_result in bm25_results:
            doc = self._index.get_document(bm25_result.passage_id)
            if doc is None:
                logger.warning(
                    "bm25_document_not_found",
                    passage_id=bm25_result.passage_id,
                )
                continue

            try:
                lang_enum = Language(doc.lang)
            except ValueError:
                logger.warning(
                    "bm25_unknown_language",
                    passage_id=doc.passage_id,
                    lang=doc.lang,
                )
                continue

            passage = Passage(
                passage_id=doc.passage_id,
                text=doc.text,
                lang=lang_enum,
            )
            retrieval_results.append(RetrievalResult(
                passage=passage,
                score=bm25_result.score,
                source="bm25",
            ))

        logger.debug(
            "bm25_retrieval",
            query_lang=query.lang.value,
            mode=mode.value,
            results_count=len(retrieval_results),
            latency_ms=f"{elapsed_ms:.2f}",
        )

        return retrieval_results

    def retrieve_with_latency(
        self,
        query: Query,
        mode: RetrievalMode,
        top_k: int,
    ) -> tuple[list[RetrievalResult], float]:
        """
        Retrieve with explicit latency measurement.

        Returns:
            (results, latency_ms)
        """
        start_time = time.perf_counter()
        results = self.retrieve(query, mode, top_k)
        latency_ms = (time.perf_counter() - start_time) * 1000
        return results, latency_ms
