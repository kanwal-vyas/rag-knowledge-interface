"""
Dense retriever — implements the DenseRetriever interface.

Wraps QdrantIndexManager + BgeM3EmbeddingProvider behind the Retriever ABC
so dense retrieval can be benchmarked interchangeably with BM25 and hybrid.

Architecture alignment:
- ADR-0003: dense = bge-m3
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
from embeddings.bge_m3 import BgeM3EmbeddingProvider
from retrieval.dense.qdrant_index import QdrantIndexManager

logger = structlog.get_logger(__name__)


class BgeM3DenseRetriever:
    """
    DenseRetriever backed by bge-m3 embeddings and Qdrant.

    Usage:
        retriever = BgeM3DenseRetriever(
            qdrant_manager=qdrant_manager,
            embedding_provider=embedding_provider,
        )
        results = retriever.retrieve(query, mode=RetrievalMode.CROSS_LINGUAL, top_k=10)
    """

    def __init__(
        self,
        qdrant_manager: QdrantIndexManager,
        embedding_provider: BgeM3EmbeddingProvider,
    ) -> None:
        self._qdrant = qdrant_manager
        self._embedder = embedding_provider

    def retrieve(
        self,
        query: Query,
        mode: RetrievalMode,
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        Retrieve top_k results using bge-m3 dense vector search.

        Args:
            query: The query object (query_text + lang).
            mode: MONOLINGUAL (filter to query.lang) or CROSS_LINGUAL (no filter).
            top_k: Number of results to return.
        """
        # Embed the query
        query_vector = self._embedder.embed_query(query.query_text, query.lang.value)

        # Determine language filter
        lang_filter: str | None = None
        if mode == RetrievalMode.MONOLINGUAL:
            lang_filter = query.lang.value

        start_time = time.perf_counter()

        # Search Qdrant
        hits = self._qdrant.search(
            query_vector=query_vector,
            top_k=top_k,
            lang_filter=lang_filter,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Convert to RetrievalResult
        results: list[RetrievalResult] = []
        for hit in hits:
            try:
                lang_enum = Language(hit["lang"])
            except (ValueError, KeyError):
                logger.warning(
                    "dense_unknown_language",
                    passage_id=hit.get("passage_id", ""),
                    lang=hit.get("lang", ""),
                )
                continue

            passage = Passage(
                passage_id=hit["passage_id"],
                text=hit["text"],
                lang=lang_enum,
            )
            results.append(RetrievalResult(
                passage=passage,
                score=hit["score"],
                source="dense",
            ))

        logger.debug(
            "dense_retrieval",
            query_lang=query.lang.value,
            mode=mode.value,
            results_count=len(results),
            latency_ms=f"{elapsed_ms:.2f}",
        )

        return results

    def retrieve_with_latency(
        self,
        query: Query,
        mode: RetrievalMode,
        top_k: int,
    ) -> tuple[list[RetrievalResult], float]:
        """Retrieve with explicit latency measurement (embed + search combined)."""
        start_time = time.perf_counter()
        results = self.retrieve(query, mode, top_k)
        latency_ms = (time.perf_counter() - start_time) * 1000
        return results, latency_ms
