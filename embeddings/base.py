"""
EmbeddingProvider interface.

bge-m3 is confirmed (Audit §4/§7) as THE embedding model for this
architecture — this is not a menu of interchangeable embedding models to
benchmark against each other, unlike retrievers/rerankers. The interface
still exists so ingestion and query-time code depend on an abstraction
rather than a concrete library call, and so a local vs. hosted bge-m3
serving strategy can be swapped without touching callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Produces dense embeddings for query and passage text."""

    @abstractmethod
    def embed_query(self, text: str, lang: str) -> list[float]:
        """Embed a single query string. Returns a dense vector."""
        raise NotImplementedError

    @abstractmethod
    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed passage texts for indexing (offline path)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Embedding vector dimensionality.

        ARCHITECTURE DETAIL MISSING — REQUIRES CONFIRMATION
        bge-m3's dense output dimensionality choice (it supports multiple
        output modes) is not specified in the available architecture
        source. Do not hardcode a value in an implementation until
        confirmed.
        """
        raise NotImplementedError
