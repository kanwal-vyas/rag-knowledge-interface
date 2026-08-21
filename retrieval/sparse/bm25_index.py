"""
BM25 sparse retrieval index.

Implements a BM25 index over the production corpus using the `rank_bm25`
library. Designed for multilingual retrieval across all 14 dataset languages.

Architecture alignment:
- ADR-0003: sparse retrieval = BM25 (NOT bge-m3 sparse output)
- Audit §5: production payload only — no evaluation fields
- Audit §7: Config 1 (monolingual) vs Config 2 (cross-lingual) via lang filter
- Phase 5 tokenization: multilingual word + bigram tokenization

IMPORTANT LIMITATION (documented per project principle):
BM25's effectiveness varies significantly across languages. English BM25
with IDF weighting works well because English has rich vocabulary diversity.
Indic scripts with more morphological variation may show lower BM25 recall.
This is a known limitation — the retrieval evaluation (Phase 9) will
quantify it rather than assume it.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import structlog

from ingestion.normalization.tokenize import tokenize_for_lang

logger = structlog.get_logger(__name__)

# Try importing rank_bm25; fall back gracefully for test environments
try:
    from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


@dataclass
class BM25Document:
    """A document in the BM25 index — production-safe fields only."""

    passage_id: str  # Canonical content-hash ID
    text: str        # Normalized passage text
    lang: str        # ISO 639-1 language code


@dataclass
class BM25Result:
    """A single BM25 search result."""

    passage_id: str
    score: float
    rank: int


@dataclass
class BM25Stats:
    """Statistics about the BM25 index."""

    document_count: int
    language_counts: dict[str, int]
    vocab_size: int
    build_time_ms: float
    index_size_estimate: str


class BM25Index:
    """
    In-memory BM25 index with multilingual tokenization.

    Usage:
        index = BM25Index()
        index.add_document("pid_1", "Some passage text", "en")
        index.add_document("pid_2", "कुछ पाठ पाठ", "hi")
        index.build()

        results = index.search("what is AI", top_k=10)
        filtered = index.search("capital of India", top_k=10, lang_filter="hi")

    IMPORTANT: This is an in-memory index. For production with millions of
    passages, consider a persistent BM25 index (e.g., Elasticsearch, or
    a serialized rank_bm25 index). The in-memory approach is correct for
    the development/experimentation phase.
    """

    def __init__(self, use_bigrams: bool = True) -> None:
        """
        Args:
            use_bigrams: Whether to use character bigram supplementation.
                        Set True for better agglutinative script support.
        """
        self._use_bigrams = use_bigrams
        self._documents: list[BM25Document] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._built = False

        # For language-filtered search: maps passage index → language
        self._lang_index: dict[str, list[int]] = {}  # lang → list of doc indices

    @property
    def is_built(self) -> bool:
        return self._built

    def add_document(self, passage_id: str, text: str, lang: str) -> None:
        """
        Add a document to the index.

        Must be called before build(). After build(), no documents can be added.

        Args:
            passage_id: Canonical passage ID (production payload only).
            text: Passage text (should be pre-normalized).
            lang: ISO 639-1 language code.
        """
        if self._built:
            raise RuntimeError("Cannot add documents after index is built. Recreate the index.")

        idx = len(self._documents)
        self._documents.append(BM25Document(
            passage_id=passage_id,
            text=text,
            lang=lang,
        ))

        # Track language → index mapping for filtered search
        if lang not in self._lang_index:
            self._lang_index[lang] = []
        self._lang_index[lang].append(idx)

    def build(self) -> BM25Stats:
        """
        Build the BM25 index from added documents.

        Returns:
            BM25Stats with build statistics.
        """
        if not HAS_BM25:
            raise ImportError(
                "rank_bm25 is required for BM25Index. "
                "Install with: pip install rank_bm25"
            )

        if self._built:
            raise RuntimeError("Index already built. Recreate to rebuild.")

        start_time = time.perf_counter()

        # Tokenize all documents
        self._tokenized_corpus = [
            tokenize_for_lang(doc.text, doc.lang)
            for doc in self._documents
        ]

        # Build BM25 index
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        self._built = True

        build_time_ms = (time.perf_counter() - start_time) * 1000

        # Compute stats
        vocab = set()
        for tokens in self._tokenized_corpus:
            vocab.update(tokens)

        lang_counts: dict[str, int] = {}
        for doc in self._documents:
            lang_counts[doc.lang] = lang_counts.get(doc.lang, 0) + 1

        stats = BM25Stats(
            document_count=len(self._documents),
            language_counts=lang_counts,
            vocab_size=len(vocab),
            build_time_ms=build_time_ms,
            index_size_estimate=f"{len(self._documents)} docs, {len(vocab)} terms",
        )

        logger.info(
            "bm25_index_built",
            documents=stats.document_count,
            vocab_size=stats.vocab_size,
            build_time_ms=f"{build_time_ms:.1f}",
            languages=list(lang_counts.keys()),
        )

        return stats

    def search(
        self,
        query: str,
        top_k: int = 10,
        lang_filter: str | None = None,
    ) -> list[BM25Result]:
        """
        Search the BM25 index.

        Args:
            query: Query text (will be tokenized).
            top_k: Number of results to return.
            lang_filter: If set, filter results to this language (ISO 639-1).

        Returns:
            List of BM25Result ranked by BM25 score (descending).
            If query tokenization produces no tokens, returns empty list.
        """
        if not self._built:
            raise RuntimeError("Index not built. Call build() first.")

        assert self._bm25 is not None

        # Tokenize query (use the first document's language as fallback,
        # but queries should be tokenized with their own language)
        query_lang = lang_filter or "en"
        query_tokens = tokenize_for_lang(query, query_lang)

        if not query_tokens:
            return []

        # Get raw BM25 scores for all documents
        scores = self._bm25.get_scores(query_tokens)

        # Apply language filter if specified
        if lang_filter is not None:
            allowed_indices = set(self._lang_index.get(lang_filter, []))
            # Mask scores for documents not in the allowed language
            filtered_scores = [
                (i, scores[i]) for i in range(len(scores))
                if i in allowed_indices
            ]
        else:
            filtered_scores = [(i, scores[i]) for i in range(len(scores))]

        # Sort by score descending, take top_k
        filtered_scores.sort(key=lambda x: x[1], reverse=True)
        top_results = filtered_scores[:top_k]

        return [
            BM25Result(
                passage_id=self._documents[idx].passage_id,
                score=float(score),
                rank=rank + 1,
            )
            for rank, (idx, score) in enumerate(top_results)
            if score > 0  # Only return positive-scoring results
        ]

    def get_document(self, passage_id: str) -> BM25Document | None:
        """Look up a document by passage_id."""
        for doc in self._documents:
            if doc.passage_id == passage_id:
                return doc
        return None

    def get_documents_by_ids(self, passage_ids: list[str]) -> list[BM25Document]:
        """Look up multiple documents by passage_id."""
        id_set = set(passage_ids)
        return [doc for doc in self._documents if doc.passage_id in id_set]

    def get_stats(self) -> dict:
        """Get index statistics without rebuilding."""
        lang_counts: dict[str, int] = {}
        for doc in self._documents:
            lang_counts[doc.lang] = lang_counts.get(doc.lang, 0) + 1
        return {
            "document_count": len(self._documents),
            "language_counts": lang_counts,
            "is_built": self._built,
            "has_bm25": self._bm25 is not None,
        }

    # -------------------------------------------------------------------
    # Persistence (JSON-based, for small development indices)
    # -------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """
        Save the index to disk (documents only — BM25 model is rebuilt on load).

        This is suitable for small development indices. For production-scale
        persistence, a dedicated search engine (Elasticsearch, etc.) would be
        more appropriate.
        """
        data = {
            "use_bigrams": self._use_bigrams,
            "documents": [
                {"passage_id": doc.passage_id, "text": doc.text, "lang": doc.lang}
                for doc in self._documents
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        logger.info("bm25_index_saved", path=str(path), documents=len(self._documents))

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        """
        Load an index from disk and rebuild the BM25 model.

        Args:
            path: Path to the saved JSON index.

        Returns:
            A BM25Index with the loaded documents and rebuilt BM25 model.
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        index = cls(use_bigrams=data["use_bigrams"])
        for doc in data["documents"]:
            index.add_document(doc["passage_id"], doc["text"], doc["lang"])
        index.build()
        logger.info("bm25_index_loaded", path=str(path), documents=len(index._documents))
        return index
