"""
BGE-M3 embedding provider implementation.

bge-m3 is confirmed (Audit §4/§7) as THE embedding model for this
architecture. This implementation:

- Uses sentence-transformers for local inference (no API key needed)
- Produces dense vectors of dimension 1024 (verified from model config)
- Supports batch embedding for offline corpus indexing
- Supports single query embedding for online retrieval
- Normalizes embeddings by default (required for cosine similarity in Qdrant)
- Supports checkpointing to avoid re-embedding unchanged passages
- Uses deterministic passage→vector association

ARCHITECTURE DETAIL MISSING — REQUIRES CONFIRMATION:
- Whether bge-m3's dense, sparse, or multi-vector output mode should be
  used for the primary retrieval path. This implementation uses the dense
  mode (1024-dim vectors), which is the most common choice for Qdrant-
  backed retrieval. The sparse output from bge-m3 could potentially
  replace the separate BM25 component, but that is a different
  architectural decision not confirmed in the available source.
"""
from __future__ import annotations

import json
from pathlib import Path

import structlog

from embeddings.base import EmbeddingProvider
from embeddings.device import detect_device, log_device_report

logger = structlog.get_logger(__name__)

# bge-m3 dense embedding dimension (verified from model config: hidden_size=1024)
BGE_M3_DENSE_DIMENSION = 1024


class BgeM3EmbeddingProvider(EmbeddingProvider):
    """
    Local bge-m3 embedding provider via sentence-transformers.

    Usage:
        provider = BgeM3EmbeddingProvider()
        # or with checkpointing:
        provider = BgeM3EmbeddingProvider(cache_dir=Path("data/embeddings_cache"))

        vectors = provider.embed_passages(["text1", "text2"])
        query_vec = provider.embed_query("query text", lang="hi")
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
        batch_size: int = 32,
        normalize: bool = True,
        cache_dir: Path | None = None,
    ) -> None:
        """
        Args:
            model_name: HuggingFace model identifier.
            device: "cpu", "cuda", "mps", or None for auto-detect.
            batch_size: Number of texts per embedding batch.
            normalize: Whether to L2-normalize vectors (recommended for cosine).
            cache_dir: If set, enables checkpointing — embeddings are saved
                      per-passage_id and skipped on re-runs.
        """
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._normalize = normalize
        self._cache_dir = cache_dir
        self._model = None  # Lazy-loaded

    def _get_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            import time

            from sentence_transformers import SentenceTransformer

            # Auto-detect device if not explicitly set
            effective_device = detect_device(self._device)
            self._device = effective_device

            logger.info(
                "loading_embedding_model",
                model=self._model_name,
                device=effective_device,
            )
            log_device_report(prefix="bge_m3")

            load_start = time.perf_counter()
            self._model = SentenceTransformer(
                self._model_name,
                device=effective_device,
            )
            load_time = time.perf_counter() - load_start

            logger.info(
                "embedding_model_loaded",
                model=self._model_name,
                dimension=self.dimension,
                device=effective_device,
                load_time_s=round(load_time, 2),
                dtype=str(self._model[0].auto_model.config.dtype)
                if hasattr(self._model, "__getitem__")
                else "unknown",
            )
        return self._model

    @property
    def dimension(self) -> int:
        return BGE_M3_DENSE_DIMENSION

    def embed_query(self, text: str, lang: str) -> list[float]:
        """
        Embed a single query string. Returns a dense vector.

        Args:
            text: Query text (will be normalized before embedding).
            lang: ISO 639-1 language code (metadata, not used for embedding
                  since bge-m3 is inherently multilingual).
        """
        model = self._get_model()
        from ingestion.normalization.text import normalize_query

        normalized = normalize_query(text)
        vector = model.encode(
            [normalized],
            normalize_embeddings=self._normalize,
            batch_size=1,
            show_progress_bar=False,
        )
        return vector[0].tolist()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """
        Batch-embed passage texts for indexing (offline path).

        Args:
            texts: List of passage texts to embed.

        Returns:
            List of dense vectors, one per input text.
        """
        if not texts:
            return []

        model = self._get_model()
        from ingestion.normalization.text import normalize_text

        normalized = [normalize_text(t) for t in texts]
        vectors = model.encode(
            normalized,
            normalize_embeddings=self._normalize,
            batch_size=self._batch_size,
            show_progress_bar=len(texts) > 100,
        )
        return [v.tolist() for v in vectors]

    # -----------------------------------------------------------------------
    # Checkpointing support
    # -----------------------------------------------------------------------

    def _cache_path(self, passage_id: str) -> Path | None:
        """Get the cache file path for a passage_id."""
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{passage_id}.json"

    def load_cached_embedding(self, passage_id: str) -> list[float] | None:
        """Load a cached embedding if available."""
        path = self._cache_path(passage_id)
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data["vector"]
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def save_cached_embedding(self, passage_id: str, vector: list[float]) -> None:
        """Save an embedding to the checkpoint cache."""
        path = self._cache_path(passage_id)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"passage_id": passage_id, "vector": vector}),
            encoding="utf-8",
        )

    def embed_passages_with_checkpointing(
        self,
        passage_ids: list[str],
        texts: list[str],
    ) -> list[list[float]]:
        """
        Embed passages with checkpointing — skip already-embedded passages.

        Args:
            passage_ids: Canonical passage_ids (for cache keys).
            texts: Corresponding passage texts.

        Returns:
            List of vectors, one per input. Cached vectors are returned
            directly; uncached vectors are computed and saved.
        """
        assert len(passage_ids) == len(texts), "passage_ids and texts must have same length"

        results: list[list[float] | None] = [None] * len(texts)
        to_embed_indices: list[int] = []
        to_embed_texts: list[str] = []

        # Load from cache where possible
        for i, pid in enumerate(passage_ids):
            cached = self.load_cached_embedding(pid)
            if cached is not None and len(cached) == self.dimension:
                results[i] = cached
            else:
                to_embed_indices.append(i)
                to_embed_texts.append(texts[i])

        # Embed remaining
        if to_embed_texts:
            logger.info(
                "embedding_batch",
                cached=len(texts) - len(to_embed_texts),
                to_embed=len(to_embed_texts),
            )
            new_vectors = self.embed_passages(to_embed_texts)
            for idx, vector in zip(to_embed_indices, new_vectors, strict=False):
                results[idx] = vector
                self.save_cached_embedding(passage_ids[idx], vector)
        else:
            logger.info("all_embeddings_cached", count=len(texts))

        # Cast: by this point all entries are populated
        return [v for v in results if v is not None]  # type: ignore[misc]
