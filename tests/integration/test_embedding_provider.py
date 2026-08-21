"""Integration tests for BgeM3EmbeddingProvider.

These tests require sentence-transformers and the bge-m3 model to be
available. They may be slow on first run (model download/loading).

Run with: pytest tests/integration/test_embedding_provider.py -v
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from embeddings.bge_m3 import BgeM3EmbeddingProvider

# Skip entire module if sentence-transformers not installed
try:
    from sentence_transformers import SentenceTransformer  # noqa: F401
    HAS_ST = True
except ImportError:
    HAS_ST = False

pytestmark = pytest.mark.skipif(not HAS_ST, reason="sentence-transformers not installed")


@pytest.fixture(scope="module")
def provider() -> BgeM3EmbeddingProvider:
    """Create a shared bge-m3 provider (loaded once per test module)."""
    return BgeM3EmbeddingProvider(device="cpu")


class TestBgeM3Dimension:
    """Verify embedding dimension matches expected 1024."""

    def test_dimension_property(self, provider: BgeM3EmbeddingProvider) -> None:
        assert provider.dimension == 1024

    def test_embed_query_returns_correct_dim(self, provider: BgeM3EmbeddingProvider) -> None:
        vec = provider.embed_query("What is AI?", lang="en")
        assert len(vec) == 1024
        assert all(isinstance(v, float) for v in vec)

    def test_embed_passages_returns_correct_dim(self, provider: BgeM3EmbeddingProvider) -> None:
        vectors = provider.embed_passages(["Hello", "World"])
        assert len(vectors) == 2
        assert all(len(v) == 1024 for v in vectors)

    def test_embedding_is_normalized(self, provider: BgeM3EmbeddingProvider) -> None:
        """With normalize=True, vectors should be approximately unit length."""
        vec = provider.embed_query("test query", lang="en")
        norm = sum(v * v for v in vec) ** 0.5
        assert abs(norm - 1.0) < 0.01, f"Expected unit vector, got norm={norm}"


class TestBgeM3Multilingual:
    """Verify bge-m3 works across languages."""

    def test_hindi_embedding(self, provider: BgeM3EmbeddingProvider) -> None:
        vec = provider.embed_query("यह एक प्रश्न है", lang="hi")
        assert len(vec) == 1024

    def test_english_embedding(self, provider: BgeM3EmbeddingProvider) -> None:
        vec = provider.embed_query("This is a question", lang="en")
        assert len(vec) == 1024

    def test_cross_lingual_similarity(self, provider: BgeM3EmbeddingProvider) -> None:
        """Translations of the same query should be more similar than unrelated queries."""
        hi_vec = provider.embed_query("भारत की राजधानी क्या है?", lang="hi")
        en_vec = provider.embed_query("What is the capital of India?", lang="en")
        unrelated = provider.embed_query("Tell me about quantum physics", lang="en")

        # Cosine similarity (vectors are normalized)
        sim_related = sum(a * b for a, b in zip(hi_vec, en_vec, strict=False))
        sim_unrelated = sum(a * b for a, b in zip(hi_vec, unrelated, strict=False))

        assert sim_related > sim_unrelated, (
            f"Cross-lingual similarity ({sim_related:.4f}) should be "
            f"greater than unrelated ({sim_unrelated:.4f})"
        )


class TestBgeM3BatchEmbedding:
    """Test batch embedding for corpus indexing."""

    def test_batch_matches_individual(self, provider: BgeM3EmbeddingProvider) -> None:
        """Batch embedding should produce same vectors as individual embedding."""
        texts = ["Alpha", "Beta", "Gamma"]
        batch_vecs = provider.embed_passages(texts)

        individual_vecs = []
        for text in texts:
            individual_vecs.append(provider.embed_query(text, lang="en"))

        for batch_vec, ind_vec in zip(batch_vecs, individual_vecs, strict=False):
            for b, i in zip(batch_vec, ind_vec, strict=False):
                assert abs(b - i) < 1e-5, "Batch and individual embeddings should match"

    def test_empty_input(self, provider: BgeM3EmbeddingProvider) -> None:
        result = provider.embed_passages([])
        assert result == []

    def test_large_batch(self, provider: BgeM3EmbeddingProvider) -> None:
        """Embedding 100 texts should work without error."""
        texts = [f"This is passage number {i}" for i in range(100)]
        vectors = provider.embed_passages(texts)
        assert len(vectors) == 100
        assert all(len(v) == 1024 for v in vectors)


class TestBgeM3Checkpointing:
    """Test checkpointing/caching of embeddings."""

    def test_save_and_load(self, provider: BgeM3EmbeddingProvider) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cached_provider = type(provider)(
                device="cpu",
                cache_dir=Path(tmpdir),
            )
            vec = cached_provider.embed_query("Test text", lang="en")
            cached_provider.save_cached_embedding("test_pid", vec)

            loaded = cached_provider.load_cached_embedding("test_pid")
            assert loaded is not None
            assert len(loaded) == 1024
            for a, b in zip(vec, loaded, strict=False):
                assert abs(a - b) < 1e-6

    def test_load_missing_returns_none(self, provider: BgeM3EmbeddingProvider) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cached_provider = type(provider)(
                device="cpu",
                cache_dir=Path(tmpdir),
            )
            assert cached_provider.load_cached_embedding("nonexistent") is None

    def test_checkpointed_embedding_matches(self, provider: BgeM3EmbeddingProvider) -> None:
        """embed_passages_with_checkpointing should match embed_passages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cached_provider = type(provider)(
                device="cpu",
                cache_dir=Path(tmpdir),
            )
            texts = ["Passage A", "Passage B"]
            pids = ["pid_A", "pid_B"]

            # First call: compute and cache
            vecs1 = cached_provider.embed_passages_with_checkpointing(pids, texts)

            # Second call: should use cache
            vecs2 = cached_provider.embed_passages_with_checkpointing(pids, texts)

            assert len(vecs1) == len(vecs2) == 2
            for v1, v2 in zip(vecs1, vecs2, strict=False):
                for a, b in zip(v1, v2, strict=False):
                    assert abs(a - b) < 1e-6
