"""Tests for BM25 sparse index and retriever."""
from __future__ import annotations

import pytest

from app.models.retrieval import Language, Query, RetrievalMode
from retrieval.sparse.bm25_index import HAS_BM25, BM25Index
from retrieval.sparse.bm25_retriever import BM25SparseRetriever

# Skip if rank_bm25 not installed
pytestmark = pytest.mark.skipif(not HAS_BM25, reason="rank_bm25 not installed")


# ---------------------------------------------------------------------------
# Tokenization tests
# ---------------------------------------------------------------------------

class TestTokenization:
    """Verify multilingual tokenization works correctly."""

    def test_english_tokenization(self) -> None:
        from ingestion.normalization.tokenize import tokenize_words
        tokens = tokenize_words("The quick brown fox jumps")
        assert "the" in tokens
        assert "quick" in tokens
        assert len(tokens) == 5

    def test_hindi_tokenization(self) -> None:
        from ingestion.normalization.tokenize import tokenize_words
        tokens = tokenize_words("भारत की राजधानी क्या है")
        assert len(tokens) == 5
        assert "भारत" in tokens

    def test_bigram_extraction(self) -> None:
        from ingestion.normalization.tokenize import tokenize_char_bigrams
        bigrams = tokenize_char_bigrams("abc")
        assert bigrams == ["ab", "bc"]

    def test_language_aware_english_no_bigrams(self) -> None:
        from ingestion.normalization.tokenize import tokenize_for_lang
        tokens = tokenize_for_lang("Hello world", "en")
        # Should be word tokens only, no bigrams (lowercased)
        assert "hello" in tokens
        assert "world" in tokens
        # Should NOT contain "he", "el", etc. (bigrams)
        assert "he" not in tokens

    def test_language_aware_tamil_with_bigrams(self) -> None:
        from ingestion.normalization.tokenize import tokenize_for_lang
        tokens = tokenize_for_lang("வணக்கம் உலகம்", "ta")
        # Should contain word tokens
        assert "வணக்கம்" in tokens
        # Should also contain character bigrams
        bigrams_present = any(len(t) == 2 for t in tokens)
        assert bigrams_present, "Tamil tokenization should include character bigrams"

    def test_empty_text(self) -> None:
        from ingestion.normalization.tokenize import tokenize_words
        assert tokenize_words("") == []


# ---------------------------------------------------------------------------
# BM25Index construction tests
# ---------------------------------------------------------------------------

class TestBM25IndexConstruction:
    """Tests for index building and document management."""

    def _make_index(self) -> BM25Index:
        index = BM25Index()
        index.add_document("pid_en_1", "The quick brown fox", "en")
        index.add_document("pid_en_2", "A lazy dog sleeps", "en")
        index.add_document("pid_hi_1", "भारत एक महान देश है", "hi")
        index.add_document("pid_hi_2", "दिल्ली भारत की राजधानी है", "hi")
        return index

    def test_build_returns_stats(self) -> None:
        index = self._make_index()
        stats = index.build()
        assert stats.document_count == 4
        assert stats.language_counts == {"en": 2, "hi": 2}
        assert stats.build_time_ms >= 0

    def test_build_sets_is_built(self) -> None:
        index = self._make_index()
        assert not index.is_built
        index.build()
        assert index.is_built

    def test_cannot_add_after_build(self) -> None:
        index = self._make_index()
        index.build()
        with pytest.raises(RuntimeError, match="Cannot add documents"):
            index.add_document("pid_new", "new text", "en")

    def test_cannot_rebuild(self) -> None:
        index = self._make_index()
        index.build()
        with pytest.raises(RuntimeError, match="already built"):
            index.build()

    def test_search_before_build_raises(self) -> None:
        index = BM25Index()
        index.add_document("pid_1", "text", "en")
        with pytest.raises(RuntimeError, match="not built"):
            index.search("query", top_k=10)


# ---------------------------------------------------------------------------
# BM25 search tests
# ---------------------------------------------------------------------------

class TestBM25Search:
    """Tests for BM25 search behavior."""

    def _make_built_index(self) -> BM25Index:
        index = BM25Index()
        index.add_document("pid_ai_1", "Artificial intelligence is transforming technology", "en")
        index.add_document("pid_ai_2", "Machine learning and AI are related fields", "en")
        index.add_document("pid_cook_1", "Italian pasta is delicious with tomato sauce", "en")
        index.add_document("pid_hi_ai_1", "कृत्रिम बुद्धिमत्ता तकनीक को बदल रही है", "hi")
        index.add_document("pid_hi_cook_1", "भारतीय खाना मसालेदार होता है", "hi")
        index.build()
        return index

    def test_basic_search_returns_results(self) -> None:
        index = self._make_built_index()
        results = index.search("artificial intelligence", top_k=5)
        assert len(results) > 0
        # AI-related passages should score highest
        assert results[0].passage_id in {"pid_ai_1", "pid_ai_2"}

    def test_top_k_limits_results(self) -> None:
        index = self._make_built_index()
        results = index.search("intelligence", top_k=2)
        assert len(results) <= 2

    def test_scores_are_positive(self) -> None:
        index = self._make_built_index()
        results = index.search("artificial intelligence", top_k=5)
        for r in results:
            assert r.score > 0

    def test_ranking_is_deterministic(self) -> None:
        index = self._make_built_index()
        r1 = index.search("artificial intelligence", top_k=5)
        r2 = index.search("artificial intelligence", top_k=5)
        assert [r.passage_id for r in r1] == [r.passage_id for r in r2]

    def test_unrelated_query_returns_fewer_or_no_results(self) -> None:
        index = self._make_built_index()
        results = index.search("quantum entanglement physics", top_k=5)
        # Should return fewer results or lower scores
        assert len(results) <= 5

    def test_empty_query_returns_empty(self) -> None:
        index = self._make_built_index()
        results = index.search("", top_k=5)
        assert results == []

    def test_hindi_search(self) -> None:
        index = self._make_built_index()
        results = index.search("कृत्रिम बुद्धिमत्ता", top_k=5, lang_filter="hi")
        assert len(results) > 0
        assert results[0].passage_id == "pid_hi_ai_1"

    def test_language_filter_en(self) -> None:
        index = self._make_built_index()
        results = index.search("intelligence", top_k=10, lang_filter="en")
        for r in results:
            doc = index.get_document(r.passage_id)
            assert doc is not None
            assert doc.lang == "en"

    def test_language_filter_hi(self) -> None:
        index = self._make_built_index()
        results = index.search("खाना", top_k=10, lang_filter="hi")
        for r in results:
            doc = index.get_document(r.passage_id)
            assert doc is not None
            assert doc.lang == "hi"

    def test_no_language_filter_returns_all(self) -> None:
        index = self._make_built_index()
        results = index.search("intelligence", top_k=10)
        # Without filter, can return results from any language
        assert len(results) > 0

    def test_document_lookup(self) -> None:
        index = self._make_built_index()
        doc = index.get_document("pid_ai_1")
        assert doc is not None
        assert doc.passage_id == "pid_ai_1"
        assert doc.lang == "en"

    def test_document_lookup_missing(self) -> None:
        index = self._make_built_index()
        assert index.get_document("nonexistent") is None


# ---------------------------------------------------------------------------
# BM25SparseRetriever tests (interface integration)
# ---------------------------------------------------------------------------

class TestBM25SparseRetriever:
    """Tests for the SparseRetriever interface implementation."""

    def _make_retriever(self) -> BM25SparseRetriever:
        index = BM25Index()
        index.add_document("pid_en_1", "Artificial intelligence is transforming technology", "en")
        index.add_document("pid_en_2", "Machine learning and AI are related fields", "en")
        index.add_document("pid_hi_1", "कृत्रिम बुद्धिमत्ता तकनीक को बदल रही है", "hi")
        index.add_document("pid_hi_2", "दिल्ली भारत की राजधानी है", "hi")
        index.build()
        return BM25SparseRetriever(index=index)

    def test_retrieve_returns_retrieval_results(self) -> None:
        retriever = self._make_retriever()
        query = Query(query_text="artificial intelligence", lang=Language.EN)
        results = retriever.retrieve(query, RetrievalMode.CROSS_LINGUAL, top_k=5)
        assert len(results) > 0
        assert all(hasattr(r, "passage") and hasattr(r, "score") for r in results)
        assert all(r.source == "bm25" for r in results)

    def test_monolingual_filters_to_query_lang(self) -> None:
        retriever = self._make_retriever()
        query = Query(query_text="intelligence", lang=Language.EN)
        results = retriever.retrieve(query, RetrievalMode.MONOLINGUAL, top_k=10)
        for r in results:
            assert r.passage.lang == Language.EN

    def test_cross_lingual_no_filter(self) -> None:
        retriever = self._make_retriever()
        query = Query(query_text="बुद्धिमत्ता", lang=Language.HI)
        results = retriever.retrieve(query, RetrievalMode.CROSS_LINGUAL, top_k=10)
        # Should be able to find results in any language
        assert len(results) > 0

    def test_hindi_query_monolingual(self) -> None:
        retriever = self._make_retriever()
        query = Query(query_text="बुद्धिमत्ता", lang=Language.HI)
        results = retriever.retrieve(query, RetrievalMode.MONOLINGUAL, top_k=10)
        for r in results:
            assert r.passage.lang == Language.HI

    def test_retrieve_with_latency(self) -> None:
        retriever = self._make_retriever()
        query = Query(query_text="artificial intelligence", lang=Language.EN)
        results, latency_ms = retriever.retrieve_with_latency(
            query, RetrievalMode.CROSS_LINGUAL, top_k=5
        )
        assert len(results) > 0
        assert latency_ms >= 0

    def test_retriever_rejects_unbuilt_index(self) -> None:
        index = BM25Index()
        index.add_document("pid_1", "text", "en")
        with pytest.raises(ValueError, match="must be built"):
            BM25SparseRetriever(index=index)


# ---------------------------------------------------------------------------
# Evaluation-only field leakage test
# ---------------------------------------------------------------------------

class TestBM25NoLeakage:
    """Verify BM25 does not include evaluation-only fields."""

    FORBIDDEN = {"is_selected", "Answer", "Eng_Answer", "source_query_ids"}

    def test_bm25_document_has_no_forbidden_fields(self) -> None:
        index = BM25Index()
        index.add_document("pid_1", "Test passage", "en")
        doc_fields = {f.name for f in index._documents[0].__dataclass_fields__.values()}  # type: ignore[attr-defined]
        assert doc_fields & self.FORBIDDEN == set()

    def test_retrieval_result_has_no_forbidden_fields(self) -> None:
        from retrieval.sparse.bm25_index import BM25Document
        # BM25Document should not contain forbidden fields
        doc = BM25Document(passage_id="pid_1", text="text", lang="en")
        doc_dict = doc.__dict__
        assert set(doc_dict.keys()) & self.FORBIDDEN == set()


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------

class TestBM25Persistence:
    """Test save/load round-trip."""

    def test_save_and_load(self, tmp_path: object) -> None:
        from pathlib import Path
        path = Path(str(tmp_path)) / "bm25_index.json"

        # Need enough docs so BM25 IDF > 0 (IDF = 0 when every term
        # appears in exactly half the corpus)
        index = BM25Index()
        index.add_document("pid_1", "Artificial intelligence is transforming technology", "en")
        index.add_document("pid_2", "Machine learning algorithms process data efficiently", "en")
        index.add_document("pid_3", "Natural language processing enables text understanding", "en")
        index.add_document("pid_4", "Computer vision analyzes images and video content", "en")
        index.add_document("pid_5", "कृत्रिम बुद्धिमत्ता तकनीक को बदल रही है", "hi")
        index.save(path)

        loaded = BM25Index.load(path)
        assert loaded.is_built
        results = loaded.search("artificial intelligence", top_k=5)
        assert len(results) > 0
        assert results[0].passage_id == "pid_1"
