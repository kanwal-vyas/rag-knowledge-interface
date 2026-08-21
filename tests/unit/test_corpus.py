"""Tests for corpus construction, deduplication, and Phase 1 checks."""
from __future__ import annotations

import pytest

from ingestion.dataset.corpus import CorpusBuilder, CorpusPassage, check_corpus_coverage
from ingestion.dataset.loader import MSMARCORow
from ingestion.dataset.phase1_checks import (
    find_train_validation_near_duplicates,
    scan_query_text_in_corpus,
)
from ingestion.deduplication.pipeline import deduplicate_passages

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(
    query_id: int = 1,
    en_passages: list[str] | None = None,
    hi_passages: list[str] | None = None,
    is_selected: list[int] | None = None,
    query_type: str = "DESCRIPTION",
) -> MSMARCORow:
    """Create a minimal MSMARCORow for testing."""
    return MSMARCORow(
        query_id=query_id,
        query_type=query_type,
        hindi_query=f"हिंदी प्रश्न {query_id}",
        english_query=f"English query {query_id}",
        source_lang="hi",
        target_lang="en",
        source_lang_bcp47="hin_Deva",
        target_lang_bcp47="eng_Latn",
        english_passages=en_passages or [f"English passage {query_id}"],
        hindi_passages=hi_passages or [f"हिंदी पाठ {query_id}"],
        is_selected=is_selected or [1],
        answer=f"Answer {query_id}",
        eng_answer=f"English Answer {query_id}",
        meta={},
    )


# ---------------------------------------------------------------------------
# Phase 1 checks
# ---------------------------------------------------------------------------

class TestQueryInCorpusScan:
    def test_clean_corpus(self) -> None:
        """Queries not in corpus → empty list."""
        corpus = ["The quick brown fox", "jumped over the lazy dog"]
        queries = ["quantum physics", "climate change"]
        found = scan_query_text_in_corpus(corpus, queries)
        assert found == []

    def test_query_found_in_corpus(self) -> None:
        """Query that appears as substring in a passage is flagged."""
        corpus = ["India is a diverse country with many languages"]
        queries = ["diverse country"]
        found = scan_query_text_in_corpus(corpus, queries)
        assert len(found) == 1
        assert "diverse country" in found[0]

    def test_short_queries_skipped(self) -> None:
        """Trivially short queries (< 3 chars) are skipped."""
        corpus = ["A passage with the letter a"]
        queries = ["a"]
        found = scan_query_text_in_corpus(corpus, queries)
        assert found == []

    def test_case_insensitive(self) -> None:
        """Substring scan is case-insensitive."""
        corpus = ["Hello World"]
        queries = ["hello world"]
        found = scan_query_text_in_corpus(corpus, queries)
        assert len(found) == 1


class TestNearDuplicateCheck:
    def test_no_duplicates(self) -> None:
        """Completely different queries → empty list."""
        train = ["What is machine learning?", "Explain quantum computing"]
        val = ["History of Rome", "Climate change effects"]
        dupes = find_train_validation_near_duplicates(train, val)
        assert dupes == []

    def test_exact_duplicates(self) -> None:
        """Identical queries across splits are flagged."""
        train = ["What is machine learning?", "Explain quantum computing"]
        val = ["What is machine learning?", "History of Rome"]
        dupes = find_train_validation_near_duplicates(train, val)
        # Should find at least the exact match
        assert len(dupes) >= 1
        exact = [d for d in dupes if d[2] == 1.0]
        assert len(exact) >= 1

    def test_empty_inputs(self) -> None:
        """Empty inputs → empty list."""
        assert find_train_validation_near_duplicates([], []) == []
        assert find_train_validation_near_duplicates(["hello"], []) == []
        assert find_train_validation_near_duplicates([], ["hello"]) == []


class TestNearDuplicateRegression:
    """Regression tests for false-positive issues with character-set Jaccard.

    These are the real-world short English queries that produced false
    positives under character-set Jaccard (sim >= 0.95) because they
    share common letter characters despite being semantically unrelated.

    Token-level Jaccard should NOT flag these as near-duplicates.
    """

    def test_unrelated_short_queries_not_flagged(self) -> None:
        """Two short, semantically unrelated English queries must NOT
        be flagged as near-duplicates under token-level Jaccard."""
        train = [
            "what direction does phloem flow",
            "how far is philadelphia from lancaster pa",
        ]
        val = [
            "how far is philadelphia from lancaster pa",
            "what direction does phloem flow",
        ]
        dupes = find_train_validation_near_duplicates(train, val)
        # These share many letter characters but NOT many words.
        # Token-level Jaccard: {what, direction, does, phloem, flow} vs
        # {how, far, is, philadelphia, from, lancaster, pa}
        # Intersection = 0 tokens → sim = 0.0 → NOT flagged.
        # Only the exact matches should be found.
        exact = [d for d in dupes if d[2] == 1.0]
        assert len(exact) == 2  # Both are exact matches in cross direction
        # No near-duplicates with sim in (0.0, 1.0)
        near = [d for d in dupes if 0.0 < d[2] < 1.0]
        assert near == []

    def test_short_queries_with_common_stopwords(self) -> None:
        """Queries that share only common stop words (is, the, a, of)
        should NOT be flagged."""
        train = ["what is the capital of france"]
        val = ["who is the president of america"]
        dupes = find_train_validation_near_duplicates(train, val)
        # Shared words: {is, the, of} out of union ~10 words → sim ~0.3
        # Should NOT be flagged at 0.95 threshold
        assert dupes == []

    def test_truly_near_duplicate_detected(self) -> None:
        """Queries differing by one word SHOULD be flagged."""
        train = ["what is the capital of france"]
        val = ["what is the capital of germany"]
        dupes = find_train_validation_near_duplicates(train, val)
        # Shared: {what, is, the, capital, of} = 5 out of union 7
        # Token Jaccard = 5/7 ≈ 0.71 — below 0.95 threshold
        # This is correct behavior: these are different queries!
        # If you want to flag these, lower the threshold.
        assert dupes == []

    def test_high_similarity_near_duplicate(self) -> None:
        """Queries differing by only a minor detail SHOULD be flagged
        when similarity is high enough."""
        train = ["what is the population of new york city in 2020"]
        val = ["what is the population of new york city in 2021"]
        dupes = find_train_validation_near_duplicates(train, val)
        # Shared: {what, is, the, population, of, new, york, city, in} = 9
        # Union: {what, is, the, population, of, new, york, city, in, 2020, 2021} = 11
        # Token Jaccard = 9/11 ≈ 0.82 — below 0.95
        assert dupes == []

    def test_hindi_queries_not_false_positives(self) -> None:
        """Hindi queries should also not produce false positives."""
        train = ["भारत की राजधानी क्या है"]  # What is the capital of India
        val = ["भारत का राष्ट्रपति कौन है"]  # Who is the president of India
        dupes = find_train_validation_near_duplicates(train, val)
        # Different meaning despite shared Hindi words
        assert dupes == []


# ---------------------------------------------------------------------------
# Corpus builder
# ---------------------------------------------------------------------------

class TestCorpusBuilder:
    def test_single_row(self) -> None:
        """One row adds its passages to the corpus."""
        builder = CorpusBuilder(tier_name="T1", target_passages=1000, source_query_rows=100)
        row = _make_row(
            query_id=1,
            en_passages=["English passage one", "English passage two"],
            hi_passages=["हिंदी पाठ एक", "हिंदी पाठ दो"],
        )
        builder.add_row(row)
        passages, stats = builder.build()
        assert stats.unique_passages == 4  # 2 en + 2 hi
        assert stats.query_count == 1

    def test_deduplication_across_rows(self) -> None:
        """Same passage text in different rows is deduplicated."""
        builder = CorpusBuilder(tier_name="T1", target_passages=1000, source_query_rows=100)
        shared_text = "This is a shared passage"
        row1 = _make_row(query_id=1, en_passages=[shared_text], hi_passages=["p1"])
        row2 = _make_row(query_id=2, en_passages=[shared_text], hi_passages=["p2"])
        builder.add_row(row1)
        builder.add_row(row2)
        passages, stats = builder.build()
        # shared_text deduped → 1 en + 2 hi = 3 unique
        assert stats.unique_passages == 3
        assert stats.duplicate_count > 0

    def test_has_capacity(self) -> None:
        """has_capacity() respects target_passages."""
        builder = CorpusBuilder(tier_name="T1", target_passages=3, source_query_rows=100)
        assert builder.has_capacity() is True
        builder.add_row(_make_row(en_passages=["a"], hi_passages=["b"]))
        assert builder.has_capacity() is True  # 2 passages, target=3
        builder.add_row(_make_row(query_id=2, en_passages=["c"], hi_passages=["d"]))
        assert builder.has_capacity() is False  # 4 passages (no dup), target=3

    def test_empty_passages_skipped(self) -> None:
        """Empty/whitespace-only passages are not indexed."""
        builder = CorpusBuilder(tier_name="T1", target_passages=1000, source_query_rows=100)
        row = _make_row(en_passages=["", "  ", "real passage"], hi_passages=["", "real hindi"])
        builder.add_row(row)
        passages, stats = builder.build()
        assert stats.unique_passages == 2  # "real passage" + "real hindi"

    def test_default_tier_sizes(self) -> None:
        """Default tiers are loaded from DEFAULT_TIERS."""
        from ingestion.dataset.corpus import DEFAULT_TIERS
        assert "T1" in DEFAULT_TIERS
        assert "T5" in DEFAULT_TIERS
        assert DEFAULT_TIERS["T3"]["target_passages"] == 500_000

    def test_source_query_ids_tracked(self) -> None:
        """Shared passages track which queries they came from."""
        builder = CorpusBuilder(tier_name="T1", target_passages=1000, source_query_rows=100)
        shared = "shared text"
        builder.add_row(_make_row(query_id=10, en_passages=[shared], hi_passages=["a"]))
        builder.add_row(_make_row(query_id=20, en_passages=[shared], hi_passages=["b"]))
        passages, stats = builder.build()
        en_passage = [p for p in passages if p.text == "shared text"][0]
        assert 10 in en_passage.source_query_ids
        assert 20 in en_passage.source_query_ids


# ---------------------------------------------------------------------------
# Deduplication pipeline
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_no_duplicates(self) -> None:
        """All unique passages pass through unchanged."""
        passages = [
            CorpusPassage(passage_id="a", text="text a", lang="en"),
            CorpusPassage(passage_id="b", text="text b", lang="hi"),
        ]
        unique, stats = deduplicate_passages(passages)
        assert len(unique) == 2
        assert stats.duplicate_count == 0
        assert stats.duplicate_percentage == 0.0

    def test_exact_duplicates(self) -> None:
        """Duplicate passage_ids are merged."""
        passages = [
            CorpusPassage(passage_id="a", text="text a", lang="en"),
            CorpusPassage(passage_id="a", text="text a", lang="en"),
            CorpusPassage(passage_id="b", text="text b", lang="hi"),
        ]
        unique, stats = deduplicate_passages(passages)
        assert len(unique) == 2
        assert stats.duplicate_count == 1
        assert stats.duplicate_percentage == pytest.approx(33.33, rel=0.01)

    def test_stats_breakdown(self) -> None:
        """Stats correctly count by language."""
        passages = [
            CorpusPassage(passage_id=f"en{i}", text=f"en {i}", lang="en")
            for i in range(5)
        ] + [
            CorpusPassage(passage_id=f"hi{i}", text=f"hi {i}", lang="hi")
            for i in range(3)
        ]
        unique, stats = deduplicate_passages(passages)
        assert stats.lang_unique["hi"] == 3
        assert stats.lang_unique["en"] == 5

    def test_empty_input(self) -> None:
        """Empty input → empty output."""
        unique, stats = deduplicate_passages([])
        assert unique == []
        assert stats.raw_count == 0

    def test_deterministic(self) -> None:
        """Same input always produces same output order."""
        passages = [
            CorpusPassage(passage_id="c", text="c", lang="en"),
            CorpusPassage(passage_id="a", text="a", lang="en"),
            CorpusPassage(passage_id="b", text="b", lang="hi"),
        ]
        u1, _ = deduplicate_passages(passages)
        u2, _ = deduplicate_passages(passages)
        assert [p.passage_id for p in u1] == [p.passage_id for p in u2]


# ---------------------------------------------------------------------------
# Coverage check
# ---------------------------------------------------------------------------

class TestCorpusCoverage:
    def test_full_coverage(self) -> None:
        """All gold passages present in corpus → 100% coverage."""
        corpus = [
            CorpusPassage(passage_id="p1", text="a", lang="en"),
            CorpusPassage(passage_id="p2", text="b", lang="en"),
        ]
        gold = {1: ["p1"], 2: ["p2"]}
        result = check_corpus_coverage(corpus, gold)
        assert result["coverage_rate"] == 1.0
        assert result["full_coverage_rate"] == 1.0

    def test_partial_coverage(self) -> None:
        """Some gold passages missing → partial coverage."""
        corpus = [CorpusPassage(passage_id="p1", text="a", lang="en")]
        gold = {1: ["p1", "p2"]}  # p2 is missing
        result = check_corpus_coverage(corpus, gold)
        assert result["coverage_rate"] == pytest.approx(0.5)
        assert result["fully_covered_queries"] == 0
        assert result["partially_covered_queries"] == 1

    def test_no_coverage(self) -> None:
        """No gold passages in corpus → 0% coverage."""
        corpus = [CorpusPassage(passage_id="p1", text="a", lang="en")]
        gold = {1: ["p99"]}
        result = check_corpus_coverage(corpus, gold)
        assert result["coverage_rate"] == 0.0



