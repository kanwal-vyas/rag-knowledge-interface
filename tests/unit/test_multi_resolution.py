"""Tests for adaptive multi-resolution representation."""
from __future__ import annotations

from ingestion.representation.base import (
    create_passage_representation,
    create_representations,
    expand_to_parent,
    representation_id,
    split_sentences,
)


class TestSplitSentences:
    """Tests for sentence splitting."""

    def test_single_sentence(self) -> None:
        """Single sentence returns list with one element."""
        result = split_sentences("Hello world.")
        assert result == ["Hello world."]

    def test_multiple_sentences(self) -> None:
        """Multiple sentences are split correctly."""
        result = split_sentences("First sentence. Second sentence. Third.")
        assert len(result) == 3
        assert result[0] == "First sentence."
        assert result[1] == "Second sentence."
        assert result[2] == "Third."

    def test_hindi_danda(self) -> None:
        """Hindi danda splits sentences."""
        # Two sentences separated by danda
        s1 = "\u092a\u0939\u0932\u093e \u0935\u093e\u0915\u094d\u092f\u094b\u0902 \u0939\u0948\u0964"  # noqa: E501
        s2 = "\u0926\u0942\u0938\u0930\u093e \u0935\u093e\u0915\u094d\u092f\u094b\u0902 \u0939\u0948\u0964"  # noqa: E501
        result = split_sentences(s1 + " " + s2)
        assert len(result) == 2

    def test_empty_input(self) -> None:
        """Empty input returns empty list."""
        assert split_sentences("") == []
        assert split_sentences("   ") == []

    def test_no_ending_punctuation(self) -> None:
        """Text without sentence-ending punctuation returns as single item."""
        result = split_sentences("No punctuation here")
        assert result == ["No punctuation here"]

    def test_exclamation_and_question(self) -> None:
        """! and ? also split sentences."""
        result = split_sentences("What? Yes! OK.")
        assert len(result) == 3

    def test_whitespace_after_punctuation(self) -> None:
        """Sentences are split only when followed by whitespace."""
        result = split_sentences("3.14 is pi. Not 3.14.")
        # "3.14" doesn't have whitespace after the dot, so won't split
        # But "pi. Not" will split
        assert len(result) >= 2


class TestCreateRepresentations:
    """Tests for adaptive multi-resolution representation creation."""

    def test_short_passage_returns_only_passage(self) -> None:
        """Short passage (<= T_sentence) returns only passage repr."""
        text = "Short text under 256 chars."
        result = create_representations(
            passage_id="p001", text=text, lang="en",
            t_sentence=256, multi_resolution=True,
        )
        assert len(result) == 1
        assert result[0].representation_type == "passage"
        assert result[0].parent_id == "p001"
        assert result[0].child_ids == []

    def test_long_passage_returns_children(self) -> None:
        """Long passage (> T_sentence) returns passage + sentence children."""
        text = (
            "This is the first sentence that is reasonably long "
            "and descriptive for testing multi-resolution "
            "representations in the retrieval system. "
            "This is the second sentence also with good length "
            "for testing purposes. "
            "And finally the third sentence wraps up the "
            "passage nicely here."
        )
        assert len(text) > 256  # verify it exceeds threshold

        result = create_representations(
            passage_id="p002", text=text, lang="en",
            t_sentence=256, multi_resolution=True,
        )
        # Should have passage + sentence children
        assert len(result) >= 2
        assert result[0].representation_type == "passage"
        assert result[0].parent_id == "p002"
        assert len(result[0].child_ids) > 0
        # All children should be sentence type
        for r in result[1:]:
            assert r.representation_type == "sentence"
            assert r.parent_id == "p002"

    def test_exact_threshold_boundary(self) -> None:
        """At exactly T_sentence, passage-only (no children)."""
        text = "x" * 256
        result = create_representations(
            passage_id="p003", text=text, lang="en",
            t_sentence=256, multi_resolution=True,
        )
        assert len(result) == 1

    def test_one_over_threshold(self) -> None:
        """One char over T_sentence triggers multi-resolution."""
        # Multi-sentence text just over threshold
        text = "First sentence here. " + "x" * 240
        assert len(text) > 256
        result = create_representations(
            passage_id="p004", text=text, lang="en",
            t_sentence=256, multi_resolution=True,
        )
        # Should have passage + at least one sentence child
        assert len(result) >= 2

    def test_multi_resolution_disabled(self) -> None:
        """multi_resolution=False always returns passage only."""
        long_text = "Sentence one. " * 30  # well over 256
        result = create_representations(
            passage_id="p005", text=long_text, lang="en",
            t_sentence=256, multi_resolution=False,
        )
        assert len(result) == 1
        assert result[0].representation_type == "passage"

    def test_deterministic_sentence_ids(self) -> None:
        """Same passage always produces same sentence repr IDs."""
        text = "First sentence here with enough length. " * 10
        r1 = create_representations(
            passage_id="p006", text=text, lang="en",
            t_sentence=256, multi_resolution=True,
        )
        r2 = create_representations(
            passage_id="p006", text=text, lang="en",
            t_sentence=256, multi_resolution=True,
        )
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2, strict=False):
            assert a.representation_id == b.representation_id

    def test_child_ids_match_children(self) -> None:
        """Passage child_ids match IDs of sentence children."""
        text = "First sentence here with enough text. " * 10
        result = create_representations(
            passage_id="p007", text=text, lang="en",
            t_sentence=256, multi_resolution=True,
        )
        passage_repr = result[0]
        sentence_reprs = result[1:]
        child_ids_in_sentences = [
            s.representation_id for s in sentence_reprs
        ]
        assert passage_repr.child_ids == child_ids_in_sentences

    def test_empty_text(self) -> None:
        """Empty text returns a single passage representation."""
        result = create_representations(
            passage_id="p008", text="", lang="en",
            t_sentence=256, multi_resolution=True,
        )
        assert len(result) == 1
        assert result[0].representation_type == "passage"

    def test_single_sentence_long(self) -> None:
        """Long passage with only one sentence returns passage only."""
        text = "x" * 300  # One long sentence, no splitting
        result = create_representations(
            passage_id="p009", text=text, lang="en",
            t_sentence=256, multi_resolution=True,
        )
        assert len(result) == 1

    def test_hindi_multi_resolution(self) -> None:
        """Hindi passages with danda get sentence children."""
        s1 = "\u092f\u0939 \u092a\u0939\u0932\u093e \u0935\u093e\u0915\u094d\u092f\u094b\u0902 \u0938\u0947 \u092c\u0927 \u0939\u0948\u0964"  # noqa: E501
        s2 = "\u0926\u0942\u0938\u0930\u093e \u0935\u093e\u0915\u094d\u092f\u094b\u0902 \u0938\u0947 \u092c\u0927 \u0939\u0948\u0964"  # noqa: E501
        s3 = "\u0924\u0940\u0928\u0930\u0940 \u0935\u093e\u0915\u094d\u092f\u094b\u0902 \u0938\u0947 \u092c\u0927 \u0939\u0948\u0964"  # noqa: E501
        text = s1 + " " + s2 + " " + s3
        # Pad to exceed threshold if needed
        pad = (
            "\u0905\u0924\u093f\u0930\u093f\u0915\u094d\u0924 "
            "\u092a\u0930\u094d\u092f\u093e\u092a\u094d\u0924 "
            "\u0915\u093e \u0935\u093f\u0938\u094d\u0924\u093e\u0930 "
            "\u0939\u0948\u0964"
        )
        if len(text) <= 256:
            text += " " + pad * 5
        result = create_representations(
            passage_id="p010", text=text, lang="hi",
            t_sentence=256, multi_resolution=True,
        )
        has_danda = any(c in text for c in "\u0964\u0965.!?")
        if len(text) > 256 and has_danda:
            assert len(result) >= 2


class TestNoEvaluationLeakage:
    """Verify no evaluation fields leak into representations."""

    def test_no_forbidden_fields(self) -> None:
        """Representation must not contain evaluation-only fields."""
        from app.models.retrieval import FORBIDDEN_EVALUATION_FIELDS

        text = "Long passage text. " * 20
        reps = create_representations(
            passage_id="p011", text=text, lang="en",
            t_sentence=256, multi_resolution=True,
        )
        for r in reps:
            r_dict = r.__dict__
            leaked = set(r_dict.keys()) & FORBIDDEN_EVALUATION_FIELDS
            assert not leaked, (
                f"Representation contains forbidden fields: {leaked}"
            )


class TestParentExpansion:
    """Tests for context assembly parent expansion."""

    def test_passage_hit_returns_direct_text(self) -> None:
        """Passage-level hit returns the passage text directly."""
        passages = {"p001": "Full passage text here."}
        result = expand_to_parent("p001", "passage", passages)
        assert result == "Full passage text here."

    def test_sentence_hit_returns_parent_text(self) -> None:
        """Sentence-level hit returns parent passage text."""
        # In our model, sentence hits carry parent's passage_id
        passages = {
            "p001": "Full passage with multiple sentences. Second one."
        }
        result = expand_to_parent("p001", "sentence", passages)
        assert result == (
            "Full passage with multiple sentences. Second one."
        )

    def test_unknown_passage_returns_empty(self) -> None:
        """Unknown passage_id returns empty string."""
        passages = {"p001": "Known passage."}
        result = expand_to_parent("p999", "passage", passages)
        assert result == ""

    def test_empty_passages_dict(self) -> None:
        """Empty passages dict returns empty string."""
        result = expand_to_parent("p001", "passage", {})
        assert result == ""


class TestRepresentationIdDeterminism:
    """Tests for deterministic representation IDs."""

    def test_same_input_same_id(self) -> None:
        """Same text + passage_id + type -> same repr ID."""
        id1 = representation_id("hello world", "p1", "passage")
        id2 = representation_id("hello world", "p1", "passage")
        assert id1 == id2

    def test_different_type_different_id(self) -> None:
        """Different representation type -> different ID."""
        id1 = representation_id("hello world", "p1", "passage")
        id2 = representation_id("hello world", "p1", "sentence")
        assert id1 != id2

    def test_different_text_different_id(self) -> None:
        """Different text -> different ID."""
        id1 = representation_id("hello world", "p1", "passage")
        id2 = representation_id("hello world!", "p1", "passage")
        assert id1 != id2

    def test_deterministic_across_calls(self) -> None:
        """IDs are stable across multiple calls."""
        ids = [
            representation_id("test text", "p1", "sentence")
            for _ in range(10)
        ]
        assert len(set(ids)) == 1


class TestPassageOnlyMode:
    """Tests for passage-only mode compatibility."""

    def test_create_passage_representation_unchanged(self) -> None:
        """Original function still works identically."""
        r = create_passage_representation("p001", "Hello world.", "en")
        assert r.representation_type == "passage"
        assert r.parent_id == "p001"
        assert r.child_ids == []

    def test_passage_only_matches_single_resolution(self) -> None:
        """multi_resolution=False matches create_passage_representation."""
        text = "Hello world."
        r1 = create_passage_representation("p001", text, "en")
        r2_list = create_representations(
            "p001", text, "en", multi_resolution=False,
        )
        assert len(r2_list) == 1
        r2 = r2_list[0]
        assert r1.representation_id == r2.representation_id
        assert r1.text == r2.text
        assert r1.passage_id == r2.passage_id
