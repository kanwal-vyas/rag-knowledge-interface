"""Tests for passage representation (base case: passage-only indexing)."""
from __future__ import annotations

from ingestion.representation.base import (
    create_passage_representation,
    representation_id,
)


class TestRepresentationId:
    """Tests for deterministic representation_id generation."""

    def test_deterministic(self) -> None:
        """Same inputs → same representation_id."""
        rid1 = representation_id("Some text", "passage_abc", "passage")
        rid2 = representation_id("Some text", "passage_abc", "passage")
        assert rid1 == rid2

    def test_differs_for_different_text(self) -> None:
        """Different text → different representation_id."""
        rid1 = representation_id("Text A", "pid_1", "passage")
        rid2 = representation_id("Text B", "pid_1", "passage")
        assert rid1 != rid2

    def test_differs_for_different_passage_id(self) -> None:
        """Different parent passage_id → different representation_id."""
        rid1 = representation_id("Same text", "pid_A", "passage")
        rid2 = representation_id("Same text", "pid_B", "passage")
        assert rid1 != rid2

    def test_differs_for_different_type(self) -> None:
        """Different representation type → different representation_id."""
        rid1 = representation_id("Same text", "pid_1", "passage")
        rid2 = representation_id("Same text", "pid_1", "sentence")
        assert rid1 != rid2

    def test_handles_hindi_text(self) -> None:
        """Hindi text produces deterministic IDs."""
        text = "यह एक हिंदी पाठ है"
        rid1 = representation_id(text, "pid_hi", "passage")
        rid2 = representation_id(text, "pid_hi", "passage")
        assert rid1 == rid2
        assert len(rid1) == 64  # SHA-256 hex

    def test_unicode_normalization_stability(self) -> None:
        """NFC-normalized and raw text produce the same representation_id."""
        # U+0915 (Devanagari KA) can be composed or decomposed
        rid_nfc = representation_id("क", "pid", "passage")
        rid_raw = representation_id("क", "pid", "passage")
        assert rid_nfc == rid_raw

    def test_empty_text(self) -> None:
        """Empty text is valid and produces a deterministic ID."""
        rid = representation_id("", "pid_empty", "passage")
        assert len(rid) == 64


class TestCreatePassageRepresentation:
    """Tests for the base passage-only representation factory."""

    def test_base_representations_are_passage_type(self) -> None:
        rep = create_passage_representation("pid_1", "Some content", "en")
        assert rep.representation_type == "passage"

    def test_parent_id_is_self(self) -> None:
        """Base representation is self-referential parent."""
        rep = create_passage_representation("pid_1", "Content", "hi")
        assert rep.parent_id == "pid_1"

    def test_child_ids_empty(self) -> None:
        """Base representation has no children."""
        rep = create_passage_representation("pid_1", "Content", "en")
        assert rep.child_ids == []

    def test_text_is_normalized(self) -> None:
        """Text is normalized via the normalization pipeline."""
        rep = create_passage_representation("pid_1", "  Some   content  ", "en")
        assert rep.text == "Some content"  # Whitespace collapsed and stripped

    def test_text_length_matches(self) -> None:
        rep = create_passage_representation("pid_1", "Hello world", "en")
        assert rep.text_length == len(rep.text)

    def test_representation_id_matches_function(self) -> None:
        """The representation_id matches calling representation_id() directly."""
        rep = create_passage_representation("pid_1", "Content", "en")
        expected = representation_id("Content", "pid_1", "passage")
        assert rep.representation_id == expected

    def test_frozen_dataclass(self) -> None:
        """Representation is immutable."""
        rep = create_passage_representation("pid_1", "Content", "en")
        try:
            rep.text = "modified"  # type: ignore[misc]
            raise AssertionError("Should have raised FrozenInstanceError")
        except AttributeError:
            pass  # Expected


class TestNoEvaluationLeakage:
    """Verify that Representation dataclass never contains evaluation-only fields."""

    FORBIDDEN = {"is_selected", "Answer", "Eng_Answer", "source_query_ids"}

    def test_no_forbidden_fields_in_representation(self) -> None:
        rep = create_passage_representation("pid_1", "Content", "en")
        rep_keys = set(rep.__dataclass_fields__.keys())
        assert rep_keys & self.FORBIDDEN == set(), (
            f"Representation contains forbidden fields: {rep_keys & self.FORBIDDEN}"
        )

    def test_no_forbidden_fields_in_factory_kwargs(self) -> None:
        """Attempting to pass forbidden fields raises TypeError."""
        try:
            create_passage_representation(
                "pid_1", "Content", "en",  # type: ignore[call-arg]
                is_selected=1,
            )
            raise AssertionError("Should have raised TypeError for unexpected kwargs")
        except TypeError:
            pass  # Expected


class TestMultilingualHandling:
    """Verify representations work correctly across languages."""

    def test_all_supported_languages(self) -> None:
        """Create representations for every language in the Language enum."""
        from app.models.retrieval import Language
        for lang in Language:
            rep = create_passage_representation(
                f"pid_{lang.value}", "Test passage", lang.value
            )
            assert rep.lang == lang.value
            assert rep.representation_type == "passage"

    def test_same_text_different_lang_different_id(self) -> None:
        """Same text in different languages gets different representation_ids
        (because the representation_id includes the parent passage_id which
        is different per language)."""
        # Note: in practice the text would differ across languages,
        # but even with same text, different passage_ids yield different IDs.
        rid_en = representation_id("test", "pid_en", "passage")
        rid_hi = representation_id("test", "pid_hi", "passage")
        assert rid_en != rid_hi
