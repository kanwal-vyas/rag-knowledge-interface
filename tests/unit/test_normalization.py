"""Tests for text normalization (ingestion/normalization/text.py)."""
from __future__ import annotations

from ingestion.normalization.text import normalize_query, normalize_text


class TestNormalizeText:
    def test_deterministic(self) -> None:
        """Same input always produces same output."""
        text = "Hello   world\t\twith\nspaces"
        assert normalize_text(text) == normalize_text(text)

    def test_nfc_normalization(self) -> None:
        """Unicode NFC: é (single codepoint) vs e + combining accent."""
        # U+00E9 (é) is already NFC
        nfc = "\u00e9"
        # U+0065 (e) + U+0301 (combining acute accent) is NFD
        nfd = "\u0065\u0301"
        assert normalize_text(nfc) == normalize_text(nfd)

    def test_whitespace_collapsing(self) -> None:
        """Multiple whitespace chars collapse to single space."""
        assert normalize_text("hello   world") == "hello world"
        assert normalize_text("a\t\tb") == "a b"
        assert normalize_text("line1\nline2") == "line1 line2"

    def test_strip(self) -> None:
        """Leading/trailing whitespace removed."""
        assert normalize_text("  hello  ") == "hello"
        assert normalize_text("\t\nhello\n\t") == "hello"

    def test_zero_width_removal(self) -> None:
        """Zero-width characters are removed."""
        # ZWSP (U+200B)
        assert normalize_text("hel\u200blo") == "hello"
        # ZWNJ (U+200C)
        assert normalize_text("hel\u200clo") == "hello"
        # ZWJ (U+200D)
        assert normalize_text("hel\u200dlo") == "hello"
        # BOM (U+FEFF)
        assert normalize_text("\ufeffhello") == "hello"
        # Soft hyphen (U+00AD)
        assert normalize_text("hel\u00adlo") == "hello"

    def test_hindi_text(self) -> None:
        """Hindi text with Devanagari script normalizes correctly."""
        hindi = "नमस्ते   दुनिया"
        result = normalize_text(hindi)
        assert result == "नमस्ते दुनिया"

    def test_empty_string(self) -> None:
        assert normalize_text("") == ""

    def test_only_whitespace(self) -> None:
        assert normalize_text("   \t\n  ") == ""

    def test_query_normalization_matches_text(self) -> None:
        """normalize_query delegates to normalize_text."""
        text = "hello   world"
        assert normalize_query(text) == normalize_text(text)


class TestNormalizationForHashing:
    """Verify normalization produces stable input for canonical hashing."""

    def test_whitespace_variants_produce_same_normalized_form(self) -> None:
        """Different whitespace patterns normalize to the same string."""
        a = normalize_text("Hello world")
        b = normalize_text("  Hello   world  ")
        c = normalize_text("Hello\tworld")
        assert a == b == c

    def test_unicode_variants_produce_same_normalized_form(self) -> None:
        """NFC/NFD Unicode variants normalize to same string."""
        a = normalize_text("café")  # NFC
        b = normalize_text("cafe\u0301")  # NFD
        assert a == b
