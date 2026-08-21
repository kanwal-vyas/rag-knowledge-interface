"""Tests for BCP 47 language code mapping (ingestion/dataset/language.py)."""
from __future__ import annotations

import pytest

from ingestion.dataset.language import (
    UnsupportedLanguageError,
    is_supported_language,
    iso639_1_for,
    resolve_language,
    supported_bcp47_codes,
    supported_iso639_1_codes,
)


class TestResolveLanguage:
    """Test resolve_language for every supported BCP 47 code."""

    def test_english(self) -> None:
        info = resolve_language("eng_Latn")
        assert info.iso639_1 == "en"
        assert info.bcp47 == "eng_Latn"
        assert info.name == "English"

    def test_hindi(self) -> None:
        info = resolve_language("hin_Deva")
        assert info.iso639_1 == "hi"
        assert info.bcp47 == "hin_Deva"
        assert info.name == "Hindi"

    def test_assamese(self) -> None:
        info = resolve_language("asm_Beng")
        assert info.iso639_1 == "as"
        assert info.name == "Assamese"

    def test_bengali(self) -> None:
        info = resolve_language("ben_Beng")
        assert info.iso639_1 == "bn"
        assert info.name == "Bengali"

    def test_gujarati(self) -> None:
        info = resolve_language("guj_Gujr")
        assert info.iso639_1 == "gu"
        assert info.name == "Gujarati"

    def test_kannada(self) -> None:
        info = resolve_language("kan_Knda")
        assert info.iso639_1 == "kn"
        assert info.name == "Kannada"

    def test_malayalam(self) -> None:
        info = resolve_language("mal_Mlym")
        assert info.iso639_1 == "ml"
        assert info.name == "Malayalam"

    def test_marathi(self) -> None:
        info = resolve_language("mar_Deva")
        assert info.iso639_1 == "mr"
        assert info.name == "Marathi"

    def test_nepali(self) -> None:
        info = resolve_language("npi_Deva")
        assert info.iso639_1 == "ne"
        assert info.name == "Nepali"

    def test_odia(self) -> None:
        info = resolve_language("ory_Orya")
        assert info.iso639_1 == "or"
        assert info.name == "Odia"

    def test_punjabi(self) -> None:
        info = resolve_language("pan_Guru")
        assert info.iso639_1 == "pa"
        assert info.name == "Punjabi"

    def test_tamil(self) -> None:
        info = resolve_language("tam_Taml")
        assert info.iso639_1 == "ta"
        assert info.name == "Tamil"

    def test_telugu(self) -> None:
        info = resolve_language("tel_Telu")
        assert info.iso639_1 == "te"
        assert info.name == "Telugu"

    def test_urdu(self) -> None:
        info = resolve_language("urd_Arab")
        assert info.iso639_1 == "ur"
        assert info.name == "Urdu"

    def test_deterministic(self) -> None:
        """Same input always produces same LanguageInfo."""
        a = resolve_language("hin_Deva")
        b = resolve_language("hin_Deva")
        assert a == b


class TestUnsupportedLanguage:
    """Unknown BCP 47 codes MUST produce an explicit error."""

    def test_unknown_code_raises_error(self) -> None:
        with pytest.raises(UnsupportedLanguageError, match="Unsupported BCP 47"):
            resolve_language("xyz_Unknown")

    def test_error_lists_supported_codes(self) -> None:
        with pytest.raises(UnsupportedLanguageError) as exc_info:
            resolve_language("qaa_Latn")
        msg = str(exc_info.value)
        assert "eng_Latn" in msg
        assert "hin_Deva" in msg

    def test_empty_string_raises_error(self) -> None:
        with pytest.raises(UnsupportedLanguageError):
            resolve_language("")

    def test_partially_valid_code_raises_error(self) -> None:
        """'eng' without script tag should fail — not in mapping."""
        with pytest.raises(UnsupportedLanguageError):
            resolve_language("eng")


class TestIso6391For:
    """Quick-lookup function."""

    def test_returns_iso_code(self) -> None:
        assert iso639_1_for("hin_Deva") == "hi"

    def test_unknown_raises(self) -> None:
        with pytest.raises(UnsupportedLanguageError):
            iso639_1_for("xyz_Fake")


class TestIsSupportedLanguage:
    def test_supported(self) -> None:
        assert is_supported_language("hin_Deva") is True
        assert is_supported_language("eng_Latn") is True

    def test_unsupported(self) -> None:
        assert is_supported_language("xyz_Unknown") is False
        assert is_supported_language("") is False


class TestSupportedCodes:
    def test_lists_are_non_empty(self) -> None:
        bcp47 = supported_bcp47_codes()
        iso = supported_iso639_1_codes()
        assert len(bcp47) >= 14
        assert len(iso) >= 14

    def test_lists_are_sorted(self) -> None:
        bcp47 = supported_bcp47_codes()
        assert bcp47 == sorted(bcp47)
        iso = supported_iso639_1_codes()
        assert iso == sorted(iso)

    def test_all_mapped_codes_present(self) -> None:
        """Every BCP 47 code resolves, and every ISO code appears."""
        for code in supported_bcp47_codes():
            info = resolve_language(code)
            assert info.iso639_1 in supported_iso639_1_codes()
