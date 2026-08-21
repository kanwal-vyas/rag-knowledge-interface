"""
BCP 47 language code mapping for MSMARCO-XI.

The dataset uses IETF BCP 47 language tags (e.g., "eng_Latn", "hin_Deva").
This module maps them to the application's canonical ISO 639-1 representation.

Architecture constraints:
- Language must be preserved as metadata throughout the pipeline.
- Unknown language codes MUST produce an explicit error, not map silently.
- The production pipeline must remain language-agnostic and multilingual.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageInfo:
    """Canonical representation of a language."""

    iso639_1: str  # Application canonical code (e.g., "en", "hi")
    bcp47: str     # Original BCP 47 code (e.g., "eng_Latn")
    name: str      # Human-readable name (e.g., "English", "Hindi")


# ---------------------------------------------------------------------------
# BCP 47 → ISO 639-1 mapping
#
# Sources present in MSMARCO-XI (verified from parquet file names and
# Phase 2 schema inspection):
#   hintrain → hin_Deva (Hindi)
#   asmtrain → asm_Beng (Assamese)
#   bentrain → ben_Beng (Bengali)
#   gujtrain → guj_Gujr (Gujarati)
#   kantrain → kan_Knda (Kannada)
#   (English is always the source_lang in every pair)
#
# The mapping covers all languages present in the dataset scope.
# If new languages are added to the dataset, extend this map.
# ---------------------------------------------------------------------------

_BCP47_TO_ISO: dict[str, str] = {
    "eng_Latn": "en",   # English (Latin script)
    "hin_Deva": "hi",   # Hindi (Devanagari script)
    "asm_Beng": "as",   # Assamese (Bengali script)
    "ben_Beng": "bn",   # Bengali (Bengali script)
    "guj_Gujr": "gu",   # Gujarati (Gujarati script)
    "kan_Knda": "kn",   # Kannada (Kannada script)
    "mal_Mlym": "ml",   # Malayalam (Malayalam script)
    "mar_Deva": "mr",   # Marathi (Devanagari script)
    "npi_Deva": "ne",   # Nepali (Devanagari script)
    "ory_Orya": "or",   # Odia (Odia script)
    "pan_Guru": "pa",   # Punjabi (Gurmukhi script)
    "tam_Taml": "ta",   # Tamil (Tamil script)
    "tel_Telu": "te",   # Telugu (Telugu script)
    "urd_Arab": "ur",   # Urdu (Arabic script)
}

# Reverse map for lookup convenience
_ISO_TO_BCP47: dict[str, str] = {v: k for k, v in _BCP47_TO_ISO.items()}

# Human-readable names
_LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "as": "Assamese",
    "bn": "Bengali",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}


class UnsupportedLanguageError(ValueError):
    """Raised when a BCP 47 language code is not in the supported set."""

    def __init__(self, bcp47_code: str) -> None:
        self.bcp47_code = bcp47_code
        supported = sorted(_BCP47_TO_ISO.keys())
        super().__init__(
            f"Unsupported BCP 47 language code: '{bcp47_code}'. "
            f"Supported codes: {supported}"
        )


def resolve_language(bcp47_code: str) -> LanguageInfo:
    """
    Resolve a BCP 47 language code to a canonical LanguageInfo.

    Args:
        bcp47_code: The BCP 47 code from the dataset (e.g., "hin_Deva").

    Returns:
        LanguageInfo with iso639_1, bcp47, and name.

    Raises:
        UnsupportedLanguageError: If the code is not in the supported set.
    """
    iso = _BCP47_TO_ISO.get(bcp47_code)
    if iso is None:
        raise UnsupportedLanguageError(bcp47_code)
    return LanguageInfo(
        iso639_1=iso,
        bcp47=bcp47_code,
        name=_LANGUAGE_NAMES.get(iso, iso),
    )


def iso639_1_for(bcp47_code: str) -> str:
    """
    Quick lookup: BCP 47 code → ISO 639-1 code.

    Raises UnsupportedLanguageError for unknown codes.
    """
    return resolve_language(bcp47_code).iso639_1


def is_supported_language(bcp47_code: str) -> bool:
    """Check whether a BCP 47 code is in the supported set."""
    return bcp47_code in _BCP47_TO_ISO


def supported_bcp47_codes() -> list[str]:
    """Return sorted list of all supported BCP 47 codes."""
    return sorted(_BCP47_TO_ISO.keys())


def supported_iso639_1_codes() -> list[str]:
    """Return sorted list of all supported ISO 639-1 codes."""
    return sorted(_BCP47_TO_ISO.values())
