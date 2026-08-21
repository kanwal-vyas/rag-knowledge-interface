r"""
Multilingual tokenization for BM25 sparse retrieval.

The 14 supported languages use scripts with varying word-boundary characteristics:

WORD-SEPARATED SCRIPTS (space-delimited):
- Latin (English) — spaces between words
- Devanagari (Hindi, Marathi, Nepali) — visible word spaces
- Bengali/Assamese — spaces between words
- Gurmukhi (Punjabi) — spaces between words
- Gujarati — spaces between words
- Arabic/Urdu — spaces between words

AGGLUTINATIVE/COMPOUNDING SCRIPTS (fewer or no explicit word spaces):
- Tamil — historically no spaces, modern text often has them
- Telugu — spaces between words in modern text
- Kannada — spaces between words in modern text
- Malayalam — spaces between words in modern text
- Odia — spaces between words in modern text

Strategy:
1. Unicode word-boundary split (\w+ plus virama/matra extensions)
   handles most Indic scripts correctly because modern MSMARCO-XI
   text uses standard word spacing.
2. For scripts where words compound without spaces, we add character bigram
   tokenization as a secondary signal — this improves recall for exact
   substring matches.
3. Stopwords are NOT removed — BM25 handles term weighting.
4. No stemming — BM25's term frequency model handles morphological variation
   to a degree, and explicit stemming would require per-language stemmers.

LIMITATIONS (documented explicitly):
- Tamil historical no-space text may under-tokenize with word-boundary split.
- The character bigram supplement partially mitigates this.
- Urdu right-to-left text: tokenization order is preserved, but BM25 ranking
  is unaffected by script directionality.
- No language-specific stemmer or lemmatizer is applied. For BM25 this is
  acceptable because BM25 penalizes long documents naturally.
"""
from __future__ import annotations

import re

# -------------------------------------------------------------------
# Unicode word-boundary tokenizer
# -------------------------------------------------------------------

# \w+ matches Unicode word characters: [a-zA-Z0-9_] plus Unicode letters/digits
# Indic virama/halant characters are combining marks (\p{M}), NOT in \w class.
# Without extending the pattern, conjuncts like क्ष (ka + virama + sha) split
# into individual characters, breaking BM25's IDF signal.
# Virama codes for supported scripts:
#   U+094D Devanagari (Hindi, Marathi, Nepali)
#   U+09CD Bengali/Assamese
#   U+0ACD Gujarati
#   U+0A4D Gurmukhi (Punjabi)
#   U+0BCD Tamil
#   U+0C4D Telugu
#   U+0CCD Kannada
#   U+0D4D Malayalam
#   U+0B4D Odia
#   U+064D Arabic (Urdu)
#
# Vowel signs (matras) are combining marks (\p{M}) not in \w class.
# Without them, Hindi words like राजधानी split into राज + धानी.
# We add the common Indic vowel signs explicitly.
_VIRAMAS = "\u094D\u09CD\u0ACD\u0A4D\u0BCD\u0C4D\u0CCD\u0D4D\u0B4D\u064D"
# Common Indic vowel signs (matras) — not exhaustive but covers
# the standard dependent vowel forms across supported scripts.
_VOWEL_SIGNS = (
    "\u093E\u093F\u0940\u0941\u0942\u0943\u0944"  # Devanagari
    "\u0947\u0948\u094B\u094C"  # Devanagari
    "\u09BE\u09BF\u09C0\u09C1\u09C2\u09C3\u09C4"  # Bengali
    "\u09C7\u09C8\u09CB\u09CC"  # Bengali
    "\u0ABE\u0ABF\u0AC0\u0AC1\u0AC2"  # Gujarati
    "\u0AC7\u0AC8\u0ACB\u0ACC"  # Gujarati
    "\u0A3E\u0A3F\u0A40\u0A41\u0A42"  # Gurmukhi
    "\u0BCA\u0BCB\u0BCC"  # Tamil
    "\u0BBE\u0BBF\u0BC0\u0BC1\u0BC2"  # Tamil
    "\u0C3E\u0C3F\u0C40\u0C41\u0C42"  # Telugu
    "\u0C46\u0C47\u0C48\u0C4B\u0C4C"  # Telugu
    "\u0CBE\u0CBF\u0CC0\u0CC1\u0CC2"  # Kannada
    "\u0CC6\u0CC7\u0CC8\u0CCA\u0CCB\u0CCC"  # Kannada
    "\u0D3E\u0D3F\u0D40\u0D41\u0D42"  # Malayalam
    "\u0D46\u0D47\u0D48\u0D4B\u0D4C"  # Malayalam
    "\u0B3E\u0B3F\u0B40\u0B41\u0B42"  # Odia
)
_EXTRA_IN_WORD = _VIRAMAS + _VOWEL_SIGNS
_WORD_RE = re.compile(rf"[\w{_EXTRA_IN_WORD}]+", re.UNICODE)

# Character bigram extraction for agglutinative scripts
_CHAR_RE = re.compile(r"[\w]", re.UNICODE)


def tokenize_words(text: str) -> list[str]:
    """
    Split text into word tokens using Unicode word boundaries.

    Handles Latin, Devanagari, Bengali, Gurmukhi, Gujarati, Arabic, Tamil,
    Telugu, Kannada, Malayalam, Odia scripts via Unicode word-boundary matching.

    Note: lowercases input for case-insensitive BM25 matching.
    This is a deliberate choice — BM25 should treat "Artificial" and
    "artificial" as the same term. For scripts without case distinction
    (Devanagari, Bengali, etc.), lower() is a no-op.
    """
    return _WORD_RE.findall(text.lower())


def tokenize_char_bigrams(text: str) -> list[str]:
    """
    Extract character bigrams from text.

    Used as a supplementary tokenization for agglutinative scripts
    (Tamil, Telugu, etc.) where word boundaries may not be explicit.
    Bigrams help capture sub-word matches.
    """
    chars = _CHAR_RE.findall(text)
    return [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]


def tokenize(text: str, use_bigrams: bool = True) -> list[str]:
    """
    Multilingual tokenizer combining word tokens and optional character bigrams.

    Args:
        text: Input text (should already be normalize_text'd).
        use_bigrams: If True, append character bigrams to word tokens.
                     This improves recall for agglutinative scripts at the
                     cost of a larger vocabulary.

    Returns:
        List of tokens (words + optional bigrams).
    """
    tokens = tokenize_words(text)
    if use_bigrams:
        tokens.extend(tokenize_char_bigrams(text))
    return tokens


# -------------------------------------------------------------------
# Language-aware tokenization (configurable)
# -------------------------------------------------------------------

# Languages where word-boundary tokenization is reliable
_WORD_SEPARATED_LANGS = frozenset({
    "en", "hi", "as", "bn", "gu", "ur", "pa", "mr", "ne", "or",
})

# Languages where character bigrams are particularly useful
_BIGRAM_SUPPLEMENT_LANGS = frozenset({
    "ta", "te", "kn", "ml",  # Tamil, Telugu, Kannada, Malayalam
})


def tokenize_for_lang(text: str, lang: str) -> list[str]:
    """
    Language-aware tokenization.

    For word-separated languages, uses word tokens only.
    For agglutinative languages, adds character bigrams.
    """
    if lang in _BIGRAM_SUPPLEMENT_LANGS:
        return tokenize(text, use_bigrams=True)
    return tokenize(text, use_bigrams=False)


# -------------------------------------------------------------------
# BM25 corpus tokenization helper
# -------------------------------------------------------------------

def build_tokenized_corpus(
    texts: list[str],
    langs: list[str],
    use_bigrams: bool = True,
) -> list[list[str]]:
    """
    Tokenize a list of texts with language metadata.

    Args:
        texts: List of passage texts.
        langs: Corresponding language codes (ISO 639-1).
        use_bigrams: Whether to use character bigrams.

    Returns:
        List of token lists, one per input text.
    """
    assert len(texts) == len(langs), "texts and langs must have same length"
    return [tokenize_for_lang(t, la) for t, la in zip(texts, langs, strict=False)]
