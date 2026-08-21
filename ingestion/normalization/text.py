"""
Text normalization for passage and query content.

Normalization must be applied BEFORE canonical passage_id hashing
(Audit §6, freeze decision 4). The normalization rules below are
stable: once gold-set data has been generated against a specific
normalization pipeline, do not change the rules without re-running
the entire corpus + gold-set construction.

Normalization is intentionally conservative — it collapses whitespace
and normalizes Unicode encoding, but does NOT perform stemming,
stopword removal, or any transformation that would change meaning.
"""
from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    """
    Deterministic text normalization applied to all passage and query
    text before hashing, indexing, or embedding.

    Steps:
    1. Unicode NFC normalization (canonical decomposition + composition)
       — ensures visually identical text maps to the same bytes.
    2. Collapse all whitespace runs (spaces, tabs, newlines, non-breaking
       spaces) to a single ASCII space.
    3. Strip leading/trailing whitespace.
    4. Remove zero-width characters (ZWJ, ZWNJ, ZWSP, soft hyphens)
       that are invisible but would change a hash.

    This function must be deterministic: same input always produces
    the same output. No randomness, no locale dependency, no network calls.
    """
    # Step 1: Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Step 2: Remove zero-width characters
    # U+200B (ZWSP), U+200C (ZWNJ), U+200D (ZWJ), U+FEFF (BOM/ZWNBS),
    # U+00AD (soft hyphen)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", text)

    # Step 3: Collapse all whitespace runs to a single space
    # \s matches space, tab, newline, carriage return, form feed, vertical tab,
    # and various Unicode space characters
    text = re.sub(r"\s+", " ", text)

    # Step 4: Strip leading/trailing whitespace
    text = text.strip()

    return text


def normalize_query(text: str) -> str:
    """
    Query normalization. Same pipeline as normalize_text for consistency,
    but named separately so it's clear when we're normalizing a query
    vs. a passage (relevant for Phase 1 query-in-corpus scan).
    """
    return normalize_text(text)
