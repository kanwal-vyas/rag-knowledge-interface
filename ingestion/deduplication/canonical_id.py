"""
Canonical passage_id generation.

Audit §6, freeze decision 4: gold-set construction and corpus dedup MUST
use the same canonical content-hash passage_id space. This is the single
shared implementation both ingestion (corpus build) and evaluation
(gold-set join) must call — having two separate hash implementations that
are "supposed to" agree is exactly the kind of silent-bug risk the Audit
flags explicitly. There must be exactly one function that produces a
passage_id, imported by both sides.

Normalization procedure (stable — do not change after gold-set generation):
1. Unicode NFC normalization (canonical decomposition + composition)
2. Remove zero-width characters (ZWJ, ZWNJ, ZWSP, soft hyphen, BOM)
3. Collapse all whitespace runs to a single ASCII space
4. Strip leading/trailing whitespace
5. Encode as UTF-8 bytes
6. SHA-256 hash → hex digest (64 hex chars)
"""

from __future__ import annotations

import hashlib

from ingestion.normalization.text import normalize_text


def canonical_passage_id(passage_text: str) -> str:
    """
    Deterministic content-hash ID for a passage.

    Uses the same normalization pipeline as ingestion and evaluation,
    so any passage text that has been normalized before hashing will
    produce the same ID regardless of whether it went through the
    ingestion pipeline or the evaluation gold-set join.

    IMPORTANT: After any gold-set data has been generated against this
    function's output, do NOT change the normalization rules or hash
    algorithm — that would silently break every existing passage_id join.
    """
    normalized = normalize_text(passage_text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
