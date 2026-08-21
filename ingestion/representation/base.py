"""
Passage representation for indexing.

Supports two modes:

1. PASSAGE_ONLY: one passage → one Representation (single-resolution)
2. ADAPTIVE_MULTI_RESOLUTION: short passages get one Representation;
   long passages (> T_sentence chars) additionally produce sentence-level
   child Representations.

Adaptive multi-resolution is approved (ADR-0002, T_sentence tuning).
Gold labels remain passage-level — a sentence-child hit counts as a
match when parent passage_id is in the gold set.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from ingestion.normalization.text import normalize_text


@dataclass(frozen=True)
class Representation:
    """
    A single indexed representation of a passage.

    In the base case (passage-only), each CorpusPassage produces exactly
    one Representation with:
      - representation_type = "passage"
      - parent_id = passage_id (self-referential)
      - child_ids = [] (no sub-representations)

    When multi-resolution is specified, child representations (e.g., sentence-
    level or paragraph-level chunks) will have:
      - representation_type = "sentence" / "paragraph" / etc.
      - parent_id = the canonical passage_id of the source passage
      - child_ids = [] (leaf representations have no children)
    """

    representation_id: str  # Deterministic content-hash ID for this representation
    passage_id: str         # Canonical passage_id of the source passage (parent)
    text: str               # The text of this specific representation
    lang: str               # ISO 639-1 language code
    representation_type: str = "passage"  # "passage", "sentence", "paragraph", etc.
    parent_id: str = ""     # Canonical passage_id of parent (empty = self is parent)
    child_ids: list[str] = field(default_factory=list)  # IDs of child representations
    text_length: int = 0    # Character count of normalized text


def representation_id(text: str, passage_id: str, rep_type: str) -> str:
    """
    Deterministic ID for a representation.

    Uses SHA-256 over (representation_type + passage_id + normalized_text)
    to ensure:
    - Same text + same passage + same type → same ID
    - Different representations of the same passage → different IDs
    - Stable across runs

    This is NOT the same as canonical_passage_id (which hashes just the
    normalized text). The representation_id includes the type and parent
    to disambiguate sub-representations.
    """
    normalized = normalize_text(text)
    content = f"{rep_type}|{passage_id}|{normalized}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def create_passage_representation(
    passage_id: str,
    text: str,
    lang: str,
) -> Representation:
    """
    Create the base (passage-level) representation for a single passage.

    This is the default single-resolution indexing model.
    """
    normalized = normalize_text(text)
    rid = representation_id(text, passage_id, "passage")
    return Representation(
        representation_id=rid,
        passage_id=passage_id,
        text=normalized,
        lang=lang,
        representation_type="passage",
        parent_id=passage_id,
        child_ids=[],
        text_length=len(normalized),
    )


# ---------------------------------------------------------------------------
# Sentence splitting for multi-resolution
# ---------------------------------------------------------------------------

# Sentence-ending punctuation for English and Indic scripts.
# Hindi danda (\u0964) and double danda (\u0965) are the primary
# sentence terminators for Devanagari scripts.
# Bengali, Gujarati, etc. also use danda.
# Tamil/Telugu/Kannada/Malayalam use their own punctuation but
# danda is common across Indic.
_SENTENCE_END_RE = re.compile(
    r'(?<=[.!?\u0964\u0965])\s+'  # after .!? or danda, followed by whitespace
)


def split_sentences(text: str) -> list[str]:
    """
    Split normalized text into sentences.

    Uses punctuation-based splitting: `.`, `!`, `?`,
    Hindi danda (`\u0964`), and double danda (`\u0965`).

    Returns a list of non-empty stripped sentence strings.
    An empty or whitespace-only input returns [].
    A single-sentence passage returns a list with one element.
    """
    text = text.strip()
    if not text:
        return []

    parts = _SENTENCE_END_RE.split(text)
    sentences = [s.strip() for s in parts if s.strip()]
    return sentences if sentences else [text]


# Default T_sentence threshold (chars). Tuned on the train-split pool.
DEFAULT_T_SENTENCE = 256


def create_representations(
    passage_id: str,
    text: str,
    lang: str,
    t_sentence: int = DEFAULT_T_SENTENCE,
    multi_resolution: bool = True,
) -> list[Representation]:
    """
    Create one or more Representations for a passage.

    Modes:
    - multi_resolution=False: always returns [passage Representation]
    - multi_resolution=True:
      - len(normalized) <= t_sentence → [passage Representation]
      - len(normalized) > t_sentence → [passage + sentence children]

    Sentence children:
      - representation_type = "sentence"
      - parent_id = canonical passage_id
      - child_ids = [] (leaf)
      - text = individual sentence text

    The passage-level Representation always exists and carries
    child_ids pointing to its sentence children.
    """
    normalized = normalize_text(text)

    # Always create the passage-level representation
    rid = representation_id(text, passage_id, "passage")
    passage_repr = Representation(
        representation_id=rid,
        passage_id=passage_id,
        text=normalized,
        lang=lang,
        representation_type="passage",
        parent_id=passage_id,
        child_ids=[],  # populated below if multi-res
        text_length=len(normalized),
    )

    if not multi_resolution or len(normalized) <= t_sentence:
        return [passage_repr]

    # Adaptive: passage is long enough → split into sentences
    sentences = split_sentences(normalized)

    # If only one sentence (or splitting failed), no children needed
    if len(sentences) <= 1:
        return [passage_repr]

    sentence_reprs: list[Representation] = []
    child_ids: list[str] = []

    for sent_text in sentences:
        s_rid = representation_id(sent_text, passage_id, "sentence")
        child_ids.append(s_rid)
        sentence_reprs.append(
            Representation(
                representation_id=s_rid,
                passage_id=passage_id,
                text=sent_text,
                lang=lang,
                representation_type="sentence",
                parent_id=passage_id,
                child_ids=[],
                text_length=len(sent_text),
            )
        )

    # Re-create passage representation with child_ids populated
    passage_repr = Representation(
        representation_id=rid,
        passage_id=passage_id,
        text=normalized,
        lang=lang,
        representation_type="passage",
        parent_id=passage_id,
        child_ids=child_ids,
        text_length=len(normalized),
    )

    return [passage_repr] + sentence_reprs


# ---------------------------------------------------------------------------
# Context assembly: parent expansion
# ---------------------------------------------------------------------------


def expand_to_parent(
    hit_passage_id: str,
    hit_type: str,
    all_passages: dict[str, str],
) -> str:
    """
    Expand a retrieval hit to its parent passage text.

    For passage-level hits: returns the passage text directly.
    For sentence-level hits: looks up parent_id and returns
    the full parent passage text.

    Args:
        hit_passage_id: The canonical passage_id from the retrieval hit.
        hit_type: "passage" or "sentence".
        all_passages: Dict mapping passage_id → full passage text.

    Returns:
        The full parent passage text, or empty string if not found.
    """
    return all_passages.get(hit_passage_id, "")
