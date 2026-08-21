from __future__ import annotations

from app.models.retrieval import Language, Passage, RetrievalMode
from ingestion.deduplication.canonical_id import canonical_passage_id


def test_retrieval_mode_has_config_1_and_2() -> None:
    """Audit §7: monolingual vs cross-lingual is an experiment with exactly two configs."""
    assert RetrievalMode.MONOLINGUAL.value == "monolingual"
    assert RetrievalMode.CROSS_LINGUAL.value == "cross_lingual"


def test_canonical_passage_id_is_deterministic() -> None:
    text = "Some passage content."
    assert canonical_passage_id(text) == canonical_passage_id(text)


def test_canonical_passage_id_differs_for_different_text() -> None:
    assert canonical_passage_id("passage a") != canonical_passage_id("passage b")


def test_canonical_passage_id_used_consistently_in_passage_model() -> None:
    text = "Content used to build a corpus entry."
    pid = canonical_passage_id(text)
    passage = Passage(passage_id=pid, text=text, lang=Language.EN)
    assert passage.passage_id == pid


def test_canonical_passage_id_normalizes_whitespace() -> None:
    """Whitespace differences should not change the canonical ID."""
    a = canonical_passage_id("hello world")
    b = canonical_passage_id("  hello   world  ")
    c = canonical_passage_id("hello\tworld")
    assert a == b == c


def test_canonical_passage_id_normalizes_unicode() -> None:
    """NFC/NFD Unicode variants should produce the same ID."""
    nfc = canonical_passage_id("café")
    nfd = canonical_passage_id("cafe\u0301")
    assert nfc == nfd


def test_canonical_passage_id_is_sha256_hex() -> None:
    """Output is a 64-char hex string (SHA-256)."""
    pid = canonical_passage_id("test")
    assert len(pid) == 64
    assert all(c in "0123456789abcdef" for c in pid)


def test_canonical_passage_id_hindi_text() -> None:
    """Hindi text produces a stable canonical ID."""
    hindi = "नमस्ते दुनिया"
    pid = canonical_passage_id(hindi)
    assert canonical_passage_id(hindi) == pid
    assert len(pid) == 64
