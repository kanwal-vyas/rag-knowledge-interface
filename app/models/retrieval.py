"""
Production retrieval data models.

CRITICAL INVARIANT (Audit §5, §10.3 — FROZEN):
`source_query_ids`, `is_selected`, `Answer`, and `Eng_Answer` must be
structurally absent from these models. They must never appear as fields
here, must never be settable via these models, and no code path that
constructs one of these models from raw dataset rows may pass evaluation
fields through, even accidentally via **kwargs or dict unpacking.

If you are implementing ingestion and find yourself needing an evaluation
field on a *production* object, stop — that is very likely the exact
leakage pattern this architecture was audited to prevent. See
evaluation/models.py for where those fields belong instead.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

# Fields that must NEVER appear on any model in this module.
# Used by tests/unit/test_model_isolation.py to assert the invariant holds.
FORBIDDEN_EVALUATION_FIELDS: frozenset[str] = frozenset(
    {
        "source_query_ids",
        "is_selected",
        "Answer",
        "Eng_Answer",
        "query_id",  # query_id specifically as passage/index metadata — see Audit §5
    }
)


class Language(str, Enum):
    """Canonical ISO 639-1 language codes used throughout the system.

    Audit §7 originally identified hi/en as the partition field for the
    monolingual vs. cross-lingual experiment. The production pipeline is
    multilingual — additional languages present in the dataset are supported
    via the ingestion language mapping layer (ingestion/dataset/language.py).

    Language codes here are ISO 639-1, mapped from BCP 47 dataset codes.
    """

    EN = "en"   # English
    HI = "hi"   # Hindi
    AS = "as"   # Assamese
    BN = "bn"   # Bengali
    GU = "gu"   # Gujarati
    KN = "kn"   # Kannada
    ML = "ml"   # Malayalam
    MR = "mr"   # Marathi
    NE = "ne"   # Nepali
    OR = "or"   # Odia
    PA = "pa"   # Punjabi
    TA = "ta"   # Tamil
    TE = "te"   # Telugu
    UR = "ur"   # Urdu


class Passage(BaseModel):
    """
    A single canonical, deduplicated corpus entry.

    `passage_id` MUST be the canonical content-hash ID used for both corpus
    dedup and gold-set joins (Audit §6, freeze decision 4). Never a
    row-local list index.
    """

    model_config = ConfigDict(extra="forbid")

    passage_id: str = Field(
        ...,
        description="Canonical content-hash ID. Same hash space as corpus dedup and gold joins.",
    )
    text: str
    lang: Language
    # Deliberately NO source_query_ids, is_selected, Answer, or Eng_Answer here.
    # Those live in evaluation/models.py, joined only by the eval harness.


class Query(BaseModel):
    """A user or benchmark query as it enters the production retrieval path."""

    model_config = ConfigDict(extra="forbid")

    query_text: str
    lang: Language
    # NOTE: query_id is intentionally omitted from the production path per
    # Audit §5 ("A field that exists in the retrieval-serving payload is a
    # field some future code path could accidentally filter or boost on").
    # If a query_id is needed for tracing, carry it in a side channel
    # (e.g. request ID / log correlation), not on this model.


class RetrievalMode(str, Enum):
    """
    Audit §7: monolingual vs. cross-lingual retrieval is an explicit
    experiment (Config 1 / Config 2), not a preset default.
    """

    MONOLINGUAL = "monolingual"  # Config 1: lang == query.lang
    CROSS_LINGUAL = "cross_lingual"  # Config 2: lang in {hi, en}, no filter


class RetrievalResult(BaseModel):
    """A single scored passage returned by a retriever."""

    model_config = ConfigDict(extra="forbid")

    passage: Passage
    score: float
    source: str = Field(
        ...,
        description=(
            'Which retriever produced this hit, e.g. "bm25", "dense", "fused", "reranked".'
        ),
    )


class RetrievalResponse(BaseModel):
    """Full result set for one query, plus enough metadata to reason about it."""

    model_config = ConfigDict(extra="forbid")

    query: Query
    mode: RetrievalMode
    results: list[RetrievalResult]
    # ARCHITECTURE DETAIL MISSING — REQUIRES CONFIRMATION
    # Fusion method (Audit mentions hybrid retrieval exists but not the
    # exact fusion algorithm) and reranker identity are not confirmed.
    # `source` field above is left free-text until that's resolved so this
    # model doesn't have to guess an enum of reranker names.
