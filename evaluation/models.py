"""
Evaluation-only data models.

This module is the ONLY place in the repository allowed to define fields
for `is_selected`, `Answer`, `Eng_Answer`, and `source_query_ids`
(Audit §5, freeze decision 3).

Structural isolation rule enforced by tests/unit/test_model_isolation.py:
nothing in this module may be imported by app/, retrieval/, generation/, or
guardrails/. It is imported only by the evaluation harness and benchmark
code. If a production module needs to import from here, that is a leakage
bug, not a missing convenience import — stop and reconsider the design.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class GoldRelevanceLabel(BaseModel):
    """
    One (query, passage) relevance judgment.

    `passage_id` MUST be the canonical content-hash ID — the same hash
    space used for corpus dedup (Audit §6, freeze decision 4). Building
    this from a row-local list index instead is the exact bug the Audit
    flags as "easy to get wrong."
    """

    model_config = ConfigDict(extra="forbid")

    query_id: str
    passage_id: str
    is_selected: bool


class AnswerReference(BaseModel):
    """
    Gold reference answer text. Audit §5: structurally separate from both
    the retrieval corpus and the generation prompt construction path —
    a more severe leak than is_selected if it ever reached generation.
    """

    model_config = ConfigDict(extra="forbid")

    query_id: str
    answer: str | None = Field(None, description="Original-language gold answer (`Answer`).")
    eng_answer: str | None = Field(None, description="English gold answer (`Eng_Answer`).")


class SourceQueryTraceability(BaseModel):
    """
    Audit §5: `source_query_ids` is removed from the production Qdrant
    payload and lives only in this offline traceability layer.
    """

    model_config = ConfigDict(extra="forbid")

    passage_id: str
    source_query_ids: list[str]


class QueryPool(str):
    """
    Marker type documenting the three-way split (Audit §3, freeze
    decision 2). Not an enum because the pool a query belongs to is a
    property of the dataset split it was drawn from, determined at
    ingestion time — this is a documentation aid for type signatures,
    not enforcement. Enforcement lives in the ingestion pipeline's split
    logic (ingestion/dataset/), which does not exist yet.
    """


TUNING_POOL = "tuning"  # train-split queries — all threshold/A-B-C decisions
BENCHMARK_POOL = "benchmark"  # validation-split queries — touched exactly once
