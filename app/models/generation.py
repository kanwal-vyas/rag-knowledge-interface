"""
Production generation data models.

CRITICAL INVARIANT (Audit §5 — "Answer / Eng_Answer" row, and freeze
decision 3): gold reference answers must never reach generation-prompt
construction. This module's models must never carry an Answer/Eng_Answer
field, and GenerationRequest must be constructible only from RetrievalResult
objects (app/models/retrieval.py), which themselves cannot carry those
fields either — see app/models/retrieval.py FORBIDDEN_EVALUATION_FIELDS.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from app.models.retrieval import Query, RetrievalResult


class Context(BaseModel):
    """Assembled context passed to the generator."""

    model_config = ConfigDict(extra="forbid")

    query: Query
    passages: list[RetrievalResult]
    # ARCHITECTURE DETAIL MISSING — REQUIRES CONFIRMATION
    # Context assembly strategy for mixed-language context (Audit §7,
    # Config 2 cross-lingual) is flagged as "an added engineering cost" but
    # no concrete assembly algorithm is specified.


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context: Context
    # ARCHITECTURE DETAIL MISSING — REQUIRES CONFIRMATION
    # Generation model/provider is not confirmed. Prompt template is not
    # specified.


class GenerationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_text: str
    grounded: bool
    # ARCHITECTURE DETAIL MISSING — REQUIRES CONFIRMATION
    # "Semantic Grounding Consistency Check" (Audit, named only) presumably
    # populates `grounded` and likely additional signal fields — exact
    # signal set not specified. Do not invent field names for those signals
    # here; extend this model only once the actual check design is known.


class GuardrailResult(BaseModel):
    """
    Placeholder shape only. The actual guardrail architecture, including
    the Semantic Grounding Consistency Check, is named but not specified
    in the available architecture source.

    ARCHITECTURE DETAIL MISSING — REQUIRES CONFIRMATION
    """

    model_config = ConfigDict(extra="forbid")

    passed: bool
    reason: str | None = None
