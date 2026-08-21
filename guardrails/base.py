"""
Guardrail interface.

ARCHITECTURE DETAIL MISSING — REQUIRES CONFIRMATION
The guardrail architecture, including the "Semantic Grounding Consistency
Check" (named explicitly in the Audit as a multi-signal check replacing a
simpler "entailment check," per project memory — but its concrete signal
set and logic are not present in the available architecture source), is
inherited from V2 and not reproduced here. Do not implement the concrete
check yet. This interface only fixes the shape: a guardrail consumes a
generated answer plus its supporting context and returns pass/fail.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.generation import Context, GenerationResponse, GuardrailResult


class Guardrail(ABC):
    @abstractmethod
    def check(self, context: Context, response: GenerationResponse) -> GuardrailResult:
        raise NotImplementedError
