"""
Generator interface.

ARCHITECTURE DETAIL MISSING — REQUIRES CONFIRMATION
No production generation model/provider is confirmed in the available
architecture source. Per project principle, the intended production
generation model is explicitly NOT assumed to be the development model
used for design work (Claude, used for architecture/review) — do not
default this interface's concrete implementation to whatever dev-time LLM
was used for design conversations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.generation import Context, GenerationResponse


class Generator(ABC):
    """Produces a grounded answer from assembled context."""

    @abstractmethod
    def generate(self, context: Context) -> GenerationResponse:
        raise NotImplementedError
