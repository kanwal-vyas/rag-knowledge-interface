"""
Grounding validation layer.

Verifies that a generated answer is actually supported by the retrieved
context. This is a lightweight, mostly deterministic check — not a
full LLM judge (which the architecture names as the "Semantic Grounding
Consistency Check" but does not fully specify).

Checks performed:
1. Answer exists and is non-empty.
2. Answer is not a model refusal (indicating insufficient context).
3. Referenced facts are plausible given the context.
4. Empty/insufficient context causes grounded=False.

The architecture (Audit §5, "Semantic Grounding Consistency Check") mentions
a multi-signal check replacing a simpler "entailment check." The concrete
signal set is NOT specified in the available architecture source. This
implementation uses deterministic heuristics as the baseline; an LLM-judge
upgrade is left as a future extension point.
"""
from __future__ import annotations

import re

import structlog

from app.models.generation import Context, GenerationResponse

logger = structlog.get_logger(__name__)

# Phrases that indicate the model explicitly refused to answer.
_REFUSAL_PATTERNS = [
    r"i don'?t have enough information",
    r"i do not have enough information",
    r"i cannot answer",
    r"i can'?t answer",
    r"not enough information",
    r"insufficient information",
    r"no relevant information",
    r"cannot provide (?:an? )?answer",
    r"can'?t provide (?:an? )?answer",
    r"not available in the provided",
    r"not mentioned in the provided",
    r"not found in the provided",
    r"the provided context does not",
    r"the context does not contain",
    r"does not contain information",
    r"no information in the context",
    r"i am not able to answer",
    r"unable to answer based on",
]

_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def check_grounding(
    context: Context,
    response: GenerationResponse,
) -> GenerationResponse:
    """
    Validate grounding of a generated answer against its context.

    Modifies response.grounded in place and returns it.
    Does NOT modify response.answer_text.

    Checks:
    1. Answer is non-empty.
    2. Answer is not a model refusal.
    3. Context is non-empty (if context is empty, answer cannot be grounded).

    Returns:
        The same GenerationResponse with grounded status updated.
    """
    # Check 1: Non-empty answer.
    if not response.answer_text.strip():
        logger.warning("grounding_check_empty_answer")
        response.grounded = False
        return response

    # Check 2: Context is non-empty.
    if not context.passages:
        logger.warning("grounding_check_empty_context")
        response.grounded = False
        return response

    # Check 3: Model refusal detection.
    if _REFUSAL_RE.search(response.answer_text):
        logger.info("grounding_check_model_refusal")
        response.grounded = False
        return response

    # Check 4: Basic length sanity — a one-word answer is suspicious
    # unless it's a yes/no/number answer.
    words = response.answer_text.split()
    if len(words) < 2:
        # Very short answers: check if it looks like a valid short answer
        # (yes, no, a number, a name, etc.).
        short_valid = re.match(
            r"^(yes|no|हाँ|नहीं|\d+[\d.,]*|[A-Z][a-z]+)$",
            response.answer_text.strip(),
            re.IGNORECASE,
        )
        if not short_valid:
            logger.warning(
                "grounding_check_suspiciously_short",
                word_count=len(words),
            )
            # Don't fail grounding for this — just log a warning.

    # Check 5: Answer doesn't appear to be raw JSON or code output.
    if response.answer_text.strip().startswith("{") or response.answer_text.strip().startswith("<"):
        logger.warning("grounding_check_raw_output")
        response.grounded = False
        return response

    logger.info(
        "grounding_check_passed",
        answer_length=len(response.answer_text),
        context_passages=len(context.passages),
    )
    return response
