"""
Guardrail implementations.

Handles:
- Empty query
- Off-topic query
- Unsafe/inappropriate query
- No relevant retrieval
- Insufficient context
- Generation failure
- Malformed model output

The system should know when NOT to answer. Returns structured refusal
reasons. Does not leak internal prompts/errors to the user.

The architecture (Audit §10.7) mentions guardrail structure as frozen
in V2 but not reproduced. This implementation covers the common cases
documented in the project brief.
"""
from __future__ import annotations

import re

import structlog

from app.models.generation import Context, GenerationResponse, GuardrailResult

logger = structlog.get_logger(__name__)

# Unsafe content patterns (basic keyword-based, not exhaustive).
_UNSAFE_PATTERNS = [
    r"\b(how to (make|build|create) (a )?(bomb|explosive|weapon))\b",
    r"\b(kill|murder|assassinate)\b.*\b(yourself|someone|people)\b",
    r"\b(suicide|self[- ]harm)\b",
    r"\b(hack|exploit|phish)\b.*\b(password|account|system)\b",
]

_UNSAFE_RE = re.compile("|".join(_UNSAFE_PATTERNS), re.IGNORECASE)

# Off-topic patterns — queries that are clearly not knowledge-seeking.
_OFF_TOPIC_PATTERNS = [
    r"^(hi|hello|hey|namaste|नमस्ते)\s*[!.]?\s*$",
    r"^(what'?s? (up|your name)|who are you)\s*[?.]?\s*$",
    r"^(thanks|thank you|धन्यवाद)\s*[!.]?\s*$",
]

_OFF_TOPIC_RE = re.compile("|".join(_OFF_TOPIC_PATTERNS), re.IGNORECASE)


class GuardrailPipeline:
    """
    Runs guardrail checks at multiple pipeline stages.

    - Pre-retrieval: validates the input query.
    - Post-retrieval: validates retrieval results.
    - Post-generation: validates the generated answer.

    Each stage returns a GuardrailResult. If passed=False, the pipeline
    should stop and return a structured refusal.
    """

    def check_query(self, query_text: str) -> GuardrailResult:
        """Pre-retrieval guardrail: validate the input query."""
        # Check 1: Empty query.
        if not query_text or not query_text.strip():
            logger.warning("guardrail_empty_query")
            return GuardrailResult(
                passed=False,
                reason="Query text is empty. Please provide a question.",
            )

        # Check 2: Too short (single character).
        stripped = query_text.strip()
        if len(stripped) < 2:
            logger.warning("guardrail_query_too_short", length=len(stripped))
            return GuardrailResult(
                passed=False,
                reason="Query is too short. Please provide a more detailed question.",
            )

        # Check 3: Unsafe content.
        if _UNSAFE_RE.search(stripped):
            logger.warning("guardrail_unsafe_content", query=stripped[:50])
            return GuardrailResult(
                passed=False,
                reason=(
                    "This query contains content that cannot be answered. "
                    "Please ask a different question."
                ),
            )

        # Check 4: Off-topic (greetings, chitchat).
        if _OFF_TOPIC_RE.match(stripped):
            logger.info("guardrail_off_topic", query=stripped[:50])
            return GuardrailResult(
                passed=False,
                reason=(
                    "This appears to be a greeting rather than a knowledge "
                    "question. Please ask a specific question about a topic."
                ),
            )

        logger.info("guardrail_query_passed", query_length=len(stripped))
        return GuardrailResult(passed=True)

    def check_retrieval(
        self, context: Context, min_passages: int = 1
    ) -> GuardrailResult:
        """Post-retrieval guardrail: validate that retrieval found enough."""
        if len(context.passages) < min_passages:
            logger.warning(
                "guardrail_insufficient_retrieval",
                passages_found=len(context.passages),
                min_required=min_passages,
            )
            return GuardrailResult(
                passed=False,
                reason=(
                    "I couldn't find enough relevant information to answer "
                    "this question. Please try rephrasing or asking about "
                    "a different topic."
                ),
            )

        logger.info(
            "guardrail_retrieval_passed",
            passages_found=len(context.passages),
        )
        return GuardrailResult(passed=True)

    def check_generation(
        self,
        context: Context,
        response: GenerationResponse,
    ) -> GuardrailResult:
        """Post-generation guardrail: validate the generated answer."""
        # Check 1: Non-empty answer.
        if not response.answer_text.strip():
            logger.warning("guardrail_empty_generation")
            return GuardrailResult(
                passed=False,
                reason="The system could not generate an answer. Please try again.",
            )

        # Check 2: Grounding status.
        if not response.grounded:
            logger.info("guardrail_grounding_failed")
            return GuardrailResult(
                passed=False,
                reason=(
                    "I don't have enough reliable information to answer "
                    "this question confidently. Please try a different question."
                ),
            )

        # Check 3: Answer length sanity.
        answer = response.answer_text.strip()
        if len(answer) > 2000:
            logger.warning(
                "guardrail_answer_too_long",
                length=len(answer),
            )
            # Don't reject — just log. Long answers are unusual but valid.

        logger.info(
            "guardrail_generation_passed",
            answer_length=len(answer),
            grounded=response.grounded,
        )
        return GuardrailResult(passed=True)
