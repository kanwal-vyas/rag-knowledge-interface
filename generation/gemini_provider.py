"""
Google Gemini generation provider.

Uses the official google-genai SDK (replaces deprecated google-generativeai).

Models (as of 2026-08):
- gemini-3.6-flash: fast, cost-effective — default for RAG generation
- gemini-2.5-flash: previous generation
- gemini-2.5-pro: heavy reasoning (overkill for RAG)

The provider is configurable:
- model: defaults to gemini-3.6-flash
- api_key: from environment (GEMINI_API_KEY)
- temperature: 0.0 for deterministic grounding
- max_output_tokens: configurable

CRITICAL: The prompt instructs the model to answer ONLY from supplied
context. If context is insufficient, the model must refuse to answer.
"""
from __future__ import annotations

import os
import re
import time

import structlog

from app.core.errors import RateLimitError
from app.models.generation import Context, GenerationResponse
from generation.base import Generator

logger = structlog.get_logger(__name__)

# System instruction — shared with DeepSeek provider for consistency.
_INSTRUCTION = """\
You are a helpful multilingual assistant. Answer the user's question \
USING ONLY the provided context passages. Do not use any outside knowledge.

Rules:
1. Answer in the SAME LANGUAGE as the user's question.
2. Base your answer ONLY on the provided context passages.
3. If the context does not contain enough information to answer, say: \
"I don't have enough information to answer this question."
4. Do NOT fabricate facts or make up information.
5. Do NOT reference specific passage numbers or IDs in your answer.
6. Be concise and direct.
7. If the context contains conflicting information, note the conflict.
"""


def _build_user_prompt(context: Context) -> str:
    """Build the user prompt with context passages and query."""
    parts: list[str] = []
    parts.append("Context passages:")
    for i, result in enumerate(context.passages, 1):
        lang_label = result.passage.lang.value.upper()
        parts.append(f"[{i}] ({lang_label}) {result.passage.text}")
    parts.append("")
    parts.append(f"Question: {context.query.query_text}")
    return "\n".join(parts)


# Refusal phrases for grounding detection (shared logic).
_REFUSAL_PHRASES = [
    "i don't have enough information",
    "i do not have enough information",
    "i cannot answer",
    "i can't answer",
    "not enough information",
    "insufficient information",
    "no relevant information",
    "cannot provide",
    "can't provide",
    "not available in the provided",
    "not mentioned in the provided",
    "not found in the provided",
    "the provided context does not",
    "the context does not contain",
    "does not contain information",
    "no information in the context",
]


def _is_rate_limit(exc: Exception) -> bool:
    """Return True if the exception is a Gemini 429 RESOURCE_EXHAUSTED."""
    try:
        from google.genai.errors import ClientError
        if isinstance(exc, ClientError) and exc.code == 429:
            return True
    except (ImportError, AttributeError):
        pass
    msg = str(exc).lower()
    return "429" in msg and ("resource_exhausted" in msg or "quota" in msg)


def _extract_retry_delay(exc: Exception) -> int | None:
    """Extract the retry delay in seconds from a Gemini 429 response."""
    try:
        from google.genai.errors import ClientError
        if isinstance(exc, ClientError) and hasattr(exc, "response_json"):
            rj = exc.response_json
            if isinstance(rj, dict):
                error_obj = rj.get("error", {})
                if isinstance(error_obj, dict):
                    delay = error_obj.get("retryDelay")
                    if isinstance(delay, str):
                        m = re.search(r"(\d+)\s*s", delay)
                        if m:
                            return int(m.group(1))
    except (ImportError, AttributeError, TypeError, ValueError):
        pass
    # Fallback: parse the string representation.
    text = str(exc)
    m = re.search(r"retryDelay.*?(\d+)s", text)
    if m:
        return int(m.group(1))
    m = re.search(r"retry.delay.*?(\d+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


class GeminiGenerator(Generator):
    """
    Generation provider using Google Gemini API (google-genai SDK).

    Requires GEMINI_API_KEY environment variable.
    """

    def __init__(
        self,
        *,
        model: str = "gemini-3.6-flash",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout

        if not self.api_key:
            logger.warning(
                "gemini_no_api_key",
                note="GEMINI_API_KEY not set. Generation will fail.",
            )

    def generate(self, context: Context) -> GenerationResponse:
        """
        Generate a grounded answer from assembled context using Gemini.

        Returns a GenerationResponse with the answer text and grounding
        status based on refusal-phrase detection.
        """
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "google-genai package is required for Gemini generation. "
                "Install with: pip install google-genai"
            ) from exc

        client = genai.Client(api_key=self.api_key)

        user_prompt = _build_user_prompt(context)

        start_time = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=_INSTRUCTION,
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                ),
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            answer_text = (response.text or "").strip()

            # Extract token usage if available.
            tokens_used = None
            if response.usage_metadata:
                tokens_used = getattr(response.usage_metadata, "total_token_count", None)

            logger.info(
                "gemini_generation_complete",
                model=self.model,
                elapsed_ms=round(elapsed_ms, 1),
                answer_length=len(answer_text),
                tokens_used=tokens_used,
            )

            # Empty response → not grounded.
            if not answer_text:
                return GenerationResponse(answer_text="", grounded=False)

            # Grounding check via refusal-phrase detection.
            answer_lower = answer_text.lower()
            grounded = not any(phrase in answer_lower for phrase in _REFUSAL_PHRASES)

            return GenerationResponse(
                answer_text=answer_text,
                grounded=grounded,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # ── Detect 429 RESOURCE_EXHAUSTED ──
            retry_seconds = _extract_retry_delay(e)
            if retry_seconds is not None or _is_rate_limit(e):
                retry_display = f"{retry_seconds} seconds" if retry_seconds else "about 1 minute"
                logger.warning(
                    "gemini_rate_limited",
                    provider="gemini",
                    model=self.model,
                    http_status=429,
                    retry_delay_seconds=retry_seconds,
                    elapsed_ms=round(elapsed_ms, 1),
                )
                raise RateLimitError(
                    f"The knowledge service is temporarily at capacity. "
                    f"Please try again in {retry_display}."
                ) from e

            logger.error(
                "gemini_generation_failed",
                model=self.model,
                elapsed_ms=round(elapsed_ms, 1),
                error=str(e),
            )
            raise
