"""
DeepSeek generation provider.

Uses DeepSeek's OpenAI-compatible API for answer generation.
The architecture does not specify a concrete generation model, but this
project is the DeepSeek implementation track. DeepSeek's API is
OpenAI-compatible, so we use the openai client library.

Current models (as of 2026-08):
- deepseek-v4-flash: fast, cost-effective — default for RAG generation
- deepseek-v4-pro: heavier reasoning model

The provider is configurable:
- model: defaults to deepseek-v4-flash
- api_key: from environment (DEEPSEEK_API_KEY)
- base_url: defaults to https://api.deepseek.com
- temperature: 0.0 for deterministic grounding
- max_tokens: configurable

CRITICAL: The prompt instructs the model to answer ONLY from supplied
context. If context is insufficient, the model must refuse to answer.
"""
from __future__ import annotations

import os
import time

import structlog

from app.models.generation import Context, GenerationResponse
from generation.base import Generator

logger = structlog.get_logger(__name__)

# System prompt for grounded RAG generation.
# The model is instructed to answer ONLY from provided context.
# If context is insufficient, it must say "I don't have enough information."
_SYSTEM_PROMPT = """\
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

    # Add context passages.
    parts.append("Context passages:")
    for i, result in enumerate(context.passages, 1):
        lang_label = result.passage.lang.value.upper()
        parts.append(f"[{i}] ({lang_label}) {result.passage.text}")

    # Add the query.
    parts.append("")
    parts.append(f"Question: {context.query.query_text}")

    return "\n".join(parts)


class DeepSeekGenerator(Generator):
    """
    Generation provider using DeepSeek's OpenAI-compatible API.

    Requires DEEPSEEK_API_KEY environment variable.
    """

    def __init__(
        self,
        *,
        model: str = "deepseek-v4-flash",
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        if not self.api_key:
            logger.warning(
                "deepseek_no_api_key",
                note="DEEPSEEK_API_KEY not set. Generation will fail.",
            )

    def generate(self, context: Context) -> GenerationResponse:
        """
        Generate a grounded answer from assembled context.

        Returns a GenerationResponse with the answer text and grounding
        status. The grounding check is a basic heuristic; the full
        grounding validation layer is in grounding.py.
        """
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required for DeepSeek generation. "
                "Install with: pip install openai"
            ) from exc

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

        user_prompt = _build_user_prompt(context)

        start_time = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            answer_text = response.choices[0].message.content or ""
            answer_text = answer_text.strip()

            logger.info(
                "deepseek_generation_complete",
                model=self.model,
                elapsed_ms=round(elapsed_ms, 1),
                answer_length=len(answer_text),
                tokens_used=(
                    response.usage.total_tokens if response.usage else None
                ),
            )

            # Basic grounding: check if model refused to answer.
            refusal_phrases = [
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
            answer_lower = answer_text.lower()
            grounded = not any(phrase in answer_lower for phrase in refusal_phrases)

            return GenerationResponse(
                answer_text=answer_text,
                grounded=grounded,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "deepseek_generation_failed",
                model=self.model,
                elapsed_ms=round(elapsed_ms, 1),
                error=str(e),
            )
            raise


class StubGenerator(Generator):
    """
    Stub generator for testing and development.

    Returns a deterministic response without calling any API.
    Used for unit tests and offline pipeline validation.
    """

    def __init__(self, answer: str = "", grounded: bool = True) -> None:
        self.answer = answer
        self.grounded = grounded

    def generate(self, context: Context) -> GenerationResponse:
        if self.answer:
            return GenerationResponse(
                answer_text=self.answer,
                grounded=self.grounded,
            )

        # Generate a stub answer from context.
        if not context.passages:
            return GenerationResponse(
                answer_text="I don't have enough information to answer this question.",
                grounded=False,
            )

        # Return a synthetic answer citing the first passage.
        first_passage = context.passages[0].passage.text
        answer = f"Based on the provided context: {first_passage[:200]}"
        return GenerationResponse(
            answer_text=answer,
            grounded=True,
        )
