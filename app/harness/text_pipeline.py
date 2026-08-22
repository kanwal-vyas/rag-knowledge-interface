"""
Text query pipeline — end-to-end orchestration.

    text query
    → validation (guardrails)
    → retrieval (BM25 / dense / hybrid)
    → context assembly (dedup, parent expansion, budget)
    → generation (LLM)
    → grounding validation
    → guardrails (post-generation)
    → final response

Voice pipeline adds STT upstream:

    audio → STT → transcript → TextPipeline
"""
from __future__ import annotations

import time

import structlog

from app.core.errors import (
    InsufficientEvidenceError,
    InvalidRequestError,
    PipelineError,
    STTFailureError,
)
from app.harness.orchestrator import HarnessResult
from app.models.generation import GenerationResponse, GuardrailResult
from app.models.latency import LatencyBreakdown
from app.models.retrieval import Language, Query, RetrievalMode
from generation.context_assembly import assemble_context
from generation.grounding import check_grounding
from guardrails.implementation import GuardrailPipeline

logger = structlog.get_logger(__name__)


class TextPipeline:
    """
    End-to-end text query pipeline.

    Orchestrates: guardrails → retrieval → context assembly → generation → grounding.

    Components are injected via constructor, not hardcoded.
    """

    def __init__(
        self,
        *,
        retriever,
        generator,
        passage_store: dict[str, str],
        stt_provider=None,
        context_max_passages: int = 10,
        context_max_chars: int = 4000,
    ) -> None:
        """
        Args:
            retriever: A Retriever instance (BM25, dense, or hybrid).
            generator: A Generator instance (DeepSeek, stub, etc.).
            passage_store: Mapping passage_id → full text for parent expansion.
            stt_provider: Optional STTProvider for voice queries.
            context_max_passages: Max passages in assembled context.
            context_max_chars: Max total characters in assembled context.
        """
        self.retriever = retriever
        self.generator = generator
        self.passage_store = passage_store
        self.stt_provider = stt_provider
        self.context_max_passages = context_max_passages
        self.context_max_chars = context_max_chars
        self.guardrails = GuardrailPipeline()

    def run(
        self,
        query_text: str,
        lang: str = "en",
        mode: RetrievalMode = RetrievalMode.CROSS_LINGUAL,
        top_k: int = 10,
    ) -> HarnessResult:
        """
        Execute the full text query pipeline.

        Returns a HarnessResult with the response, guardrail status,
        and latency breakdown. On any unrecoverable error, returns
        a HarnessResult with the error set.
        """
        overall_start = time.perf_counter()
        latency = LatencyBreakdown(total_ms=0, is_estimated=False)

        try:
            # ── Stage 1: Pre-retrieval guardrails ──
            guardrail_start = time.perf_counter()
            guardrail_result = self.guardrails.check_query(query_text)
            if not guardrail_result.passed:
                latency.guardrail_ms = (
                    time.perf_counter() - guardrail_start
                ) * 1000
                latency.total_ms = (
                    time.perf_counter() - overall_start
                ) * 1000
                return HarnessResult(
                    response=None,
                    guardrail=guardrail_result,
                    latency=latency,
                )
            latency.guardrail_ms = (time.perf_counter() - guardrail_start) * 1000

            # ── Stage 2: Build query ──
            try:
                language = Language(lang)
            except ValueError as exc:
                langs = [lang_mem.value for lang_mem in Language]
                raise InvalidRequestError(
                    f"Unsupported language: {lang}. "
                    f"Supported: {langs}"
                ) from exc

            query = Query(query_text=query_text, lang=language)

            # ── Stage 3: Retrieval ──
            retrieval_start = time.perf_counter()
            try:
                results = self.retriever.retrieve(query, mode, top_k)
            except Exception as e:
                latency.sparse_retrieval_ms = (
                    time.perf_counter() - retrieval_start
                ) * 1000
                latency.total_ms = (
                    time.perf_counter() - overall_start
                ) * 1000
                raise PipelineError(f"Retrieval failed: {e}") from e

            latency.sparse_retrieval_ms = (
                time.perf_counter() - retrieval_start
            ) * 1000

            # ── Stage 4: Context assembly ──
            # Handle empty retrieval results gracefully.
            if not results:
                latency.total_ms = (
                    time.perf_counter() - overall_start
                ) * 1000
                return HarnessResult(
                    response=GenerationResponse(
                        answer_text="",
                        grounded=False,
                    ),
                    guardrail=GuardrailResult(
                        passed=False,
                        reason="No relevant passages found for this query.",
                    ),
                    latency=latency,
                )

            assembly_start = time.perf_counter()
            try:
                context = assemble_context(
                    query=query,
                    results=results,
                    passage_store=self.passage_store,
                    max_passages=self.context_max_passages,
                    max_chars=self.context_max_chars,
                )
            except ValueError as e:
                latency.context_assembly_ms = (
                    time.perf_counter() - assembly_start
                ) * 1000
                latency.total_ms = (
                    time.perf_counter() - overall_start
                ) * 1000
                raise InsufficientEvidenceError(str(e)) from e

            latency.context_assembly_ms = (
                time.perf_counter() - assembly_start
            ) * 1000

            # ── Stage 5: Post-retrieval guardrails ──
            guardrail_start = time.perf_counter()
            retrieval_guard = self.guardrails.check_retrieval(context)
            if not retrieval_guard.passed:
                latency.guardrail_ms = (
                    latency.guardrail_ms or 0
                ) + (time.perf_counter() - guardrail_start) * 1000
                latency.total_ms = (
                    time.perf_counter() - overall_start
                ) * 1000
                return HarnessResult(
                    response=None,
                    guardrail=retrieval_guard,
                    latency=latency,
                )
            latency.guardrail_ms = (
                latency.guardrail_ms or 0
            ) + (time.perf_counter() - guardrail_start) * 1000

            # ── Stage 6: Generation ──
            generation_start = time.perf_counter()
            try:
                response = self.generator.generate(context)
            except Exception as e:
                latency.generation_ms = (
                    time.perf_counter() - generation_start
                ) * 1000
                latency.total_ms = (
                    time.perf_counter() - overall_start
                ) * 1000
                raise PipelineError(f"Generation failed: {e}") from e

            latency.generation_ms = (
                time.perf_counter() - generation_start
            ) * 1000

            # ── Stage 7: Grounding validation ──
            grounding_start = time.perf_counter()
            response = check_grounding(context, response)
            latency.generation_ms = (
                latency.generation_ms or 0
            ) + (time.perf_counter() - grounding_start) * 1000

            # ── Stage 8: Post-generation guardrails ──
            guardrail_start = time.perf_counter()
            gen_guard = self.guardrails.check_generation(context, response)
            latency.guardrail_ms = (
                latency.guardrail_ms or 0
            ) + (time.perf_counter() - guardrail_start) * 1000

            latency.total_ms = (
                time.perf_counter() - overall_start
            ) * 1000

            try:
                logger.info(
                    "text_pipeline_complete",
                    query=query_text[:50],
                    lang=lang,
                    results_count=len(results),
                    context_passages=len(context.passages),
                    grounded=response.grounded,
                    total_ms=round(latency.total_ms, 1),
                )
            except UnicodeEncodeError:
                logger.info(
                    "text_pipeline_complete",
                    lang=lang,
                    results_count=len(results),
                    context_passages=len(context.passages),
                    grounded=response.grounded,
                    total_ms=round(latency.total_ms, 1),
                )

            return HarnessResult(
                response=response,
                guardrail=gen_guard,
                latency=latency,
            )

        except PipelineError:
            latency.total_ms = (
                time.perf_counter() - overall_start
            ) * 1000
            raise
        except Exception as e:
            latency.total_ms = (
                time.perf_counter() - overall_start
            ) * 1000
            # Log safely — Windows cp1252 console can't encode all Unicode
            try:
                logger.error("text_pipeline_unexpected_error", error=str(e))
            except UnicodeEncodeError:
                logger.error("text_pipeline_unexpected_error", error=repr(e))
            raise PipelineError(f"Pipeline failed: {e}") from e

    def run_voice(
        self,
        audio_bytes: bytes,
        audio_format: str,
        mode: RetrievalMode = RetrievalMode.CROSS_LINGUAL,
        top_k: int = 10,
        language_hint: str | None = None,
    ) -> HarnessResult:
        """
        Execute the voice query pipeline.

        Flow: audio → STT → transcript → TextPipeline.run()

        Args:
            audio_bytes: Raw audio data.
            audio_format: MIME type or format string (e.g., "wav", "mp3").
            mode: Retrieval mode (MONOLINGUAL or CROSS_LINGUAL).
            top_k: Number of retrieval results.
            language_hint: Optional ISO 639-1 language code hint forwarded to the STT provider.

        Returns:
            HarnessResult with response, guardrail status, and latency.
        """
        overall_start = time.perf_counter()
        latency = LatencyBreakdown(total_ms=0, is_estimated=False)

        # ── Stage 1: STT ──
        if not self.stt_provider:
            latency.total_ms = (
                time.perf_counter() - overall_start
            ) * 1000
            raise STTFailureError(
                "No STT provider configured. "
                "Pass stt_provider to TextPipeline constructor."
            )

        stt_start = time.perf_counter()
        try:
            transcription = self.stt_provider.transcribe(
                audio_bytes=audio_bytes,
                audio_format=audio_format,
                language_hint=language_hint,
            )
        except Exception as e:
            latency.stt_ms = (
                time.perf_counter() - stt_start
            ) * 1000
            latency.total_ms = (
                time.perf_counter() - overall_start
            ) * 1000
            raise STTFailureError(f"STT failed: {e}") from e

        latency.stt_ms = (time.perf_counter() - stt_start) * 1000

        if not transcription.text.strip():
            latency.total_ms = (
                time.perf_counter() - overall_start
            ) * 1000
            raise STTFailureError("STT returned empty transcript.")

        # ── Stage 2: Use detected language from STT ──
        lang = transcription.lang or "en"

        try:
            logger.info(
                "voice_pipeline_stt_complete",
                transcript=transcription.text[:50],
                lang=lang,
                stt_ms=round(latency.stt_ms or 0, 1),
            )
        except UnicodeEncodeError:
            logger.info(
                "voice_pipeline_stt_complete",
                lang=lang,
                stt_ms=round(latency.stt_ms or 0, 1),
            )

        # ── Stage 3: Delegate to text pipeline ──
        try:
            text_result = self.run(
                query_text=transcription.text,
                lang=lang,
                mode=mode,
                top_k=top_k,
            )
            latency.sparse_retrieval_ms = text_result.latency.sparse_retrieval_ms
            latency.context_assembly_ms = text_result.latency.context_assembly_ms
            latency.generation_ms = text_result.latency.generation_ms
            latency.guardrail_ms = text_result.latency.guardrail_ms
            latency.total_ms = (
                time.perf_counter() - overall_start
            ) * 1000
            return HarnessResult(
                response=text_result.response,
                guardrail=text_result.guardrail,
                latency=latency,
                transcript=transcription.text,
                detected_language=lang,
            )
        except PipelineError as e:
            # Preserve STT result when generation/retrieval fails.
            latency.total_ms = (
                time.perf_counter() - overall_start
            ) * 1000
            return HarnessResult(
                response=GenerationResponse(answer_text="", grounded=False),
                guardrail=GuardrailResult(passed=False, reason=e.detail),
                latency=latency,
                transcript=transcription.text,
                detected_language=lang,
            )
