"""
API routes wired to real TextPipeline.

POST /query: text query through full RAG pipeline
POST /voice/query: audio through STT then RAG pipeline
GET /health: service status and provider readiness
"""
from __future__ import annotations

import io
import json
import time
import uuid
from typing import Any

import structlog
from fastapi import APIRouter, File, Form, Request, UploadFile
from pydantic import BaseModel, ConfigDict

from app.core.config import get_settings
from app.core.errors import (
    PipelineError,
    STTFailureError,
)
from app.harness.orchestrator import HarnessResult
from app.models.latency import LatencyBreakdown
from app.models.retrieval import RetrievalMode

logger = structlog.get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Audio format normalization
# ---------------------------------------------------------------------------

# MIME type / extension aliases → canonical format name.
_AUDIO_FORMAT_ALIASES: dict[str, str] = {
    "mpeg": "mp3",
    "mpeg3": "mp3",
    "x-mpeg-3": "mp3",
    "x-mp3": "mp3",
    "x-wav": "wav",
    "wave": "wav",
    "x-aac": "aac",
    "x-flac": "flac",
    "x-aiff": "aiff",
    "x-ms-wma": "wma",
    "x-m4a": "mp4",
    "pcm_s16le": "pcm",
    "pcm_l16": "pcm",
    "pcm_raw": "pcm",
}


def _normalize_audio_format(raw: str) -> str:
    """Normalize a MIME type or file extension to a canonical audio format.

    Handles browser Content-Types like ``audio/webm;codecs=opus`` and
    maps common MIME aliases to the short names expected by Sarvam.

    Examples::
        "audio/webm;codecs=opus" → "webm"
        "webm;codecs=opus"       → "webm"
        "audio/ogg;codecs=opus"  → "ogg"
        "audio/mpeg"             → "mp3"
        "audio/wav"              → "wav"
        "webm"                   → "webm"
    """
    fmt = raw.strip().lower()

    # 1. Strip MIME prefix (e.g. "audio/").
    if "/" in fmt:
        fmt = fmt.rsplit("/", 1)[-1]

    # 2. Strip codec parameters (e.g. ";codecs=opus").
    fmt = fmt.split(";", 1)[0].strip()

    # 3. Strip any trailing whitespace or semicolons.
    fmt = fmt.strip().rstrip("; ")

    # 4. Map known aliases to canonical names.
    fmt = _AUDIO_FORMAT_ALIASES.get(fmt, fmt)

    return fmt


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    environment: str
    providers: dict[str, str]


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_text: str
    lang: str = "en"
    retrieval_mode: RetrievalMode = RetrievalMode.CROSS_LINGUAL
    top_k: int = 10


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    grounded: bool
    request_id: str
    latency: dict[str, Any]


class GuardrailResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    reason: str | None = None
    request_id: str


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error_code: str
    detail: str
    request_id: str


class VoiceQueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transcript: str
    detected_language: str
    answer: str
    grounded: bool
    request_id: str
    latency: dict[str, Any]


# ---------------------------------------------------------------------------
# Pipeline singleton (lazily initialized)
# ---------------------------------------------------------------------------

_pipeline = None  # TextPipeline | None, typed lazily


def _get_pipeline() -> Any:
    """Get or create the TextPipeline singleton."""
    global _pipeline  # noqa: PLW0603
    if _pipeline is not None:
        return _pipeline

    from app.harness.text_pipeline import TextPipeline
    from retrieval.sparse.bm25_index import BM25Index
    from retrieval.sparse.bm25_retriever import BM25SparseRetriever

    settings = get_settings()

    passage_store: dict[str, str] = {}
    bm25_index: BM25Index

    if settings.demo_mode:
        # Load prebuilt demo index from artifacts/demo/
        from pathlib import Path
        index_path = Path(settings.demo_index_path)
        store_path = Path(settings.demo_passage_store_path)

        if index_path.exists() and store_path.exists():
            logger.info("loading_demo_index", index=str(index_path))
            bm25_index = BM25Index.load(index_path)
            passage_store = json.loads(store_path.read_text(encoding="utf-8"))
            logger.info(
                "demo_index_loaded",
                bm25_docs=bm25_index.get_stats()["document_count"],
                passages=len(passage_store),
            )
        else:
            logger.warning(
                "demo_index_not_found",
                index=str(index_path),
                store=str(store_path),
                note="Run: python -m scripts.prepare_demo_index",
            )
            # Fall back to empty index
            bm25_index = BM25Index()
            bm25_index.add_document("_placeholder", "placeholder", "en")
            bm25_index.build()
    else:
        # Empty index for non-demo mode
        bm25_index = BM25Index()
        bm25_index.add_document("_placeholder", "placeholder", "en")
        bm25_index.build()

    retriever = BM25SparseRetriever(index=bm25_index)

    # Generator — Gemini (primary) → DeepSeek (fallback) → Stub.
    generator: Any
    if settings.gemini_api_key:
        from generation.gemini_provider import GeminiGenerator
        generator = GeminiGenerator(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model_name,
        )
        logger.info("generator_selected", provider="gemini", model=settings.gemini_model_name)
    elif settings.generation_api_key:
        from generation.deepseek_provider import DeepSeekGenerator
        generator = DeepSeekGenerator(
            api_key=settings.generation_api_key,
            model=settings.generation_model_name or "deepseek-v4-flash",
        )
        logger.info("generator_selected", provider="deepseek", model=settings.generation_model_name)
    else:
        from generation.deepseek_provider import StubGenerator
        generator = StubGenerator()
        logger.warning("no_generation_key", note="Using StubGenerator")

    # STT provider — Sarvam if API key available, otherwise None.
    stt_provider: Any = None
    sarvam_key = settings.stt_api_key
    if sarvam_key:
        from app.services.sarvam_stt import SarvamSTTProvider
        stt_provider = SarvamSTTProvider(api_key=sarvam_key)
    else:
        logger.warning("no_sarvam_key", note="Voice queries will fail")

    _pipeline = TextPipeline(
        retriever=retriever,
        generator=generator,
        passage_store=passage_store,
        stt_provider=stt_provider,
    )

    return _pipeline


def _format_latency(latency: LatencyBreakdown) -> dict[str, Any]:
    """Convert LatencyBreakdown to a JSON-serializable dict."""
    return {
        "total_ms": round(latency.total_ms, 1),
        "stt_ms": round(latency.stt_ms, 1) if latency.stt_ms is not None else None,
        "sparse_retrieval_ms": (
            round(latency.sparse_retrieval_ms, 1)
            if latency.sparse_retrieval_ms is not None
            else None
        ),
        "dense_retrieval_ms": (
            round(latency.dense_retrieval_ms, 1)
            if latency.dense_retrieval_ms is not None
            else None
        ),
        "context_assembly_ms": (
            round(latency.context_assembly_ms, 1)
            if latency.context_assembly_ms is not None
            else None
        ),
        "generation_ms": (
            round(latency.generation_ms, 1)
            if latency.generation_ms is not None
            else None
        ),
        "guardrail_ms": (
            round(latency.guardrail_ms, 1)
            if latency.guardrail_ms is not None
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Service health check with provider readiness."""
    settings = get_settings()

    providers = {
        "generation": (
            "gemini" if settings.gemini_api_key
            else ("deepseek" if settings.generation_api_key else "stub")
        ),
        "stt": "sarvam" if settings.stt_api_key else "unavailable",
        "demo_mode": str(settings.demo_mode).lower(),
    }

    return HealthResponse(
        status="ok",
        environment=settings.environment,
        providers=providers,
    )


@router.post("/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    """Execute a text RAG query through the full pipeline."""
    request_id = str(uuid.uuid4())

    try:
        pipeline = _get_pipeline()
        result: HarnessResult = pipeline.run(
            query_text=body.query_text,
            lang=body.lang,
            mode=body.retrieval_mode,
            top_k=body.top_k,
        )
    except PipelineError as e:
        logger.error("query_pipeline_error", error_code=e.error_code, detail=e.detail)
        return QueryResponse(
            answer="",
            grounded=False,
            request_id=request_id,
            latency={"total_ms": 0, "error": e.detail},
        )
    except Exception as e:
        logger.error("query_unexpected_error", error=str(e))
        return QueryResponse(
            answer="",
            grounded=False,
            request_id=request_id,
            latency={"total_ms": 0, "error": str(e)},
        )

    # Check guardrail refusal or generation error.
    if result.guardrail and not result.guardrail.passed:
        latency_dict = _format_latency(result.latency)
        if result.guardrail.reason:
            latency_dict["error"] = result.guardrail.reason
        return QueryResponse(
            answer="",
            grounded=False,
            request_id=request_id,
            latency=latency_dict,
        )

    answer = result.response.answer_text if result.response else ""
    grounded = result.response.grounded if result.response else False

    return QueryResponse(
        answer=answer,
        grounded=grounded,
        request_id=request_id,
        latency=_format_latency(result.latency),
    )


@router.post("/voice/query", response_model=VoiceQueryResponse)
async def voice_query(
    request: Request,
    file: UploadFile = File(...),  # noqa: B008
    lang: str = Form(default=""),  # noqa: B008
    retrieval_mode: RetrievalMode = Form(default=RetrievalMode.CROSS_LINGUAL),  # noqa: B008
    top_k: int = Form(default=10),  # noqa: B008
) -> VoiceQueryResponse:
    """Execute a voice RAG query: audio → STT → retrieval → generation."""
    request_id = str(uuid.uuid4())

    # Read audio bytes.
    audio_bytes = await file.read()
    logger.info(
        "voice_audio_received",
        byte_length=len(audio_bytes),
        content_type=file.content_type,
        filename=file.filename,
        first_16_hex=audio_bytes[:16].hex() if len(audio_bytes) >= 16 else audio_bytes.hex(),
        last_8_hex=audio_bytes[-8:].hex() if len(audio_bytes) >= 8 else audio_bytes.hex(),
    )
    if not audio_bytes:
        return VoiceQueryResponse(
            transcript="",
            detected_language="",
            answer="",
            grounded=False,
            request_id=request_id,
            latency={"total_ms": 0, "error": "Empty audio file"},
        )

    # Determine audio format from content type or filename,
    # then normalize (e.g. "audio/webm;codecs=opus" → "webm").
    raw_format = "wav"
    if file.content_type and "/" in file.content_type:
        raw_format = file.content_type
    elif file.filename and "." in file.filename:
        raw_format = file.filename
    audio_format = _normalize_audio_format(raw_format)

    # ── WebM structural inspection ──
    if audio_format == "webm":
        from app.services.webm_inspect import inspect_webm
        webm_info = inspect_webm(audio_bytes)
        tracks_info = []
        for t in webm_info.tracks:
            tracks_info.append({"codec": t.codec_id, "rate": t.sample_rate, "ch": t.channels})
        logger.info(
            "voice_webm_inspect",
            valid=webm_info.valid,
            doctype=webm_info.doctype,
            duration_ms=webm_info.duration_ms,
            timecode_scale=webm_info.timecode_scale,
            total_bytes=webm_info.total_bytes,
            tracks=tracks_info,
            first_16_hex=webm_info.first_bytes_hex,
            error=webm_info.error,
        )

    logger.info(
        "voice_audio_normalized",
        raw_format=raw_format,
        audio_format=audio_format,
        byte_length=len(audio_bytes),
    )

    try:
        pipeline = _get_pipeline()
        result: HarnessResult = pipeline.run_voice(
            audio_bytes=audio_bytes,
            audio_format=audio_format,
            mode=retrieval_mode,
            top_k=top_k,
            language_hint=lang or None,
        )
    except STTFailureError as e:
        logger.error("voice_stt_error", detail=e.detail)
        return VoiceQueryResponse(
            transcript="",
            detected_language="",
            answer="",
            grounded=False,
            request_id=request_id,
            latency={"total_ms": 0, "error": e.detail},
        )
    except PipelineError as e:
        logger.error("voice_pipeline_error", error_code=e.error_code, detail=e.detail)
        return VoiceQueryResponse(
            transcript="",
            detected_language="",
            answer="",
            grounded=False,
            request_id=request_id,
            latency={"total_ms": 0, "error": e.detail},
        )
    except Exception as e:
        logger.error("voice_unexpected_error", error=str(e))
        return VoiceQueryResponse(
            transcript="",
            detected_language="",
            answer="",
            grounded=False,
            request_id=request_id,
            latency={"total_ms": 0, "error": str(e)},
        )

    # Extract transcript from latency metadata (STT stores it in the log).
    # The pipeline doesn't return the transcript directly, so we need to
    # get it from the STT result. For now, we'll extract it from the
    # guardrail/response context.
    answer = result.response.answer_text if result.response else ""
    grounded = result.response.grounded if result.response else False

    # Guardrail refusal or generation error.
    if result.guardrail and not result.guardrail.passed:
        latency_dict = _format_latency(result.latency)
        if result.guardrail.reason:
            latency_dict["error"] = result.guardrail.reason
        return VoiceQueryResponse(
            transcript=result.transcript or "",
            detected_language=result.detected_language or "",
            answer="",
            grounded=False,
            request_id=request_id,
            latency=latency_dict,
        )

    return VoiceQueryResponse(
        transcript=result.transcript or "",
        detected_language=result.detected_language or "en",
        answer=answer,
        grounded=grounded,
        request_id=request_id,
        latency=_format_latency(result.latency),
    )


# ---------------------------------------------------------------------------
# Diagnostic endpoint: isolate WebM/Opus format issues
# ---------------------------------------------------------------------------

@router.post("/diagnose/audio")
async def diagnose_audio(
    file: UploadFile = File(...),  # noqa: B008
    lang: str = Form(default="en"),  # noqa: B008
) -> dict[str, Any]:
    """Diagnostic endpoint to test audio format compatibility with Sarvam.

    Uploads the same audio in two formats:
    1. Original WebM/Opus (as received from browser)
    2. Extracted Opus rewrapped as OGG/Opus

    Both are sent to Sarvam and the results are compared.
    """
    audio_bytes = await file.read()
    request_id = str(uuid.uuid4())

    # Determine format.
    raw_format = "wav"
    if file.content_type and "/" in file.content_type:
        raw_format = file.content_type
    elif file.filename and "." in file.filename:
        raw_format = file.filename
    audio_format = _normalize_audio_format(raw_format)

    # WebM structural inspection.
    webm_diag: dict[str, Any] = {}
    if audio_format == "webm":
        from app.services.webm_inspect import inspect_webm
        wi = inspect_webm(audio_bytes)
        webm_diag = {
            "valid": wi.valid,
            "doctype": wi.doctype,
            "duration_ms": wi.duration_ms,
            "timecode_scale": wi.timecode_scale,
            "total_bytes": wi.total_bytes,
            "tracks": [
                {"codec": t.codec_id, "rate": t.sample_rate, "ch": t.channels}
                for t in wi.tracks
            ],
            "first_16_hex": wi.first_bytes_hex,
            "last_8_hex": wi.last_bytes_hex,
            "error": wi.error,
        }

    # Direct Sarvam test with original WebM.
    webm_result = await _direct_sarvam_test(
        audio_bytes, audio_format, lang, f"audio.{audio_format}",
    )

    # Extract Opus and rewrap as OGG.
    ogg_diag: dict[str, Any] = {}
    ogg_result: dict[str, Any] = {}
    if audio_format == "webm":
        from app.services.opus_extract import webm_to_ogg_opus
        ogg_bytes, ogg_extract_diag = webm_to_ogg_opus(audio_bytes)
        ogg_diag = ogg_extract_diag

        if ogg_bytes:
            ogg_result = await _direct_sarvam_test(
                ogg_bytes, "ogg", lang, "audio.ogg",
            )
        else:
            ogg_result = {"error": ogg_extract_diag.get("error", "Extraction failed")}

    return {
        "request_id": request_id,
        "original_format": audio_format,
        "original_bytes": len(audio_bytes),
        "webm_inspection": webm_diag,
        "sarvam_webm_result": webm_result,
        "ogg_extraction": ogg_diag,
        "sarvam_ogg_result": ogg_result,
    }


async def _direct_sarvam_test(
    audio_bytes: bytes,
    audio_format: str,
    lang: str,
    filename: str,
) -> dict[str, Any]:
    """Send audio directly to Sarvam and return the raw result.

    Does NOT go through the pipeline — sends directly to the API.
    """
    settings = get_settings()
    api_key = settings.stt_api_key
    if not api_key:
        return {"error": "SARVAM_API_KEY not configured"}

    try:
        import httpx
    except ImportError:
        return {"error": "httpx not installed"}

    # Build BCP-47 language code.
    from app.services.sarvam_stt import _ISO_TO_BCP47, _SARVAM_STT_URL
    hint = (lang or "").lower().strip()
    bcp47 = _ISO_TO_BCP47.get(hint) if hint else None
    language_code = bcp47 if bcp47 else "unknown"

    # Build MIME type.
    if audio_format == "webm":
        mime = "audio/webm"
    elif audio_format == "ogg":
        mime = "audio/ogg"
    elif audio_format == "wav":
        mime = "audio/wav"
    else:
        mime = f"audio/{audio_format}"

    files = {
        "file": (filename, io.BytesIO(audio_bytes), mime),
    }
    data: dict[str, str] = {
        "model": "saaras:v3",
        "mode": "transcribe",
        "language_code": language_code,
    }
    headers = {"api-subscription-key": api_key}

    logger.info(
        "diagnose_sarvam_request",
        audio_bytes=len(audio_bytes),
        audio_format=audio_format,
        filename=filename,
        mime=mime,
        language_code=language_code,
    )

    start = time.time()
    try:
        resp = httpx.post(
            _SARVAM_STT_URL,
            files=files,
            data=data,
            headers=headers,
            timeout=30.0,
        )
        elapsed_ms = round((time.time() - start) * 1000, 1)

        if resp.status_code == 200:
            result = resp.json()
            logger.info(
                "diagnose_sarvam_success",
                audio_format=audio_format,
                transcript=result.get("transcript", "")[:80],
                language_code=result.get("language_code"),
                elapsed_ms=elapsed_ms,
            )
            return {
                "status": 200,
                "transcript": result.get("transcript", ""),
                "language_code": result.get("language_code"),
                "request_id": result.get("request_id"),
                "elapsed_ms": elapsed_ms,
                "all_keys": list(result.keys()),
            }
        else:
            logger.warning(
                "diagnose_sarvam_error",
                audio_format=audio_format,
                status=resp.status_code,
                body=resp.text[:200],
            )
            return {
                "status": resp.status_code,
                "error": resp.text[:500],
                "elapsed_ms": elapsed_ms,
            }
    except Exception as e:
        elapsed_ms = round((time.time() - start) * 1000, 1)
        return {"error": str(e), "elapsed_ms": elapsed_ms}
