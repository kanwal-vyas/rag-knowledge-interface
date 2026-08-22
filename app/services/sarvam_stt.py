"""
Sarvam AI Speech-to-Text provider.

Uses Sarvam's Saaras v3 model for multilingual ASR across 22 Indian
languages + English.

API: https://api.sarvam.ai/speech-to-text
Model: saaras:v3
Modes: transcribe, translate, verbatim, translit, codemix

Authentication: api-subscription-key header (SARVAM_API_KEY env var).

Supported audio formats: WAV, MP3, AAC, FLAC, OGG, AIFF, AMR, WMA, WEBM, PCM.
Max duration: 30 seconds (REST API).

BCP-47 language codes returned by Sarvam (e.g., hi-IN, en-IN) are mapped
to ISO 639-1 for internal use.
"""
from __future__ import annotations

import io
import os
import time

import structlog

from app.core.errors import (
    InvalidAudioError,
    STTFailureError,
)
from app.services.stt import STTProvider, TranscriptionResult

logger = structlog.get_logger(__name__)

# Sarvam API endpoint.
_SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"

# Supported audio formats (MIME types and extensions).
_SUPPORTED_FORMATS = frozenset({
    "wav", "x-wav", "wave",
    "mp3", "mpeg", "mpeg3", "x-mpeg-3", "x-mp3",
    "aac", "x-aac",
    "flac", "x-flac",
    "ogg", "opus",
    "aiff", "x-aiff",
    "amr",
    "x-ms-wma",
    "webm",
    "mp4", "x-m4a",
    "pcm_s16le", "pcm_l16", "pcm_raw",
})

# BCP-47 to ISO 639-1 mapping for supported languages.
_BCP47_TO_ISO: dict[str, str] = {
    "hi-IN": "hi", "en-IN": "en", "bn-IN": "bn",
    "ta-IN": "ta", "te-IN": "te", "mr-IN": "mr",
    "gu-IN": "gu", "kn-IN": "kn", "ml-IN": "ml",
    "pa-IN": "pa", "or-IN": "or", "as-IN": "as",
    "ne-IN": "ne", "ur-IN": "ur",
    "sa-IN": "sa", "sat-IN": "sat",
    "mai-IN": "mai", "mni-IN": "mni",
    "doi-IN": "doi", "ks-IN": "ks",
    "sd-IN": "sd", "gom-IN": "gom",
}

# Reverse mapping: ISO 639-1 → BCP-47 (for sending language hints to Sarvam).
_ISO_TO_BCP47: dict[str, str] = {v: k for k, v in _BCP47_TO_ISO.items()}

# Maximum audio size: 10MB.
_MAX_AUDIO_SIZE = 10 * 1024 * 1024

# Maximum audio duration for REST API: 30 seconds.
_MAX_DURATION_SECONDS = 30


def _map_language_code(bcp47_code: str | None) -> str:
    """Map BCP-47 language code to ISO 639-1."""
    if not bcp47_code:
        return "en"  # Default to English if no language detected.

    # Try exact match first.
    iso = _BCP47_TO_ISO.get(bcp47_code)
    if iso:
        return iso

    # Try language-only part (e.g., "hi-IN" → "hi").
    lang_part = bcp47_code.split("-")[0].lower()
    if lang_part in _BCP47_TO_ISO.values():
        return lang_part

    logger.warning("unknown_language_code", bcp47=bcp47_code)
    return lang_part if lang_part else "en"


def _validate_audio(audio_bytes: bytes, audio_format: str) -> None:
    """Validate audio input before sending to Sarvam API."""
    if not audio_bytes:
        raise InvalidAudioError("Audio data is empty.")

    if len(audio_bytes) > _MAX_AUDIO_SIZE:
        raise InvalidAudioError(
            f"Audio file too large: {len(audio_bytes)} bytes "
            f"(max: {_MAX_AUDIO_SIZE} bytes)."
        )

    fmt_lower = audio_format.lower().strip()
    if fmt_lower not in _SUPPORTED_FORMATS:
        raise InvalidAudioError(
            f"Unsupported audio format: {audio_format}. "
            f"Supported: {sorted(_SUPPORTED_FORMATS)}"
        )


class SarvamSTTProvider(STTProvider):
    """
    Sarvam AI STT provider using Saaras v3.

    Requires SARVAM_API_KEY environment variable.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "saaras:v3",
        mode: str = "transcribe",
        timeout: float = 30.0,
    ) -> None:
        """
        Args:
            api_key: Sarvam API subscription key. Falls back to SARVAM_API_KEY env var.
            model: Sarvam model identifier. Default: saaras:v3.
            mode: Transcription mode (transcribe, translate, verbatim, translit, codemix).
            timeout: HTTP request timeout in seconds.
        """
        self.api_key = api_key or os.environ.get("SARVAM_API_KEY", "")
        self.model = model
        self.mode = mode
        self.timeout = timeout

        if not self.api_key:
            logger.warning(
                "sarvam_no_api_key",
                note="SARVAM_API_KEY not set. STT will fail.",
            )

    def transcribe(
        self,
        audio_bytes: bytes,
        audio_format: str,
        language_hint: str | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio using Sarvam Saaras v3.

        Args:
            audio_bytes: Raw audio data.
            audio_format: MIME type or format string.
            language_hint: Optional ISO 639-1 language code hint.

        Returns:
            TranscriptionResult with transcript and metadata.

        Raises:
            InvalidAudioError: If audio is invalid.
            STTFailureError: If API call fails.
        """
        # Validate input.
        _validate_audio(audio_bytes, audio_format)

        if not self.api_key:
            raise STTFailureError(
                "SARVAM_API_KEY is not configured. "
                "Set the environment variable or pass api_key to the constructor."
            )

        try:
            import httpx
        except ImportError as exc:
            raise STTFailureError(
                "httpx package is required for Sarvam STT. "
                "Install with: pip install httpx"
            ) from exc

        # Convert WebM/Opus to OGG/Opus for more reliable Sarvam transcription.
        # Chrome MediaRecorder produces WebM/Opus that Sarvam sometimes mishandles.
        # Extracting the raw Opus frames and rewrapping as OGG gives Sarvam a
        # format it processes correctly, while preserving the exact audio content.
        send_bytes = audio_bytes
        send_format = audio_format
        if audio_format == "webm":
            try:
                from app.services.opus_extract import webm_to_ogg_opus
                ogg_bytes, diag = webm_to_ogg_opus(audio_bytes)
                packet_count = diag.get("opus_packets_found", 0)
                if ogg_bytes and packet_count > 0:
                    send_bytes = ogg_bytes
                    send_format = "ogg"
                    logger.info(
                        "webm_to_ogg_converted",
                        original_bytes=len(audio_bytes),
                        ogg_bytes=len(ogg_bytes),
                        opus_packets=packet_count,
                    )
                else:
                    logger.warning(
                        "webm_to_ogg_failed",
                        note="No Opus packets extracted; sending original WebM.",
                        diag_error=diag.get("error"),
                    )
            except Exception as e:
                logger.warning(
                    "webm_to_ogg_error",
                    error=str(e),
                    note="Conversion failed; sending original WebM.",
                )

        # Prepare the multipart form data.
        # Sarvam expects: file, model, mode, and optionally language_code.
        files = {
            "file": (f"audio.{send_format}", io.BytesIO(send_bytes), f"audio/{send_format}"),
        }
        data: dict[str, str] = {
            "model": self.model,
            "mode": self.mode,
        }

        # Map language hint to BCP-47 language_code for Sarvam.
        # When no hint is provided, send "unknown" so Sarvam auto-detects.
        hint = (language_hint or "").lower().strip()
        bcp47 = _ISO_TO_BCP47.get(hint) if hint else None
        if bcp47:
            data["language_code"] = bcp47
            logger.info("sarvam_language_hint", iso=hint, bcp47=bcp47)
        else:
            data["language_code"] = "unknown"
            if hint:
                logger.warning(
                    "sarvam_unsupported_language_hint",
                    iso=hint,
                    note="Unknown hint; Sarvam will auto-detect.",
                )

        headers = {
            "api-subscription-key": self.api_key,
        }

        start_time = time.perf_counter()
        logger.info(
            "sarvam_request_sending",
            audio_bytes=len(audio_bytes),
            audio_format=audio_format,
            filename=files["file"][0],
            content_type=files["file"][2],
            first_16_hex=audio_bytes[:16].hex() if len(audio_bytes) >= 16 else audio_bytes.hex(),
            last_8_hex=audio_bytes[-8:].hex() if len(audio_bytes) >= 8 else audio_bytes.hex(),
            model=self.model,
            mode=self.mode,
            language_code=data.get("language_code"),
        )
        try:
            response = httpx.post(
                _SARVAM_STT_URL,
                files=files,
                data=data,
                headers=headers,
                timeout=self.timeout,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            if response.status_code == 200:
                result = response.json()
                transcript = result.get("transcript", "")
                lang_code = result.get("language_code")
                request_id = result.get("request_id")

                iso_lang = _map_language_code(lang_code)

                logger.info(
                    "sarvam_stt_success",
                    language=iso_lang,
                    bcp47=lang_code,
                    transcript=transcript[:80],
                    transcript_length=len(transcript),
                    request_id=request_id,
                    elapsed_ms=round(elapsed_ms, 1),
                    response_keys=list(result.keys()),
                )

                return TranscriptionResult(
                    text=transcript,
                    lang=iso_lang,
                    confidence=None,  # Sarvam doesn't return confidence in v3.
                    request_id=request_id,
                    provider="sarvam",
                    latency_ms=round(elapsed_ms, 1),
                )

            elif response.status_code == 422:
                raise InvalidAudioError(
                    f"Sarvam API rejected audio: {response.text}"
                )
            elif response.status_code == 429:
                raise STTFailureError(
                    "Sarvam API rate limit exceeded. Please retry later."
                )
            elif response.status_code == 403:
                raise STTFailureError(
                    "Sarvam API authentication failed. Check SARVAM_API_KEY."
                )
            elif response.status_code == 503:
                raise STTFailureError(
                    "Sarvam API service overloaded. Please retry with backoff."
                )
            else:
                raise STTFailureError(
                    f"Sarvam API error {response.status_code}: {response.text}"
                )

        except (InvalidAudioError, STTFailureError):
            raise
        except httpx.TimeoutException:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            raise STTFailureError(
                f"Sarvam API timeout after {elapsed_ms:.0f}ms."
            ) from None
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "sarvam_stt_error",
                elapsed_ms=round(elapsed_ms, 1),
                error=str(e),
            )
            raise STTFailureError(f"Sarvam STT failed: {e}") from e
