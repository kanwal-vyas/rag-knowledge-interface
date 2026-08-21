"""
Stub STT provider for testing and development.

Returns deterministic transcriptions without calling any external API.
"""
from __future__ import annotations

import structlog

from app.services.stt import STTProvider, TranscriptionResult

logger = structlog.get_logger(__name__)


class StubSTTProvider(STTProvider):
    """
    Stub STT provider that returns pre-configured transcriptions.

    Used for unit tests and offline pipeline validation.
    """

    def __init__(
        self,
        text: str = "Hello, what is the capital of India?",
        lang: str = "en",
        confidence: float = 0.95,
    ) -> None:
        self.text = text
        self.lang = lang
        self.confidence = confidence

    def transcribe(
        self,
        audio_bytes: bytes,
        audio_format: str,
        language_hint: str | None = None,
    ) -> TranscriptionResult:
        if not audio_bytes:
            from app.core.errors import InvalidAudioError
            raise InvalidAudioError("Audio data is empty.")

        logger.info(
            "stub_stt_transcribe",
            audio_size=len(audio_bytes),
            format=audio_format,
        )

        return TranscriptionResult(
            text=self.text,
            lang=language_hint or self.lang,
            confidence=self.confidence,
            request_id="stub-request-id",
            provider="stub",
            latency_ms=0.1,
        )
