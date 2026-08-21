"""
STTProvider interface.

Provider: Sarvam AI (confirmed by project requirement).
Model: Saaras v3 — state-of-the-art ASR for 22 Indian languages + English.
Endpoint: https://api.sarvam.ai/speech-to-text

This interface is deliberately provider-agnostic so any
candidate can be benchmarked without touching callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict


class TranscriptionResult(BaseModel):
    """Structured transcription output from STT provider."""
    model_config = ConfigDict(extra="forbid")

    text: str
    lang: str
    confidence: float | None = None
    request_id: str | None = None
    provider: str = "unknown"
    latency_ms: float | None = None


class STTProvider(ABC):
    """Speech-to-text contract."""

    @abstractmethod
    def transcribe(
        self,
        audio_bytes: bytes,
        audio_format: str,
        language_hint: str | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio_bytes: Raw audio data.
            audio_format: MIME type or format string (e.g., "wav", "mp3").
            language_hint: Optional ISO 639-1 language code hint.

        Returns:
            TranscriptionResult with transcript text and metadata.

        Raises:
            STTFailureError: If transcription fails.
            InvalidAudioError: If audio is invalid or unsupported.
        """
        raise NotImplementedError
