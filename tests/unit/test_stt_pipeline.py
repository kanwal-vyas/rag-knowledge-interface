"""
Tests for STT provider, audio validation, voice pipeline, and language mapping.

Covers:
- Audio validation (empty, oversized, unsupported format)
- Sarvam STT provider configuration
- Language code mapping (BCP-47 → ISO 639-1)
- Stub STT provider
- Voice pipeline integration
- Error handling (no STT provider, empty transcript)
- End-to-end voice → text pipeline
"""
from __future__ import annotations

import pytest

from app.core.errors import InvalidAudioError, STTFailureError
from app.services.sarvam_stt import (
    _BCP47_TO_ISO,
    _ISO_TO_BCP47,
    _map_language_code,
    _validate_audio,
)
from app.services.stub_stt import StubSTTProvider

# ---------------------------------------------------------------------------
# Audio Validation Tests
# ---------------------------------------------------------------------------

class TestAudioValidation:
    """Tests for audio input validation."""

    def test_empty_audio_rejected(self) -> None:
        """Empty audio bytes raise InvalidAudioError."""
        with pytest.raises(InvalidAudioError, match="empty"):
            _validate_audio(b"", "wav")

    def test_oversized_audio_rejected(self) -> None:
        """Audio exceeding 10MB is rejected."""
        large_audio = b"\x00" * (10 * 1024 * 1024 + 1)
        with pytest.raises(InvalidAudioError, match="too large"):
            _validate_audio(large_audio, "wav")

    def test_unsupported_format_rejected(self) -> None:
        """Unsupported audio format is rejected."""
        with pytest.raises(InvalidAudioError, match="Unsupported"):
            _validate_audio(b"some audio data", "xyz")

    def test_wav_accepted(self) -> None:
        """WAV format is accepted."""
        _validate_audio(b"RIFF....WAVEfmt", "wav")

    def test_mp3_accepted(self) -> None:
        """MP3 format is accepted."""
        _validate_audio(b"ID3...", "mp3")

    def test_flac_accepted(self) -> None:
        """FLAC format is accepted."""
        _validate_audio(b"fLaC...", "flac")

    def test_ogg_accepted(self) -> None:
        """OGG format is accepted."""
        _validate_audio(b"OggS...", "ogg")

    def test_case_insensitive_format(self) -> None:
        """Format check is case-insensitive."""
        _validate_audio(b"data", "WAV")
        _validate_audio(b"data", "Mp3")


# ---------------------------------------------------------------------------
# Language Code Mapping Tests
# ---------------------------------------------------------------------------

class TestLanguageCodeMapping:
    """Tests for BCP-47 to ISO 639-1 language code mapping."""

    def test_hindi_mapping(self) -> None:
        """BCP-47 hi-IN maps to ISO hi."""
        assert _map_language_code("hi-IN") == "hi"

    def test_english_mapping(self) -> None:
        """BCP-47 en-IN maps to ISO en."""
        assert _map_language_code("en-IN") == "en"

    def test_bengali_mapping(self) -> None:
        """BCP-47 bn-IN maps to ISO bn."""
        assert _map_language_code("bn-IN") == "bn"

    def test_tamil_mapping(self) -> None:
        """BCP-47 ta-IN maps to ISO ta."""
        assert _map_language_code("ta-IN") == "ta"

    def test_telugu_mapping(self) -> None:
        """BCP-47 te-IN maps to ISO te."""
        assert _map_language_code("te-IN") == "te"

    def test_gujarati_mapping(self) -> None:
        """BCP-47 gu-IN maps to ISO gu."""
        assert _map_language_code("gu-IN") == "gu"

    def test_kannada_mapping(self) -> None:
        """BCP-47 kn-IN maps to ISO kn."""
        assert _map_language_code("kn-IN") == "kn"

    def test_malayalam_mapping(self) -> None:
        """BCP-47 ml-IN maps to ISO ml."""
        assert _map_language_code("ml-IN") == "ml"

    def test_punjabi_mapping(self) -> None:
        """BCP-47 pa-IN maps to ISO pa."""
        assert _map_language_code("pa-IN") == "pa"

    def test_unknown_code_fallback(self) -> None:
        """Unknown BCP-47 code falls back to language part."""
        assert _map_language_code("xx-YY") == "xx"

    def test_none_defaults_to_english(self) -> None:
        """None language code defaults to English."""
        assert _map_language_code(None) == "en"

    def test_empty_string_defaults_to_english(self) -> None:
        """Empty language code defaults to English."""
        assert _map_language_code("") == "en"

    def test_all_mapped_codes_present(self) -> None:
        """All BCP-47 codes in the mapping have valid ISO counterparts."""
        for bcp47, iso in _BCP47_TO_ISO.items():
            assert len(iso) >= 2, f"ISO code {iso} for {bcp47} is too short"
            assert bcp47.endswith("-IN"), f"BCP-47 code {bcp47} doesn't end with -IN"

    def test_mapping_deterministic(self) -> None:
        """Language mapping is deterministic."""
        for code in ["hi-IN", "en-IN", "bn-IN", "ta-IN"]:
            assert _map_language_code(code) == _map_language_code(code)


# ---------------------------------------------------------------------------
# Stub STT Provider Tests
# ---------------------------------------------------------------------------

class TestStubSTTProvider:
    """Tests for the stub STT provider."""

    def test_stub_returns_transcript(self) -> None:
        """Stub returns configured transcript."""
        stub = StubSTTProvider(text="Hello world", lang="en")
        result = stub.transcribe(b"fake audio", "wav")
        assert result.text == "Hello world"
        assert result.lang == "en"
        assert result.provider == "stub"

    def test_stub_empty_audio_rejected(self) -> None:
        """Stub rejects empty audio."""
        stub = StubSTTProvider()
        with pytest.raises(InvalidAudioError):
            stub.transcribe(b"", "wav")

    def test_stub_respects_language_hint(self) -> None:
        """Stub uses language hint when provided."""
        stub = StubSTTProvider(text="test", lang="en")
        result = stub.transcribe(b"audio", "wav", language_hint="hi")
        assert result.lang == "hi"

    def test_stub_has_metadata(self) -> None:
        """Stub returns metadata fields."""
        stub = StubSTTProvider()
        result = stub.transcribe(b"audio", "wav")
        assert result.request_id is not None
        assert result.latency_ms is not None
        assert result.confidence is not None


# ---------------------------------------------------------------------------
# Voice Pipeline Tests
# ---------------------------------------------------------------------------

class TestVoicePipeline:
    """End-to-end voice pipeline tests."""

    def _make_pipeline(self, stt_text: str = "capital of India", stt_lang: str = "en"):
        """Create a TextPipeline with stub components."""
        from app.harness.text_pipeline import TextPipeline

        passage_store = {
            "p001": "The capital of India is New Delhi.",
            "p002": "Python is a programming language.",
        }

        class MockRetriever:
            def retrieve(self, query, mode, top_k):
                from app.models.retrieval import Language, Passage, RetrievalResult
                return [
                    RetrievalResult(
                        passage=Passage(
                            passage_id="p001",
                            text=passage_store["p001"],
                            lang=Language("en"),
                        ),
                        score=0.9,
                        source="mock",
                    ),
                ]

        from generation.deepseek_provider import StubGenerator

        return TextPipeline(
            retriever=MockRetriever(),
            generator=StubGenerator(answer="New Delhi.", grounded=True),
            passage_store=passage_store,
            stt_provider=StubSTTProvider(text=stt_text, lang=stt_lang),
        )

    def test_voice_pipeline_success(self) -> None:
        """Voice pipeline completes end-to-end."""
        pipeline = self._make_pipeline(stt_text="capital of India")
        result = pipeline.run_voice(b"fake audio data", "wav")
        assert result.response is not None
        assert result.response.answer_text
        assert result.latency.total_ms > 0
        assert result.latency.stt_ms is not None

    def test_voice_pipeline_no_stt_provider(self) -> None:
        """Voice pipeline fails without STT provider."""
        from app.harness.text_pipeline import TextPipeline

        pipeline = TextPipeline(
            retriever=None,
            generator=None,
            passage_store={},
            stt_provider=None,
        )
        with pytest.raises(STTFailureError, match="No STT provider"):
            pipeline.run_voice(b"audio", "wav")

    def test_voice_pipeline_empty_transcript(self) -> None:
        """Voice pipeline fails on empty transcript."""
        pipeline = self._make_pipeline(stt_text="")
        # Empty transcript after STT should raise STTFailureError.
        with pytest.raises(STTFailureError, match="empty transcript"):
            pipeline.run_voice(b"audio", "wav")

    def test_voice_pipeline_stt_failure(self) -> None:
        """Voice pipeline handles STT failure gracefully."""
        from app.harness.text_pipeline import TextPipeline

        class FailingSTT:
            def transcribe(self, audio_bytes, audio_format, language_hint=None):
                raise STTFailureError("API timeout")

        pipeline = TextPipeline(
            retriever=None,
            generator=None,
            passage_store={},
            stt_provider=FailingSTT(),
        )
        with pytest.raises(STTFailureError):
            pipeline.run_voice(b"audio", "wav")

    def test_voice_latency_recorded(self) -> None:
        """Voice pipeline records STT latency."""
        pipeline = self._make_pipeline()
        result = pipeline.run_voice(b"audio data", "wav")
        assert result.latency.stt_ms is not None
        assert result.latency.stt_ms >= 0
        assert result.latency.sparse_retrieval_ms is not None
        assert result.latency.context_assembly_ms is not None

    def test_voice_hindi_stt(self) -> None:
        """Voice pipeline handles Hindi STT output."""
        pipeline = self._make_pipeline(
            stt_text="भारत की राजधानी",
            stt_lang="hi",
        )
        result = pipeline.run_voice(b"hindi audio", "wav")
        assert result.response is not None
        assert result.latency.stt_ms is not None

    def _make_pipeline_with_recording_stt(self, received_hints: list):
        """Create a pipeline with a RecordingSTT that captures language_hint."""
        from app.harness.text_pipeline import TextPipeline
        from generation.deepseek_provider import StubGenerator

        class RecordingSTT:
            def transcribe(self, audio_bytes, audio_format, language_hint=None):
                received_hints.append(language_hint)
                from app.services.stt import TranscriptionResult
                return TranscriptionResult(
                    text="capital of India", lang="en", provider="stub",
                )

        passage_store = {"p001": "The capital of India is New Delhi."}

        class MockRetriever:
            def retrieve(self, query, mode, top_k):
                from app.models.retrieval import Language, Passage, RetrievalResult
                return [
                    RetrievalResult(
                        passage=Passage(
                            passage_id="p001",
                            text=passage_store["p001"],
                            lang=Language("en"),
                        ),
                        score=0.9,
                        source="mock",
                    ),
                ]

        return TextPipeline(
            retriever=MockRetriever(),
            generator=StubGenerator(answer="New Delhi.", grounded=True),
            passage_store=passage_store,
            stt_provider=RecordingSTT(),
        )

    def test_voice_pipeline_forwards_language_hint(self) -> None:
        """run_voice() forwards language_hint to the STT provider."""
        received_hints: list[str | None] = []
        pipeline = self._make_pipeline_with_recording_stt(received_hints)
        pipeline.run_voice(b"audio", "wav", language_hint="en")
        assert received_hints == ["en"]

    def test_voice_pipeline_no_hint_defaults_to_none(self) -> None:
        """run_voice() passes None when no language_hint is given."""
        received_hints: list[str | None] = []
        pipeline = self._make_pipeline_with_recording_stt(received_hints)
        pipeline.run_voice(b"audio", "wav")
        assert received_hints == [None]


# ---------------------------------------------------------------------------
# ISO → BCP-47 Reverse Mapping Tests
# ---------------------------------------------------------------------------

class TestISOTOBCP47:
    """Tests for ISO 639-1 to BCP-47 mapping used when sending hints to Sarvam."""

    def test_reverse_mapping_completeness(self) -> None:
        """Every BCP-47 code has a corresponding ISO entry."""
        for bcp47, iso in _BCP47_TO_ISO.items():
            assert iso in _ISO_TO_BCP47, f"Missing reverse mapping for {iso}"
            assert _ISO_TO_BCP47[iso] == bcp47

    def test_english_reverse(self) -> None:
        """en maps to en-IN."""
        assert _ISO_TO_BCP47["en"] == "en-IN"

    def test_hindi_reverse(self) -> None:
        """hi maps to hi-IN."""
        assert _ISO_TO_BCP47["hi"] == "hi-IN"

    def test_gujarati_reverse(self) -> None:
        """gu maps to gu-IN."""
        assert _ISO_TO_BCP47["gu"] == "gu-IN"

    def test_known_codes_roundtrip(self) -> None:
        """Every code roundtrips: BCP-47 → ISO → BCP-47."""
        for bcp47, iso in _BCP47_TO_ISO.items():
            assert _ISO_TO_BCP47.get(iso) == bcp47


# ---------------------------------------------------------------------------
# Sarvam Language Hint Integration Tests
# ---------------------------------------------------------------------------

class TestSarvamLanguageHint:
    """Tests for language_hint behavior in SarvamSTTProvider.

    Uses monkeypatch on httpx.post and io.BytesIO since transcribe() does
    local imports inside the method body.
    """

    @staticmethod
    def _patch_httpx(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
        """Patch httpx.post to capture data; return the sent_data dict."""
        sent_data: dict[str, str] = {}

        class FakeResponse:
            status_code = 200

            def json(self) -> dict[str, str]:
                return {"transcript": "hello", "language_code": "en-IN"}

        def fake_post(url: str, **kwargs: object) -> FakeResponse:  # noqa: ANN002
            sent_data.update(kwargs.get("data", {}))  # type: ignore[arg-type]
            return FakeResponse()

        monkeypatch.setattr("httpx.post", fake_post)
        return sent_data

    def test_en_hint_sends_en_IN(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When language_hint='en', the API request contains language_code=en-IN."""
        from app.services.sarvam_stt import SarvamSTTProvider

        sent_data = self._patch_httpx(monkeypatch)
        provider = SarvamSTTProvider(api_key="test-key")
        result = provider.transcribe(b"audio data", "wav", language_hint="en")
        assert result.text == "hello"
        assert sent_data.get("language_code") == "en-IN"

    def test_hi_hint_sends_hi_IN(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When language_hint='hi', the API request contains language_code=hi-IN."""
        from app.services.sarvam_stt import SarvamSTTProvider

        sent_data = self._patch_httpx(monkeypatch)
        provider = SarvamSTTProvider(api_key="test-key")
        result = provider.transcribe(b"audio data", "wav", language_hint="hi")
        assert result.text == "hello"
        assert sent_data.get("language_code") == "hi-IN"

    def test_gu_hint_sends_gu_IN(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When language_hint='gu', the API request contains language_code=gu-IN."""
        from app.services.sarvam_stt import SarvamSTTProvider

        sent_data = self._patch_httpx(monkeypatch)
        provider = SarvamSTTProvider(api_key="test-key")
        provider.transcribe(b"audio data", "wav", language_hint="gu")
        assert sent_data.get("language_code") == "gu-IN"

    def test_none_hint_sends_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When language_hint=None, language_code='unknown' triggers auto-detect."""
        from app.services.sarvam_stt import SarvamSTTProvider

        sent_data = self._patch_httpx(monkeypatch)
        provider = SarvamSTTProvider(api_key="test-key")
        provider.transcribe(b"audio data", "wav", language_hint=None)
        assert sent_data.get("language_code") == "unknown"

    def test_empty_hint_sends_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When language_hint='', language_code='unknown' triggers auto-detect."""
        from app.services.sarvam_stt import SarvamSTTProvider

        sent_data = self._patch_httpx(monkeypatch)
        provider = SarvamSTTProvider(api_key="test-key")
        provider.transcribe(b"audio data", "wav", language_hint="")
        assert sent_data.get("language_code") == "unknown"

    def test_unsupported_hint_sends_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When language_hint is unsupported, language_code='unknown' triggers auto-detect."""
        from app.services.sarvam_stt import SarvamSTTProvider

        sent_data = self._patch_httpx(monkeypatch)
        provider = SarvamSTTProvider(api_key="test-key")
        provider.transcribe(b"audio data", "wav", language_hint="xyz")
        assert sent_data.get("language_code") == "unknown"

    def test_hint_is_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Language hints are matched case-insensitively."""
        from app.services.sarvam_stt import SarvamSTTProvider

        sent_data = self._patch_httpx(monkeypatch)
        provider = SarvamSTTProvider(api_key="test-key")
        provider.transcribe(b"audio data", "wav", language_hint="HI")
        assert sent_data.get("language_code") == "hi-IN"
        assert sent_data.get("model") == "saaras:v3"
        assert sent_data.get("mode") == "transcribe"
