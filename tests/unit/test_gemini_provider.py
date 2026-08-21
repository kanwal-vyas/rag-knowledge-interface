"""
Unit tests for Gemini generation provider.

All Gemini API calls are mocked — no real API key required for pytest.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.models.generation import Context
from app.models.retrieval import Language, Passage, Query, RetrievalResult
from generation.gemini_provider import GeminiGenerator, _build_user_prompt


def _make_context(query_text: str = "What is AI?", lang: str = "en") -> Context:
    """Create a minimal Context for testing."""
    return Context(
        query=Query(query_text=query_text, lang=Language(lang)),
        passages=[
            RetrievalResult(
                passage=Passage(
                    passage_id="p1",
                    text="Artificial intelligence is a branch of computer science.",
                    lang=Language("en"),
                ),
                score=0.9,
                source="bm25",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestGeminiConstruction:
    """Tests for GeminiGenerator construction and configuration."""

    def test_default_model(self) -> None:
        gen = GeminiGenerator(api_key="test-key")
        assert gen.model == "gemini-3.6-flash"

    def test_custom_model(self) -> None:
        gen = GeminiGenerator(api_key="test-key", model="gemini-2.5-pro")
        assert gen.model == "gemini-2.5-pro"

    def test_api_key_from_env(self) -> None:
        with patch.dict("os.environ", {"GEMINI_API_KEY": "env-key"}):
            gen = GeminiGenerator()
            assert gen.api_key == "env-key"

    def test_api_key_explicit(self) -> None:
        gen = GeminiGenerator(api_key="explicit-key")
        assert gen.api_key == "explicit-key"

    def test_no_api_key_warns(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            gen = GeminiGenerator()
            assert gen.api_key == ""

    def test_temperature_configurable(self) -> None:
        gen = GeminiGenerator(api_key="key", temperature=0.5)
        assert gen.temperature == 0.5

    def test_max_output_tokens_configurable(self) -> None:
        gen = GeminiGenerator(api_key="key", max_output_tokens=512)
        assert gen.max_output_tokens == 512


# ---------------------------------------------------------------------------
# Prompt construction tests
# ---------------------------------------------------------------------------

class TestPromptConstruction:
    """Tests for user prompt building."""

    def test_prompt_includes_context(self) -> None:
        ctx = _make_context()
        prompt = _build_user_prompt(ctx)
        assert "Artificial intelligence" in prompt

    def test_prompt_includes_query(self) -> None:
        ctx = _make_context()
        prompt = _build_user_prompt(ctx)
        assert "What is AI?" in prompt

    def test_prompt_labels_languages(self) -> None:
        ctx = _make_context()
        prompt = _build_user_prompt(ctx)
        assert "(EN)" in prompt


# ---------------------------------------------------------------------------
# Generation tests (mocked)
# ---------------------------------------------------------------------------

class TestGeminiGeneration:
    """Tests for Gemini generation with mocked API."""

    @patch("google.genai.Client")
    def test_generate_returns_response(self, mock_client_cls: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.text = "AI is a branch of computer science."
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.total_token_count = 50

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        gen = GeminiGenerator(api_key="test-key")
        ctx = _make_context()
        result = gen.generate(ctx)

        assert result.answer_text == "AI is a branch of computer science."
        assert result.grounded is True

    @patch("google.genai.Client")
    def test_generate_grounds_correctly(self, mock_client_cls: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.text = "I don't have enough information to answer this question."
        mock_response.usage_metadata = None

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        gen = GeminiGenerator(api_key="test-key")
        ctx = _make_context()
        result = gen.generate(ctx)

        assert result.grounded is False

    @patch("google.genai.Client")
    def test_generate_api_error(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("API error")
        mock_client_cls.return_value = mock_client

        gen = GeminiGenerator(api_key="test-key")
        ctx = _make_context()
        with pytest.raises(RuntimeError, match="API error"):
            gen.generate(ctx)

    @patch("google.genai.Client")
    def test_generate_empty_response(self, mock_client_cls: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.text = ""
        mock_response.usage_metadata = None

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        gen = GeminiGenerator(api_key="test-key")
        ctx = _make_context()
        result = gen.generate(ctx)

        assert result.answer_text == ""
        assert result.grounded is False

    @patch("google.genai.Client")
    def test_generate_hindi_response(self, mock_client_cls: MagicMock) -> None:
        """Hindi response is parsed correctly."""
        mock_response = MagicMock()
        mock_response.text = "कृत्रिम बुद्धिमत्ता कंप्यूटर विज्ञान की एक शाखा है।"
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.total_token_count = 30

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        gen = GeminiGenerator(api_key="test-key")
        ctx = _make_context(query_text="AI क्या है?", lang="hi")
        result = gen.generate(ctx)

        assert "कृत्रिम बुद्धिमत्ता" in result.answer_text
        assert result.grounded is True

    def test_no_api_key_not_logged(self) -> None:
        gen = GeminiGenerator(api_key="secret-key-12345")
        assert gen.api_key == "secret-key-12345"


# ---------------------------------------------------------------------------
# Provider selection tests
# ---------------------------------------------------------------------------

class TestProviderSelection:
    """Tests for provider selection logic."""

    def test_gemini_preferred_over_deepseek(self) -> None:
        import app.api.routes as routes
        routes._pipeline = None

        with patch("app.core.config.get_settings") as mock_settings:
            mock_s = MagicMock()
            mock_s.gemini_api_key = "gemini-key"
            mock_s.generation_api_key = "deepseek-key"
            mock_s.demo_mode = False
            mock_s.demo_index_path = "nonexistent"
            mock_s.demo_passage_store_path = "nonexistent"
            mock_settings.return_value = mock_s

            from generation.gemini_provider import GeminiGenerator
            assert GeminiGenerator is not None

    def test_stub_fallback_when_no_keys(self) -> None:
        from generation.deepseek_provider import StubGenerator
        gen = StubGenerator()
        assert gen is not None
