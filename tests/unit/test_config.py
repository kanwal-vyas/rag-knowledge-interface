from __future__ import annotations

from app.core.config import Settings, get_settings


def test_settings_load_with_defaults() -> None:
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.environment in {"development", "staging", "production"}


def test_settings_confirmed_embedding_model_default() -> None:
    """bge-m3 is confirmed (Audit §4/§7) — default must not silently drift."""
    settings = get_settings()
    assert settings.embedding_model_name == "BAAI/bge-m3"


def test_settings_stt_provider_configurable() -> None:
    """STT provider is configurable via env var, not hardcoded in code."""
    # The code default in Settings class is None — provider must be set via .env
    fields = Settings.model_fields
    assert fields["stt_provider"].default is None
    # Runtime value comes from .env (Sarvam is the configured provider)
    settings = get_settings()
    # stt_provider may be None (no .env) or 'sarvam' (configured)
    assert settings.stt_provider in (None, "sarvam")
