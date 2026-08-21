"""
Typed configuration.

Secrets and deployment-specific values come from environment variables
(see .env.example). Do not hardcode secrets or invent provider defaults —
fields with no confirmed architectural default are left as `None` /
empty string rather than given a plausible-looking default value, so a
missing configuration fails loudly instead of silently running against a
guessed default.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "development"
    log_level: str = "INFO"

    # Qdrant — confirmed as the production vector store.
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection_name: str | None = None  # ARCHITECTURE DETAIL MISSING

    # Embedding — bge-m3 confirmed.
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_endpoint_url: str | None = None

    # STT — ARCHITECTURE DETAIL MISSING, no default provider assumed.
    stt_provider: str | None = None
    stt_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("STT_API_KEY", "SARVAM_API_KEY"),
    )

    # Generation — Gemini (primary) or DeepSeek (fallback).
    generation_provider: str | None = None  # "gemini" or "deepseek"
    generation_api_key: str | None = None
    generation_model_name: str | None = None

    # Gemini-specific
    gemini_api_key: str | None = None
    gemini_model_name: str = "gemini-3.6-flash"

    dev_review_llm_api_key: str | None = None

    # Reranker — ARCHITECTURE DETAIL MISSING which of A/B/C.
    reranker_option: str | None = None

    # Demo mode — load prebuilt BM25 index from artifacts/demo/
    # CORS — comma-separated list of allowed origins.
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173"

    demo_mode: bool = False
    demo_index_path: str = "artifacts/demo/bm25_index.json"
    demo_passage_store_path: str = "artifacts/demo/passage_store.json"


def get_settings() -> Settings:
    """Factory rather than a module-level singleton, so tests can override
    environment variables cleanly without import-order caching bugs."""
    return Settings()
