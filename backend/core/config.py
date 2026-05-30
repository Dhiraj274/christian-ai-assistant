"""
Core application configuration using Pydantic Settings.
All config values are loaded from environment variables / .env file.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with strict validation.
    All secrets must be provided via environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    app_env: Literal["development", "staging", "production"] = "development"
    app_secret_key: str = Field(default="dev-secret-change-in-production")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --- LLM Providers ---
    # Anthropic no longer required — OpenAI handles all LLM tasks
    openai_api_key: str = Field(default="")

    # --- Vector DB ---
    pinecone_api_key: str = Field(default="")
    pinecone_index_name: str = Field(default="christian-ai-bible")
    pinecone_environment: str = Field(default="us-east-1-aws")

    # --- Observability ---
    langfuse_secret_key: str = Field(default="")
    langfuse_public_key: str = Field(default="")
    langfuse_host: str = Field(default="https://cloud.langfuse.com")

    # --- Database ---
    sqlite_db_path: str = Field(default="./backend/data/bible.db")

    # --- Rate Limiting ---
    rate_limit_per_minute: int = Field(default=30)

    # --- CORS ---
    allowed_origins: str = Field(default="http://localhost:3000")

    # --- LLM Model Selections ---
    # gpt-4o-mini: best quality/cost ratio — $0.15/1M tokens, 90% of gpt-4o quality
    # Used for ALL tasks: generation, routing, safety, verification, embeddings
    primary_llm_model: str = Field(default="gpt-4o-mini")
    fast_llm_model: str = Field(default="gpt-4o-mini")
    embedding_model: str = Field(default="text-embedding-3-small")

    # --- Agent Config ---
    max_retriever_results: int = Field(default=8)
    max_verifier_retries: int = Field(default=2)
    conversation_window_size: int = Field(default=10)

    @computed_field  # type: ignore[misc]
    @property
    def allowed_origins_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    @computed_field  # type: ignore[misc]
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @computed_field  # type: ignore[misc]
    @property
    def observability_enabled(self) -> bool:
        """Langfuse is only enabled when keys are provided."""
        return bool(self.langfuse_secret_key and self.langfuse_public_key)

    @computed_field  # type: ignore[misc]
    @property
    def openai_configured(self) -> bool:
        """True if OpenAI API key is set (required for all LLM tasks)."""
        return bool(self.openai_api_key)


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings singleton.
    Use FastAPI Depends(get_settings) for dependency injection.
    """
    return Settings()
