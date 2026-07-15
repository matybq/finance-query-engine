"""Application settings loaded from environment variables and .env files."""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


COLLECTION_NAME = "airbnb_10k_fy2025"


class Settings(BaseSettings):
    openrouter_api_key: SecretStr
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_provider: Literal["openrouter", "openai"] = "openrouter"
    llm_model: str = "openai/gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    openai_api_key: SecretStr
    data_raw_dir: Path = Path("data/raw")
    data_processed_dir: Path = Path("data/processed")
    chroma_persist_dir: Path | None = None
    langsmith_tracing: bool | None = None
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str | None = None
    langsmith_endpoint: str | None = None
    langsmith_workspace_id: str | None = None
    langchain_callbacks_background: bool | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def set_chroma_persist_dir(self) -> Self:
        if self.chroma_persist_dir is None:
            self.chroma_persist_dir = self.data_processed_dir / "chroma"
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
