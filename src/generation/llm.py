"""LLM client construction helpers."""

from langchain_openai import ChatOpenAI

from src.config import Settings


def build_llm(settings: Settings) -> ChatOpenAI:
    """Create the chat model configured by application settings."""
    if settings.llm_provider == "openai":
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
        )

    if settings.llm_provider == "openrouter":
        return ChatOpenAI(
            model=settings.llm_model,
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
        )

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
