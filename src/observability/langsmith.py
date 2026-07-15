"""LangSmith tracing configuration helpers."""

from __future__ import annotations

import os
from collections.abc import MutableMapping

from pydantic import SecretStr

from src.config import Settings

LANGSMITH_ENV_FIELDS = {
    "LANGSMITH_TRACING": "langsmith_tracing",
    "LANGSMITH_API_KEY": "langsmith_api_key",
    "LANGSMITH_PROJECT": "langsmith_project",
    "LANGSMITH_ENDPOINT": "langsmith_endpoint",
    "LANGSMITH_WORKSPACE_ID": "langsmith_workspace_id",
    "LANGCHAIN_CALLBACKS_BACKGROUND": "langchain_callbacks_background",
}


def _format_env_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, SecretStr):
        value = value.get_secret_value()
    if isinstance(value, bool):
        return "true" if value else "false"

    env_value = str(value)
    return env_value or None


def configure_langsmith(settings: Settings, environ: MutableMapping[str, str] = os.environ) -> None:
    """Apply optional LangSmith settings to the process environment.

    LangChain/LangGraph read LangSmith configuration from environment variables
    at runtime. Pydantic loads `.env` values for this app, so this bridges those
    settings into `os.environ` without making tracing required.
    """
    for env_name, field_name in LANGSMITH_ENV_FIELDS.items():
        env_value = _format_env_value(getattr(settings, field_name))
        if env_value is not None:
            environ[env_name] = env_value
