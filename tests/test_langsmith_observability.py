from pydantic import SecretStr

from src.config import Settings
from src.observability.langsmith import configure_langsmith


def make_settings(**overrides) -> Settings:
    return Settings(
        openrouter_api_key=SecretStr("test-openrouter-key"),
        openai_api_key=SecretStr("test-openai-key"),
        **overrides,
    )


def test_configure_langsmith_sets_configured_environment_values() -> None:
    settings = make_settings(
        langsmith_tracing=True,
        langsmith_api_key="test-langsmith-key",
        langsmith_project="finance-query-engine-dev",
        langsmith_endpoint="https://api.smith.langchain.com",
        langsmith_workspace_id="workspace-id",
        langchain_callbacks_background=False,
    )
    environ: dict[str, str] = {}

    configure_langsmith(settings, environ=environ)

    assert environ == {
        "LANGSMITH_TRACING": "true",
        "LANGSMITH_API_KEY": "test-langsmith-key",
        "LANGSMITH_PROJECT": "finance-query-engine-dev",
        "LANGSMITH_ENDPOINT": "https://api.smith.langchain.com",
        "LANGSMITH_WORKSPACE_ID": "workspace-id",
        "LANGCHAIN_CALLBACKS_BACKGROUND": "false",
    }


def test_configure_langsmith_leaves_existing_environment_when_unconfigured() -> None:
    settings = make_settings()
    environ = {"LANGSMITH_TRACING": "true"}

    configure_langsmith(settings, environ=environ)

    assert environ == {"LANGSMITH_TRACING": "true"}
