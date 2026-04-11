from __future__ import annotations

import importlib.util

import pytest

from src.workflow_ingest import (
    EmbeddingProviderConfig,
    ProviderEndpointConfig,
    WorkflowProviderSettings,
    build_chat_model_for_role,
    build_embedding_function,
)


pytestmark = [pytest.mark.workflow, pytest.mark.ci]


def _require_module(module_name: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        pytest.skip(f"{module_name} is not installed")


CHAT_PROVIDER_CASES = [
    pytest.param(
        "ocr-gemini",
        "ocr",
        "gemini",
        "langchain_google_genai",
        "gemini-2.5-flash",
        id="ocr-gemini",
    ),
    pytest.param(
        "ocr-ollama",
        "ocr",
        "ollama",
        "langchain_ollama",
        "llava:latest",
        id="ocr-ollama",
    ),
    pytest.param(
        "parser-openai",
        "parser",
        "openai",
        "langchain_openai",
        "gpt-4.1-mini",
        id="parser-openai",
    ),
    pytest.param(
        "parser-vertex",
        "parser",
        "vertex",
        "langchain_google_vertexai",
        "gemini-2.5-pro",
        id="parser-vertex",
    ),
]


@pytest.mark.parametrize("case_name, role, provider, module_name, model", CHAT_PROVIDER_CASES)
def test_chat_model_provider_matrix_skips_if_backend_missing(
    case_name: str,
    role: str,
    provider: str,
    module_name: str,
    model: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_module(module_name)

    if provider == "gemini":
        monkeypatch.setenv("GOOGLE_API_KEY", "dummy-key")

    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="gemini", model="gemini-2.5-flash"),
        parser=ProviderEndpointConfig(
            provider=provider,
            model=model,
            api_key_env="GOOGLE_API_KEY" if provider == "gemini" else None,
        ),
    )
    chat = build_chat_model_for_role(role, settings)

    assert chat is not None, case_name
    assert hasattr(chat, "with_structured_output"), case_name


EMBEDDING_PROVIDER_CASES = [
    pytest.param(
        "embed-openai",
        "openai",
        "langchain_openai",
        "text-embedding-3-small",
        id="embed-openai",
    ),
    pytest.param(
        "embed-ollama",
        "ollama",
        "langchain_ollama",
        "nomic-embed-text",
        id="embed-ollama",
    ),
    pytest.param(
        "embed-vertex",
        "vertex",
        "langchain_google_vertexai",
        "text-embedding-004",
        id="embed-vertex",
    ),
]


@pytest.mark.parametrize("case_name, provider, module_name, model", EMBEDDING_PROVIDER_CASES)
def test_embedding_provider_matrix_skips_if_backend_missing(
    case_name: str,
    provider: str,
    module_name: str,
    model: str,
) -> None:
    _require_module(module_name)

    emb = build_embedding_function(EmbeddingProviderConfig(provider=provider, model=model, dimension=3))
    vectors = emb(["alpha", "beta"])

    assert emb is not None, case_name
    assert len(vectors) == 2, case_name
    assert len(vectors[0]) > 0, case_name
    assert len(vectors[1]) == len(vectors[0]), case_name
