from __future__ import annotations

import importlib
import importlib.util

import pytest

from kg_doc_parser.workflow_ingest import (
    EmbeddingProviderConfig,
    ProviderEndpointConfig,
    WorkflowProviderSettings,
    build_chat_model_for_role,
    build_embedding_function,
)


# Provider constructors are kg-doc-parser model-selection coverage.  Several
# adapters validate real SDK credentials during construction, so this file is
# intentionally outside Kogwistar ADR-015 compatibility CI until fully faked.
pytestmark = [pytest.mark.workflow, pytest.mark.llm_real]


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
        "parser-azure",
        "parser",
        "azure",
        "langchain_openai",
        "gpt-4o",
        id="parser-azure",
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

    if provider == "gemini" or role == "ocr":
        monkeypatch.setenv("GOOGLE_API_KEY", "dummy-key")
    if provider == "azure":
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-key")

    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="gemini", model="gemini-2.5-flash"),
        parser=ProviderEndpointConfig(
            provider=provider,
            model=model,
            base_url="https://example.openai.azure.com/" if provider == "azure" else None,
            api_version="2024-12-01-preview" if provider == "azure" else None,
            api_key_env="AZURE_OPENAI_API_KEY" if provider == "azure" else ("GOOGLE_API_KEY" if provider == "gemini" else None),
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_module(module_name)

    module = importlib.import_module(module_name)

    class _FakeEmbeddings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def embed_documents(self, texts):
            return [[float(len(text)), float(index + 1), 0.0] for index, text in enumerate(texts)]

    if provider == "openai":
        monkeypatch.setattr(module, "OpenAIEmbeddings", _FakeEmbeddings)
        monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
    elif provider == "ollama":
        monkeypatch.setattr(module, "OllamaEmbeddings", _FakeEmbeddings)
    elif provider == "vertex":
        monkeypatch.setattr(module, "VertexAIEmbeddings", _FakeEmbeddings)

    emb = build_embedding_function(EmbeddingProviderConfig(provider=provider, model=model, dimension=3))
    vectors = emb(["alpha", "beta"])

    assert emb is not None, case_name
    assert len(vectors) == 2, case_name
    assert len(vectors[0]) > 0, case_name
    assert len(vectors[1]) == len(vectors[0]), case_name


def test_workflow_provider_settings_from_env_normalizes_azure_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KG_DOC_PARSER_PROVIDER", "azure_openai")
    monkeypatch.setenv("KG_DOC_PARSER_MODEL", "gpt-5-mini")
    monkeypatch.setenv("KG_DOC_PARSER_BASE_URL", "https://example.openai.azure.com/")
    monkeypatch.setenv("KG_DOC_PARSER_API_KEY_ENV", "OPENAI_API_KEY_GPT5_MINI")

    settings = WorkflowProviderSettings.from_env()

    assert settings.parser.provider == "azure"
    assert settings.parser.model == "gpt-5-mini"
    assert settings.parser.base_url == "https://example.openai.azure.com/"
    assert settings.parser.api_key_env == "OPENAI_API_KEY_GPT5_MINI"


def test_azure_chat_model_uses_gpt5_temperature_one(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("langchain_openai")
    captured: dict[str, object] = {}

    class _FakeAzureChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def with_structured_output(self, schema, include_raw: bool = True):
            raise AssertionError("not needed for this test")

    monkeypatch.setattr(module, "AzureChatOpenAI", _FakeAzureChatOpenAI)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-key")
    monkeypatch.setenv("OPENAI_API_VERSION", "2024-12-01-preview")

    chat = build_chat_model_for_role(
        "parser",
        WorkflowProviderSettings(
            parser=ProviderEndpointConfig(
                provider="azure",
                model="gpt-5-nano",
                base_url="https://example.openai.azure.com/",
                api_key_env="AZURE_OPENAI_API_KEY",
            )
        ),
    )

    assert chat is not None
    assert captured["temperature"] == 1.0
    assert captured["azure_deployment"] == "gpt-5-nano"
    assert captured["api_version"] == "2024-12-01-preview"
