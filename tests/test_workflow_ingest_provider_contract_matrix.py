from __future__ import annotations

import sys
from types import ModuleType
from typing import ClassVar

import pytest
from kg_doc_parser.workflow_ingest import (
    ProviderEndpointConfig,
    WorkflowProviderSettings,
    build_chat_model_for_role,
    providers,
)

pytestmark = [pytest.mark.workflow, pytest.mark.ci]


class _OfflineChatModel:
    constructed: ClassVar[list[dict[str, object]]] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.constructed.append(kwargs)

    def with_structured_output(self, schema: object, include_raw: bool = True, **kwargs: object):
        del schema, include_raw, kwargs
        return self


def _install_offline_sdk_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "langchain_ollama",
        ModuleType("langchain_ollama"),
    )
    sys.modules["langchain_ollama"].ChatOllama = _OfflineChatModel

    openai = ModuleType("langchain_openai")
    openai.ChatOpenAI = _OfflineChatModel
    openai.AzureChatOpenAI = _OfflineChatModel
    monkeypatch.setitem(sys.modules, "langchain_openai", openai)

    google_genai = ModuleType("langchain_google_genai")
    google_genai.ChatGoogleGenerativeAI = _OfflineChatModel
    monkeypatch.setitem(sys.modules, "langchain_google_genai", google_genai)

    vertex = ModuleType("langchain_google_vertexai")
    vertex.ChatVertexAI = _OfflineChatModel
    monkeypatch.setitem(sys.modules, "langchain_google_vertexai", vertex)


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("fake", "fake-parser"),
        ("ollama", "qwen2.5:7b"),
        ("openai", "gpt-4.1-mini"),
        ("azure", "gpt-4.1-mini"),
        ("gemini", "gemini-2.5-flash"),
        ("vertex", "gemini-2.5-pro"),
    ],
)
def test_declared_chat_provider_constructors_have_one_common_contract(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    model: str,
) -> None:
    _install_offline_sdk_modules(monkeypatch)
    _OfflineChatModel.constructed.clear()
    if provider in {"openai", "azure"}:
        monkeypatch.setenv("TEST_OPENAI_TOKEN", "offline-token")
    if provider == "gemini":
        monkeypatch.setenv("TEST_GOOGLE_TOKEN", "offline-token")

    endpoint = ProviderEndpointConfig(
        provider=provider,
        model=model,
        base_url="https://example.invalid" if provider == "azure" else None,
        api_version="2024-12-01-preview" if provider == "azure" else None,
        api_key_env=(
            "TEST_OPENAI_TOKEN"
            if provider in {"openai", "azure"}
            else "TEST_GOOGLE_TOKEN"
            if provider == "gemini"
            else None
        ),
    )

    chat = build_chat_model_for_role(
        "parser",
        WorkflowProviderSettings(parser=endpoint),
    )

    assert chat is not None
    assert hasattr(chat, "with_structured_output")
    if provider != "fake":
        assert _OfflineChatModel.constructed


def test_codex_constructor_uses_the_same_structured_provider_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _OfflineBridge:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def with_structured_output(self, schema: object, include_raw: bool = True, **kwargs: object):
            del schema, include_raw, kwargs
            return self

    monkeypatch.setattr(providers, "StructuredBridgeChatModel", _OfflineBridge)
    monkeypatch.setenv("TEST_CODEX_TOKEN", "offline-token")

    chat = build_chat_model_for_role(
        "parser",
        WorkflowProviderSettings(
            parser=ProviderEndpointConfig(
                provider="codex",
                model="gpt-5.6-luna",
                base_url="http://bridge.invalid",
                api_key_env="TEST_CODEX_TOKEN",
            )
        ),
    )

    assert chat is not None
    assert hasattr(chat, "with_structured_output")
    assert captured["endpoint"] == "http://bridge.invalid"
    assert captured["model"] == "gpt-5.6-luna"
