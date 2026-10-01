from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest
from kg_doc_parser.workflow_ingest import ProviderEndpointConfig
from kg_doc_parser.workflow_ingest.providers import build_chat_model

pytestmark = pytest.mark.ci


def _fake_provider_module(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    class_name: str,
    captured: list[dict[str, Any]],
) -> None:
    module = ModuleType(module_name)

    class FakeChatModel:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)

    setattr(module, class_name, FakeChatModel)
    monkeypatch.setitem(sys.modules, module_name, module)


def test_openai_adapter_enforces_configured_retry_and_output_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []
    callbacks = [object()]
    _fake_provider_module(monkeypatch, "langchain_openai", "ChatOpenAI", captured)

    model = build_chat_model(
        ProviderEndpointConfig(
            provider="openai",
            model="Ternary-Bonsai-2-27B-PTQ1_0",
            base_url="http://127.0.0.1:8181/v1",
            max_retries=0,
            max_output_tokens=4096,
        ),
        callbacks=callbacks,
    )

    assert model is not None
    assert captured == [
        {
            "model": "Ternary-Bonsai-2-27B-PTQ1_0",
            "temperature": 0.1,
            "callbacks": callbacks,
            "max_retries": 0,
            "max_tokens": 4096,
            "base_url": "http://127.0.0.1:8181/v1",
        }
    ]


@pytest.mark.parametrize(
    ("provider", "module_name", "class_name", "expected_output_key"),
    [
        ("anthropic", "langchain_anthropic", "ChatAnthropic", "max_tokens"),
        ("azure", "langchain_openai", "AzureChatOpenAI", "max_tokens"),
        ("gemini", "langchain_google_genai", "ChatGoogleGenerativeAI", "max_output_tokens"),
        ("vertex", "langchain_google_vertexai", "ChatVertexAI", "max_output_tokens"),
        ("ollama", "langchain_ollama", "ChatOllama", "num_predict"),
    ],
)
def test_provider_adapters_map_output_cap_to_native_option(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    module_name: str,
    class_name: str,
    expected_output_key: str,
) -> None:
    captured: list[dict[str, Any]] = []
    _fake_provider_module(monkeypatch, module_name, class_name, captured)

    build_chat_model(
        ProviderEndpointConfig(
            provider=provider,
            model="test-model",
            base_url="http://localhost:1234" if provider in {"azure", "ollama"} else None,
            max_retries=0,
            max_output_tokens=128,
        )
    )

    assert captured[0][expected_output_key] == 128
    if provider != "ollama":
        assert captured[0]["max_retries"] == 0


@pytest.mark.parametrize("value", [0, -1])
def test_provider_output_token_cap_must_be_positive(value: int) -> None:
    with pytest.raises(ValueError, match="max_output_tokens"):
        ProviderEndpointConfig(provider="openai", max_output_tokens=value)


@pytest.mark.parametrize("value", [-1, -10])
def test_provider_retry_count_cannot_be_negative(value: int) -> None:
    with pytest.raises(ValueError, match="max_retries"):
        ProviderEndpointConfig(provider="openai", max_retries=value)


def test_claude_provider_name_normalizes_to_optional_anthropic_adapter() -> None:
    assert ProviderEndpointConfig(provider="anthropic", model="claude-sonnet").provider == "anthropic"
