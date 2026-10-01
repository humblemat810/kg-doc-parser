from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from kg_doc_parser.workflow_ingest.providers import (
    ProviderEndpointConfig,
    build_chat_model,
)

pytestmark = [pytest.mark.workflow, pytest.mark.ci]


@pytest.mark.parametrize(
    ("provider", "model", "api_key_env", "expected_key"),
    [
        ("openai", "gpt-4.1-mini", None, "model"),
        ("azure", "gpt-5-mini", "AZURE_OPENAI_API_KEY", "azure_deployment"),
    ],
)
def test_openai_compatible_chat_models_receive_optional_output_cap(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    model: str,
    api_key_env: str | None,
    expected_key: str,
) -> None:
    captured: dict[str, object] = {}

    class _FakeChatOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    class _FakeAzureChatOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        SimpleNamespace(ChatOpenAI=_FakeChatOpenAI, AzureChatOpenAI=_FakeAzureChatOpenAI),
    )
    monkeypatch.setenv("KG_DOC_PARSER_MAX_OUTPUT_TOKENS", "2048")
    monkeypatch.delenv("KOGWISTAR_MAINTENANCE_MAX_OUTPUT_TOKENS", raising=False)
    if api_key_env:
        monkeypatch.setenv(api_key_env, "test-token")

    build_chat_model(
        ProviderEndpointConfig(
            provider=provider,
            model=model,
            api_key_env=api_key_env,
        )
    )

    assert captured[expected_key] == model
    assert captured["max_tokens"] == 2048


def test_maintenance_output_cap_takes_precedence_over_parser_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeChatOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        SimpleNamespace(ChatOpenAI=_FakeChatOpenAI),
    )
    monkeypatch.setenv("KG_DOC_PARSER_MAX_OUTPUT_TOKENS", "2048")
    monkeypatch.setenv("KOGWISTAR_MAINTENANCE_MAX_OUTPUT_TOKENS", "1024")

    build_chat_model(ProviderEndpointConfig(provider="openai", model="gpt-4.1-mini"))

    assert captured["max_tokens"] == 1024


@pytest.mark.parametrize("value", ["0", "-5", "not-an-integer"])
def test_invalid_output_token_cap_fails_before_provider_construction(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("KG_DOC_PARSER_MAX_OUTPUT_TOKENS", value)
    monkeypatch.delenv("KOGWISTAR_MAINTENANCE_MAX_OUTPUT_TOKENS", raising=False)

    with pytest.raises(ValueError):
        build_chat_model(ProviderEndpointConfig(provider="openai", model="gpt-4.1-mini"))
