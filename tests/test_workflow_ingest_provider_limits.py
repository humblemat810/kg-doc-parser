from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from types import ModuleType
from typing import Any
from uuid import UUID

import pytest
from kg_doc_parser.workflow_ingest import ProviderEndpointConfig
from kg_doc_parser.workflow_ingest.models import FailureCategory
from kg_doc_parser.workflow_ingest.providers import (
    WorkflowProviderSettings,
    _normalize_provider_name,
    build_chat_model,
    invoke_with_timeout,
    provider_call_metrics_snapshot,
)
from kg_doc_parser.workflow_ingest.serialization import json_safe, safe_json_dumps

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
    assert _normalize_provider_name("Claude") == "anthropic"
    assert ProviderEndpointConfig(provider="anthropic", model="claude-sonnet").provider == "anthropic"


def test_provider_timeout_returns_result_without_leaking_into_caller() -> None:
    diagnostics: dict[str, object] = {}
    assert invoke_with_timeout(
        lambda: {"usage_metadata": {"input_tokens": 10, "output_tokens": 6, "ttft_ms": 12}},
        timeout_seconds=1.0,
        diagnostics=diagnostics,
        operation="test_call",
        attempt_index=2,
        call_role="proposal",
        strategy="layer_boundary",
    )["usage_metadata"]["output_tokens"] == 6
    assert diagnostics["operation"] == "test_call"
    assert diagnostics["call_role"] == "proposal"
    assert diagnostics["strategy"] == "layer_boundary"
    assert diagnostics["attempt_index"] == 2
    assert diagnostics["input_tokens"] == 10
    assert diagnostics["output_tokens"] == 6
    assert diagnostics["ttft_ms"] == 12
    assert diagnostics["elapsed_ms"] >= 0
    assert float(diagnostics["throughput_tokens_per_second"]) > 0
    assert diagnostics["success"] is True
    assert diagnostics["timed_out"] is False


def test_provider_timeout_bounds_a_stalled_local_model() -> None:
    release = threading.Event()

    def stalled_call() -> str:
        release.wait()
        return "late"

    diagnostics: dict[str, object] = {}
    with pytest.raises(TimeoutError, match="exceeded"):
        invoke_with_timeout(stalled_call, timeout_seconds=0.01, diagnostics=diagnostics)
    assert diagnostics["timed_out"] is True
    assert diagnostics["underlying_call_alive"] is True
    assert diagnostics["failure_type"] == "timeout"
    assert diagnostics["attempt_index"] == 1
    assert diagnostics["elapsed_ms"] >= 0
    assert diagnostics["input_tokens"] is None
    assert diagnostics["output_tokens"] is None
    release.set()


def test_provider_timeout_inflight_limit_prevents_retry_multiplication() -> None:
    release = threading.Event()
    started = threading.Event()

    def stalled_call() -> str:
        started.set()
        release.wait()
        return "late"

    first_error: list[BaseException] = []

    def run_first() -> None:
        try:
            invoke_with_timeout(stalled_call, timeout_seconds=1.0, max_in_flight=1)
        except Exception as exc:  # noqa: BLE001 - report any worker failure to the test
            first_error.append(exc)

    worker = threading.Thread(target=run_first)
    worker.start()
    assert started.wait(1.0)
    diagnostics: dict[str, object] = {}
    with pytest.raises(RuntimeError, match="in-flight call limit"):
        invoke_with_timeout(
            lambda: "duplicate",
            timeout_seconds=1.0,
            max_in_flight=1,
            diagnostics=diagnostics,
        )
    assert diagnostics["failure_type"] == "in_flight_limit"
    release.set()
    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert first_error == []


def test_provider_failure_diagnostics_use_stable_transport_category() -> None:
    diagnostics: dict[str, object] = {}

    def failed_call() -> str:
        raise ValueError("provider rejected request")

    with pytest.raises(ValueError, match="provider rejected"):
        invoke_with_timeout(failed_call, timeout_seconds=1.0, diagnostics=diagnostics)
    assert diagnostics["failure_type"] == "transport/provider_exception"
    assert diagnostics["error_type"] == "ValueError"


def test_failure_category_vocabulary_covers_provider_and_workflow_outcomes() -> None:
    from typing import get_args

    assert set(get_args(FailureCategory)) == {
        "timeout",
        "transport/provider_exception",
        "structured_output_parse_failure",
        "semantic_rejection",
        "anchor_ambiguity",
        "repair_failure",
        "retry",
        "fallback",
        "rollback",
        "in_flight_limit",
    }


def test_provider_metrics_expose_completion_and_orphaned_timeout_counts() -> None:
    before = provider_call_metrics_snapshot()
    release = threading.Event()
    finished = threading.Event()

    def stalled_call() -> str:
        release.wait()
        finished.set()
        return "late"

    with pytest.raises(TimeoutError, match="exceeded"):
        invoke_with_timeout(stalled_call, timeout_seconds=0.01)
    release.set()
    assert finished.wait(1.0)
    deadline = time.monotonic() + 1.0
    after = provider_call_metrics_snapshot()
    while int(after["active_calls"]) and time.monotonic() < deadline:
        time.sleep(0.01)
        after = provider_call_metrics_snapshot()

    assert int(after["calls_started"]) >= int(before["calls_started"]) + 1
    assert int(after["timeouts_observed"]) >= int(before["timeouts_observed"]) + 1
    assert int(after["orphaned_calls"]) >= int(before["orphaned_calls"]) + 1
    assert int(after["calls_completed"]) >= int(before["calls_completed"]) + 1
    assert int(after["active_calls"]) == 0


def test_provider_safe_json_serializes_nested_structured_values() -> None:
    @dataclass
    class _Metadata:
        identifier: UUID
        values: set[str]

    payload: dict[str, Any] = {
        "metadata": _Metadata(UUID("00000000-0000-0000-0000-000000000001"), {"a", "b"}),
    }
    payload["self"] = payload
    converted = json_safe(payload)
    assert converted["metadata"]["identifier"] == "00000000-0000-0000-0000-000000000001"
    assert sorted(converted["metadata"]["values"]) == ["a", "b"]
    assert converted["self"] == "<cycle>"
    assert '"metadata"' in safe_json_dumps(payload, sort_keys=True)


def test_workflow_settings_expose_bounded_frontier_and_boundary_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KG_DOC_PARSER_FRONTIER_BATCH_SIZE", "2")
    monkeypatch.setenv("KG_DOC_PARSER_BOUNDARY_MAX_POINTS", "96")
    monkeypatch.setenv("KG_DOC_PARSER_BOUNDARY_MAX_REPAIR_SHIFT_CHARS", "24")
    monkeypatch.setenv("KG_DOC_PARSER_PROPOSAL_TIMEOUT_SECONDS", "19")
    monkeypatch.setenv("KG_DOC_PARSER_REVIEW_TIMEOUT_SECONDS", "17")
    monkeypatch.setenv("KG_DOC_PARSER_TRIAGE_TIMEOUT_SECONDS", "13")
    monkeypatch.setenv("KG_DOC_PARSER_RETRY_BACKOFF_SECONDS", "0.5")
    settings = WorkflowProviderSettings.from_env()
    assert settings.layer_frontier_batch_size == 2
    assert settings.boundary_max_points == 96
    assert settings.boundary_max_repair_shift_chars == 24
    assert settings.proposal_timeout_seconds == 19
    assert settings.review_timeout_seconds == 17
    assert settings.triage_timeout_seconds == 13
    assert settings.parser.retry_backoff_seconds == 0.5


def test_openai_reasoning_effort_is_transmitted_as_request_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []
    _fake_provider_module(monkeypatch, "langchain_openai", "ChatOpenAI", captured)
    build_chat_model(
        ProviderEndpointConfig(
            provider="openai",
            model="local-reasoning-model",
            reasoning_effort="none",
        )
    )
    assert captured[0]["model_kwargs"] == {"reasoning_effort": "none"}
