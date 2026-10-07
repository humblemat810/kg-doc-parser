"""Provider-neutral OCR, parser, and embedding adapters.

This module keeps vendor-specific imports behind small factory functions so the
workflow code can stay neutral. The concrete vendor is selected by config, not
by the caller.

Quick examples
--------------
- OCR with Google GenAI:
  - KG_DOC_OCR_PROVIDER=gemini
  - KG_DOC_OCR_MODEL=gemini-2.5-flash

- OCR with a local Ollama vision model:
  - KG_DOC_OCR_PROVIDER=ollama
  - KG_DOC_OCR_MODEL=llava:latest
  - KG_DOC_OCR_BASE_URL=http://127.0.0.1:11434

- Parser/LLM with OpenAI Chat Completions:
  - KG_DOC_PARSER_PROVIDER=openai
  - KG_DOC_PARSER_MODEL=gpt-4.1-mini
  - KG_DOC_PARSER_API_KEY_ENV=OPENAI_API_KEY

- Parser/LLM with Google Vertex AI:
  - KG_DOC_PARSER_PROVIDER=vertex
  - KG_DOC_PARSER_MODEL=gemini-2.5-pro
  - KG_DOC_PARSER_PROJECT=my-project
  - KG_DOC_PARSER_LOCATION=us-central1

- Parser/LLM with Ollama:
  - KG_DOC_PARSER_PROVIDER=ollama
  - KG_DOC_PARSER_MODEL=llama3.1
  - KG_DOC_PARSER_BASE_URL=http://127.0.0.1:11434

- Embeddings with a fake deterministic function for CI:
  - KG_DOC_EMBED_PROVIDER=fake
  - KG_DOC_EMBED_MODEL=kg-doc-parser-workflow-embedding-v1

Cookbook example
----------------
If you are parsing a cooking recipe, you can keep OCR on Gemini but route the
parser to OpenAI or Ollama:

    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="gemini", model="gemini-2.5-flash"),
        parser=ProviderEndpointConfig(
            provider="openai",
            model="gpt-4.1-mini",
            api_key_env="OPENAI_API_KEY",
        ),
    )

That means the OCR step extracts the page text, and the parser step can then
turn the recipe into structured fields such as ingredients, tools, actions,
and inferred sections without changing workflow orchestration.
"""

from __future__ import annotations

import math
import os
import queue
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    ClassVar,
    Literal,
    Protocol,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    runtime_checkable,
)

from kogwistar.llm_tasks.providers import (
    ProviderChainChatModel,
    StructuredBridgeChatModel,
    SupportsStructuredOutput,
    bridge_messages,
)
from kogwistar.typing_interfaces import EmbeddingFunctionLike
from pydantic import BaseModel, Field, model_validator
from pydantic_core import PydanticUndefined
from pydantic_extension.model_slicing import BackendField, FrontendField
from pydantic_extension.model_slicing.mixin import (
    DtoField,
    ExcludeMode,
    LLMField,
    ModeSlicingMixin,
)

# Compatibility aliases keep older parser integrations source-compatible while
# the implementations now live in the shared Kogwistar layer.
CodexBridgeChatModel = StructuredBridgeChatModel
_codex_messages = bridge_messages

TStructuredModel = TypeVar("TStructuredModel", bound=BaseModel)
ChatProviderName = Literal["anthropic", "gemini", "ollama", "openai", "azure", "vertex", "fake", "codex"]
EmbeddingProviderName = Literal["fake", "openai", "vertex", "ollama"]
ProposalMode = Literal["children", "boundaries"]

_PROVIDER_IN_FLIGHT_LOCK = threading.Lock()
_PROVIDER_IN_FLIGHT = 0
_PROVIDER_METRICS: dict[str, object] = {
    "calls_started": 0,
    "calls_completed": 0,
    "calls_succeeded": 0,
    "calls_failed": 0,
    "timeouts_observed": 0,
    "orphaned_calls": 0,
    "in_flight_rejections": 0,
    "active_calls": 0,
    "failure_counts": {},
}


def _provider_failure_category(exc: BaseException) -> str:
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, RuntimeError) and "in-flight call limit" in str(exc):
        return "in_flight_limit"
    return "transport/provider_exception"


def _record_provider_completion(*, success: bool, failure_type: str | None = None) -> None:
    with _PROVIDER_IN_FLIGHT_LOCK:
        _PROVIDER_METRICS["calls_completed"] = int(_PROVIDER_METRICS["calls_completed"]) + 1
        if success:
            _PROVIDER_METRICS["calls_succeeded"] = int(_PROVIDER_METRICS["calls_succeeded"]) + 1
            return
        _PROVIDER_METRICS["calls_failed"] = int(_PROVIDER_METRICS["calls_failed"]) + 1
        failure_counts = _PROVIDER_METRICS["failure_counts"]
        if not isinstance(failure_counts, dict):
            failure_counts = {}
            _PROVIDER_METRICS["failure_counts"] = failure_counts
        failure_counts[failure_type or "transport/provider_exception"] = (
            int(failure_counts.get(failure_type or "transport/provider_exception", 0)) + 1
        )


def provider_call_metrics_snapshot() -> dict[str, object]:
    """Return a stable, process-local snapshot of provider call health.

    The counters describe the shared invocation boundary, not a vendor SDK.
    ``orphaned_calls`` counts calls whose caller timed out while the daemon
    worker was still alive; ``active_calls`` shows whether those workers have
    subsequently drained.
    """

    with _PROVIDER_IN_FLIGHT_LOCK:
        snapshot = dict(_PROVIDER_METRICS)
        failure_counts = _PROVIDER_METRICS.get("failure_counts", {})
        snapshot["failure_counts"] = dict(failure_counts) if isinstance(failure_counts, dict) else {}
        snapshot["active_calls"] = _PROVIDER_IN_FLIGHT
        return snapshot


_PROVIDER_USAGE_FIELDS: dict[str, tuple[str, ...]] = {
    "input_tokens": ("input_tokens", "prompt_tokens", "input_token_count", "prompt_token_count"),
    "output_tokens": ("output_tokens", "completion_tokens", "output_token_count", "completion_token_count"),
    "ttft_ms": ("ttft_ms", "time_to_first_token_ms", "first_token_latency_ms"),
}


def _provider_usage_metrics(value: object) -> dict[str, int | float]:
    """Extract optional provider usage metadata without serializing the response."""

    pending: list[tuple[object, int]] = [(value, 0)]
    seen: set[int] = set()
    found: dict[str, int | float] = {}
    while pending:
        candidate, depth = pending.pop()
        if candidate is None or id(candidate) in seen or depth > 4:
            continue
        seen.add(id(candidate))
        if isinstance(candidate, Mapping):
            items = candidate.items()
        else:
            model_dump = getattr(candidate, "model_dump", None)
            if callable(model_dump):
                try:
                    dumped = model_dump()
                except Exception:  # noqa: BLE001 - usage metadata is best effort.
                    dumped = None
                items = dumped.items() if isinstance(dumped, Mapping) else ()
            else:
                items = (
                    (name, getattr(candidate, name))
                    for names in _PROVIDER_USAGE_FIELDS.values()
                    for name in (*names, "usage", "usage_metadata", "response_metadata", "metadata", "raw")
                    if hasattr(candidate, name)
                )
        nested: list[object] = []
        for key, item in items:
            key_text = str(key)
            for metric_name, aliases in _PROVIDER_USAGE_FIELDS.items():
                if key_text in aliases and metric_name not in found and isinstance(item, (int, float)) and not isinstance(item, bool):
                    found[metric_name] = item
            if key_text in {"usage", "usage_metadata", "response_metadata", "metadata", "raw", "additional_kwargs"}:
                nested.append(item)
        pending.extend((item, depth + 1) for item in nested)
    return found


def _reserve_provider_slot(max_in_flight: int) -> bool:
    global _PROVIDER_IN_FLIGHT
    with _PROVIDER_IN_FLIGHT_LOCK:
        if _PROVIDER_IN_FLIGHT >= max_in_flight:
            _PROVIDER_METRICS["in_flight_rejections"] = int(_PROVIDER_METRICS["in_flight_rejections"]) + 1
            return False
        _PROVIDER_IN_FLIGHT += 1
        _PROVIDER_METRICS["calls_started"] = int(_PROVIDER_METRICS["calls_started"]) + 1
        _PROVIDER_METRICS["active_calls"] = _PROVIDER_IN_FLIGHT
        return True


def _release_provider_slot() -> None:
    global _PROVIDER_IN_FLIGHT
    with _PROVIDER_IN_FLIGHT_LOCK:
        _PROVIDER_IN_FLIGHT = max(0, _PROVIDER_IN_FLIGHT - 1)
        _PROVIDER_METRICS["active_calls"] = _PROVIDER_IN_FLIGHT


class StructuredPayloadFactory(Protocol):
    """Build a deterministic structured-output payload for one schema."""

    def __call__(self, schema: type[BaseModel], /) -> dict[str, object]: ...


class _FakeStructuredResponse:
    def __init__(self, schema: type[TStructuredModel], payload: dict[str, object]) -> None:
        self.schema = schema
        self.payload = payload

    def invoke(self, messages: object, config: object = None) -> dict[str, object]:
        parsed = self.schema.model_validate(self.payload)
        return {"parsed": parsed, "raw": None, "parsing_error": None}


class FakeChatModel:
    """Minimal structured-output compatible chat model for tests."""

    def __init__(
        self, *, payload_factory: StructuredPayloadFactory | None = None
    ) -> None:
        self.payload_factory = payload_factory or _default_schema_payload

    def with_structured_output(
        self,
        schema: type[TStructuredModel],
        include_raw: bool = True,
        **kwargs: object,
    ) -> _FakeStructuredResponse:
        _ = kwargs
        payload = self.payload_factory(schema)
        return _FakeStructuredResponse(schema, payload)


def _default_schema_payload(schema: type[BaseModel]) -> dict[str, object]:
    def _value_for_field(field: object) -> object:
        annotation = getattr(field, "annotation", None)
        origin = get_origin(annotation)
        args = get_args(annotation)
        if annotation is str:
            return ""
        if annotation is bool:
            return False
        if annotation is int:
            return 0
        if annotation is float:
            return 0.0
        if origin is list or annotation is list:
            return []
        if origin is dict or annotation is dict:
            return {}
        if origin is tuple:
            return []
        if origin is Literal and args:
            return args[0]
        if origin is Union and type(None) in args:
            return None
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return _default_schema_payload(annotation)
        default = getattr(field, "default", PydanticUndefined)
        if default is not PydanticUndefined and default is not None:
            return default
        return None

    payload: dict[str, object] = {}
    for name, field in getattr(schema, "model_fields", {}).items():
        value = _value_for_field(field)
        if value is not None:
            payload[name] = value
    return payload


@runtime_checkable
class ChatModelProvider(Protocol):
    def build(self, *, callbacks: list[object] | None = None) -> SupportsStructuredOutput: ...


@runtime_checkable
class EmbeddingFunctionProvider(Protocol):
    def build(self) -> EmbeddingFunctionLike: ...


class ProviderEndpointConfig(ModeSlicingMixin, BaseModel):
    default_include_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}
    include_unmarked_for_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}

    provider: Annotated[
        Literal[
            "anthropic",
            "gemini",
            "ollama",
            "openai",
            "azure",
            "vertex",
            "fake",
            "codex",
        ],
        DtoField(),
        BackendField(),
        FrontendField(),
        LLMField(),
    ] = "gemini"
    model: Annotated[str, DtoField(), BackendField(), FrontendField(), LLMField()] = "gemini-2.5-flash"
    temperature: Annotated[float, DtoField(), BackendField(), FrontendField(), LLMField()] = 0.1
    reasoning_effort: Annotated[
        str | None,
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = None
    base_url: Annotated[
        str | None,
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = None
    api_key_env: Annotated[
        str | None,
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = None
    api_version: Annotated[
        str | None,
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = None
    project: Annotated[
        str | None,
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = None
    location: Annotated[
        str | None,
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = None
    max_retries: Annotated[int, DtoField(), BackendField(), FrontendField(), LLMField()] = Field(
        default=2, ge=0
    )
    max_output_tokens: Annotated[
        int | None, DtoField(), BackendField(), FrontendField(), LLMField()
    ] = Field(default=None, gt=0)
    timeout_seconds: Annotated[
        float,
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = Field(default=120.0, gt=0)
    retry_backoff_seconds: Annotated[
        float,
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = Field(default=0.25, ge=0.0, le=30.0)
    retry_backoff_max_seconds: Annotated[
        float,
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = Field(default=5.0, ge=0.0, le=120.0)
    max_in_flight_calls: Annotated[
        int,
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = Field(default=1, ge=1, le=32)
    fallback_specs: list[ProviderEndpointConfig] = Field(default_factory=list, exclude=True)


class EmbeddingProviderConfig(ModeSlicingMixin, BaseModel):
    default_include_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}
    include_unmarked_for_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}

    provider: Annotated[
        Literal["fake", "openai", "vertex", "ollama"],
        DtoField(),
        BackendField(),
        FrontendField(),
        LLMField(),
    ] = "fake"
    model: Annotated[str, DtoField(), BackendField(), FrontendField(), LLMField()] = "kg-doc-parser-workflow-embedding-v1"
    dimension: Annotated[int, DtoField(), BackendField(), FrontendField(), LLMField()] = 2
    base_url: Annotated[
        str | None,
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = None
    api_key_env: Annotated[
        str | None,
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = None
    project: Annotated[
        str | None,
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = None
    location: Annotated[
        str | None,
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = None
    max_sequence_length: Annotated[
        int | None, DtoField(), BackendField(), FrontendField(), LLMField()
    ] = None
    crop_token_budget: Annotated[
        int | None, DtoField(), BackendField(), FrontendField(), LLMField()
    ] = None
    tokenizer_fingerprint: Annotated[
        str | None, DtoField(), BackendField(), FrontendField(), LLMField()
    ] = None

    @model_validator(mode="after")
    def validate_token_limits(self) -> EmbeddingProviderConfig:
        if self.max_sequence_length is not None and self.max_sequence_length <= 0:
            raise ValueError("embedding max_sequence_length must be positive")
        if self.crop_token_budget is not None and self.crop_token_budget <= 0:
            raise ValueError("embedding crop_token_budget must be positive")
        if (
            self.max_sequence_length is not None
            and self.crop_token_budget is not None
            and self.crop_token_budget > self.max_sequence_length
        ):
            raise ValueError("embedding crop_token_budget cannot exceed max_sequence_length")
        return self


def _normalize_provider_name(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized == "azure_openai":
        return "azure"
    if normalized == "claude":
        return "anthropic"
    return normalized


def _is_gpt5_model(model: str | None) -> bool:
    normalized = str(model or "").strip().lower()
    return normalized.startswith(("gpt-5", "gpt5"))


def _chat_temperature_for_model(model: str | None, requested_temperature: float) -> float:
    if _is_gpt5_model(model):
        return 1.0
    return requested_temperature


def _configured_max_output_tokens() -> int | None:
    """Return an optional cap for OpenAI-compatible structured responses.

    Local OpenAI-compatible servers otherwise choose a large default, which
    can let malformed structured-output responses consume a worker lease.
    Keep this opt-in so existing provider behavior is unchanged.
    """

    raw = os.getenv("KOGWISTAR_MAINTENANCE_MAX_OUTPUT_TOKENS") or os.getenv(
        "KG_DOC_PARSER_MAX_OUTPUT_TOKENS"
    )
    if not raw:
        return None
    value = int(raw)
    if value < 1:
        raise ValueError("max output tokens must be positive")
    return value


class WorkflowProviderSettings(ModeSlicingMixin, BaseModel):
    default_include_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}
    include_unmarked_for_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}

    proposal_mode: Annotated[
        Literal["children", "boundaries"],
        DtoField(),
        BackendField(),
        FrontendField(),
        LLMField(),
    ] = "children"
    parse_strategy: Annotated[
        Literal["auto", "layer_excerpt", "layer_boundary", "page_index"],
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = "auto"
    parse_strategy_order: Annotated[
        tuple[Literal["layer_excerpt", "layer_boundary", "page_index"], ...],
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = ("layer_excerpt", "layer_boundary", "page_index")
    triage_enabled: Annotated[
        bool,
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = True
    page_index_summary_enabled: Annotated[
        bool,
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = True
    page_index_hierarchical_summary_enabled: Annotated[
        bool,
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = False
    layer_frontier_batch_size: Annotated[
        int,
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = Field(default=1, ge=1)
    boundary_max_points: Annotated[
        int,
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = Field(default=128, ge=8, le=512)
    boundary_max_repair_shift_chars: Annotated[
        int,
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = Field(default=8, ge=0, le=4096)
    proposal_timeout_seconds: Annotated[
        float | None,
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = Field(default=None, gt=0)
    review_timeout_seconds: Annotated[
        float | None,
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = Field(default=None, gt=0)
    triage_timeout_seconds: Annotated[
        float | None,
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = Field(default=None, gt=0)
    ocr: Annotated[ProviderEndpointConfig, DtoField(), BackendField(), FrontendField(), LLMField()] = Field(
        default_factory=ProviderEndpointConfig
    )
    parser: Annotated[ProviderEndpointConfig, DtoField(), BackendField(), FrontendField(), LLMField()] = Field(
        default_factory=ProviderEndpointConfig
    )
    embedding: Annotated[EmbeddingProviderConfig, DtoField(), BackendField(), FrontendField(), LLMField()] = Field(
        default_factory=EmbeddingProviderConfig
    )

    @model_validator(mode="after")
    def _check_parse_strategy_order(self) -> WorkflowProviderSettings:
        expected = {"layer_excerpt", "layer_boundary", "page_index"}
        if len(self.parse_strategy_order) != 3 or set(self.parse_strategy_order) != expected:
            raise ValueError(
                "parse_strategy_order must contain layer_excerpt, layer_boundary, and page_index exactly once"
            )
        return self

    @classmethod
    def from_env(cls) -> WorkflowProviderSettings:
        def _env(name: str, default: str | None = None) -> str | None:
            value = os.getenv(name)
            return value if value not in {None, ""} else default

        return cls(
            proposal_mode=cast(ProposalMode, str(_env("KG_DOC_PARSER_PROPOSAL_MODE", "children"))),
            parse_strategy=cast(
                Literal["auto", "layer_excerpt", "layer_boundary", "page_index"],
                str(_env("KG_DOC_PARSER_PARSE_STRATEGY", "auto")),
            ),
            parse_strategy_order=tuple(
                value.strip()
                for value in str(
                    _env("KG_DOC_PARSER_PARSE_STRATEGY_ORDER", "layer_excerpt,layer_boundary,page_index")
                ).split(",")
                if value.strip()
            ),
            triage_enabled=str(_env("KG_DOC_PARSER_TRIAGE_ENABLED", "1")).lower()
            not in {"0", "false", "no", "off"},
            page_index_summary_enabled=str(_env("KG_DOC_PARSER_PAGE_INDEX_SUMMARY_ENABLED", "1")).lower()
            not in {"0", "false", "no", "off"},
            page_index_hierarchical_summary_enabled=str(
                _env("KG_DOC_PARSER_PAGE_INDEX_HIERARCHICAL_SUMMARY_ENABLED", "0")
            ).lower()
            not in {"0", "false", "no", "off"},
            layer_frontier_batch_size=int(_env("KG_DOC_PARSER_FRONTIER_BATCH_SIZE", "1") or "1"),
            boundary_max_points=int(_env("KG_DOC_PARSER_BOUNDARY_MAX_POINTS", "128") or "128"),
            boundary_max_repair_shift_chars=int(
                _env("KG_DOC_PARSER_BOUNDARY_MAX_REPAIR_SHIFT_CHARS", "8") or "8"
            ),
            proposal_timeout_seconds=(
                float(value) if (value := _env("KG_DOC_PARSER_PROPOSAL_TIMEOUT_SECONDS")) else None
            ),
            review_timeout_seconds=(
                float(value) if (value := _env("KG_DOC_PARSER_REVIEW_TIMEOUT_SECONDS")) else None
            ),
            triage_timeout_seconds=(
                float(value) if (value := _env("KG_DOC_PARSER_TRIAGE_TIMEOUT_SECONDS")) else None
            ),
            ocr=ProviderEndpointConfig(
                provider=cast(ChatProviderName, _normalize_provider_name(_env("KG_DOC_OCR_PROVIDER", "gemini"))),
                model=str(_env("KG_DOC_OCR_MODEL", "gemini-2.5-flash")),
                temperature=float(_env("KG_DOC_OCR_TEMPERATURE", "0.1") or "0.1"),
                reasoning_effort=_env("KG_DOC_OCR_REASONING_EFFORT"),
                base_url=_env("KG_DOC_OCR_BASE_URL"),
                api_key_env=_env("KG_DOC_OCR_API_KEY_ENV"),
                api_version=_env("KG_DOC_OCR_API_VERSION"),
                project=_env("KG_DOC_OCR_PROJECT"),
                location=_env("KG_DOC_OCR_LOCATION"),
                max_retries=int(_env("KG_DOC_OCR_MAX_RETRIES", "2") or "2"),
                timeout_seconds=float(_env("KG_DOC_OCR_TIMEOUT_SECONDS", "120") or "120"),
                retry_backoff_seconds=float(_env("KG_DOC_OCR_RETRY_BACKOFF_SECONDS", "0.25") or "0.25"),
                retry_backoff_max_seconds=float(_env("KG_DOC_OCR_RETRY_BACKOFF_MAX_SECONDS", "5") or "5"),
                max_in_flight_calls=int(_env("KG_DOC_OCR_MAX_IN_FLIGHT", "1") or "1"),
            ),
            parser=ProviderEndpointConfig(
                provider=cast(ChatProviderName, _normalize_provider_name(_env("KG_DOC_PARSER_PROVIDER", "gemini"))),
                model=str(_env("KG_DOC_PARSER_MODEL", "gemini-2.5-flash")),
                temperature=float(_env("KG_DOC_PARSER_TEMPERATURE", "0.1") or "0.1"),
                reasoning_effort=_env("KG_DOC_PARSER_REASONING_EFFORT"),
                base_url=_env("KG_DOC_PARSER_BASE_URL"),
                api_key_env=_env("KG_DOC_PARSER_API_KEY_ENV"),
                api_version=_env("KG_DOC_PARSER_API_VERSION"),
                project=_env("KG_DOC_PARSER_PROJECT"),
                location=_env("KG_DOC_PARSER_LOCATION"),
                max_retries=int(_env("KG_DOC_PARSER_MAX_RETRIES", "2") or "2"),
                timeout_seconds=float(_env("KG_DOC_PARSER_TIMEOUT_SECONDS", "120") or "120"),
                retry_backoff_seconds=float(_env("KG_DOC_PARSER_RETRY_BACKOFF_SECONDS", "0.25") or "0.25"),
                retry_backoff_max_seconds=float(_env("KG_DOC_PARSER_RETRY_BACKOFF_MAX_SECONDS", "5") or "5"),
                max_in_flight_calls=int(_env("KG_DOC_PARSER_MAX_IN_FLIGHT", "1") or "1"),
            ),
            embedding=EmbeddingProviderConfig(
                provider=cast(EmbeddingProviderName, str(_env("KG_DOC_EMBED_PROVIDER", "fake"))),
                model=str(_env("KG_DOC_EMBED_MODEL", "kg-doc-parser-workflow-embedding-v1")),
                dimension=int(_env("KG_DOC_EMBED_DIMENSION", "2") or "2"),
                base_url=_env("KG_DOC_EMBED_BASE_URL"),
                api_key_env=_env("KG_DOC_EMBED_API_KEY_ENV"),
                project=_env("KG_DOC_EMBED_PROJECT"),
                location=_env("KG_DOC_EMBED_LOCATION"),
                max_sequence_length=(
                    int(value) if (value := _env("KG_DOC_EMBED_MAX_SEQUENCE_LENGTH")) else None
                ),
                crop_token_budget=(
                    int(value) if (value := _env("KG_DOC_EMBED_CROP_TOKEN_BUDGET")) else None
                ),
                tokenizer_fingerprint=_env("KG_DOC_EMBED_TOKENIZER_FINGERPRINT"),
            ),
        )


def invoke_with_timeout(
    callable_obj: Callable[[], object],
    *,
    timeout_seconds: float,
    diagnostics: dict[str, object] | None = None,
    operation: str = "provider_call",
    max_in_flight: int = 1,
    attempt_index: int = 1,
    call_role: str | None = None,
    strategy: str | None = None,
) -> object:
    """Run one provider call with a hard wall-clock bound.

    Provider SDKs do not expose one consistent timeout argument. A daemon
    thread keeps a stalled local HTTP client from blocking the parser workflow
    forever; the workflow records the timeout and can take its deterministic
    fallback path. The provider call itself must remain side-effect free until
    its structured result has been accepted by the host.
    """
    if timeout_seconds <= 0:
        raise ValueError("provider timeout_seconds must be positive")
    if max_in_flight < 1:
        raise ValueError("provider max_in_flight must be positive")
    if attempt_index < 1:
        raise ValueError("provider attempt_index must be positive")
    started = time.monotonic()
    if diagnostics is not None:
        diagnostics.update(
            {
                "operation": operation,
                "call_role": call_role or operation,
                "strategy": strategy,
                "attempt_index": attempt_index,
                "timeout_seconds": timeout_seconds,
                "timed_out": False,
                "underlying_call_alive": False,
                "max_in_flight": max_in_flight,
                "elapsed_ms": None,
                "ttft_ms": None,
                "input_tokens": None,
                "output_tokens": None,
                "throughput_tokens_per_second": None,
            }
        )
    if not _reserve_provider_slot(max_in_flight):
        if diagnostics is not None:
            diagnostics.update(
                {
                    "success": False,
                    "failure_type": "in_flight_limit",
                    "elapsed_seconds": time.monotonic() - started,
                    "elapsed_ms": int((time.monotonic() - started) * 1000),
                }
            )
        raise RuntimeError(f"provider in-flight call limit reached ({max_in_flight})")
    result_queue: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

    def _run() -> None:
        try:
            result = callable_obj()
            _record_provider_completion(success=True)
            result_queue.put((True, result))
        except Exception as exc:  # noqa: BLE001 - preserve provider exceptions for the caller
            _record_provider_completion(success=False, failure_type=_provider_failure_category(exc))
            result_queue.put((False, exc))
        finally:
            _release_provider_slot()

    worker = threading.Thread(target=_run, name="kg-doc-parser-provider", daemon=True)
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        elapsed_seconds = time.monotonic() - started
        with _PROVIDER_IN_FLIGHT_LOCK:
            _PROVIDER_METRICS["timeouts_observed"] = int(_PROVIDER_METRICS["timeouts_observed"]) + 1
            _PROVIDER_METRICS["orphaned_calls"] = int(_PROVIDER_METRICS["orphaned_calls"]) + 1
        if diagnostics is not None:
            diagnostics.update(
                {
                    "timed_out": True,
                    "underlying_call_alive": True,
                    "elapsed_seconds": elapsed_seconds,
                    "elapsed_ms": int(elapsed_seconds * 1000),
                    "failure_type": "timeout",
                }
            )
        raise TimeoutError(f"provider call exceeded {timeout_seconds:g}s")
    succeeded, value = result_queue.get_nowait()
    elapsed_seconds = time.monotonic() - started
    if diagnostics is not None:
        diagnostics["elapsed_seconds"] = elapsed_seconds
        diagnostics["elapsed_ms"] = int(elapsed_seconds * 1000)
    if succeeded:
        if diagnostics is not None:
            usage = _provider_usage_metrics(value)
            diagnostics.update(usage)
            output_tokens = usage.get("output_tokens")
            if isinstance(output_tokens, (int, float)) and elapsed_seconds > 0:
                diagnostics["throughput_tokens_per_second"] = output_tokens / elapsed_seconds
            diagnostics["success"] = True
        return value
    if diagnostics is not None:
        diagnostics.update(
            {
                "success": False,
                "failure_type": _provider_failure_category(cast(BaseException, value)),
                "error_type": type(value).__name__,
            }
        )
    raise cast(Exception, value)


def _embedding_vector(text: str, *, dimension: int) -> list[float]:
    checksum = sum(ord(ch) for ch in text or "")
    return [
        float((len(text) + idx + 1) % 97 + 1 + (checksum % 13))
        for idx in range(max(1, dimension))
    ]


def _validate_embedding_vectors(
    value: Any,
    *,
    dimension: int,
    provider: str,
) -> list[list[float]]:
    """Reject provider output that cannot satisfy the declared vector profile."""
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{provider} embedding response must be a sequence of vectors")
    vectors: list[list[float]] = []
    for index, vector in enumerate(value):
        if not isinstance(vector, (list, tuple)):
            raise TypeError(f"{provider} embedding {index} must be a numeric vector")
        if len(vector) != dimension:
            raise ValueError(
                f"{provider} returned embedding dimension {len(vector)}; "
                f"expected configured dimension {dimension}"
            )
        row: list[float] = []
        for coordinate in vector:
            if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
                raise TypeError(f"{provider} returned a non-numeric embedding value")
            numeric = float(coordinate)
            if not math.isfinite(numeric):
                raise ValueError(f"{provider} returned a non-finite embedding value")
            row.append(numeric)
        vectors.append(row)
    return vectors


@dataclass
class _CallableEmbeddingFunction:
    name_value: str
    dimension: int
    provider: str

    def name(self) -> str:
        return self.name_value

    def __call__(self, documents_or_texts: list[str]) -> list[list[float]]:
        vectors = []
        for value in documents_or_texts:
            vectors.append(_embedding_vector(str(value or ""), dimension=self.dimension))
        return vectors


def build_embedding_function(
    spec: EmbeddingProviderConfig | None = None,
) -> EmbeddingFunctionLike:
    """Build a configurable embedding callable.

    Supported providers currently include fake, OpenAI, Vertex AI, and Ollama.
    The fake provider is deterministic and preferred for unit tests.
    """
    spec = spec or EmbeddingProviderConfig()
    if spec.provider == "fake":
        return _CallableEmbeddingFunction(
            name_value=spec.model,
            dimension=spec.dimension,
            provider=spec.provider,
        )

    def _build_langchain_embeddings() -> Any:
        if spec.provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            kwargs: dict[str, Any] = {
                "model": spec.model,
                "dimensions": spec.dimension,
            }
            if spec.base_url:
                kwargs["base_url"] = spec.base_url
            if spec.api_key_env and os.getenv(spec.api_key_env):
                kwargs["api_key"] = os.getenv(spec.api_key_env)
            return OpenAIEmbeddings(**kwargs)
        if spec.provider == "vertex":
            from langchain_google_vertexai import (  # pyright: ignore[reportMissingImports]
                VertexAIEmbeddings,
            )

            kwargs = {"model_name": spec.model}
            if spec.project:
                kwargs["project"] = spec.project
            if spec.location:
                kwargs["location"] = spec.location
            return VertexAIEmbeddings(**kwargs)
        if spec.provider == "ollama":
            from langchain_ollama import (  # pyright: ignore[reportMissingImports]
                OllamaEmbeddings,
            )

            kwargs = {"model": spec.model}
            if spec.base_url:
                kwargs["base_url"] = spec.base_url
            return OllamaEmbeddings(**kwargs)
        raise ValueError(f"unsupported embedding provider: {spec.provider}")

    embeddings = _build_langchain_embeddings()

    class _LangChainEmbeddingFunction:
        def name(self) -> str:
            return spec.model

        def __call__(self, documents_or_texts: list[str]) -> list[list[float]]:
            texts = [str(value or "") for value in documents_or_texts]
            if hasattr(embeddings, "embed_documents"):
                result = embeddings.embed_documents(texts)
                return _validate_embedding_vectors(
                    result, dimension=spec.dimension, provider=spec.provider
                )
            if hasattr(embeddings, "embed_query"):
                result = [embeddings.embed_query(text) for text in texts]
                return _validate_embedding_vectors(
                    result, dimension=spec.dimension, provider=spec.provider
                )
            raise TypeError(f"unsupported embedding backend: {type(embeddings)!r}")

    return _LangChainEmbeddingFunction()


def build_chat_model(
    spec: ProviderEndpointConfig | None = None,
    *,
    callbacks: list[object] | None = None,
) -> SupportsStructuredOutput:
    """Build a vendor-specific chat model behind a stable adapter boundary.

    Supported providers include Anthropic, Gemini, OpenAI-compatible, Ollama,
    Vertex, the Codex bridge, and fake. Vendor integrations are imported only
    when selected; optional provider packages need not be installed otherwise.
    """
    spec = spec or ProviderEndpointConfig()
    callbacks = callbacks or []
    if spec.provider == "fake":
        return cast(SupportsStructuredOutput, FakeChatModel())
    if spec.provider == "codex":
        if not spec.base_url or not spec.api_key_env or not os.getenv(spec.api_key_env):
            raise ValueError("codex provider requires base_url and a configured api_key_env token")
        timeout = float(os.getenv("KOGWISTAR_MAINTENANCE_CODEX_TIMEOUT_SECONDS", "300"))
        return StructuredBridgeChatModel(
            endpoint=spec.base_url,
            token=os.environ[spec.api_key_env],
            model=spec.model,
            timeout_seconds=timeout,
            max_retries=spec.max_retries,
        )
    if spec.provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs: dict[str, Any] = {
            "model": spec.model,
            "temperature": spec.temperature,
            "callbacks": callbacks,
            "max_retries": spec.max_retries,
        }
        if spec.max_output_tokens is not None:
            kwargs["max_output_tokens"] = spec.max_output_tokens
        if spec.api_key_env and os.getenv(spec.api_key_env):
            kwargs["google_api_key"] = os.getenv(spec.api_key_env)
        return cast(SupportsStructuredOutput, ChatGoogleGenerativeAI(**kwargs))
    if spec.provider == "openai":
        max_output_tokens = (
            spec.max_output_tokens
            if spec.max_output_tokens is not None
            else _configured_max_output_tokens()
        )
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": spec.model,
            "temperature": _chat_temperature_for_model(spec.model, spec.temperature),
            "callbacks": callbacks,
            "max_retries": spec.max_retries,
        }
        if max_output_tokens is not None:
            kwargs["max_tokens"] = max_output_tokens
        if spec.reasoning_effort:
            kwargs["model_kwargs"] = {"reasoning_effort": spec.reasoning_effort}
        if spec.base_url:
            kwargs["base_url"] = spec.base_url
        if spec.api_key_env and os.getenv(spec.api_key_env):
            kwargs["api_key"] = os.getenv(spec.api_key_env)
        return cast(SupportsStructuredOutput, ChatOpenAI(**kwargs))
    if spec.provider == "azure":
        max_output_tokens = (
            spec.max_output_tokens
            if spec.max_output_tokens is not None
            else _configured_max_output_tokens()
        )
        from langchain_openai import AzureChatOpenAI

        kwargs = {
            "azure_deployment": spec.model,
            "temperature": _chat_temperature_for_model(spec.model, spec.temperature),
            "callbacks": callbacks,
            "max_retries": spec.max_retries,
        }
        if max_output_tokens is not None:
            kwargs["max_tokens"] = max_output_tokens
        if spec.reasoning_effort:
            kwargs["model_kwargs"] = {"reasoning_effort": spec.reasoning_effort}
        if spec.base_url:
            kwargs["azure_endpoint"] = spec.base_url
        if spec.api_version:
            kwargs["api_version"] = spec.api_version
        elif os.getenv("OPENAI_API_VERSION"):
            kwargs["api_version"] = os.getenv("OPENAI_API_VERSION")
        elif os.getenv("AZURE_OPENAI_API_VERSION"):
            kwargs["api_version"] = os.getenv("AZURE_OPENAI_API_VERSION")
        if spec.api_key_env and os.getenv(spec.api_key_env):
            kwargs["api_key"] = os.getenv(spec.api_key_env)
        return cast(SupportsStructuredOutput, AzureChatOpenAI(**kwargs))
    if spec.provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError(
                "Anthropic provider selected; install the optional langchain-anthropic package"
            ) from exc

        kwargs = {
            "model": spec.model,
            "temperature": spec.temperature,
            "callbacks": callbacks,
            "max_retries": spec.max_retries,
        }
        if spec.max_output_tokens is not None:
            kwargs["max_tokens"] = spec.max_output_tokens
        if spec.base_url:
            kwargs["base_url"] = spec.base_url
        if spec.api_key_env and os.getenv(spec.api_key_env):
            kwargs["anthropic_api_key"] = os.getenv(spec.api_key_env)
        return cast(SupportsStructuredOutput, ChatAnthropic(**kwargs))
    if spec.provider == "ollama":
        from langchain_ollama import ChatOllama  # pyright: ignore[reportMissingImports]

        kwargs = {
            "model": spec.model,
            "temperature": _chat_temperature_for_model(spec.model, spec.temperature),
            "callbacks": callbacks,
        }
        if spec.max_output_tokens is not None:
            kwargs["num_predict"] = spec.max_output_tokens
        if spec.base_url:
            kwargs["base_url"] = spec.base_url
        return cast(SupportsStructuredOutput, ChatOllama(**kwargs))
    if spec.provider == "vertex":
        from langchain_google_vertexai import (  # pyright: ignore[reportMissingImports]
            ChatVertexAI,
        )

        kwargs = {
            "model": spec.model,
            "temperature": _chat_temperature_for_model(spec.model, spec.temperature),
            "callbacks": callbacks,
            "max_retries": spec.max_retries,
        }
        if spec.max_output_tokens is not None:
            kwargs["max_output_tokens"] = spec.max_output_tokens
        if spec.project:
            kwargs["project"] = spec.project
        if spec.location:
            kwargs["location"] = spec.location
        return cast(SupportsStructuredOutput, ChatVertexAI(**kwargs))
    raise ValueError(f"unsupported chat provider: {spec.provider}")


def build_chat_model_for_role(
    role: Literal["ocr", "parser"],
    spec: WorkflowProviderSettings | None = None,
    *,
    callbacks: list[object] | None = None,
) -> SupportsStructuredOutput:
    """Build the chat model used for either OCR or parsing.

    Examples:
    - role="ocr" with KG_DOC_OCR_PROVIDER=gemini for image OCR.
    - role="parser" with KG_DOC_PARSER_PROVIDER=openai for recipe extraction.
    - role="parser" with KG_DOC_PARSER_PROVIDER=ollama for local models.
    """
    settings = spec or WorkflowProviderSettings.from_env()
    chat_spec = settings.ocr if role == "ocr" else settings.parser
    if chat_spec.fallback_specs:
        models = [
            (item.provider, build_chat_model(item, callbacks=callbacks))
            for item in [chat_spec, *chat_spec.fallback_specs]
        ]
        return ProviderChainChatModel(models)
    return build_chat_model(chat_spec, callbacks=callbacks)
