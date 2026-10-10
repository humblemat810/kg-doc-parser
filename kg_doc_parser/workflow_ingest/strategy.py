"""Bounded parser-strategy selection for workflow ingestion.

The deterministic order is the safety net.  A provider may recommend a route,
but it cannot bypass the host's allowed strategies or the provider timeout.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Protocol, cast

from .serialization import JsonValue
from pydantic import BaseModel, Field, model_validator

from ..llm_structured_output import StructuredOutputRunnable
from .providers import (
    ProviderDiagnosticsSink,
    WorkflowProviderSettings,
    build_chat_model_for_role,
    invoke_with_timeout,
)

ParseStrategy = Literal["layer_excerpt", "layer_boundary", "page_index"]
ParseStrategyRequest = Literal["auto", "layer_excerpt", "layer_boundary", "page_index"]
ParseStrategySource = Literal["config", "hardcoded_fallback", "llm_triage", "llm_triage_fallback"]

HARD_CODED_STRATEGY_PRIORITY: tuple[ParseStrategy, ...] = (
    "layer_excerpt",
    "layer_boundary",
    "page_index",
)


class ParseStrategyAssessment(BaseModel):
    strategy: ParseStrategy
    advantages: list[str] = Field(default_factory=list, max_length=4)
    limitations: list[str] = Field(default_factory=list, max_length=4)


class ParseStrategyTriage(BaseModel):
    selected_strategy: ParseStrategy
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(default="", max_length=800)
    assessments: list[ParseStrategyAssessment] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def _require_complete_assessments(self) -> ParseStrategyTriage:
        assessments_by_strategy = {item.strategy: item for item in self.assessments}
        expected = set(HARD_CODED_STRATEGY_PRIORITY)
        if set(assessments_by_strategy) != expected:
            missing = sorted(expected.difference(assessments_by_strategy))
            extra = sorted(set(assessments_by_strategy).difference(expected))
            details = []
            if missing:
                details.append(f"missing={','.join(missing)}")
            if extra:
                details.append(f"unexpected={','.join(extra)}")
            raise ValueError(f"strategy triage must assess all allowed strategies ({'; '.join(details)})")
        if any(
            not any(str(value).strip() for value in item.advantages)
            or not any(str(value).strip() for value in item.limitations)
            for item in assessments_by_strategy.values()
        ):
            raise ValueError("strategy triage assessments require advantages and limitations")
        return self


class ParseStrategyDecision(BaseModel):
    selected_strategy: ParseStrategy
    source: ParseStrategySource
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(default="", max_length=800)
    assessments: list[ParseStrategyAssessment] = Field(default_factory=list, max_length=3)
    fallback_order: tuple[ParseStrategy, ...] = HARD_CODED_STRATEGY_PRIORITY


class StrategyTriageFn(Protocol):
    def __call__(self, context: Mapping[str, JsonValue], /) -> ParseStrategyTriage: ...


def hardcoded_strategy(
    requested: ParseStrategyRequest,
    *,
    strategy_order: tuple[ParseStrategy, ...] = HARD_CODED_STRATEGY_PRIORITY,
    disabled_strategies: set[ParseStrategy] | None = None,
    reason: str = "deterministic priority policy",
) -> ParseStrategyDecision:
    disabled = disabled_strategies or set()
    if set(strategy_order) != set(HARD_CODED_STRATEGY_PRIORITY) or len(strategy_order) != len(HARD_CODED_STRATEGY_PRIORITY):
        raise ValueError("strategy_order must contain each parser strategy exactly once")
    order: tuple[ParseStrategy, ...]
    if requested == "auto":
        order = strategy_order
    else:
        explicit = cast(ParseStrategy, requested)
        order = cast(
            tuple[ParseStrategy, ...],
            (explicit,)
            + tuple(
                cast(ParseStrategy, item)
                for item in strategy_order
                if item != explicit
            ),
        )
    available = cast(
        tuple[ParseStrategy, ...],
        tuple(item for item in order if item not in disabled),
    )
    if not available:
        raise ValueError("all parser strategies are disabled for this layer")
    if requested != "auto" and requested not in disabled:
        explicit = cast(ParseStrategy, requested)
        return ParseStrategyDecision(
            selected_strategy=explicit,
            source="config",
            confidence=1.0,
            rationale="explicit parse strategy requested by the caller",
            fallback_order=order,
        )
    return ParseStrategyDecision(
        selected_strategy=available[0],
        source="hardcoded_fallback",
        confidence=1.0,
        rationale=reason,
        fallback_order=order,
    )


def _triage_prompt(context: Mapping[str, JsonValue]) -> str:
    return (
        "Choose one parser strategy for this bounded document summary.\n"
        "Consider the trade-offs explicitly: layer_excerpt preserves verbatim leaf evidence and is the preferred "
        "default; layer_boundary is useful when structural cut points are clearer than excerpts; page_index is a "
        "deterministic structural fallback for headings and page sections.\n"
        "Assess every allowed strategy exactly once, with at least one advantage and one limitation for each, "
        "before selecting one.\n"
        "Return only the structured schema. Do not invent source facts.\n"
        f"Context: {context}"
    )


def build_llm_strategy_triage(
    provider_settings: WorkflowProviderSettings,
    *,
    diagnostics_sink: ProviderDiagnosticsSink | None = None,
) -> StrategyTriageFn:
    """Build a provider-backed triage callable with the parser timeout."""

    chat = build_chat_model_for_role("parser", provider_settings)
    from langchain_core.messages import HumanMessage, SystemMessage

    structured = cast(
        StructuredOutputRunnable[ParseStrategyTriage],
        chat.with_structured_output(ParseStrategyTriage, include_raw=True),
    )

    def _triage(context: Mapping[str, JsonValue]) -> ParseStrategyTriage:
        diagnostics: dict[str, object] = {}
        try:
            response = invoke_with_timeout(
                lambda: structured.invoke(
                    [
                        SystemMessage(content="You are a conservative parser-strategy triage classifier."),
                        HumanMessage(content=_triage_prompt(context)),
                    ]
                ),
                timeout_seconds=(
                    provider_settings.triage_timeout_seconds
                    or provider_settings.parser.timeout_seconds
                ),
                diagnostics=diagnostics,
                operation="parse_strategy_triage",
                max_in_flight=provider_settings.parser.max_in_flight_calls,
                attempt_index=1,
                call_role="triage",
                strategy="triage",
            )
            parsed = response.get("parsed") if isinstance(response, dict) else response
            if parsed is None:
                error = response.get("parsing_error") if isinstance(response, dict) else None
                raise ValueError(f"strategy triage parsing failed: {error!r}")
            result = parsed if isinstance(parsed, ParseStrategyTriage) else ParseStrategyTriage.model_validate(parsed)
        except Exception as exc:
            diagnostics.update(
                {
                    "success": False,
                    "failure_type": diagnostics.get("failure_type", "structured_output_parse_failure"),
                    "error_type": type(exc).__name__,
                }
            )
            if diagnostics_sink is not None:
                diagnostics_sink(dict(diagnostics))
            raise
        if diagnostics_sink is not None:
            diagnostics_sink(dict(diagnostics))
        return result

    return _triage


def select_parse_strategy(
    *,
    requested: ParseStrategyRequest,
    context: Mapping[str, JsonValue],
    triage_enabled: bool,
    triage_fn: StrategyTriageFn | None = None,
    strategy_order: tuple[ParseStrategy, ...] = HARD_CODED_STRATEGY_PRIORITY,
    disabled_strategies: set[ParseStrategy] | None = None,
    minimum_confidence: float = 0.55,
) -> ParseStrategyDecision:
    disabled = disabled_strategies or set()
    configured = hardcoded_strategy(
        requested, strategy_order=strategy_order, disabled_strategies=disabled
    )
    if requested != "auto" or not triage_enabled or triage_fn is None:
        return configured
    try:
        triage = triage_fn(context)
        if triage.confidence < minimum_confidence or triage.selected_strategy in disabled:
            return configured.model_copy(
                update={
                    "source": "llm_triage_fallback",
                    "rationale": (
                        f"triage confidence {triage.confidence:.2f} below {minimum_confidence:.2f}"
                        if triage.confidence < minimum_confidence
                        else f"triage selected disabled strategy {triage.selected_strategy}"
                    ),
                }
            )
        return ParseStrategyDecision(
            selected_strategy=triage.selected_strategy,
            source="llm_triage",
            confidence=triage.confidence,
            rationale=triage.rationale,
            assessments=list(triage.assessments),
            fallback_order=strategy_order,
        )
    except Exception as exc:  # noqa: BLE001 - triage must never block deterministic parsing.
        return configured.model_copy(
            update={
                "source": "llm_triage_fallback",
                "rationale": f"triage failed: {type(exc).__name__}: {exc}",
            }
        )


__all__ = [
    "HARD_CODED_STRATEGY_PRIORITY",
    "ParseStrategy",
    "ParseStrategyAssessment",
    "ParseStrategyDecision",
    "ParseStrategyRequest",
    "ParseStrategyTriage",
    "StrategyTriageFn",
    "build_llm_strategy_triage",
    "hardcoded_strategy",
    "select_parse_strategy",
]
