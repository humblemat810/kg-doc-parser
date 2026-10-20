from __future__ import annotations

from typing import cast

from kogwistar.runtime import MappingStepResolver, StepContext
from kogwistar.runtime.models import (
    StepRunResult,
)

from .handler_state import (
    parser_source_map as _parser_source_map,
)
from .handler_support import (
    register_step as _register_step,
)
from .handler_support import (
    success as _success,
)
from .model_validation import validate_backend_payload
from .models import (
    CurrentLayerContext,
    ParseSessionState,
    StrategyExecutionRecord,
    WorkflowIngestInput,
)
from .runtime_contracts import WorkflowRuntimeDeps
from .serialization import JsonValue
from .strategy import (
    ParseStrategy,
    build_llm_strategy_triage,
    select_parse_strategy,
)


def register_strategy_triage_step(
    resolver: MappingStepResolver,
    *,
    runtime_deps: WorkflowRuntimeDeps,
) -> None:
    @_register_step(
        resolver, step_name="triage_parse_strategy", runtime_deps=runtime_deps
    )
    def _triage_parse_strategy(ctx: StepContext) -> StepRunResult:
        current_layer_context = validate_backend_payload(
            CurrentLayerContext, ctx.state_view["current_layer_context"]
        )
        parse_session = validate_backend_payload(
            ParseSessionState, ctx.state_view["parse_session"]
        )
        normalized_input = validate_backend_payload(
            WorkflowIngestInput, ctx.state_view["normalized_input"]
        )
        layer_metadata = dict(current_layer_context.metadata)
        raw_disabled = layer_metadata.get("disabled_strategies")
        disabled_strategies: set[ParseStrategy] = {
            cast(ParseStrategy, value)
            for value in (raw_disabled if isinstance(raw_disabled, list) else [])
            if value in {"layer_excerpt", "layer_boundary", "page_index"}
        }
        settings = runtime_deps.get("provider_settings")
        triage_fn = runtime_deps.get("triage_strategy_fn")
        if normalized_input.parse_strategy is not None:
            requested = normalized_input.parse_strategy
        elif settings is not None:
            requested = settings.parse_strategy
        elif triage_fn is not None:
            requested = "auto"
        elif normalized_input.parse_strategy_order is not None:
            # An explicit cascade is meaningful even for direct callers that
            # do not construct provider settings. Do not let the legacy
            # excerpt-first default reorder the caller's permutation.
            requested = "auto"
        else:
            # Preserve the existing runtime-dependency configuration for
            # callers that have not opted into provider strategy triage.
            requested = (
                "layer_boundary"
                if parse_session.metadata.get("default_split_strategy")
                == "boundary_first"
                else "layer_excerpt"
            )
        configured_order = tuple(
            normalized_input.parse_strategy_order
            or (
                settings.parse_strategy_order
                if settings is not None
                else ("layer_excerpt", "layer_boundary", "page_index")
            )
        )
        triage_enabled = (
            normalized_input.triage_enabled
            if normalized_input.triage_enabled is not None
            else (
                settings.triage_enabled
                if settings is not None
                else triage_fn is not None
            )
        )
        page_index_summary_enabled = (
            normalized_input.page_index_summary_enabled
            if normalized_input.page_index_summary_enabled is not None
            else (settings.page_index_summary_enabled if settings is not None else True)
        )
        page_index_hierarchical_summary_enabled = (
            normalized_input.page_index_hierarchical_summary_enabled
            if normalized_input.page_index_hierarchical_summary_enabled is not None
            else (
                settings.page_index_hierarchical_summary_enabled
                if settings is not None
                else False
            )
        )
        triage_build_error: str | None = None
        if (
            triage_fn is None
            and settings is not None
            and triage_enabled
            and requested == "auto"
        ):
            try:
                triage_fn = build_llm_strategy_triage(
                    settings,
                    diagnostics_sink=runtime_deps.get("provider_diagnostics_sink"),
                )
            except Exception as exc:  # noqa: BLE001 - unavailable providers use deterministic fallback.
                triage_build_error = (
                    f"triage provider unavailable: {type(exc).__name__}: {exc}"
                )
        parent_context: list[dict[str, JsonValue]] = []
        parser_source_map = _parser_source_map(ctx.state_view.get("parser_source_map"))
        for parent_id, title in zip(
            current_layer_context.parent_node_ids,
            current_layer_context.parent_titles,
        ):
            pointers = current_layer_context.parent_content_pointers_by_id.get(
                parent_id, []
            )
            excerpts = []
            for pointer in pointers[:3]:
                source = parser_source_map.get(pointer.source_cluster_id, {})
                raw_source_text = (
                    source.get("text") if isinstance(source, dict) else None
                )
                source_text = (
                    raw_source_text if isinstance(raw_source_text, str) else ""
                )
                end = (
                    len(source_text) if pointer.end_char == -1 else pointer.end_char + 1
                )
                excerpts.append(source_text[max(0, pointer.start_char) : end][:500])
            parent_context.append(
                {
                    "node_id": parent_id,
                    "title": title,
                    "depth": current_layer_context.depth,
                    "source_excerpts": excerpts,
                }
            )
        context = {
            "layer_depth": current_layer_context.depth,
            "parent_count": len(current_layer_context.parent_node_ids),
            "parents": parent_context,
            "current_strategy": current_layer_context.split_strategy,
            "default_priority": ["layer_excerpt", "layer_boundary", "page_index"],
        }
        try:
            decision = select_parse_strategy(
                requested=requested,
                context=context,
                # A configured provider enables model triage; an injected triage
                # function is also a deliberate caller opt-in.  Without either,
                # the deterministic priority policy is used.
                triage_enabled=triage_enabled,
                triage_fn=triage_fn,  # type: ignore[arg-type]
                strategy_order=configured_order,
                disabled_strategies=disabled_strategies,
            )
        except ValueError as exc:
            with ctx.state_write as st:
                st["workflow_errors"] = [str(exc)]
                st["strategy_selection_error"] = str(exc)
            return _success()
        if triage_build_error is not None and decision.source == "hardcoded_fallback":
            decision = decision.model_copy(
                update={
                    "source": "llm_triage_fallback",
                    "rationale": triage_build_error,
                }
            )
        metadata = dict(parse_session.metadata)
        metadata.update(
            cast(
                dict[str, JsonValue],
                {
                    "parse_strategy": decision.selected_strategy,
                    "parse_strategy_source": decision.source,
                    "parse_strategy_confidence": decision.confidence,
                    "parse_strategy_rationale": decision.rationale,
                    "parse_strategy_assessments": [
                        assessment.model_dump(mode="json")
                        for assessment in decision.assessments
                    ],
                    "parse_strategy_fallback_order": list(decision.fallback_order),
                    "disabled_strategies": sorted(disabled_strategies),
                    "page_index_summary_enabled": page_index_summary_enabled,
                    "page_index_hierarchical_summary_enabled": page_index_hierarchical_summary_enabled,
                },
            )
        )
        selected_split_strategy = (
            "boundary_first"
            if decision.selected_strategy == "layer_boundary"
            else "excerpt_first"
        )
        strategy_history = list(parse_session.strategy_history)
        strategy_switch_count = parse_session.strategy_switch_count
        if strategy_history[-1:] != [selected_split_strategy]:
            strategy_history.append(selected_split_strategy)
            strategy_switch_count += 1
        if decision.selected_strategy == "layer_boundary":
            updated_session = parse_session.model_copy(
                update={
                    "split_strategy": "boundary_first",
                    "fallback_split_strategy": "excerpt_first",
                    "strategy_history": strategy_history,
                    "strategy_switch_count": strategy_switch_count,
                    "metadata": metadata,
                }
            )
        else:
            updated_session = parse_session.model_copy(
                update={
                    "split_strategy": "excerpt_first",
                    "fallback_split_strategy": "boundary_first",
                    "strategy_history": strategy_history,
                    "strategy_switch_count": strategy_switch_count,
                    "metadata": metadata,
                }
            )
        updated_context = current_layer_context.model_copy(
            update={
                "split_strategy": (
                    "boundary_first"
                    if decision.selected_strategy == "layer_boundary"
                    else "excerpt_first"
                ),
                "metadata": {
                    **current_layer_context.metadata,
                    "parse_strategy": decision.selected_strategy,
                    "parse_strategy_source": decision.source,
                    "parse_strategy_confidence": decision.confidence,
                    "parse_strategy_rationale": decision.rationale,
                    "parse_strategy_assessments": [
                        assessment.model_dump(mode="json")
                        for assessment in decision.assessments
                    ],
                    "disabled_strategies": sorted(disabled_strategies),
                    "page_index_attempted": False,
                    "page_index_summary_enabled": page_index_summary_enabled,
                    "page_index_hierarchical_summary_enabled": page_index_hierarchical_summary_enabled,
                },
                "retry_count": 0,
            }
        )
        with ctx.state_write as st:
            st["parse_session"] = updated_session.model_dump(
                field_mode="backend", dump_format="json"
            )
            st["current_layer_context"] = updated_context.model_dump(
                field_mode="backend", dump_format="json"
            )
            st["parse_strategy_decision"] = decision.model_dump(mode="json")
        raw_attempt_counts = ctx.state_view.get("strategy_attempt_counts")
        attempt_counts = {
            str(key): int(value)
            for key, value in (
                raw_attempt_counts.items()
                if isinstance(raw_attempt_counts, dict)
                else []
            )
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        attempt = int(attempt_counts.get(decision.selected_strategy, 0)) + 1
        attempt_counts[decision.selected_strategy] = attempt
        selected_record = StrategyExecutionRecord(
            strategy=decision.selected_strategy,
            depth=current_layer_context.depth,
            parent_node_ids=list(current_layer_context.parent_node_ids),
            attempt=attempt,
            event="selected",
        )
        # Routing is intentionally selected by the persisted predicate edges.
        # No provider response or handler shortcut may bypass disabled methods.
        return _success(
            state_update=[
                ("u", {"strategy_attempt_counts": attempt_counts}),
                (
                    "a",
                    {
                        "strategy_execution_history": selected_record.model_dump(
                            mode="json"
                        )
                    },
                ),
            ]
        )
