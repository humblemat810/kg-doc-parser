from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Literal, Protocol, TypedDict, cast

from kogwistar.json_types import JsonValue
from kogwistar.runtime import MappingStepResolver
from kogwistar.runtime.models import (
    RunFailure,
    RunSuccess,
    RunSuspended,
    StateUpdate,
    StepRunResult,
)
from kogwistar.runtime.runtime import StepContext

from .adapters import (
    build_authoritative_source_map,
    build_parser_input_dict,
    build_parser_source_map,
    select_primary_collection,
)
from .cache import WorkflowLLMCallCache
from .clients import CanonicalGraphPersistenceClient
from .models import (
    CanonicalGraphWriteResult,
    CurrentLayerContext,
    CurrentLayerResult,
    CurrentLayerReview,
    GroundedSourceRecord,
    LayerFrontierItem,
    ParseSessionState,
    StrategyExecutionRecord,
    ValidationReport,
    WorkflowExportBundle,
    WorkflowIngestInput,
)
from .page_index import PageIndexSourceFormat, parse_page_index_layer
from .parser_core import (
    ParserPayload,
    ParserSourceMap,
    ParseSemanticFn,
    ProposeLayerFn,
    ReviewLayerFn,
    SplitStrategy,
    apply_cud_update,
    check_layer_coverage,
    commit_layer_children,
    dedupe_and_filter_layer,
    default_parse_semantic_fn,
    enqueue_next_layer_frontier,
    finalize_semantic_tree,
    initialize_parse_session,
    prepare_layer_frontier,
    propose_layer_breakdown,
    repair_layer_candidates,
    requeue_failed_frontier_items,
    review_layer,
    switch_split_strategy,
    validate_layer_commit,
)
from .probe import WorkflowProbe, emit_probe_event
from .providers import ProviderDiagnosticsSink, WorkflowProviderSettings
from .semantics import (
    HydratedTextPointer,
    SemanticNode,
    classify_terminal_coverage_status,
    compute_pointer_coverage,
    compute_terminal_content_coverage,
    correct_and_validate_pointer,
    semantic_tree_to_kge_payload,
)
from .strategy import (
    ParseStrategy,
    StrategyTriageFn,
    build_llm_strategy_triage,
    select_parse_strategy,
)

_LOGGER = logging.getLogger(__name__)


def _strategy_attempt(state_view: Mapping[str, object], strategy: str) -> int:
    raw = state_view.get("strategy_attempt_counts")
    if not isinstance(raw, dict):
        return 1
    value = raw.get(strategy)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return 1


def _parser_payload(value: object) -> ParserPayload:
    """Decode one JSON object from runtime state into parser payload shape."""

    if not isinstance(value, dict):
        raise TypeError("parser payload must be a JSON object")
    return dict(value)


def _parser_source_map(value: object | None) -> ParserSourceMap:
    """Decode the nested parser source map carried through checkpoint state."""

    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("parser source map must be a JSON object")
    source_map: ParserSourceMap = {}
    for cluster_id, payload in value.items():
        if not isinstance(payload, dict):
            raise TypeError(f"parser source cluster {cluster_id!r} must be a JSON object")
        source_map[str(cluster_id)] = dict(payload)
    return source_map


def _correct_parser_pointer(
    pointer: HydratedTextPointer,
    parser_source_map: ParserSourceMap,
) -> HydratedTextPointer | None:
    """Adapt rich parser payloads to the text-only pointer validator."""

    text_source_map: dict[str, dict[str, JsonValue]] = {}
    for cluster_id, payload in parser_source_map.items():
        text = payload.get("text", "")
        text_source_map[cluster_id] = {"text": text if isinstance(text, str) else str(text)}
    return correct_and_validate_pointer(pointer, text_source_map)


def _state_object_map(value: object) -> dict[str, dict[str, JsonValue]]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): cast(dict[str, JsonValue], item)
        for key, item in value.items()
        if isinstance(item, dict)
    }


def _state_list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _state_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _state_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _coverage_map(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): _state_float(item)
        for key, item in value.items()
    }


def _page_index_source_format(inp: WorkflowIngestInput) -> PageIndexSourceFormat:
    """Read the document format carried by the normalized collection metadata."""

    collection = select_primary_collection(inp)
    candidates: list[object] = [collection.metadata.get("source_format")]
    for page in collection.pages:
        candidates.append(page.metadata.get("source_format"))
        for unit in page.units:
            candidates.append(unit.metadata.get("source_format"))
    for value in candidates:
        if value in {"text", "markdown"}:
            return cast(Literal["text", "markdown"], value)
    return "text"


class StepHandler(Protocol):
    """Execute one parser workflow step against the runtime context."""

    def __call__(self, context: StepContext, /) -> StepRunResult: ...


class WorkflowRuntimeDeps(TypedDict, total=False):
    """Optional, typed dependencies injected into workflow steps.

    The state carried by the runtime remains JSON-like and intentionally
    dynamic. This contract applies only to executable collaborators and
    bounded workflow policy, so a malformed dependency cannot silently pass
    through as an arbitrary object.
    """

    probe: WorkflowProbe | None
    parse_semantic_fn: ParseSemanticFn
    propose_layer_fn: ProposeLayerFn
    review_layer_fn: ReviewLayerFn
    llm_cache: WorkflowLLMCallCache
    graph_persistence_client: CanonicalGraphPersistenceClient
    persistence_mode: str
    kg_authority: str
    max_depth: int
    allow_review: bool
    split_strategy: SplitStrategy
    fallback_split_strategy: SplitStrategy
    max_review_retries: int
    layer_frontier_batch_size: int
    coverage_threshold: float
    provider_settings: WorkflowProviderSettings
    triage_strategy_fn: StrategyTriageFn
    provider_diagnostics_sink: ProviderDiagnosticsSink


def _build_export_bundle(
    *,
    ctx: StepContext,
    runtime_deps: WorkflowRuntimeDeps | None = None,
) -> WorkflowExportBundle:
    normalized = WorkflowIngestInput.model_validate(ctx.state_view["normalized_input"])
    collection = select_primary_collection(normalized)
    tree = SemanticNode.model_validate(ctx.state_view["semantic_tree"])
    graph_payload = semantic_tree_to_kge_payload(tree, doc_id=collection.collection_id)
    deps = runtime_deps or {}
    persistence_mode = str(ctx.state_view.get("persistence_mode", deps.get("persistence_mode", "local_debug")))
    kg_authority = str(ctx.state_view.get("kg_authority", deps.get("kg_authority", "local")))
    return WorkflowExportBundle(
        graph_payload=graph_payload,
        authoritative_source_map=cast(
            dict[str, GroundedSourceRecord],
            ctx.state_view["authoritative_source_map"],
        ),
        embedding_spaces=collection.embedding_spaces,
        consolidation_candidates=[],
        retrieval_metadata=cast(
            dict[str, JsonValue],
            {
                "embedding_spaces": collection.embedding_spaces,
                "supports_hybrid_retrieval": True,
                "supports_split_embedding_spaces": True,
                "collection_modality": collection.modality,
            },
        ),
        persistence_mode="server_canonical" if persistence_mode == "server_canonical" else "local_debug",
        kg_authority="server" if kg_authority == "server" else "local",
        canonical_write_confirmed=False,
        parser_owner="local",
        server_parser_used=False,
        persisted_to_knowledge_engine=False,
    )


def _success(
    next_step: str | None = None,
    *,
    state_update: list[StateUpdate] | None = None,
) -> RunSuccess:
    return RunSuccess(
        conversation_node_id=None,
        state_update=state_update or [],
        _route_next=[] if next_step is None else [next_step],
    )


def _probe_snapshot(state_view: dict[str, object]) -> dict[str, object]:
    snapshot: dict[str, object] = {
        "state_keys": sorted(k for k, v in state_view.items() if v is not None),
    }
    frontier = state_view.get("layer_frontier_queue")
    if isinstance(frontier, list):
        snapshot["frontier_size"] = len(frontier)
    current_layer_context = state_view.get("current_layer_context")
    if isinstance(current_layer_context, dict):
        snapshot["current_depth"] = current_layer_context.get("depth")
        snapshot["retry_count"] = current_layer_context.get("retry_count")
        snapshot["split_strategy"] = current_layer_context.get("split_strategy")
    parse_session = state_view.get("parse_session")
    if isinstance(parse_session, dict):
        snapshot["parse_mode"] = parse_session.get("mode")
        snapshot["session_depth"] = parse_session.get("current_depth")
        snapshot["session_strategy"] = parse_session.get("split_strategy")
        snapshot["strategy_switch_count"] = parse_session.get("strategy_switch_count")
    return snapshot


def _progress_bar(done: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return "?" * width
    filled = min(width, max(0, round((done / total) * width)))
    return ("█" * filled) + ("░" * (width - filled))


def _log_runtime_progress(*, step_name: str, state_view: dict[str, object]) -> None:
    current_layer_context = state_view.get("current_layer_context")
    parse_session = state_view.get("parse_session")
    if not isinstance(current_layer_context, dict) or not isinstance(parse_session, dict):
        return
    depth = int(current_layer_context.get("depth", 0))
    max_depth = int(parse_session.get("max_depth", 10) or 10)
    layer_num = depth + 1
    retry_count = int(current_layer_context.get("retry_count", 0))
    max_retries = int(current_layer_context.get("max_retries", 3) or 3)
    strategy = str(current_layer_context.get("split_strategy", "excerpt_first"))
    fallback_strategy = str(parse_session.get("fallback_split_strategy", "boundary_first"))
    strategy_idx = 2 if strategy == fallback_strategy else 1
    strategy_cap = 2
    phase = step_name.replace("_", " ")
    pct = round((layer_num / max_depth) * 100) if max_depth > 0 else 0
    _LOGGER.info(
        "⏳ layer %s/%s | %3s%% | %s | iter %s/%s | strategy %s/%s %s | %s",
        layer_num,
        max_depth,
        pct,
        _progress_bar(layer_num, max_depth),
        retry_count + 1,
        max_retries,
        strategy_idx,
        strategy_cap,
        strategy,
        phase,
    )


def _register_step(
    resolver: MappingStepResolver,
    *,
    step_name: str,
    runtime_deps: WorkflowRuntimeDeps,
) -> Callable[[StepHandler], StepHandler]:
    probe = runtime_deps.get("probe")

    def decorator(fn: StepHandler) -> StepHandler:
        @resolver.register(step_name)
        def _wrapped(ctx: StepContext) -> StepRunResult:
            _log_runtime_progress(step_name=step_name, state_view=dict(ctx.state_view))
            emit_probe_event(
                probe,
                "workflow.step_started",
                step=step_name,
                **_probe_snapshot(dict(ctx.state_view)),
            )
            try:
                result = fn(ctx)
            except Exception as exc:
                emit_probe_event(
                    probe,
                    "workflow.step_exception",
                    step=step_name,
                    error=repr(exc),
                    **_probe_snapshot(dict(ctx.state_view)),
                )
                raise
            if isinstance(result, RunFailure):
                status = "failure"
            elif isinstance(result, RunSuspended):
                status = "suspended"
            else:
                status = "success"
            emit_probe_event(
                probe,
                "workflow.step_finished",
                step=step_name,
                status=status,
                route_next=list(getattr(result, "_route_next", []) or []),
                **_probe_snapshot(dict(ctx.state_view)),
            )
            return result

        return cast(StepHandler, _wrapped)

    return decorator


def register_base_ingest_steps(
    resolver: MappingStepResolver,
    *,
    runtime_deps: WorkflowRuntimeDeps,
) -> None:
    @_register_step(resolver, step_name="start", runtime_deps=runtime_deps)
    def _start(ctx: StepContext) -> StepRunResult:
        return _success("normalize_input")

    @_register_step(resolver, step_name="normalize_input", runtime_deps=runtime_deps)
    def _normalize_input(ctx: StepContext) -> StepRunResult:
        payload = WorkflowIngestInput.model_validate(ctx.state_view["input"]).model_dump(
            field_mode="backend",
            dump_format="json",
        )
        with ctx.state_write as st:
            st["normalized_input"] = payload
        return _success("build_source_map")

    @_register_step(resolver, step_name="build_source_map", runtime_deps=runtime_deps)
    def _build_source_map(ctx: StepContext) -> StepRunResult:
        normalized = WorkflowIngestInput.model_validate(ctx.state_view["normalized_input"])
        authoritative_source_map = build_authoritative_source_map(normalized)
        collection = select_primary_collection(normalized)
        parser_input_dict = build_parser_input_dict(collection)
        parser_source_map = build_parser_source_map(authoritative_source_map)
        with ctx.state_write as st:
            st["authoritative_source_map"] = {
                k: v.model_dump(field_mode="backend", dump_format="json")
                for k, v in authoritative_source_map.items()
            }
            st["parser_input_dict"] = parser_input_dict
            st["parser_source_map"] = parser_source_map
        return _success("init_parse_session")

    @_register_step(resolver, step_name="init_parse_session", runtime_deps=runtime_deps)
    def _init_parse_session(ctx: StepContext) -> StepRunResult:
        normalized = WorkflowIngestInput.model_validate(ctx.state_view["normalized_input"])
        collection = select_primary_collection(normalized)
        parse_semantic_fn = runtime_deps.get("parse_semantic_fn", default_parse_semantic_fn)
        propose_layer_fn = runtime_deps.get("propose_layer_fn")
        session, frontier, root = initialize_parse_session(
            collection=collection,
            parser_input_dict=_parser_payload(ctx.state_view["parser_input_dict"]),
            parser_source_map=_parser_source_map(ctx.state_view["parser_source_map"]),
            max_depth=int(runtime_deps.get("max_depth", 10)),
            allow_review=bool(runtime_deps.get("allow_review", True)),
            split_strategy=runtime_deps.get("split_strategy", "excerpt_first"),
            fallback_split_strategy=runtime_deps.get("fallback_split_strategy", "boundary_first"),
            parse_semantic_fn=parse_semantic_fn if propose_layer_fn is None else None,
        )
        with ctx.state_write as st:
            st["parse_session"] = session.model_dump(field_mode="backend", dump_format="json")
            st["layer_frontier_queue"] = [
                item.model_dump(field_mode="backend", dump_format="json") for item in frontier
            ]
            st["semantic_tree"] = root.model_dump()
        return _success("check_frontier_remaining")


def register_layerwise_parser_steps(
    resolver: MappingStepResolver,
    *,
    runtime_deps: WorkflowRuntimeDeps,
) -> None:
    @_register_step(resolver, step_name="triage_parse_strategy", runtime_deps=runtime_deps)
    def _triage_parse_strategy(ctx: StepContext) -> StepRunResult:
        current_layer_context = CurrentLayerContext.model_validate(
            ctx.state_view["current_layer_context"]
        )
        parse_session = ParseSessionState.model_validate(ctx.state_view["parse_session"])
        normalized_input = WorkflowIngestInput.model_validate(ctx.state_view["normalized_input"])
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
                if parse_session.metadata.get("default_split_strategy") == "boundary_first"
                else "layer_excerpt"
            )
        configured_order = tuple(
            normalized_input.parse_strategy_order
            or (settings.parse_strategy_order if settings is not None else ("layer_excerpt", "layer_boundary", "page_index"))
        )
        triage_enabled = (
            normalized_input.triage_enabled
            if normalized_input.triage_enabled is not None
            else (settings.triage_enabled if settings is not None else triage_fn is not None)
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
        if triage_fn is None and settings is not None and triage_enabled and requested == "auto":
            try:
                triage_fn = build_llm_strategy_triage(
                    settings,
                    diagnostics_sink=runtime_deps.get("provider_diagnostics_sink"),
                )
            except Exception as exc:  # noqa: BLE001 - unavailable providers use deterministic fallback.
                triage_build_error = f"triage provider unavailable: {type(exc).__name__}: {exc}"
        parent_context: list[dict[str, JsonValue]] = []
        parser_source_map = _parser_source_map(ctx.state_view.get("parser_source_map"))
        for parent_id, title in zip(
            current_layer_context.parent_node_ids,
            current_layer_context.parent_titles,
        ):
            pointers = current_layer_context.parent_content_pointers_by_id.get(parent_id, [])
            excerpts = []
            for pointer in pointers[:3]:
                source = parser_source_map.get(pointer.source_cluster_id, {})
                source_text = str(source.get("text", ""))
                end = len(source_text) if pointer.end_char == -1 else pointer.end_char + 1
                excerpts.append(source_text[max(0, pointer.start_char):end][:500])
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
                    assessment.model_dump(mode="json") for assessment in decision.assessments
                ],
                "parse_strategy_fallback_order": list(decision.fallback_order),
                "disabled_strategies": sorted(disabled_strategies),
                    "page_index_summary_enabled": page_index_summary_enabled,
                    "page_index_hierarchical_summary_enabled": page_index_hierarchical_summary_enabled,
                },
            )
        )
        selected_split_strategy = (
            "boundary_first" if decision.selected_strategy == "layer_boundary" else "excerpt_first"
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
                        assessment.model_dump(mode="json") for assessment in decision.assessments
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
            st["parse_session"] = updated_session.model_dump(field_mode="backend", dump_format="json")
            st["current_layer_context"] = updated_context.model_dump(
                field_mode="backend", dump_format="json"
            )
            st["parse_strategy_decision"] = decision.model_dump(mode="json")
        raw_attempt_counts = ctx.state_view.get("strategy_attempt_counts")
        attempt_counts = {
            str(key): int(value)
            for key, value in (raw_attempt_counts.items() if isinstance(raw_attempt_counts, dict) else [])
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
                ("a", {"strategy_execution_history": selected_record.model_dump(mode="json")}),
            ]
        )

    @_register_step(resolver, step_name="page_index_layer", runtime_deps=runtime_deps)
    def _page_index_layer(ctx: StepContext) -> StepRunResult:
        current_layer_context = CurrentLayerContext.model_validate(ctx.state_view["current_layer_context"])
        normalized_input = WorkflowIngestInput.model_validate(ctx.state_view["normalized_input"])
        parser_source_map = _parser_source_map(ctx.state_view.get("parser_source_map"))
        source_format = _page_index_source_format(normalized_input)
        candidates = []
        for parent_id, parent_title in zip(
            current_layer_context.parent_node_ids,
            current_layer_context.parent_titles,
        ):
            candidates.extend(
                parse_page_index_layer(
                    parent_id=parent_id,
                    parent_title=parent_title,
                    parent_pointers=current_layer_context.parent_content_pointers_by_id.get(parent_id, []),
                    parser_source_map=parser_source_map,
                    source_format=source_format,
                    summary_enabled=bool(current_layer_context.metadata.get("page_index_summary_enabled", True)),
                )
            )
        result = CurrentLayerResult(
            children=candidates,
            satisfied=bool(candidates),
            metadata={
                "parse_strategy": "page_index",
                "page_index_layer_only": True,
                "allow_empty_layer": not bool(candidates),
                "atomic_retained": not bool(candidates),
            },
        )
        with ctx.state_write as st:
            context = current_layer_context.model_copy(
                update={
                    "metadata": {
                        **current_layer_context.metadata,
                        "page_index_attempted": True,
                    }
                }
            )
            st["current_layer_context"] = context.model_dump(
                field_mode="backend", dump_format="json"
            )
            st["current_layer_result"] = result.model_dump(
                field_mode="backend", dump_format="json"
            )
        return _success()

    @_register_step(resolver, step_name="check_frontier_remaining", runtime_deps=runtime_deps)
    def _check_frontier_remaining(ctx: StepContext) -> StepRunResult:
        queue = ctx.state_view.get("layer_frontier_queue") or []
        if queue:
            return _success("prepare_layer_frontier")
        return _success("finalize_semantic_tree")

    @_register_step(resolver, step_name="prepare_layer_frontier", runtime_deps=runtime_deps)
    def _prepare_layer_frontier(ctx: StepContext) -> StepRunResult:
        parse_session = ParseSessionState.model_validate(ctx.state_view["parse_session"])
        semantic_tree = SemanticNode.model_validate(ctx.state_view["semantic_tree"])
        configured_batch_size = runtime_deps.get("layer_frontier_batch_size")
        if configured_batch_size is None:
            settings = runtime_deps.get("provider_settings")
            configured_batch_size = (
                getattr(settings, "layer_frontier_batch_size", None)
                if settings is not None
                else 1
            )
        raw_frontier = ctx.state_view.get("layer_frontier_queue")
        frontier_items = raw_frontier if isinstance(raw_frontier, list) else []
        context, remaining, updated_session = prepare_layer_frontier(
            parse_session=parse_session,
            frontier_queue=[
                LayerFrontierItem.model_validate(item)
                for item in frontier_items
            ],
            semantic_tree=semantic_tree,
            max_retries=int(runtime_deps.get("max_review_retries", 3)),
            max_items=(int(configured_batch_size) if configured_batch_size is not None else None),
        )
        with ctx.state_write as st:
            st["parse_session"] = updated_session.model_dump(field_mode="backend", dump_format="json")
            st["current_layer_context"] = context.model_dump(field_mode="backend", dump_format="json")
            st["layer_frontier_queue"] = [
                item.model_dump(field_mode="backend", dump_format="json") for item in remaining
            ]
        return _success("triage_parse_strategy")

    @_register_step(resolver, step_name="propose_layer_breakdown", runtime_deps=runtime_deps)
    def _propose_layer_breakdown(ctx: StepContext) -> StepRunResult:
        normalized = WorkflowIngestInput.model_validate(ctx.state_view["normalized_input"])
        collection = select_primary_collection(normalized)
        parse_session = ParseSessionState.model_validate(ctx.state_view["parse_session"])
        current_layer_context = CurrentLayerContext.model_validate(ctx.state_view["current_layer_context"])
        semantic_tree = SemanticNode.model_validate(ctx.state_view["semantic_tree"])
        result = propose_layer_breakdown(
            collection=collection,
            parser_input_dict=_parser_payload(ctx.state_view["parser_input_dict"]),
            parser_source_map=_parser_source_map(ctx.state_view["parser_source_map"]),
            parse_session=parse_session,
            current_layer_context=current_layer_context,
            semantic_tree=semantic_tree,
            propose_layer_fn=runtime_deps.get("propose_layer_fn"),
            llm_cache=runtime_deps.get("llm_cache"),
        )
        with ctx.state_write as st:
            st["current_layer_result"] = result.model_dump(field_mode="backend", dump_format="json")
        return _success()

    @_register_step(resolver, step_name="review_cud_proposal", runtime_deps=runtime_deps)
    def _review_cud_proposal(ctx: StepContext) -> StepRunResult:
        parse_session = ParseSessionState.model_validate(ctx.state_view["parse_session"])
        current_layer_context = CurrentLayerContext.model_validate(ctx.state_view["current_layer_context"])
        current_layer_result = CurrentLayerResult.model_validate(ctx.state_view["current_layer_result"])
        reviewed, updated_session = review_layer(
            parse_session=parse_session,
            current_layer_context=current_layer_context,
            current_layer_result=current_layer_result,
            parser_source_map=_parser_source_map(ctx.state_view["parser_source_map"]),
            review_layer_fn=runtime_deps.get("review_layer_fn"),
            llm_cache=runtime_deps.get("llm_cache"),
        )
        updated_context = current_layer_context.model_copy(
            update={"retry_count": current_layer_context.retry_count + 1}
        )
        with ctx.state_write as st:
            st["parse_session"] = updated_session.model_dump(field_mode="backend", dump_format="json")
            st["current_layer_context"] = updated_context.model_dump(
                field_mode="backend",
                dump_format="json",
            )
            st["current_layer_review"] = reviewed.model_dump(field_mode="backend", dump_format="json")
        return _success("apply_cud_update")

    @_register_step(resolver, step_name="apply_cud_update", runtime_deps=runtime_deps)
    def _apply_cud_update(ctx: StepContext) -> StepRunResult:
        current_layer_result = CurrentLayerResult.model_validate(ctx.state_view["current_layer_result"])
        current_layer_review = CurrentLayerReview.model_validate(ctx.state_view["current_layer_review"])
        updated_result = apply_cud_update(
            current_layer_result=current_layer_result,
            current_layer_review=current_layer_review,
        )
        with ctx.state_write as st:
            st["current_layer_result"] = updated_result.model_dump(
                field_mode="backend",
                dump_format="json",
            )
        return _success("check_layer_coverage")

    @_register_step(resolver, step_name="check_layer_coverage", runtime_deps=runtime_deps)
    def _check_layer_coverage(ctx: StepContext) -> StepRunResult:
        current_layer_context = CurrentLayerContext.model_validate(ctx.state_view["current_layer_context"])
        current_layer_result = CurrentLayerResult.model_validate(ctx.state_view["current_layer_result"])
        current_layer_review = CurrentLayerReview.model_validate(ctx.state_view["current_layer_review"])
        coverage_ok, coverage_notes = check_layer_coverage(
            current_layer_context=current_layer_context,
            current_layer_result=current_layer_result,
            current_layer_review=current_layer_review,
        )
        merged_review = current_layer_review.model_copy(
            update={
                "coverage_ok": coverage_ok,
                "review_notes": list(current_layer_review.review_notes) + [
                    note
                    for note in coverage_notes
                    if note not in current_layer_review.review_notes
                ],
            }
        )
        with ctx.state_write as st:
            st["current_layer_review"] = merged_review.model_dump(
                field_mode="backend",
                dump_format="json",
            )
        return _success("check_layer_satisfaction")

    @_register_step(resolver, step_name="check_layer_satisfaction", runtime_deps=runtime_deps)
    def _check_layer_satisfaction(ctx: StepContext) -> StepRunResult:
        current_layer_context = CurrentLayerContext.model_validate(ctx.state_view["current_layer_context"])
        current_layer_result = CurrentLayerResult.model_validate(ctx.state_view["current_layer_result"])
        current_layer_review = CurrentLayerReview.model_validate(ctx.state_view["current_layer_review"])
        coverage_ok = (
            current_layer_review.coverage_ok is True
            and not current_layer_review.metadata.get("review_failure")
        )
        has_conflicts = bool(
            current_layer_review.overlap_conflicts
            or current_layer_review.coverage_gap_notes
            or current_layer_review.duplicate_child_notes
        )
        if current_layer_result.satisfied is False or not coverage_ok or has_conflicts:
            # Disable the failed operator for this parent layer. When triage
            # is enabled, even PageIndex may be retried through the remaining
            # policy-approved operators; deterministic routing still exhausts
            # the configured cascade in order.
            reasons = list(current_layer_review.review_notes)
            if current_layer_result.satisfied is False:
                reasons.append("layer marked unsatisfied")
            if not coverage_ok:
                reasons.append("layer coverage check failed")
            if has_conflicts:
                reasons.append(
                    f"layer has {len(current_layer_review.overlap_conflicts)} overlap conflicts, "
                    f"{len(current_layer_review.coverage_gap_notes)} coverage gaps, "
                    f"{len(current_layer_review.duplicate_child_notes)} duplicates"
                )
            strategy = str(current_layer_context.metadata.get("parse_strategy", "layer_excerpt"))
            raw_disabled = current_layer_context.metadata.get("disabled_strategies")
            disabled = {
                str(value)
                for value in (raw_disabled if isinstance(raw_disabled, list) else [])
            }
            disabled.add(strategy)
            updated_context = current_layer_context.model_copy(
                update={
                    "retry_count": current_layer_context.retry_count + 1,
                    "metadata": {
                        **current_layer_context.metadata,
                        "disabled_strategies": sorted(disabled),
                        "last_strategy_failure": strategy,
                        "last_strategy_failure_reasons": reasons,
                    },
                }
            )
            remaining = {"layer_excerpt", "layer_boundary", "page_index"} - disabled
            with ctx.state_write as st:
                st["current_layer_context"] = updated_context.model_dump(
                    field_mode="backend", dump_format="json"
                )
            event = StrategyExecutionRecord(
                strategy=strategy,  # type: ignore[arg-type]
                depth=current_layer_context.depth,
                parent_node_ids=list(current_layer_context.parent_node_ids),
                attempt=_strategy_attempt(ctx.state_view, strategy),
                event="failed",
                failure_type="retry",
                reasons=reasons[:12],
            )
            if remaining:
                return _success(
                    state_update=[("a", {"strategy_execution_history": event.model_dump(mode="json")})]
                )
            error_message = (
                f"layer satisfaction retries exhausted at depth {current_layer_context.depth}; "
                "all configured strategies are exhausted"
            )
            with ctx.state_write as st:
                st["workflow_errors"] = [error_message, *reasons]
            return _success(
                state_update=[("a", {"strategy_execution_history": event.model_dump(mode="json")})]
            )
        strategy = str(current_layer_context.metadata.get("parse_strategy", "layer_excerpt"))
        event = StrategyExecutionRecord(
            strategy=strategy,  # type: ignore[arg-type]
            depth=current_layer_context.depth,
            parent_node_ids=list(current_layer_context.parent_node_ids),
            attempt=_strategy_attempt(ctx.state_view, strategy),
            event="succeeded",
        )
        return _success(
            state_update=[("a", {"strategy_execution_history": event.model_dump(mode="json")})]
        )

    @_register_step(resolver, step_name="switch_split_strategy", runtime_deps=runtime_deps)
    def _switch_split_strategy(ctx: StepContext) -> StepRunResult:
        parse_session = ParseSessionState.model_validate(ctx.state_view["parse_session"])
        current_layer_context = CurrentLayerContext.model_validate(ctx.state_view["current_layer_context"])
        updated_session, updated_context = switch_split_strategy(
            parse_session=parse_session,
            current_layer_context=current_layer_context,
        )
        with ctx.state_write as st:
            st["parse_session"] = updated_session.model_dump(field_mode="backend", dump_format="json")
            st["current_layer_context"] = updated_context.model_dump(field_mode="backend", dump_format="json")
            st["current_layer_review"] = None
        return _success("propose_layer_breakdown")

    @_register_step(resolver, step_name="repair_layer_pointers", runtime_deps=runtime_deps)
    def _repair_layer_pointers(ctx: StepContext) -> StepRunResult:
        current_layer_result = CurrentLayerResult.model_validate(ctx.state_view["current_layer_result"])
        repaired, repaired_count = repair_layer_candidates(
            current_layer_result=current_layer_result,
            parser_source_map=_parser_source_map(ctx.state_view["parser_source_map"]),
            correct_pointer_fn=_correct_parser_pointer,
        )
        with ctx.state_write as st:
            st["current_layer_result"] = repaired.model_dump(field_mode="backend", dump_format="json")
            st["corrected_pointer_count"] = _state_int(
                ctx.state_view.get("corrected_pointer_count")
            ) + repaired_count
        return _success("dedupe_and_filter_layer")

    @_register_step(resolver, step_name="dedupe_and_filter_layer", runtime_deps=runtime_deps)
    def _dedupe_and_filter_layer(ctx: StepContext) -> StepRunResult:
        current_layer_context = CurrentLayerContext.model_validate(ctx.state_view["current_layer_context"])
        current_layer_result = CurrentLayerResult.model_validate(ctx.state_view["current_layer_result"])
        filtered = dedupe_and_filter_layer(
            current_layer_context=current_layer_context,
            current_layer_result=current_layer_result,
        )
        with ctx.state_write as st:
            st["current_layer_result"] = filtered.model_dump(field_mode="backend", dump_format="json")
        return _success("validate_layer_commit")

    @_register_step(resolver, step_name="validate_layer_commit", runtime_deps=runtime_deps)
    def _validate_layer_commit(ctx: StepContext) -> StepRunResult:
        current_layer_context = CurrentLayerContext.model_validate(ctx.state_view["current_layer_context"])
        current_layer_result = CurrentLayerResult.model_validate(ctx.state_view["current_layer_result"])
        review = validate_layer_commit(
            current_layer_context=current_layer_context,
            current_layer_result=current_layer_result,
            parser_source_map=_parser_source_map(ctx.state_view.get("parser_source_map")),
        )
        with ctx.state_write as st:
            if review.updated_result is not None:
                st["current_layer_result"] = review.updated_result.model_dump(
                    field_mode="backend",
                    dump_format="json",
                )
            st["current_layer_review"] = review.model_dump(
                field_mode="backend",
                dump_format="json",
            )
        return _success("commit_layer_children" if review.satisfied else "check_layer_satisfaction")

    @_register_step(resolver, step_name="commit_layer_children", runtime_deps=runtime_deps)
    def _commit_layer_children(ctx: StepContext) -> StepRunResult:
        semantic_tree = SemanticNode.model_validate(ctx.state_view["semantic_tree"])
        current_layer_context = CurrentLayerContext.model_validate(ctx.state_view["current_layer_context"])
        current_layer_result = CurrentLayerResult.model_validate(ctx.state_view["current_layer_result"])
        current_layer_review = CurrentLayerReview.model_validate(ctx.state_view["current_layer_review"])
        updated_tree = commit_layer_children(
            semantic_tree=semantic_tree,
            current_layer_result=current_layer_result,
            current_depth=current_layer_context.depth,
            parent_node_ids=current_layer_context.parent_node_ids,
        )
        with ctx.state_write as st:
            st["semantic_tree"] = updated_tree.model_dump()
            failed_parent_ids = list(current_layer_review.failed_parent_node_ids)
            if failed_parent_ids:
                queued = requeue_failed_frontier_items(
                    frontier_queue=[
                        LayerFrontierItem.model_validate(item)
                        for item in _state_list(ctx.state_view.get("layer_frontier_queue"))
                    ],
                    parent_node_ids=failed_parent_ids,
                    depth=current_layer_context.depth,
                )
                st["layer_frontier_queue"] = [
                    item.model_dump(field_mode="backend", dump_format="json")
                    for item in queued
                ]
        return _success("check_children_expandable")

    @_register_step(resolver, step_name="check_children_expandable", runtime_deps=runtime_deps)
    def _check_children_expandable(ctx: StepContext) -> StepRunResult:
        current_layer_result = CurrentLayerResult.model_validate(ctx.state_view["current_layer_result"])
        if any(child.expandable for child in current_layer_result.children):
            return _success("enqueue_next_layer_frontier")
        return _success("check_frontier_remaining")

    @_register_step(resolver, step_name="enqueue_next_layer_frontier", runtime_deps=runtime_deps)
    def _enqueue_next_layer_frontier(ctx: StepContext) -> StepRunResult:
        parse_session = ParseSessionState.model_validate(ctx.state_view["parse_session"])
        current_layer_context = CurrentLayerContext.model_validate(ctx.state_view["current_layer_context"])
        current_layer_result = CurrentLayerResult.model_validate(ctx.state_view["current_layer_result"])
        frontier_queue = [
            LayerFrontierItem.model_validate(item)
            for item in _state_list(ctx.state_view.get("layer_frontier_queue"))
        ]
        updated_queue = enqueue_next_layer_frontier(
            frontier_queue=frontier_queue,
            current_layer_context=current_layer_context,
            current_layer_result=current_layer_result,
            parse_session=parse_session,
        )
        with ctx.state_write as st:
            st["layer_frontier_queue"] = [
                item.model_dump(field_mode="backend", dump_format="json") for item in updated_queue
            ]
            st["current_layer_context"] = None
            st["current_layer_result"] = None
            st["current_layer_review"] = None
        return _success("check_frontier_remaining")

    @_register_step(resolver, step_name="finalize_semantic_tree", runtime_deps=runtime_deps)
    def _finalize_semantic_tree(ctx: StepContext) -> StepRunResult:
        tree = finalize_semantic_tree(
            SemanticNode.model_validate(ctx.state_view["semantic_tree"]),
            parser_source_map=_parser_source_map(ctx.state_view.get("parser_source_map")),
        )
        with ctx.state_write as st:
            st["semantic_tree"] = tree.model_dump()
        return _success("validate_tree")


def register_postparse_steps(
    resolver: MappingStepResolver,
    *,
    runtime_deps: WorkflowRuntimeDeps,
) -> None:
    @_register_step(resolver, step_name="validate_tree", runtime_deps=runtime_deps)
    def _validate_tree(ctx: StepContext) -> StepRunResult:
        tree = SemanticNode.model_validate(ctx.state_view["semantic_tree"])
        authoritative_source_map = _state_object_map(ctx.state_view["authoritative_source_map"])
        text_only_map = {
            unit_id: {"text": rec["parser_text"], "id": unit_id}
            for unit_id, rec in authoritative_source_map.items()
            if rec.get("participates_in_semantic_text", True)
        }
        coverage = compute_pointer_coverage(tree, text_only_map)
        terminal_coverage = compute_terminal_content_coverage(tree, text_only_map)
        report = ValidationReport(
            overall_text_coverage=_state_float(coverage.get("overall")),
            per_cluster_coverage=_coverage_map(coverage.get("per_cluster")),
            terminal_coverage=terminal_coverage,
            terminal_coverage_status=classify_terminal_coverage_status(tree, terminal_coverage),
            coverage_basis=str(terminal_coverage.get("coverage_basis", "legacy_union")),
            corrected_pointer_count=_state_int(ctx.state_view.get("corrected_pointer_count")),
            validation_notes=[],
        )
        bundle = _build_export_bundle(ctx=ctx, runtime_deps=runtime_deps)
        threshold = float(runtime_deps.get("coverage_threshold", 1.0))
        if not terminal_coverage.get("valid", False):
            overall = _state_float(terminal_coverage.get("overall"))
            ownership_message = (
                "terminal content ownership is incomplete or invalid: "
                f"overall={overall:.3f}"
            )
            # Preserve the older threshold diagnostic when incomplete
            # ownership also lowers coverage.  Callers can therefore
            # distinguish the quality failure without losing the stricter
            # terminal-ownership reason.
            error_message = (
                f"text coverage below threshold: {report.overall_text_coverage:.3f} < {threshold:.3f}; "
                f"{ownership_message}"
                if overall < threshold
                else ownership_message
            )
            with ctx.state_write as st:
                st["validation_report"] = report.model_dump(field_mode="backend", dump_format="json")
                st["export_bundle"] = bundle.model_dump(field_mode="backend", dump_format="json")
            return RunFailure(
                conversation_node_id=None,
                state_update=[],
                update={
                    "workflow_errors": [error_message],
                    "validation_report": report.model_dump(field_mode="backend", dump_format="json"),
                },
                errors=[error_message],
            )
        if _state_float(terminal_coverage.get("overall")) < threshold:
            error_message = (
                f"text coverage below threshold: {report.overall_text_coverage:.3f} < {threshold:.3f}"
            )
            with ctx.state_write as st:
                st["validation_report"] = report.model_dump(field_mode="backend", dump_format="json")
                st["export_bundle"] = bundle.model_dump(field_mode="backend", dump_format="json")
            return RunFailure(
                conversation_node_id=None,
                state_update=[],
                update={
                    "workflow_errors": [error_message],
                    "validation_report": report.model_dump(field_mode="backend", dump_format="json"),
                },
                errors=[error_message],
            )
        with ctx.state_write as st:
            st["validation_report"] = report.model_dump(field_mode="backend", dump_format="json")
        return _success("export_graph")

    @_register_step(resolver, step_name="export_graph", runtime_deps=runtime_deps)
    def _export_graph(ctx: StepContext) -> StepRunResult:
        bundle = _build_export_bundle(ctx=ctx, runtime_deps=runtime_deps)
        with ctx.state_write as st:
            st["export_bundle"] = bundle.model_dump(field_mode="backend", dump_format="json")
        return _success("persist_canonical_graph")

    @_register_step(resolver, step_name="persist_canonical_graph", runtime_deps=runtime_deps)
    def _persist_canonical_graph(ctx: StepContext) -> StepRunResult:
        bundle = WorkflowExportBundle.model_validate(ctx.state_view["export_bundle"])
        persistence_client = runtime_deps.get("graph_persistence_client")
        if persistence_client is None:
            return RunFailure(
                conversation_node_id=None,
                state_update=[],
                errors=["no graph persistence client configured"],
            )
        try:
            write_result = persistence_client.persist_graph_payload(bundle)
        except Exception as exc:  # noqa: BLE001 - persistence failures become workflow failures
            # Preserve the server-canonical authority contract on failure: a
            # failed remote write must not silently become a local-debug write.
            with ctx.state_write as st:
                st["export_bundle"] = bundle.model_dump(
                    field_mode="backend",
                    dump_format="json",
                )
            return RunFailure(
                conversation_node_id=None,
                state_update=[],
                errors=[f"canonical graph persistence failed: {exc}"],
            )
        if isinstance(write_result, dict):
            write_result = CanonicalGraphWriteResult.model_validate(write_result)
        emit_probe_event(
            runtime_deps.get("probe"),
            "workflow.persistence_result",
            persistence_mode=write_result.persistence_mode,
            kg_authority=write_result.kg_authority,
            canonical_write_confirmed=write_result.canonical_write_confirmed,
            nodes_written=write_result.nodes_written,
            edges_written=write_result.edges_written,
            transport=write_result.transport,
        )
        updated_bundle = bundle.model_copy(
            update={
                "persistence_mode": write_result.persistence_mode,
                "kg_authority": write_result.kg_authority,
                "canonical_write_confirmed": write_result.canonical_write_confirmed,
                "server_parser_used": write_result.server_parser_used,
                "canonical_write_result": write_result,
                "persisted_to_knowledge_engine": write_result.persistence_mode == "local_debug"
                and (write_result.nodes_written > 0 or write_result.edges_written > 0),
            }
        )
        with ctx.state_write as st:
            st["canonical_write_result"] = write_result.model_dump(
                field_mode="backend",
                dump_format="json",
            )
            st["export_bundle"] = updated_bundle.model_dump(
                field_mode="backend",
                dump_format="json",
            )
        return _success("end")

    @_register_step(resolver, step_name="end", runtime_deps=runtime_deps)
    def _end(ctx: StepContext) -> StepRunResult:
        return _success(None)

    @_register_step(resolver, step_name="parse_failure", runtime_deps=runtime_deps)
    def _parse_failure(ctx: StepContext) -> StepRunResult:
        errors = [str(value) for value in _state_list(ctx.state_view.get("workflow_errors"))]
        if not errors:
            errors = ["layer parsing failed after all strategies were exhausted"]
        return RunFailure(
            conversation_node_id=None,
            state_update=[],
            update={"workflow_errors": errors},
            errors=errors,
        )


def build_ingest_step_resolver(
    *,
    deps: Mapping[str, object] | None = None,
) -> MappingStepResolver:
    runtime_deps = cast(WorkflowRuntimeDeps, dict(deps or {}))
    resolver = MappingStepResolver()
    resolver.set_state_schema(
        {
            "normalized_input": "u",
            "authoritative_source_map": "u",
            "parser_input_dict": "u",
            "parser_source_map": "u",
            "parse_session": "u",
            "layer_frontier_queue": "u",
            "current_layer_context": "u",
            "current_layer_result": "u",
            "current_layer_review": "u",
            "semantic_tree": "u",
            "corrected_pointer_count": "u",
            "validation_report": "u",
            "export_bundle": "u",
            "canonical_write_result": "u",
            "workflow_errors": "a",
            "strategy_selection_error": "u",
            "strategy_attempt_counts": "u",
            "strategy_execution_history": "a",
        }
    )
    register_base_ingest_steps(resolver, runtime_deps=runtime_deps)
    register_layerwise_parser_steps(resolver, runtime_deps=runtime_deps)
    register_postparse_steps(resolver, runtime_deps=runtime_deps)
    return resolver
