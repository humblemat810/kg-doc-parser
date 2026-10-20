from __future__ import annotations

from typing import Literal, cast

from kogwistar.runtime import MappingStepResolver, StepContext
from kogwistar.runtime.models import (
    StepRunResult,
)

from .adapters import (
    select_primary_collection,
)
from .cache import workflow_cache_namespace
from .handler_state import (
    correct_parser_pointer as _correct_parser_pointer,
)
from .handler_state import (
    parser_payload as _parser_payload,
)
from .handler_state import (
    parser_source_map as _parser_source_map,
)
from .handler_state import (
    state_int as _state_int,
)
from .handler_state import (
    state_object_list as _state_object_list,
)
from .handler_state import (
    strategy_attempt as _strategy_attempt,
)
from .handler_strategy_steps import register_strategy_triage_step
from .handler_support import (
    register_step as _register_step,
)
from .handler_support import (
    success as _success,
)
from .model_validation import validate_backend_payload
from .models import (
    CurrentLayerContext,
    CurrentLayerResult,
    CurrentLayerReview,
    LayerFrontierItem,
    ParseSessionState,
    StrategyExecutionRecord,
    WorkflowIngestInput,
)
from .page_index import PageIndexSourceFormat, parse_page_index_layer
from .parser_core import (
    apply_cud_update,
    check_layer_coverage,
    commit_layer_children,
    dedupe_and_filter_layer,
    enqueue_next_layer_frontier,
    finalize_semantic_tree,
    prepare_layer_frontier,
    propose_layer_breakdown,
    repair_layer_candidates,
    requeue_failed_frontier_items,
    review_layer,
    switch_split_strategy,
    validate_layer_commit,
)
from .runtime_contracts import WorkflowRuntimeDeps
from .semantics import (
    SemanticNode,
)


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


def register_layerwise_parser_steps(
    resolver: MappingStepResolver,
    *,
    runtime_deps: WorkflowRuntimeDeps,
) -> None:
    register_strategy_triage_step(resolver, runtime_deps=runtime_deps)

    @_register_step(resolver, step_name="page_index_layer", runtime_deps=runtime_deps)
    def _page_index_layer(ctx: StepContext) -> StepRunResult:
        current_layer_context = validate_backend_payload(
            CurrentLayerContext, ctx.state_view["current_layer_context"]
        )
        normalized_input = validate_backend_payload(
            WorkflowIngestInput, ctx.state_view["normalized_input"]
        )
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
                    parent_pointers=current_layer_context.parent_content_pointers_by_id.get(
                        parent_id, []
                    ),
                    parser_source_map=parser_source_map,
                    source_format=source_format,
                    summary_enabled=bool(
                        current_layer_context.metadata.get(
                            "page_index_summary_enabled", True
                        )
                    ),
                )
            )
        result = CurrentLayerResult(
            children=candidates,
            satisfied=bool(candidates),
            atomic_retained=not bool(candidates),
            metadata={
                "parse_strategy": "page_index",
                "page_index_layer_only": True,
                "allow_empty_layer": not bool(candidates),
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

    @_register_step(
        resolver, step_name="check_frontier_remaining", runtime_deps=runtime_deps
    )
    def _check_frontier_remaining(ctx: StepContext) -> StepRunResult:
        queue = ctx.state_view.get("layer_frontier_queue") or []
        if queue:
            return _success("prepare_layer_frontier")
        return _success("finalize_semantic_tree")

    @_register_step(
        resolver, step_name="prepare_layer_frontier", runtime_deps=runtime_deps
    )
    def _prepare_layer_frontier(ctx: StepContext) -> StepRunResult:
        parse_session = validate_backend_payload(
            ParseSessionState, ctx.state_view["parse_session"]
        )
        semantic_tree = SemanticNode.model_validate(ctx.state_view["semantic_tree"])
        configured_batch_size = runtime_deps.get("layer_frontier_batch_size")
        if configured_batch_size is None:
            settings = runtime_deps.get("provider_settings")
            configured_batch_size = (
                settings.layer_frontier_batch_size if settings is not None else 1
            )
        raw_frontier = ctx.state_view.get("layer_frontier_queue")
        frontier_items = raw_frontier if isinstance(raw_frontier, list) else []
        context, remaining, updated_session = prepare_layer_frontier(
            parse_session=parse_session,
            frontier_queue=[
                validate_backend_payload(LayerFrontierItem, item)
                for item in frontier_items
            ],
            semantic_tree=semantic_tree,
            max_retries=int(runtime_deps.get("max_review_retries", 3)),
            max_items=(
                int(configured_batch_size)
                if configured_batch_size is not None
                else None
            ),
        )
        with ctx.state_write as st:
            st["parse_session"] = updated_session.model_dump(
                field_mode="backend", dump_format="json"
            )
            st["current_layer_context"] = context.model_dump(
                field_mode="backend", dump_format="json"
            )
            st["layer_frontier_queue"] = [
                item.model_dump(field_mode="backend", dump_format="json")
                for item in remaining
            ]
        return _success("triage_parse_strategy")

    @_register_step(
        resolver, step_name="propose_layer_breakdown", runtime_deps=runtime_deps
    )
    def _propose_layer_breakdown(ctx: StepContext) -> StepRunResult:
        normalized = validate_backend_payload(
            WorkflowIngestInput, ctx.state_view["normalized_input"]
        )
        collection = select_primary_collection(normalized)
        parse_session = validate_backend_payload(
            ParseSessionState, ctx.state_view["parse_session"]
        )
        current_layer_context = validate_backend_payload(
            CurrentLayerContext, ctx.state_view["current_layer_context"]
        )
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
            cache_namespace=workflow_cache_namespace(
                runtime_deps.get("provider_settings")
            ),
        )
        with ctx.state_write as st:
            st["current_layer_result"] = result.model_dump(
                field_mode="backend", dump_format="json"
            )
        return _success()

    @_register_step(
        resolver, step_name="review_cud_proposal", runtime_deps=runtime_deps
    )
    def _review_cud_proposal(ctx: StepContext) -> StepRunResult:
        parse_session = validate_backend_payload(
            ParseSessionState, ctx.state_view["parse_session"]
        )
        current_layer_context = validate_backend_payload(
            CurrentLayerContext, ctx.state_view["current_layer_context"]
        )
        current_layer_result = validate_backend_payload(
            CurrentLayerResult, ctx.state_view["current_layer_result"]
        )
        reviewed, updated_session = review_layer(
            parse_session=parse_session,
            current_layer_context=current_layer_context,
            current_layer_result=current_layer_result,
            parser_source_map=_parser_source_map(ctx.state_view["parser_source_map"]),
            review_layer_fn=runtime_deps.get("review_layer_fn"),
            llm_cache=runtime_deps.get("llm_cache"),
            cache_namespace=workflow_cache_namespace(
                runtime_deps.get("provider_settings")
            ),
        )
        updated_context = current_layer_context.model_copy(
            update={"retry_count": current_layer_context.retry_count + 1}
        )
        with ctx.state_write as st:
            st["parse_session"] = updated_session.model_dump(
                field_mode="backend", dump_format="json"
            )
            st["current_layer_context"] = updated_context.model_dump(
                field_mode="backend",
                dump_format="json",
            )
            st["current_layer_review"] = reviewed.model_dump(
                field_mode="backend", dump_format="json"
            )
        return _success("apply_cud_update")

    @_register_step(resolver, step_name="apply_cud_update", runtime_deps=runtime_deps)
    def _apply_cud_update(ctx: StepContext) -> StepRunResult:
        current_layer_result = validate_backend_payload(
            CurrentLayerResult, ctx.state_view["current_layer_result"]
        )
        current_layer_review = validate_backend_payload(
            CurrentLayerReview, ctx.state_view["current_layer_review"]
        )
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

    @_register_step(
        resolver, step_name="check_layer_coverage", runtime_deps=runtime_deps
    )
    def _check_layer_coverage(ctx: StepContext) -> StepRunResult:
        current_layer_context = validate_backend_payload(
            CurrentLayerContext, ctx.state_view["current_layer_context"]
        )
        current_layer_result = validate_backend_payload(
            CurrentLayerResult, ctx.state_view["current_layer_result"]
        )
        current_layer_review = validate_backend_payload(
            CurrentLayerReview, ctx.state_view["current_layer_review"]
        )
        coverage_ok, coverage_notes = check_layer_coverage(
            current_layer_context=current_layer_context,
            current_layer_result=current_layer_result,
            current_layer_review=current_layer_review,
        )
        merged_review = current_layer_review.model_copy(
            update={
                "coverage_ok": coverage_ok,
                "review_notes": list(current_layer_review.review_notes)
                + [
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

    @_register_step(
        resolver, step_name="check_layer_satisfaction", runtime_deps=runtime_deps
    )
    def _check_layer_satisfaction(ctx: StepContext) -> StepRunResult:
        current_layer_context = validate_backend_payload(
            CurrentLayerContext, ctx.state_view["current_layer_context"]
        )
        current_layer_result = validate_backend_payload(
            CurrentLayerResult, ctx.state_view["current_layer_result"]
        )
        current_layer_review = validate_backend_payload(
            CurrentLayerReview, ctx.state_view["current_layer_review"]
        )
        coverage_ok = (
            current_layer_review.coverage_ok is True
            and not current_layer_review.metadata.get("review_failure")
        )
        has_conflicts = bool(
            current_layer_review.overlap_conflicts
            or current_layer_review.coverage_gap_notes
            or current_layer_review.duplicate_child_notes
            or current_layer_review.structural_issues
            or current_layer_review.validation_diagnostics
        )
        # A mixed batch may contain valid refinements for some parents and a
        # rejected refinement for others.  Do not discard the valid subset or
        # let the provider's aggregate verdict hide the failed parents.  Run
        # the deterministic commit gate here so the next steps can commit the
        # validated subset and requeue only the failed parents.
        if not current_layer_review.metadata.get("review_failure") and (
            current_layer_result.satisfied is False or not coverage_ok or has_conflicts
        ):
            partial_review = validate_layer_commit(
                current_layer_context=current_layer_context,
                current_layer_result=current_layer_result,
                parser_source_map=_parser_source_map(
                    ctx.state_view.get("parser_source_map")
                ),
            )
            if (
                partial_review.committable_parent_node_ids
                and partial_review.failed_parent_node_ids
            ):
                with ctx.state_write as st:
                    st["current_layer_result"] = (
                        partial_review.updated_result or current_layer_result
                    ).model_dump(field_mode="backend", dump_format="json")
                    st["current_layer_review"] = partial_review.model_dump(
                        field_mode="backend", dump_format="json"
                    )
                return _success("repair_layer_pointers")
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
                    f"{len(current_layer_review.duplicate_child_notes)} duplicates, "
                    f"{len(current_layer_review.structural_issues)} structural issues, "
                    f"{len(current_layer_review.validation_diagnostics)} validation diagnostics"
                )
            reasons.extend(
                f"{diagnostic.phase} validation: {diagnostic.message}"
                for diagnostic in current_layer_review.validation_diagnostics
            )
            strategy = str(
                current_layer_context.metadata.get("parse_strategy", "layer_excerpt")
            )
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
                    state_update=[
                        (
                            "a",
                            {
                                "strategy_execution_history": event.model_dump(
                                    mode="json"
                                )
                            },
                        )
                    ]
                )
            error_message = (
                f"layer satisfaction retries exhausted at depth {current_layer_context.depth}; "
                "all configured strategies are exhausted"
            )
            with ctx.state_write as st:
                st["workflow_errors"] = [error_message, *reasons]
            return _success(
                state_update=[
                    ("a", {"strategy_execution_history": event.model_dump(mode="json")})
                ]
            )
        strategy = str(
            current_layer_context.metadata.get("parse_strategy", "layer_excerpt")
        )
        event = StrategyExecutionRecord(
            strategy=strategy,  # type: ignore[arg-type]
            depth=current_layer_context.depth,
            parent_node_ids=list(current_layer_context.parent_node_ids),
            attempt=_strategy_attempt(ctx.state_view, strategy),
            event="succeeded",
        )
        return _success(
            state_update=[
                ("a", {"strategy_execution_history": event.model_dump(mode="json")})
            ]
        )

    @_register_step(
        resolver, step_name="switch_split_strategy", runtime_deps=runtime_deps
    )
    def _switch_split_strategy(ctx: StepContext) -> StepRunResult:
        parse_session = validate_backend_payload(
            ParseSessionState, ctx.state_view["parse_session"]
        )
        current_layer_context = validate_backend_payload(
            CurrentLayerContext, ctx.state_view["current_layer_context"]
        )
        updated_session, updated_context = switch_split_strategy(
            parse_session=parse_session,
            current_layer_context=current_layer_context,
        )
        with ctx.state_write as st:
            st["parse_session"] = updated_session.model_dump(
                field_mode="backend", dump_format="json"
            )
            st["current_layer_context"] = updated_context.model_dump(
                field_mode="backend", dump_format="json"
            )
            st["current_layer_review"] = None
        return _success("propose_layer_breakdown")

    @_register_step(
        resolver, step_name="repair_layer_pointers", runtime_deps=runtime_deps
    )
    def _repair_layer_pointers(ctx: StepContext) -> StepRunResult:
        current_layer_result = validate_backend_payload(
            CurrentLayerResult, ctx.state_view["current_layer_result"]
        )
        repaired, repaired_count = repair_layer_candidates(
            current_layer_result=current_layer_result,
            parser_source_map=_parser_source_map(ctx.state_view["parser_source_map"]),
            correct_pointer_fn=_correct_parser_pointer,
        )
        with ctx.state_write as st:
            st["current_layer_result"] = repaired.model_dump(
                field_mode="backend", dump_format="json"
            )
            st["corrected_pointer_count"] = (
                _state_int(ctx.state_view.get("corrected_pointer_count"))
                + repaired_count
            )
        return _success("dedupe_and_filter_layer")

    @_register_step(
        resolver, step_name="dedupe_and_filter_layer", runtime_deps=runtime_deps
    )
    def _dedupe_and_filter_layer(ctx: StepContext) -> StepRunResult:
        current_layer_context = validate_backend_payload(
            CurrentLayerContext, ctx.state_view["current_layer_context"]
        )
        current_layer_result = validate_backend_payload(
            CurrentLayerResult, ctx.state_view["current_layer_result"]
        )
        filtered = dedupe_and_filter_layer(
            current_layer_context=current_layer_context,
            current_layer_result=current_layer_result,
        )
        with ctx.state_write as st:
            st["current_layer_result"] = filtered.model_dump(
                field_mode="backend", dump_format="json"
            )
        return _success("validate_layer_commit")

    @_register_step(
        resolver, step_name="validate_layer_commit", runtime_deps=runtime_deps
    )
    def _validate_layer_commit(ctx: StepContext) -> StepRunResult:
        current_layer_context = validate_backend_payload(
            CurrentLayerContext, ctx.state_view["current_layer_context"]
        )
        current_layer_result = validate_backend_payload(
            CurrentLayerResult, ctx.state_view["current_layer_result"]
        )
        review = validate_layer_commit(
            current_layer_context=current_layer_context,
            current_layer_result=current_layer_result,
            parser_source_map=_parser_source_map(
                ctx.state_view.get("parser_source_map")
            ),
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
        return _success(
            "commit_layer_children" if review.satisfied else "check_layer_satisfaction"
        )

    @_register_step(
        resolver, step_name="commit_layer_children", runtime_deps=runtime_deps
    )
    def _commit_layer_children(ctx: StepContext) -> StepRunResult:
        semantic_tree = SemanticNode.model_validate(ctx.state_view["semantic_tree"])
        current_layer_context = validate_backend_payload(
            CurrentLayerContext, ctx.state_view["current_layer_context"]
        )
        current_layer_result = validate_backend_payload(
            CurrentLayerResult, ctx.state_view["current_layer_result"]
        )
        current_layer_review = validate_backend_payload(
            CurrentLayerReview, ctx.state_view["current_layer_review"]
        )
        parse_session = validate_backend_payload(
            ParseSessionState, ctx.state_view["parse_session"]
        )
        updated_tree = commit_layer_children(
            semantic_tree=semantic_tree,
            current_layer_result=current_layer_result,
            current_depth=current_layer_context.depth,
            parent_node_ids=current_layer_review.committable_parent_node_ids,
            parser_source_map=_parser_source_map(
                ctx.state_view.get("parser_source_map")
            ),
            max_depth=parse_session.max_depth,
        )
        with ctx.state_write as st:
            st["semantic_tree"] = updated_tree.model_dump()
            failed_parent_ids = list(current_layer_review.failed_parent_node_ids)
            if failed_parent_ids:
                queued = requeue_failed_frontier_items(
                    frontier_queue=[
                        validate_backend_payload(LayerFrontierItem, item)
                        for item in _state_object_list(
                            ctx.state_view.get("layer_frontier_queue")
                        )
                    ],
                    parent_node_ids=failed_parent_ids,
                    depth=current_layer_context.depth,
                )
                st["layer_frontier_queue"] = [
                    item.model_dump(field_mode="backend", dump_format="json")
                    for item in queued
                ]
        return _success("check_children_expandable")

    @_register_step(
        resolver, step_name="check_children_expandable", runtime_deps=runtime_deps
    )
    def _check_children_expandable(ctx: StepContext) -> StepRunResult:
        current_layer_result = validate_backend_payload(
            CurrentLayerResult, ctx.state_view["current_layer_result"]
        )
        if any(child.expandable for child in current_layer_result.children):
            return _success("enqueue_next_layer_frontier")
        return _success("check_frontier_remaining")

    @_register_step(
        resolver, step_name="enqueue_next_layer_frontier", runtime_deps=runtime_deps
    )
    def _enqueue_next_layer_frontier(ctx: StepContext) -> StepRunResult:
        parse_session = validate_backend_payload(
            ParseSessionState, ctx.state_view["parse_session"]
        )
        current_layer_context = validate_backend_payload(
            CurrentLayerContext, ctx.state_view["current_layer_context"]
        )
        current_layer_result = validate_backend_payload(
            CurrentLayerResult, ctx.state_view["current_layer_result"]
        )
        frontier_queue = [
            validate_backend_payload(LayerFrontierItem, item)
            for item in _state_object_list(ctx.state_view.get("layer_frontier_queue"))
        ]
        updated_queue = enqueue_next_layer_frontier(
            frontier_queue=frontier_queue,
            current_layer_context=current_layer_context,
            current_layer_result=current_layer_result,
            parse_session=parse_session,
        )
        with ctx.state_write as st:
            st["layer_frontier_queue"] = [
                item.model_dump(field_mode="backend", dump_format="json")
                for item in updated_queue
            ]
            st["current_layer_context"] = None
            st["current_layer_result"] = None
            st["current_layer_review"] = None
        return _success("check_frontier_remaining")

    @_register_step(
        resolver, step_name="finalize_semantic_tree", runtime_deps=runtime_deps
    )
    def _finalize_semantic_tree(ctx: StepContext) -> StepRunResult:
        parse_session = validate_backend_payload(
            ParseSessionState, ctx.state_view["parse_session"]
        )
        tree = finalize_semantic_tree(
            SemanticNode.model_validate(ctx.state_view["semantic_tree"]),
            parser_source_map=_parser_source_map(
                ctx.state_view.get("parser_source_map")
            ),
            max_depth=parse_session.max_depth,
        )
        with ctx.state_write as st:
            st["semantic_tree"] = tree.model_dump()
        return _success("validate_tree")
