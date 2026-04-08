from __future__ import annotations

from typing import Any

from kogwistar.runtime import MappingStepResolver
from kogwistar.runtime.models import RunFailure, RunSuccess, RunSuspended

from .adapters import (
    build_authoritative_source_map,
    build_parser_input_dict,
    build_parser_source_map,
    select_primary_collection,
)
from .models import (
    CanonicalGraphWriteResult,
    CurrentLayerContext,
    CurrentLayerReview,
    CurrentLayerResult,
    LayerFrontierItem,
    ParseSessionState,
    ValidationReport,
    WorkflowExportBundle,
    WorkflowIngestInput,
)
from .parser_core import (
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
    review_layer,
)
from .probe import emit_probe_event
from .semantics import (
    SemanticNode,
    compute_pointer_coverage,
    correct_and_validate_pointer,
    semantic_tree_to_kge_payload,
)


def _success(next_step: str | None = None) -> RunSuccess:
    return RunSuccess(
        conversation_node_id=None,
        state_update=[],
        _route_next=[] if next_step is None else [next_step],
    )


def _probe_snapshot(state_view: dict[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "state_keys": sorted(k for k, v in state_view.items() if v is not None),
    }
    frontier = state_view.get("layer_frontier_queue")
    if isinstance(frontier, list):
        snapshot["frontier_size"] = len(frontier)
    current_layer_context = state_view.get("current_layer_context")
    if isinstance(current_layer_context, dict):
        snapshot["current_depth"] = current_layer_context.get("depth")
        snapshot["retry_count"] = current_layer_context.get("retry_count")
    parse_session = state_view.get("parse_session")
    if isinstance(parse_session, dict):
        snapshot["parse_mode"] = parse_session.get("mode")
        snapshot["session_depth"] = parse_session.get("current_depth")
    return snapshot


def _register_step(
    resolver: MappingStepResolver,
    *,
    step_name: str,
    runtime_deps: dict[str, Any],
):
    probe = runtime_deps.get("probe")

    def decorator(fn):
        @resolver.register(step_name)
        def _wrapped(ctx):
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

        return _wrapped

    return decorator


def register_base_ingest_steps(resolver: MappingStepResolver, *, runtime_deps: dict[str, Any]) -> None:
    @_register_step(resolver, step_name="start", runtime_deps=runtime_deps)
    def _start(ctx):
        return _success("normalize_input")

    @_register_step(resolver, step_name="normalize_input", runtime_deps=runtime_deps)
    def _normalize_input(ctx):
        payload = WorkflowIngestInput.model_validate(ctx.state_view["input"]).model_dump(
            field_mode="backend",
            dump_format="json",
        )
        with ctx.state_write as st:
            st["normalized_input"] = payload
        return _success("build_source_map")

    @_register_step(resolver, step_name="build_source_map", runtime_deps=runtime_deps)
    def _build_source_map(ctx):
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
    def _init_parse_session(ctx):
        normalized = WorkflowIngestInput.model_validate(ctx.state_view["normalized_input"])
        collection = select_primary_collection(normalized)
        parse_semantic_fn = runtime_deps.get("parse_semantic_fn", default_parse_semantic_fn)
        propose_layer_fn = runtime_deps.get("propose_layer_fn")
        session, frontier, root = initialize_parse_session(
            collection=collection,
            parser_input_dict=ctx.state_view["parser_input_dict"],
            parser_source_map=ctx.state_view["parser_source_map"],
            max_depth=int(runtime_deps.get("max_depth", 10)),
            allow_review=bool(runtime_deps.get("allow_review", True)),
            parse_semantic_fn=parse_semantic_fn if propose_layer_fn is None else None,
        )
        with ctx.state_write as st:
            st["parse_session"] = session.model_dump(field_mode="backend", dump_format="json")
            st["layer_frontier_queue"] = [
                item.model_dump(field_mode="backend", dump_format="json") for item in frontier
            ]
            st["semantic_tree"] = root.model_dump(mode="json")
        return _success("check_frontier_remaining")


def register_layerwise_parser_steps(resolver: MappingStepResolver, *, runtime_deps: dict[str, Any]) -> None:
    @_register_step(resolver, step_name="check_frontier_remaining", runtime_deps=runtime_deps)
    def _check_frontier_remaining(ctx):
        queue = ctx.state_view.get("layer_frontier_queue") or []
        if queue:
            return _success("prepare_layer_frontier")
        return _success("finalize_semantic_tree")

    @_register_step(resolver, step_name="prepare_layer_frontier", runtime_deps=runtime_deps)
    def _prepare_layer_frontier(ctx):
        parse_session = ParseSessionState.model_validate(ctx.state_view["parse_session"])
        semantic_tree = SemanticNode.model_validate(ctx.state_view["semantic_tree"])
        context, remaining, updated_session = prepare_layer_frontier(
            parse_session=parse_session,
            frontier_queue=[
                LayerFrontierItem.model_validate(item)
                for item in ctx.state_view["layer_frontier_queue"]
            ],
            semantic_tree=semantic_tree,
            max_retries=int(runtime_deps.get("max_review_retries", 3)),
        )
        with ctx.state_write as st:
            st["parse_session"] = updated_session.model_dump(field_mode="backend", dump_format="json")
            st["current_layer_context"] = context.model_dump(field_mode="backend", dump_format="json")
            st["layer_frontier_queue"] = [
                item.model_dump(field_mode="backend", dump_format="json") for item in remaining
            ]
        return _success("propose_layer_breakdown")

    @_register_step(resolver, step_name="propose_layer_breakdown", runtime_deps=runtime_deps)
    def _propose_layer_breakdown(ctx):
        normalized = WorkflowIngestInput.model_validate(ctx.state_view["normalized_input"])
        collection = select_primary_collection(normalized)
        parse_session = ParseSessionState.model_validate(ctx.state_view["parse_session"])
        current_layer_context = CurrentLayerContext.model_validate(ctx.state_view["current_layer_context"])
        semantic_tree = SemanticNode.model_validate(ctx.state_view["semantic_tree"])
        result = propose_layer_breakdown(
            collection=collection,
            parser_input_dict=ctx.state_view["parser_input_dict"],
            parser_source_map=ctx.state_view["parser_source_map"],
            parse_session=parse_session,
            current_layer_context=current_layer_context,
            semantic_tree=semantic_tree,
            propose_layer_fn=runtime_deps.get("propose_layer_fn"),
            llm_cache=runtime_deps.get("llm_cache"),
        )
        with ctx.state_write as st:
            st["current_layer_result"] = result.model_dump(field_mode="backend", dump_format="json")
        return _success("review_cud_proposal")

    @_register_step(resolver, step_name="review_cud_proposal", runtime_deps=runtime_deps)
    def _review_cud_proposal(ctx):
        parse_session = ParseSessionState.model_validate(ctx.state_view["parse_session"])
        current_layer_context = CurrentLayerContext.model_validate(ctx.state_view["current_layer_context"])
        current_layer_result = CurrentLayerResult.model_validate(ctx.state_view["current_layer_result"])
        reviewed, updated_session = review_layer(
            parse_session=parse_session,
            current_layer_context=current_layer_context,
            current_layer_result=current_layer_result,
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
    def _apply_cud_update(ctx):
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
    def _check_layer_coverage(ctx):
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
    def _check_layer_satisfaction(ctx):
        current_layer_context = CurrentLayerContext.model_validate(ctx.state_view["current_layer_context"])
        current_layer_result = CurrentLayerResult.model_validate(ctx.state_view["current_layer_result"])
        current_layer_review = CurrentLayerReview.model_validate(ctx.state_view["current_layer_review"])
        coverage_ok = current_layer_review.coverage_ok is not False
        if current_layer_result.satisfied is False or not coverage_ok:
            if current_layer_context.retry_count >= current_layer_context.max_retries:
                reasons = list(current_layer_review.review_notes)
                if current_layer_result.satisfied is False:
                    reasons.append("layer marked unsatisfied")
                if not coverage_ok:
                    reasons.append("layer coverage check failed")
                error_message = (
                    f"layer satisfaction retries exhausted at depth {current_layer_context.depth}"
                )
                return RunFailure(
                    conversation_node_id=None,
                    state_update=[],
                    update={"workflow_errors": [error_message, *reasons]},
                errors=[error_message],
            )
            return _success("propose_layer_breakdown")
        return _success("repair_layer_pointers")

    @_register_step(resolver, step_name="repair_layer_pointers", runtime_deps=runtime_deps)
    def _repair_layer_pointers(ctx):
        current_layer_result = CurrentLayerResult.model_validate(ctx.state_view["current_layer_result"])
        repaired, repaired_count = repair_layer_candidates(
            current_layer_result=current_layer_result,
            parser_source_map=ctx.state_view["parser_source_map"],
            correct_pointer_fn=correct_and_validate_pointer,
        )
        with ctx.state_write as st:
            st["current_layer_result"] = repaired.model_dump(field_mode="backend", dump_format="json")
            st["corrected_pointer_count"] = int(ctx.state_view.get("corrected_pointer_count", 0)) + repaired_count
        return _success("dedupe_and_filter_layer")

    @_register_step(resolver, step_name="dedupe_and_filter_layer", runtime_deps=runtime_deps)
    def _dedupe_and_filter_layer(ctx):
        current_layer_context = CurrentLayerContext.model_validate(ctx.state_view["current_layer_context"])
        current_layer_result = CurrentLayerResult.model_validate(ctx.state_view["current_layer_result"])
        filtered = dedupe_and_filter_layer(
            current_layer_context=current_layer_context,
            current_layer_result=current_layer_result,
        )
        with ctx.state_write as st:
            st["current_layer_result"] = filtered.model_dump(field_mode="backend", dump_format="json")
        return _success("commit_layer_children")

    @_register_step(resolver, step_name="commit_layer_children", runtime_deps=runtime_deps)
    def _commit_layer_children(ctx):
        semantic_tree = SemanticNode.model_validate(ctx.state_view["semantic_tree"])
        current_layer_context = CurrentLayerContext.model_validate(ctx.state_view["current_layer_context"])
        current_layer_result = CurrentLayerResult.model_validate(ctx.state_view["current_layer_result"])
        updated_tree = commit_layer_children(
            semantic_tree=semantic_tree,
            current_layer_result=current_layer_result,
            current_depth=current_layer_context.depth,
        )
        with ctx.state_write as st:
            st["semantic_tree"] = updated_tree.model_dump(mode="json")
        return _success("check_children_expandable")

    @_register_step(resolver, step_name="check_children_expandable", runtime_deps=runtime_deps)
    def _check_children_expandable(ctx):
        current_layer_result = CurrentLayerResult.model_validate(ctx.state_view["current_layer_result"])
        if any(child.expandable for child in current_layer_result.children):
            return _success("enqueue_next_layer_frontier")
        return _success("check_frontier_remaining")

    @_register_step(resolver, step_name="enqueue_next_layer_frontier", runtime_deps=runtime_deps)
    def _enqueue_next_layer_frontier(ctx):
        parse_session = ParseSessionState.model_validate(ctx.state_view["parse_session"])
        current_layer_context = CurrentLayerContext.model_validate(ctx.state_view["current_layer_context"])
        current_layer_result = CurrentLayerResult.model_validate(ctx.state_view["current_layer_result"])
        frontier_queue = [
            LayerFrontierItem.model_validate(item)
            for item in ctx.state_view.get("layer_frontier_queue", [])
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
    def _finalize_semantic_tree(ctx):
        tree = finalize_semantic_tree(SemanticNode.model_validate(ctx.state_view["semantic_tree"]))
        with ctx.state_write as st:
            st["semantic_tree"] = tree.model_dump(mode="json")
        return _success("validate_tree")


def register_postparse_steps(resolver: MappingStepResolver, *, runtime_deps: dict[str, Any]) -> None:
    @_register_step(resolver, step_name="validate_tree", runtime_deps=runtime_deps)
    def _validate_tree(ctx):
        tree = SemanticNode.model_validate(ctx.state_view["semantic_tree"])
        authoritative_source_map = ctx.state_view["authoritative_source_map"]
        text_only_map = {
            unit_id: {"text": rec["parser_text"], "id": unit_id}
            for unit_id, rec in authoritative_source_map.items()
            if rec.get("participates_in_semantic_text", True)
        }
        coverage = compute_pointer_coverage(tree, text_only_map)
        report = ValidationReport(
            overall_text_coverage=coverage["overall"],
            per_cluster_coverage=coverage["per_cluster"],
            corrected_pointer_count=int(ctx.state_view.get("corrected_pointer_count", 0)),
            validation_notes=[],
        )
        threshold = float(runtime_deps.get("coverage_threshold", 0.99))
        if report.overall_text_coverage < threshold:
            error_message = (
                f"text coverage below threshold: {report.overall_text_coverage:.3f} < {threshold:.3f}"
            )
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
    def _export_graph(ctx):
        normalized = WorkflowIngestInput.model_validate(ctx.state_view["normalized_input"])
        collection = select_primary_collection(normalized)
        tree = SemanticNode.model_validate(ctx.state_view["semantic_tree"])
        graph_payload = semantic_tree_to_kge_payload(tree, doc_id=collection.collection_id)
        persistence_mode = str(runtime_deps.get("persistence_mode", "local_debug"))
        kg_authority = str(runtime_deps.get("kg_authority", "local"))
        bundle = WorkflowExportBundle(
            graph_payload=graph_payload,
            authoritative_source_map=ctx.state_view["authoritative_source_map"],
            embedding_spaces=collection.embedding_spaces,
            consolidation_candidates=[],
            retrieval_metadata={
                "embedding_spaces": collection.embedding_spaces,
                "supports_hybrid_retrieval": True,
                "supports_split_embedding_spaces": True,
                "collection_modality": collection.modality,
            },
            persistence_mode="server_canonical" if persistence_mode == "server_canonical" else "local_debug",
            kg_authority="server" if kg_authority == "server" else "local",
            canonical_write_confirmed=False,
            parser_owner="local",
            server_parser_used=False,
            persisted_to_knowledge_engine=False,
        )
        with ctx.state_write as st:
            st["export_bundle"] = bundle.model_dump(field_mode="backend", dump_format="json")
        return _success("persist_canonical_graph")

    @_register_step(resolver, step_name="persist_canonical_graph", runtime_deps=runtime_deps)
    def _persist_canonical_graph(ctx):
        bundle = WorkflowExportBundle.model_validate(ctx.state_view["export_bundle"])
        persistence_client = runtime_deps.get("graph_persistence_client")
        if persistence_client is None:
            return RunFailure(
                conversation_node_id=None,
                state_update=[],
                errors=["no graph persistence client configured"],
            )
        write_result = persistence_client.persist_graph_payload(bundle)
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
    def _end(ctx):
        return _success(None)


def build_ingest_step_resolver(*, deps: dict[str, Any] | None = None) -> MappingStepResolver:
    runtime_deps = dict(deps or {})
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
        }
    )
    register_base_ingest_steps(resolver, runtime_deps=runtime_deps)
    register_layerwise_parser_steps(resolver, runtime_deps=runtime_deps)
    register_postparse_steps(resolver, runtime_deps=runtime_deps)
    return resolver
