from __future__ import annotations

from typing import Any, Callable

from kogwistar.runtime import MappingStepResolver
from kogwistar.runtime.models import RunFailure, RunSuccess

from .adapters import (
    build_authoritative_source_map,
    build_parser_input_dict,
    build_parser_source_map,
    select_primary_collection,
)
from .models import ValidationReport, WorkflowExportBundle, WorkflowIngestInput
from .models import CanonicalGraphWriteResult
from .semantics import (
    SemanticNode,
    compute_pointer_coverage,
    correct_and_validate_pointer,
    semantic_tree_to_kge_payload,
)


def default_parse_semantic_fn(
    *,
    collection,
    parser_input_dict: dict[str, Any],
    parser_source_map: dict[str, dict[str, Any]],
):
    from ..semantic_document_splitting_layerwise_edits import build_document_tree

    return build_document_tree(
        doc_id=collection.collection_id,
        llm_input_dict=parser_input_dict,
        source_map=parser_source_map,
    )


def _repair_tree_pointers(
    root: SemanticNode, source_map: dict[str, dict[str, Any]]
) -> tuple[SemanticNode, int]:
    repaired_count = 0

    def walk(node: SemanticNode) -> SemanticNode:
        nonlocal repaired_count
        repaired_ptrs = []
        for ptr in node.total_content_pointers:
            fixed = correct_and_validate_pointer(ptr, source_map)
            if fixed is None:
                raise ValueError(f"unrecoverable pointer for node {node.title!r}")
            if fixed.model_dump() != ptr.model_dump():
                repaired_count += 1
            repaired_ptrs.append(fixed)
        repaired_children = [walk(child) for child in node.child_nodes]
        payload = node.model_dump(mode="json")
        payload["total_content_pointers"] = [p.model_dump(mode="json") for p in repaired_ptrs]
        payload["child_nodes"] = [c.model_dump(mode="json") for c in repaired_children]
        return SemanticNode.model_validate(payload)

    return walk(root), repaired_count


def build_ingest_step_resolver(*, deps: dict[str, Any] | None = None) -> MappingStepResolver:
    runtime_deps = dict(deps or {})
    resolver = MappingStepResolver()
    resolver.set_state_schema(
        {
            "normalized_input": "u",
            "authoritative_source_map": "u",
            "parser_input_dict": "u",
            "parser_source_map": "u",
            "semantic_tree": "u",
            "validation_report": "u",
            "export_bundle": "u",
            "canonical_write_result": "u",
            "workflow_errors": "a",
        }
    )

    @resolver.register("start")
    def _start(ctx):
        return RunSuccess(conversation_node_id=None, state_update=[], _route_next=["normalize_input"])

    @resolver.register("normalize_input")
    def _normalize_input(ctx):
        payload = WorkflowIngestInput.model_validate(ctx.state_view["input"]).model_dump(
            field_mode="backend",
            dump_format="json",
        )
        with ctx.state_write as st:
            st["normalized_input"] = payload
        return RunSuccess(conversation_node_id=None, state_update=[], _route_next=["build_source_map"])

    @resolver.register("build_source_map")
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
        return RunSuccess(conversation_node_id=None, state_update=[], _route_next=["parse_semantic"])

    @resolver.register("parse_semantic")
    def _parse_semantic(ctx):
        normalized = WorkflowIngestInput.model_validate(ctx.state_view["normalized_input"])
        collection = select_primary_collection(normalized)
        parse_semantic_fn: Callable[..., Any] = runtime_deps.get(
            "parse_semantic_fn",
            default_parse_semantic_fn,
        )
        tree = parse_semantic_fn(
            collection=collection,
            parser_input_dict=ctx.state_view["parser_input_dict"],
            parser_source_map=ctx.state_view["parser_source_map"],
        )
        if isinstance(tree, tuple):
            tree = tree[0]
        if isinstance(tree, dict):
            tree = SemanticNode.model_validate(tree)
        with ctx.state_write as st:
            st["semantic_tree"] = tree.model_dump(mode="json")
        return RunSuccess(conversation_node_id=None, state_update=[], _route_next=["correct_pointers"])

    @resolver.register("correct_pointers")
    def _correct_pointers(ctx):
        tree = SemanticNode.model_validate(ctx.state_view["semantic_tree"])
        parser_source_map = ctx.state_view["parser_source_map"]
        repaired_tree, repaired_count = _repair_tree_pointers(tree, parser_source_map)
        with ctx.state_write as st:
            st["semantic_tree"] = repaired_tree.model_dump(mode="json")
            st["corrected_pointer_count"] = repaired_count
        return RunSuccess(conversation_node_id=None, state_update=[], _route_next=["validate_tree"])

    @resolver.register("validate_tree")
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
        return RunSuccess(
            conversation_node_id=None,
            state_update=[],
            _route_next=["export_graph"],
        )

    @resolver.register("export_graph")
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
        return RunSuccess(
            conversation_node_id=None,
            state_update=[],
            _route_next=["persist_canonical_graph"],
        )

    @resolver.register("persist_canonical_graph")
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
        return RunSuccess(conversation_node_id=None, state_update=[], _route_next=["end"])

    @resolver.register("end")
    def _end(ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    return resolver
