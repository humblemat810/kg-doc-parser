from __future__ import annotations

import logging
from collections.abc import Iterable

from kogwistar.engine_core.models import Grounding, Span
from kogwistar.runtime.models import WorkflowEdge, WorkflowNode

DEFAULT_WORKFLOW_ID = "kg_doc_parser.ingest.v1"
_LOGGER = logging.getLogger(__name__)


def _progress_bar(done: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return "?" * width
    filled = min(width, max(0, round((done / total) * width)))
    return ("█" * filled) + ("░" * (width - filled))


def _grounding(workflow_id: str) -> Grounding:
    return Grounding(spans=[Span.from_dummy_for_workflow(workflow_id)])


def _workflow_node(
    *,
    workflow_id: str,
    node_id: str,
    op: str,
    start: bool = False,
    terminal: bool = False,
) -> WorkflowNode:
    return WorkflowNode(
        id=node_id,
        label=node_id.split("|")[-1],
        type="entity",
        doc_id=node_id,
        summary=op,
        properties={},
        metadata={
            "entity_type": "workflow_node",
            "workflow_id": workflow_id,
            "wf_op": op,
            "wf_start": bool(start),
            "wf_terminal": bool(terminal),
            "wf_version": "v1",
        },
        mentions=[_grounding(workflow_id)],
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )


def _workflow_edge(
    *,
    workflow_id: str,
    edge_id: str,
    src: str,
    dst: str,
    predicate: str | None = None,
    is_default: bool = True,
) -> WorkflowEdge:
    dst_name = dst.split("|")[-1]
    return WorkflowEdge(
        id=edge_id,
        label=dst_name,
        type="relationship",
        doc_id=edge_id,
        summary=f"next:{dst_name}",
        properties={},
        source_ids=[src],
        target_ids=[dst],
        relation="wf_next",
        source_edge_ids=[],
        target_edge_ids=[],
        metadata={
            "entity_type": "workflow_edge",
            "workflow_id": workflow_id,
            "wf_priority": 100,
            "wf_predicate": predicate,
            "wf_is_default": is_default,
            "wf_multiplicity": "one",
            "wf_version": "v1",
        },
        mentions=[_grounding(workflow_id)],
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )


def build_ingest_workflow_design(
    workflow_id: str = DEFAULT_WORKFLOW_ID,
) -> tuple[list[WorkflowNode], list[WorkflowEdge]]:
    node_specs = [
        ("start", "start", True, False),
        ("normalize_input", "normalize_input", False, False),
        ("build_source_map", "build_source_map", False, False),
        ("init_parse_session", "init_parse_session", False, False),
        ("check_frontier_remaining", "check_frontier_remaining", False, False),
        ("prepare_layer_frontier", "prepare_layer_frontier", False, False),
        ("triage_parse_strategy", "triage_parse_strategy", False, False),
        ("layer_excerpt_method", "propose_layer_breakdown", False, False),
        ("layer_boundary_method", "propose_layer_breakdown", False, False),
        ("page_index_layer", "page_index_layer", False, False),
        ("review_cud_proposal", "review_cud_proposal", False, False),
        ("apply_cud_update", "apply_cud_update", False, False),
        ("check_layer_coverage", "check_layer_coverage", False, False),
        ("check_layer_satisfaction", "check_layer_satisfaction", False, False),
        ("repair_layer_pointers", "repair_layer_pointers", False, False),
        ("dedupe_and_filter_layer", "dedupe_and_filter_layer", False, False),
        ("validate_layer_commit", "validate_layer_commit", False, False),
        ("commit_layer_children", "commit_layer_children", False, False),
        ("check_children_expandable", "check_children_expandable", False, False),
        ("enqueue_next_layer_frontier", "enqueue_next_layer_frontier", False, False),
        ("finalize_semantic_tree", "finalize_semantic_tree", False, False),
        ("validate_tree", "validate_tree", False, False),
        ("export_graph", "export_graph", False, False),
        ("persist_canonical_graph", "persist_canonical_graph", False, False),
        ("parse_failure", "parse_failure", False, True),
        ("end", "end", False, True),
    ]
    nodes = [
        _workflow_node(
            workflow_id=workflow_id,
            node_id=f"wf|{workflow_id}|{suffix}",
            op=op,
            start=start,
            terminal=terminal,
        )
        for suffix, op, start, terminal in node_specs
    ]
    node_by_suffix = {node.safe_get_id().split("|")[-1]: node for node in nodes}
    edge_pairs: Iterable[tuple[str, str] | tuple[str, str, str]] = [
        ("start", "normalize_input"),
        ("normalize_input", "build_source_map"),
        ("build_source_map", "init_parse_session"),
        ("init_parse_session", "check_frontier_remaining"),
        ("check_frontier_remaining", "prepare_layer_frontier"),
        ("check_frontier_remaining", "finalize_semantic_tree"),
        ("prepare_layer_frontier", "triage_parse_strategy"),
        ("triage_parse_strategy", "layer_excerpt_method", "parse_strategy_layer_excerpt"),
        ("triage_parse_strategy", "layer_boundary_method", "parse_strategy_layer_boundary"),
        ("triage_parse_strategy", "page_index_layer", "parse_strategy_page_index"),
        ("triage_parse_strategy", "parse_failure", "strategy_selection_failed"),
        ("layer_excerpt_method", "review_cud_proposal", "strategy_attempted"),
        ("layer_boundary_method", "review_cud_proposal", "strategy_attempted"),
        ("review_cud_proposal", "apply_cud_update"),
        ("apply_cud_update", "check_layer_coverage"),
        ("check_layer_coverage", "check_layer_satisfaction"),
        ("check_layer_satisfaction", "repair_layer_pointers", "layer_satisfied"),
        ("check_layer_satisfaction", "repair_layer_pointers", "batch_has_repair_candidates"),
        ("check_layer_satisfaction", "triage_parse_strategy", "strategy_failed_with_remaining"),
        ("check_layer_satisfaction", "parse_failure", "all_strategies_exhausted"),
        ("repair_layer_pointers", "dedupe_and_filter_layer"),
        ("dedupe_and_filter_layer", "validate_layer_commit"),
        ("validate_layer_commit", "commit_layer_children", "commit_candidates_valid"),
        ("validate_layer_commit", "check_layer_satisfaction", "commit_candidates_invalid"),
        ("commit_layer_children", "check_children_expandable"),
        ("check_children_expandable", "enqueue_next_layer_frontier"),
        ("check_children_expandable", "check_frontier_remaining"),
        ("enqueue_next_layer_frontier", "check_frontier_remaining"),
        ("finalize_semantic_tree", "validate_tree"),
        ("page_index_layer", "review_cud_proposal", "strategy_attempted"),
        ("validate_tree", "export_graph"),
        ("export_graph", "persist_canonical_graph"),
        ("persist_canonical_graph", "end"),
    ]
    edges = [
        _workflow_edge(
            workflow_id=workflow_id,
            edge_id=f"wf|{workflow_id}|e|{index}|{src}->{dst}",
            src=node_by_suffix[src].safe_get_id(),
            dst=node_by_suffix[dst].safe_get_id(),
            predicate=predicate if len(edge_spec) == 3 else None,
            is_default=len(edge_spec) == 2,
        )
        for index, edge_spec in enumerate(edge_pairs)
        for src, dst, *predicate_values in [edge_spec]
        for predicate in [predicate_values[0] if predicate_values else None]
    ]
    return nodes, edges


def ensure_ingest_workflow_design(workflow_engine, workflow_id: str = DEFAULT_WORKFLOW_ID) -> None:
    nodes, edges = build_ingest_workflow_design(workflow_id)
    total = len(nodes) + len(edges)
    done = 0
    _LOGGER.info(
        "⏳ workflow design | 0/%s |   0%% | %s | bootstrap %s nodes + %s edges",
        total,
        _progress_bar(0, total),
        len(nodes),
        len(edges),
    )
    for node_idx, node in enumerate(nodes, start=1):
        node_id = node.safe_get_id()
        if not workflow_engine.persist.exists_node(node_id):
            workflow_engine.write.add_node(node)
        done += 1
        _LOGGER.info(
            "⏳ workflow design | %s/%s | %3s%% | %s | node %s/%s | %s",
            done,
            total,
            round((done / total) * 100),
            _progress_bar(done, total),
            node_idx,
            len(nodes),
            node.label,
        )
    for edge_idx, edge in enumerate(edges, start=1):
        edge_id = edge.safe_get_id()
        if not workflow_engine.persist.exists_edge(edge_id):
            workflow_engine.write.add_edge(edge)
        done += 1
        _LOGGER.info(
            "⏳ workflow design | %s/%s | %3s%% | %s | edge %s/%s | %s",
            done,
            total,
            round((done / total) * 100),
            _progress_bar(done, total),
            edge_idx,
            len(edges),
            edge.label,
        )
