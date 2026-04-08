from __future__ import annotations

from typing import Iterable

from kogwistar.engine_core.models import Grounding, Span
from kogwistar.runtime.models import WorkflowEdge, WorkflowNode


DEFAULT_WORKFLOW_ID = "kg_doc_parser.ingest.v1"


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
) -> WorkflowEdge:
    return WorkflowEdge(
        id=edge_id,
        label="wf_next",
        type="relationship",
        doc_id=edge_id,
        summary="next",
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
            "wf_is_default": True,
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
        ("parse_semantic", "parse_semantic", False, False),
        ("correct_pointers", "correct_pointers", False, False),
        ("validate_tree", "validate_tree", False, False),
        ("export_graph", "export_graph", False, False),
        ("persist_canonical_graph", "persist_canonical_graph", False, False),
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
    node_by_suffix = {node.id.split("|")[-1]: node for node in nodes}
    edge_pairs: Iterable[tuple[str, str]] = [
        ("start", "normalize_input"),
        ("normalize_input", "build_source_map"),
        ("build_source_map", "parse_semantic"),
        ("parse_semantic", "correct_pointers"),
        ("correct_pointers", "validate_tree"),
        ("validate_tree", "export_graph"),
        ("export_graph", "persist_canonical_graph"),
        ("persist_canonical_graph", "end"),
    ]
    edges = [
        _workflow_edge(
            workflow_id=workflow_id,
            edge_id=f"wf|{workflow_id}|e|{src}->{dst}",
            src=node_by_suffix[src].safe_get_id(),
            dst=node_by_suffix[dst].safe_get_id(),
        )
        for src, dst in edge_pairs
    ]
    return nodes, edges


def ensure_ingest_workflow_design(workflow_engine, workflow_id: str = DEFAULT_WORKFLOW_ID) -> None:
    nodes, edges = build_ingest_workflow_design(workflow_id)
    for node in nodes:
        node_id = node.safe_get_id()
        if not workflow_engine.persist.exists_node(node_id):
            workflow_engine.write.add_node(node)
    for edge in edges:
        edge_id = edge.safe_get_id()
        if not workflow_engine.persist.exists_edge(edge_id):
            workflow_engine.write.add_edge(edge)
