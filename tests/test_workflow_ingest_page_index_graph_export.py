"""Graph projection contracts for hierarchical PageIndex trees."""

from __future__ import annotations

import pytest

from kg_doc_parser.workflow_ingest.semantics import (
    HydratedTextPointer,
    SemanticNode,
    semantic_tree_to_kge_payload,
)

pytestmark = [pytest.mark.workflow, pytest.mark.ci]


def test_graph_export_consumes_hierarchy_for_identical_source_evidence() -> None:
    source_text = "alpha beta gamma delta"
    source_map = {"unit": {"text": source_text}}
    spans = {
        name: HydratedTextPointer(
            source_cluster_id="unit",
            start_char=start,
            end_char=end,
            verbatim_text=source_text[start : end + 1],
        )
        for name, start, end in (
            ("alpha", 0, 4),
            ("beta", 6, 9),
            ("gamma", 11, 15),
            ("delta", 17, 21),
        )
    }

    def _tree(first_group: tuple[str, str], second_group: tuple[str, str]) -> SemanticNode:
        leaves = {
            name: SemanticNode(
                node_id=name,
                parent_id="group-a" if name in first_group else "group-b",
                title=name,
                total_content_pointers=[spans[name]],
                level_from_root=2,
            )
            for name in spans
        }
        return SemanticNode(
            node_id="root",
            title="Root",
            node_type="DOCUMENT_ROOT",
            child_nodes=[
                SemanticNode(
                    node_id="group-a",
                    parent_id="root",
                    title="First group",
                    level_from_root=1,
                    child_nodes=[leaves[name] for name in first_group],
                ),
                SemanticNode(
                    node_id="group-b",
                    parent_id="root",
                    title="Second group",
                    level_from_root=1,
                    child_nodes=[leaves[name] for name in second_group],
                ),
            ],
        )

    first = semantic_tree_to_kge_payload(
        _tree(("alpha", "beta"), ("gamma", "delta")),
        doc_id="hierarchy-metamorphic",
        parser_source_map=source_map,
    )
    second = semantic_tree_to_kge_payload(
        _tree(("alpha", "gamma"), ("beta", "delta")),
        doc_id="hierarchy-metamorphic",
        parser_source_map=source_map,
    )

    def _parent_child_pairs(payload: dict[str, object]) -> set[tuple[str, str]]:
        edges = payload["edges"]
        assert isinstance(edges, list)
        return {
            (str(edge["source_ids"][0]), str(edge["target_ids"][0]))
            for edge in edges
            if isinstance(edge, dict)
        }

    first_pairs = _parent_child_pairs(first)
    second_pairs = _parent_child_pairs(second)
    assert first_pairs != second_pairs
    assert ("group-a", "alpha") in first_pairs
    assert ("group-a", "gamma") in second_pairs


def test_graph_export_preserves_mixed_multimodal_sibling_reading_order() -> None:
    source = "FIGURECAPTIONTABLEBODY"
    ordered_items = (
        ("figure", "FIGURE", 0, 6, "image_region"),
        ("caption", "CAPTION", 6, 13, "ocr_text"),
        ("table", "TABLE", 13, 18, "table"),
        ("body", "BODY", 18, 22, "ocr_text"),
    )
    children = [
        SemanticNode(
            node_id=node_id,
            parent_id="root",
            title=title,
            node_type="TEXT" if modality == "ocr_text" else "NON_TEXT",
            total_content_pointers=[
                HydratedTextPointer(
                    source_cluster_id="page-1",
                    start_char=start,
                    end_char=end - 1,
                    verbatim_text=source[start:end],
                )
            ],
            metadata={"modality": modality},
        )
        for node_id, title, start, end, modality in ordered_items
    ]
    root = SemanticNode(
        node_id="root",
        title="Document",
        node_type="DOCUMENT_ROOT",
        child_nodes=children,
    )

    payload = semantic_tree_to_kge_payload(
        root,
        doc_id="multimodal-order",
        parser_source_map={"page-1": {"text": source}},
    )

    edges = payload["edges"]
    nodes = payload["nodes"]
    assert isinstance(edges, list)
    assert isinstance(nodes, list)
    assert [edge["target_ids"][0] for edge in edges] == [
        "figure",
        "caption",
        "table",
        "body",
    ]
    assert [node["id"] for node in nodes] == [
        "root",
        "figure",
        "caption",
        "table",
        "body",
    ]
