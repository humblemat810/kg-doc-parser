from __future__ import annotations

from uuid import uuid4

import pytest

from kg_doc_parser.semantic_document_splitting_layerwise_edits import (
    HydratedTextPointer,
    SemanticNode,
    kge_payload_to_semantic_tree,
    semantic_tree_to_kge_payload,
)

pytestmark = [pytest.mark.ci]


def _legacy_pointer(cluster_id: str, text: str) -> HydratedTextPointer:
    return HydratedTextPointer(
        source_cluster_id=cluster_id,
        start_char=0,
        end_char=len(text) - 1,
        verbatim_text=text,
    )


def _legacy_tree(child_count: int) -> SemanticNode:
    root_id = uuid4()
    children = [
        SemanticNode(
            node_id=uuid4(),
            parent_id=root_id,
            node_type="TEXT_FLOW",
            title=f"Child {index}",
            total_content_pointers=[_legacy_pointer(f"p1_c{index}", f"Text {index}")],
            child_nodes=[],
            level_from_root=1,
        )
        for index in range(child_count)
    ]
    return SemanticNode(
        node_id=root_id,
        parent_id=None,
        node_type="DOCUMENT_ROOT",
        title="Document",
        total_content_pointers=[],
        child_nodes=children,
        level_from_root=0,
    )


def test_legacy_export_rejects_a_unary_tree_before_persistence() -> None:
    with pytest.raises(ValueError, match="unary semantic node"):
        semantic_tree_to_kge_payload(_legacy_tree(1))


def test_legacy_export_and_import_preserve_a_valid_branching_tree() -> None:
    payload = semantic_tree_to_kge_payload(_legacy_tree(2))
    restored = kge_payload_to_semantic_tree(payload)

    assert len(restored.child_nodes) == 2
    assert {child.title for child in restored.child_nodes} == {"Child 0", "Child 1"}


def test_legacy_import_rejects_a_persisted_unary_tree() -> None:
    payload = semantic_tree_to_kge_payload(_legacy_tree(2))
    child_id = payload["edges"][0]["target_ids"][0]
    payload["edges"] = [edge for edge in payload["edges"] if edge["target_ids"][0] == child_id]
    payload["nodes"] = [node for node in payload["nodes"] if node["id"] in {payload["edges"][0]["source_ids"][0], child_id}]

    with pytest.raises(ValueError, match="unary semantic node"):
        kge_payload_to_semantic_tree(payload)
