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


def _legacy_source_map(child_count: int) -> dict[str, dict[str, object]]:
    return {
        f"p1_c{index}": {"text": f"Text {index}"}
        for index in range(child_count)
    }


def test_legacy_export_rejects_a_unary_tree_before_persistence() -> None:
    with pytest.raises(ValueError, match="unary semantic node"):
        semantic_tree_to_kge_payload(
            _legacy_tree(1), parser_source_map=_legacy_source_map(1)
        )


def test_legacy_export_and_import_preserve_a_valid_branching_tree() -> None:
    payload = semantic_tree_to_kge_payload(
        _legacy_tree(2), parser_source_map=_legacy_source_map(2)
    )
    restored = kge_payload_to_semantic_tree(payload)

    assert len(restored.child_nodes) == 2
    assert {child.title for child in restored.child_nodes} == {"Child 0", "Child 1"}


def test_legacy_import_rejects_a_persisted_unary_tree() -> None:
    payload = semantic_tree_to_kge_payload(
        _legacy_tree(2), parser_source_map=_legacy_source_map(2)
    )
    child_id = payload["edges"][0]["target_ids"][0]
    payload["edges"] = [edge for edge in payload["edges"] if edge["target_ids"][0] == child_id]
    payload["nodes"] = [node for node in payload["nodes"] if node["id"] in {payload["edges"][0]["source_ids"][0], child_id}]

    with pytest.raises(ValueError, match="unary semantic node"):
        kge_payload_to_semantic_tree(payload)


def test_legacy_export_rejects_a_parent_id_mismatch():
    root = _legacy_tree(2)
    root.child_nodes[0].parent_id = "wrong-parent"

    with pytest.raises(ValueError, match="parent mismatch"):
        semantic_tree_to_kge_payload(root, parser_source_map=_legacy_source_map(2))


def test_legacy_export_requires_authoritative_source_map() -> None:
    with pytest.raises(ValueError, match="parser_source_map"):
        semantic_tree_to_kge_payload(_legacy_tree(2))


def test_legacy_export_rejects_pointer_excerpt_that_disagrees_with_source() -> None:
    tree = _legacy_tree(2)
    tree.child_nodes[0].total_content_pointers[0].verbatim_text = "forged"

    with pytest.raises(ValueError, match="authoritative source slice"):
        semantic_tree_to_kge_payload(
            tree, parser_source_map=_legacy_source_map(2)
        )
