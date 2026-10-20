"""Layer-level PageIndex contracts and commit behavior."""

from __future__ import annotations

import pytest

import kg_doc_parser.workflow_ingest.page_index as page_index_module
from kg_doc_parser.workflow_ingest import (
    BlockAssignment,
    BlockAssignmentBatch,
    CandidateBlock,
    PageIndexParseResult,
    parse_page_index_layer,
)
from kg_doc_parser.workflow_ingest.models import CurrentLayerResult, LayerChildCandidate
from kg_doc_parser.workflow_ingest.parser_core import commit_layer_children
from kg_doc_parser.workflow_ingest.semantics import HydratedTextPointer, SemanticNode

pytestmark = [pytest.mark.workflow, pytest.mark.ci]


def test_page_index_module_exports_hybrid_primitives() -> None:
    assert hasattr(page_index_module, "CandidateBlock")
    assert hasattr(page_index_module, "BlockAssignment")
    assert hasattr(page_index_module, "BlockAssignmentBatch")
    assert hasattr(page_index_module, "PageIndexValidationResult")
    assert CandidateBlock is page_index_module.CandidateBlock
    assert BlockAssignment is page_index_module.BlockAssignment
    assert BlockAssignmentBatch is page_index_module.BlockAssignmentBatch
    assert PageIndexParseResult is page_index_module.PageIndexParseResult


def test_page_index_layer_refines_only_direct_children_of_one_parent() -> None:
    source_text = "# Parent\n\nParent introduction.\n\n## Child\n\nChild detail.\n"
    candidates = parse_page_index_layer(
        parent_id="parent",
        parent_title="Parent",
        parent_pointers=[
            HydratedTextPointer(
                source_cluster_id="unit-1",
                start_char=0,
                end_char=len(source_text) - 1,
                verbatim_text=source_text,
            )
        ],
        parser_source_map={"unit-1": {"text": source_text}},
        source_format="markdown",
    )

    assert candidates
    assert {candidate.parent_node_id for candidate in candidates} == {"parent"}
    assert all(candidate.metadata["page_index_layer_only"] is True for candidate in candidates)
    child_heading = next(candidate for candidate in candidates if candidate.title == "Child")
    assert child_heading.expandable is True
    assert [child.title for child in child_heading.child_candidates] == [
        "Child",
        "Child detail.",
    ]
    assert child_heading.child_candidates[0].node_type == "HEADING_TEXT"
    assert child_heading.child_candidates[0].parent_node_id == child_heading.node_id
    assert all(candidate.title != "Child detail." for candidate in candidates)


def test_page_index_materialized_title_leaf_survives_layer_commit() -> None:
    pointer = HydratedTextPointer(
        source_cluster_id="unit-1",
        start_char=0,
        end_char=4,
        verbatim_text="Title",
    )
    heading = LayerChildCandidate(
        node_id="heading",
        parent_node_id="root",
        title="Title",
        node_type="HEADING",
        total_content_pointers=[pointer],
        child_candidates=[
            LayerChildCandidate(
                node_id="heading-text",
                parent_node_id="heading",
                title="Title",
                node_type="HEADING_TEXT",
                total_content_pointers=[pointer],
                expandable=False,
            ),
            LayerChildCandidate(
                node_id="heading-body",
                parent_node_id="heading",
                title="Body",
                node_type="TEXT_FLOW",
                total_content_pointers=[pointer],
                expandable=False,
            ),
        ],
    )
    sibling = LayerChildCandidate(
        node_id="heading-sibling",
        parent_node_id="root",
        title="Other",
        node_type="TEXT_FLOW",
        total_content_pointers=[
            HydratedTextPointer(
                source_cluster_id="unit-1",
                start_char=5,
                end_char=9,
                verbatim_text="Other",
            )
        ],
    )
    committed = commit_layer_children(
        semantic_tree=SemanticNode(node_id="root", parent_id=None, title="Root", node_type="ROOT"),
        current_layer_result=CurrentLayerResult(children=[heading, sibling], satisfied=True),
        current_depth=0,
    )

    assert committed.child_nodes[0].title == "Title"
    assert committed.child_nodes[0].child_nodes[0].node_type == "HEADING_TEXT"
