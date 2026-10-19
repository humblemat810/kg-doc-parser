"""Deterministic plain-text and Markdown PageIndex behavior."""

from __future__ import annotations

import pytest
from page_index_test_support import fixture_text, normalized_node_signature

import kg_doc_parser.workflow_ingest.page_index as page_index_module
from kg_doc_parser.workflow_ingest import PageIndexBlockSpec, parse_page_index_document

pytestmark = [pytest.mark.workflow, pytest.mark.ci]


def test_page_index_heuristic_plain_text_and_markdown_share_structure() -> None:
    text_result = parse_page_index_document(
        document_id="page-index-text",
        title="Page Index Document",
        raw_text=fixture_text("sample_page_index.txt"),
        source_format="text",
        mode="heuristic",
    )
    markdown_result = parse_page_index_document(
        document_id="page-index-markdown",
        title="Page Index Document",
        raw_text=fixture_text("sample_page_index.md"),
        source_format="markdown",
        mode="heuristic",
    )

    assert normalized_node_signature(text_result.semantic_tree) == normalized_node_signature(
        markdown_result.semantic_tree
    )


def test_page_index_heading_container_projects_source_text_to_leaf() -> None:
    result = parse_page_index_document(
        document_id="page-index-heading-leaf",
        title="Page Index Document",
        raw_text="# Results\n\nIntro.\n\n## Measurements\n\nBody.\n",
        source_format="markdown",
        mode="heuristic",
    )

    page = result.semantic_tree
    heading = page.child_nodes[0]
    assert heading.node_type == "HEADING_TEXT"
    assert heading.title == "Results"
    assert heading.metadata["page_index_role"] == "heading_text"
    assert heading.total_content_pointers[0].verbatim_text == "# Results"
    assert heading.summary == ""
    assert heading.metadata["summary_unavailable"] is True

    measurements = next(node for node in page.child_nodes if node.title == "Measurements")
    assert measurements.node_type == "HEADING"
    assert measurements.total_content_pointers == []
    assert measurements.aggregate_content_pointers[0].verbatim_text == "## Measurements\n\nBody."
    measurement_heading = measurements.child_nodes[0]
    assert measurement_heading.node_type == "HEADING_TEXT"
    assert measurement_heading.total_content_pointers[0].verbatim_text == "## Measurements"


def test_page_index_repeated_heading_titles_have_distinct_stable_ids() -> None:
    result = parse_page_index_document(
        document_id="page-index-repeated-heading",
        title="Page Index Document",
        raw_text="# Repeat\n\nFirst body.\n\n# Repeat\n\nSecond body.\n",
        source_format="markdown",
        mode="heuristic",
    )

    repeated = [
        node
        for node in result.semantic_tree.child_nodes
        if node.metadata.get("page_index_role") == "heading_container"
        and node.title == "Repeat"
    ]

    assert len(repeated) == 2
    assert repeated[0].node_id != repeated[1].node_id
    assert repeated[0].total_content_pointers == []
    assert repeated[1].total_content_pointers == []


def test_page_index_heading_ids_ignore_title_edits_at_same_source_occurrence() -> None:
    def materialize(title: str):
        spec = PageIndexBlockSpec(
            title=title,
            node_type="HEADING",
            excerpt="# Stable occurrence",
            source_role="heading",
        )
        nodes, _ = page_index_module._materialize_block_tree(
            block_specs=[spec],
            page_text="# Stable occurrence",
            unit_id="unit-1",
            parent_id="page-1",
            level_from_root=2,
        )
        return nodes[0]

    original = materialize("Original title")
    edited = materialize("Edited title")

    assert original.node_id == edited.node_id
    assert original.metadata["page_index_role"] == "heading_leaf"
    assert original.total_content_pointers == edited.total_content_pointers
