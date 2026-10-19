"""Provider assignment acceptance and fallback for PageIndex parsing."""

from __future__ import annotations

import pytest
from page_index_provider_test_support import install_fake_page_index_chat

from kg_doc_parser.workflow_ingest import (
    BlockAssignment,
    BlockAssignmentBatch,
    ProviderEndpointConfig,
    WorkflowProviderSettings,
    parse_page_index_document,
)

pytestmark = [pytest.mark.workflow]

_RAW_TEXT = "# Root\n\nIntro paragraph.\n\n## Child\n\n- Term item\n"


def _parse_with_assignment_payload(
    monkeypatch: pytest.MonkeyPatch,
    *,
    document_id: str,
    assignments: BlockAssignmentBatch | None,
    parsing_error: str | None = None,
):
    install_fake_page_index_chat(
        monkeypatch,
        assignment_payload=assignments,
        assignment_parsing_error=parsing_error,
    )
    settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(
            provider="ollama",
            model="fake",
            base_url="http://127.0.0.1:11434",
        )
    )
    return parse_page_index_document(
        document_id=document_id,
        title="Page Index Document",
        raw_text=_RAW_TEXT,
        source_format="markdown",
        mode="ollama",
        provider_settings=settings,
    )


def test_page_index_ollama_flat_assignment_parses_and_assembles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assignments = BlockAssignmentBatch(
        assignments=[
            BlockAssignment(block_id="p0001-b001", parent_id=None, node_type="SECTION", title="Root"),
            BlockAssignment(block_id="p0001-b002", parent_id="p0001-b001", node_type="PARAGRAPH", title="Intro paragraph"),
            BlockAssignment(block_id="p0001-b003", parent_id="p0001-b001", node_type="SUBSECTION", title="Child"),
            BlockAssignment(block_id="p0001-b004", parent_id="p0001-b003", node_type="TERM", title="Term item"),
        ]
    )
    result = _parse_with_assignment_payload(
        monkeypatch,
        document_id="page-index-ollama-flat",
        assignments=assignments,
    )

    assert result.diagnostics["assignment_mode"] == "ollama_flat_assignment"
    assert result.diagnostics["fallback_reason"] is None
    root = next(node for node in result.semantic_tree.child_nodes if node.title == "Root")
    assert root.node_type == "HEADING_TEXT"
    assert root.total_content_pointers[0].verbatim_text == "# Root"
    child = next(node for node in result.semantic_tree.child_nodes if node.title == "Child")
    assert child.child_nodes[-1].title == "Term item"


def test_page_index_ollama_malformed_payload_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _parse_with_assignment_payload(
        monkeypatch,
        document_id="page-index-ollama-malformed",
        assignments=None,
        parsing_error="malformed json",
    )

    assert result.diagnostics["assignment_mode"] == "deterministic_fallback"
    assert result.diagnostics["fallback_reason"] == "llm_unavailable_or_invalid"
    assert result.semantic_tree.child_nodes[0].title == "Root"


def test_page_index_ollama_invalid_assignments_fall_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad_assignments = BlockAssignmentBatch(
        assignments=[
            BlockAssignment(block_id="p0001-b001", parent_id=None, node_type="SECTION", title="Root"),
            BlockAssignment(block_id="p0001-b002", parent_id="p0001-b001", node_type="PARAGRAPH", title="Intro paragraph"),
        ]
    )
    result = _parse_with_assignment_payload(
        monkeypatch,
        document_id="page-index-ollama-fallback",
        assignments=bad_assignments,
    )

    assert result.diagnostics["assignment_mode"] == "deterministic_fallback"
    assert result.diagnostics["validation_errors"]
    assert result.semantic_tree.child_nodes[0].title == "Root"


def test_page_index_ollama_salvages_valid_branches_after_local_assignment_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assignments = BlockAssignmentBatch(
        assignments=[
            BlockAssignment(block_id="p0001-b001", parent_id=None, node_type="SECTION", title="Provider Root"),
            BlockAssignment(block_id="p0001-b002", parent_id="p0001-b001", node_type="PARAGRAPH", title="Provider intro"),
            BlockAssignment(block_id="p0001-b003", parent_id="p0001-b001", node_type="SUBSECTION", title="Provider child"),
            BlockAssignment(block_id="p0001-b004", parent_id="missing-parent", node_type="TERM", title="Provider term"),
        ]
    )
    result = _parse_with_assignment_payload(
        monkeypatch,
        document_id="page-index-ollama-branch-salvage",
        assignments=assignments,
    )

    assert result.diagnostics["assignment_mode"] == "ollama_branch_local_salvage", result.diagnostics
    assert result.diagnostics["branch_local_salvage"][0]["invalid_block_ids"] == ["p0001-b004"]
    root = next(node for node in result.semantic_tree.child_nodes if node.title == "Provider Root")
    assert root.node_type == "HEADING_TEXT"
    child = next(node for node in result.semantic_tree.child_nodes if node.title == "Provider child")
    term = next(node for node in child.child_nodes if node.title == "Term item")
    assert term.total_content_pointers[0].verbatim_text == "- Term item"


def test_page_index_ollama_flat_but_shallow_assignment_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assignments = BlockAssignmentBatch(
        assignments=[
            BlockAssignment(block_id="p0001-b001", parent_id=None, node_type="SECTION", title="Root"),
            BlockAssignment(block_id="p0001-b002", parent_id="p0001-b001", node_type="PARAGRAPH", title="Intro paragraph"),
            BlockAssignment(block_id="p0001-b003", parent_id="p0001-b001", node_type="PARAGRAPH", title="Child"),
            BlockAssignment(block_id="p0001-b004", parent_id="p0001-b001", node_type="PARAGRAPH", title="Term item"),
        ]
    )
    result = _parse_with_assignment_payload(
        monkeypatch,
        document_id="page-index-ollama-shallow",
        assignments=assignments,
    )

    assert result.diagnostics["assignment_mode"] == "deterministic_fallback"
    assert any(
        "heading structure was flattened" in error
        for error in result.diagnostics["validation_errors"]
    )
    assert result.semantic_tree.child_nodes[0].title == "Root"
