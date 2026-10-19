"""Excerpt-refinement acceptance, fallback, and grounding tests."""

from __future__ import annotations

import pytest
from page_index_provider_test_support import (
    install_fake_page_index_chat as _install_fake_page_index_chat,
)
from page_index_test_support import (
    normalized_node_signature as _normalized_node_signature,
)

import kg_doc_parser.workflow_ingest.page_index as page_index_module
import kg_doc_parser.workflow_ingest.page_index_contracts as page_index_contracts_module
import kg_doc_parser.workflow_ingest.page_index_refinement as page_index_refinement_module
from kg_doc_parser.workflow_ingest import (
    BlockAssignment,
    BlockAssignmentBatch,
    ProviderEndpointConfig,
    WorkflowProviderSettings,
    parse_page_index_document,
)

pytestmark = [pytest.mark.workflow]

def test_page_index_refinement_disabled_by_default_preserves_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_text = "# Root\n\nIntro paragraph.\n\n## Child\n\n- Term item\n"
    assignments = BlockAssignmentBatch(
        assignments=[
            BlockAssignment(block_id="p0001-b001", parent_id=None, node_type="SECTION", title="Root"),
            BlockAssignment(block_id="p0001-b002", parent_id="p0001-b001", node_type="PARAGRAPH", title="Intro paragraph"),
            BlockAssignment(block_id="p0001-b003", parent_id="p0001-b001", node_type="SUBSECTION", title="Child"),
            BlockAssignment(block_id="p0001-b004", parent_id="p0001-b003", node_type="TERM", title="Term item"),
        ]
    )
    _install_fake_page_index_chat(monkeypatch, assignment_payload=assignments)

    provider_settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(provider="ollama", model="fake", base_url="http://127.0.0.1:11434")
    )
    default_result = parse_page_index_document(
        document_id="page-index-ollama-default-refine",
        title="Page Index Document",
        raw_text=raw_text,
        source_format="markdown",
        mode="ollama",
        provider_settings=provider_settings,
    )
    explicit_result = parse_page_index_document(
        document_id="page-index-ollama-explicit-refine-off",
        title="Page Index Document",
        raw_text=raw_text,
        source_format="markdown",
        mode="ollama",
        provider_settings=provider_settings,
        refine_excerpts=False,
    )

    assert _normalized_node_signature(default_result.semantic_tree) == _normalized_node_signature(explicit_result.semantic_tree)
    assert default_result.diagnostics["refine_excerpts_enabled"] is False
    assert default_result.diagnostics["refine_excerpts_attempted"] == 0


def test_page_index_refinement_rejects_shorter_leaf_substring(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_text = "# Root\n\nIntro paragraph.\n\n## Child\n\n- Term item\n"
    refinement = page_index_contracts_module.ExcerptRefinementBatch(
        suggestions=[
            page_index_contracts_module.ExcerptRefinementSuggestion(path_id="0.0", excerpt="Intro paragraph"),
        ]
    )
    _install_fake_page_index_chat(
        monkeypatch,
        assignment_payload=BlockAssignmentBatch(assignments=[]),
        refinement_payload=refinement,
    )

    block_specs = [
        page_index_module.PageIndexBlockSpec(
            title="Root",
            node_type="SECTION",
            excerpt="# Root",
            child_nodes=[
                page_index_module.PageIndexBlockSpec(
                    title="Intro paragraph",
                    node_type="PARAGRAPH",
                    excerpt="Intro paragraph.",
                    child_nodes=[],
                )
            ],
        )
    ]

    provider_settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(provider="ollama", model="fake", base_url="http://127.0.0.1:11434")
    )
    refined, diagnostics = page_index_refinement_module.refine_page_index_block_excerpts(
        block_specs=block_specs,
        page_text=raw_text,
        page_number=1,
        unit_id="page-1",
        provider_settings=provider_settings,
    )

    assert refined[0].child_nodes[0].excerpt == "Intro paragraph."
    assert diagnostics["refine_excerpts_enabled"] is True
    assert diagnostics["refine_excerpts_accepted"] == 0
    assert diagnostics["refine_excerpts_rejected"] == 1
    assert diagnostics["refine_excerpts_fallback"] is False


def test_page_index_refinement_rejects_non_substring_and_keeps_original(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_text = "# Root\n\nIntro paragraph.\n\n## Child\n\n- Term item\n"
    assignments = BlockAssignmentBatch(
        assignments=[
            BlockAssignment(block_id="p0001-b001", parent_id=None, node_type="SECTION", title="Root"),
            BlockAssignment(block_id="p0001-b002", parent_id="p0001-b001", node_type="PARAGRAPH", title="Intro paragraph"),
            BlockAssignment(block_id="p0001-b003", parent_id="p0001-b001", node_type="SUBSECTION", title="Child"),
            BlockAssignment(block_id="p0001-b004", parent_id="p0001-b003", node_type="TERM", title="Term item"),
        ]
    )
    refinement = page_index_contracts_module.ExcerptRefinementBatch(
        suggestions=[
            page_index_contracts_module.ExcerptRefinementSuggestion(path_id="0.0", excerpt="not present"),
        ]
    )
    _install_fake_page_index_chat(monkeypatch, assignment_payload=assignments, refinement_payload=refinement)

    provider_settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(provider="ollama", model="fake", base_url="http://127.0.0.1:11434")
    )
    result = parse_page_index_document(
        document_id="page-index-ollama-refine-reject",
        title="Page Index Document",
        raw_text=raw_text,
        source_format="markdown",
        mode="ollama",
        provider_settings=provider_settings,
        refine_excerpts=True,
    )
    baseline_result = parse_page_index_document(
        document_id="page-index-ollama-refine-reject-baseline",
        title="Page Index Document",
        raw_text=raw_text,
        source_format="markdown",
        mode="ollama",
        provider_settings=provider_settings,
    )

    root = result.semantic_tree
    baseline_root = baseline_result.semantic_tree
    assert root.child_nodes[0].total_content_pointers[0].verbatim_text == baseline_root.child_nodes[0].total_content_pointers[0].verbatim_text
    assert result.diagnostics["refine_excerpts_enabled"] is True
    assert result.diagnostics["refine_excerpts_accepted"] == 0
    assert result.diagnostics["refine_excerpts_rejected"] == 1
    assert result.diagnostics["refine_excerpts_fallback"] is False


def test_page_index_refinement_rejects_whole_page_and_duplicate_excerpts(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_text = "# Root\n\nAlpha paragraph.\n\nBeta paragraph.\n"
    assignments = BlockAssignmentBatch(
        assignments=[
            BlockAssignment(block_id="p0001-b001", parent_id=None, node_type="SECTION", title="Root"),
            BlockAssignment(block_id="p0001-b002", parent_id="p0001-b001", node_type="PARAGRAPH", title="Alpha paragraph"),
            BlockAssignment(block_id="p0001-b003", parent_id="p0001-b001", node_type="PARAGRAPH", title="Beta paragraph"),
        ]
    )
    refinement = page_index_contracts_module.ExcerptRefinementBatch(
        suggestions=[
            page_index_contracts_module.ExcerptRefinementSuggestion(path_id="0", excerpt=raw_text),
            page_index_contracts_module.ExcerptRefinementSuggestion(path_id="0.0", excerpt="Alpha paragraph."),
            page_index_contracts_module.ExcerptRefinementSuggestion(path_id="0.1", excerpt="Alpha paragraph."),
        ]
    )
    _install_fake_page_index_chat(monkeypatch, assignment_payload=assignments, refinement_payload=refinement)

    provider_settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(provider="ollama", model="fake", base_url="http://127.0.0.1:11434")
    )
    result = parse_page_index_document(
        document_id="page-index-ollama-refine-dup",
        title="Page Index Document",
        raw_text=raw_text,
        source_format="markdown",
        mode="ollama",
        provider_settings=provider_settings,
        refine_excerpts=True,
    )
    baseline_result = parse_page_index_document(
        document_id="page-index-ollama-refine-dup-baseline",
        title="Page Index Document",
        raw_text=raw_text,
        source_format="markdown",
        mode="ollama",
        provider_settings=provider_settings,
    )

    root = result.semantic_tree
    baseline_root = baseline_result.semantic_tree
    assert root.total_content_pointers
    assert baseline_root.total_content_pointers
    root_heading = root.child_nodes[0]
    baseline_heading = baseline_root.child_nodes[0]
    assert root_heading.node_type == "HEADING_TEXT"
    assert root_heading.total_content_pointers[0].verbatim_text == baseline_heading.total_content_pointers[0].verbatim_text
    assert root.child_nodes[0].total_content_pointers[0].verbatim_text == baseline_root.child_nodes[0].total_content_pointers[0].verbatim_text
    assert root.child_nodes[1].total_content_pointers[0].verbatim_text == baseline_root.child_nodes[1].total_content_pointers[0].verbatim_text
    assert result.diagnostics["refine_excerpts_enabled"] is True
    assert result.diagnostics["refine_excerpts_accepted"] == 0
    assert result.diagnostics["refine_excerpts_rejected"] == 3
    assert result.diagnostics["refine_excerpts_fallback"] is False


def test_page_index_refinement_transport_failure_keeps_original_excerpts(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_text = "# Root\n\nIntro paragraph.\n\n## Child\n\n- Term item\n"
    assignments = BlockAssignmentBatch(
        assignments=[
            BlockAssignment(block_id="p0001-b001", parent_id=None, node_type="SECTION", title="Root"),
            BlockAssignment(block_id="p0001-b002", parent_id="p0001-b001", node_type="PARAGRAPH", title="Intro paragraph"),
            BlockAssignment(block_id="p0001-b003", parent_id="p0001-b001", node_type="SUBSECTION", title="Child"),
            BlockAssignment(block_id="p0001-b004", parent_id="p0001-b003", node_type="TERM", title="Term item"),
        ]
    )
    _install_fake_page_index_chat(
        monkeypatch,
        assignment_payload=assignments,
        refinement_parsing_error="malformed refinement payload",
    )

    provider_settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(provider="ollama", model="fake", base_url="http://127.0.0.1:11434")
    )
    result = parse_page_index_document(
        document_id="page-index-ollama-refine-fallback",
        title="Page Index Document",
        raw_text=raw_text,
        source_format="markdown",
        mode="ollama",
        provider_settings=provider_settings,
        refine_excerpts=True,
    )
    baseline_result = parse_page_index_document(
        document_id="page-index-ollama-refine-fallback-baseline",
        title="Page Index Document",
        raw_text=raw_text,
        source_format="markdown",
        mode="ollama",
        provider_settings=provider_settings,
    )

    root = result.semantic_tree
    baseline_root = baseline_result.semantic_tree
    assert root.child_nodes[0].total_content_pointers[0].verbatim_text == baseline_root.child_nodes[0].total_content_pointers[0].verbatim_text
    assert result.diagnostics["refine_excerpts_enabled"] is True
    assert result.diagnostics["refine_excerpts_fallback"] is True
    assert result.diagnostics["refine_excerpts_accepted"] == 0
    assert result.diagnostics["refine_excerpts_rejected"] == result.diagnostics["refine_excerpts_attempted"]


def test_page_index_refinement_reports_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_text = "# Root\n\nIntro paragraph.\n"
    assignments = BlockAssignmentBatch(
        assignments=[
            BlockAssignment(block_id="p0001-b001", parent_id=None, node_type="SECTION", title="Root"),
            BlockAssignment(block_id="p0001-b002", parent_id="p0001-b001", node_type="PARAGRAPH", title="Intro paragraph"),
        ]
    )
    refinement = page_index_contracts_module.ExcerptRefinementBatch(
        suggestions=[
            page_index_contracts_module.ExcerptRefinementSuggestion(path_id="0.0", excerpt="Intro paragraph"),
        ]
    )
    _install_fake_page_index_chat(monkeypatch, assignment_payload=assignments, refinement_payload=refinement)

    class _UsageCallback:
        recorded_call_count = 0

        def __init__(self) -> None:
            self.call_keys: list[str] = []

        def record_untracked_call(self, call_key: str) -> None:
            self.call_keys.append(call_key)
            self.recorded_call_count += 1

    usage_callback = _UsageCallback()
    provider_settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(provider="openai", model="fake")
    )
    result = parse_page_index_document(
        document_id="page-index-openai-refine-counts",
        title="Page Index Document",
        raw_text=raw_text,
        source_format="markdown",
        mode="openai",
        provider_settings=provider_settings,
        callbacks=[usage_callback],
        refine_excerpts=True,
    )

    assert result.diagnostics["refine_excerpts_enabled"] is True
    assert result.diagnostics["refine_excerpts_attempted"] == 2
    assert result.diagnostics["refine_excerpts_accepted"] == 0
    assert result.diagnostics["refine_excerpts_rejected"] == 1
    assert result.diagnostics["refine_excerpts_fallback"] is False
    assert usage_callback.call_keys == ["page-1-first", "refine-page-1"]


def test_page_index_refinement_preserves_terminal_coverage_when_punctuation_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shorter model excerpt must not remove authoritative terminal text."""
    raw_text = "# Root\n\nIntro paragraph.\n"
    assignments = BlockAssignmentBatch(
        assignments=[
            BlockAssignment(block_id="p0001-b001", parent_id=None, node_type="SECTION", title="Root"),
            BlockAssignment(
                block_id="p0001-b002",
                parent_id="p0001-b001",
                node_type="PARAGRAPH",
                title="Intro paragraph",
            ),
        ]
    )
    refinement = page_index_contracts_module.ExcerptRefinementBatch(
        suggestions=[
            page_index_contracts_module.ExcerptRefinementSuggestion(
                path_id="0.0", excerpt="Intro paragraph"
            ),
        ]
    )
    _install_fake_page_index_chat(
        monkeypatch,
        assignment_payload=assignments,
        refinement_payload=refinement,
    )

    result = parse_page_index_document(
        document_id="page-index-refinement-terminal-coverage",
        title="Page Index Document",
        raw_text=raw_text,
        source_format="markdown",
        mode="openai",
        provider_settings=WorkflowProviderSettings(
            parser=ProviderEndpointConfig(provider="openai", model="fake")
        ),
        refine_excerpts=True,
    )

    assert result.coverage["overall"] == pytest.approx(1.0)
    assert result.diagnostics["refine_excerpts_accepted"] == 0
    assert result.diagnostics["refine_excerpts_rejected"] == 1
    pending = [result.semantic_tree]
    pointers = []
    while pending:
        node = pending.pop()
        pointers.extend(node.total_content_pointers)
        pointers.extend(node.aggregate_content_pointers)
        pending.extend(node.child_nodes)
    assert any(pointer.verbatim_text == "Intro paragraph." for pointer in pointers)
