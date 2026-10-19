"""Hierarchical summary behavior for PageIndex provider workflows."""

from __future__ import annotations

import json

import pytest
from page_index_provider_test_support import install_fake_page_index_chat

import kg_doc_parser.workflow_ingest.page_index as page_index_module
import kg_doc_parser.workflow_ingest.page_index_provider as page_index_provider_module
import kg_doc_parser.workflow_ingest.page_index_refinement as page_index_refinement_module
from kg_doc_parser.workflow_ingest import (
    BlockAssignmentBatch,
    ProviderEndpointConfig,
    WorkflowProviderSettings,
    parse_page_index_document,
)

pytestmark = [pytest.mark.workflow]


def test_page_index_hierarchical_summary_passes_parent_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_text = "# Root\n\nRoot detail.\n\n## Child\n\nChild detail.\n"
    candidates = page_index_module._extract_candidate_blocks(
        raw_text,
        page_number=1,
        source_format="markdown",
    )
    assignments = page_index_module._deterministic_block_assignments(candidates)
    assignment_payload = BlockAssignmentBatch(
        assignments=[
            assignment.model_copy(update={"summary": f"initial-{index}"})
            for index, assignment in enumerate(assignments)
        ]
    )
    captured: list[object] = []

    class _SummaryStructured:
        def __init__(self, schema: object) -> None:
            self.schema = schema

        def invoke(self, messages: list[object]) -> dict[str, object]:
            captured.extend(messages)
            if self.schema is page_index_module.BlockAssignmentBatch:
                return {"parsed": assignment_payload}
            if self.schema is page_index_module.HierarchicalSummaryBatch:
                prompt = str(messages[-1].content)
                entries = json.loads(prompt.split("Blocks: ", 1)[1])
                return {
                    "parsed": page_index_module.HierarchicalSummaryBatch(
                        assignments=[
                            page_index_module.HierarchicalSummaryAssignment(
                                path_id=str(entry["path_id"]),
                                summary=f"summary-{entry['path_id']}",
                            )
                            for entry in entries
                        ]
                    )
                }
            raise AssertionError(f"unexpected structured schema: {self.schema!r}")

    class _SummaryChat:
        def with_structured_output(
            self, schema: object, include_raw: bool = True, **kwargs: object
        ) -> _SummaryStructured:
            del include_raw, kwargs
            return _SummaryStructured(schema)

    monkeypatch.setattr(
        page_index_provider_module,
        "build_chat_model_for_role",
        lambda *args, **kwargs: _SummaryChat(),
    )
    monkeypatch.setattr(
        page_index_refinement_module,
        "build_chat_model_for_role",
        lambda *args, **kwargs: _SummaryChat(),
    )
    diagnostics: list[dict[str, object]] = []
    settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(provider="ollama", model="test-model")
    )

    result = parse_page_index_document(
        document_id="hierarchical-summary",
        title="Hierarchy",
        raw_text=raw_text,
        source_format="markdown",
        mode="ollama",
        provider_settings=settings,
        provider_diagnostics_sink=diagnostics.append,
        hierarchical_summary_enabled=True,
    )

    assert result.semantic_tree.child_nodes
    page_diagnostics = result.diagnostics["page_diagnostics"][0]
    assert page_diagnostics["hierarchical_summary_enabled"] is True
    assert page_diagnostics["hierarchical_summary_accepted"] > 0
    assert any(record["operation"] == "page_index_hierarchical_summary" for record in diagnostics)
    assert any(
        '"parent_summary": "summary-' in str(message.content)
        for message in captured
        if hasattr(message, "content")
    )


def test_page_index_hierarchical_summary_provider_failure_keeps_initial_summaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_text = "# Root\n\nRoot detail.\n\n## Child\n\nChild detail.\n"
    candidates = page_index_module._extract_candidate_blocks(
        raw_text,
        page_number=1,
        source_format="markdown",
    )
    assignments = page_index_module._deterministic_block_assignments(candidates)
    initial_summaries = [assignment.summary for assignment in assignments]

    install_fake_page_index_chat(
        monkeypatch,
        assignment_payload=BlockAssignmentBatch(assignments=assignments),
    )

    def _raise_on_summary_model(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("summary model unavailable")

    monkeypatch.setattr(
        page_index_refinement_module,
        "build_chat_model_for_role",
        _raise_on_summary_model,
    )
    diagnostics: list[dict[str, object]] = []
    settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(provider="ollama", model="test-model")
    )

    result = parse_page_index_document(
        document_id="hierarchical-summary-fallback",
        title="Hierarchy",
        raw_text=raw_text,
        source_format="markdown",
        mode="ollama",
        provider_settings=settings,
        provider_diagnostics_sink=diagnostics.append,
        hierarchical_summary_enabled=True,
    )

    actual_summaries: list[str] = []

    def _collect_summaries(node: object) -> None:
        if node.node_type != "DOCUMENT_ROOT":
            actual_summaries.append(node.summary)
        for child in node.child_nodes:
            _collect_summaries(child)

    _collect_summaries(result.semantic_tree)
    assert actual_summaries
    assert all(not summary.startswith("summary-") for summary in actual_summaries)
    assert all(not summary.startswith("summary-") for summary in initial_summaries)
    page_diagnostics = result.diagnostics["page_diagnostics"][0]
    assert page_diagnostics["hierarchical_summary_fallback"] is True
    assert page_diagnostics["hierarchical_summary_accepted"] == 0
    summary_failures = [
        record
        for record in diagnostics
        if record["operation"] == "page_index_hierarchical_summary"
    ]
    assert summary_failures
    assert summary_failures[0]["failure_type"] == "transport/provider_exception"


def test_page_index_hierarchical_summary_requires_ordinary_summaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_page_index_chat(
        monkeypatch,
        assignment_payload=BlockAssignmentBatch(
            assignments=[
                page_index_module.BlockAssignment(
                    block_id="p0001-b001",
                    parent_id=None,
                    node_type="SECTION",
                    title="Root",
                )
            ]
        ),
    )
    settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(provider="ollama", model="test-model")
    )

    result = parse_page_index_document(
        document_id="hierarchical-summary-disabled",
        title="Hierarchy",
        raw_text="# Root\n\nRoot detail.\n",
        source_format="markdown",
        mode="ollama",
        provider_settings=settings,
        summary_enabled=False,
        hierarchical_summary_enabled=True,
    )

    page_diagnostics = result.diagnostics["page_diagnostics"][0]
    assert page_diagnostics["hierarchical_summary_enabled"] is False
    assert page_diagnostics["hierarchical_summary_call_count"] == 0
