from __future__ import annotations

from pathlib import Path

import kg_doc_parser.workflow_ingest.page_index as page_index_module
import pytest
from _kogwistar_test_helpers import build_workflow_engine_triplet
from kg_doc_parser.workflow_ingest import (
    BlockAssignment,
    BlockAssignmentBatch,
    CurrentLayerContext,
    CurrentLayerResult,
    CurrentLayerReview,
    HydratedTextPointer,
    LayerChildCandidate,
    ProviderEndpointConfig,
    WorkflowIngestInput,
    WorkflowProviderSettings,
    parse_page_index_document,
)
from kg_doc_parser.workflow_ingest.parser_core import detect_layer_invariants
from kg_doc_parser.workflow_ingest.service import run_ingest_workflow

pytestmark = [pytest.mark.workflow, pytest.mark.ci]


def _long_table() -> str:
    return (Path(__file__).parent / "fixtures" / "page_index" / "title_long_table.md").read_text(
        encoding="utf-8"
    )


def _install_atomic_table_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StructuredOutput:
        def invoke(self, _messages):
            return {
                "parsed": BlockAssignmentBatch(
                    assignments=[
                        BlockAssignment(
                            block_id="p0001-b001",
                            parent_id=None,
                            node_type="SECTION",
                            title="Sample Document Title",
                        ),
                        BlockAssignment(
                            block_id="p0001-b002",
                            parent_id="p0001-b001",
                            node_type="TABLE",
                            title="ID Description Value",
                        ),
                    ]
                )
            }

    class _FakeChat:
        def with_structured_output(self, _schema, include_raw=True, **_kwargs):
            assert include_raw is True
            return _StructuredOutput()

    monkeypatch.setattr(page_index_module, "build_chat_model_for_role", lambda *args, **kwargs: _FakeChat())


@pytest.mark.parametrize("mode", ["heuristic", "ollama"])
def test_corrective_long_table_page_index_matrix(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if mode == "ollama":
        _install_atomic_table_provider(monkeypatch)

    result = parse_page_index_document(
        document_id=f"corrective-long-table-{mode}",
        title="Sample Document Title",
        raw_text=_long_table(),
        source_format="markdown",
        mode=mode,
        provider_settings=WorkflowProviderSettings(
            parser=ProviderEndpointConfig(provider="ollama", model="test-provider")
        ),
        summary_enabled=False,
    )

    assert result.coverage["overall"] == pytest.approx(1.0)
    assert result.semantic_tree.child_nodes
    assert result.authoritative_source_map.keys() == result.parser_source_map.keys()
    assert not any("too broad" in error for error in result.diagnostics.get("validation_errors", []))


def _full_source_pointer(inp: WorkflowIngestInput) -> HydratedTextPointer:
    unit = inp.collections[0].pages[0].units[0]
    text = unit.text or ""
    return HydratedTextPointer(
        source_cluster_id=f"{inp.request_id}|p1_t0",
        start_char=0,
        end_char=max(0, len(text) - 1),
        verbatim_text=text,
    )


@pytest.mark.parametrize("strategy", ["excerpt_first", "boundary_first", "page_index"])
@pytest.mark.parametrize("coverage_ratio", [0.95, 0.99])
def test_partial_non_whitespace_grounding_is_rejected_for_each_route(
    strategy: str,
    coverage_ratio: float,
) -> None:
    text = _long_table()
    source_cluster_id = "partial-grounding|p1_t0"
    cutoff = max(1, int(len(text) * coverage_ratio))
    parent_pointer = HydratedTextPointer(
        source_cluster_id=source_cluster_id,
        start_char=0,
        end_char=len(text) - 1,
        verbatim_text=text,
    )
    partial_pointer = HydratedTextPointer(
        source_cluster_id=source_cluster_id,
        start_char=0,
        end_char=cutoff - 1,
        verbatim_text=text[:cutoff],
    )
    split_strategy = "boundary_first" if strategy == "boundary_first" else "excerpt_first"
    context = CurrentLayerContext(
        depth=0,
        parent_node_ids=["root"],
        parent_titles=["Sample Document Title"],
        parent_content_pointers_by_id={"root": [parent_pointer]},
        split_strategy=split_strategy,
        metadata={"parse_strategy": strategy},
    )
    result = CurrentLayerResult(
        children=[
            LayerChildCandidate(
                node_id="root|partial",
                parent_node_id="root",
                title="Partial table",
                node_type="TABLE",
                total_content_pointers=[partial_pointer],
                expandable=False,
            )
        ],
        satisfied=True,
    )

    coverage_ok, satisfied, _overlaps, gaps, _duplicates, _notes = detect_layer_invariants(
        current_layer_context=context,
        current_layer_result=result,
        parser_source_map={source_cluster_id: {"text": text}},
    )

    assert coverage_ok is False
    assert satisfied is False
    assert gaps
    assert gaps[0].gap_start == cutoff


@pytest.mark.parametrize("strategy", ["excerpt_first", "boundary_first", "page_index"])
def test_corrective_long_table_workflow_strategy_matrix(strategy: str) -> None:
    workflow_engine, conversation_engine, knowledge_engine = build_workflow_engine_triplet(
        Path("tests") / ".tmp_corrective_matrix" / strategy,
        "in_memory",
    )
    inp = WorkflowIngestInput.from_text(
        document_id=f"corrective-workflow-{strategy}",
        text=_long_table(),
        title="Sample Document Title",
    )
    inp = inp.model_copy(
        update={
            "parse_strategy": "page_index" if strategy == "page_index" else None,
            "triage_enabled": False if strategy == "page_index" else None,
            "collections": [
                inp.collections[0].model_copy(update={"metadata": {"source_format": "markdown"}})
            ],
        }
    )

    def propose_layer(*, current_layer_context, **_kwargs) -> CurrentLayerResult:
        parent_id = current_layer_context.parent_node_ids[0]
        return CurrentLayerResult(
            children=[
                LayerChildCandidate(
                    node_id=f"{parent_id}|table",
                    parent_node_id=parent_id,
                    title="ID Description Value",
                    node_type="TABLE",
                    total_content_pointers=[_full_source_pointer(inp)],
                    expandable=False,
                )
            ],
            satisfied=True,
        )

    def review_layer(*, current_layer_context, current_layer_result, **_kwargs) -> CurrentLayerReview:
        return CurrentLayerReview(
            updated_result=current_layer_result,
            coverage_ok=True,
            satisfied=True,
            strategy_used=current_layer_context.split_strategy,
            review_notes=["deterministic corrective matrix review"],
        )

    dependencies = {
        "split_strategy": "boundary_first" if strategy == "boundary_first" else "excerpt_first",
        "propose_layer_fn": propose_layer,
        "review_layer_fn": review_layer,
        "fallback_split_strategy": "boundary_first",
        "max_review_retries": 0,
        "max_depth": 1,
    }
    run, bundle = run_ingest_workflow(
        inp=inp,
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        deps=dependencies,
    )

    assert run.status == "succeeded", run.final_state.get("workflow_errors")
    assert bundle is not None
    expected_strategy = {
        "excerpt_first": "layer_excerpt",
        "boundary_first": "layer_boundary",
        "page_index": "page_index",
    }[strategy]
    assert any(
        event["event"] == "selected" and event["strategy"] == expected_strategy
        for event in run.final_state["strategy_execution_history"]
    )
    assert any(node["label"] == "ID Description Value" for node in bundle.graph_payload["nodes"])
