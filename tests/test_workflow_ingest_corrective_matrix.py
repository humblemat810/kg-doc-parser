from __future__ import annotations

from pathlib import Path

import kg_doc_parser.workflow_ingest.handlers as handlers_module
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
    assert run.final_state["validation_report"]["terminal_coverage_status"] == "complete"


@pytest.mark.parametrize("strategy", ["excerpt_first", "boundary_first", "page_index"])
def test_durable_batch_commits_valid_parent_and_requeues_failed_parent(
    strategy: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mixed frontier batch must not discard a valid sibling on repair."""
    workflow_engine, conversation_engine, knowledge_engine = build_workflow_engine_triplet(
        Path("tests") / ".tmp_corrective_matrix" / "mixed_parent_batch",
        "in_memory",
    )
    text = "Alpha detail\nBeta detail"
    inp = WorkflowIngestInput.from_text(
        document_id="corrective-mixed-parent-batch",
        text=text,
        title="Mixed Parent Batch",
    )
    inp = inp.model_copy(
        update={
            "parse_strategy": "page_index" if strategy == "page_index" else None,
            "triage_enabled": False if strategy == "page_index" else None,
        }
    )
    source_cluster_id = f"{inp.request_id}|p1_t0"
    newline = text.index("\n")
    alpha_start, alpha_end = 0, newline - 1
    beta_start, beta_end = newline + 1, len(text) - 1
    root_id = f"{inp.request_id}|root"
    alpha_id = f"{root_id}|alpha"
    beta_id = f"{root_id}|beta"
    calls: list[tuple[str, ...]] = []

    def pointer(start: int, end: int) -> HydratedTextPointer:
        return HydratedTextPointer(
            source_cluster_id=source_cluster_id,
            start_char=start,
            end_char=end,
            verbatim_text=text[start : end + 1],
        )

    def child(*, node_id: str, parent_node_id: str, start: int, end: int) -> LayerChildCandidate:
        return LayerChildCandidate(
            node_id=node_id,
            parent_node_id=parent_node_id,
            title=node_id.rsplit("|", 1)[-1],
            node_type="TEXT_FLOW",
            total_content_pointers=[pointer(start, end)],
            expandable=False,
        )

    def propose_layer(*, current_layer_context, **_kwargs) -> CurrentLayerResult:
        parent_ids = tuple(current_layer_context.parent_node_ids)
        calls.append(parent_ids)
        if current_layer_context.depth == 0:
            return CurrentLayerResult(
                children=[
                    LayerChildCandidate(
                        node_id=alpha_id,
                        parent_node_id=root_id,
                        title="alpha",
                        node_type="SECTION",
                        total_content_pointers=[pointer(alpha_start, alpha_end)],
                        expandable=True,
                    ),
                    LayerChildCandidate(
                        node_id=beta_id,
                        parent_node_id=root_id,
                        title="beta",
                        node_type="SECTION",
                        total_content_pointers=[pointer(beta_start, beta_end)],
                        expandable=True,
                    ),
                ],
                satisfied=True,
            )
        if set(parent_ids) == {alpha_id, beta_id}:
            return CurrentLayerResult(
                children=[child(node_id=f"{alpha_id}|leaf", parent_node_id=alpha_id, start=alpha_start, end=alpha_end)],
                satisfied=True,
            )
        assert parent_ids == (beta_id,)
        return CurrentLayerResult(
            children=[child(node_id=f"{beta_id}|leaf", parent_node_id=beta_id, start=beta_start, end=beta_end)],
            satisfied=True,
        )

    if strategy == "page_index":
        page_index_calls: dict[str, int] = {}

        def page_index_layer(*, parent_id, **_kwargs) -> list[LayerChildCandidate]:
            page_index_calls[parent_id] = page_index_calls.get(parent_id, 0) + 1
            if parent_id == root_id:
                return [
                    LayerChildCandidate(
                        node_id=alpha_id,
                        parent_node_id=root_id,
                        title="alpha",
                        node_type="SECTION",
                        total_content_pointers=[pointer(alpha_start, alpha_end)],
                        expandable=True,
                    ),
                    LayerChildCandidate(
                        node_id=beta_id,
                        parent_node_id=root_id,
                        title="beta",
                        node_type="SECTION",
                        total_content_pointers=[pointer(beta_start, beta_end)],
                        expandable=True,
                    ),
                ]
            if parent_id == beta_id and page_index_calls[parent_id] == 1:
                return []
            if parent_id == alpha_id:
                return [child(node_id=f"{alpha_id}|leaf", parent_node_id=alpha_id, start=alpha_start, end=alpha_end)]
            assert parent_id == beta_id
            return [child(node_id=f"{beta_id}|leaf", parent_node_id=beta_id, start=beta_start, end=beta_end)]

        monkeypatch.setattr(handlers_module, "parse_page_index_layer", page_index_layer)

    def review_layer(*, current_layer_context, current_layer_result, **_kwargs) -> CurrentLayerReview:
        return CurrentLayerReview(
            updated_result=current_layer_result,
            coverage_ok=True,
            satisfied=True,
            strategy_used=current_layer_context.split_strategy,
            review_notes=["deterministic mixed-parent batch review"],
        )

    run, bundle = run_ingest_workflow(
        inp=inp,
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        deps={
            "layer_frontier_batch_size": 2,
            "split_strategy": "boundary_first" if strategy == "boundary_first" else "excerpt_first",
            "propose_layer_fn": propose_layer,
            "review_layer_fn": review_layer,
            "max_review_retries": 0,
            "max_depth": 2,
        },
    )

    assert run.status == "succeeded", run.final_state.get("workflow_errors")
    assert bundle is not None
    if strategy == "page_index":
        assert page_index_calls[beta_id] == 2
    else:
        assert any(set(call) == {alpha_id, beta_id} for call in calls)
        assert (beta_id,) in calls
    node_ids = {node["id"] for node in bundle.graph_payload["nodes"]}
    assert {f"{alpha_id}|leaf", f"{beta_id}|leaf"} <= node_ids
    assert run.final_state["layer_frontier_queue"] == []
    assert run.final_state["validation_report"]["terminal_coverage_status"] == "complete"


def test_durable_empty_provider_is_retried_before_atomic_page_index_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow_engine, conversation_engine, knowledge_engine = build_workflow_engine_triplet(
        Path("tests") / ".tmp_corrective_matrix" / "invalid_zero_child",
        "in_memory",
    )
    inp = WorkflowIngestInput.from_text(
        document_id="corrective-invalid-zero-child",
        text="This source must not disappear.",
        title="Invalid Zero Child",
    )

    def propose_layer(*, current_layer_context, **_kwargs) -> CurrentLayerResult:
        return CurrentLayerResult(
            children=[],
            satisfied=True,
            metadata={"strategy": current_layer_context.split_strategy},
        )

    def review_layer(*, current_layer_result, **_kwargs) -> CurrentLayerReview:
        return CurrentLayerReview(
            updated_result=current_layer_result,
            coverage_ok=True,
            satisfied=True,
            review_notes=["fake reviewer cannot authorize empty refinement"],
        )

    # The normal cascade is allowed to reach deterministic PageIndex as its
    # final fallback.  This test isolates the durable failure contract by
    # making every selected strategy return the same invalid empty result.
    monkeypatch.setattr(handlers_module, "parse_page_index_layer", lambda **_kwargs: [])

    run, bundle = run_ingest_workflow(
        inp=inp,
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        deps={
            "propose_layer_fn": propose_layer,
            "review_layer_fn": review_layer,
            "max_review_retries": 0,
            "max_depth": 1,
        },
    )

    assert run.status == "succeeded", run.final_state.get("workflow_errors")
    assert bundle is not None
    assert run.final_state["validation_report"]["terminal_coverage_status"] == "atomic_valid"
    history = run.final_state["strategy_execution_history"]
    assert any(event["event"] == "failed" for event in history)
    assert any(
        event["event"] == "selected" and event["strategy"] == "page_index"
        for event in history
    )


def test_durable_invalid_pointer_refinement_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    workflow_engine, conversation_engine, knowledge_engine = build_workflow_engine_triplet(
        Path("tests") / ".tmp_corrective_matrix" / "invalid_pointer_refinement",
        "in_memory",
    )
    inp = WorkflowIngestInput.from_text(
        document_id="corrective-invalid-pointer",
        text="This source must not disappear.",
        title="Invalid Pointer",
    )

    def propose_layer(*, current_layer_context, **_kwargs) -> CurrentLayerResult:
        return CurrentLayerResult(
            children=[],
            satisfied=True,
            metadata={"strategy": current_layer_context.split_strategy},
        )

    def review_layer(*, current_layer_result, **_kwargs) -> CurrentLayerReview:
        return CurrentLayerReview(
            updated_result=current_layer_result,
            coverage_ok=True,
            satisfied=True,
            review_notes=["fake reviewer cannot authorize empty refinement"],
        )

    def invalid_page_index_layer(*, parent_id, **_kwargs) -> list[LayerChildCandidate]:
        return [
            LayerChildCandidate(
                node_id=f"{parent_id}|invalid",
                parent_node_id=parent_id,
                title="invalid pointer",
                node_type="TEXT_FLOW",
                total_content_pointers=[],
                expandable=False,
            )
        ]

    monkeypatch.setattr(handlers_module, "parse_page_index_layer", invalid_page_index_layer)

    run, bundle = run_ingest_workflow(
        inp=inp,
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        deps={
            "propose_layer_fn": propose_layer,
            "review_layer_fn": review_layer,
            "max_review_retries": 0,
            "max_depth": 1,
        },
    )

    assert run.status in {"failed", "failure"}
    assert bundle is None
    errors = "\n".join(str(item) for item in run.final_state.get("workflow_errors", []))
    assert "all configured strategies are exhausted" in errors


def test_durable_duplicate_text_at_distinct_occurrences_is_preserved() -> None:
    workflow_engine, conversation_engine, knowledge_engine = build_workflow_engine_triplet(
        Path("tests") / ".tmp_corrective_matrix" / "duplicate_occurrences",
        "in_memory",
    )
    text = "Shared content\nShared content"
    inp = WorkflowIngestInput.from_text(
        document_id="corrective-duplicate-occurrences",
        text=text,
        title="Duplicate Occurrences",
    )
    source_cluster_id = f"{inp.request_id}|p1_t0"
    first_end = text.index("\n") - 1
    second_start = first_end + 2

    def pointer(start: int, end: int) -> HydratedTextPointer:
        return HydratedTextPointer(
            source_cluster_id=source_cluster_id,
            start_char=start,
            end_char=end,
            verbatim_text=text[start : end + 1],
        )

    def propose_layer(*, current_layer_context, **_kwargs) -> CurrentLayerResult:
        parent_id = current_layer_context.parent_node_ids[0]
        return CurrentLayerResult(
            children=[
                LayerChildCandidate(
                    node_id=f"{parent_id}|first",
                    parent_node_id=parent_id,
                    title="Shared content",
                    node_type="TEXT_FLOW",
                    total_content_pointers=[pointer(0, first_end)],
                    expandable=False,
                ),
                LayerChildCandidate(
                    node_id=f"{parent_id}|second",
                    parent_node_id=parent_id,
                    title="Shared content",
                    node_type="TEXT_FLOW",
                    total_content_pointers=[pointer(second_start, len(text) - 1)],
                    expandable=False,
                ),
            ],
            satisfied=True,
        )

    def review_layer(*, current_layer_result, **_kwargs) -> CurrentLayerReview:
        return CurrentLayerReview(
            updated_result=current_layer_result,
            coverage_ok=True,
            satisfied=True,
            review_notes=["deterministic duplicate-occurrence review"],
        )

    run, bundle = run_ingest_workflow(
        inp=inp,
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        deps={
            "propose_layer_fn": propose_layer,
            "review_layer_fn": review_layer,
            "max_review_retries": 0,
            "max_depth": 1,
        },
    )

    assert run.status == "succeeded", run.final_state.get("workflow_errors")
    assert bundle is not None
    shared_nodes = [
        node for node in bundle.graph_payload["nodes"] if node["label"] == "Shared content"
    ]
    assert len(shared_nodes) == 2
    assert {
        tuple(pointer_item["start_char"] for pointer_item in node["mentions"][0]["spans"])
        for node in shared_nodes
    } == {(0,), (second_start,)}
