from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from _kogwistar_test_helpers import build_workflow_engine_triplet
from kg_doc_parser.workflow_ingest.models import (
    CurrentLayerContext,
    CurrentLayerResult,
    CurrentLayerReview,
    LayerChildCandidate,
    ParseSessionState,
    WorkflowIngestInput,
)
from kg_doc_parser.workflow_ingest.parser_core import (
    initialize_parse_session,
    prepare_layer_frontier,
    review_layer,
    switch_split_strategy,
)
from kg_doc_parser.workflow_ingest.semantics import HydratedTextPointer
from kg_doc_parser.workflow_ingest.service import run_ingest_workflow


pytestmark = [pytest.mark.workflow]


@pytest.fixture(
    params=[
        pytest.param("in_memory", id="in_memory", marks=pytest.mark.ci),
        pytest.param("chroma", id="chroma", marks=pytest.mark.ci_full),
    ]
)
def workflow_backend_kind(request):
    return request.param


def _scratch(name: str) -> Path:
    root = Path("tests") / ".tmp_workflow_ingest_layerwise"
    path = root / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _segment_pointer(unit_id: str, text: str, fragment: str) -> HydratedTextPointer:
    start = text.index(fragment)
    end = start + len(fragment) - 1
    return HydratedTextPointer(
        source_cluster_id=unit_id,
        start_char=start,
        end_char=end,
        verbatim_text=fragment,
    )


@pytest.mark.ci
def test_initialize_parse_session_seeds_root_frontier_in_workflow_mode():
    inp = WorkflowIngestInput.from_text(
        document_id="layer-doc",
        text="Alpha clause\nBeta clause\nGamma clause",
        title="Layer Doc",
    )
    collection = inp.collections[0]
    parser_source_map = {
        "layer-doc|p1_t0": {
            "id": "layer-doc|p1_t0",
            "text": "Alpha clause\nBeta clause\nGamma clause",
            "participates_in_semantic_text": True,
        }
    }
    parser_input_dict = {"document_filename": "Layer Doc", "pages": []}

    session, frontier, root = initialize_parse_session(
        collection=collection,
        parser_input_dict=parser_input_dict,
        parser_source_map=parser_source_map,
        parse_semantic_fn=None,
    )

    assert session.mode == "workflow_layered"
    assert len(frontier) == 1
    assert frontier[0].parent_node_id == root.node_id
    assert frontier[0].depth == 0


@pytest.mark.ci
def test_initialize_parse_session_normalizes_legacy_uuid_tree():
    inp = WorkflowIngestInput.from_text(
        document_id="legacy-uuid-doc",
        text="Alpha clause\nBeta clause",
        title="Legacy UUID Doc",
    )
    legacy_root_id = uuid4()
    legacy_child_id = uuid4()
    legacy_pointer_id = "p1_c0"
    class _LegacyLikeNode:
        def __init__(
            self,
            *,
            node_id,
            parent_id,
            node_type,
            title,
            child_nodes,
            total_content_pointers,
            level_from_root,
        ):
            self.node_id = node_id
            self.parent_id = parent_id
            self.node_type = node_type
            self.title = title
            self.child_nodes = child_nodes
            self.total_content_pointers = total_content_pointers
            self.level_from_root = level_from_root

        def model_dump(self, mode: str = "json"):
            def _stringify(value):
                if hasattr(value, "hex") and getattr(value, "version", None) is not None:
                    return str(value)
                if isinstance(value, list):
                    return [_stringify(item) for item in value]
                if isinstance(value, dict):
                    return {key: _stringify(item) for key, item in value.items()}
                return value

            return {
                "node_id": _stringify(self.node_id) if mode == "json" else self.node_id,
                "parent_id": _stringify(self.parent_id) if mode == "json" else self.parent_id,
                "node_type": self.node_type,
                "title": self.title,
                "total_content_pointers": [pointer.model_dump(mode=mode) for pointer in self.total_content_pointers],
                "child_nodes": [child.model_dump(mode=mode) for child in self.child_nodes],
                "level_from_root": self.level_from_root,
            }

    legacy_tree = _LegacyLikeNode(
        node_id=legacy_root_id,
        parent_id=None,
        node_type="DOCUMENT_ROOT",
        title="Legacy UUID Doc",
        total_content_pointers=[],
        child_nodes=[
            _LegacyLikeNode(
                node_id=legacy_child_id,
                parent_id=legacy_root_id,
                node_type="TEXT_FLOW",
                title="Legacy Child",
                child_nodes=[],
                total_content_pointers=[
                    HydratedTextPointer(
                        source_cluster_id=legacy_pointer_id,
                        start_char=0,
                        end_char=11,
                        verbatim_text="Alpha clause",
                    )
                ],
                level_from_root=1,
            )
        ],
        level_from_root=0,
    )

    session, frontier, root = initialize_parse_session(
        collection=inp.collections[0],
        parser_input_dict={"document_filename": "Legacy UUID Doc", "pages": []},
        parser_source_map={
            "legacy-uuid-doc|p1_t0": {
                "id": "legacy-uuid-doc|p1_t0",
                "text": "Alpha clause\nBeta clause",
                "page_number": 1,
                "cluster_number": 0,
                "participates_in_semantic_text": True,
            }
        },
        parse_semantic_fn=lambda **kwargs: legacy_tree,
    )

    assert session.mode == "legacy_compat"
    assert isinstance(root.node_id, str)
    assert frontier[0].parent_node_id == root.node_id
    assert root.child_nodes == []
    assert session.compat_full_tree is not None
    assert session.compat_full_tree["child_nodes"][0]["total_content_pointers"][0]["source_cluster_id"] == "legacy-uuid-doc|p1_t0"


@pytest.mark.ci
def test_prepare_layer_frontier_pulls_one_bfs_depth_group():
    inp = WorkflowIngestInput.from_text(
        document_id="layer-doc",
        text="Alpha clause",
        title="Layer Doc",
    )
    session, frontier, root = initialize_parse_session(
        collection=inp.collections[0],
        parser_input_dict={"document_filename": "Layer Doc", "pages": []},
        parser_source_map={
            "layer-doc|p1_t0": {"id": "layer-doc|p1_t0", "text": "Alpha clause", "participates_in_semantic_text": True}
        },
        parse_semantic_fn=None,
    )
    frontier.extend(
        [
            frontier[0].model_copy(update={"parent_node_id": "node-a", "depth": 1, "order": 1}),
            frontier[0].model_copy(update={"parent_node_id": "node-b", "depth": 1, "order": 2}),
        ]
    )

    context, remaining, updated = prepare_layer_frontier(
        parse_session=session,
        frontier_queue=frontier,
        semantic_tree=root,
    )

    assert context.depth == 0
    assert context.parent_node_ids == [root.node_id]
    assert len(remaining) == 2
    assert updated.current_depth == 0


@pytest.mark.ci
def test_review_layer_reports_overlap_and_gap_conflicts():
    inp = WorkflowIngestInput.from_text(
        document_id="conflict-doc",
        text="Alpha clause\nBeta clause\nGamma clause",
        title="Conflict Doc",
    )
    collection = inp.collections[0]
    parser_source_map = {
        "conflict-doc|p1_t0": {
            "id": "conflict-doc|p1_t0",
            "text": "Alpha clause\nBeta clause\nGamma clause",
            "participates_in_semantic_text": True,
        }
    }
    session, frontier, root = initialize_parse_session(
        collection=collection,
        parser_input_dict={"document_filename": "Conflict Doc", "pages": []},
        parser_source_map=parser_source_map,
        parse_semantic_fn=None,
    )
    context, _, _ = prepare_layer_frontier(
        parse_session=session,
        frontier_queue=frontier,
        semantic_tree=root,
    )
    unit_id = "conflict-doc|p1_t0"
    text = parser_source_map[unit_id]["text"]
    result = CurrentLayerResult(
        children=[
            LayerChildCandidate(
                node_id="node-a",
                parent_node_id=context.parent_node_ids[0],
                title="Alpha",
                node_type="TEXT_FLOW",
                total_content_pointers=[_segment_pointer(unit_id, text, "Alpha clause\nBeta clause")],
                expandable=False,
            ),
            LayerChildCandidate(
                node_id="node-b",
                parent_node_id=context.parent_node_ids[0],
                title="Beta",
                node_type="TEXT_FLOW",
                total_content_pointers=[_segment_pointer(unit_id, text, "Beta clause")],
                expandable=False,
            ),
        ],
        satisfied=True,
        reasoning_history=[],
    )

    reviewed, _ = review_layer(
        parse_session=session,
        current_layer_context=context,
        current_layer_result=result,
        parser_source_map=parser_source_map,
    )

    assert reviewed.satisfied is False
    assert reviewed.overlap_conflicts
    assert reviewed.overlap_conflicts[0].parent_node_id == context.parent_node_ids[0]
    assert reviewed.coverage_ok is False
    assert reviewed.coverage_gap_notes


@pytest.mark.ci
def test_switch_split_strategy_updates_session_and_context():
    session = ParseSessionState(
        collection_id="doc",
        root_node_id="doc|root",
        split_strategy="excerpt_first",
        fallback_split_strategy="boundary_first",
        strategy_history=["excerpt_first"],
    )
    context = CurrentLayerContext(
        depth=0,
        parent_node_ids=["doc|root"],
        parent_titles=["Doc"],
        split_strategy="excerpt_first",
        retry_count=1,
        max_retries=1,
    )

    updated_session, updated_context = switch_split_strategy(
        parse_session=session,
        current_layer_context=context,
    )

    assert updated_session.split_strategy == "boundary_first"
    assert updated_session.strategy_history == ["excerpt_first", "boundary_first"]
    assert updated_session.strategy_switch_count == 1
    assert updated_context.split_strategy == "boundary_first"
    assert updated_context.retry_count == 0


def test_layerwise_workflow_retries_same_layer_then_enqueues_next_layer(workflow_backend_kind):
    scratch = _scratch("layer_retry")
    workflow_engine, conversation_engine, knowledge_engine = build_workflow_engine_triplet(
        scratch / "engines", workflow_backend_kind
    )
    inp = WorkflowIngestInput.from_text(
        document_id="layer-retry-doc",
        text="Alpha clause\nBeta clause\nGamma clause",
        title="Layer Retry Doc",
    )
    unit_id = f"{inp.request_id}|p1_t0"
    text = inp.collections[0].pages[0].units[0].text or ""
    review_calls = {"depth0": 0}

    def _propose_layer_fn(*, current_layer_context, **kwargs):
        if current_layer_context.depth == 0:
            return CurrentLayerResult(
                children=[
                    LayerChildCandidate(
                        node_id="node-section-a",
                        parent_node_id=current_layer_context.parent_node_ids[0],
                        title="Section A",
                        node_type="TEXT_FLOW",
                        total_content_pointers=[
                            _segment_pointer(unit_id, text, "Alpha clause\nBeta clause\n")
                        ],
                        expandable=True,
                    ),
                    LayerChildCandidate(
                        node_id="node-section-b",
                        parent_node_id=current_layer_context.parent_node_ids[0],
                        title="Section B",
                        node_type="TEXT_FLOW",
                        total_content_pointers=[_segment_pointer(unit_id, text, "Gamma clause")],
                        expandable=False,
                    ),
                ],
                satisfied=None,
                reasoning_history=[],
            )
        return CurrentLayerResult(
            children=[
                LayerChildCandidate(
                    node_id="node-alpha",
                    parent_node_id=current_layer_context.parent_node_ids[0],
                    title="Alpha",
                    node_type="TEXT_FLOW",
                    total_content_pointers=[_segment_pointer(unit_id, text, "Alpha clause")],
                    expandable=False,
                ),
                LayerChildCandidate(
                    node_id="node-beta",
                    parent_node_id=current_layer_context.parent_node_ids[0],
                    title="Beta",
                    node_type="TEXT_FLOW",
                    total_content_pointers=[_segment_pointer(unit_id, text, "Beta clause")],
                    expandable=False,
                ),
            ],
            satisfied=True,
            reasoning_history=[],
        )

    def _review_layer_fn(*, current_layer_context, current_layer_result, **kwargs):
        if current_layer_context.depth == 0 and review_calls["depth0"] == 0:
            review_calls["depth0"] += 1
            return current_layer_result.model_copy(update={"satisfied": False})
        return current_layer_result.model_copy(update={"satisfied": True})

    run, bundle = run_ingest_workflow(
        inp=inp,
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        deps={
            "propose_layer_fn": _propose_layer_fn,
            "review_layer_fn": _review_layer_fn,
        },
    )

    assert run.status == "succeeded"
    assert bundle is not None
    assert review_calls["depth0"] == 1
    assert "parse_session" in run.final_state
    assert bundle.graph_payload["nodes"]
    labels = {node["label"] for node in bundle.graph_payload["nodes"]}
    assert {"Section A", "Section B", "Alpha", "Beta"}.issubset(labels)


def test_layerwise_workflow_switches_strategy_after_overlap_retry_exhaustion(
    workflow_backend_kind,
):
    scratch = _scratch("layer_strategy_switch")
    workflow_engine, conversation_engine, knowledge_engine = build_workflow_engine_triplet(
        scratch / "engines", workflow_backend_kind
    )
    inp = WorkflowIngestInput.from_text(
        document_id="layer-strategy-doc",
        text="Alpha clause\nBeta clause\nGamma clause",
        title="Layer Strategy Doc",
    )
    unit_id = f"{inp.request_id}|p1_t0"
    text = inp.collections[0].pages[0].units[0].text or ""

    def _propose_layer_fn(*, current_layer_context, **kwargs):
        if current_layer_context.split_strategy == "excerpt_first":
            return CurrentLayerResult(
                children=[
                    LayerChildCandidate(
                        node_id="node-section-a",
                        parent_node_id=current_layer_context.parent_node_ids[0],
                        title="Section A",
                        node_type="TEXT_FLOW",
                        total_content_pointers=[_segment_pointer(unit_id, text, "Alpha clause\nBeta clause")],
                        expandable=False,
                    ),
                    LayerChildCandidate(
                        node_id="node-section-b",
                        parent_node_id=current_layer_context.parent_node_ids[0],
                        title="Section B",
                        node_type="TEXT_FLOW",
                        total_content_pointers=[_segment_pointer(unit_id, text, "Beta clause\nGamma clause")],
                        expandable=False,
                    ),
                ],
                satisfied=True,
                reasoning_history=[],
            )
        return CurrentLayerResult(
            children=[
                LayerChildCandidate(
                    node_id="node-alpha",
                    parent_node_id=current_layer_context.parent_node_ids[0],
                    title="Alpha",
                    node_type="TEXT_FLOW",
                    total_content_pointers=[_segment_pointer(unit_id, text, "Alpha clause\n")],
                    expandable=False,
                ),
                LayerChildCandidate(
                    node_id="node-beta",
                    parent_node_id=current_layer_context.parent_node_ids[0],
                    title="Beta",
                    node_type="TEXT_FLOW",
                    total_content_pointers=[_segment_pointer(unit_id, text, "Beta clause\n")],
                    expandable=False,
                ),
                LayerChildCandidate(
                    node_id="node-gamma",
                    parent_node_id=current_layer_context.parent_node_ids[0],
                    title="Gamma",
                    node_type="TEXT_FLOW",
                    total_content_pointers=[_segment_pointer(unit_id, text, "Gamma clause")],
                    expandable=False,
                ),
            ],
            satisfied=True,
            reasoning_history=[],
        )

    run, bundle = run_ingest_workflow(
        inp=inp,
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        deps={
            "propose_layer_fn": _propose_layer_fn,
            "max_review_retries": 1,
        },
    )

    assert run.status == "succeeded"
    assert bundle is not None
    assert run.final_state["parse_session"]["strategy_history"] == ["excerpt_first", "boundary_first"]
    assert run.final_state["parse_session"]["strategy_switch_count"] == 1


def test_layerwise_workflow_fails_when_satisfaction_retries_exhaust(workflow_backend_kind):
    scratch = _scratch("layer_fail")
    workflow_engine, conversation_engine, knowledge_engine = build_workflow_engine_triplet(
        scratch / "engines", workflow_backend_kind
    )
    inp = WorkflowIngestInput.from_text(
        document_id="layer-fail-doc",
        text="Alpha clause",
        title="Layer Fail Doc",
    )
    unit_id = f"{inp.request_id}|p1_t0"
    text = inp.collections[0].pages[0].units[0].text or ""

    def _propose_layer_fn(*, current_layer_context, **kwargs):
        return CurrentLayerResult(
            children=[
                LayerChildCandidate(
                    node_id="node-a",
                    parent_node_id=current_layer_context.parent_node_ids[0],
                    title="Section A",
                    node_type="TEXT_FLOW",
                    total_content_pointers=[_segment_pointer(unit_id, text, "Alpha clause")],
                    expandable=False,
                )
            ],
            satisfied=None,
            reasoning_history=[],
        )

    def _review_layer_fn(*, current_layer_result, **kwargs):
        return current_layer_result.model_copy(update={"satisfied": False})

    run, bundle = run_ingest_workflow(
        inp=inp,
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        deps={
            "propose_layer_fn": _propose_layer_fn,
            "review_layer_fn": _review_layer_fn,
            "max_review_retries": 2,
        },
    )

    assert bundle is None
    assert run.status in {"failed", "failure"}
    assert any("layer satisfaction retries exhausted" in err for err in run.final_state["workflow_errors"])


def test_layerwise_workflow_retries_when_cud_coverage_check_fails(workflow_backend_kind):
    scratch = _scratch("layer_coverage_retry")
    workflow_engine, conversation_engine, knowledge_engine = build_workflow_engine_triplet(
        scratch / "engines", workflow_backend_kind
    )
    inp = WorkflowIngestInput.from_text(
        document_id="layer-coverage-doc",
        text="Alpha clause\nBeta clause",
        title="Layer Coverage Doc",
    )
    unit_id = f"{inp.request_id}|p1_t0"
    text = inp.collections[0].pages[0].units[0].text or ""
    review_calls = {"count": 0}

    def _propose_layer_fn(*, current_layer_context, **kwargs):
        if review_calls["count"] == 0:
            return CurrentLayerResult(
                children=[
                    LayerChildCandidate(
                        node_id="node-section-a",
                        parent_node_id=current_layer_context.parent_node_ids[0],
                        title="Section A",
                        node_type="TEXT_FLOW",
                        total_content_pointers=[_segment_pointer(unit_id, text, "Alpha clause")],
                        expandable=False,
                    ),
                ],
                satisfied=True,
                reasoning_history=[],
            )
        return CurrentLayerResult(
            children=[
                LayerChildCandidate(
                    node_id="node-section-a",
                    parent_node_id=current_layer_context.parent_node_ids[0],
                    title="Section A",
                    node_type="TEXT_FLOW",
                    total_content_pointers=[
                        _segment_pointer(unit_id, text, "Alpha clause\nBeta clause")
                    ],
                    expandable=False,
                )
            ],
            satisfied=True,
            reasoning_history=[],
        )

    def _review_layer_fn(*, current_layer_result, **kwargs):
        review_calls["count"] += 1
        return CurrentLayerReview(
            updated_result=current_layer_result,
            coverage_ok=True,
            satisfied=True,
        )

    run, bundle = run_ingest_workflow(
        inp=inp,
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        deps={
            "propose_layer_fn": _propose_layer_fn,
            "review_layer_fn": _review_layer_fn,
            "coverage_threshold": 0.95,
        },
    )

    assert run.status == "succeeded"
    assert bundle is not None
    assert review_calls["count"] == 2


def test_legacy_compat_and_layerwise_paths_produce_equivalent_labels(workflow_backend_kind):
    scratch = _scratch("layer_equivalent")
    workflow_engine, conversation_engine, knowledge_engine = build_workflow_engine_triplet(
        scratch / "engines", workflow_backend_kind
    )
    inp = WorkflowIngestInput.from_text(
        document_id="layer-equiv-doc",
        text="Alpha clause\nBeta clause\nGamma clause",
        title="Layer Equivalent Doc",
    )
    unit_id = f"{inp.request_id}|p1_t0"
    text = inp.collections[0].pages[0].units[0].text or ""

    def _full_tree(*, collection, parser_input_dict, parser_source_map):
        from kg_doc_parser.workflow_ingest.semantics import SemanticNode

        root = SemanticNode(
            node_id=f"{collection.collection_id}|root",
            title=collection.title,
            node_type="DOCUMENT_ROOT",
            total_content_pointers=[],
            child_nodes=[],
            level_from_root=0,
        )
        section_a = SemanticNode(
            node_id="node-section-a",
            parent_id=root.node_id,
            title="Section A",
            node_type="TEXT_FLOW",
            total_content_pointers=[_segment_pointer(unit_id, text, "Alpha clause\nBeta clause\n")],
            child_nodes=[
                SemanticNode(
                    node_id="node-alpha",
                    parent_id="node-section-a",
                    title="Alpha",
                    node_type="TEXT_FLOW",
                    total_content_pointers=[_segment_pointer(unit_id, text, "Alpha clause")],
                    child_nodes=[],
                    level_from_root=2,
                ),
                SemanticNode(
                    node_id="node-beta",
                    parent_id="node-section-a",
                    title="Beta",
                    node_type="TEXT_FLOW",
                    total_content_pointers=[_segment_pointer(unit_id, text, "Beta clause")],
                    child_nodes=[],
                    level_from_root=2,
                ),
            ],
            level_from_root=1,
        )
        section_b = SemanticNode(
            node_id="node-section-b",
            parent_id=root.node_id,
            title="Section B",
            node_type="TEXT_FLOW",
            total_content_pointers=[_segment_pointer(unit_id, text, "Gamma clause")],
            child_nodes=[],
            level_from_root=1,
        )
        root.child_nodes.extend([section_a, section_b])
        return root

    def _propose_layer_fn(*, current_layer_context, **kwargs):
        if current_layer_context.depth == 0:
            return CurrentLayerResult(
                children=[
                    LayerChildCandidate(
                        node_id="node-section-a",
                        parent_node_id=current_layer_context.parent_node_ids[0],
                        title="Section A",
                        node_type="TEXT_FLOW",
                        total_content_pointers=[
                            _segment_pointer(unit_id, text, "Alpha clause\nBeta clause\n")
                        ],
                        expandable=True,
                    ),
                    LayerChildCandidate(
                        node_id="node-section-b",
                        parent_node_id=current_layer_context.parent_node_ids[0],
                        title="Section B",
                        node_type="TEXT_FLOW",
                        total_content_pointers=[_segment_pointer(unit_id, text, "Gamma clause")],
                        expandable=False,
                    ),
                ],
                satisfied=True,
                reasoning_history=[],
            )
        return CurrentLayerResult(
            children=[
                LayerChildCandidate(
                    node_id="node-alpha",
                    parent_node_id=current_layer_context.parent_node_ids[0],
                    title="Alpha",
                    node_type="TEXT_FLOW",
                    total_content_pointers=[_segment_pointer(unit_id, text, "Alpha clause")],
                    expandable=False,
                ),
                LayerChildCandidate(
                    node_id="node-beta",
                    parent_node_id=current_layer_context.parent_node_ids[0],
                    title="Beta",
                    node_type="TEXT_FLOW",
                    total_content_pointers=[_segment_pointer(unit_id, text, "Beta clause")],
                    expandable=False,
                ),
            ],
            satisfied=True,
            reasoning_history=[],
        )

    compat_run, compat_bundle = run_ingest_workflow(
        inp=inp,
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        deps={"parse_semantic_fn": _full_tree},
    )
    layer_run, layer_bundle = run_ingest_workflow(
        inp=inp,
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        deps={"propose_layer_fn": _propose_layer_fn},
    )

    assert compat_run.status == "succeeded"
    assert layer_run.status == "succeeded"
    assert compat_bundle is not None
    assert layer_bundle is not None
    compat_labels = {node["label"] for node in compat_bundle.graph_payload["nodes"]}
    layer_labels = {node["label"] for node in layer_bundle.graph_payload["nodes"]}
    assert compat_labels == layer_labels
