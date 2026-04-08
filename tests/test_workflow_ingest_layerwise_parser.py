from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from _kogwistar_test_helpers import load_kogwistar_fake_backend
from src.workflow_ingest.models import CurrentLayerResult, CurrentLayerReview, LayerChildCandidate, WorkflowIngestInput
from src.workflow_ingest.parser_core import initialize_parse_session, prepare_layer_frontier
from src.workflow_ingest.semantics import HydratedTextPointer
from src.workflow_ingest.service import build_default_engines, run_ingest_workflow


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


def test_layerwise_workflow_retries_same_layer_then_enqueues_next_layer():
    scratch = _scratch("layer_retry")
    fake_backend = load_kogwistar_fake_backend()
    workflow_engine, conversation_engine, knowledge_engine = build_default_engines(
        scratch / "engines",
        backend_factory=fake_backend,
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


def test_layerwise_workflow_fails_when_satisfaction_retries_exhaust():
    scratch = _scratch("layer_fail")
    fake_backend = load_kogwistar_fake_backend()
    workflow_engine, conversation_engine, knowledge_engine = build_default_engines(
        scratch / "engines",
        backend_factory=fake_backend,
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


def test_layerwise_workflow_retries_when_cud_coverage_check_fails():
    scratch = _scratch("layer_coverage_retry")
    fake_backend = load_kogwistar_fake_backend()
    workflow_engine, conversation_engine, knowledge_engine = build_default_engines(
        scratch / "engines",
        backend_factory=fake_backend,
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
        if review_calls["count"] == 1:
            return CurrentLayerReview(
                updated_result=current_layer_result,
                coverage_ok=False,
                satisfied=True,
                review_notes=["coverage not yet good enough"],
            )
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


def test_legacy_compat_and_layerwise_paths_produce_equivalent_labels():
    scratch = _scratch("layer_equivalent")
    fake_backend = load_kogwistar_fake_backend()
    workflow_engine, conversation_engine, knowledge_engine = build_default_engines(
        scratch / "engines",
        backend_factory=fake_backend,
    )
    inp = WorkflowIngestInput.from_text(
        document_id="layer-equiv-doc",
        text="Alpha clause\nBeta clause\nGamma clause",
        title="Layer Equivalent Doc",
    )
    unit_id = f"{inp.request_id}|p1_t0"
    text = inp.collections[0].pages[0].units[0].text or ""

    def _full_tree(*, collection, parser_input_dict, parser_source_map):
        from src.workflow_ingest.semantics import SemanticNode

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
