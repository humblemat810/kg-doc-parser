"""Durable OCR graph publication and checkpoint-recovery integration tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from _kogwistar_test_helpers import (
    build_workflow_engine_triplet,
    drain_phase1_indexes_until_idle,
)
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from ocr_workflow_support import (
    draw_test_image as _draw_test_image,
)
from ocr_workflow_support import (
    fake_ocr_response as _fake_ocr_response,
)
from ocr_workflow_support import (
    scratch_path as _scratch,
)

from kg_doc_parser.models import (
    NonTextCluster,
    OCRClusterResponse,
    TextCluster,
)
from kg_doc_parser.workflow_ingest import (
    EmbeddingProviderConfig,
    OCRImagePayload,
    ProviderEndpointConfig,
    WorkflowProviderSettings,
    run_ocr_ingest_workflow,
)
from kg_doc_parser.workflow_ingest.document_grouping import GroupProposal
from kg_doc_parser.workflow_ingest.document_models import DocumentTree
from kg_doc_parser.workflow_ingest.models import (
    CurrentLayerResult,
    CurrentLayerReview,
    LayerChildCandidate,
)
from kg_doc_parser.workflow_ingest.ocr_pipeline import OCRQualityPolicy
from kg_doc_parser.workflow_ingest.ocr_state_store import OCRWorkflowStateStore
from kg_doc_parser.workflow_ingest.semantics import HydratedTextPointer, SemanticNode
from kg_doc_parser.workflow_ingest.service import _TinyEmbeddingFunction

pytestmark = [pytest.mark.workflow]


def _terminate_interrupted_worker(process: subprocess.Popen[str]) -> None:
    if os.name == "posix":
        import signal

        os.kill(process.pid, signal.SIGKILL)
    else:
        process.terminate()


def _grounded_semantic_tree(*, collection, parser_input_dict, parser_source_map):
    del parser_input_dict
    root = SemanticNode(
        title=collection.title,
        node_type="DOCUMENT_ROOT",
        total_content_pointers=[],
        child_nodes=[],
        level_from_root=0,
    )
    for unit_id, record in parser_source_map.items():
        if not record.get("participates_in_semantic_text", True):
            continue
        text = record["text"]
        root.child_nodes.append(
            SemanticNode(
                title=f"section:{unit_id}",
                node_type="TEXT_FLOW",
                total_content_pointers=[
                    HydratedTextPointer(
                        source_cluster_id=unit_id,
                        start_char=0,
                        end_char=max(0, len(text) - 1),
                        verbatim_text=text,
                    )
                ],
                child_nodes=[],
                level_from_root=1,
                parent_id=root.node_id,
            )
        )
    return root

@pytest.mark.ci_full
def test_ocr_root_graph_publication_survives_durable_engine_reopen() -> None:
    pytest.importorskip("chromadb")
    scratch = _scratch("ocr_durable_graph_reopen")
    image_path = scratch / "images" / "page_1.png"
    _draw_test_image(image_path, lines=["durable OCR alpha", "durable OCR beta"])
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="durable-ocr"),
        parser=ProviderEndpointConfig(provider="fake", model="durable-parser"),
        embedding=EmbeddingProviderConfig(
            provider="fake", model="durable-embedding", dimension=2
        ),
    )
    engine_dir = scratch / "engines"
    workflow_engine, conversation_engine, unused_memory_knowledge = (
        build_workflow_engine_triplet(engine_dir, "in_memory")
    )
    knowledge_engine = GraphKnowledgeEngine(
        persist_directory=str(engine_dir / "knowledge"),
        kg_graph_type="knowledge",
        embedding_function=_TinyEmbeddingFunction(),
    )
    engines = (workflow_engine, conversation_engine, knowledge_engine)

    def _runner(
        image_path: Path,
        page_number: int,
        provider_settings: WorkflowProviderSettings,
    ) -> OCRClusterResponse:
        del image_path, provider_settings
        return _fake_ocr_response(page_number, "durable OCR alpha").model_copy(
            update={
                "OCR_text_clusters": [
                    TextCluster(
                        text="durable OCR alpha",
                        bb_x_min=55.0,
                        bb_x_max=300.0,
                        bb_y_min=55.0,
                        bb_y_max=80.0,
                        cluster_number=0,
                    ),
                    TextCluster(
                        text="durable OCR beta",
                        bb_x_min=55.0,
                        bb_x_max=300.0,
                        bb_y_min=135.0,
                        bb_y_max=160.0,
                        cluster_number=1,
                    ),
                ],
                "meaningful_ordering": [0, 1],
            }
        )

    try:
        run, bundle, _artifacts = run_ocr_ingest_workflow(
            document_id="ocr-durable-graph-reopen",
            title="Durable OCR graph reopen",
            output_dir=scratch / "artifacts",
            image_payloads=[OCRImagePayload(page_number=1, image_path=str(image_path))],
            provider_settings=settings,
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            knowledge_engine=knowledge_engine,
            ocr_runner=_runner,
            deps={"parse_semantic_fn": _grounded_semantic_tree},
            run_id="ocr-durable-graph-reopen-run",
        )
        assert run.status == "succeeded"
        assert bundle is not None
        drain_phase1_indexes_until_idle(*engines)
        expected_node_ids = {
            str(record["id"])
            for record in bundle.graph_payload["nodes"]
            if isinstance(record, dict) and isinstance(record.get("id"), str)
        }
        expected_edge_ids = {
            str(record["id"])
            for record in bundle.graph_payload["edges"]
            if isinstance(record, dict) and isinstance(record.get("id"), str)
        }
        assert expected_node_ids
        assert expected_edge_ids
        assert len(expected_node_ids) == len(bundle.graph_payload["nodes"])
        assert len(expected_edge_ids) == len(bundle.graph_payload["edges"])
        assert all(
            knowledge_engine.persist.exists_node(node_id)
            for node_id in expected_node_ids
        )
        assert all(
            knowledge_engine.persist.exists_edge(edge_id)
            for edge_id in expected_edge_ids
        )
    finally:
        for engine in engines:
            engine.close()
        unused_memory_knowledge.close()

    reopened_knowledge = GraphKnowledgeEngine(
        persist_directory=str(engine_dir / "knowledge"),
        kg_graph_type="knowledge",
        embedding_function=_TinyEmbeddingFunction(),
    )
    try:
        assert all(
            reopened_knowledge.persist.exists_node(node_id)
            for node_id in expected_node_ids
        )
        assert all(
            reopened_knowledge.persist.exists_edge(edge_id)
            for edge_id in expected_edge_ids
        )
        stored_node_ids = {
            node.id for node in reopened_knowledge.read.get_nodes(limit=1000)
        }
        stored_edge_ids = {
            edge.id for edge in reopened_knowledge.read.get_edges(limit=1000)
        }
        assert stored_node_ids == expected_node_ids
        assert stored_edge_ids == expected_edge_ids
    finally:
        reopened_knowledge.close()


@pytest.mark.ci_full
def test_ocr_root_fresh_process_resume_publishes_durable_graph() -> None:
    """Durable fresh-process recovery matches an uninterrupted canonical run."""
    pytest.importorskip("chromadb")
    scratch = _scratch("ocr_fresh_resume_durable_graph")
    for page_number in (1, 2):
        _draw_test_image(
            scratch / "source" / f"page-{page_number}.png",
            lines=[f"durable resumed page {page_number}"],
        )
    worker = Path(__file__).with_name("_ocr_root_fresh_process_worker.py")
    repo_root = Path(__file__).resolve().parents[1]
    child_env = dict(os.environ)
    child_env["OCR_TEST_LEASE_MS"] = "1000"
    child_env["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(repo_root), child_env.get("PYTHONPATH", "")) if path
    )

    interrupted = subprocess.Popen(
        [sys.executable, str(worker), str(scratch), "durable-hold"],
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    marker = scratch / "page-2-started"
    deadline = time.monotonic() + 90.0
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert marker.exists(), "worker did not reach page two before termination"
    _terminate_interrupted_worker(interrupted)
    interrupted.wait(timeout=30)
    assert interrupted.returncode != 0

    resumed = subprocess.run(
        [sys.executable, str(worker), str(scratch), "durable-resume"],
        capture_output=True,
        text=True,
        check=False,
        env=child_env,
        timeout=120,
    )
    assert resumed.returncode == 0, resumed.stderr
    result = json.loads((scratch / "resume-result.json").read_text(encoding="utf-8"))
    assert result == {
        "status": "succeeded",
        "run_id": "ocr|fresh-root-document",
        "calls": [2],
        "has_grouping_tree": True,
    }

    publication = json.loads(
        (scratch / "durable-resume-lineage.json").read_text(encoding="utf-8")
    )
    expected_node_ids = set(publication["graph_node_ids"])
    expected_edge_ids = set(publication["graph_edge_ids"])
    assert expected_node_ids
    assert expected_edge_ids
    assert len(publication["graph_node_ids"]) == len(expected_node_ids)
    assert len(publication["graph_edge_ids"]) == len(expected_edge_ids)
    assert set(publication["persisted_graph_node_ids"]) == expected_node_ids
    assert set(publication["persisted_graph_edge_ids"]) == expected_edge_ids
    lineage = publication["lineage"]
    assert len(lineage) == 3
    assert {entry["workflow_id"] for entry in lineage} == {
        "kg_doc_parser.ocr_pages.v1",
        "kg_doc_parser.ocr_grouping.v1",
        "kg_doc_parser.ingest.v1",
    }
    assert {entry["parent_run_id"] for entry in lineage} == {"ocr|fresh-root-document"}
    assert len({entry["child_run_id"] for entry in lineage}) == 3
    assert len({entry["id"] for entry in lineage}) == 3

    reopened_knowledge = GraphKnowledgeEngine(
        persist_directory=str(scratch / "engines" / "durable-knowledge"),
        kg_graph_type="knowledge",
        embedding_function=_TinyEmbeddingFunction(),
    )
    try:
        assert {
            node.id for node in reopened_knowledge.read.get_nodes(limit=1000)
        } == expected_node_ids
        assert {
            edge.id for edge in reopened_knowledge.read.get_edges(limit=1000)
        } == expected_edge_ids
    finally:
        reopened_knowledge.close()

    reference = scratch / "uninterrupted-reference"
    for page_number in (1, 2):
        _draw_test_image(
            reference / "source" / f"page-{page_number}.png",
            lines=[f"durable resumed page {page_number}"],
        )
    baseline = subprocess.run(
        [sys.executable, str(worker), str(reference), "durable-reference"],
        capture_output=True,
        text=True,
        check=False,
        env=child_env,
        timeout=120,
    )
    assert baseline.returncode == 0, baseline.stderr
    reference_publication = json.loads(
        (reference / "durable-reference-lineage.json").read_text(encoding="utf-8")
    )
    assert set(reference_publication["graph_node_ids"]) == expected_node_ids
    assert set(reference_publication["graph_edge_ids"]) == expected_edge_ids
    assert (
        reference_publication["graph_payload_sha256"]
        == publication["graph_payload_sha256"]
    )
    assert (
        reference_publication["grouping_tree_sha256"]
        == publication["grouping_tree_sha256"]
    )


@pytest.mark.ci_full
def test_ocr_summary_is_not_regenerated_after_fresh_process_resume() -> None:
    """A completed grounded summary survives a crash before semantic publication."""
    pytest.importorskip("chromadb")
    scratch = _scratch("ocr_summary_fresh_resume_durable_graph")
    for page_number in (1, 2):
        _draw_test_image(
            scratch / "source" / f"page-{page_number}.png",
            lines=[f"summary resume page {page_number}"],
        )
    worker = Path(__file__).with_name("_ocr_root_fresh_process_worker.py")
    repo_root = Path(__file__).resolve().parents[1]
    child_env = dict(os.environ)
    child_env["OCR_TEST_LEASE_MS"] = "1000"
    child_env["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(repo_root), child_env.get("PYTHONPATH", "")) if path
    )

    interrupted_ocr = subprocess.Popen(
        [sys.executable, str(worker), str(scratch), "durable-summary-hold"],
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    page_marker = scratch / "page-2-started"
    deadline = time.monotonic() + 90.0
    while not page_marker.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    if not page_marker.exists():
        _terminate_interrupted_worker(interrupted_ocr)
        stdout, stderr = interrupted_ocr.communicate(timeout=30)
        pytest.fail(
            "worker did not reach page two before termination\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    _terminate_interrupted_worker(interrupted_ocr)
    interrupted_ocr.wait(timeout=30)
    assert interrupted_ocr.returncode != 0

    first_resume = subprocess.Popen(
        [sys.executable, str(worker), str(scratch), "durable-summary-publish-hold"],
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    summary_marker = scratch / "grounded-summary-ready"
    deadline = time.monotonic() + 120.0
    while not summary_marker.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    if not summary_marker.exists():
        _terminate_interrupted_worker(first_resume)
        stdout, stderr = first_resume.communicate(timeout=30)
        pytest.fail(
            "semantic parser did not receive the accepted summary\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    _terminate_interrupted_worker(first_resume)
    first_resume.wait(timeout=30)
    assert first_resume.returncode != 0
    assert (scratch / "summary-calls.txt").read_text(encoding="utf-8").splitlines() == [
        "called"
    ]

    resumed = subprocess.run(
        [sys.executable, str(worker), str(scratch), "durable-summary-resume"],
        capture_output=True,
        text=True,
        check=False,
        env=child_env,
        timeout=120,
    )
    assert resumed.returncode == 0, resumed.stderr
    result = json.loads((scratch / "resume-result.json").read_text(encoding="utf-8"))
    assert result == {
        "status": "succeeded",
        "run_id": "ocr|fresh-root-document",
        "calls": [],
        "has_grouping_tree": True,
        "grouping_summary_count": 1,
    }
    assert (scratch / "summary-calls.txt").read_text(encoding="utf-8").splitlines() == [
        "called"
    ]
    publication = json.loads(
        (scratch / "durable-summary-resume-lineage.json").read_text(encoding="utf-8")
    )
    expected_node_ids = set(publication["graph_node_ids"])
    expected_edge_ids = set(publication["graph_edge_ids"])
    assert expected_node_ids
    assert expected_edge_ids
    assert set(publication["persisted_graph_node_ids"]) == expected_node_ids
    assert set(publication["persisted_graph_edge_ids"]) == expected_edge_ids
    reopened_knowledge = GraphKnowledgeEngine(
        persist_directory=str(scratch / "engines" / "durable-knowledge"),
        kg_graph_type="knowledge",
        embedding_function=_TinyEmbeddingFunction(),
    )
    try:
        assert {
            node.id for node in reopened_knowledge.read.get_nodes(limit=1000)
        } == expected_node_ids
        assert {
            edge.id for edge in reopened_knowledge.read.get_edges(limit=1000)
        } == expected_edge_ids
    finally:
        reopened_knowledge.close()


@pytest.mark.ci
def test_ocr_grouping_and_semantic_graph_preserve_mixed_reading_order() -> None:
    scratch = _scratch("ocr_multimodal_grouping_export_order")
    image_path = scratch / "images" / "page_1.png"
    _draw_test_image(image_path, lines=["Text A", "Caption", "Text B"])
    document_id = "ocr-multimodal-order-doc"
    source_order = (
        ("text-a", f"{document_id}|p1_t0"),
        ("figure", f"{document_id}|p1_i1"),
        ("caption", f"{document_id}|p1_t2"),
        ("text-b", f"{document_id}|p1_t3"),
    )
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        parser=ProviderEndpointConfig(provider="fake", model="fake-parser"),
        embedding=EmbeddingProviderConfig(
            provider="fake", model="fake-embed", dimension=1
        ),
        parse_strategy="layer_excerpt",
        triage_enabled=False,
    )

    def _mixed_ocr_runner(
        image_path: Path,
        page_number: int,
        provider_settings: WorkflowProviderSettings,
    ) -> OCRClusterResponse:
        del image_path, provider_settings
        response = _fake_ocr_response(page_number, "Text A")
        return response.model_copy(
            update={
                "OCR_text_clusters": [
                    TextCluster(
                        text="Text A",
                        bb_x_min=10.0,
                        bb_x_max=500.0,
                        bb_y_min=10.0,
                        bb_y_max=30.0,
                        cluster_number=0,
                    ),
                    TextCluster(
                        text="Caption",
                        bb_x_min=10.0,
                        bb_x_max=500.0,
                        bb_y_min=40.0,
                        bb_y_max=60.0,
                        cluster_number=2,
                    ),
                    TextCluster(
                        text="Text B",
                        bb_x_min=10.0,
                        bb_x_max=500.0,
                        bb_y_min=70.0,
                        bb_y_max=90.0,
                        cluster_number=3,
                    ),
                ],
                "non_text_objects": [
                    NonTextCluster(
                        description="Figure 1",
                        bb_x_min=10.0,
                        bb_x_max=500.0,
                        bb_y_min=30.0,
                        bb_y_max=40.0,
                        cluster_number=1,
                    )
                ],
                "meaningful_ordering": [0, 1, 2, 3],
                "source_item_count": 4,
            }
        )

    workflow_engine, conversation_engine, knowledge_engine = (
        build_workflow_engine_triplet(scratch / "engines", "in_memory")
    )
    semantic_source_orders: list[tuple[str, ...]] = []

    def _group_aware_proposal_fn(
        *, current_layer_context, parser_input_dict, parser_source_map, **kwargs
    ) -> CurrentLayerResult:
        del kwargs
        raw_tree = parser_input_dict.get("document_tree")
        assert isinstance(raw_tree, dict)
        tree = DocumentTree.model_validate(raw_tree)
        nodes_by_id = {node.node_id: node for node in tree.nodes}
        root = nodes_by_id[tree.root_id]
        assert [child.node_id for child in root.children] == [
            next(node.node_id for node in tree.nodes if node.kind == "section"),
            f"leaf:{source_order[2][1]}",
            f"leaf:{source_order[3][1]}",
        ]
        group = next(node for node in tree.nodes if node.kind == "section")
        assert [child.node_id for child in group.children] == [
            f"leaf:{source_order[0][1]}",
            f"leaf:{source_order[1][1]}",
        ]
        group_id = group.node_id
        grouped_item_ids = tuple(
            nodes_by_id[child.node_id].parts[0].item_id for child in group.children
        )
        ordered_item_ids = (
            *grouped_item_ids,
            source_order[2][1],
            source_order[3][1],
        )
        semantic_source_orders.append(ordered_item_ids)

        def _candidate(item_id: str, parent_id: str) -> LayerChildCandidate:
            record = parser_source_map[item_id]
            text = record["text"]
            assert isinstance(text, str) and text
            return LayerChildCandidate(
                node_id=f"semantic:{item_id}",
                parent_node_id=parent_id,
                title=text,
                node_type="SOURCE_ITEM",
                total_content_pointers=[
                    HydratedTextPointer(
                        source_cluster_id=item_id,
                        start_char=0,
                        end_char=len(text) - 1,
                        verbatim_text=text,
                    )
                ],
                expandable=False,
                metadata={"modality": record["modality"]},
            )

        parent_id = current_layer_context.parent_node_ids[0]
        return CurrentLayerResult(
            children=[
                LayerChildCandidate(
                    node_id=f"semantic:{group_id}",
                    parent_node_id=parent_id,
                    title="Text A and Figure",
                    node_type="TEXT_FLOW",
                    child_candidates=[
                        _candidate(item_id, f"semantic:{group_id}")
                        for item_id in grouped_item_ids
                    ],
                    expandable=False,
                    metadata={"source_group_id": group_id},
                ),
                *(
                    _candidate(item_id, parent_id)
                    for item_id in (source_order[2][1], source_order[3][1])
                ),
            ],
            satisfied=True,
        )

    def _group_aware_review_fn(*, current_layer_result, **kwargs) -> CurrentLayerReview:
        del kwargs
        return CurrentLayerReview(
            updated_result=current_layer_result,
            coverage_ok=True,
            satisfied=True,
        )

    run, bundle, artifacts = run_ocr_ingest_workflow(
        document_id=document_id,
        title="OCR mixed reading order",
        output_dir=scratch / "artifacts",
        image_payloads=[OCRImagePayload(page_number=1, image_path=str(image_path))],
        provider_settings=settings,
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        ocr_runner=_mixed_ocr_runner,
        quality_policy=OCRQualityPolicy(mode="permissive"),
        grouping_proposals=(
            GroupProposal(
                kind="section",
                members=(f"leaf:{source_order[0][1]}", f"leaf:{source_order[1][1]}"),
            ),
        ),
        deps={
            "propose_layer_fn": _group_aware_proposal_fn,
            "review_layer_fn": _group_aware_review_fn,
        },
    )

    assert run.status == "succeeded"
    assert bundle is not None
    assert artifacts.document_tree is not None
    expected_source_order = tuple(item_id for _, item_id in source_order)
    assert semantic_source_orders
    assert all(order == expected_source_order for order in semantic_source_orders)
    assert bundle.grouping_tree == artifacts.document_tree.model_dump(mode="json")
    edges = bundle.graph_payload["edges"]
    assert isinstance(edges, list)
    graph_nodes = bundle.graph_payload["nodes"]
    assert isinstance(graph_nodes, list)
    labels_by_id = {
        node["id"]: node["label"]
        for node in graph_nodes
        if isinstance(node, dict)
        and isinstance(node.get("id"), str)
        and isinstance(node.get("label"), str)
    }
    target_labels = [
        labels_by_id[edge["target_ids"][0]]
        for edge in edges
        if isinstance(edge, dict)
        and isinstance(edge.get("target_ids"), list)
        and edge["target_ids"]
    ]
    # The graph's text-only semantic projection omits the figure description;
    # the complete multimodal order remains authoritative in grouping_tree.
    assert target_labels == ["Text A", "Caption", "Text B"]


@pytest.mark.ci
def test_failed_ocr_child_retries_from_kogwistar_checkpoint_without_replaying_pages() -> (
    None
):
    """A failed nested child is retried by Kogwistar, not by a parser fork."""
    scratch = _scratch("ocr_failed_child_retry")
    payloads: list[OCRImagePayload] = []
    for page_number in (1, 2):
        image_path = scratch / "images" / f"page-{page_number}.png"
        _draw_test_image(image_path, lines=[f"retry page {page_number}"])
        payloads.append(
            OCRImagePayload(page_number=page_number, image_path=str(image_path))
        )
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="retry-ocr"),
        parser=ProviderEndpointConfig(provider="fake", model="retry-parser"),
        embedding=EmbeddingProviderConfig(
            provider="fake", model="retry-embed", dimension=1
        ),
    )
    calls: dict[int, int] = {}

    def _fail_once_on_page_two(
        image_path: Path,
        page_number: int,
        provider_settings: WorkflowProviderSettings,
    ) -> OCRClusterResponse:
        del image_path, provider_settings
        calls[page_number] = calls.get(page_number, 0) + 1
        if page_number == 2 and calls[page_number] == 1:
            raise RuntimeError("simulated OCR child failure")
        return _fake_ocr_response(page_number, f"retry page {page_number}")

    workflow_engine, conversation_engine, knowledge_engine = (
        build_workflow_engine_triplet(scratch / "engines", "in_memory")
    )
    run_id = "ocr|failed-child-retry"
    with pytest.raises(RuntimeError, match="OCR root workflow"):
        run_ocr_ingest_workflow(
            document_id="ocr-failed-child-retry-doc",
            title="OCR Failed Child Retry",
            output_dir=scratch / "artifacts",
            image_payloads=payloads,
            provider_settings=settings,
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            knowledge_engine=knowledge_engine,
            ocr_runner=_fail_once_on_page_two,
            deps={"parse_semantic_fn": _grounded_semantic_tree},
            run_id=run_id,
        )

    resumed, bundle, _artifacts = run_ocr_ingest_workflow(
        document_id="ocr-failed-child-retry-doc",
        title="OCR Failed Child Retry",
        output_dir=scratch / "artifacts",
        image_payloads=payloads,
        provider_settings=settings,
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        ocr_runner=_fail_once_on_page_two,
        deps={"parse_semantic_fn": _grounded_semantic_tree},
        run_id=run_id,
        resume_from_checkpoint=True,
    )

    assert resumed.status == "succeeded"
    assert bundle is not None
    assert calls == {1: 1, 2: 2}
    lineage_edges = [
        edge
        for edge in conversation_engine.read.get_edges(limit=1000)
        if (edge.metadata or {}).get("relation") == "wf_invoked"
        and (edge.metadata or {}).get("parent_run_id") == run_id
    ]
    page_edges = [
        edge
        for edge in lineage_edges
        if (edge.metadata or {}).get("workflow_id") == "kg_doc_parser.ocr_pages.v1"
    ]
    assert len(page_edges) == 1
    assert (page_edges[0].metadata or {}).get("child_run_id") == f"{run_id}|ocr-pages"


@pytest.mark.ci
def test_ocr_root_retries_after_grouping_sqlite_commit_before_nested_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A root resume reconciles a grouping commit made before its child checkpoint."""
    scratch = _scratch("ocr_root_grouping_commit_crash")
    image_path = scratch / "images" / "page-1.png"
    _draw_test_image(image_path, lines=["root grouping crash boundary"])
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="root-crash-ocr"),
        parser=ProviderEndpointConfig(provider="fake", model="root-crash-parser"),
        embedding=EmbeddingProviderConfig(
            provider="fake", model="root-crash-embed", dimension=1
        ),
    )
    workflow_engine, conversation_engine, knowledge_engine = (
        build_workflow_engine_triplet(scratch / "engines", "in_memory")
    )
    calls = {"ocr": 0, "parse": 0}

    def _runner(
        image_path: Path,
        page_number: int,
        provider_settings: WorkflowProviderSettings,
    ) -> OCRClusterResponse:
        del image_path, provider_settings
        calls["ocr"] += 1
        return _fake_ocr_response(
            page_number, "root grouping crash boundary"
        ).model_copy(
            update={
                "OCR_text_clusters": [
                    TextCluster(
                        text="root grouping crash",
                        bb_x_min=10.0,
                        bb_x_max=500.0,
                        bb_y_min=10.0,
                        bb_y_max=50.0,
                        cluster_number=0,
                    ),
                    TextCluster(
                        text="boundary",
                        bb_x_min=10.0,
                        bb_x_max=500.0,
                        bb_y_min=60.0,
                        bb_y_max=100.0,
                        cluster_number=1,
                    ),
                ],
                "meaningful_ordering": [0, 1],
            }
        )

    def _parse_semantic_fn(*, collection, parser_input_dict, parser_source_map):
        calls["parse"] += 1
        return _grounded_semantic_tree(
            collection=collection,
            parser_input_dict=parser_input_dict,
            parser_source_map=parser_source_map,
        )

    original_save = OCRWorkflowStateStore.save_document_tree
    crashed = False

    def _save_then_crash(
        store: OCRWorkflowStateStore,
        *args: object,
        **kwargs: object,
    ) -> str:
        nonlocal crashed
        revision = original_save(store, *args, **kwargs)  # type: ignore[arg-type]
        if not crashed:
            crashed = True
            raise RuntimeError("simulated root crash after grouping SQLite commit")
        return revision

    monkeypatch.setattr(OCRWorkflowStateStore, "save_document_tree", _save_then_crash)
    with pytest.raises(RuntimeError, match="OCR root workflow"):
        run_ocr_ingest_workflow(
            document_id="ocr-root-grouping-crash-doc",
            title="OCR Root Grouping Crash",
            output_dir=scratch / "artifacts",
            image_payloads=[OCRImagePayload(page_number=1, image_path=str(image_path))],
            provider_settings=settings,
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            knowledge_engine=knowledge_engine,
            ocr_runner=_runner,
            deps={"parse_semantic_fn": _parse_semantic_fn},
            run_id="ocr|root-grouping-crash",
        )

    monkeypatch.undo()
    resumed, bundle, artifacts = run_ocr_ingest_workflow(
        document_id="ocr-root-grouping-crash-doc",
        title="OCR Root Grouping Crash",
        output_dir=scratch / "artifacts",
        image_payloads=[OCRImagePayload(page_number=1, image_path=str(image_path))],
        provider_settings=settings,
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        ocr_runner=_runner,
        deps={"parse_semantic_fn": _parse_semantic_fn},
        run_id="ocr|root-grouping-crash",
        resume_from_checkpoint=True,
    )

    assert resumed.status == "succeeded"
    assert bundle is not None
    assert artifacts.document_tree is not None
    assert calls == {"ocr": 1, "parse": 1}
    assert bundle.grouping_tree == artifacts.document_tree.model_dump(mode="json")
    checkpoint = OCRWorkflowStateStore(
        artifacts.state_db_path
    ).load_grouping_checkpoint_metadata(
        document_id="ocr-root-grouping-crash-doc",
        grouping_policy_version="baseline-v1",
    )
    assert checkpoint is not None


@pytest.mark.ci
def test_ocr_root_retries_after_semantic_graph_commit_before_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A graph commit before semantic checkpointing is safely replayable."""
    scratch = _scratch("ocr_root_semantic_commit_crash")
    image_path = scratch / "images" / "page-1.png"
    _draw_test_image(image_path, lines=["semantic graph crash boundary"])
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="semantic-crash-ocr"),
        parser=ProviderEndpointConfig(provider="fake", model="semantic-crash-parser"),
        embedding=EmbeddingProviderConfig(
            provider="fake", model="semantic-crash-embed", dimension=1
        ),
    )
    workflow_engine, conversation_engine, knowledge_engine = (
        build_workflow_engine_triplet(scratch / "engines", "in_memory")
    )
    calls = {"ocr": 0, "parse": 0}

    def _runner(
        image_path: Path,
        page_number: int,
        provider_settings: WorkflowProviderSettings,
    ) -> OCRClusterResponse:
        del image_path, provider_settings
        calls["ocr"] += 1
        return _fake_ocr_response(
            page_number, "semantic graph crash boundary"
        ).model_copy(
            update={
                "OCR_text_clusters": [
                    TextCluster(
                        text="semantic graph crash",
                        bb_x_min=10.0,
                        bb_x_max=500.0,
                        bb_y_min=10.0,
                        bb_y_max=50.0,
                        cluster_number=0,
                    ),
                    TextCluster(
                        text="boundary",
                        bb_x_min=10.0,
                        bb_x_max=500.0,
                        bb_y_min=60.0,
                        bb_y_max=100.0,
                        cluster_number=1,
                    ),
                ],
                "meaningful_ordering": [0, 1],
            }
        )

    def _parse_semantic_fn(*, collection, parser_input_dict, parser_source_map):
        calls["parse"] += 1
        return _grounded_semantic_tree(
            collection=collection,
            parser_input_dict=parser_input_dict,
            parser_source_map=parser_source_map,
        )

    from kg_doc_parser.workflow_ingest.clients import DirectRuntimeIngestClient

    original_persist = DirectRuntimeIngestClient.persist_graph_payload
    committed = False
    first_graph_node_ids: set[str] = set()
    first_graph_edge_ids: set[str] = set()

    def _persist_then_crash(client, bundle):
        nonlocal committed
        result = original_persist(client, bundle)
        if not committed:
            first_graph_node_ids.update(
                str(record["id"])
                for record in bundle.graph_payload["nodes"]
                if isinstance(record, dict) and "id" in record
            )
            first_graph_edge_ids.update(
                str(record["id"])
                for record in bundle.graph_payload["edges"]
                if isinstance(record, dict) and "id" in record
            )
            committed = True
            raise RuntimeError("simulated semantic graph commit before checkpoint")
        return result

    monkeypatch.setattr(
        DirectRuntimeIngestClient, "persist_graph_payload", _persist_then_crash
    )
    run_id = "ocr|root-semantic-commit-crash"
    with pytest.raises(RuntimeError, match="OCR root workflow"):
        run_ocr_ingest_workflow(
            document_id="ocr-root-semantic-crash-doc",
            title="OCR Root Semantic Crash",
            output_dir=scratch / "artifacts",
            image_payloads=[OCRImagePayload(page_number=1, image_path=str(image_path))],
            provider_settings=settings,
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            knowledge_engine=knowledge_engine,
            ocr_runner=_runner,
            deps={"parse_semantic_fn": _parse_semantic_fn},
            run_id=run_id,
        )

    assert first_graph_node_ids
    monkeypatch.undo()

    resumed, bundle, _artifacts = run_ocr_ingest_workflow(
        document_id="ocr-root-semantic-crash-doc",
        title="OCR Root Semantic Crash",
        output_dir=scratch / "artifacts",
        image_payloads=[OCRImagePayload(page_number=1, image_path=str(image_path))],
        provider_settings=settings,
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        ocr_runner=_runner,
        deps={"parse_semantic_fn": _parse_semantic_fn},
        run_id=run_id,
        resume_from_checkpoint=True,
    )

    assert resumed.status == "succeeded"
    assert bundle is not None
    assert calls == {"ocr": 1, "parse": 2}
    final_node_ids = {
        node.id
        for node in knowledge_engine.read.get_nodes(limit=1000)
        if node.id in first_graph_node_ids
    }
    final_edge_ids = {
        edge.id
        for edge in knowledge_engine.read.get_edges(limit=1000)
        if edge.id in first_graph_edge_ids
    }
    assert final_node_ids == first_graph_node_ids
    assert final_edge_ids == first_graph_edge_ids
    assert bundle.canonical_write_result is not None
    assert bundle.canonical_write_result.nodes_written == 0
    assert bundle.canonical_write_result.edges_written == 0

