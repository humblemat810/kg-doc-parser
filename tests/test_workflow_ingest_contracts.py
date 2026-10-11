from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from _kogwistar_test_helpers import (
    build_workflow_engine_triplet,
    drain_phase1_indexes_until_idle,
    drain_phase1_indexes_with_workers_until_idle,
)

from kg_doc_parser.workflow_ingest.adapters import (
    build_authoritative_source_map,
    build_parser_input_dict,
    normalize_ocr_pages,
)
from kg_doc_parser.workflow_ingest.clients import (
    ServerCanonicalKgClient,
    UnsupportedClientOperation,
)
from kg_doc_parser.workflow_ingest.design import build_ingest_workflow_design
from kg_doc_parser.workflow_ingest.models import (
    BoundingBox,
    GroundedSourceRecord,
    NormalizedPage,
    NormalizedSourceCollection,
    SourceUnit,
    WorkflowExportBundle,
    WorkflowIngestInput,
)
from kg_doc_parser.workflow_ingest.semantics import HydratedTextPointer, SemanticNode
from kg_doc_parser.workflow_ingest.service import run_ingest_workflow

pytestmark = [pytest.mark.workflow]


@pytest.fixture(
    params=[
        pytest.param("in_memory", id="in_memory", marks=pytest.mark.ci),
        pytest.param("chroma", id="chroma", marks=pytest.mark.slow),
    ]
)
def workflow_backend_kind(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param("eager", id="eager"),
        pytest.param("worker", id="worker"),
    ]
)
def workflow_index_mode(request):
    return request.param


def _local_scratch_dir(name: str) -> Path:
    root = Path("tests") / ".tmp_workflow_ingest"
    path = root / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fake_semantic_tree(*, collection, parser_input_dict, parser_source_map):
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
        ptr = HydratedTextPointer(
            source_cluster_id=unit_id,
            start_char=0,
            end_char=max(0, len(text) - 1),
            verbatim_text=text,
        )
        root.child_nodes.append(
            SemanticNode(
                title=f"section:{unit_id}",
                node_type="TEXT_FLOW",
                total_content_pointers=[ptr],
                child_nodes=[],
                level_from_root=1,
                parent_id=root.node_id,
            )
        )
    return root


def _partial_semantic_tree(*, collection, parser_input_dict, parser_source_map):
    root = SemanticNode(
        title=collection.title,
        node_type="DOCUMENT_ROOT",
        total_content_pointers=[],
        child_nodes=[],
        level_from_root=0,
    )
    first_id = next(iter(parser_source_map))
    text = parser_source_map[first_id]["text"]
    partial_end = max(0, (len(text) // 2) - 1)
    root.child_nodes.append(
        SemanticNode(
            title="partial",
            node_type="TEXT_FLOW",
            total_content_pointers=[
                HydratedTextPointer(
                    source_cluster_id=first_id,
                    start_char=0,
                    end_char=partial_end,
                    verbatim_text=text[: partial_end + 1],
                )
            ],
            child_nodes=[],
            level_from_root=1,
            parent_id=root.node_id,
        )
    )
    return root


class _FakeServerPersistenceClient:
    def __init__(self, *, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.calls: list[dict] = []

    def persist_graph_payload(self, bundle):
        self.calls.append(bundle.graph_payload)
        if self.should_fail:
            raise RuntimeError("server canonical persistence unavailable")
        return {
            "persistence_mode": "server_canonical",
            "kg_authority": "server",
            "canonical_write_confirmed": True,
            "nodes_written": len(bundle.graph_payload.get("nodes", [])),
            "edges_written": len(bundle.graph_payload.get("edges", [])),
            "transport": "server_client",
            "server_parser_used": False,
        }


@pytest.mark.ci
def test_workflow_input_from_text_and_llm_slicing():
    inp = WorkflowIngestInput.from_text(document_id="doc-1", text="hello world", title="Doc 1")
    assert inp.collections[0].pages[0].units[0].text == "hello world"

    unit = SourceUnit(
        modality="ocr_text",
        text="Clause 1",
        page_number=1,
        cluster_number=7,
        embedding_space="default_text",
        metadata={"internal": True},
    )
    llm_view = unit.model_dump(field_mode="llm")
    backend_view = unit.model_dump(field_mode="backend")

    assert "embedding_space" not in llm_view
    assert "metadata" not in llm_view
    assert backend_view["embedding_space"] == "default_text"
    assert backend_view["metadata"]["internal"] is True


@pytest.mark.ci
def test_normalized_collection_rejects_ambiguous_identity_and_embedding_labels() -> None:
    page = NormalizedPage(
        page_number=1,
        units=[SourceUnit(modality="text", text="Alpha", page_number=1, cluster_number=0)],
    )
    with pytest.raises(ValueError, match="collection_id must be non-empty"):
        NormalizedSourceCollection(
            collection_id=" ",
            title="Doc",
            modality="text",
            pages=[page],
        )
    with pytest.raises(ValueError, match="embedding_spaces must be unique"):
        NormalizedSourceCollection(
            collection_id="doc",
            title="Doc",
            modality="text",
            pages=[page],
            embedding_spaces=["default_text", "default_text"],
        )


@pytest.mark.ci
def test_workflow_input_rejects_duplicate_collection_ids() -> None:
    page = NormalizedPage(
        page_number=1,
        units=[SourceUnit(modality="text", text="Alpha", page_number=1, cluster_number=0)],
    )
    collection = NormalizedSourceCollection(
        collection_id="doc",
        title="Doc",
        modality="text",
        pages=[page],
    )
    with pytest.raises(ValueError, match="collection_id values must be unique"):
        WorkflowIngestInput(collections=[collection, collection.model_copy(deep=True)])


@pytest.mark.ci
def test_normalize_ocr_and_source_map_contract():
    inp = normalize_ocr_pages(
        document_id="ocr-doc",
        title="OCR Doc",
        pages=[
            {
                "pdf_page_num": 1,
                "printed_page_number": "1",
                "contains_table": False,
                "OCR_text_clusters": [
                    {
                        "text": "Alpha clause",
                        "bb_x_min": 1,
                        "bb_x_max": 2,
                        "bb_y_min": 3,
                        "bb_y_max": 4,
                        "cluster_number": 0,
                    }
                ],
                "non_text_objects": [
                    {
                        "description": "signature block",
                        "bb_x_min": 5,
                        "bb_x_max": 6,
                        "bb_y_min": 7,
                        "bb_y_max": 8,
                        "cluster_number": 0,
                    }
                ],
            }
        ],
    )
    source_map = build_authoritative_source_map(inp)

    assert len(source_map) == 2
    assert len(set(source_map.keys())) == 2
    assert any(record.modality == "ocr_text" for record in source_map.values())
    assert any(record.modality == "image_region" for record in source_map.values())
    assert any(unit_id.endswith("_t0") for unit_id in source_map)
    assert any(unit_id.endswith("_i0") for unit_id in source_map)


@pytest.mark.ci
@pytest.mark.parametrize("page_number", [True, 1.5, 0, -1, "not-a-number"])
def test_normalize_ocr_rejects_ambiguous_page_numbers(page_number):
    with pytest.raises(ValueError, match="positive integer"):
        normalize_ocr_pages(
            document_id="ocr-invalid-page",
            title="OCR Invalid Page",
            pages=[{"pdf_page_num": page_number}],  # type: ignore[list-item]
        )


@pytest.mark.ci
def test_normalize_ocr_rejects_duplicate_page_numbers():
    with pytest.raises(ValueError, match="duplicate OCR pdf_page_num"):
        normalize_ocr_pages(
            document_id="ocr-duplicate-page",
            title="OCR Duplicate Page",
            pages=[
                {"pdf_page_num": 1},
                {"pdf_page_num": "1"},  # type: ignore[dict-item]
            ],
        )


@pytest.mark.ci
@pytest.mark.parametrize(
    ("pages", "message"),
    [
        ([None], "each OCR page must be an object"),
        ([{}], "each OCR page requires pdf_page_num"),
        ([{"pdf_page_num": 1, "OCR_text_clusters": None}], "OCR cluster collections must be lists"),
        ([{"pdf_page_num": 1, "non_text_objects": {}}], "OCR cluster collections must be lists"),
        ([{"pdf_page_num": 1, "OCR_text_clusters": [None]}], "each OCR text cluster must be an object"),
        ([{"pdf_page_num": 1, "non_text_objects": [None]}], "each OCR non-text object must be an object"),
    ],
)
def test_normalize_ocr_rejects_malformed_container_shapes(pages, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        normalize_ocr_pages(document_id="doc", title="Doc", pages=pages)


@pytest.mark.ci
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bb_x_min", float("nan")),
        ("bb_y_max", float("inf")),
        ("bb_x_min", True),
        ("cluster_number", -1),
        ("cluster_number", 1.5),
        ("cluster_number", True),
    ],
)
def test_normalize_ocr_rejects_malformed_coordinates_and_clusters(field, value):
    page = {
        "pdf_page_num": 1,
        "OCR_text_clusters": [
            {
                "text": "Alpha",
                "bb_x_min": 0,
                "bb_x_max": 1,
                "bb_y_min": 0,
                "bb_y_max": 1,
                "cluster_number": 0,
            }
        ],
    }
    page["OCR_text_clusters"][0][field] = value
    with pytest.raises((TypeError, ValueError)):
        normalize_ocr_pages(
            document_id="ocr-invalid-payload",
            title="OCR Invalid Payload",
            pages=[page],  # type: ignore[list-item]
        )


@pytest.mark.ci
def test_normalize_ocr_rejects_inverted_boxes_and_cluster_collisions():
    with pytest.raises(ValueError, match="minimums"):
        normalize_ocr_pages(
            document_id="ocr-inverted-box",
            title="OCR Inverted Box",
            pages=[
                {
                    "pdf_page_num": 1,
                    "OCR_text_clusters": [
                        {
                            "text": "Alpha",
                            "bb_x_min": 2,
                            "bb_x_max": 1,
                            "bb_y_min": 0,
                            "bb_y_max": 1,
                            "cluster_number": 0,
                        }
                    ],
                }
            ],
        )
    with pytest.raises(ValueError, match="duplicate OCR text cluster_number"):
        normalize_ocr_pages(
            document_id="ocr-duplicate-cluster",
            title="OCR Duplicate Cluster",
            pages=[
                {
                    "pdf_page_num": 1,
                    "OCR_text_clusters": [
                        {"text": "Alpha", "cluster_number": 0},
                        {"text": "Beta", "cluster_number": 0},
                    ],
                }
            ],
        )


@pytest.mark.ci
@pytest.mark.parametrize(
    "values",
    [
        {"x_min": -1, "x_max": 1, "y_min": 0, "y_max": 1},
        {"x_min": 0, "x_max": 1, "y_min": 2, "y_max": 1},
        {"x_min": 0, "x_max": float("nan"), "y_min": 0, "y_max": 1},
        {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": float("inf")},
        {"x_min": True, "x_max": 1, "y_min": 0, "y_max": 1},
        {"x_min": "0", "x_max": 1, "y_min": 0, "y_max": 1},
    ],
)
def test_bounding_box_model_rejects_invalid_coordinates(values: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        BoundingBox.model_validate(values)


@pytest.mark.ci
def test_bounding_box_model_accepts_zero_area_and_pixel_coordinates() -> None:
    box = BoundingBox(x_min=0, x_max=1400, y_min=0, y_max=1000)
    assert box.x_max == 1400


@pytest.mark.ci
def test_source_map_generated_cluster_skips_reserved_explicit_cluster():
    inp = normalize_ocr_pages(
        document_id="ocr-reserved-cluster",
        title="OCR Reserved Cluster",
        pages=[
            {
                "pdf_page_num": 1,
                "OCR_text_clusters": [
                    {"text": "Explicit", "cluster_number": 0},
                    {"text": "Generated"},
                ],
            }
        ],
    )

    source_map = build_authoritative_source_map(inp)
    assert set(source_map) == {
        "ocr-reserved-cluster|p1_t0",
        "ocr-reserved-cluster|p1_t1",
    }


@pytest.mark.ci
def test_parser_input_and_source_map_share_reserved_cluster_allocation() -> None:
    inp = normalize_ocr_pages(
        document_id="ocr-roundtrip-cluster",
        title="OCR Roundtrip Cluster",
        pages=[
            {
                "pdf_page_num": 1,
                "OCR_text_clusters": [
                    {"text": "Explicit", "cluster_number": 10},
                    {"text": "Generated"},
                ],
            }
        ],
    )

    source_map = build_authoritative_source_map(inp)
    parser_page = build_parser_input_dict(inp.collections[0])["pages"][0]
    parser_clusters = parser_page["OCR_text_clusters"]
    assert [cluster["cluster_number"] for cluster in parser_clusters] == [10, 11]
    assert set(source_map) == {
        "ocr-roundtrip-cluster|p1_t10",
        "ocr-roundtrip-cluster|p1_t11",
    }
    assert source_map["ocr-roundtrip-cluster|p1_t10"].cluster_number == 10
    assert source_map["ocr-roundtrip-cluster|p1_t11"].cluster_number == 11


@pytest.mark.ci
def test_normalized_collection_rejects_duplicate_pages_and_non_strict_coordinates():
    with pytest.raises(ValueError, match="unique page_number"):
        WorkflowIngestInput(
            collections=[
                {
                    "collection_id": "doc",
                    "title": "Doc",
                    "modality": "text",
                    "pages": [
                        {"page_number": 1, "units": []},
                        {"page_number": 1, "units": []},
                    ],
                }
            ]
        )

    with pytest.raises(ValueError):
        SourceUnit(modality="text", text="text", page_number=True)

    with pytest.raises(ValueError, match="embedding_space"):
        SourceUnit(modality="text", text="text", embedding_space="   ")

    with pytest.raises(ValueError, match="unit_id must be non-empty"):
        SourceUnit(modality="text", text="text", unit_id="   ")

    with pytest.raises(ValueError, match="page_number must match"):
        NormalizedPage(
            page_number=2,
            units=[SourceUnit(modality="text", text="text", page_number=1)],
        )

    with pytest.raises(ValueError, match="description or source_uri"):
        SourceUnit(modality="pure_image", source_uri="   ")


def test_fake_workflow_run_text_success_and_knowledge_persist(
    workflow_backend_kind,
    workflow_index_mode,
    monkeypatch,
):
    scratch_dir = _local_scratch_dir("text_success")
    workflow_engine, conversation_engine, knowledge_engine = build_workflow_engine_triplet(
        scratch_dir / "engines", workflow_backend_kind
    )
    if workflow_index_mode == "worker":
        monkeypatch.setattr(workflow_engine, "reconcile_indexes", lambda *a, **k: 0)
        monkeypatch.setattr(conversation_engine, "reconcile_indexes", lambda *a, **k: 0)
        monkeypatch.setattr(knowledge_engine, "reconcile_indexes", lambda *a, **k: 0)
    inp = WorkflowIngestInput.from_text(
        document_id="wf-doc",
        text="Alpha clause\nBeta clause",
        title="Workflow Doc",
    )

    run, bundle = run_ingest_workflow(
        inp=inp,
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        deps={"parse_semantic_fn": _fake_semantic_tree},
    )
    if workflow_index_mode == "worker":
        drain_phase1_indexes_with_workers_until_idle(
            workflow_engine, conversation_engine, knowledge_engine
        )
    else:
        drain_phase1_indexes_until_idle(
            workflow_engine, conversation_engine, knowledge_engine
        )

    assert run.status == "succeeded"
    assert bundle is not None
    assert bundle.persisted_to_knowledge_engine is True
    assert bundle.persistence_mode == "local_debug"
    assert bundle.kg_authority == "local"
    assert bundle.canonical_write_confirmed is False
    assert bundle.retrieval_metadata["supports_split_embedding_spaces"] is True
    assert knowledge_engine.persist.exists_node(bundle.graph_payload["nodes"][0]["id"])
    assert run.final_state["strategy_attempt_counts"]
    assert all(count >= 1 for count in run.final_state["strategy_attempt_counts"].values())
    history = run.final_state["strategy_execution_history"]
    assert any(event["event"] == "selected" for event in history)
    assert any(event["event"] == "succeeded" for event in history)


def test_fake_workflow_run_ocr_success(workflow_backend_kind):
    scratch_dir = _local_scratch_dir("ocr_success")
    workflow_engine, conversation_engine, knowledge_engine = build_workflow_engine_triplet(
        scratch_dir / "engines", workflow_backend_kind
    )
    inp = normalize_ocr_pages(
        document_id="ocr-workflow",
        title="OCR Workflow",
        pages=[
            {
                "pdf_page_num": 1,
                "printed_page_number": "1",
                "contains_table": False,
                "OCR_text_clusters": [
                    {
                        "text": "Clause 1",
                        "bb_x_min": 1,
                        "bb_x_max": 2,
                        "bb_y_min": 3,
                        "bb_y_max": 4,
                        "cluster_number": 10,
                    },
                    {
                        "text": "Clause 2",
                        "bb_x_min": 11,
                        "bb_x_max": 12,
                        "bb_y_min": 13,
                        "bb_y_max": 14,
                        "cluster_number": 11,
                    },
                ],
                "non_text_objects": [
                    {
                        "description": "stamp image",
                        "bb_x_min": 5,
                        "bb_x_max": 6,
                        "bb_y_min": 7,
                        "bb_y_max": 8,
                        "cluster_number": 3,
                    }
                ],
            }
        ],
    )

    run, bundle = run_ingest_workflow(
        inp=inp,
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        deps={"parse_semantic_fn": _fake_semantic_tree},
    )
    drain_phase1_indexes_until_idle(workflow_engine, conversation_engine, knowledge_engine)

    assert run.status == "succeeded"
    assert bundle is not None
    assert "image" in bundle.embedding_spaces
    assert bundle.authoritative_source_map


def test_fake_workflow_validation_failure_is_structured(workflow_backend_kind):
    scratch_dir = _local_scratch_dir("validation_failure")
    workflow_engine, conversation_engine, knowledge_engine = build_workflow_engine_triplet(
        scratch_dir / "engines", workflow_backend_kind
    )
    inp = WorkflowIngestInput(
        request_id="bad-doc",
        collections=[
            WorkflowIngestInput.from_text(
                document_id="bad-doc",
                text="First section\nSecond section",
                title="Bad Doc",
            ).collections[0]
        ],
    )

    run, bundle = run_ingest_workflow(
        inp=inp,
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        deps={
            "parse_semantic_fn": _partial_semantic_tree,
            "coverage_threshold": 0.99,
        },
    )
    drain_phase1_indexes_until_idle(workflow_engine, conversation_engine, knowledge_engine)

    assert bundle is not None
    assert run.status in {"failed", "failure"}
    assert any("text coverage below threshold" in err for err in run.final_state["workflow_errors"])
    attempt_counts = run.final_state["strategy_attempt_counts"]
    assert attempt_counts
    assert all(count >= 1 for count in attempt_counts.values())
    history = run.final_state["strategy_execution_history"]
    assert any(event["event"] == "selected" for event in history)


@pytest.mark.ci
def test_source_map_preserves_explicit_unit_ids_and_cluster_identity():
    inp = WorkflowIngestInput(
        request_id="stable-doc",
        collections=[
            WorkflowIngestInput.from_text(
                document_id="stable-doc",
                text="Preserved",
                title="Stable Doc",
            ).collections[0].model_copy(
                update={
                    "pages": [
                        WorkflowIngestInput.from_text(
                            document_id="stable-doc",
                            text="Preserved",
                            title="Stable Doc",
                        ).collections[0].pages[0].model_copy(
                            update={
                                "units": [
                                    SourceUnit(
                                        unit_id="custom-unit-7",
                                        modality="text",
                                        text="Preserved",
                                        page_number=1,
                                        cluster_number=7,
                                        metadata={"source": "explicit"},
                                    )
                                ]
                            }
                        )
                    ]
                }
            )
        ],
    )

    source_map = build_authoritative_source_map(inp)
    record = source_map["custom-unit-7"]

    assert record.unit_id == "custom-unit-7"
    assert record.cluster_number == 7
    assert record.metadata["original_cluster_number"] == 7
    assert record.metadata["source"] == "explicit"


@pytest.mark.ci
def test_invalid_source_unit_payload_fails_cleanly():
    try:
        SourceUnit(modality="image_region", page_number=1, cluster_number=0)
    except ValueError as exc:
        assert "require description or source_uri" in str(exc)
    else:
        raise AssertionError("expected validation error for incomplete image_region payload")


@pytest.mark.ci
def test_workflow_design_matches_expected_step_sequence():
    nodes, edges = build_ingest_workflow_design()

    assert [node.metadata["wf_op"] for node in nodes] == [
        "start",
        "normalize_input",
        "build_source_map",
        "init_parse_session",
        "check_frontier_remaining",
        "prepare_layer_frontier",
        "triage_parse_strategy",
        "propose_layer_breakdown",
        "propose_layer_breakdown",
        "page_index_layer",
        "review_cud_proposal",
        "apply_cud_update",
        "check_layer_coverage",
        "check_layer_satisfaction",
        "repair_layer_pointers",
        "dedupe_and_filter_layer",
        "validate_layer_commit",
        "commit_layer_children",
        "check_children_expandable",
        "enqueue_next_layer_frontier",
        "finalize_semantic_tree",
        "validate_tree",
        "export_graph",
        "persist_canonical_graph",
        "parse_failure",
        "end",
    ]
    assert edges[0].source_ids[0].endswith("|start")
    assert edges[-1].target_ids[0].endswith("|end")
    edge_pairs = {
        (edge.source_ids[0].split("|")[-1], edge.target_ids[0].split("|")[-1])
        for edge in edges
    }
    assert ("prepare_layer_frontier", "triage_parse_strategy") in edge_pairs
    assert ("triage_parse_strategy", "page_index_layer") in edge_pairs
    assert ("triage_parse_strategy", "layer_excerpt_method") in edge_pairs
    assert ("triage_parse_strategy", "layer_boundary_method") in edge_pairs
    assert ("check_layer_satisfaction", "parse_failure") in edge_pairs
    assert ("check_layer_satisfaction", "triage_parse_strategy") in edge_pairs
    assert ("check_layer_satisfaction", "repair_layer_pointers") in edge_pairs
    assert ("dedupe_and_filter_layer", "validate_layer_commit") in edge_pairs
    assert ("validate_layer_commit", "commit_layer_children") in edge_pairs
    assert ("validate_layer_commit", "check_layer_satisfaction") in edge_pairs
    assert ("page_index_layer", "review_cud_proposal") in edge_pairs
    assert ("page_index_layer", "validate_tree") not in edge_pairs
    guarded = {
        (src, dst): edge.metadata.get("wf_predicate")
        for edge in edges
        for src, dst in [(edge.source_ids[0].split("|")[-1], edge.target_ids[0].split("|")[-1])]
    }
    assert guarded[("triage_parse_strategy", "layer_boundary_method")] == "parse_strategy_layer_boundary"
    assert guarded[("check_layer_satisfaction", "parse_failure")] == "all_strategies_exhausted"
    assert guarded[("check_layer_satisfaction", "repair_layer_pointers")] == "batch_has_repair_candidates"


@pytest.mark.ci
def test_export_bundle_llm_slicing_excludes_backend_grounding_data():
    bundle = WorkflowExportBundle(
        graph_payload={"nodes": [{"id": "n1"}], "edges": []},
        authoritative_source_map={
            "u1": GroundedSourceRecord(
                unit_id="u1",
                collection_id="doc-1",
                modality="text",
                page_number=1,
                cluster_number=0,
                text="Alpha",
                parser_text="Alpha",
                embedding_space="default_text",
                participates_in_semantic_text=True,
            )
        },
        embedding_spaces=["default_text", "image"],
        retrieval_metadata={"supports_split_embedding_spaces": True},
        persisted_to_knowledge_engine=True,
    )

    llm_view = bundle.model_dump(field_mode="llm")
    backend_view = bundle.model_dump(field_mode="backend")

    assert "authoritative_source_map" not in llm_view
    assert "embedding_spaces" not in llm_view
    assert backend_view["authoritative_source_map"]["u1"]["parser_text"] == "Alpha"


def test_pointer_correction_is_applied_before_validation(workflow_backend_kind):
    scratch_dir = _local_scratch_dir("pointer_repair")
    workflow_engine, conversation_engine, knowledge_engine = build_workflow_engine_triplet(
        scratch_dir / "engines", workflow_backend_kind
    )
    inp = WorkflowIngestInput.from_text(
        document_id="repair-doc",
        text="Alpha clause",
        title="Repair Doc",
    )

    def _misaligned_tree(*, collection, parser_input_dict, parser_source_map):
        unit_id, record = next(iter(parser_source_map.items()))
        text = record["text"]
        return SemanticNode(
            title=collection.title,
            node_type="DOCUMENT_ROOT",
            total_content_pointers=[],
            child_nodes=[
                SemanticNode(
                    title="repair-me",
                    node_type="TEXT_FLOW",
                    total_content_pointers=[
                        HydratedTextPointer(
                            source_cluster_id=unit_id,
                            start_char=0,
                            end_char=len(text) - 1,
                            verbatim_text=text + " trailing noise",
                        )
                    ],
                    child_nodes=[],
                    level_from_root=1,
                )
            ],
            level_from_root=0,
        )

    run, bundle = run_ingest_workflow(
        inp=inp,
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        deps={"parse_semantic_fn": _misaligned_tree},
    )
    drain_phase1_indexes_until_idle(workflow_engine, conversation_engine, knowledge_engine)

    assert run.status == "succeeded"
    assert bundle is not None
    assert run.final_state["corrected_pointer_count"] == 1


def test_server_canonical_client_uses_server_write_and_not_local_kg(workflow_backend_kind):
    scratch_dir = _local_scratch_dir("server_canonical")
    workflow_engine, conversation_engine, knowledge_engine = build_workflow_engine_triplet(
        scratch_dir / "engines", workflow_backend_kind
    )
    server_client = _FakeServerPersistenceClient()
    client = ServerCanonicalKgClient(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        persistence_client=server_client,
    )
    called = {"parse": 0}

    def _tracking_parse(*, collection, parser_input_dict, parser_source_map):
        called["parse"] += 1
        return _fake_semantic_tree(
            collection=collection,
            parser_input_dict=parser_input_dict,
            parser_source_map=parser_source_map,
        )

    result = client.run_ingest(
        inp=WorkflowIngestInput.from_text(
            document_id="server-doc",
            text="Alpha clause\nBeta clause",
            title="Server Canonical Doc",
        ),
        deps={"parse_semantic_fn": _tracking_parse},
    )
    drain_phase1_indexes_until_idle(workflow_engine, conversation_engine, knowledge_engine)

    assert result.status == "succeeded"
    assert result.bundle is not None
    assert called["parse"] == 1
    assert len(server_client.calls) == 1
    assert result.bundle.persistence_mode == "server_canonical"
    assert result.bundle.kg_authority == "server"
    assert result.bundle.canonical_write_confirmed is True
    assert result.bundle.server_parser_used is False
    assert result.bundle.persisted_to_knowledge_engine is False
    assert not knowledge_engine.persist.exists_node(result.bundle.graph_payload["nodes"][0]["id"])


def test_server_canonical_persistence_failure_keeps_exported_bundle_in_state(
    workflow_backend_kind,
):
    scratch_dir = _local_scratch_dir("server_canonical_failure")
    workflow_engine, conversation_engine, _knowledge_engine = build_workflow_engine_triplet(
        scratch_dir / "engines", workflow_backend_kind
    )
    client = ServerCanonicalKgClient(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        persistence_client=_FakeServerPersistenceClient(should_fail=True),
    )

    result = client.run_ingest(
        inp=WorkflowIngestInput.from_text(
            document_id="server-fail-doc",
            text="Alpha clause\nBeta clause",
            title="Server Canonical Failure",
        ),
        deps={"parse_semantic_fn": _fake_semantic_tree},
    )
    drain_phase1_indexes_until_idle(workflow_engine, conversation_engine)

    assert result.status in {"failed", "failure"}
    assert result.bundle is not None
    assert result.bundle.persistence_mode == "server_canonical"
    assert result.bundle.canonical_write_confirmed is False
    assert "export_bundle" in result.final_state
    assert "canonical_write_result" not in result.final_state


def test_server_canonical_resume_and_trace_are_explicitly_unsupported(workflow_backend_kind):
    scratch_dir = _local_scratch_dir("server_unsupported")
    workflow_engine, conversation_engine, _knowledge_engine = build_workflow_engine_triplet(
        scratch_dir / "engines", workflow_backend_kind
    )
    client = ServerCanonicalKgClient(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        persistence_client=_FakeServerPersistenceClient(),
    )

    for fn in (
        lambda: client.resume_ingest(),
        lambda: client.get_run_trace(run_id="run-1"),
        lambda: client.get_latest_checkpoint(run_id="run-1"),
    ):
        try:
            fn()
        except UnsupportedClientOperation:
            pass
        else:
            raise AssertionError("expected explicit unsupported client operation")
