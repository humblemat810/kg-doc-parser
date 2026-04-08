from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from _kogwistar_test_helpers import load_kogwistar_fake_backend
from src.workflow_ingest.clients import ServerCanonicalKgClient, UnsupportedClientOperation
from src.workflow_ingest.adapters import (
    build_authoritative_source_map,
    normalize_ocr_pages,
)
from src.workflow_ingest.design import build_ingest_workflow_design
from src.workflow_ingest.models import GroundedSourceRecord, SourceUnit, WorkflowExportBundle, WorkflowIngestInput
from src.workflow_ingest.semantics import HydratedTextPointer, SemanticNode
from src.workflow_ingest.service import build_default_engines, run_ingest_workflow


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


def test_fake_workflow_run_text_success_and_knowledge_persist():
    scratch_dir = _local_scratch_dir("text_success")
    fake_backend = load_kogwistar_fake_backend()
    workflow_engine, conversation_engine, knowledge_engine = build_default_engines(
        scratch_dir / "engines",
        backend_factory=fake_backend,
    )
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

    assert run.status == "succeeded"
    assert bundle is not None
    assert bundle.persisted_to_knowledge_engine is True
    assert bundle.persistence_mode == "local_debug"
    assert bundle.kg_authority == "local"
    assert bundle.canonical_write_confirmed is False
    assert bundle.retrieval_metadata["supports_split_embedding_spaces"] is True
    assert knowledge_engine.persist.exists_node(bundle.graph_payload["nodes"][0]["id"])


def test_fake_workflow_run_ocr_success():
    scratch_dir = _local_scratch_dir("ocr_success")
    fake_backend = load_kogwistar_fake_backend()
    workflow_engine, conversation_engine, knowledge_engine = build_default_engines(
        scratch_dir / "engines",
        backend_factory=fake_backend,
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

    assert run.status == "succeeded"
    assert bundle is not None
    assert "image" in bundle.embedding_spaces
    assert bundle.authoritative_source_map


def test_fake_workflow_validation_failure_is_structured():
    scratch_dir = _local_scratch_dir("validation_failure")
    fake_backend = load_kogwistar_fake_backend()
    workflow_engine, conversation_engine, knowledge_engine = build_default_engines(
        scratch_dir / "engines",
        backend_factory=fake_backend,
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

    assert bundle is None
    assert run.status in {"failed", "failure"}
    assert any("text coverage below threshold" in err for err in run.final_state["workflow_errors"])


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


def test_invalid_source_unit_payload_fails_cleanly():
    try:
        SourceUnit(modality="image_region", page_number=1, cluster_number=0)
    except Exception as exc:
        assert "require description or source_uri" in str(exc)
    else:
        raise AssertionError("expected validation error for incomplete image_region payload")


def test_workflow_design_matches_expected_step_sequence():
    nodes, edges = build_ingest_workflow_design()

    assert [node.metadata["wf_op"] for node in nodes] == [
        "start",
        "normalize_input",
        "build_source_map",
        "parse_semantic",
        "correct_pointers",
        "validate_tree",
        "export_graph",
        "persist_canonical_graph",
        "end",
    ]
    assert edges[0].source_ids[0].endswith("|start")
    assert edges[-1].target_ids[0].endswith("|end")


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


def test_pointer_correction_is_applied_before_validation():
    scratch_dir = _local_scratch_dir("pointer_repair")
    fake_backend = load_kogwistar_fake_backend()
    workflow_engine, conversation_engine, knowledge_engine = build_default_engines(
        scratch_dir / "engines",
        backend_factory=fake_backend,
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

    assert run.status == "succeeded"
    assert bundle is not None
    assert run.final_state["corrected_pointer_count"] == 1


def test_server_canonical_client_uses_server_write_and_not_local_kg():
    scratch_dir = _local_scratch_dir("server_canonical")
    fake_backend = load_kogwistar_fake_backend()
    workflow_engine, conversation_engine, knowledge_engine = build_default_engines(
        scratch_dir / "engines",
        backend_factory=fake_backend,
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


def test_server_canonical_persistence_failure_keeps_exported_bundle_in_state():
    scratch_dir = _local_scratch_dir("server_canonical_failure")
    fake_backend = load_kogwistar_fake_backend()
    workflow_engine, conversation_engine, _knowledge_engine = build_default_engines(
        scratch_dir / "engines",
        backend_factory=fake_backend,
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

    assert result.status in {"failed", "failure"}
    assert result.bundle is not None
    assert result.bundle.persistence_mode == "server_canonical"
    assert result.bundle.canonical_write_confirmed is False
    assert "export_bundle" in result.final_state
    assert "canonical_write_result" not in result.final_state


def test_server_canonical_resume_and_trace_are_explicitly_unsupported():
    scratch_dir = _local_scratch_dir("server_unsupported")
    fake_backend = load_kogwistar_fake_backend()
    workflow_engine, conversation_engine, _knowledge_engine = build_default_engines(
        scratch_dir / "engines",
        backend_factory=fake_backend,
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
