from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from uuid import uuid4

import pytest

from _kogwistar_test_helpers import load_kogwistar_fake_backend
from src.workflow_ingest import (
    DocumentTreeApiPersistenceClient,
    ServerCanonicalKgClient,
    WorkflowIngestInput,
)
from src.workflow_ingest.semantics import HydratedTextPointer, SemanticNode
from src.workflow_ingest.service import build_default_engines

pytestmark = [pytest.mark.workflow, pytest.mark.ci_full]

if str(os.getenv("KG_DOC_ENABLE_SERVER_E2E_CI_FULL") or "").strip().lower() not in {
    "1",
    "true",
    "yes",
    "on",
}:
    pytest.skip(
        "set KG_DOC_ENABLE_SERVER_E2E_CI_FULL=1 to run real server canonical persistence e2e tests",
        allow_module_level=True,
    )

pytest.importorskip("chromadb")
pytest.importorskip("fastapi")
pytest.importorskip("fastmcp")


def _scratch(name: str) -> Path:
    root = Path("tests") / ".tmp_workflow_ingest_server"
    path = root / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _server_modules() -> list[str]:
    return [
        "kogwistar.server.resources",
        "kogwistar.server.bootstrap",
        "kogwistar.server_mcp_with_admin",
    ]


def _load_isolated_server_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    base_dir = tmp_path / "server-data"
    monkeypatch.setenv("GKE_BACKEND", "chroma")
    monkeypatch.setenv("GKE_PERSIST_DIRECTORY", str(base_dir))
    monkeypatch.setenv("AUTH_MODE", "dev")
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "FALSE")
    monkeypatch.setenv("KOGWISTAR_LOG_LEVEL", "WARNING")

    for module_name in _server_modules():
        sys.modules.pop(module_name, None)

    server_module = importlib.import_module("kogwistar.server_mcp_with_admin")
    return TestClient(server_module.app)


def _fake_semantic_tree(*, collection, parser_input_dict, parser_source_map):
    root = SemanticNode(
        node_id=f"{collection.collection_id}|root",
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
                node_id=f"{collection.collection_id}|{unit_id}",
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


def _single_node_semantic_tree(*, collection, parser_input_dict, parser_source_map):
    return SemanticNode(
        node_id=f"{collection.collection_id}|root",
        title=collection.title,
        node_type="DOCUMENT_ROOT",
        total_content_pointers=[],
        child_nodes=[],
        level_from_root=0,
    )


@pytest.mark.integration
def test_server_canonical_roundtrip_uses_real_fastapi_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    with _load_isolated_server_app(tmp_path, monkeypatch) as api_client:
        scratch = _scratch("server_roundtrip")
        workflow_engine, conversation_engine, knowledge_engine = build_default_engines(
            scratch / "engines",
            backend_factory=load_kogwistar_fake_backend(),
        )
        ingest_client = ServerCanonicalKgClient(
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            persistence_client=DocumentTreeApiPersistenceClient(
                client=api_client,
                transport="fastapi_testclient",
            ),
        )
        doc_id = f"server-e2e-{uuid4().hex[:8]}"
        result = ingest_client.run_ingest(
            inp=WorkflowIngestInput.from_text(
                document_id=doc_id,
                text="Alpha clause\nBeta clause",
                title="Server E2E",
            ),
            deps={"parse_semantic_fn": _fake_semantic_tree},
        )

        assert result.status == "succeeded"
        assert result.bundle is not None
        assert result.bundle.persistence_mode == "server_canonical"
        assert result.bundle.kg_authority == "server"
        assert result.bundle.canonical_write_confirmed is True
        assert result.bundle.server_parser_used is False
        assert not knowledge_engine.persist.exists_node(result.bundle.graph_payload["nodes"][0]["id"])

        viz = api_client.get(f"/api/viz/d3.json?doc_id={doc_id}&mode=reify")
        assert viz.status_code == 200
        graph = viz.json()
        node_ids = {node["id"] for node in graph.get("nodes", [])}
        expected_ids = {node["id"] for node in result.bundle.graph_payload["nodes"]}

        assert expected_ids.issubset(node_ids)


@pytest.mark.integration
def test_server_canonical_duplicate_write_is_stable_for_fixed_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    with _load_isolated_server_app(tmp_path, monkeypatch) as api_client:
        scratch = _scratch("server_duplicate")
        workflow_engine, conversation_engine, _knowledge_engine = build_default_engines(
            scratch / "engines",
            backend_factory=load_kogwistar_fake_backend(),
        )
        ingest_client = ServerCanonicalKgClient(
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            persistence_client=DocumentTreeApiPersistenceClient(
                client=api_client,
                transport="fastapi_testclient",
            ),
        )
        doc_id = f"server-idempotent-{uuid4().hex[:8]}"
        inp = WorkflowIngestInput.from_text(
            document_id=doc_id,
            text="Only one stable root",
            title="Stable Root",
        )

        first = ingest_client.run_ingest(
            inp=inp,
            deps={"parse_semantic_fn": _single_node_semantic_tree},
        )
        second = ingest_client.run_ingest(
            inp=inp,
            deps={"parse_semantic_fn": _single_node_semantic_tree},
        )

        assert first.status == "succeeded"
        assert second.status == "succeeded"
        assert first.bundle is not None
        assert second.bundle is not None
        assert first.bundle.canonical_write_confirmed is True
        assert second.bundle.canonical_write_confirmed is True

        viz = api_client.get(f"/api/viz/d3.json?doc_id={doc_id}&mode=reify")
        assert viz.status_code == 200
        graph = viz.json()
        root_id = f"{doc_id}|root"
        matching_nodes = [node for node in graph.get("nodes", []) if node.get("id") == root_id]

        assert len(matching_nodes) == 1
