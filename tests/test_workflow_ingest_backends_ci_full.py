from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from uuid import uuid4

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine

from _kogwistar_test_helpers import load_kogwistar_fake_backend
from src.workflow_ingest.models import WorkflowIngestInput
from src.workflow_ingest.semantics import HydratedTextPointer, SemanticNode
from src.workflow_ingest.service import _TinyEmbeddingFunction, run_ingest_workflow


def _scratch(name: str) -> Path:
    root = Path("tests") / ".tmp_workflow_ingest_backends"
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


def _build_engine_triplet(base_dir: Path, backend_kind: str):
    emb = _TinyEmbeddingFunction()
    if backend_kind == "fake":
        build_fake_backend = load_kogwistar_fake_backend()
        backend_factory = build_fake_backend
        backend = None
    elif backend_kind == "chroma":
        if os.getenv("KG_DOC_ENABLE_CHROMA_CI_FULL", "").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            pytest.skip("set KG_DOC_ENABLE_CHROMA_CI_FULL=1 to run chroma ci_full workflow tests")
        backend_factory = None
        backend = None
    elif backend_kind == "pg":
        pytest.importorskip("sqlalchemy")
        dsn = os.getenv("KG_DOC_PG_DSN")
        if not dsn:
            pytest.skip("KG_DOC_PG_DSN is required for pgvector ci_full tests")
        from sqlalchemy import create_engine
        from kogwistar.engine_core.postgres_backend import PgVectorBackend

        engine = create_engine(dsn)
        schema_prefix = f"kgdoc_{uuid4().hex[:8]}"
        return (
            GraphKnowledgeEngine(
                persist_directory=str(base_dir / "workflow_meta"),
                kg_graph_type="workflow",
                embedding_function=emb,
                backend=PgVectorBackend(engine=engine, embedding_dim=2, schema=f"{schema_prefix}_wf"),
            ),
            GraphKnowledgeEngine(
                persist_directory=str(base_dir / "conversation_meta"),
                kg_graph_type="conversation",
                embedding_function=emb,
                backend=PgVectorBackend(engine=engine, embedding_dim=2, schema=f"{schema_prefix}_conv"),
            ),
            GraphKnowledgeEngine(
                persist_directory=str(base_dir / "knowledge_meta"),
                kg_graph_type="knowledge",
                embedding_function=emb,
                backend=PgVectorBackend(engine=engine, embedding_dim=2, schema=f"{schema_prefix}_kg"),
            ),
        )
    else:
        raise ValueError(f"unsupported backend kind: {backend_kind}")

    return (
        GraphKnowledgeEngine(
            persist_directory=str(base_dir / "workflow"),
            kg_graph_type="workflow",
            embedding_function=emb,
            backend_factory=backend_factory,
            backend=backend,
        ),
        GraphKnowledgeEngine(
            persist_directory=str(base_dir / "conversation"),
            kg_graph_type="conversation",
            embedding_function=emb,
            backend_factory=backend_factory,
            backend=backend,
        ),
        GraphKnowledgeEngine(
            persist_directory=str(base_dir / "knowledge"),
            kg_graph_type="knowledge",
            embedding_function=emb,
            backend_factory=backend_factory,
            backend=backend,
        ),
    )


@pytest.mark.workflow
@pytest.mark.ci_full
@pytest.mark.parametrize("backend_kind", ["fake", "chroma"])
def test_workflow_ingest_runs_on_ci_full_backends(backend_kind: str):
    scratch = _scratch(f"ci_full_{backend_kind}")
    workflow_engine, conversation_engine, knowledge_engine = _build_engine_triplet(
        scratch / "engines", backend_kind
    )
    inp = WorkflowIngestInput.from_text(
        document_id=f"ci-full-{backend_kind}",
        text="Alpha clause\nBeta clause",
        title=f"Workflow {backend_kind}",
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
    assert knowledge_engine.persist.exists_node(bundle.graph_payload["nodes"][0]["id"])


@pytest.mark.workflow
@pytest.mark.ci_full
def test_workflow_ingest_pgvector_ci_full():
    scratch = _scratch("ci_full_pg")
    workflow_engine, conversation_engine, knowledge_engine = _build_engine_triplet(
        scratch / "engines", "pg"
    )
    inp = WorkflowIngestInput.from_text(
        document_id="ci-full-pg",
        text="Alpha clause\nBeta clause",
        title="Workflow pg",
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
