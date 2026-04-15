from __future__ import annotations

from pathlib import Path
from typing import Any

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.runtime.runtime import WorkflowRuntime

from .clients import DirectRuntimeIngestClient
from .design import DEFAULT_WORKFLOW_ID
from .handlers import build_ingest_step_resolver
from .models import IngestRunResult, WorkflowIngestInput
from .providers import WorkflowProviderSettings, build_embedding_function


class _TinyEmbeddingFunction:
    _name = "kg-doc-parser-workflow-embedding-v1"

    def name(self):
        return self._name

    def __call__(self, input):
        vectors = []
        for value in input:
            text = str(value or "")
            checksum = float((sum(ord(ch) for ch in text) % 97) + 1)
            vectors.append([float(len(text) + 1), checksum])
        return vectors


def build_default_engines(
    base_dir: str | Path,
    *,
    embedding_function=None,
    backend_factory=None,
    provider_settings: WorkflowProviderSettings | None = None,
):
    base_dir = Path(base_dir)
    provider_settings = provider_settings or WorkflowProviderSettings.from_env()
    # One embedding function is still wired per engine instance. The workflow
    # can carry embedding-space metadata, but engine-level routing is a future
    # Kogwistar concern.
    embedding = embedding_function or build_embedding_function(provider_settings.embedding)
    workflow_engine = GraphKnowledgeEngine(
        persist_directory=str(base_dir / "workflow"),
        kg_graph_type="workflow",
        embedding_function=embedding,
        backend_factory=backend_factory,
    )
    conversation_engine = GraphKnowledgeEngine(
        persist_directory=str(base_dir / "conversation"),
        kg_graph_type="conversation",
        embedding_function=embedding,
        backend_factory=backend_factory,
    )
    knowledge_engine = GraphKnowledgeEngine(
        persist_directory=str(base_dir / "knowledge"),
        kg_graph_type="knowledge",
        embedding_function=embedding,
        backend_factory=backend_factory,
    )
    return workflow_engine, conversation_engine, knowledge_engine


def build_runtime(*, workflow_engine, conversation_engine, deps: dict[str, Any] | None = None):
    resolver = build_ingest_step_resolver(deps=deps)
    return WorkflowRuntime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=resolver.resolve,
        predicate_registry={},
        trace=False,
    )


def run_ingest_workflow(
    *,
    inp: WorkflowIngestInput,
    workflow_engine,
    conversation_engine,
    knowledge_engine=None,
    workflow_id: str = DEFAULT_WORKFLOW_ID,
    deps: dict[str, Any] | None = None,
):
    client = DirectRuntimeIngestClient(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
    )
    result = client.run_ingest(
        inp=inp,
        workflow_id=workflow_id,
        deps=deps,
    )
    return _legacy_run_result(result)


def _legacy_run_result(result: IngestRunResult):
    class _RunCompat:
        def __init__(self, *, run_id: str, final_state: dict[str, Any], status: str) -> None:
            self.run_id = run_id
            self.final_state = final_state
            self.status = status

    return (
        _RunCompat(
            run_id=result.handle.run_id,
            final_state=result.final_state,
            status=result.status,
        ),
        result.bundle,
    )
