from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.storage_backend import StorageBackend
from kogwistar.runtime.runtime import WorkflowRuntime
from kogwistar.typing_interfaces import EmbeddingFunctionLike

from .clients import DirectRuntimeIngestClient
from .design import DEFAULT_WORKFLOW_ID
from .handlers import build_ingest_step_resolver
from .models import IngestRunResult, WorkflowExportBundle, WorkflowIngestInput
from .providers import WorkflowProviderSettings, build_embedding_function


@dataclass(slots=True)
class _RunCompat:
    run_id: str
    final_state: dict[str, Any]
    status: str


class _TinyEmbeddingFunction:
    _name = "kg-doc-parser-workflow-embedding-v1"

    def name(self) -> str:
        return self._name

    def __call__(self, documents_or_texts: list[str], /) -> list[list[float]]:
        vectors = []
        for value in documents_or_texts:
            text = str(value or "")
            checksum = float((sum(ord(ch) for ch in text) % 97) + 1)
            vectors.append([float(len(text) + 1), checksum])
        return vectors


class StorageBackendFactory(Protocol):
    """Build the parser's selected storage backend for one graph engine."""

    def __call__(self, engine: GraphKnowledgeEngine, /) -> StorageBackend: ...


def build_default_engines(
    base_dir: str | Path,
    *,
    embedding_function: EmbeddingFunctionLike | None = None,
    backend_factory: StorageBackendFactory | None = None,
    provider_settings: WorkflowProviderSettings | None = None,
    conversation_persistence_mode: Literal["single_stage", "two_stage"] = "single_stage",
) -> tuple[GraphKnowledgeEngine, GraphKnowledgeEngine, GraphKnowledgeEngine]:
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
        persistence_mode=conversation_persistence_mode,
    )
    knowledge_engine = GraphKnowledgeEngine(
        persist_directory=str(base_dir / "knowledge"),
        kg_graph_type="knowledge",
        embedding_function=embedding,
        backend_factory=backend_factory,
    )
    return workflow_engine, conversation_engine, knowledge_engine


def build_runtime(
    *,
    workflow_engine: GraphKnowledgeEngine,
    conversation_engine: GraphKnowledgeEngine,
    deps: dict[str, Any] | None = None,
) -> WorkflowRuntime:
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
    workflow_engine: GraphKnowledgeEngine,
    conversation_engine: GraphKnowledgeEngine,
    knowledge_engine: GraphKnowledgeEngine | None = None,
    workflow_id: str = DEFAULT_WORKFLOW_ID,
    deps: dict[str, Any] | None = None,
    run_id: str | None = None,
    resume_from_checkpoint: bool = False,
) -> tuple[_RunCompat, WorkflowExportBundle | None]:
    client = DirectRuntimeIngestClient(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
    )
    result = client.run_ingest(
        inp=inp,
        workflow_id=workflow_id,
        deps=deps,
        run_id=run_id,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    return _legacy_run_result(result)


def _legacy_run_result(result: IngestRunResult) -> tuple[_RunCompat, WorkflowExportBundle | None]:
    return (
        _RunCompat(
            run_id=result.handle.run_id,
            final_state=result.final_state,
            status=result.status,
        ),
        result.bundle,
    )
