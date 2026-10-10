from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.storage_backend import StorageBackend
from kogwistar.runtime.runtime import WorkflowRuntime
from kogwistar.typing_interfaces import EmbeddingFunctionLike

from .clients import DirectRuntimeIngestClient
from .design import DEFAULT_WORKFLOW_ID
from .handlers import build_ingest_step_resolver
from .models import IngestRunResult, WorkflowExportBundle, WorkflowIngestInput
from .providers import WorkflowProviderSettings, build_embedding_function

if TYPE_CHECKING:
    from kogwistar.runtime.contract import Predicate, WorkflowEdgeInfo
else:
    class WorkflowEdgeInfo(Protocol):
        """Runtime fallback for Core releases without the newer contract export."""

        dst: str

    class Predicate(Protocol):
        """Runtime fallback for the Core predicate callback contract."""

        def __call__(
            self,
            edge: WorkflowEdgeInfo,
            state: Mapping[str, object],
            result: object,
        ) -> bool: ...


WorkflowState = Mapping[str, object]
WorkflowPredicates = dict[str, Predicate]


def _string_set(value: object) -> set[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return set()
    return {str(item) for item in value}


def workflow_predicates() -> WorkflowPredicates:
    """Guards for parser strategy transitions.

    These are deliberately derived from persisted state only.  The provider
    never controls the route directly, and a retry cannot re-enable a method
    that the satisfaction check disabled.
    """

    def _context(state: WorkflowState) -> dict[str, object]:
        value = state.get("current_layer_context")
        return value if isinstance(value, dict) else {}

    def _metadata(state: WorkflowState) -> dict[str, object]:
        value = _context(state).get("metadata")
        return value if isinstance(value, dict) else {}

    def _strategy_name(edge: WorkflowEdgeInfo) -> str:
        target = str(edge.dst).split("|")[-1]
        return {
            "layer_excerpt_method": "layer_excerpt",
            "layer_boundary_method": "layer_boundary",
            "page_index_layer": "page_index",
        }.get(target, "")

    def _selected_strategy(edge: WorkflowEdgeInfo, state: WorkflowState, result: object) -> bool:
        del result
        return str(_metadata(state).get("parse_strategy")) == _strategy_name(edge)

    def _failed_with_remaining(edge: WorkflowEdgeInfo, state: WorkflowState, result: object) -> bool:
        del edge, result
        disabled = _string_set(_metadata(state).get("disabled_strategies", []))
        return bool(disabled) and len(disabled) < 3

    def _exhausted(edge: WorkflowEdgeInfo, state: WorkflowState, result: object) -> bool:
        del edge, result
        return len(_string_set(_metadata(state).get("disabled_strategies", []))) >= 3

    def _strategy_selection_failed(edge: WorkflowEdgeInfo, state: WorkflowState, result: object) -> bool:
        del edge, result
        return bool(state.get("strategy_selection_error"))

    def _commit_candidates_valid(edge: WorkflowEdgeInfo, state: WorkflowState, result: object) -> bool:
        del edge, result
        review = state.get("current_layer_review")
        return isinstance(review, dict) and bool(review.get("metadata", {}).get("commit_validation")) and review.get("satisfied") is True

    def _commit_candidates_invalid(edge: WorkflowEdgeInfo, state: WorkflowState, result: object) -> bool:
        del edge, result
        review = state.get("current_layer_review")
        return isinstance(review, dict) and bool(review.get("metadata", {}).get("commit_validation")) and review.get("satisfied") is False

    def _batch_has_repair_candidates(edge: WorkflowEdgeInfo, state: WorkflowState, result: object) -> bool:
        del edge, result
        context = _context(state)
        review = state.get("current_layer_review")
        result = state.get("current_layer_result")
        if not isinstance(review, dict) or not isinstance(result, dict):
            return False
        if review.get("metadata", {}).get("review_failure"):
            return False
        parent_ids = context.get("parent_node_ids")
        children = result.get("children")
        if not isinstance(parent_ids, list) or len(parent_ids) < 2 or not isinstance(children, list):
            return False
        return any(
            isinstance(child, dict) and child.get("parent_node_id") in parent_ids
            for child in children
        )

    def _satisfied(edge: WorkflowEdgeInfo, state: WorkflowState, result: object) -> bool:
        del edge, result
        review = state.get("current_layer_review")
        result = state.get("current_layer_result")
        if not isinstance(review, dict) or not isinstance(result, dict):
            return False
        return (
            review.get("coverage_ok") is True
            and not review.get("metadata", {}).get("review_failure")
            and bool(result.get("satisfied", False))
            and not review.get("overlap_conflicts")
            and not review.get("coverage_gap_notes")
            and not review.get("duplicate_child_notes")
        )

    return {
        "parse_strategy_layer_excerpt": _selected_strategy,
        "parse_strategy_layer_boundary": _selected_strategy,
        "parse_strategy_page_index": _selected_strategy,
        "strategy_attempted": lambda edge, state, result: True,
        "layer_satisfied": _satisfied,
        "strategy_failed_with_remaining": _failed_with_remaining,
        "all_strategies_exhausted": _exhausted,
        "strategy_selection_failed": _strategy_selection_failed,
        "commit_candidates_valid": _commit_candidates_valid,
        "commit_candidates_invalid": _commit_candidates_invalid,
        "batch_has_repair_candidates": _batch_has_repair_candidates,
    }


# Keep the old private name available for callers and tests that imported it
# before the helper became part of the runtime construction API.
_workflow_predicates = workflow_predicates


@dataclass(slots=True)
class _RunCompat:
    run_id: str
    final_state: Mapping[str, object]
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
    deps: Mapping[str, object] | None = None,
) -> WorkflowRuntime:
    resolver = build_ingest_step_resolver(deps=deps)
    return WorkflowRuntime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=resolver.resolve,
        predicate_registry=workflow_predicates(),
        trace=False,
    )


def run_ingest_workflow(
    *,
    inp: WorkflowIngestInput,
    workflow_engine: GraphKnowledgeEngine,
    conversation_engine: GraphKnowledgeEngine,
    knowledge_engine: GraphKnowledgeEngine | None = None,
    workflow_id: str = DEFAULT_WORKFLOW_ID,
    provider_settings: WorkflowProviderSettings | None = None,
    deps: Mapping[str, object] | None = None,
    run_id: str | None = None,
    resume_from_checkpoint: bool = False,
) -> tuple[_RunCompat, WorkflowExportBundle | None]:
    effective_deps = dict(deps or {})
    if provider_settings is not None:
        effective_deps.setdefault("provider_settings", provider_settings)
    client = DirectRuntimeIngestClient(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
    )
    result = client.run_ingest(
        inp=inp,
        workflow_id=workflow_id,
        deps=effective_deps,
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
