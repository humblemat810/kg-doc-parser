from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TypedDict, cast

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.storage_backend import StorageBackend
from kogwistar.runtime.contract import Predicate as RuntimeDesignPredicate
from kogwistar.runtime.contract import WorkflowEdgeInfo
from kogwistar.runtime.runtime import WorkflowRuntime
from kogwistar.typing_interfaces import EmbeddingFunctionLike

from .clients import DirectRuntimeIngestClient
from .design import DEFAULT_WORKFLOW_ID
from .handlers import build_ingest_step_resolver
from .models import IngestRunResult, WorkflowExportBundle, WorkflowIngestInput
from .providers import WorkflowProviderSettings, build_embedding_function
from .runtime_contracts import WorkflowRuntimeDeps
from .serialization import JsonValue

WorkflowState = Mapping[str, object]


class _WorkflowPredicate(Protocol):
    """Parser-side predicate shape before the runtime adapter boundary."""

    def __call__(
        self,
        edge: WorkflowEdgeInfo,
        state: WorkflowState,
        result: object,
    ) -> bool: ...


WorkflowPredicates = dict[str, _WorkflowPredicate]


class _LayerContext(TypedDict, total=False):
    metadata: _LayerMetadata
    parent_node_ids: list[str]


class _LayerMetadata(TypedDict, total=False):
    parse_strategy: str
    disabled_strategies: list[str]
    commit_validation: bool
    review_failure: str


class _LayerReview(TypedDict, total=False):
    metadata: _LayerMetadata
    satisfied: bool
    coverage_ok: bool
    overlap_conflicts: list[JsonValue]
    coverage_gap_notes: list[JsonValue]
    duplicate_child_notes: list[JsonValue]
    structural_issues: list[JsonValue]
    validation_diagnostics: list[JsonValue]


class _LayerChild(TypedDict, total=False):
    parent_node_id: str


class _LayerResult(TypedDict, total=False):
    satisfied: bool
    children: list[_LayerChild]


def _string_set(value: object) -> set[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return set()
    return {item for item in value if isinstance(item, str)}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, str)]


def _mapping(value: object) -> Mapping[str, JsonValue]:
    """Narrow a persisted layer payload to the JSON value union."""

    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return cast(Mapping[str, JsonValue], value)
    return {}


def _metadata_mapping(value: object) -> _LayerMetadata:
    raw = _mapping(value)
    metadata: _LayerMetadata = {}
    parse_strategy = raw.get("parse_strategy")
    if isinstance(parse_strategy, str):
        metadata["parse_strategy"] = parse_strategy
    disabled_strategies = _string_list(raw.get("disabled_strategies"))
    if disabled_strategies:
        metadata["disabled_strategies"] = disabled_strategies
    commit_validation = raw.get("commit_validation")
    if isinstance(commit_validation, bool):
        metadata["commit_validation"] = commit_validation
    review_failure = raw.get("review_failure")
    if isinstance(review_failure, str):
        metadata["review_failure"] = review_failure
    return metadata


def _context(state: WorkflowState) -> _LayerContext:
    raw = _mapping(state.get("current_layer_context"))
    context: _LayerContext = {}
    parent_node_ids = _string_list(raw.get("parent_node_ids"))
    if parent_node_ids:
        context["parent_node_ids"] = parent_node_ids
    metadata = _metadata_mapping(raw.get("metadata"))
    if metadata:
        context["metadata"] = metadata
    return context


def _metadata(state: WorkflowState) -> _LayerMetadata:
    return _metadata_mapping(_context(state).get("metadata"))


def _review(state: WorkflowState) -> _LayerReview:
    raw = _mapping(state.get("current_layer_review"))
    review: _LayerReview = {}
    metadata = _metadata_mapping(raw.get("metadata"))
    if metadata:
        review["metadata"] = metadata
    for field_name in ("satisfied", "coverage_ok"):
        value = raw.get(field_name)
        if isinstance(value, bool):
            review[field_name] = value
    for field_name in (
        "overlap_conflicts",
        "coverage_gap_notes",
        "duplicate_child_notes",
        "structural_issues",
        "validation_diagnostics",
    ):
        value = raw.get(field_name)
        if isinstance(value, list):
            review[field_name] = value
    return review


def _result(state: WorkflowState) -> _LayerResult:
    raw = _mapping(state.get("current_layer_result"))
    result: _LayerResult = {}
    satisfied = raw.get("satisfied")
    if isinstance(satisfied, bool):
        result["satisfied"] = satisfied
    children = _children(raw.get("children"))
    if children:
        result["children"] = children
    return result


def _children(value: object) -> list[_LayerChild]:
    if not isinstance(value, list):
        return []
    children: list[_LayerChild] = []
    for child in value:
        if not isinstance(child, Mapping):
            continue
        parent_node_id = child.get("parent_node_id")
        if isinstance(parent_node_id, str):
            children.append({"parent_node_id": parent_node_id})
    return children


def workflow_predicates() -> WorkflowPredicates:
    """Guards for parser strategy transitions.

    These are deliberately derived from persisted state only.  The provider
    never controls the route directly, and a retry cannot re-enable a method
    that the satisfaction check disabled.
    """

    def _strategy_name(edge: WorkflowEdgeInfo) -> str:
        target = str(edge.dst).split("|")[-1]
        return {
            "layer_excerpt_method": "layer_excerpt",
            "layer_boundary_method": "layer_boundary",
            "page_index_layer": "page_index",
        }.get(target, "")

    def _selected_strategy(edge: WorkflowEdgeInfo, state: WorkflowState, result: object) -> bool:
        del result
        return str(_metadata(state).get("parse_strategy", "")) == _strategy_name(edge)

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
        review = _review(state)
        metadata = _mapping(review.get("metadata"))
        return bool(metadata.get("commit_validation")) and review.get("satisfied") is True

    def _commit_candidates_invalid(edge: WorkflowEdgeInfo, state: WorkflowState, result: object) -> bool:
        del edge, result
        review = _review(state)
        metadata = _mapping(review.get("metadata"))
        return bool(metadata.get("commit_validation")) and review.get("satisfied") is False

    def _batch_has_repair_candidates(edge: WorkflowEdgeInfo, state: WorkflowState, result: object) -> bool:
        del edge, result
        context = _context(state)
        review = _review(state)
        result = _result(state)
        if _mapping(review.get("metadata")).get("review_failure"):
            return False
        parent_ids = set(_string_list(context.get("parent_node_ids")))
        children = _children(result.get("children"))
        if len(parent_ids) < 2 or not children:
            return False
        return any(child.get("parent_node_id") in parent_ids for child in children)

    def _satisfied(edge: WorkflowEdgeInfo, state: WorkflowState, result: object) -> bool:
        del edge, result
        review = _review(state)
        result = _result(state)
        metadata = _mapping(review.get("metadata"))
        return (
            review.get("coverage_ok") is True
            and not metadata.get("review_failure")
            and bool(result.get("satisfied", False))
            and not review.get("overlap_conflicts")
            and not review.get("coverage_gap_notes")
            and not review.get("duplicate_child_notes")
            and not review.get("structural_issues")
            and not review.get("validation_diagnostics")
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

    def __call__(self, input: list[str], /) -> list[list[float]]:
        vectors = []
        for value in input:
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
        predicate_registry=cast(
            dict[str, RuntimeDesignPredicate], workflow_predicates()
        ),
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
    effective_deps = cast(WorkflowRuntimeDeps, dict(deps or {}))
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
