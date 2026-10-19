"""Kogwistar-runtime execution for OCR grouping and derived summaries.

The SQLite store remains an artifact/cache index.  The durable workflow state,
step trace, and resume frontier are owned by ``WorkflowRuntime``.  Provider
callables and the state store are injected dependencies and never serialized
into checkpoint state.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Protocol,
    TypedDict,
    cast,
)

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.runtime import MappingStepResolver
from kogwistar.runtime.contract import Predicate as RuntimeDesignPredicate
from kogwistar.runtime.models import (
    RunFailure,
    RunSuccess,
    StateUpdate,
    StepRunResult,
    WorkflowInvocationRequest,
)
from kogwistar.runtime.runtime import (
    RunResult,
    StepContext,
    WorkflowRuntime,
)

from .design import (
    DEFAULT_OCR_GROUPING_WORKFLOW_ID,
    DEFAULT_OCR_PAGE_WORKFLOW_ID,
    DEFAULT_OCR_WORKFLOW_ID,
    DEFAULT_WORKFLOW_ID,
    ensure_ingest_workflow_design,
    ensure_ocr_grouping_workflow_design,
    ensure_ocr_page_workflow_design,
    ensure_ocr_workflow_design,
)
from .document_adapters import (
    apply_grouping_proposals,
    attach_document_tree_metadata,
    suggest_cross_page_continuations,
)
from .document_grouping import (
    GroupProposal,
    GroupProposalBatch,
    GroupProposalGenerator,
    build_grouping_pass_context,
    grouping_frontier_windows,
    grouping_policy_identity,
    resolve_grouping_proposals,
    validate_grouping_window_proposals,
)
from .document_models import AdaptiveGroupingPolicy, DocumentTree, SummaryPolicy
from .handlers import build_ingest_step_resolver
from .models import WorkflowExportBundle, WorkflowIngestInput
from .ocr_pipeline import (
    GroundedSummaryGenerator,
    OCRPageJSON,
    OCRPagePreparation,
    OCRPageProcessingContext,
    OCRSourcePlan,
    OCRWorkflowArtifacts,
    OCRWorkflowStateStore,
    SummaryGenerator,
    _finalize_ocr_workflow_artifacts,
    _generate_tree_summaries,
    _process_ocr_page,
    _summary_generator_identity,
    _sync_progress_from_state,
    _write_document_tree,
    _write_summary,
    build_production_grounded_summary_generator,
    build_production_group_proposal_generator,
)
from .ocr_runtime_contracts import (
    OCRCheckpointPatch as _OCRCheckpointPatch,
)
from .ocr_runtime_contracts import (
    OCRWorkflowState as _OCRWorkflowState,
)
from .ocr_runtime_contracts import (
    ResumeIdentityKey,
    ResumeIdentityValue,
)
from .ocr_runtime_state import (
    int_list as _int_list,
)
from .ocr_runtime_state import (
    json_pages as _json_pages,
)
from .ocr_runtime_state import (
    page_cursor as _page_cursor,
)
from .ocr_runtime_state import (
    page_sources as _page_sources,
)
from .ocr_runtime_state import (
    resume_identity_value as _resume_identity_value,
)
from .ocr_runtime_state import (
    state_bool as _state_bool,
)
from .ocr_runtime_state import (
    state_optional_string as _state_optional_string,
)
from .ocr_runtime_state import (
    state_string as _state_string,
)
from .ocr_runtime_state import (
    typed_child_state as _typed_child_state,
)
from .ocr_runtime_state import (
    typed_state as _typed_state,
)
from .runtime_contracts import WorkflowRuntimeDeps
from .service import workflow_predicates

if TYPE_CHECKING:
    from .providers import WorkflowProviderSettings


class StepResolver(Protocol):
    def __call__(self, op: str) -> Callable[[StepContext], StepRunResult]: ...

_OCR_ROOT_STATE_VERSION = "ocr-root-v3"
_OCR_PAGE_STATE_VERSION = "ocr-page-v2"


class KogwistarRuntimeCompatibilityError(RuntimeError):
    """Raised when a requested OCR recovery contract is not available."""


class _WorkflowRuntimeArgs(TypedDict):
    workflow_engine: GraphKnowledgeEngine
    conversation_engine: GraphKnowledgeEngine
    step_resolver: StepResolver
    predicate_registry: dict[str, RuntimeDesignPredicate]
    trace: bool


class _WorkflowRuntimeLeaseArgs(_WorkflowRuntimeArgs, total=False):
    run_ownership_lease_ms: int


def _workflow_runtime(
    *,
    workflow_engine: GraphKnowledgeEngine,
    conversation_engine: GraphKnowledgeEngine,
    step_resolver: StepResolver,
    predicate_registry: dict[str, RuntimeDesignPredicate],
    trace: bool,
    run_ownership_lease_ms: int | None = None,
) -> WorkflowRuntime:
    """Construct a runtime across released Kogwistar API generations.

    Lease-aware ownership is required whenever the caller requests it.
    Kogwistar releases before the lease constructor argument may still be used
    for non-recovery workflows, but must not silently downgrade a recovery run
    to an unowned execution.  Do not catch arbitrary constructor ``TypeError``
    values: inspect the public signature and fail at this dependency boundary.
    """
    runtime_args: _WorkflowRuntimeArgs = {
        "workflow_engine": workflow_engine,
        "conversation_engine": conversation_engine,
        "step_resolver": step_resolver,
        "predicate_registry": predicate_registry,
        "trace": trace,
    }
    if run_ownership_lease_ms is not None:
        try:
            supports_lease = "run_ownership_lease_ms" in inspect.signature(
                WorkflowRuntime
            ).parameters
        except (TypeError, ValueError):
            supports_lease = False
        if not supports_lease:
            raise KogwistarRuntimeCompatibilityError(
                "Kogwistar WorkflowRuntime does not support "
                "run_ownership_lease_ms; install the lease-aware release "
                "before running resumable OCR workflows"
            )
        lease_args: _WorkflowRuntimeLeaseArgs = {
            **runtime_args,
            "run_ownership_lease_ms": run_ownership_lease_ms,
        }
        return cast(Callable[..., WorkflowRuntime], WorkflowRuntime)(**lease_args)
    return cast(Callable[..., WorkflowRuntime], WorkflowRuntime)(**runtime_args)


def _checkpoint_repair_cursor(patch: Mapping[str, object]) -> int:
    """Read the required rewind cursor and fail closed on malformed state."""

    cursor = patch.get("page_cursor")
    if type(cursor) is not int or cursor < 0:
        raise ValueError("OCR checkpoint repair requires a non-negative page_cursor")
    return cursor


def _success(
    next_step: str | None,
    *,
    state_update: list[StateUpdate] | None = None,
) -> RunSuccess:
    return RunSuccess(
        conversation_node_id=None,
        state_update=state_update or [],
        _route_next=[] if next_step is None else [next_step],
    )


def _tree(state: _OCRWorkflowState) -> DocumentTree:
    payload = state.get("tree")
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise TypeError("OCR grouping runtime state has no tree")
    return DocumentTree.model_validate(payload)


def _proposals(value: object) -> tuple[GroupProposal, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(GroupProposal.model_validate(item) for item in value)


def _pass_count(state: _OCRWorkflowState) -> int:
    value = state.get("pass_index", 0)
    return value if type(value) is int and value >= 0 else 0


def _adaptive_round(state: _OCRWorkflowState) -> int:
    value = state.get("adaptive_grouping_round", 0)
    if type(value) is not int or value < 0:
        raise ValueError("adaptive grouping round must be a non-negative integer")
    return value


def _adaptive_window(state: _OCRWorkflowState) -> int:
    value = state.get("adaptive_grouping_window", 0)
    if type(value) is not int or value < 0:
        raise ValueError("adaptive grouping window cursor must be a non-negative integer")
    return value


def _adaptive_grouping_complete(tree: DocumentTree, *, policy_identity: str) -> bool:
    root = next(node for node in tree.nodes if node.node_id == tree.root_id)
    return root.annotations.get("grouping:adaptive_complete") == policy_identity


def _mark_adaptive_grouping_complete(
    tree: DocumentTree, *, policy_identity: str
) -> DocumentTree:
    root = next(node for node in tree.nodes if node.node_id == tree.root_id)
    updated_root = root.model_copy(
        update={
            "annotations": {
                **root.annotations,
                "grouping:adaptive_complete": policy_identity,
            }
        }
    )
    nodes = tuple(
        updated_root if node.node_id == tree.root_id else node for node in tree.nodes
    )
    return tree.model_copy(update={"nodes": nodes})


def _recovery_generation(state: _OCRWorkflowState) -> int:
    value = state.get("ocr_page_recovery_generation", 0)
    if value is None:
        return 0
    if type(value) is not int or value < 0:
        raise ValueError(
            "OCR page recovery generation must be a non-negative integer"
        )
    return value


def _validate_root_resume_identity(
    state: _OCRWorkflowState,
    *,
    document_id: str,
    source_fingerprint: str,
    source_namespace: str,
    configuration_fingerprints: Sequence[str],
    grouping_policy_version: str,
    parse_generation: str,
    require_match: bool,
) -> None:
    """Reject a root checkpoint built from a different immutable input set."""
    if not require_match:
        return
    expected: dict[ResumeIdentityKey, ResumeIdentityValue] = {
        "ocr_root_state_version": _OCR_ROOT_STATE_VERSION,
        "document_id": document_id,
        "source_fingerprint": source_fingerprint,
        "source_namespace": source_namespace,
        "ocr_configuration_fingerprints": list(configuration_fingerprints),
        "grouping_policy_version": grouping_policy_version,
        "parse_generation": parse_generation,
    }
    missing = [key for key in expected if key not in state]
    if missing:
        raise RuntimeError(
            "cannot resume OCR root checkpoint without immutable identity fields: "
            + ", ".join(sorted(missing))
        )
    mismatches = [
        key
        for key, expected_value in expected.items()
        if _resume_identity_value(state, key) != expected_value
    ]
    if mismatches:
        raise RuntimeError(
            "OCR root checkpoint identity mismatch; refusing to mix persisted state for: "
            + ", ".join(sorted(mismatches))
        )


def _validate_root_invocation_lineage(
    conversation_engine: GraphKnowledgeEngine,
    *,
    root_run_id: str,
    recovery_generation: int,
) -> None:
    """Require one persisted edge for each accepted root child workflow.

    The workflow result and artifact bundle prove that outputs exist, but not
    that page, grouping, and semantic execution belonged to this persisted
    root.  The runtime's ``wf_invoked`` edges are the durable lineage record;
    duplicate or missing edges must fail closed before graph publication.
    """
    suffix = f"|recovery-{recovery_generation}" if recovery_generation else ""
    expected = {
        (
            "kg_doc_parser.ocr_pages.v1",
            "ocr-pages-v1" if not recovery_generation else f"ocr-pages-recovery-v{recovery_generation}",
            f"{root_run_id}|ocr-pages{suffix}",
        ),
        (
            "kg_doc_parser.ocr_grouping.v1",
            "grouping-v1" if not recovery_generation else f"grouping-recovery-v{recovery_generation}",
            f"{root_run_id}|grouping{suffix}",
        ),
        (
            "kg_doc_parser.ingest.v1",
            "semantic-v1" if not recovery_generation else f"semantic-recovery-v{recovery_generation}",
            f"{root_run_id}|semantic{suffix}",
        ),
    }
    expected_edge_ids = [
        f"wf_invoked|{root_run_id}|{child_run_id}"
        for _workflow_id, _invocation_key, child_run_id in expected
    ]
    counts: dict[tuple[str, str, str], int] = {}
    for edge in conversation_engine.read.get_edges(
        ids=expected_edge_ids,
        limit=len(expected_edge_ids),
    ):
        metadata = edge.metadata or {}
        if metadata.get("relation") != "wf_invoked":
            continue
        if metadata.get("parent_run_id") != root_run_id:
            continue
        key = (
            str(metadata.get("workflow_id")),
            str(metadata.get("invocation_key")),
            str(metadata.get("child_run_id")),
        )
        counts[key] = counts.get(key, 0) + 1
    missing = expected - counts.keys()
    duplicate = sorted(key for key, count in counts.items() if key in expected and count != 1)
    if missing or duplicate:
        details: list[str] = []
        if missing:
            details.append(f"missing={sorted(missing)!r}")
        if duplicate:
            details.append(f"duplicate={duplicate!r}")
        raise RuntimeError(
            f"OCR root persisted invocation lineage is invalid for {root_run_id!r}: "
            + "; ".join(details)
        )


def _reused_pages_from_trace(
    conversation_engine: GraphKnowledgeEngine, run_id: str
) -> list[int]:
    """Rebuild the compatibility reuse summary from compact per-step deltas."""
    nodes = list(
        conversation_engine.read.get_nodes(
            where={
                "$and": [
                    {"entity_type": "workflow_step_exec"},
                    {"run_id": run_id},
                ]
            },
            limit=None,
        )
    )
    outcomes: dict[int, tuple[int, int, int, bool]] = {}
    for node in nodes:
        metadata = node.metadata or {}
        result_raw = metadata.get("result_json")
        if not isinstance(result_raw, str):
            continue
        try:
            result = json.loads(result_raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(result, dict):
            continue
        updates = result.get("state_update", [])
        if not isinstance(updates, list):
            continue
        step_sequence = metadata.get("step_seq", 0)
        if type(step_sequence) is not int:
            step_sequence = 0
        attempt = metadata.get("attempt", 0)
        if type(attempt) is not int:
            attempt = 0
        for update_index, update in enumerate(updates):
            if not isinstance(update, (list, tuple)) or len(update) != 2:
                continue
            operation, payload = update
            if not isinstance(payload, dict):
                continue
            page_cursor = payload.get("page_cursor")
            if (
                type(page_cursor) is int
                and page_cursor >= 1
                and "last_reused_page" in payload
            ):
                reused_page = payload.get("last_reused_page")
                outcomes[page_cursor] = (
                    step_sequence,
                    attempt,
                    update_index,
                    reused_page == page_cursor,
                )
            reused_page = payload.get("last_reused_page")
            if type(reused_page) is int and reused_page >= 1:
                outcomes[reused_page] = (
                    step_sequence,
                    attempt,
                    update_index,
                    True,
                )
            # Read pre-compaction append deltas when resuming an old run.
            if operation == "a":
                legacy_page = payload.get("reused_pages")
                if type(legacy_page) is int and legacy_page >= 1:
                    outcomes[legacy_page] = (
                        step_sequence,
                        attempt,
                        update_index,
                        True,
                    )
    return [
        page_number
        for page_number, (_step, _attempt, _update, was_reused) in sorted(
            outcomes.items(), key=lambda item: item[1][:3]
        )
        if was_reused
    ]


def _page_workflow_initial_state(source_plan: OCRSourcePlan) -> _OCRWorkflowState:
    return {
        "ocr_page_state_version": _OCR_PAGE_STATE_VERSION,
        "source_fingerprint": source_plan.input_fingerprint,
        "legacy_dir": str(source_plan.legacy_dir),
        "page_cursor": 0,
    }


def _validate_page_resume_identity(
    state: _OCRWorkflowState,
    *,
    document_id: str,
    source_fingerprint: str,
    configuration_fingerprints: Sequence[str],
) -> None:
    """Reject a page checkpoint built from a different immutable input set."""
    expected: dict[ResumeIdentityKey, ResumeIdentityValue] = {
        "ocr_page_state_version": _OCR_PAGE_STATE_VERSION,
        "document_id": document_id,
        "source_fingerprint": source_fingerprint,
        "ocr_configuration_fingerprints": list(configuration_fingerprints),
    }
    missing = [key for key in expected if key not in state]
    if missing:
        raise RuntimeError(
            "cannot resume OCR page checkpoint without immutable identity fields: "
            + ", ".join(sorted(missing))
        )
    mismatches = [
        key
        for key, expected_value in expected.items()
        if _resume_identity_value(state, key) != expected_value
    ]
    if mismatches:
        raise RuntimeError(
            "OCR page checkpoint identity mismatch; refusing to mix persisted state for: "
            + ", ".join(sorted(mismatches))
        )


def _reconcile_page_checkpoint(
    *,
    state: _OCRWorkflowState,
    context: OCRPageProcessingContext,
    source_plan: OCRSourcePlan,
    start_index: int = 0,
) -> _OCRCheckpointPatch | None:
    """Find the first checkpointed page whose accepted artifact is no longer valid.

    Runtime checkpoints and the SQLite/artifact cache are separate durable
    stores.  A crash can commit one without the other, or an operator can
    tamper with an artifact between resumes.  Never advance past that seam:
    truncate the in-memory checkpoint to the verified prefix and let the
    normal page step replay the first unaccepted page.
    """
    cursor = _page_cursor(state)
    # New checkpoints keep the immutable source manifest in the invocation
    # context, not in every Kogwistar state snapshot. Older checkpoints remain
    # readable through their persisted page_sources field.
    sources = (
        _page_sources(state)
        if "page_sources" in state
        else source_plan.page_sources
    )
    completed = _int_list(state, "completed_pages") if "completed_pages" in state else []
    reused = _int_list(state, "reused_pages") if "reused_pages" in state else []
    if start_index < 0 or start_index > cursor:
        raise ValueError("OCR checkpoint reconciliation start index is out of range")
    for index in range(start_index, cursor):
        page_number, image_path = sources[index]
        image_digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        page_path = source_plan.legacy_dir / f"page_{page_number}.json"
        if context.state_store.reconcile_orphan_artifact(
            document_id=context.document_id,
            page_number=page_number,
            content_hash=image_digest,
            artifact_path=page_path,
            config_fingerprints=context.configuration_fingerprints,
        ):
            continue
        if context.state_store.should_skip_page(
            document_id=context.document_id,
            page_number=page_number,
            stage="ocr",
            content_hash=image_digest,
            artifact_path=page_path,
            config_fingerprint=context.configuration_fingerprints,
        ):
            continue
        repair: _OCRCheckpointPatch = {"page_cursor": index}
        if "completed_pages" in state:
            repair["completed_pages"] = completed[:index]
        if "reused_pages" in state:
            repair["reused_pages"] = [
                number for number in reused[:index] if number in completed[:index]
            ]
        # Preserve the old in-checkpoint payload only when repairing a legacy
        # checkpoint. New checkpoints stay artifact-backed and compact.
        if "raw_pages" in state:
            repair["raw_pages"] = _json_pages(
                state, source_plan=source_plan
            )[:index]
        return repair
    return None


def _validate_root_page_result(
    *,
    page_result: _OCRWorkflowState,
    context: OCRPageProcessingContext,
    source_plan: OCRSourcePlan,
) -> None:
    """Fail closed if a terminal page child no longer owns valid artifacts.

    A parent resume can skip a terminal child invocation.  Recheck the
    artifact/SQLite seam here before treating the serialized child result as
    accepted output; otherwise a deleted or tampered page artifact could be
    replaced by stale checkpoint data without rerunning OCR.
    """
    sources = source_plan.page_sources
    cursor = _page_cursor(page_result)
    expected_pages = [page_number for page_number, _image_path in sources]
    legacy_completed = (
        _int_list(page_result, "completed_pages")
        if "completed_pages" in page_result
        else []
    )
    completed = legacy_completed or expected_pages[:cursor]
    if cursor != len(sources) or completed != expected_pages:
        raise RuntimeError("OCR root page child completed with an incomplete page checkpoint")
    repair = _reconcile_page_checkpoint(
        state=page_result,
        context=context,
        source_plan=source_plan,
    )
    if repair is not None:
        raise RuntimeError(
            "OCR root page checkpoint is no longer backed by accepted artifacts; "
            "resume must retry the page child"
        )


def _semantic_parse_generation(recovery_generation: int) -> str:
    """Return the immutable semantic-parser generation for one root run."""

    return (
        "semantic-v1"
        if recovery_generation == 0
        else f"semantic-recovery-v{recovery_generation}"
    )
def _page_result_needs_recovery(
    *,
    page_result: _OCRWorkflowState,
    context: OCRPageProcessingContext,
    source_plan: OCRSourcePlan,
) -> bool:
    """Report whether a terminal page child must be regenerated."""
    sources = source_plan.page_sources
    expected_pages = [page_number for page_number, _image_path in sources]
    legacy_completed = (
        _int_list(page_result, "completed_pages")
        if "completed_pages" in page_result
        else []
    )
    completed = legacy_completed or expected_pages[: _page_cursor(page_result)]
    if (
        _page_cursor(page_result) != len(sources)
        or completed != expected_pages
    ):
        return True
    return _reconcile_page_checkpoint(
        state=page_result,
        context=context,
        source_plan=source_plan,
    ) is not None


def _root_page_artifacts_need_recovery(
    *,
    context: OCRPageProcessingContext,
    source_plan: OCRSourcePlan,
) -> bool:
    """Detect a broken accepted page set before a terminal root resume.

    Kogwistar correctly treats a succeeded run as terminal. When an artifact
    is lost after that terminal checkpoint, create a new root incarnation
    rather than asking the runtime to re-execute a completed run.
    """
    for page_number, image_path in source_plan.page_sources:
        image_digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        page_path = source_plan.legacy_dir / f"page_{page_number}.json"
        context.state_store.reconcile_orphan_artifact(
            document_id=context.document_id,
            page_number=page_number,
            content_hash=image_digest,
            artifact_path=page_path,
            config_fingerprints=context.configuration_fingerprints,
        )
        if not context.state_store.should_skip_page(
            document_id=context.document_id,
            page_number=page_number,
            stage="ocr",
            content_hash=image_digest,
            artifact_path=page_path,
            config_fingerprint=context.configuration_fingerprints,
        ):
            return True
    return False


def _validate_existing_page_identities(
    *,
    context: OCRPageProcessingContext,
    source_plan: OCRSourcePlan,
) -> None:
    """Reject provider changes before recovery can replay any page.

    An incomplete page child may require a fresh root incarnation because a
    completed runtime cannot be re-executed.  That recovery path must still
    honor the original OCR configuration identity before it schedules work.
    """
    expected = set(context.configuration_fingerprints)
    for page_number, _image_path in source_plan.page_sources:
        state = context.state_store.get_page_state(
            document_id=context.document_id,
            page_number=page_number,
            stage="ocr",
        )
        if (
            state is not None
            and state.config_fingerprint is not None
            and state.config_fingerprint not in expected
        ):
            raise RuntimeError(
                "OCR root checkpoint identity mismatch; refusing to mix persisted "
                "OCR configuration for page "
                f"{page_number}"
            )


def _next_root_recovery_run_id(
    *,
    conversation_engine: GraphKnowledgeEngine,
    base_run_id: str,
) -> tuple[str, int]:
    """Choose an unused deterministic root recovery incarnation."""
    generation = 1
    while True:
        candidate = f"{base_run_id}|recovery-{generation}"
        completed = conversation_engine.read.get_nodes(
            where={
                "$and": [
                    {"entity_type": "workflow_completed"},
                    {"run_id": candidate},
                ]
            },
            limit=1,
        )
        if not completed:
            return candidate, generation
        generation += 1


def _build_page_resolver(
    *,
    context: OCRPageProcessingContext,
    source_plan: OCRSourcePlan,
    progress_path: Path,
) -> MappingStepResolver:
    """Build the page child resolver for both standalone and nested runs."""
    resolver = MappingStepResolver()
    # The checkpoint prefix is checked once when execution reaches it, then
    # each newly accepted page is checked on the next loop iteration. A fresh
    # resolver on process restart naturally revalidates the restored prefix.
    verified_prefix_cursor = 0
    resolver.set_state_schema(
        {
            "ocr_page_state_version": "u",
            "document_id": "u",
            "source_fingerprint": "u",
            "ocr_configuration_fingerprints": "u",
            "page_sources": "u",
            "legacy_dir": "u",
            "page_cursor": "u",
            "raw_pages": "u",
            "completed_pages": "u",
            "reused_pages": "u",
            "last_failed_page": "u",
            "last_reused_page": "u",
        }
    )

    @resolver.register("start")
    def _start(ctx: StepContext) -> StepRunResult:
        _validate_page_resume_identity(
            _typed_state(ctx.state_view),
            document_id=context.document_id,
            source_fingerprint=source_plan.input_fingerprint,
            configuration_fingerprints=context.configuration_fingerprints,
        )
        return _success("process_ocr_page")

    @resolver.register("process_ocr_page")
    def _process(ctx: StepContext) -> StepRunResult:
        nonlocal verified_prefix_cursor
        state = _typed_state(ctx.state_view)
        repair = _reconcile_page_checkpoint(
            state=state,
            context=context,
            source_plan=source_plan,
            start_index=verified_prefix_cursor,
        )
        if repair is not None:
            verified_prefix_cursor = _checkpoint_repair_cursor(repair)
            repair_update: StateUpdate = ("u", dict(repair))
            return _success("process_ocr_page", state_update=[repair_update])
        sources = source_plan.page_sources
        cursor = _page_cursor(state)
        verified_prefix_cursor = cursor
        if cursor >= len(sources):
            return _success("finalize_ocr_pages")
        page_number, image_path = sources[cursor]
        outcome = _process_ocr_page(
            context=context,
            done_count=cursor + 1,
            page_number=page_number,
            image_path=image_path,
            image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
            page_json_path=source_plan.legacy_dir / f"page_{page_number}.json",
            page_image_path=source_plan.legacy_dir / f"page_{page_number}{image_path.suffix or '.png'}",
        )
        if outcome.status == "failed":
            return RunFailure(
                conversation_node_id=None,
                state_update=[("u", {"last_failed_page": page_number})],
                errors=[f"OCR page {page_number} failed after all candidates"],
            )
        if outcome.page_number != page_number or outcome.page is None:
            raise RuntimeError("OCR page processing produced an inconsistent outcome")

        updates: list[StateUpdate] = [
            (
                "u",
                {
                    "page_cursor": cursor + 1,
                    "last_reused_page": (
                        page_number if outcome.status == "reused" else None
                    ),
                },
            ),
        ]
        # Stop carrying historical cumulative page lists in the new runtime
        # state. Older list deltas remain in Kogwistar's step trace and can be
        # interpreted when returning compatibility summaries.
        if "completed_pages" in state or "reused_pages" in state:
            updates.append(("u", {"completed_pages": [], "reused_pages": []}))
        # Old checkpoints may contain cumulative raw_pages. Once the page
        # artifact has been accepted, drop that payload and use the artifact
        # directory for subsequent reads and checkpoints.
        if state.get("raw_pages") is not None:
            updates.append(("u", {"raw_pages": None}))
        verified_prefix_cursor = cursor + 1
        return _success("process_ocr_page", state_update=updates)

    @resolver.register("finalize_ocr_pages")
    def _finalize(ctx: StepContext) -> StepRunResult:
        nonlocal verified_prefix_cursor
        state = _typed_state(ctx.state_view)
        repair = _reconcile_page_checkpoint(
            state=state,
            context=context,
            source_plan=source_plan,
        )
        if repair is not None:
            verified_prefix_cursor = _checkpoint_repair_cursor(repair)
            return _success("process_ocr_page", state_update=[("u", dict(repair))])
        context.state_store.refresh_document_completion(document_id=context.document_id)
        # SQLite is the live page-status source; the full JSON mirror is an
        # O(page-count) snapshot, so publish it once after the page loop.
        _sync_progress_from_state(
            state_store=context.state_store,
            document_id=context.document_id,
            title=context.title,
            source_kind=context.source_kind,
            total_pages=context.total_pages,
            progress_path=progress_path,
        )
        return _success("end")

    @resolver.register("end")
    def _end(ctx: StepContext) -> StepRunResult:
        del ctx
        return _success(None)
    return resolver


def run_ocr_page_workflow(
    *,
    workflow_engine: GraphKnowledgeEngine,
    conversation_engine: GraphKnowledgeEngine,
    context: OCRPageProcessingContext,
    source_plan: OCRSourcePlan,
    progress_path: Path,
    run_id: str,
    resume_from_checkpoint: bool = False,
) -> tuple[list[OCRPageJSON], list[int], list[int], str]:
    """Process OCR pages as persisted Kogwistar steps.

    Provider clients and the SQLite store stay in the injected context. Only
    JSON page identity, cursor, and accepted artifact payloads enter runtime
    checkpoints.
    """
    ensure_ocr_page_workflow_design(workflow_engine)
    resolver = _build_page_resolver(
        context=context,
        source_plan=source_plan,
        progress_path=progress_path,
    )

    runtime = WorkflowRuntime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=resolver.resolve,
        predicate_registry={},
        trace=True,
    )
    initial_state = _page_workflow_initial_state(source_plan)
    initial_state.update(
        {
            "document_id": context.document_id,
            "ocr_configuration_fingerprints": list(context.configuration_fingerprints),
        }
    )
    result = (
        runtime.resume_from_latest_checkpoint(
            run_id=run_id,
            workflow_id=DEFAULT_OCR_PAGE_WORKFLOW_ID,
            conversation_id=f"ocr:{context.document_id}",
            turn_node_id=f"ocr:{context.document_id}:pages",
        )
        if resume_from_checkpoint
        else runtime.run(
            workflow_id=DEFAULT_OCR_PAGE_WORKFLOW_ID,
            conversation_id=f"ocr:{context.document_id}",
            turn_node_id=f"ocr:{context.document_id}:pages",
            initial_state=dict(initial_state),
            run_id=run_id,
        )
    )
    if result.status != "succeeded":
        raise RuntimeError(f"OCR page workflow {result.status}: {result.errors}")
    final_state = _typed_state(result.final_state)
    return (
        _json_pages(final_state, source_plan=source_plan),
        [page_number for page_number, _path in source_plan.page_sources],
        _reused_pages_from_trace(conversation_engine, result.run_id),
        result.run_id,
    )


def _validated_grouping_checkpoint(
    *,
    state_store: OCRWorkflowStateStore,
    document_id: str,
    tree: DocumentTree,
    grouping_policy_version: str,
) -> tuple[DocumentTree | None, str | None]:
    """Validate persisted grouping state before a workflow can resume.

    A terminal runtime checkpoint may return without executing the parser's
    restore step. Validate the SQLite tree/frontier seam before requesting
    that resume so terminal state cannot bypass corruption detection.
    """
    try:
        restored = state_store.load_document_tree(
            document_id=document_id,
            source_fingerprint=tree.source_fingerprint(),
            grouping_policy_version=grouping_policy_version,
        )
    except ValueError as exc:
        if "frontier" in str(exc):
            raise RuntimeError(
                "grouping checkpoint frontier does not match the restored tree"
            ) from exc
        raise RuntimeError(f"persisted grouping checkpoint is invalid: {exc}") from exc
    revision = state_store.document_tree_revision(document_id=document_id)
    if restored is None:
        if revision is not None:
            raise RuntimeError(
                "grouping checkpoint exists but does not match the current source or policy"
            )
        return None, None

    persisted_frontier = state_store.load_grouping_frontier(
        document_id=document_id,
        source_fingerprint=restored.source_fingerprint(),
        grouping_policy_version=grouping_policy_version,
    )
    if persisted_frontier is None:
        raise RuntimeError(
            "grouping checkpoint exists but its frontier is missing or mismatched"
        )
    restored_root = next(
        node for node in restored.nodes if node.node_id == restored.root_id
    )
    expected_frontier = (
        tuple(child.node_id for child in restored_root.children)
        if restored_root.children
        else (restored_root.node_id,)
    )
    if persisted_frontier != expected_frontier:
        raise RuntimeError("grouping checkpoint frontier does not match the restored tree")
    return restored, revision


def _register_grouping_steps(
    resolver: MappingStepResolver,
    *,
    state_store: OCRWorkflowStateStore,
    summary_policy: SummaryPolicy | None,
    summary_generator: SummaryGenerator | None,
    grounded_summary_generator: GroundedSummaryGenerator | None,
    adaptive_grouping_policy: AdaptiveGroupingPolicy,
    grouping_proposer: GroupProposalGenerator | None,
) -> None:
    @resolver.register("start")
    def _start(ctx: StepContext) -> StepRunResult:
        return _success("restore_grouping_checkpoint")

    @resolver.register("restore_grouping_checkpoint")
    def _restore(ctx: StepContext) -> StepRunResult:
        state = _typed_state(ctx.state_view)
        tree = _tree(state)
        document_id = _state_string(state, "document_id", tree.document_id)
        policy_version = _state_string(state, "grouping_policy_version", "baseline-v1")
        restored, revision = _validated_grouping_checkpoint(
            state_store=state_store,
            document_id=document_id,
            tree=tree,
            grouping_policy_version=policy_version,
        )
        if restored is not None:
            payload = restored.model_dump(mode="json")
            return _success(
                "apply_grouping_pass",
                state_update=[
                    (
                        "u",
                        {
                            "tree": payload,
                            "checkpoint_restored": True,
                            "checkpoint_absent": False,
                            "checkpoint_revision": revision,
                            "adaptive_grouping_complete": _adaptive_grouping_complete(
                                restored, policy_identity=policy_version
                            ),
                        },
                    )
                ],
            )
        if state_store.document_tree_revision(document_id=document_id) is not None:
            raise RuntimeError("grouping checkpoint exists but does not match the current source or policy")
        return _success("apply_grouping_pass")

    @resolver.register("apply_grouping_pass")
    def _apply(ctx: StepContext) -> StepRunResult:
        state = _typed_state(ctx.state_view)
        tree = _tree(state)
        pass_index = _pass_count(state)
        raw_passes = state.get("grouping_passes", [])
        passes = raw_passes if isinstance(raw_passes, list) else []
        if pass_index < len(passes):
            proposals = _proposals(passes[pass_index])
            policy_version = _state_string(state, "grouping_policy_version", "baseline-v1")
            tree = apply_grouping_proposals(
                tree,
                proposals,
                grouping_policy_version=f"{policy_version}:{pass_index + 1}",
            )
            return _success(
                "apply_grouping_pass",
                state_update=[
                    (
                        "u",
                        {"tree": tree.model_dump(mode="json"), "pass_index": pass_index + 1},
                    )
                ],
            )
        if not adaptive_grouping_policy.enabled or _state_bool(
            state, "adaptive_grouping_complete"
        ):
            return _success("generate_grouping_summaries")
        if grouping_proposer is None:
            raise RuntimeError(
                "adaptive grouping is enabled without a proposal generator"
            )

        policy_version = _state_string(state, "grouping_policy_version", "baseline-v1")
        round_index = _adaptive_round(state)
        history = state.get("adaptive_grouping_history", [])
        if round_index >= adaptive_grouping_policy.maximum_rounds:
            completed_tree = _mark_adaptive_grouping_complete(
                tree, policy_identity=policy_version
            )
            history.append(
                {
                    "round": round_index,
                    "outcome": "maximum_rounds_reached",
                }
            )
            return _success(
                "generate_grouping_summaries",
                state_update=[
                    (
                        "u",
                        {
                            "tree": completed_tree.model_dump(mode="json"),
                            "adaptive_grouping_complete": True,
                            "adaptive_grouping_history": history,
                        },
                    )
                ],
            )

        root = next(node for node in tree.nodes if node.node_id == tree.root_id)
        frontier = tuple(child.node_id for child in root.children)
        windows = grouping_frontier_windows(frontier, adaptive_grouping_policy)
        window_index = _adaptive_window(state)
        pending_window_proposals = state.get("adaptive_grouping_window_proposals", [])
        if window_index > len(windows):
            raise ValueError("adaptive grouping window cursor exceeds the current frontier")
        if window_index < len(windows):
            window = windows[window_index]
            context = build_grouping_pass_context(
                tree,
                pass_index=round_index,
                frontier_ids=window,
                maximum_evidence_items_per_candidate=(
                    adaptive_grouping_policy.maximum_evidence_items_per_candidate
                ),
                maximum_text_chars_per_candidate=(
                    adaptive_grouping_policy.maximum_text_chars_per_candidate
                ),
            )
            batch = grouping_proposer(context)
            if not isinstance(batch, GroupProposalBatch):
                raise TypeError(
                    "grouping proposal generator must return GroupProposalBatch"
                )
            validated_window_proposals = validate_grouping_window_proposals(
                frontier=frontier,
                window=window,
                proposals=batch.proposals,
            )
            pending_window_proposals.extend(
                proposal.model_dump(mode="json")
                for proposal in validated_window_proposals
            )
            return _success(
                "apply_grouping_pass",
                state_update=[
                    (
                        "u",
                        {
                            "adaptive_grouping_window": window_index + 1,
                            "adaptive_grouping_window_proposals": pending_window_proposals,
                        },
                    )
                ],
            )

        proposals = resolve_grouping_proposals(
            frontier=frontier,
            proposals=_proposals(pending_window_proposals),
        )
        if not proposals:
            completed_tree = _mark_adaptive_grouping_complete(
                tree, policy_identity=policy_version
            )
            history.append(
                {
                    "round": round_index + 1,
                    "input_frontier": list(frontier),
                    "window_count": len(windows),
                    "proposals": [],
                    "outcome": "no_supported_groups",
                }
            )
            return _success(
                "generate_grouping_summaries",
                state_update=[
                    (
                        "u",
                        {
                            "tree": completed_tree.model_dump(mode="json"),
                            "adaptive_grouping_complete": True,
                            "adaptive_grouping_history": history,
                        },
                    )
                ],
            )

        next_tree = apply_grouping_proposals(
            tree,
            proposals,
            grouping_policy_version=f"{policy_version}:adaptive:{round_index + 1}",
        )
        next_root = next(node for node in next_tree.nodes if node.node_id == next_tree.root_id)
        if len(next_root.children) >= len(frontier):
            raise RuntimeError("adaptive grouping proposal did not reduce the frontier")
        history.append(
            {
                "round": round_index + 1,
                "input_frontier": list(frontier),
                "proposals": [proposal.model_dump(mode="json") for proposal in proposals],
                "output_frontier": [child.node_id for child in next_root.children],
                "outcome": "groups_accepted",
            }
        )
        return _success(
            "apply_grouping_pass",
            state_update=[
                (
                    "u",
                    {
                        "tree": next_tree.model_dump(mode="json"),
                        "adaptive_grouping_round": round_index + 1,
                        "adaptive_grouping_window": 0,
                        "adaptive_grouping_window_proposals": [],
                        "adaptive_grouping_history": history,
                    },
                )
            ],
        )

    @resolver.register("generate_grouping_summaries")
    def _summaries(ctx: StepContext) -> StepRunResult:
        state = _typed_state(ctx.state_view)
        tree = _tree(state)
        document_id = _state_string(state, "document_id", tree.document_id)
        policy_version = _state_string(state, "grouping_policy_version", "baseline-v1")
        checkpoint_revision = _state_optional_string(state, "checkpoint_revision")
        updated = _generate_tree_summaries(
            tree,
            document_id=document_id,
            state_store=state_store,
            policy=summary_policy,
            generator=summary_generator,
            grounded_generator=grounded_summary_generator,
            grouping_policy_version=policy_version,
            expected_tree_revision=(
                checkpoint_revision
                if checkpoint_revision is not None
                else None
            ),
            require_no_existing_tree=_state_bool(state, "checkpoint_absent"),
        )
        current_revision = state_store.document_tree_revision(document_id=document_id)
        return _success(
            "persist_grouping_result",
            state_update=[
                (
                    "u",
                    {
                        "tree": updated.model_dump(mode="json"),
                        "summaries_generated": (summary_generator is not None
                        or grounded_summary_generator is not None)
                        and bool(summary_policy and summary_policy.enabled),
                        "checkpoint_revision": current_revision,
                        "checkpoint_absent": current_revision is None,
                    },
                )
            ],
        )

    @resolver.register("persist_grouping_result")
    def _persist(ctx: StepContext) -> StepRunResult:
        state = _typed_state(ctx.state_view)
        tree = _tree(state)
        checkpoint_revision = _state_optional_string(state, "checkpoint_revision")
        revision = state_store.save_document_tree(
            document_id=_state_string(state, "document_id", tree.document_id),
            tree=tree,
            grouping_policy_version=_state_string(state, "grouping_policy_version", "baseline-v1"),
            expected_tree_revision=(
                checkpoint_revision
                if checkpoint_revision is not None
                else None
            ),
            require_no_existing_tree=_state_bool(state, "checkpoint_absent"),
            last_runtime_step="persist_grouping_result",
        )
        return _success("end", state_update=[("u", {"checkpoint_revision": revision})])

    @resolver.register("end")
    def _end(ctx: StepContext) -> StepRunResult:
        del ctx
        return _success(None)


def run_grouping_workflow(
    *,
    workflow_engine: GraphKnowledgeEngine,
    conversation_engine: GraphKnowledgeEngine,
    state_store: OCRWorkflowStateStore,
    tree: DocumentTree,
    document_id: str,
    grouping_policy_version: str,
    grouping_proposals: Sequence[GroupProposal] = (),
    grouping_passes: Sequence[Sequence[GroupProposal]] = (),
    summary_policy: SummaryPolicy | None = None,
    summary_generator: SummaryGenerator | None = None,
    grounded_summary_generator: GroundedSummaryGenerator | None = None,
    adaptive_grouping_policy: AdaptiveGroupingPolicy | None = None,
    grouping_proposer: GroupProposalGenerator | None = None,
    run_id: str | None = None,
    resume_from_checkpoint: bool = False,
    run_ownership_lease_ms: int | None = None,
) -> tuple[DocumentTree, str]:
    """Run grouping as a persisted Kogwistar workflow and return its tree."""
    if grouping_proposals and grouping_passes:
        raise ValueError("provide grouping_proposals or grouping_passes, not both")

    explicit_passes = list(grouping_passes)
    if grouping_proposals:
        explicit_passes = [tuple(grouping_proposals)]
    effective_adaptive_policy = adaptive_grouping_policy or AdaptiveGroupingPolicy(
        enabled=grouping_proposer is not None
    )
    if effective_adaptive_policy.enabled != (grouping_proposer is not None):
        raise ValueError(
            "enabled adaptive grouping requires exactly one grouping proposer"
        )
    effective_policy_version = grouping_policy_version
    if effective_adaptive_policy.enabled:
        if grouping_proposer is None:
            raise RuntimeError("adaptive grouping proposer was not constructed")
        proposer_fingerprint = grouping_proposer.configuration_fingerprint
        if not proposer_fingerprint:
            raise ValueError("grouping proposer configuration fingerprint cannot be empty")
        effective_policy_version = grouping_policy_identity(
            policy_version=grouping_policy_version,
            policy=effective_adaptive_policy,
            proposer_fingerprint=proposer_fingerprint,
        )
    if not explicit_passes:
        explicit_passes = [suggest_cross_page_continuations(tree)]
    ensure_ocr_grouping_workflow_design(workflow_engine)
    resolver = MappingStepResolver()
    _register_grouping_steps(
        resolver,
        state_store=state_store,
        summary_policy=summary_policy,
        summary_generator=summary_generator,
        grounded_summary_generator=grounded_summary_generator,
        adaptive_grouping_policy=effective_adaptive_policy,
        grouping_proposer=grouping_proposer,
    )
    runtime = _workflow_runtime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=resolver.resolve,
        predicate_registry={},
        trace=True,
        run_ownership_lease_ms=run_ownership_lease_ms,
    )
    effective_run_id = run_id or f"ocr-grouping|{document_id}"
    if resume_from_checkpoint:
        _validated_grouping_checkpoint(
            state_store=state_store,
            document_id=document_id,
            tree=tree,
            grouping_policy_version=effective_policy_version,
        )
    existing_revision = state_store.document_tree_revision(document_id=document_id)
    result = (
        runtime.resume_from_latest_checkpoint(
            run_id=effective_run_id,
            workflow_id=DEFAULT_OCR_GROUPING_WORKFLOW_ID,
            conversation_id=f"ocr:{document_id}",
            turn_node_id=f"ocr:{document_id}:grouping",
        )
        if resume_from_checkpoint
        else runtime.run(
            workflow_id=DEFAULT_OCR_GROUPING_WORKFLOW_ID,
            conversation_id=f"ocr:{document_id}",
            turn_node_id=f"ocr:{document_id}:grouping",
            initial_state={
                "document_id": document_id,
                "tree": tree.model_dump(mode="json"),
                "grouping_policy_version": effective_policy_version,
                "grouping_passes": [
                    [proposal.model_dump(mode="json") for proposal in current_pass]
                    for current_pass in explicit_passes
                ],
                "pass_index": 0,
                "adaptive_grouping_round": 0,
                "adaptive_grouping_window": 0,
                "adaptive_grouping_history": [],
                "adaptive_grouping_window_proposals": [],
                "adaptive_grouping_complete": False,
                "checkpoint_revision": existing_revision,
                "checkpoint_absent": existing_revision is None,
            },
            run_id=effective_run_id,
        )
    )
    if result.status != "succeeded":
        raise RuntimeError(f"OCR grouping workflow {result.status}: {result.errors}")
    return _tree(_typed_state(result.final_state)), result.run_id


def run_ocr_root_workflow(
    *,
    workflow_engine: GraphKnowledgeEngine,
    conversation_engine: GraphKnowledgeEngine,
    knowledge_engine: GraphKnowledgeEngine | None,
    preparation: OCRPagePreparation,
    provider_settings: WorkflowProviderSettings,
    document_id: str,
    title: str,
    grouping_policy_version: str,
    source_namespace: str | None = None,
    grouping_proposals: Sequence[GroupProposal] = (),
    grouping_passes: Sequence[Sequence[GroupProposal]] = (),
    summary_policy: SummaryPolicy | None = None,
    summary_generator: SummaryGenerator | None = None,
    grounded_summary_generator: GroundedSummaryGenerator | None = None,
    adaptive_grouping_policy: AdaptiveGroupingPolicy | None = None,
    grouping_proposer: GroupProposalGenerator | None = None,
    deps: WorkflowRuntimeDeps | None = None,
    run_id: str | None = None,
    resume_from_checkpoint: bool = False,
    run_ownership_lease_ms: int | None = None,
) -> tuple[RunResult, WorkflowExportBundle | None, OCRWorkflowArtifacts]:
    """Run OCR, grouping, and semantic parsing under one persisted parent.

    Each expensive stage remains a separately versioned child workflow.  The
    parent stores explicit invocation records and stable child run IDs, so a
    restart resumes the same lineage instead of merely concatenating unrelated
    run names.
    """
    if grouping_proposals and grouping_passes:
        raise ValueError("provide grouping_proposals or grouping_passes, not both")

    if source_namespace is not None and (
        not source_namespace.strip()
        or source_namespace != source_namespace.strip()
    ):
        raise ValueError(
            "OCR source_namespace must be non-empty and have no outer whitespace"
        )
    source_namespace_identity = source_namespace or ""

    from .clients import DirectRuntimeIngestClient
    from .providers import WorkflowProviderSettings

    if not isinstance(provider_settings, WorkflowProviderSettings):
        raise TypeError("provider_settings must be WorkflowProviderSettings")
    effective_adaptive_policy = adaptive_grouping_policy or AdaptiveGroupingPolicy(
        enabled=(
            grouping_proposer is not None
            or provider_settings.parser.provider != "fake"
        )
    )
    if effective_adaptive_policy.enabled and grouping_proposer is None:
        if provider_settings.parser.provider == "fake":
            raise ValueError(
                "adaptive grouping is enabled with the fake parser provider; "
                "supply a test proposer or disable adaptive grouping"
            )
        grouping_proposer = build_production_group_proposal_generator(provider_settings)
    elif not effective_adaptive_policy.enabled and grouping_proposer is not None:
        raise ValueError("grouping proposer was supplied while adaptive grouping is disabled")
    effective_summary_policy = summary_policy
    effective_grounded_summary_generator = grounded_summary_generator
    if (
        effective_adaptive_policy.enabled
        and provider_settings.parser.provider != "fake"
    ):
        effective_summary_policy = effective_summary_policy or SummaryPolicy()
        if summary_generator is None and effective_grounded_summary_generator is None:
            effective_grounded_summary_generator = (
                build_production_grounded_summary_generator(provider_settings)
            )
    if effective_adaptive_policy.enabled:
        if grouping_proposer is None:
            raise RuntimeError("adaptive grouping proposal generator was not constructed")
        proposer_fingerprint = grouping_proposer.configuration_fingerprint
        if not proposer_fingerprint:
            raise ValueError("grouping proposer configuration fingerprint cannot be empty")
        summary_fingerprint: str | None = None
        active_summary_generator = (
            summary_generator
            if summary_generator is not None
            else effective_grounded_summary_generator
        )
        if (
            effective_summary_policy is not None
            and effective_summary_policy.enabled
            and active_summary_generator is not None
        ):
            summary_identity_payload = json.dumps(
                [
                    effective_summary_policy.model_dump(mode="json"),
                    _summary_generator_identity(active_summary_generator),
                ],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            summary_fingerprint = hashlib.sha256(
                summary_identity_payload.encode("utf-8")
            ).hexdigest()
        grouping_policy_version = grouping_policy_identity(
            policy_version=grouping_policy_version,
            policy=effective_adaptive_policy,
            proposer_fingerprint=proposer_fingerprint,
            summary_fingerprint=summary_fingerprint,
        )
    explicit_passes = list(grouping_passes)
    if grouping_proposals:
        explicit_passes = [tuple(grouping_proposals)]

    ensure_ocr_workflow_design(workflow_engine)
    ensure_ocr_page_workflow_design(workflow_engine)
    ensure_ocr_grouping_workflow_design(workflow_engine)
    ensure_ingest_workflow_design(workflow_engine, workflow_id=DEFAULT_WORKFLOW_ID)

    page_resolver = _build_page_resolver(
        context=preparation.page_context,
        source_plan=preparation.source_plan,
        progress_path=preparation.progress_path,
    )
    grouping_resolver = MappingStepResolver()
    _register_grouping_steps(
        grouping_resolver,
        state_store=preparation.state_store,
        summary_policy=effective_summary_policy,
        summary_generator=summary_generator,
        grounded_summary_generator=effective_grounded_summary_generator,
        adaptive_grouping_policy=effective_adaptive_policy,
        grouping_proposer=grouping_proposer,
    )
    graph_client = DirectRuntimeIngestClient(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
    )
    semantic_deps = {
        "knowledge_engine": knowledge_engine,
        "persistence_mode": "local_debug",
        "kg_authority": "local",
        "graph_persistence_client": graph_client,
        "provider_settings": provider_settings,
        **dict(deps or {}),
    }
    semantic_resolver = build_ingest_step_resolver(deps=semantic_deps)

    root_resolver = MappingStepResolver()
    prepared_page_payloads: list[OCRPageJSON] | None = None
    prepared_reused_pages: list[int] | None = None
    root_resolver.set_state_schema(
        {
            "ocr_page_result": "u",
            "ocr_page_recovery_generation": "u",
            "workflow_input": "u",
            "tree": "u",
            "grouping_result": "u",
            "semantic_result": "u",
            "semantic_tree": "u",
            "export_bundle": "u",
            "ocr_root_state_version": "u",
            "ocr_root_lineage_id": "u",
            "source_fingerprint": "u",
            "source_namespace": "u",
            "ocr_configuration_fingerprints": "u",
            "parse_generation": "u",
            "checkpoint_revision": "u",
            "checkpoint_absent": "u",
            "grouping_passes": "u",
            "adaptive_grouping_round": "u",
            "adaptive_grouping_window": "u",
            "adaptive_grouping_history": "u",
            "adaptive_grouping_window_proposals": "u",
            "adaptive_grouping_complete": "u",
            **page_resolver._state_schema,
            **semantic_resolver._state_schema,
        }
    )

    @root_resolver.register("ocr_root_start")
    def _root_start(ctx: StepContext) -> StepRunResult:
        del ctx
        return _success("ocr_root_invoke_pages")

    @root_resolver.register("ocr_root_invoke_pages", is_nested=True)
    def _root_invoke_pages(ctx: StepContext) -> StepRunResult:
        state = _typed_state(ctx.state_view)
        recovery_generation = _recovery_generation(state)
        recovery_suffix = (
            f"|recovery-{recovery_generation}" if recovery_generation else ""
        )
        child_run_id = f"{ctx.run_id}|ocr-pages{recovery_suffix}"
        return RunSuccess(
            conversation_node_id=None,
            state_update=[],
            workflow_invocations=[
                WorkflowInvocationRequest(
                    workflow_id=DEFAULT_OCR_PAGE_WORKFLOW_ID,
                    initial_state={
                        **_page_workflow_initial_state(preparation.source_plan),
                        "document_id": preparation.page_context.document_id,
                        "ocr_configuration_fingerprints": list(
                            preparation.page_context.configuration_fingerprints
                        ),
                    },
                    result_state_key="ocr_page_result",
                    run_id=child_run_id,
                    invocation_key=(
                        "ocr-pages-v1"
                        if not recovery_generation
                        else f"ocr-pages-recovery-v{recovery_generation}"
                    ),
                    conversation_id=f"ocr:{document_id}",
                    turn_node_id=f"ocr:{document_id}:pages",
                )
            ],
            _route_next=["ocr_root_prepare_grouping_input"],
        )

    @root_resolver.register("ocr_root_prepare_grouping_input")
    def _root_prepare_grouping(ctx: StepContext) -> StepRunResult:
        nonlocal prepared_page_payloads, prepared_reused_pages
        state = _typed_state(ctx.state_view)
        page_result = _typed_child_state(
            state.get("ocr_page_result"),
            label="OCR page child",
        )
        if _page_result_needs_recovery(
            page_result=page_result,
            context=preparation.page_context,
            source_plan=preparation.source_plan,
        ):
            generation = _recovery_generation(state) + 1
            return _success(
                "ocr_root_invoke_pages",
                state_update=[("u", {"ocr_page_recovery_generation": generation})],
            )
        raw_pages = _json_pages(
            page_result, source_plan=preparation.source_plan
        )
        prepared_page_payloads = raw_pages
        completed_pages = [
            page_number
            for page_number, _image_path in preparation.source_plan.page_sources
        ]
        if "completed_pages" in page_result:
            legacy_completed = _int_list(page_result, "completed_pages")
            if legacy_completed:
                completed_pages = legacy_completed
        recovery_generation = _recovery_generation(state)
        recovery_suffix = (
            f"|recovery-{recovery_generation}" if recovery_generation else ""
        )
        page_run_id = f"{ctx.run_id}|ocr-pages{recovery_suffix}"
        reused_pages = _reused_pages_from_trace(conversation_engine, page_run_id)
        prepared_reused_pages = reused_pages
        if "reused_pages" in page_result:
            legacy_reused = _int_list(page_result, "reused_pages")
            if legacy_reused:
                reused_pages = legacy_reused
        artifacts = _finalize_ocr_workflow_artifacts(
            document_id=document_id,
            title=title,
            raw_pages=raw_pages,
            completed_pages=completed_pages,
            reused_pages=reused_pages,
            rendered_dir=preparation.source_plan.rendered_dir,
            legacy_dir=preparation.source_plan.legacy_dir,
            state_db_path=preparation.state_db_path,
            progress_path=preparation.progress_path,
            summary_path=preparation.summary_path,
            provider_settings=provider_settings,
            state_store=preparation.state_store,
            restore_grouping_checkpoint=False,
            persist_grouping_checkpoint=False,
            publish_artifacts=False,
            grouping_policy_version=grouping_policy_version,
        )
        if artifacts.document_tree is None:
            raise ValueError("OCR normalization produced no document tree")
        if not explicit_passes:
            passes = [suggest_cross_page_continuations(artifacts.document_tree)]
        else:
            passes = explicit_passes
        return _success(
            "ocr_root_invoke_grouping",
            state_update=[
                (
                    "u",
                    {
                        "workflow_input": artifacts.workflow_input.model_dump(
                            field_mode="backend", dump_format="json"
                        ),
                        "tree": artifacts.document_tree.model_dump(mode="json"),
                        "grouping_passes": [
                            [proposal.model_dump(mode="json") for proposal in current_pass]
                            for current_pass in passes
                        ],
                        "completed_pages": completed_pages,
                        "reused_pages": reused_pages,
                    },
                )
            ],
        )

    @root_resolver.register("ocr_root_invoke_grouping", is_nested=True)
    def _root_invoke_grouping(ctx: StepContext) -> StepRunResult:
        state = _typed_state(ctx.state_view)
        tree = _tree(state)
        existing_revision = preparation.state_store.document_tree_revision(document_id=document_id)
        recovery_generation = _recovery_generation(state)
        recovery_suffix = (
            f"|recovery-{recovery_generation}" if recovery_generation else ""
        )
        return RunSuccess(
            conversation_node_id=None,
            state_update=[],
            workflow_invocations=[
                WorkflowInvocationRequest(
                    workflow_id=DEFAULT_OCR_GROUPING_WORKFLOW_ID,
                    initial_state={
                        "document_id": document_id,
                        "tree": tree.model_dump(mode="json"),
                        "grouping_policy_version": grouping_policy_version,
                        "grouping_passes": state.get("grouping_passes", []),
                        "pass_index": 0,
                        "adaptive_grouping_round": 0,
                        "adaptive_grouping_window": 0,
                        "adaptive_grouping_history": [],
                        "adaptive_grouping_window_proposals": [],
                        "adaptive_grouping_complete": False,
                        "checkpoint_revision": existing_revision,
                        "checkpoint_absent": existing_revision is None,
                    },
                    result_state_key="grouping_result",
                    run_id=f"{ctx.run_id}|grouping{recovery_suffix}",
                    invocation_key=(
                        "grouping-v1"
                        if not recovery_generation
                        else f"grouping-recovery-v{recovery_generation}"
                    ),
                    conversation_id=f"ocr:{document_id}",
                    turn_node_id=f"ocr:{document_id}:grouping",
                )
            ],
            _route_next=["ocr_root_prepare_semantic_input"],
        )

    @root_resolver.register("ocr_root_prepare_semantic_input")
    def _root_prepare_semantic(ctx: StepContext) -> StepRunResult:
        state = _typed_state(ctx.state_view)
        grouping_result = _typed_child_state(
            state.get("grouping_result"),
            label="grouping child",
        )
        grouped_tree = _tree(grouping_result)
        workflow_input_payload = state.get("workflow_input")
        if not isinstance(workflow_input_payload, dict):
            raise TypeError("OCR root has no persisted semantic workflow input")
        workflow_input = WorkflowIngestInput.model_validate(workflow_input_payload)
        primary_collection = workflow_input.collections[0]
        workflow_input = workflow_input.model_copy(
            update={
                "collections": [
                    collection.model_copy(update={"source_namespace": source_namespace})
                    if collection.collection_id == primary_collection.collection_id
                    else collection
                    for collection in workflow_input.collections
                ]
            }
        )
        attach_document_tree_metadata(workflow_input, grouped_tree)
        return _success(
            "ocr_root_invoke_semantic",
            state_update=[
                (
                    "u",
                    {
                        "tree": grouped_tree.model_dump(mode="json"),
                        "workflow_input": workflow_input.model_dump(
                            field_mode="backend", dump_format="json"
                        ),
                    },
                )
            ],
        )

    @root_resolver.register("ocr_root_invoke_semantic", is_nested=True)
    def _root_invoke_semantic(ctx: StepContext) -> StepRunResult:
        state = _typed_state(ctx.state_view)
        input_payload = state.get("workflow_input")
        if not isinstance(input_payload, dict):
            raise TypeError("OCR root has no semantic workflow input")
        recovery_generation = _recovery_generation(state)
        parse_generation = state.get("parse_generation")
        expected_parse_generation = _semantic_parse_generation(recovery_generation)
        if parse_generation != expected_parse_generation:
            raise RuntimeError(
                "OCR root parse-generation identity mismatch; refusing to invoke "
                f"semantic parser with {parse_generation!r}, expected "
                f"{expected_parse_generation!r}"
            )
        recovery_suffix = (
            f"|recovery-{recovery_generation}" if recovery_generation else ""
        )
        return RunSuccess(
            conversation_node_id=None,
            state_update=[],
            workflow_invocations=[
                WorkflowInvocationRequest(
                    workflow_id=DEFAULT_WORKFLOW_ID,
                    initial_state={"input": input_payload},
                    result_state_key="semantic_result",
                    run_id=f"{ctx.run_id}|semantic{recovery_suffix}",
                    invocation_key=parse_generation,
                    conversation_id=f"ingest:{document_id}",
                    turn_node_id=f"ingest:{document_id}:turn",
                )
            ],
            _route_next=["ocr_root_finalize"],
        )

    @root_resolver.register("ocr_root_finalize")
    def _root_finalize(ctx: StepContext) -> StepRunResult:
        state = _typed_state(ctx.state_view)
        semantic_result = _typed_child_state(
            state.get("semantic_result"),
            label="semantic child",
        )
        bundle = semantic_result.get("export_bundle")
        if not isinstance(bundle, dict):
            raise TypeError("semantic child did not publish an export bundle")
        grouped_tree = _tree(state)
        exported_grouping_tree = bundle.get("grouping_tree")
        if not isinstance(exported_grouping_tree, dict):
            raise TypeError("semantic child did not preserve the accepted grouping tree")
        if exported_grouping_tree != grouped_tree.model_dump(mode="json"):
            raise ValueError("semantic child grouping tree differs from the accepted OCR grouping")
        semantic_tree = semantic_result.get("semantic_tree")
        if not isinstance(semantic_tree, dict):
            raise TypeError("semantic child did not publish a semantic tree")
        return _success(
            "ocr_root_end",
            state_update=[
                (
                    "u",
                    {
                        "export_bundle": bundle,
                        "semantic_tree": semantic_tree,
                    },
                )
            ],
        )

    @root_resolver.register("ocr_root_end")
    def _root_end(ctx: StepContext) -> StepRunResult:
        del ctx
        return _success(None)

    # Child workflow designs reuse operation names such as ``start`` and
    # ``end``. Dispatch by persisted workflow ID, not by a mutable global mode.
    root_ops = set(root_resolver.ops)
    child_ops = page_resolver.ops | grouping_resolver.ops | semantic_resolver.ops
    dispatch = MappingStepResolver()
    dispatch.set_state_schema(root_resolver._state_schema)

    for op in root_ops | child_ops:
        is_nested = op in {"ocr_root_invoke_pages", "ocr_root_invoke_grouping", "ocr_root_invoke_semantic"}

        @dispatch.register(op, is_nested=is_nested)
        def _dispatch(ctx: StepContext, operation: str = op) -> StepRunResult:
            if ctx.workflow_id == DEFAULT_OCR_WORKFLOW_ID:
                _validate_root_resume_identity(
                    _typed_state(ctx.state_view),
                    document_id=document_id,
                    source_fingerprint=preparation.source_plan.input_fingerprint,
                    source_namespace=source_namespace_identity,
                    configuration_fingerprints=preparation.page_context.configuration_fingerprints,
                    grouping_policy_version=grouping_policy_version,
                    parse_generation=_semantic_parse_generation(recovery_generation),
                    require_match=resume_from_checkpoint,
                )
                return root_resolver.resolve(operation)(ctx)
            if ctx.workflow_id == DEFAULT_OCR_PAGE_WORKFLOW_ID:
                return page_resolver.resolve(operation)(ctx)
            if ctx.workflow_id == DEFAULT_OCR_GROUPING_WORKFLOW_ID:
                return grouping_resolver.resolve(operation)(ctx)
            if ctx.workflow_id == DEFAULT_WORKFLOW_ID:
                return semantic_resolver.resolve(operation)(ctx)
            raise ValueError(f"unsupported OCR workflow child {ctx.workflow_id!r}")

    runtime = _workflow_runtime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=dispatch.resolve,
        predicate_registry=cast(
            dict[str, RuntimeDesignPredicate], workflow_predicates()
        ),
        trace=True,
        run_ownership_lease_ms=run_ownership_lease_ms,
    )
    effective_run_id = run_id or f"ocr|{document_id}"
    root_resume = resume_from_checkpoint
    recovery_generation = 0
    if resume_from_checkpoint:
        _validate_existing_page_identities(
            context=preparation.page_context,
            source_plan=preparation.source_plan,
        )
    if resume_from_checkpoint and _root_page_artifacts_need_recovery(
        context=preparation.page_context,
        source_plan=preparation.source_plan,
    ):
        effective_run_id, recovery_generation = _next_root_recovery_run_id(
            conversation_engine=conversation_engine,
            base_run_id=effective_run_id,
        )
        root_resume = False
    result = (
        runtime.resume_from_latest_checkpoint(
            run_id=effective_run_id,
            workflow_id=DEFAULT_OCR_WORKFLOW_ID,
            conversation_id=f"ocr:{document_id}",
            turn_node_id=f"ocr:{document_id}:root",
        )
        if root_resume
        else runtime.run(
            workflow_id=DEFAULT_OCR_WORKFLOW_ID,
            conversation_id=f"ocr:{document_id}",
            turn_node_id=f"ocr:{document_id}:root",
            initial_state={
                "document_id": document_id,
                "title": title,
                "ocr_root_state_version": _OCR_ROOT_STATE_VERSION,
                "source_fingerprint": preparation.source_plan.input_fingerprint,
                "source_namespace": source_namespace_identity,
                "ocr_configuration_fingerprints": list(
                    preparation.page_context.configuration_fingerprints
                ),
                "grouping_policy_version": grouping_policy_version,
                "parse_generation": _semantic_parse_generation(recovery_generation),
                "ocr_page_recovery_generation": recovery_generation,
                "ocr_root_lineage_id": run_id or f"ocr|{document_id}",
                "grouping_proposals": [proposal.model_dump(mode="json") for proposal in grouping_proposals],
                "grouping_passes": [],
            },
            run_id=effective_run_id,
        )
    )
    if result.status != "succeeded":
        raise RuntimeError(f"OCR root workflow {result.status}: {result.errors}")
    if effective_run_id != (run_id or f"ocr|{document_id}"):
        # Recovery uses a fresh persisted incarnation because released runtimes
        # treat a terminal run as immutable. Keep that storage identity internal;
        # callers continue to observe the stable logical root identity.
        result = replace(result, run_id=run_id or f"ocr|{document_id}")
    final_state = _typed_state(result.final_state)
    _validate_root_resume_identity(
        final_state,
        document_id=document_id,
        source_fingerprint=preparation.source_plan.input_fingerprint,
        source_namespace=source_namespace_identity,
        configuration_fingerprints=preparation.page_context.configuration_fingerprints,
        grouping_policy_version=grouping_policy_version,
        parse_generation=_semantic_parse_generation(_recovery_generation(final_state)),
        require_match=True,
    )
    _validate_root_invocation_lineage(
        conversation_engine,
        root_run_id=effective_run_id,
        recovery_generation=_recovery_generation(final_state),
    )
    page_result = _typed_child_state(
        final_state.get("ocr_page_result"),
        label="OCR root page child",
    )
    if prepared_page_payloads is None:
        # A terminal root resume skips its resolver steps. Revalidate and
        # materialize that accepted generation once here; normal execution
        # reuses the exact page snapshot already consumed by grouping.
        _validate_root_page_result(
            page_result=page_result,
            context=preparation.page_context,
            source_plan=preparation.source_plan,
        )
        raw_pages = _json_pages(
            page_result, source_plan=preparation.source_plan
        )
    else:
        raw_pages = prepared_page_payloads
    completed_pages = [
        page_number
        for page_number, _image_path in preparation.source_plan.page_sources
    ]
    if "completed_pages" in page_result:
        legacy_completed = _int_list(page_result, "completed_pages")
        if legacy_completed:
            completed_pages = legacy_completed
    recovery_generation = _recovery_generation(final_state)
    recovery_suffix = (
        f"|recovery-{recovery_generation}" if recovery_generation else ""
    )
    if prepared_page_payloads is not None:
        reused_pages = prepared_reused_pages or []
    else:
        page_run_id = f"{effective_run_id}|ocr-pages{recovery_suffix}"
        reused_pages = _reused_pages_from_trace(conversation_engine, page_run_id)
    if "reused_pages" in page_result:
        legacy_reused = _int_list(page_result, "reused_pages")
        if legacy_reused:
            reused_pages = legacy_reused
    # A terminal child invocation is not executed again during parent resume,
    # so its prior reuse list is not regenerated in the child result state.
    # Reconcile the parent-facing artifact summary from the accepted page set;
    # this is reporting only and never turns an incomplete page into success.
    if resume_from_checkpoint and completed_pages and not reused_pages:
        reused_pages = list(completed_pages)
    grouped_tree = _tree(final_state)
    artifacts = _finalize_ocr_workflow_artifacts(
        document_id=document_id,
        title=title,
        raw_pages=raw_pages,
        completed_pages=completed_pages,
        reused_pages=reused_pages,
        rendered_dir=preparation.source_plan.rendered_dir,
        legacy_dir=preparation.source_plan.legacy_dir,
        state_db_path=preparation.state_db_path,
        progress_path=preparation.progress_path,
        summary_path=preparation.summary_path,
        provider_settings=provider_settings,
        state_store=preparation.state_store,
        restore_grouping_checkpoint=False,
        persist_grouping_checkpoint=False,
        publish_artifacts=False,
        grouping_policy_version=grouping_policy_version,
    )
    workflow_input_payload = final_state.get("workflow_input")
    if not isinstance(workflow_input_payload, dict):
        raise TypeError("OCR root completed without workflow input")
    artifacts.workflow_input = WorkflowIngestInput.model_validate(workflow_input_payload)
    artifacts.document_tree = grouped_tree
    _write_document_tree(preparation.source_plan.legacy_dir / "document-tree.json", grouped_tree)
    _write_summary(
        artifacts,
        document_id=document_id,
        provider_settings=provider_settings,
        state_store=preparation.state_store,
    )
    raw_bundle = final_state.get("export_bundle")
    bundle = WorkflowExportBundle.model_validate(raw_bundle) if isinstance(raw_bundle, dict) else None
    return result, bundle, artifacts


__all__ = [
    "KogwistarRuntimeCompatibilityError",
    "run_grouping_workflow",
    "run_ocr_root_workflow",
]
