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
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Literal, TypedDict, cast

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.runtime import MappingStepResolver
from kogwistar.runtime.contract import Predicate
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
    StepResolver,
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
from .document_grouping import GroupProposal
from .document_models import DocumentTree, SummaryPolicy
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
    _sync_progress_from_state,
    _write_document_tree,
    _write_summary,
)
from .serialization import JsonObject
from .service import workflow_predicates

_OCR_ROOT_STATE_VERSION = "ocr-root-v2"
_OCR_PAGE_STATE_VERSION = "ocr-page-v2"


class KogwistarRuntimeCompatibilityError(RuntimeError):
    """Raised when a requested OCR recovery contract is not available."""


class _WorkflowRuntimeArgs(TypedDict):
    workflow_engine: GraphKnowledgeEngine
    conversation_engine: GraphKnowledgeEngine
    step_resolver: StepResolver
    predicate_registry: dict[str, Predicate]
    trace: bool


class _WorkflowRuntimeLeaseArgs(_WorkflowRuntimeArgs, total=False):
    run_ownership_lease_ms: int


def _workflow_runtime(
    *,
    workflow_engine: GraphKnowledgeEngine,
    conversation_engine: GraphKnowledgeEngine,
    step_resolver: StepResolver,
    predicate_registry: dict[str, Predicate],
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
        return WorkflowRuntime(**lease_args)
    return WorkflowRuntime(**runtime_args)


class _PageSourceState(TypedDict):
    page_number: int
    image_path: str


class _OCRCheckpointPatch(TypedDict, total=False):
    """Validated partial state used to rewind an accepted page prefix."""

    page_cursor: int
    completed_pages: list[int]
    reused_pages: list[int]
    raw_pages: list[OCRPageJSON]


PageListKey = Literal["completed_pages", "reused_pages"]


class _OCRWorkflowState(TypedDict, total=False):
    ocr_page_state_version: str
    ocr_root_state_version: str
    document_id: str
    source_fingerprint: str
    ocr_configuration_fingerprints: list[str]
    grouping_policy_version: str
    tree: JsonObject
    pass_index: int
    ocr_page_recovery_generation: int
    page_cursor: int
    page_sources: list[_PageSourceState]
    legacy_dir: str
    raw_pages: list[OCRPageJSON]
    completed_pages: list[int]
    reused_pages: list[int]
    last_failed_page: int
    checkpoint_revision: str
    grouping_passes: list[list[JsonObject]]
    workflow_input: JsonObject
    grouping_result: JsonObject
    semantic_result: JsonObject
    semantic_tree: JsonObject
    ocr_page_result: JsonObject
    export_bundle: JsonObject
    checkpoint_restored: bool
    checkpoint_absent: bool
    summaries_generated: bool


def _typed_state(value: Mapping[str, object]) -> _OCRWorkflowState:
    """Narrow persisted workflow state once at the runtime boundary.

    Checkpoint payloads are decoded from JSON, so the runtime still validates
    values at each use.  The TypedDict keeps known keys from degrading to
    ``object`` after the boundary has been crossed.
    """
    if not all(isinstance(key, str) for key in value):
        raise TypeError("OCR workflow state keys must be strings")

    string_keys = (
        "ocr_page_state_version",
        "ocr_root_state_version",
        "document_id",
        "source_fingerprint",
        "grouping_policy_version",
        "legacy_dir",
        "checkpoint_revision",
    )
    for key in string_keys:
        field = value.get(key)
        if field is not None and not isinstance(field, str):
            raise TypeError(f"OCR workflow state field {key!r} must be a string")

    integer_keys = (
        "pass_index",
        "ocr_page_recovery_generation",
        "page_cursor",
        "last_failed_page",
    )
    for key in integer_keys:
        field = value.get(key)
        if field is not None and type(field) is not int:
            raise TypeError(f"OCR workflow state field {key!r} must be an integer")

    boolean_keys = ("checkpoint_restored", "checkpoint_absent", "summaries_generated")
    for key in boolean_keys:
        field = value.get(key)
        if field is not None and type(field) is not bool:
            raise TypeError(f"OCR workflow state field {key!r} must be a boolean")

    for key in ("tree", "workflow_input", "grouping_result", "semantic_result", "semantic_tree", "ocr_page_result", "export_bundle"):
        field = value.get(key)
        if field is not None and (
            not isinstance(field, dict) or not all(isinstance(item_key, str) for item_key in field)
        ):
            raise TypeError(f"OCR workflow state field {key!r} must be a JSON object")

    for key in ("completed_pages", "reused_pages"):
        field = value.get(key)
        if field is not None and (
            not isinstance(field, list) or any(type(item) is not int for item in field)
        ):
            raise TypeError(f"OCR workflow state field {key!r} must be a list of integers")

    page_sources = value.get("page_sources")
    if page_sources is not None:
        if not isinstance(page_sources, list):
            raise TypeError("OCR workflow state field 'page_sources' must be a list")
        for item in page_sources:
            if not isinstance(item, dict) or type(item.get("page_number")) is not int or not isinstance(
                item.get("image_path"), str
            ):
                raise TypeError("OCR workflow state page_sources contains a malformed entry")

    return cast(_OCRWorkflowState, value)


def _typed_child_state(value: object, *, label: str) -> _OCRWorkflowState:
    """Validate and narrow a JSON child-state payload at a workflow seam."""
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise TypeError(f"{label} did not return a JSON object state")
    try:
        return _typed_state(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} returned invalid typed state: {exc}") from exc


def _state_string(state: Mapping[str, object], key: str, default: str) -> str:
    value = state.get(key)
    return value if isinstance(value, str) else default


def _state_optional_string(state: Mapping[str, object], key: str) -> str | None:
    value = state.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"OCR workflow state field {key!r} must be a string")
    return value


def _state_bool(state: Mapping[str, object], key: str, default: bool = False) -> bool:
    value = state.get(key)
    return value if type(value) is bool else default


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


def _recovery_generation(state: _OCRWorkflowState) -> int:
    value = state.get("ocr_page_recovery_generation", 0)
    if value is None:
        return 0
    if type(value) is not int or value < 0:
        raise ValueError(
            "OCR page recovery generation must be a non-negative integer"
        )
    return value


def _page_cursor(state: _OCRWorkflowState) -> int:
    value = state.get("page_cursor", 0)
    return value if type(value) is int and value >= 0 else 0


def _page_sources(state: _OCRWorkflowState) -> list[tuple[int, Path]]:
    value = state.get("page_sources", [])
    if not isinstance(value, list):
        raise TypeError("OCR page runtime state has no page source list")
    sources: list[tuple[int, Path]] = []
    for item in value:
        if not isinstance(item, dict):
            raise TypeError("OCR page runtime source entry is malformed")
        source = cast(_PageSourceState, item)
        page_number: int = source["page_number"]
        image_path: str = source["image_path"]
        if type(page_number) is not int or page_number < 1:
            raise ValueError("OCR page runtime source page number is malformed")
        if not isinstance(image_path, str) or not image_path:
            raise ValueError("OCR page runtime source path is malformed")
        sources.append((page_number, Path(image_path)))
    return sources


def _json_pages(state: _OCRWorkflowState) -> list[OCRPageJSON]:
    """Load accepted pages without copying their payload into every checkpoint.

    ``raw_pages`` is retained as a compatibility path for checkpoints created
    before artifact-backed page state was introduced. New checkpoints persist
    only the artifact directory and cursor, then reload the accepted JSON
    artifacts when a downstream step needs the page payloads.
    """
    value = state.get("raw_pages")
    if value is not None:
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise ValueError("OCR page runtime raw page state is malformed")
        return [cast(OCRPageJSON, item) for item in value]

    legacy_dir_value = state.get("legacy_dir")
    if not isinstance(legacy_dir_value, str) or not legacy_dir_value:
        raise ValueError("OCR page runtime state has no legacy artifact directory")
    cursor = _page_cursor(state)
    pages: list[OCRPageJSON] = []
    for page_number, _image_path in _page_sources(state)[:cursor]:
        page_path = Path(legacy_dir_value) / f"page_{page_number}.json"
        try:
            payload = json.loads(page_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"OCR page artifact is not readable: {page_path}") from exc
        if not isinstance(payload, dict):
            raise TypeError(f"OCR page artifact is not a JSON object: {page_path}")
        pages.append(cast(OCRPageJSON, payload))
    return pages


def _int_list(state: _OCRWorkflowState, key: PageListKey) -> list[int]:
    value = state.get(key, [])
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ValueError(f"OCR page runtime state field {key!r} is malformed")
    result: list[int] = [item for item in value if type(item) is int]
    if len(result) != len(value):
        raise ValueError(f"OCR page runtime state field {key!r} is malformed")
    return result


def _validate_root_resume_identity(
    state: _OCRWorkflowState,
    *,
    document_id: str,
    source_fingerprint: str,
    configuration_fingerprints: Sequence[str],
    grouping_policy_version: str,
    require_match: bool,
) -> None:
    """Reject a root checkpoint built from a different immutable input set."""
    if not require_match:
        return
    expected: dict[str, object] = {
        "ocr_root_state_version": _OCR_ROOT_STATE_VERSION,
        "document_id": document_id,
        "source_fingerprint": source_fingerprint,
        "ocr_configuration_fingerprints": list(configuration_fingerprints),
        "grouping_policy_version": grouping_policy_version,
    }
    missing = [key for key in expected if key not in state]
    if missing:
        raise RuntimeError(
            "cannot resume OCR root checkpoint without immutable identity fields: "
            + ", ".join(sorted(missing))
        )
    mismatches = [
        key for key, expected_value in expected.items() if state.get(key) != expected_value
    ]
    if mismatches:
        raise RuntimeError(
            "OCR root checkpoint identity mismatch; refusing to mix persisted state for: "
            + ", ".join(sorted(mismatches))
        )


def _page_workflow_initial_state(source_plan: OCRSourcePlan) -> _OCRWorkflowState:
    return {
        "ocr_page_state_version": _OCR_PAGE_STATE_VERSION,
        "source_fingerprint": source_plan.input_fingerprint,
        "page_sources": [
            {"page_number": page_number, "image_path": str(image_path)}
            for page_number, image_path in source_plan.page_sources
        ],
        "legacy_dir": str(source_plan.legacy_dir),
        "page_cursor": 0,
        "completed_pages": [],
        "reused_pages": [],
    }


def _validate_page_resume_identity(
    state: _OCRWorkflowState,
    *,
    document_id: str,
    source_fingerprint: str,
    configuration_fingerprints: Sequence[str],
) -> None:
    """Reject a page checkpoint built from a different immutable input set."""
    expected: dict[str, object] = {
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
        key for key, expected_value in expected.items() if state.get(key) != expected_value
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
) -> _OCRCheckpointPatch | None:
    """Find the first checkpointed page whose accepted artifact is no longer valid.

    Runtime checkpoints and the SQLite/artifact cache are separate durable
    stores.  A crash can commit one without the other, or an operator can
    tamper with an artifact between resumes.  Never advance past that seam:
    truncate the in-memory checkpoint to the verified prefix and let the
    normal page step replay the first unaccepted page.
    """
    cursor = _page_cursor(state)
    sources = _page_sources(state)
    completed = _int_list(state, "completed_pages")
    reused = _int_list(state, "reused_pages")
    for index, (page_number, image_path) in enumerate(sources[:cursor]):
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
        repair: _OCRCheckpointPatch = {
            "page_cursor": index,
            "completed_pages": completed[:index],
            "reused_pages": [number for number in reused[:index] if number in completed[:index]],
        }
        # Preserve the old in-checkpoint payload only when repairing a legacy
        # checkpoint. New checkpoints stay artifact-backed and compact.
        if "raw_pages" in state:
            repair["raw_pages"] = _json_pages(state)[:index]
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
    sources = _page_sources(page_result)
    cursor = _page_cursor(page_result)
    completed = _int_list(page_result, "completed_pages")
    raw_pages = _json_pages(page_result)
    expected_pages = [page_number for page_number, _image_path in sources]
    if cursor != len(sources) or completed != expected_pages or len(raw_pages) != len(sources):
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


def _page_result_needs_recovery(
    *,
    page_result: _OCRWorkflowState,
    context: OCRPageProcessingContext,
    source_plan: OCRSourcePlan,
) -> bool:
    """Report whether a terminal page child must be regenerated."""
    sources = _page_sources(page_result)
    expected_pages = [page_number for page_number, _image_path in sources]
    if (
        _page_cursor(page_result) != len(sources)
        or _int_list(page_result, "completed_pages") != expected_pages
        or len(_json_pages(page_result)) != len(sources)
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
        state = _typed_state(ctx.state_view)
        repair = _reconcile_page_checkpoint(
            state=state,
            context=context,
            source_plan=source_plan,
        )
        if repair is not None:
            repair_update: StateUpdate = ("u", dict(repair))
            return _success("process_ocr_page", state_update=[repair_update])
        sources = _page_sources(state)
        cursor = _page_cursor(state)
        if cursor >= len(sources):
            return _success("finalize_ocr_pages")
        page_number, image_path = sources[cursor]
        raw_pages = _json_pages(state)
        completed_pages = _int_list(state, "completed_pages")
        reused_pages = _int_list(state, "reused_pages")
        failures: list[int] = []
        _process_ocr_page(
            context=context,
            done_count=cursor + 1,
            page_number=page_number,
            image_path=image_path,
            image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
            page_json_path=source_plan.legacy_dir / f"page_{page_number}.json",
            page_image_path=source_plan.legacy_dir / f"page_{page_number}{image_path.suffix or '.png'}",
            raw_pages=raw_pages,
            completed_pages=completed_pages,
            reused_pages=reused_pages,
            page_failures=failures,
        )
        if failures:
            return RunFailure(
                conversation_node_id=None,
                state_update=[("u", {"last_failed_page": page_number})],
                errors=[f"OCR page {page_number} failed after all candidates"],
            )
        _sync_progress_from_state(
            state_store=context.state_store,
            document_id=context.document_id,
            title=context.title,
            source_kind=context.source_kind,
            total_pages=context.total_pages,
            progress_path=progress_path,
        )
        return _success(
            "process_ocr_page",
            state_update=[
                (
                    "u",
                    {
                        "page_cursor": cursor + 1,
                        "completed_pages": completed_pages,
                        "reused_pages": reused_pages,
                    },
                )
            ],
        )

    @resolver.register("finalize_ocr_pages")
    def _finalize(ctx: StepContext) -> StepRunResult:
        context.state_store.refresh_document_completion(document_id=context.document_id)
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
            initial_state=cast(dict[str, object], initial_state),
            run_id=run_id,
        )
    )
    if result.status != "succeeded":
        raise RuntimeError(f"OCR page workflow {result.status}: {result.errors}")
    return (
        _json_pages(_typed_state(result.final_state)),
        _int_list(_typed_state(result.final_state), "completed_pages"),
        _int_list(_typed_state(result.final_state), "reused_pages"),
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
    restored = state_store.load_document_tree(
        document_id=document_id,
        source_fingerprint=tree.source_fingerprint(),
        grouping_policy_version=grouping_policy_version,
    )
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
        return _success("generate_grouping_summaries")

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
            grouping_policy_version=grouping_policy_version,
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
                "grouping_policy_version": grouping_policy_version,
                "grouping_passes": [
                    [proposal.model_dump(mode="json") for proposal in current_pass]
                    for current_pass in explicit_passes
                ],
                "pass_index": 0,
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
    provider_settings: object,
    document_id: str,
    title: str,
    grouping_policy_version: str,
    grouping_proposals: Sequence[GroupProposal] = (),
    grouping_passes: Sequence[Sequence[GroupProposal]] = (),
    summary_policy: SummaryPolicy | None = None,
    summary_generator: SummaryGenerator | None = None,
    grounded_summary_generator: GroundedSummaryGenerator | None = None,
    deps: Mapping[str, object] | None = None,
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

    from .clients import DirectRuntimeIngestClient
    from .providers import WorkflowProviderSettings

    if not isinstance(provider_settings, WorkflowProviderSettings):
        raise TypeError("provider_settings must be WorkflowProviderSettings")
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
        summary_policy=summary_policy,
        summary_generator=summary_generator,
        grounded_summary_generator=grounded_summary_generator,
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
            "ocr_configuration_fingerprints": "u",
            "checkpoint_revision": "u",
            "checkpoint_absent": "u",
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
        raw_pages = _json_pages(page_result)
        completed_pages = _int_list(page_result, "completed_pages")
        reused_pages = _int_list(page_result, "reused_pages")
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
        attach_document_tree_metadata(workflow_input, grouped_tree)
        _write_document_tree(preparation.source_plan.legacy_dir / "document-tree.json", grouped_tree)
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
                    invocation_key=(
                        "semantic-v1"
                        if not recovery_generation
                        else f"semantic-recovery-v{recovery_generation}"
                    ),
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
                    configuration_fingerprints=preparation.page_context.configuration_fingerprints,
                    grouping_policy_version=grouping_policy_version,
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
        predicate_registry=workflow_predicates(),
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
                "ocr_configuration_fingerprints": list(
                    preparation.page_context.configuration_fingerprints
                ),
                "grouping_policy_version": grouping_policy_version,
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
        configuration_fingerprints=preparation.page_context.configuration_fingerprints,
        grouping_policy_version=grouping_policy_version,
        require_match=root_resume,
    )
    page_result = _typed_child_state(
        final_state.get("ocr_page_result"),
        label="OCR root page child",
    )
    _validate_root_page_result(
        page_result=page_result,
        context=preparation.page_context,
        source_plan=preparation.source_plan,
    )
    raw_pages = _json_pages(page_result)
    completed_pages = _int_list(page_result, "completed_pages")
    reused_pages = _int_list(page_result, "reused_pages")
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
