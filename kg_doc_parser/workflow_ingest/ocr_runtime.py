"""Kogwistar-runtime execution for OCR grouping and derived summaries.

The SQLite store remains an artifact/cache index.  The durable workflow state,
step trace, and resume frontier are owned by ``WorkflowRuntime``.  Provider
callables and the state store are injected dependencies and never serialized
into checkpoint state.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.runtime import MappingStepResolver
from kogwistar.runtime.models import (
    RunFailure,
    RunSuccess,
    StateUpdate,
    StepRunResult,
    WorkflowInvocationRequest,
)
from kogwistar.runtime.runtime import RunResult, StepContext, WorkflowRuntime

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
from .service import workflow_predicates

_OCR_ROOT_STATE_VERSION = "ocr-root-v2"
_OCR_PAGE_STATE_VERSION = "ocr-page-v2"


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


def _tree(state: Mapping[str, object]) -> DocumentTree:
    payload = state.get("tree")
    if not isinstance(payload, dict):
        raise TypeError("OCR grouping runtime state has no tree")
    return DocumentTree.model_validate(payload)


def _proposals(value: object) -> tuple[GroupProposal, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(GroupProposal.model_validate(item) for item in value)


def _pass_count(state: Mapping[str, object]) -> int:
    value = state.get("pass_index", 0)
    return value if type(value) is int and value >= 0 else 0


def _page_cursor(state: Mapping[str, object]) -> int:
    value = state.get("page_cursor", 0)
    return value if type(value) is int and value >= 0 else 0


def _page_sources(state: Mapping[str, object]) -> list[tuple[int, Path]]:
    value = state.get("page_sources", [])
    if not isinstance(value, list):
        raise TypeError("OCR page runtime state has no page source list")
    sources: list[tuple[int, Path]] = []
    for item in value:
        if not isinstance(item, dict) or type(item.get("page_number")) is not int:
            raise ValueError("OCR page runtime source entry is malformed")
        image_path = item.get("image_path")
        if not isinstance(image_path, str) or not image_path:
            raise ValueError("OCR page runtime source path is malformed")
        sources.append((int(item["page_number"]), Path(image_path)))
    return sources


def _json_pages(state: Mapping[str, object]) -> list[OCRPageJSON]:
    value = state.get("raw_pages", [])
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("OCR page runtime raw page state is malformed")
    return [cast(OCRPageJSON, item) for item in value]


def _int_list(state: Mapping[str, object], key: str) -> list[int]:
    value = state.get(key, [])
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ValueError(f"OCR page runtime state field {key!r} is malformed")
    return [int(item) for item in value]


def _validate_root_resume_identity(
    state: Mapping[str, object],
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


def _page_workflow_initial_state(source_plan: OCRSourcePlan) -> dict[str, object]:
    return {
        "ocr_page_state_version": _OCR_PAGE_STATE_VERSION,
        "source_fingerprint": source_plan.input_fingerprint,
        "page_sources": [
            {"page_number": page_number, "image_path": str(image_path)}
            for page_number, image_path in source_plan.page_sources
        ],
        "page_cursor": 0,
        "raw_pages": [],
        "completed_pages": [],
        "reused_pages": [],
    }


def _validate_page_resume_identity(
    state: Mapping[str, object],
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
    state: Mapping[str, object],
    context: OCRPageProcessingContext,
    source_plan: OCRSourcePlan,
) -> dict[str, object] | None:
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
    raw_pages = _json_pages(state)
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
        return {
            "page_cursor": index,
            "raw_pages": raw_pages[:index],
            "completed_pages": completed[:index],
            "reused_pages": [number for number in reused[:index] if number in completed[:index]],
        }
    return None


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
            ctx.state_view,
            document_id=context.document_id,
            source_fingerprint=source_plan.input_fingerprint,
            configuration_fingerprints=context.configuration_fingerprints,
        )
        return _success("process_ocr_page")

    @resolver.register("process_ocr_page")
    def _process(ctx: StepContext) -> StepRunResult:
        repair = _reconcile_page_checkpoint(
            state=ctx.state_view,
            context=context,
            source_plan=source_plan,
        )
        if repair is not None:
            return _success("process_ocr_page", state_update=[("u", repair)])
        sources = _page_sources(ctx.state_view)
        cursor = _page_cursor(ctx.state_view)
        if cursor >= len(sources):
            return _success("finalize_ocr_pages")
        page_number, image_path = sources[cursor]
        raw_pages = _json_pages(ctx.state_view)
        completed_pages = _int_list(ctx.state_view, "completed_pages")
        reused_pages = _int_list(ctx.state_view, "reused_pages")
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
                        "raw_pages": raw_pages,
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
            initial_state=initial_state,
            run_id=run_id,
        )
    )
    if result.status != "succeeded":
        raise RuntimeError(f"OCR page workflow {result.status}: {result.errors}")
    return (
        _json_pages(result.final_state),
        _int_list(result.final_state, "completed_pages"),
        _int_list(result.final_state, "reused_pages"),
        result.run_id,
    )


def _register_grouping_steps(
    resolver: MappingStepResolver,
    *,
    state_store: OCRWorkflowStateStore,
    summary_policy: SummaryPolicy | None,
    summary_generator: SummaryGenerator | None,
) -> None:
    @resolver.register("start")
    def _start(ctx: StepContext) -> StepRunResult:
        return _success("restore_grouping_checkpoint")

    @resolver.register("restore_grouping_checkpoint")
    def _restore(ctx: StepContext) -> StepRunResult:
        tree = _tree(ctx.state_view)
        document_id = str(ctx.state_view.get("document_id", tree.document_id))
        policy_version = str(ctx.state_view.get("grouping_policy_version", "baseline-v1"))
        restored = state_store.load_document_tree(
            document_id=document_id,
            source_fingerprint=tree.source_fingerprint(),
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
                            "checkpoint_revision": state_store.document_tree_revision(
                                document_id=document_id
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
        tree = _tree(ctx.state_view)
        pass_index = _pass_count(ctx.state_view)
        raw_passes = ctx.state_view.get("grouping_passes", [])
        passes = raw_passes if isinstance(raw_passes, list) else []
        if pass_index < len(passes):
            proposals = _proposals(passes[pass_index])
            policy_version = str(ctx.state_view.get("grouping_policy_version", "baseline-v1"))
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
        tree = _tree(ctx.state_view)
        document_id = str(ctx.state_view.get("document_id", tree.document_id))
        policy_version = str(ctx.state_view.get("grouping_policy_version", "baseline-v1"))
        updated = _generate_tree_summaries(
            tree,
            document_id=document_id,
            state_store=state_store,
            policy=summary_policy,
            generator=summary_generator,
            grouping_policy_version=policy_version,
        )
        return _success(
            "persist_grouping_result",
            state_update=[
                (
                    "u",
                    {
                        "tree": updated.model_dump(mode="json"),
                        "summaries_generated": summary_generator is not None
                        and bool(summary_policy and summary_policy.enabled),
                    },
                )
            ],
        )

    @resolver.register("persist_grouping_result")
    def _persist(ctx: StepContext) -> StepRunResult:
        tree = _tree(ctx.state_view)
        revision = state_store.save_document_tree(
            document_id=str(ctx.state_view.get("document_id", tree.document_id)),
            tree=tree,
            grouping_policy_version=str(ctx.state_view.get("grouping_policy_version", "baseline-v1")),
            expected_tree_revision=(
                str(ctx.state_view["checkpoint_revision"])
                if ctx.state_view.get("checkpoint_revision") is not None
                else None
            ),
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
    run_id: str | None = None,
    resume_from_checkpoint: bool = False,
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
    )
    runtime = WorkflowRuntime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=resolver.resolve,
        predicate_registry={},
        trace=True,
    )
    effective_run_id = run_id or f"ocr-grouping|{document_id}"
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
            },
            run_id=effective_run_id,
        )
    )
    if result.status != "succeeded":
        raise RuntimeError(f"OCR grouping workflow {result.status}: {result.errors}")
    return _tree(result.final_state), result.run_id


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
    deps: Mapping[str, object] | None = None,
    run_id: str | None = None,
    resume_from_checkpoint: bool = False,
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
            "workflow_input": "u",
            "tree": "u",
            "grouping_result": "u",
            "semantic_result": "u",
            "semantic_tree": "u",
            "export_bundle": "u",
            "ocr_root_state_version": "u",
            "source_fingerprint": "u",
            "ocr_configuration_fingerprints": "u",
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
        child_run_id = f"{ctx.run_id}|ocr-pages"
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
                    invocation_key="ocr-pages-v1",
                    conversation_id=f"ocr:{document_id}",
                    turn_node_id=f"ocr:{document_id}:pages",
                )
            ],
            _route_next=["ocr_root_prepare_grouping_input"],
        )

    @root_resolver.register("ocr_root_prepare_grouping_input")
    def _root_prepare_grouping(ctx: StepContext) -> StepRunResult:
        page_result = ctx.state_view.get("ocr_page_result")
        if not isinstance(page_result, dict):
            raise TypeError("OCR page child did not return a JSON state")
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
        tree = _tree(ctx.state_view)
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
                        "grouping_passes": ctx.state_view.get("grouping_passes", []),
                        "pass_index": 0,
                    },
                    result_state_key="grouping_result",
                    run_id=f"{ctx.run_id}|grouping",
                    invocation_key="grouping-v1",
                    conversation_id=f"ocr:{document_id}",
                    turn_node_id=f"ocr:{document_id}:grouping",
                )
            ],
            _route_next=["ocr_root_prepare_semantic_input"],
        )

    @root_resolver.register("ocr_root_prepare_semantic_input")
    def _root_prepare_semantic(ctx: StepContext) -> StepRunResult:
        grouping_result = ctx.state_view.get("grouping_result")
        if not isinstance(grouping_result, dict):
            raise TypeError("grouping child did not return a JSON state")
        grouped_tree = _tree(grouping_result)
        workflow_input = WorkflowIngestInput.model_validate(ctx.state_view["workflow_input"])
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
        input_payload = ctx.state_view.get("workflow_input")
        if not isinstance(input_payload, dict):
            raise TypeError("OCR root has no semantic workflow input")
        return RunSuccess(
            conversation_node_id=None,
            state_update=[],
            workflow_invocations=[
                WorkflowInvocationRequest(
                    workflow_id=DEFAULT_WORKFLOW_ID,
                    initial_state={"input": input_payload},
                    result_state_key="semantic_result",
                    run_id=f"{ctx.run_id}|semantic",
                    invocation_key="semantic-v1",
                    conversation_id=f"ingest:{document_id}",
                    turn_node_id=f"ingest:{document_id}:turn",
                )
            ],
            _route_next=["ocr_root_finalize"],
        )

    @root_resolver.register("ocr_root_finalize")
    def _root_finalize(ctx: StepContext) -> StepRunResult:
        semantic_result = ctx.state_view.get("semantic_result")
        if not isinstance(semantic_result, dict):
            raise TypeError("semantic child did not return a JSON state")
        bundle = semantic_result.get("export_bundle")
        if not isinstance(bundle, dict):
            raise TypeError("semantic child did not publish an export bundle")
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
                    ctx.state_view,
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

    runtime = WorkflowRuntime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=dispatch.resolve,
        predicate_registry=workflow_predicates(),
        trace=True,
    )
    effective_run_id = run_id or f"ocr|{document_id}"
    result = (
        runtime.resume_from_latest_checkpoint(
            run_id=effective_run_id,
            workflow_id=DEFAULT_OCR_WORKFLOW_ID,
            conversation_id=f"ocr:{document_id}",
            turn_node_id=f"ocr:{document_id}:root",
        )
        if resume_from_checkpoint
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
                "grouping_proposals": [proposal.model_dump(mode="json") for proposal in grouping_proposals],
                "grouping_passes": [],
            },
            run_id=effective_run_id,
        )
    )
    if result.status != "succeeded":
        raise RuntimeError(f"OCR root workflow {result.status}: {result.errors}")
    final_state = result.final_state
    _validate_root_resume_identity(
        final_state,
        document_id=document_id,
        source_fingerprint=preparation.source_plan.input_fingerprint,
        configuration_fingerprints=preparation.page_context.configuration_fingerprints,
        grouping_policy_version=grouping_policy_version,
        require_match=resume_from_checkpoint,
    )
    page_result = final_state.get("ocr_page_result")
    if not isinstance(page_result, dict):
        raise TypeError("OCR root completed without page result")
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
    artifacts.workflow_input = WorkflowIngestInput.model_validate(final_state["workflow_input"])
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


__all__ = ["run_grouping_workflow", "run_ocr_root_workflow"]
