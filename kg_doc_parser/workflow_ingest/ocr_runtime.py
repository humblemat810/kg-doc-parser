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
    StepRunResult,
)
from kogwistar.runtime.runtime import (
    RunResult,
    StepContext,
    WorkflowRuntime,
)

from .design import (
    DEFAULT_OCR_WORKFLOW_ID,
    DEFAULT_WORKFLOW_ID,
    ensure_ingest_workflow_design,
    ensure_ocr_grouping_workflow_design,
    ensure_ocr_page_workflow_design,
    ensure_ocr_workflow_design,
)
from .document_grouping import (
    GroupProposal,
    GroupProposalGenerator,
    grouping_policy_identity,
)
from .document_models import AdaptiveGroupingPolicy, DocumentTree, SummaryPolicy
from .handlers import build_ingest_step_resolver
from .models import WorkflowExportBundle
from .ocr_generators import build_production_group_proposal_generator
from .ocr_pipeline import (
    GroundedSummaryGenerator,
    OCRPageJSON,
    OCRPagePreparation,
    OCRPageProcessingContext,
    OCRSourcePlan,
    OCRWorkflowArtifacts,
    OCRWorkflowStateStore,
    SummaryGenerator,
    _summary_generator_identity,
    build_production_grounded_summary_generator,
)
from .ocr_runtime_contracts import (
    OCR_PAGE_STATE_VERSION as _OCR_PAGE_STATE_VERSION,  # noqa: F401
)
from .ocr_runtime_contracts import (
    OCR_ROOT_STATE_VERSION as _OCR_ROOT_STATE_VERSION,
)
from .ocr_runtime_contracts import (
    OCRWorkflowState as _OCRWorkflowState,  # noqa: F401 - compatibility re-export
)
from .ocr_runtime_finalization import finalize_ocr_root_result
from .ocr_runtime_grouping_steps import _register_grouping_steps
from .ocr_runtime_lineage import (
    next_root_recovery_run_id as _next_root_recovery_run_id,
)
from .ocr_runtime_lineage import (
    semantic_parse_generation as _semantic_parse_generation,
)
from .ocr_runtime_page_state import (
    reconcile_page_checkpoint as _reconcile_page_checkpoint,  # noqa: F401
)
from .ocr_runtime_page_state import (
    root_page_artifacts_need_recovery as _root_page_artifacts_need_recovery,
)
from .ocr_runtime_page_state import (
    validate_existing_page_identities as _validate_existing_page_identities,
)
from .ocr_runtime_root_steps import build_ocr_root_step_resolver
from .runtime_contracts import WorkflowRuntimeDeps
from .service import workflow_predicates

if TYPE_CHECKING:
    from .providers import WorkflowProviderSettings


class StepResolver(Protocol):
    def __call__(self, op: str) -> Callable[[StepContext], StepRunResult]: ...


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
            supports_lease = (
                "run_ownership_lease_ms"
                in inspect.signature(WorkflowRuntime).parameters
            )
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
    """Public entrypoint for persisted page-level OCR execution."""
    from .ocr_runtime_page import run_ocr_page_workflow as run_page_workflow

    return run_page_workflow(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        context=context,
        source_plan=source_plan,
        progress_path=progress_path,
        run_id=run_id,
        resume_from_checkpoint=resume_from_checkpoint,
    )


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
    """Run grouping through the dedicated persisted workflow module."""
    from .ocr_runtime_grouping import run_grouping_workflow as run_grouping

    return run_grouping(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        state_store=state_store,
        tree=tree,
        document_id=document_id,
        grouping_policy_version=grouping_policy_version,
        grouping_proposals=grouping_proposals,
        grouping_passes=grouping_passes,
        summary_policy=summary_policy,
        summary_generator=summary_generator,
        grounded_summary_generator=grounded_summary_generator,
        adaptive_grouping_policy=adaptive_grouping_policy,
        grouping_proposer=grouping_proposer,
        run_id=run_id,
        resume_from_checkpoint=resume_from_checkpoint,
        run_ownership_lease_ms=run_ownership_lease_ms,
    )


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
        not source_namespace.strip() or source_namespace != source_namespace.strip()
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
            grouping_proposer is not None or provider_settings.parser.provider != "fake"
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
        raise ValueError(
            "grouping proposer was supplied while adaptive grouping is disabled"
        )
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
            raise RuntimeError(
                "adaptive grouping proposal generator was not constructed"
            )
        proposer_fingerprint = grouping_proposer.configuration_fingerprint
        if not proposer_fingerprint:
            raise ValueError(
                "grouping proposer configuration fingerprint cannot be empty"
            )
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

    from .ocr_runtime_page import build_page_resolver

    page_resolver = build_page_resolver(
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

    root_resolver_build = build_ocr_root_step_resolver(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        preparation=preparation,
        provider_settings=provider_settings,
        document_id=document_id,
        title=title,
        grouping_policy_version=grouping_policy_version,
        source_namespace=source_namespace,
        source_namespace_identity=source_namespace_identity,
        explicit_passes=explicit_passes,
        resume_from_checkpoint=resume_from_checkpoint,
        page_resolver=page_resolver,
        grouping_resolver=grouping_resolver,
        semantic_resolver=semantic_resolver,
    )

    runtime = _workflow_runtime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=root_resolver_build.step_resolver.resolve,
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
                "grouping_proposals": [
                    proposal.model_dump(mode="json") for proposal in grouping_proposals
                ],
                "grouping_passes": [],
            },
            run_id=effective_run_id,
        )
    )
    return finalize_ocr_root_result(
        result=result,
        effective_run_id=effective_run_id,
        logical_run_id=run_id or f"ocr|{document_id}",
        resume_from_checkpoint=resume_from_checkpoint,
        conversation_engine=conversation_engine,
        preparation=preparation,
        provider_settings=provider_settings,
        document_id=document_id,
        title=title,
        source_namespace_identity=source_namespace_identity,
        grouping_policy_version=grouping_policy_version,
        prepared_page_payloads=root_resolver_build.artifacts.prepared_page_payloads,
        prepared_reused_pages=root_resolver_build.artifacts.prepared_reused_pages,
    )


__all__ = [
    "KogwistarRuntimeCompatibilityError",
    "run_grouping_workflow",
    "run_ocr_root_workflow",
]
