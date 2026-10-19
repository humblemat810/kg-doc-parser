"""Standalone persisted OCR grouping workflow entrypoint."""

from __future__ import annotations

from collections.abc import Sequence

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.runtime import MappingStepResolver

from .design import (
    DEFAULT_OCR_GROUPING_WORKFLOW_ID,
    ensure_ocr_grouping_workflow_design,
)
from .document_adapters import suggest_cross_page_continuations
from .document_grouping import (
    GroupProposal,
    GroupProposalGenerator,
    grouping_policy_identity,
)
from .document_models import AdaptiveGroupingPolicy, DocumentTree, SummaryPolicy
from .ocr_pipeline import (
    GroundedSummaryGenerator,
    OCRWorkflowStateStore,
    SummaryGenerator,
)
from .ocr_runtime import (
    _register_grouping_steps,
    _tree,
    _typed_state,
    _workflow_runtime,
)
from .ocr_runtime_grouping_state import validated_grouping_checkpoint


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
        validated_grouping_checkpoint(
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
