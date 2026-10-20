"""Kogwistar step resolver for OCR grouping, summaries, and adaptive passes."""

from __future__ import annotations

from kogwistar.runtime import MappingStepResolver
from kogwistar.runtime.models import StepRunResult
from kogwistar.runtime.runtime import StepContext

from .document_adapters import apply_grouping_proposals
from .document_grouping import (
    GroupProposal,
    GroupProposalBatch,
    GroupProposalGenerator,
    build_grouping_pass_context,
    grouping_frontier_windows,
    resolve_grouping_proposals,
    validate_grouping_window_proposals,
)
from .document_models import AdaptiveGroupingPolicy, DocumentTree, SummaryPolicy
from .ocr_pipeline import (
    GroundedSummaryGenerator,
    OCRWorkflowStateStore,
    SummaryGenerator,
    _generate_tree_summaries,
)
from .ocr_runtime_common import document_tree_from_state, workflow_success
from .ocr_runtime_contracts import OCRWorkflowState
from .ocr_runtime_grouping_state import validated_grouping_checkpoint
from .ocr_runtime_state import (
    state_bool,
    state_optional_string,
    state_string,
    typed_state,
)


def _proposals(value: object) -> tuple[GroupProposal, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(GroupProposal.model_validate(item) for item in value)


def _pass_count(state: OCRWorkflowState) -> int:
    value = state.get("pass_index", 0)
    return value if type(value) is int and value >= 0 else 0


def _adaptive_round(state: OCRWorkflowState) -> int:
    value = state.get("adaptive_grouping_round", 0)
    if type(value) is not int or value < 0:
        raise ValueError("adaptive grouping round must be a non-negative integer")
    return value


def _adaptive_window(state: OCRWorkflowState) -> int:
    value = state.get("adaptive_grouping_window", 0)
    if type(value) is not int or value < 0:
        raise ValueError(
            "adaptive grouping window cursor must be a non-negative integer"
        )
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
        return workflow_success("restore_grouping_checkpoint")

    @resolver.register("restore_grouping_checkpoint")
    def _restore(ctx: StepContext) -> StepRunResult:
        state = typed_state(ctx.state_view)
        tree = document_tree_from_state(state)
        document_id = state_string(state, "document_id", tree.document_id)
        policy_version = state_string(state, "grouping_policy_version", "baseline-v1")
        restored, revision = validated_grouping_checkpoint(
            state_store=state_store,
            document_id=document_id,
            tree=tree,
            grouping_policy_version=policy_version,
        )
        if restored is not None:
            payload = restored.model_dump(mode="json")
            return workflow_success(
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
            raise RuntimeError(
                "grouping checkpoint exists but does not match the current source or policy"
            )
        return workflow_success("apply_grouping_pass")

    @resolver.register("apply_grouping_pass")
    def _apply(ctx: StepContext) -> StepRunResult:
        state = typed_state(ctx.state_view)
        tree = document_tree_from_state(state)
        pass_index = _pass_count(state)
        raw_passes = state.get("grouping_passes", [])
        passes = raw_passes if isinstance(raw_passes, list) else []
        if pass_index < len(passes):
            proposals = _proposals(passes[pass_index])
            policy_version = state_string(
                state, "grouping_policy_version", "baseline-v1"
            )
            tree = apply_grouping_proposals(
                tree,
                proposals,
                grouping_policy_version=f"{policy_version}:{pass_index + 1}",
            )
            return workflow_success(
                "apply_grouping_pass",
                state_update=[
                    (
                        "u",
                        {
                            "tree": tree.model_dump(mode="json"),
                            "pass_index": pass_index + 1,
                        },
                    )
                ],
            )
        if not adaptive_grouping_policy.enabled or state_bool(
            state, "adaptive_grouping_complete"
        ):
            return workflow_success("generate_grouping_summaries")
        if grouping_proposer is None:
            raise RuntimeError(
                "adaptive grouping is enabled without a proposal generator"
            )

        policy_version = state_string(state, "grouping_policy_version", "baseline-v1")
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
            return workflow_success(
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
            raise ValueError(
                "adaptive grouping window cursor exceeds the current frontier"
            )
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
            return workflow_success(
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
            return workflow_success(
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
        next_root = next(
            node for node in next_tree.nodes if node.node_id == next_tree.root_id
        )
        if len(next_root.children) >= len(frontier):
            raise RuntimeError("adaptive grouping proposal did not reduce the frontier")
        history.append(
            {
                "round": round_index + 1,
                "input_frontier": list(frontier),
                "proposals": [
                    proposal.model_dump(mode="json") for proposal in proposals
                ],
                "output_frontier": [child.node_id for child in next_root.children],
                "outcome": "groups_accepted",
            }
        )
        return workflow_success(
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
        state = typed_state(ctx.state_view)
        tree = document_tree_from_state(state)
        document_id = state_string(state, "document_id", tree.document_id)
        policy_version = state_string(state, "grouping_policy_version", "baseline-v1")
        checkpoint_revision = state_optional_string(state, "checkpoint_revision")
        updated = _generate_tree_summaries(
            tree,
            document_id=document_id,
            state_store=state_store,
            policy=summary_policy,
            generator=summary_generator,
            grounded_generator=grounded_summary_generator,
            grouping_policy_version=policy_version,
            expected_tree_revision=(
                checkpoint_revision if checkpoint_revision is not None else None
            ),
            require_no_existing_tree=state_bool(state, "checkpoint_absent"),
        )
        current_revision = state_store.document_tree_revision(document_id=document_id)
        return workflow_success(
            "persist_grouping_result",
            state_update=[
                (
                    "u",
                    {
                        "tree": updated.model_dump(mode="json"),
                        "summaries_generated": (
                            summary_generator is not None
                            or grounded_summary_generator is not None
                        )
                        and bool(summary_policy and summary_policy.enabled),
                        "checkpoint_revision": current_revision,
                        "checkpoint_absent": current_revision is None,
                    },
                )
            ],
        )

    @resolver.register("persist_grouping_result")
    def _persist(ctx: StepContext) -> StepRunResult:
        state = typed_state(ctx.state_view)
        tree = document_tree_from_state(state)
        checkpoint_revision = state_optional_string(state, "checkpoint_revision")
        revision = state_store.save_document_tree(
            document_id=state_string(state, "document_id", tree.document_id),
            tree=tree,
            grouping_policy_version=state_string(
                state, "grouping_policy_version", "baseline-v1"
            ),
            expected_tree_revision=(
                checkpoint_revision if checkpoint_revision is not None else None
            ),
            require_no_existing_tree=state_bool(state, "checkpoint_absent"),
            last_runtime_step="persist_grouping_result",
        )
        return workflow_success(
            "end", state_update=[("u", {"checkpoint_revision": revision})]
        )

    @resolver.register("end")
    def _end(ctx: StepContext) -> StepRunResult:
        del ctx
        return workflow_success(None)
