"""Durable summary generation over accepted OCR document trees."""

from __future__ import annotations

import hashlib
import json

from .document_adapters import apply_summary_updates, select_summary_targets
from .document_models import DocumentNode, DocumentTree, SummaryPolicy
from .ocr_generators import (
    _effective_summary_policy,
    _summary_generator_identity,
)
from .ocr_pipeline_contracts import GroundedSummaryGenerator, SummaryGenerator
from .ocr_state_store import GroupingCheckpointConflict, OCRWorkflowStateStore


def _generate_tree_summaries(
    tree: DocumentTree,
    *,
    document_id: str,
    state_store: OCRWorkflowStateStore,
    policy: SummaryPolicy | None,
    generator: SummaryGenerator | None,
    grounded_generator: GroundedSummaryGenerator | None = None,
    grouping_policy_version: str,
    expected_tree_revision: str | None = None,
    require_no_existing_tree: bool = False,
    last_runtime_step: str = "generate_grouping_summaries",
) -> DocumentTree:
    """Run optional bounded summary generation without making OCR depend on it."""
    if (generator is None and grounded_generator is None) or policy is None or not policy.enabled:
        return tree
    if generator is not None and grounded_generator is not None:
        raise ValueError("provide generator or grounded_generator, not both")
    items = {item.item_id: item for item in tree.source_items}
    current = tree
    active_generator = generator if generator is not None else grounded_generator
    assert active_generator is not None
    effective_policy = _effective_summary_policy(policy, active_generator)
    targets = select_summary_targets(current, effective_policy)
    if not targets:
        return current
    for node_id in targets:
        evidence_fingerprint = current.summary_input_fingerprint(node_id)
        policy_payload = effective_policy.model_dump(mode="json")
        generator_name = _summary_generator_identity(active_generator)
        input_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "evidence": evidence_fingerprint,
                    "tree_revision": current.tree_revision,
                    "grouping_policy_version": grouping_policy_version,
                    "summary_policy": policy_payload,
                    "generation_fingerprint": effective_policy.generation_fingerprint,
                    "generator": generator_name,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        # Recompute the node's evidence through the tree contract; the
        # generator receives only source items in that subtree.
        node_map = {node.node_id: node for node in current.nodes}
        covered: set[str] = set()

        def collect(
            current_id: str,
            node_map: dict[str, DocumentNode] = node_map,
            covered: set[str] = covered,
        ) -> None:
            node = node_map[current_id]
            if node.parts:
                covered.update(part.item_id for part in node.parts)
                return
            for child in node.children:
                collect(child.node_id)

        collect(node_id)
        evidence_slices = current.evidence_slices(node_id, tuple(covered))
        evidence_ids = tuple(dict.fromkeys(item.item_id for item in evidence_slices))
        if grounded_generator is None:
            # The compatibility callback receives whole SourceItem objects. It
            # must not see text outside the exact interval owned by this node.
            intervals: dict[str, list[tuple[int, int]]] = {}
            for evidence_slice in evidence_slices:
                if evidence_slice.text is None:
                    continue
                if evidence_slice.start is None or evidence_slice.end is None:
                    raise ValueError(
                        "legacy SummaryGenerator cannot receive text evidence "
                        "without explicit source bounds; migrate to "
                        "GroundedSummaryGenerator"
                    )
                intervals.setdefault(evidence_slice.item_id, []).append(
                    (evidence_slice.start, evidence_slice.end)
                )
            for item_id, item_intervals in intervals.items():
                source_text = items[item_id].text
                if source_text is None:
                    raise ValueError("text evidence resolved to a non-text source item")
                cursor = 0
                for start, end in sorted(item_intervals):
                    if start != cursor:
                        raise ValueError(
                            "legacy SummaryGenerator cannot receive partial or "
                            "discontiguous source ownership; migrate to "
                            "GroundedSummaryGenerator"
                        )
                    cursor = end
                if cursor != len(source_text):
                    raise ValueError(
                        "legacy SummaryGenerator cannot receive partial source "
                        "ownership; migrate to GroundedSummaryGenerator"
                    )
        summary = state_store.load_completed_summary(
            document_id=document_id,
            node_id=node_id,
            input_fingerprint=input_fingerprint,
        )
        if summary is not None:
            current_node = node_map[node_id]
            if current_node.summary == summary:
                # The accepted journal row already represents this exact
                # generation. Do not create another accepted attempt or
                # advance the tree revision on a retry.
                continue
            # A crash may have committed the journal row after the tree write
            # was interrupted. Repair only the missing derived field; the
            # accepted summary remains the single canonical journal result.
            updated = apply_summary_updates(current, {node_id: summary})
            expected_tree_revision = state_store.save_document_tree(
                document_id=document_id,
                tree=updated,
                grouping_policy_version=grouping_policy_version,
                expected_tree_revision=expected_tree_revision,
                require_no_existing_tree=require_no_existing_tree,
                last_runtime_step=last_runtime_step,
            )
            require_no_existing_tree = False
            current = updated
            continue

        state_store.record_summary_attempt(
            document_id=document_id,
            node_id=node_id,
            input_fingerprint=input_fingerprint,
            status="pending",
        )
        try:
            if grounded_generator is not None:
                summary = grounded_generator(
                    tree=current,
                    node_id=node_id,
                    evidence_slices=evidence_slices,
                    input_fingerprint=evidence_fingerprint,
                )
            else:
                assert generator is not None
                summary = generator(
                    tree=current,
                    node_id=node_id,
                    source_items=tuple(items[item_id] for item_id in evidence_ids),
                    input_fingerprint=evidence_fingerprint,
                )
            summary = summary.model_copy(
                update={
                    "generation_fingerprint": effective_policy.generation_fingerprint,
                    "evidence_slice_fingerprint": current.summary_slice_fingerprint(
                        node_id, evidence_ids
                    )
                }
            )
            # The journal is the authoritative accepted result. Commit it
            # before the derived tree field so a crash can replay the journal
            # entry and repair the tree without invoking the provider again.
            state_store.record_summary_attempt(
                document_id=document_id,
                node_id=node_id,
                input_fingerprint=input_fingerprint,
                status=summary.status,
                summary=summary,
            )
            accepted_summary = state_store.load_completed_summary(
                document_id=document_id,
                node_id=node_id,
                input_fingerprint=input_fingerprint,
            )
            if accepted_summary is None:
                raise RuntimeError(
                    "summary journal did not return its accepted result"
                )
            summary = accepted_summary
            updated = apply_summary_updates(current, {node_id: summary})
            expected_tree_revision = state_store.save_document_tree(
                document_id=document_id,
                tree=updated,
                grouping_policy_version=grouping_policy_version,
                expected_tree_revision=expected_tree_revision,
                require_no_existing_tree=require_no_existing_tree,
                last_runtime_step=last_runtime_step,
            )
            require_no_existing_tree = False
            current = updated
        except GroupingCheckpointConflict:
            # A stale writer must fail the workflow so the caller can restore
            # the newer checkpoint; treating this as an optional summary error
            # would hide a lost-write race.
            raise
        except Exception as exc:  # noqa: BLE001
            state_store.record_summary_attempt(
                document_id=document_id,
                node_id=node_id,
                input_fingerprint=input_fingerprint,
                status="failed",
                error_message=str(exc),
            )
    return current
