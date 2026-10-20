"""Typed semantic-layer validation, commit, frontier, and tree finalization."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from .models import (
    CurrentLayerContext,
    CurrentLayerResult,
    CurrentLayerReview,
    LayerChildCandidate,
    LayerCoverageGap,
    LayerDuplicateChildNote,
    LayerFrontierItem,
    LayerSpanConflict,
    LayerStructuralIssue,
    LayerValidationDiagnostic,
    ParseSessionState,
)
from .parser_core_diagnostics import (
    _child_pointer_fingerprint,
    _layer_structural_issues,
    _normalize_text,
)
from .parser_core_validation import detect_layer_invariants
from .semantics import (
    SemanticNode,
    pointer_source_validation_error,
    validate_semantic_tree_integrity,
)
from .serialization import JsonObject, JsonValue

ParserSourceMap = dict[str, JsonObject]


def _atomic_pointer_override_errors(
    *,
    parent_id: str,
    current_layer_result: CurrentLayerResult,
    parser_source_map: ParserSourceMap | None,
) -> list[str]:
    """Validate optional atomic pointers before a review can authorize them."""

    raw_pointers = current_layer_result.atomic_content_pointers_by_parent.get(parent_id)
    if raw_pointers is None:
        return []
    if not raw_pointers:
        return [f"parent {parent_id}: atomic pointer override must be a non-empty list"]
    if parser_source_map is None:
        return [f"parent {parent_id}: atomic pointer override requires an authoritative source map"]

    errors: list[str] = []
    for index, pointer in enumerate(raw_pointers):
        error = pointer_source_validation_error(pointer, parser_source_map)
        if error is not None:
            errors.append(f"parent {parent_id}: atomic pointer override {index}: {error}")
    return errors


def _diagnostic_applies_to_parent(
    diagnostic: LayerValidationDiagnostic,
    *,
    parent_id: str,
    children: Sequence[LayerChildCandidate],
) -> bool:
    """Keep parent-local diagnostics from poisoning unrelated valid work."""

    if not diagnostic.location:
        return True
    if parent_id in diagnostic.location:
        return True
    child_ids = {
        child.node_id
        for child in children
        if child.parent_node_id == parent_id
    }
    return any(item in child_ids for item in diagnostic.location)


def validate_layer_commit(
    *,
    current_layer_context: CurrentLayerContext,
    current_layer_result: CurrentLayerResult,
    parser_source_map: ParserSourceMap | None = None,
) -> CurrentLayerReview:
    """Revalidate the final candidate set immediately before persistence.

    Review and pointer repair may change the candidate set. This gate is kept
    separate from the earlier review so no stale reviewer verdict can authorize
    a partially repaired or deduplicated layer.
    """

    parent_ids = list(current_layer_context.parent_node_ids)
    validation_diagnostics = list(current_layer_result.validation_diagnostics)
    if not parent_ids:
        coverage_ok, satisfied, overlap_conflicts, coverage_gaps, duplicate_notes, notes = (
            detect_layer_invariants(
                current_layer_context=current_layer_context,
                current_layer_result=current_layer_result,
                parser_source_map=parser_source_map,
            )
        )
        satisfied = satisfied and not validation_diagnostics
        return CurrentLayerReview(
            updated_result=current_layer_result.model_copy(
                update={"satisfied": satisfied}
            ),
            coverage_ok=coverage_ok,
            satisfied=satisfied,
            strategy_used=current_layer_context.split_strategy,
            overlap_conflicts=overlap_conflicts,
            coverage_gap_notes=coverage_gaps,
            duplicate_child_notes=duplicate_notes,
            structural_issues=_layer_structural_issues(
                current_layer_context=current_layer_context,
                current_layer_result=current_layer_result,
            ),
            validation_diagnostics=validation_diagnostics,
            review_notes=["post-repair pre-commit validation", *notes[:20]],
            metadata={
                "commit_validation": True,
                "validation_diagnostics": [
                    item.model_dump(field_mode="backend", dump_format="json")
                    for item in validation_diagnostics
                ],
            },
            committable_parent_node_ids=parent_ids if satisfied else [],
            failed_parent_node_ids=[] if satisfied else parent_ids,
        )

    titles = dict(zip(parent_ids, current_layer_context.parent_titles))
    valid_parent_ids: list[str] = []
    failed_parent_ids: list[str] = []
    valid_children: list[LayerChildCandidate] = []
    overlap_conflicts: list[LayerSpanConflict] = []
    coverage_gaps: list[LayerCoverageGap] = []
    duplicate_notes: list[LayerDuplicateChildNote] = []
    structural_issues: list[LayerStructuralIssue] = []
    notes: list[str] = []
    child_parents: dict[str, set[str]] = {}
    for child in current_layer_result.children:
        child_parents.setdefault(child.node_id, set()).add(child.parent_node_id)
    for child_id, child_parent_ids in child_parents.items():
        if len(child_parent_ids) > 1:
            diagnostic = LayerValidationDiagnostic(
                phase="commit",
                error_type="duplicate_child_id",
                message=(
                    f"child node id {child_id!r} is proposed under multiple parents: "
                    + ", ".join(sorted(child_parent_ids))
                ),
                location=[child_id, *sorted(child_parent_ids)],
            )
            if diagnostic not in validation_diagnostics:
                validation_diagnostics.append(diagnostic)
    for parent_id in parent_ids:
        local_diagnostics = [
            diagnostic
            for diagnostic in validation_diagnostics
            if _diagnostic_applies_to_parent(
                diagnostic,
                parent_id=parent_id,
                children=current_layer_result.children,
            )
        ]
        local_context = current_layer_context.model_copy(
            update={
                "parent_node_ids": [parent_id],
                "parent_titles": [titles.get(parent_id, parent_id)],
                "parent_content_pointers_by_id": {
                    parent_id: current_layer_context.parent_content_pointers_by_id.get(parent_id, [])
                },
            }
        )
        local_result = current_layer_result.model_copy(
            update={
                "children": [
                    child for child in current_layer_result.children if child.parent_node_id == parent_id
                ],
                "validation_diagnostics": local_diagnostics,
            }
        )
        local_coverage_ok, local_satisfied, local_overlaps, local_gaps, local_duplicates, local_notes = (
            detect_layer_invariants(
                current_layer_context=local_context,
                current_layer_result=local_result,
                parser_source_map=parser_source_map,
            )
        )
        atomic_override_errors = _atomic_pointer_override_errors(
            parent_id=parent_id,
            current_layer_result=local_result,
            parser_source_map=parser_source_map,
        )
        if atomic_override_errors:
            local_coverage_ok = False
            local_satisfied = False
            local_notes.extend(atomic_override_errors)
        overlap_conflicts.extend(local_overlaps)
        coverage_gaps.extend(local_gaps)
        duplicate_notes.extend(local_duplicates)
        structural_issues.extend(
            _layer_structural_issues(
                current_layer_context=local_context,
                current_layer_result=local_result,
            )
        )
        notes.extend(f"parent {parent_id}: {note}" for note in local_notes[:12])
        if not local_diagnostics and local_coverage_ok and local_satisfied:
            valid_parent_ids.append(parent_id)
            valid_children.extend(local_result.children)
        else:
            failed_parent_ids.append(parent_id)

    partial_commit = bool(valid_parent_ids and failed_parent_ids)
    updated_result = current_layer_result.model_copy(
        update={
            "children": valid_children,
            "satisfied": bool(valid_parent_ids),
            # A partially committable result must be safe to pass to the
            # mutation primitive. Failed-parent diagnostics remain on the
            # review object and drive requeue, but must not block verified
            # parents from being committed.
            "validation_diagnostics": (
                [] if valid_parent_ids else validation_diagnostics
            ),
        }
    )
    return CurrentLayerReview(
        updated_result=updated_result,
        coverage_ok=not failed_parent_ids,
        satisfied=bool(valid_parent_ids),
        strategy_used=current_layer_context.split_strategy,
        overlap_conflicts=overlap_conflicts,
        coverage_gap_notes=coverage_gaps,
        duplicate_child_notes=duplicate_notes,
        structural_issues=structural_issues,
        validation_diagnostics=validation_diagnostics,
        review_notes=["post-repair pre-commit validation", *notes[:20]],
        metadata={
            "commit_validation": True,
            "committable_parent_node_ids": cast(JsonValue, valid_parent_ids),
            "failed_parent_node_ids": cast(JsonValue, failed_parent_ids),
            "partial_commit": partial_commit,
            "validation_diagnostics": [
                item.model_dump(field_mode="backend", dump_format="json")
                for item in validation_diagnostics
            ],
        },
        committable_parent_node_ids=valid_parent_ids,
        failed_parent_node_ids=failed_parent_ids,
    )


def commit_layer_children(
    *,
    semantic_tree: SemanticNode,
    current_layer_result: CurrentLayerResult,
    current_depth: int,
    parent_node_ids: list[str] | None = None,
    parser_source_map: ParserSourceMap | None = None,
    max_depth: int | None = None,
) -> SemanticNode:
    # This function is also a public primitive; do not rely on callers having
    # run the workflow's preceding validation step.
    finalize_semantic_tree(
        semantic_tree,
        parser_source_map=parser_source_map,
        max_depth=max_depth,
    )
    tree = SemanticNode.model_validate(semantic_tree.model_dump())
    existing_nodes: dict[str, SemanticNode] = {}
    pending_nodes = [tree]
    while pending_nodes:
        node = pending_nodes.pop()
        node_id = str(node.node_id)
        if node_id in existing_nodes:
            raise ValueError(f"duplicate node id {node_id!r}")
        existing_nodes[node_id] = node
        pending_nodes.extend(node.child_nodes)
    if current_layer_result.validation_diagnostics:
        raise ValueError("layer contains validation diagnostics and cannot be committed")
    grouped_counts: dict[str, int] = {}
    for child in current_layer_result.children:
        grouped_counts[child.parent_node_id] = grouped_counts.get(child.parent_node_id, 0) + 1
        if len(child.child_candidates) == 1:
            raise ValueError(
                f"nested unary refinement under proposed child {child.node_id!r}"
            )
    if current_layer_result.children:
        if any(count == 1 for count in grouped_counts.values()):
            raise ValueError("unary refinement cannot be committed")
    elif not current_layer_result.atomic_retained:
        raise ValueError("empty layer requires explicit atomic retention")
    if (
        parent_node_ids
        and not current_layer_result.children
        and current_layer_result.atomic_retained
        and parser_source_map is None
    ):
        raise ValueError("atomic retention requires an authoritative source map")
    atomic_retained = current_layer_result.atomic_retained
    if parent_node_ids and atomic_retained:
        atomic_override_errors = [
            error
            for parent_id in parent_node_ids
            for error in _atomic_pointer_override_errors(
                parent_id=parent_id,
                current_layer_result=current_layer_result,
                parser_source_map=parser_source_map,
            )
        ]
        if atomic_override_errors:
            raise ValueError("; ".join(atomic_override_errors[:8]))
    if parent_node_ids:
        omitted_parent_ids = [
            parent_id for parent_id in parent_node_ids if grouped_counts.get(parent_id, 0) == 0
        ]
        undeclared_omissions = [
            parent_id
            for parent_id in omitted_parent_ids
            if atomic_retained
            and current_layer_result.atomic_parent_ids
            and parent_id not in current_layer_result.atomic_parent_ids
        ]
        if undeclared_omissions:
            raise ValueError(
                "layer omitted parents without an explicit atomic disposition: "
                + ", ".join(undeclared_omissions)
            )
        if omitted_parent_ids and not atomic_retained:
            raise ValueError(
                "layer omitted parents without atomic retention: "
                + ", ".join(omitted_parent_ids)
            )
        if omitted_parent_ids:
            if parser_source_map is None:
                raise ValueError("atomic retention requires an authoritative source map")
            for parent_id in omitted_parent_ids:
                parent = existing_nodes.get(parent_id)
                if parent is None:
                    raise ValueError(f"atomic parent does not exist: {parent_id}")
                parent_pointers = list(
                    parent.total_content_pointers or parent.aggregate_content_pointers
                )
                if not parent_pointers:
                    raise ValueError(
                        f"atomic parent {parent_id!r} has no verified source pointer"
                    )
                pointer_errors = [
                    pointer_source_validation_error(pointer, parser_source_map)
                    for pointer in parent_pointers
                ]
                invalid_pointer_errors = [error for error in pointer_errors if error is not None]
                if invalid_pointer_errors:
                    raise ValueError(
                        f"atomic parent {parent_id!r} has invalid source pointer: "
                        + "; ".join(invalid_pointer_errors[:4])
                    )
    proposed_parent_ids = set(grouped_counts)
    missing_parent_ids = sorted(proposed_parent_ids - set(existing_nodes))
    if missing_parent_ids:
        raise ValueError(
            "proposed child parent does not exist: " + ", ".join(missing_parent_ids)
        )
    if parent_node_ids is not None:
        allowed_parent_ids = set(parent_node_ids)
        out_of_scope_parent_ids = sorted(proposed_parent_ids - allowed_parent_ids)
        if out_of_scope_parent_ids:
            raise ValueError(
                "proposed child parent is outside current layer: "
                + ", ".join(out_of_scope_parent_ids)
            )
    retained_parent_ids = {
        str(node_id) for node_id in (parent_node_ids or [])
    }
    atomic_pointers_by_parent = current_layer_result.atomic_content_pointers_by_parent
    children_by_parent: dict[str, dict[str, SemanticNode]] = {}
    for child in current_layer_result.children:
        parent_children = children_by_parent.setdefault(child.parent_node_id, {})
        if child.node_id in parent_children:
            raise ValueError(f"duplicate proposed child node id {child.node_id!r}")
        materialized_ids: set[str] = set()
        materialized_children: list[SemanticNode] = []
        for materialized in child.child_candidates:
            if materialized.node_id in materialized_ids:
                raise ValueError(
                    f"duplicate materialized child node id {materialized.node_id!r}"
                )
            materialized_ids.add(materialized.node_id)
            materialized_children.append(
                SemanticNode(
                    node_id=materialized.node_id,
                    parent_id=child.node_id,
                    node_type=materialized.node_type,
                    title=materialized.title,
                    total_content_pointers=list(materialized.total_content_pointers),
                    child_nodes=[],
                    level_from_root=current_depth + 2,
                    metadata=dict(materialized.metadata),
                )
            )
        parent_children[child.node_id] = SemanticNode(
            node_id=child.node_id,
            parent_id=child.parent_node_id,
            node_type=child.node_type,
            title=child.title,
            total_content_pointers=list(child.total_content_pointers),
            child_nodes=materialized_children,
            level_from_root=current_depth + 1,
            metadata=dict(child.metadata),
        )

    rebuilt: dict[int, SemanticNode] = {}
    pending: list[tuple[SemanticNode, bool]] = [(tree, False)]
    while pending:
        node, exiting = pending.pop()
        if not exiting:
            pending.append((node, True))
            pending.extend((child, False) for child in reversed(node.child_nodes))
            continue
        updated_children = [rebuilt[id(child)] for child in node.child_nodes]
        proposed_by_id = children_by_parent.get(str(node.node_id), {})
        if proposed_by_id:
            existing_by_id = {str(child.node_id): child for child in updated_children}
            for child_id, proposed in proposed_by_id.items():
                existing = existing_by_id.get(child_id)
                if existing is not None and existing.child_nodes and not proposed.child_nodes:
                    proposed = proposed.model_copy(update={"child_nodes": existing.child_nodes})
                existing_by_id[child_id] = proposed
            updated_children = list(existing_by_id.values())
        payload = node.model_dump()
        payload["child_nodes"] = [child.model_dump() for child in updated_children]
        node_id = str(node.node_id)
        if atomic_retained and not grouped_counts.get(node_id, 0) and node_id in retained_parent_ids:
            metadata = dict(payload.get("metadata") or {})
            metadata["atomic_retained"] = True
            payload["metadata"] = metadata
            raw_pointers = atomic_pointers_by_parent.get(node_id)
            if raw_pointers is not None:
                payload["total_content_pointers"] = [pointer.model_dump() for pointer in raw_pointers]
        rebuilt[id(node)] = SemanticNode.model_validate(payload)

    committed = rebuilt[id(tree)]
    return finalize_semantic_tree(
        committed,
        parser_source_map=parser_source_map,
        max_depth=max_depth,
    )


def requeue_failed_frontier_items(
    *,
    frontier_queue: list[LayerFrontierItem],
    parent_node_ids: list[str],
    depth: int,
) -> list[LayerFrontierItem]:
    """Requeue only failed batch parents without duplicating frontier work."""
    queued = list(frontier_queue)
    existing = {(item.parent_node_id, item.depth) for item in queued}
    next_order = max((item.order for item in queued), default=-1) + 1
    for parent_id in parent_node_ids:
        key = (parent_id, depth)
        if key in existing:
            continue
        queued.append(
            LayerFrontierItem(
                parent_node_id=parent_id,
                depth=depth,
                order=next_order,
            )
        )
        existing.add(key)
        next_order += 1
    return queued


def enqueue_next_layer_frontier(
    *,
    frontier_queue: list[LayerFrontierItem],
    current_layer_context: CurrentLayerContext,
    current_layer_result: CurrentLayerResult,
    parse_session: ParseSessionState,
) -> list[LayerFrontierItem]:
    queued = list(frontier_queue)
    next_depth = current_layer_context.depth + 1
    if next_depth >= parse_session.max_depth:
        return queued
    next_order = max([item.order for item in queued], default=-1) + 1
    existing_frontier = {
        (item.parent_node_id, item.depth)
        for item in queued
    }
    children_by_parent: dict[str, list[LayerChildCandidate]] = {}
    for child in current_layer_result.children:
        children_by_parent.setdefault(child.parent_node_id, []).append(child)
    for parent_id, children in children_by_parent.items():
        if len(children) == 1:
            raise ValueError(f"unary refinement cannot enter the frontier for {parent_id!r}")
        parent_fingerprint = tuple(
            (
                pointer.source_cluster_id,
                pointer.start_char,
                pointer.end_char,
                _normalize_text(pointer.verbatim_text),
            )
            for pointer in current_layer_context.parent_content_pointers_by_id.get(parent_id, [])
        )
        if any(_child_pointer_fingerprint(child) == parent_fingerprint for child in children):
            raise ValueError(f"unchanged refinement cannot enter the frontier for {parent_id!r}")
    for child in current_layer_result.children:
        frontier_key = (child.node_id, next_depth)
        if child.expandable and frontier_key not in existing_frontier:
            queued.append(
                LayerFrontierItem(
                    parent_node_id=child.node_id,
                    depth=next_depth,
                    order=next_order,
                )
            )
            existing_frontier.add(frontier_key)
            next_order += 1
    return queued


def find_semantic_node(root: SemanticNode, node_id: str) -> SemanticNode | None:
    """Find a node without recursing forever on an unvalidated tree."""
    pending = [root]
    seen: set[int] = set()
    while pending:
        node = pending.pop()
        marker = id(node)
        if marker in seen:
            continue
        seen.add(marker)
        if str(node.node_id) == str(node_id):
            return node
        pending.extend(reversed(node.child_nodes))
    return None


def finalize_semantic_tree(
    semantic_tree: SemanticNode,
    *,
    parser_source_map: ParserSourceMap | None = None,
    max_depth: int | None = None,
) -> SemanticNode:
    """Validate the replayed tree before it reaches export or persistence."""
    validate_semantic_tree_integrity(
        semantic_tree,
        parser_source_map=parser_source_map,
        max_depth=max_depth,
    )
    return semantic_tree


def coerce_layer_child(value: object) -> LayerChildCandidate:
    if isinstance(value, LayerChildCandidate):
        return value
    if isinstance(value, dict):
        return LayerChildCandidate.model_validate(value)
    raise TypeError("unsupported layer child candidate")
