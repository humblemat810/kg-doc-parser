"""Deterministic structural diagnostics for proposed semantic layers."""

from __future__ import annotations

from .models import (
    CurrentLayerContext,
    CurrentLayerResult,
    LayerChildCandidate,
    LayerStructuralIssue,
)


def _normalize_text(text: str) -> str:
    return "".join(text.split()).lower()


def _child_pointer_fingerprint(
    child: LayerChildCandidate,
) -> tuple[tuple[str, int, int, str], ...]:
    return tuple(
        (
            pointer.source_cluster_id,
            pointer.start_char,
            pointer.end_char,
            _normalize_text(pointer.verbatim_text),
        )
        for pointer in child.total_content_pointers
    )


def _layer_structural_issues(
    *,
    current_layer_context: CurrentLayerContext,
    current_layer_result: CurrentLayerResult,
) -> list[LayerStructuralIssue]:
    """Return typed structural failures without changing the tuple API."""

    issues: list[LayerStructuralIssue] = []
    for parent_id in current_layer_context.parent_node_ids:
        children = [
            child for child in current_layer_result.children if child.parent_node_id == parent_id
        ]
        if len(children) == 1:
            child = children[0]
            parent_fingerprint = tuple(
                (
                    pointer.source_cluster_id,
                    pointer.start_char,
                    pointer.end_char,
                    _normalize_text(pointer.verbatim_text),
                )
                for pointer in current_layer_context.parent_content_pointers_by_id.get(parent_id, [])
            )
            unchanged = _child_pointer_fingerprint(child) == parent_fingerprint
            issues.append(
                LayerStructuralIssue(
                    issue_kind="unchanged_refinement" if unchanged else "unary_refinement",
                    parent_node_id=parent_id,
                    node_id=child.node_id,
                    message=(
                        "refinement returned one child with the parent's exact source span"
                        if unchanged
                        else "refinement returned one child; a meaningful split requires at least two children"
                    ),
                )
            )
        for child in children:
            if len(child.child_candidates) == 1:
                nested = child.child_candidates[0]
                issues.append(
                    LayerStructuralIssue(
                        issue_kind="nested_unary_refinement",
                        parent_node_id=child.node_id,
                        node_id=nested.node_id,
                        message="materialized child tree contains a unary refinement",
                    )
                )
    return issues
