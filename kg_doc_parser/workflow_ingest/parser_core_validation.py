"""Source-grounded structural and coverage validation for semantic layers."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TypeAlias

from .models import (
    CurrentLayerContext,
    CurrentLayerResult,
    LayerChildCandidate,
    LayerCoverageGap,
    LayerDuplicateChildNote,
    LayerSpanConflict,
)
from .parser_core_diagnostics import _child_pointer_fingerprint, _normalize_text
from .parser_core_legacy import _source_text
from .semantics import (
    HydratedTextPointer,
    pointer_source_validation_error,
)
from .serialization import JsonObject

ParserSourceMap: TypeAlias = dict[str, JsonObject]  # noqa: UP040 - preserve Python 3.11/PyPy syntax


def _pointer_end(
    pointer: HydratedTextPointer,
    source_map: ParserSourceMap | None = None,
) -> int:
    if pointer.end_char != -1:
        return pointer.end_char
    if source_map is not None:
        text = _source_text(source_map.get(pointer.source_cluster_id))
        if text:
            return max(0, len(text) - 1)
    return pointer.start_char


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged: list[tuple[int, int]] = []
    ordered_intervals = sorted(intervals)
    cur_s, cur_e = min(ordered_intervals)
    for start, end in ordered_intervals[1:]:
        if start <= cur_e + 1:
            cur_e = max(cur_e, end)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = start, end
    merged.append((cur_s, cur_e))
    return merged


def _has_meaningful_gap_text(text: str) -> bool:
    return bool("".join(text.split()))


def _overlapping_sibling_pointer_pairs(
    children: list[LayerChildCandidate],
    source_map: ParserSourceMap | None,
) -> Iterator[
    tuple[int, int, int, int, HydratedTextPointer, HydratedTextPointer, int, int]
]:
    """Return overlapping pointers in the same order as the former pairwise scan."""

    # For small layers, direct iteration avoids sorting/allocating a sweep index.
    # Dense layers also benefit because nearly every pair is a real diagnostic.
    if len(children) <= 64:
        for left_index, left_child in enumerate(children):
            for right_index in range(left_index + 1, len(children)):
                right_child = children[right_index]
                for left_pointer_index, left_pointer in enumerate(
                    left_child.total_content_pointers
                ):
                    left_end = _pointer_end(left_pointer, source_map)
                    for right_pointer_index, right_pointer in enumerate(
                        right_child.total_content_pointers
                    ):
                        if (
                            left_pointer.source_cluster_id
                            != right_pointer.source_cluster_id
                        ):
                            continue
                        right_end = _pointer_end(right_pointer, source_map)
                        overlap_start = max(
                            left_pointer.start_char, right_pointer.start_char
                        )
                        overlap_end = min(left_end, right_end)
                        if overlap_start <= overlap_end:
                            yield (
                                left_index,
                                right_index,
                                left_pointer_index,
                                right_pointer_index,
                                left_pointer,
                                right_pointer,
                                overlap_start,
                                overlap_end,
                            )
        return

    intervals_by_source: dict[
        str, list[tuple[int, int, int, int, HydratedTextPointer]]
    ] = {}
    for child_index, child_obj in enumerate(children):
        child = child_obj
        for pointer_index, pointer in enumerate(child.total_content_pointers):
            intervals_by_source.setdefault(pointer.source_cluster_id, []).append(
                (
                    pointer.start_char,
                    _pointer_end(pointer, source_map),
                    child_index,
                    pointer_index,
                    pointer,
                )
            )

    candidates: list[
        tuple[int, int, int, int, HydratedTextPointer, HydratedTextPointer, int, int]
    ] = []
    for intervals in intervals_by_source.values():
        ordered = sorted(
            intervals, key=lambda item: (item[0], item[1], item[2], item[3])
        )
        active: list[tuple[int, int, int, int, HydratedTextPointer]] = []
        for current in ordered:
            start, end, child_index, pointer_index, pointer = current
            active = [item for item in active if item[1] >= start]
            if end < start:
                continue
            for other in active:
                _, other_end, other_child_index, other_pointer_index, other_pointer = (
                    other
                )
                if (
                    other_child_index == child_index
                    or other_end < start
                    or end < other[0]
                ):
                    continue
                if other_child_index < child_index:
                    candidates.append(
                        (
                            other_child_index,
                            child_index,
                            other_pointer_index,
                            pointer_index,
                            other_pointer,
                            pointer,
                            max(other[0], start),
                            min(other_end, end),
                        )
                    )
                else:
                    candidates.append(
                        (
                            child_index,
                            other_child_index,
                            pointer_index,
                            other_pointer_index,
                            pointer,
                            other_pointer,
                            max(start, other[0]),
                            min(end, other_end),
                        )
                    )
            active.append(current)

    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    yield from candidates


def detect_layer_invariants(
    *,
    current_layer_context: CurrentLayerContext,
    current_layer_result: CurrentLayerResult,
    parser_source_map: ParserSourceMap | None = None,
) -> tuple[
    bool,
    bool,
    list[LayerSpanConflict],
    list[LayerCoverageGap],
    list[LayerDuplicateChildNote],
    list[str],
]:
    """Check identity, branching, sibling spans, and parent source coverage."""

    overlap_conflicts: list[LayerSpanConflict] = []
    coverage_gaps: list[LayerCoverageGap] = []
    duplicate_notes: list[LayerDuplicateChildNote] = []
    review_notes: list[str] = []
    seen_overlap_pairs: set[tuple[str, str, str, str, int, int, str]] = set()
    seen_child_ids: dict[str, str] = {}
    children_by_parent: dict[str, list[LayerChildCandidate]] = {}
    pointer_ends: dict[int, int] = {}

    def pointer_end(pointer: HydratedTextPointer) -> int:
        key = id(pointer)
        if key not in pointer_ends:
            pointer_ends[key] = _pointer_end(pointer, parser_source_map)
        return pointer_ends[key]

    for child in current_layer_result.children:
        children_by_parent.setdefault(child.parent_node_id, []).append(child)
        previous_parent = seen_child_ids.get(child.node_id)
        if previous_parent is not None:
            duplicate_notes.append(
                LayerDuplicateChildNote(
                    parent_node_id=child.parent_node_id,
                    child_node_id=child.node_id,
                    duplicate_of_child_node_id=child.node_id,
                    reason=(
                        "duplicate child node id; first occurrence was under "
                        f"parent {previous_parent}"
                    ),
                )
            )
            review_notes.append(
                f"duplicate child node id: {child.node_id} under parents "
                f"{previous_parent} and {child.parent_node_id}"
            )
        else:
            seen_child_ids[child.node_id] = child.parent_node_id

    parent_pointers = current_layer_context.parent_content_pointers_by_id or {}
    for parent_id in current_layer_context.parent_node_ids:
        parent_children = children_by_parent.get(parent_id, [])
        if not parent_children:
            explicitly_atomic = current_layer_result.atomic_retained and (
                not current_layer_result.atomic_parent_ids
                or parent_id in current_layer_result.atomic_parent_ids
            )
            if explicitly_atomic:
                atomic_pointers = parent_pointers.get(parent_id, [])
                if not atomic_pointers:
                    coverage_gaps.append(
                        LayerCoverageGap(
                            parent_node_id=parent_id,
                            source_cluster_id="<missing-parent-pointer>",
                            gap_start=0,
                            gap_end=0,
                            expected_text="atomic retention requires a verified parent pointer",
                        )
                    )
                    review_notes.append(
                        f"parent {parent_id} requested atomic retention without a verified parent pointer"
                    )
                elif parser_source_map is not None:
                    for parent_pointer in atomic_pointers:
                        pointer_error = pointer_source_validation_error(
                            parent_pointer, parser_source_map
                        )
                        if pointer_error is not None:
                            coverage_gaps.append(
                                LayerCoverageGap(
                                    parent_node_id=parent_id,
                                    source_cluster_id=parent_pointer.source_cluster_id,
                                    gap_start=max(parent_pointer.start_char, 0),
                                    gap_end=pointer_end(parent_pointer),
                                    expected_text=pointer_error,
                                )
                            )
                            review_notes.append(
                                f"atomic parent pointer for {parent_id} is invalid: {pointer_error}"
                            )
                else:
                    coverage_gaps.append(
                        LayerCoverageGap(
                            parent_node_id=parent_id,
                            source_cluster_id="<missing-source-map>",
                            gap_start=0,
                            gap_end=0,
                            expected_text="atomic retention requires an authoritative source map",
                        )
                    )
                    review_notes.append(
                        f"parent {parent_id} cannot be atomically retained without a source map"
                    )
                if not any(gap.parent_node_id == parent_id for gap in coverage_gaps):
                    review_notes.append(
                        f"parent {parent_id} retained as an explicit atomic unit"
                    )
            else:
                for parent_pointer in parent_pointers.get(parent_id, []):
                    parent_end = pointer_end(parent_pointer)
                    cluster_text = _source_text(
                        (parser_source_map or {}).get(parent_pointer.source_cluster_id)
                    )
                    expected_text = cluster_text[
                        parent_pointer.start_char : parent_end + 1
                    ]
                    if _has_meaningful_gap_text(expected_text):
                        coverage_gaps.append(
                            LayerCoverageGap(
                                parent_node_id=parent_id,
                                source_cluster_id=parent_pointer.source_cluster_id,
                                gap_start=parent_pointer.start_char,
                                gap_end=parent_end,
                                expected_text=expected_text,
                            )
                        )
                review_notes.append(
                    f"parent {parent_id} returned no children without explicit atomic retention"
                )
            continue

        if len(parent_children) == 1:
            only_child = parent_children[0]
            parent_fingerprint = tuple(
                (
                    pointer.source_cluster_id,
                    pointer.start_char,
                    pointer.end_char,
                    _normalize_text(pointer.verbatim_text),
                )
                for pointer in parent_pointers.get(parent_id, [])
            )
            child_fingerprint = _child_pointer_fingerprint(only_child)
            reason = (
                "unchanged refinement: the only child has the parent's exact source span"
                if child_fingerprint == parent_fingerprint
                else "unary refinement: a meaningful split requires at least two children"
            )
            review_notes.append(f"parent {parent_id}: {reason}")

        for child in parent_children:
            if len(child.child_candidates) == 1:
                review_notes.append(
                    f"nested unary refinement under child {child.node_id} of parent {parent_id}"
                )

        seen_signatures: dict[
            tuple[str, tuple[tuple[str, int, int, str], ...]], str
        ] = {}
        for child in parent_children:
            signature = (child.title.strip().lower(), _child_pointer_fingerprint(child))
            if signature in seen_signatures:
                duplicate_of = seen_signatures[signature]
                duplicate_notes.append(
                    LayerDuplicateChildNote(
                        parent_node_id=parent_id,
                        child_node_id=child.node_id,
                        duplicate_of_child_node_id=duplicate_of,
                        reason="duplicate child proposal under the same parent",
                    )
                )
                review_notes.append(
                    f"duplicate child proposal under parent {parent_id}: "
                    f"{child.node_id} duplicates {duplicate_of}"
                )
            else:
                seen_signatures[signature] = child.node_id

        def record_overlap(
            left_child: LayerChildCandidate,
            right_child: LayerChildCandidate,
            left_pointer: HydratedTextPointer,
            right_pointer: HydratedTextPointer,
            overlap_start: int,
            overlap_end: int,
            current_parent_id: str = parent_id,
        ) -> None:
            left_end = pointer_end(left_pointer)
            right_end = pointer_end(right_pointer)
            conflict_kind = (
                "duplicate"
                if left_pointer.start_char == right_pointer.start_char
                and left_end == right_end
                and _normalize_text(left_pointer.verbatim_text)
                == _normalize_text(right_pointer.verbatim_text)
                else "overlap"
            )
            pair_key = (
                current_parent_id,
                left_child.node_id,
                right_child.node_id,
                left_pointer.source_cluster_id,
                overlap_start,
                overlap_end,
                conflict_kind,
            )
            if pair_key in seen_overlap_pairs:
                return
            seen_overlap_pairs.add(pair_key)
            overlap_conflicts.append(
                LayerSpanConflict(
                    parent_node_id=current_parent_id,
                    left_child_id=left_child.node_id,
                    right_child_id=right_child.node_id,
                    source_cluster_id=left_pointer.source_cluster_id,
                    left_span=left_pointer,
                    right_span=right_pointer,
                    overlap_start=overlap_start,
                    overlap_end=overlap_end,
                    conflict_kind=conflict_kind,
                )
            )
            review_notes.append(
                f"{conflict_kind} between {left_child.node_id} and "
                f"{right_child.node_id} on {left_pointer.source_cluster_id}:"
                f"{overlap_start}-{overlap_end}"
            )

        if len(parent_children) <= 64:
            for left_index, left_child in enumerate(parent_children):
                for right_child in parent_children[left_index + 1 :]:
                    for left_pointer in left_child.total_content_pointers:
                        for right_pointer in right_child.total_content_pointers:
                            if (
                                left_pointer.source_cluster_id
                                != right_pointer.source_cluster_id
                            ):
                                continue
                            left_end = _pointer_end(left_pointer, parser_source_map)
                            right_end = _pointer_end(right_pointer, parser_source_map)
                            overlap_start = max(
                                left_pointer.start_char, right_pointer.start_char
                            )
                            overlap_end = min(left_end, right_end)
                            if overlap_start > overlap_end:
                                continue
                            conflict_kind = (
                                "duplicate"
                                if left_pointer.start_char == right_pointer.start_char
                                and left_end == right_end
                                and _normalize_text(left_pointer.verbatim_text)
                                == _normalize_text(right_pointer.verbatim_text)
                                else "overlap"
                            )
                            pair_key = (
                                parent_id,
                                left_child.node_id,
                                right_child.node_id,
                                left_pointer.source_cluster_id,
                                overlap_start,
                                overlap_end,
                                conflict_kind,
                            )
                            if pair_key in seen_overlap_pairs:
                                continue
                            seen_overlap_pairs.add(pair_key)
                            overlap_conflicts.append(
                                LayerSpanConflict(
                                    parent_node_id=parent_id,
                                    left_child_id=left_child.node_id,
                                    right_child_id=right_child.node_id,
                                    source_cluster_id=left_pointer.source_cluster_id,
                                    left_span=left_pointer,
                                    right_span=right_pointer,
                                    overlap_start=overlap_start,
                                    overlap_end=overlap_end,
                                    conflict_kind=conflict_kind,
                                )
                            )
                            review_notes.append(
                                f"{conflict_kind} between {left_child.node_id} and "
                                f"{right_child.node_id} on {left_pointer.source_cluster_id}:"
                                f"{overlap_start}-{overlap_end}"
                            )
        else:
            for (
                left_index,
                right_index,
                _,
                _,
                left_pointer,
                right_pointer,
                overlap_start,
                overlap_end,
            ) in _overlapping_sibling_pointer_pairs(parent_children, parser_source_map):
                record_overlap(
                    parent_children[left_index],
                    parent_children[right_index],
                    left_pointer,
                    right_pointer,
                    overlap_start,
                    overlap_end,
                )

        for parent_pointer in parent_pointers.get(parent_id, []):
            parent_pointer_error = (
                pointer_source_validation_error(parent_pointer, parser_source_map)
                if parser_source_map is not None
                else None
            )
            if parent_pointer_error is not None:
                coverage_gaps.append(
                    LayerCoverageGap(
                        parent_node_id=parent_id,
                        source_cluster_id=parent_pointer.source_cluster_id,
                        gap_start=max(parent_pointer.start_char, 0),
                        gap_end=_pointer_end(parent_pointer, parser_source_map),
                        expected_text=parent_pointer_error,
                    )
                )
                review_notes.append(
                    f"invalid parent pointer for {parent_id}: {parent_pointer_error}"
                )
                continue
            parent_end = _pointer_end(parent_pointer, parser_source_map)
            parent_start = max(parent_pointer.start_char, 0)
            if parent_end < parent_start:
                continue
            child_intervals: list[tuple[int, int]] = []
            for child in parent_children:
                if not child.total_content_pointers:
                    review_notes.append(
                        f"child {child.node_id} under {parent_id} has no content pointer"
                    )
                    continue
                for child_pointer in child.total_content_pointers:
                    pointer_error = (
                        pointer_source_validation_error(
                            child_pointer, parser_source_map
                        )
                        if parser_source_map is not None
                        else None
                    )
                    if pointer_error is not None:
                        coverage_gaps.append(
                            LayerCoverageGap(
                                parent_node_id=parent_id,
                                source_cluster_id=child_pointer.source_cluster_id,
                                gap_start=max(child_pointer.start_char, 0),
                                gap_end=pointer_end(child_pointer),
                                expected_text=pointer_error,
                            )
                        )
                        review_notes.append(
                            f"invalid child pointer {child.node_id} under {parent_id}: "
                            f"{pointer_error}"
                        )
                        continue
                    if (
                        child_pointer.source_cluster_id
                        != parent_pointer.source_cluster_id
                    ):
                        continue
                    child_end = pointer_end(child_pointer)
                    if (
                        child_pointer.start_char < parent_start
                        or child_end > parent_end
                    ):
                        coverage_gaps.append(
                            LayerCoverageGap(
                                parent_node_id=parent_id,
                                source_cluster_id=child_pointer.source_cluster_id,
                                gap_start=max(child_pointer.start_char, 0),
                                gap_end=child_end,
                                expected_text="child pointer is outside parent extent",
                            )
                        )
                        review_notes.append(
                            f"child pointer {child.node_id} is outside parent {parent_id}"
                        )
                        continue
                    child_intervals.append(
                        (max(child_pointer.start_char, 0), child_end)
                    )
            merged = _merge_intervals(child_intervals)
            cluster_text = (
                _source_text(parser_source_map.get(parent_pointer.source_cluster_id))
                if parser_source_map
                else ""
            )
            cursor = parent_start
            for start, end in merged:
                if start > cursor:
                    gap_start = cursor
                    gap_end = min(start - 1, parent_end)
                    if gap_start <= gap_end:
                        gap_text = (
                            cluster_text[gap_start : gap_end + 1]
                            if cluster_text
                            else ""
                        )
                        if not _has_meaningful_gap_text(gap_text):
                            cursor = max(cursor, gap_end + 1)
                        else:
                            coverage_gaps.append(
                                LayerCoverageGap(
                                    parent_node_id=parent_id,
                                    source_cluster_id=parent_pointer.source_cluster_id,
                                    gap_start=gap_start,
                                    gap_end=gap_end,
                                    expected_text=gap_text,
                                )
                            )
                            review_notes.append(
                                f"gap in parent {parent_id} for "
                                f"{parent_pointer.source_cluster_id}: {gap_start}-{gap_end}"
                            )
                cursor = max(cursor, end + 1)
                if cursor > parent_end:
                    break
            if cursor <= parent_end:
                gap_text = cluster_text[cursor : parent_end + 1] if cluster_text else ""
                if _has_meaningful_gap_text(gap_text):
                    coverage_gaps.append(
                        LayerCoverageGap(
                            parent_node_id=parent_id,
                            source_cluster_id=parent_pointer.source_cluster_id,
                            gap_start=cursor,
                            gap_end=parent_end,
                            expected_text=gap_text,
                        )
                    )
                    review_notes.append(
                        f"gap in parent {parent_id} for "
                        f"{parent_pointer.source_cluster_id}: {cursor}-{parent_end}"
                    )

    coverage_ok = not coverage_gaps
    has_unary_nested_child = any(
        len(child.child_candidates) == 1 for child in current_layer_result.children
    )
    branching_ok = all(
        len(children_by_parent.get(parent_id, [])) != 1
        and not has_unary_nested_child
        for parent_id in current_layer_context.parent_node_ids
    )
    satisfied = (
        coverage_ok and not overlap_conflicts and not duplicate_notes and branching_ok
    )
    return (
        coverage_ok,
        satisfied,
        overlap_conflicts,
        coverage_gaps,
        duplicate_notes,
        review_notes,
    )
