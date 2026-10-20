"""Boundary proposal validation and source-grounded layer assembly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import cast

from kogwistar.utils import SourcePointerValidationError, validate_source_pointer

from .layerwise_boundaries import (
    MAX_BOUNDARY_REPAIR_SHIFT_CHARS,
    AnchorMatchMode,
    BoundaryKind,
    _boundary_cutpoint_legality_reason,
    _BoundaryAnchorResolution,
    _classify_boundary_kind,
    _legal_cutpoints_for_text,
    _resolve_boundary_anchor,
)
from .layerwise_boundary_contracts import (
    _AcceptedBoundaryCutpoint,
    _BoundaryAcceptedUnit,
    _BoundaryCoverageReport,
    _BoundaryGapRange,
    _BoundaryRange,
    _BoundaryRefinementCutpoint,
    _BoundaryRefinementPromptContext,
    _BoundaryReviewDecisionSummary,
    _BoundarySegment,
)
from .layerwise_source_spans import (
    _offset_value,
    _pointer_field,
    _pointer_source_cluster_id,
    _pointer_span_bounds,
    _pointer_text,
    _record_text,
    _trim_text,
)
from .models import (
    BoundaryCutpoint,
    BoundaryReviewBatch,
    BoundaryReviewDecision,
    BoundaryUnitSummary,
    CurrentLayerContext,
    CurrentLayerResult,
    FailureCategory,
    LayerChildCandidate,
    LayerReasoningEntry,
    LLMBoundaryProposalBatch,
    LLMCurrentLayerResult,
    LLMLayerChildCandidate,
)
from .semantics import HydratedTextPointer
from .serialization import JsonValue
from .layerwise_boundary_prompts import (
    _make_boundary_summary,
    _trim_multiline_text,
)
def _proposal_validation_reason(
    *,
    parsed: LLMCurrentLayerResult,
    current_layer_context: CurrentLayerContext,
    parser_source_map: Mapping[str, Mapping[str, object]],
) -> str | None:
    children = list(parsed.children)
    if not children:
        if parsed.satisfied is True:
            return None
        return "proposal returned no children without satisfied=true"
    current_parent_ids = set(current_layer_context.parent_node_ids)
    child_parent_ids = {str(child.parent_node_id or "") for child in children}
    if current_parent_ids and not child_parent_ids.issubset(current_parent_ids):
        return "proposal referenced parent ids outside the current layer"
    children_by_parent: dict[str, list[LLMLayerChildCandidate]] = {}
    parent_pointers = dict(current_layer_context.parent_content_pointers_by_id)
    source_text_by_cluster = {
        str(source_cluster_id): _record_text(record)
        for source_cluster_id, record in parser_source_map.items()
    }
    for child in children:
        pointers = list(child.total_content_pointers)
        if not pointers:
            return "proposal child missing total_content_pointers"
        parent_id = str(child.parent_node_id or "")
        if not parent_id:
            return "proposal child missing parent_node_id"
        children_by_parent.setdefault(parent_id, []).append(child)
        for pointer in pointers:
            verbatim_text = _pointer_field(pointer, "verbatim_text")
            try:
                validate_source_pointer(
                    pointer,
                    source_text_by_cluster=source_text_by_cluster,
                    parent_pointers=list(parent_pointers.get(parent_id) or []),
                    end_mode="inclusive",
                    require_source_cluster=True,
                    require_source_text=True,
                    require_parent_containment=True,
                    require_text_match=verbatim_text is not None,
                )
            except SourcePointerValidationError as exc:
                return _proposal_pointer_validation_reason(exc)
    parent_titles = dict(
        zip(
            current_layer_context.parent_node_ids,
            current_layer_context.parent_titles,
            strict=False,
        )
    )
    for parent_id, parent_children in children_by_parent.items():
        if len(parent_children) == 1:
            _ = parent_titles.get(parent_id)
            return "proposal collapsed a parent into a single child without a real breakdown"
    return None


def _proposal_pointer_validation_reason(exc: SourcePointerValidationError) -> str:
    if exc.code in {"missing_source_cluster", "source_not_found"}:
        return "proposal pointer referenced source outside the supplied source map"
    if exc.code == "span_not_int":
        return "proposal pointer missing character span integers"
    if exc.code == "invalid_span":
        return "proposal pointer used an invalid character span"
    if exc.code == "out_of_bounds":
        return "proposal pointer used a character span outside the source text"
    if exc.code == "outside_parent_span":
        return "proposal pointer used a character span outside the current parent span"
    if exc.code in {"missing_text", "text_mismatch"}:
        return "proposal pointer verbatim_text did not match the source span"
    return "proposal pointer failed source span validation"


def _boundary_validation_reason(
    *,
    parsed: LLMBoundaryProposalBatch,
    current_layer_context: CurrentLayerContext,
    parser_source_map: Mapping[str, Mapping[str, object]],
) -> str | None:
    cutpoints = list(parsed.cutpoints)
    if not cutpoints:
        if parsed.satisfied is True:
            return None
        return "boundary proposal returned no cutpoints without satisfied=true"
    current_parent_ids = set(current_layer_context.parent_node_ids)
    parent_pointers = dict(current_layer_context.parent_content_pointers_by_id)
    seen: set[tuple[str, str, int]] = set()
    previous_key: tuple[str, str, int] | None = None
    for cutpoint in cutpoints:
        parent_node_id = str(cutpoint.parent_node_id or "")
        source_cluster_id = str(cutpoint.source_cluster_id or "")
        cut_offset = cutpoint.cut_offset
        text_before_cut = str(cutpoint.text_before_cut or "")
        text_after_cut = str(cutpoint.text_after_cut or "")
        cut_reason = str(cutpoint.cut_reason or cutpoint.reason or "")
        current_key = (
            parent_node_id,
            source_cluster_id,
            cut_offset if type(cut_offset) is int else -1,
        )
        if previous_key is not None and current_key < previous_key:
            return "boundary proposal cutpoints must be sorted by parent, source cluster, then cut offset"
        previous_key = current_key
        if not parent_node_id or parent_node_id not in current_parent_ids:
            return "boundary proposal referenced parent ids outside the current layer"
        if not source_cluster_id or source_cluster_id not in parser_source_map:
            return "boundary proposal referenced source outside the supplied source map"
        if type(cut_offset) is not int:
            return "boundary proposal missing integer cut_offset"
        if not text_before_cut.strip() or not text_after_cut.strip():
            return (
                "boundary proposal missing text_before_cut or text_after_cut evidence"
            )
        if not cut_reason.strip():
            return "boundary proposal missing cut_reason"
        pointer_candidates = list(parent_pointers.get(parent_node_id) or [])
        pointer = None
        for parent_pointer in pointer_candidates:
            if _pointer_source_cluster_id(parent_pointer) == source_cluster_id:
                pointer = parent_pointer
                break
        if pointer is None:
            return "boundary proposal referenced a source cluster not present in the current parent context"
        start_char, end_char_exclusive = _pointer_span_bounds(
            pointer, parser_source_map=parser_source_map
        )
        if cut_offset <= start_char or cut_offset >= end_char_exclusive:
            return "boundary proposal used a cut_offset outside the parent span"
        key = (parent_node_id, source_cluster_id, cut_offset)
        if key in seen:
            return "boundary proposal contained duplicate cutpoints"
        seen.add(key)
    return None


def _boundary_refinement_targets(
    *,
    parsed: LLMBoundaryProposalBatch,
    review_decisions: list[BoundaryReviewDecision],
) -> list[BoundaryCutpoint]:
    targets: list[BoundaryCutpoint] = []
    for cutpoint, decision in zip(parsed.cutpoints, review_decisions, strict=False):
        if decision.decision != "needs_refinement":
            continue
        targets.append(cutpoint)
    return targets


def _boundary_refinement_prompt_context(
    *,
    current_layer_context: CurrentLayerContext,
    parser_source_map: Mapping[str, Mapping[str, object]],
    target: BoundaryCutpoint,
) -> _BoundaryRefinementPromptContext:
    parent_ids = list(current_layer_context.parent_node_ids)
    parent_pointers = dict(current_layer_context.parent_content_pointers_by_id)
    target_parent_pointers = list(parent_pointers.get(target.parent_node_id) or [])
    target_pointer = None
    for pointer in target_parent_pointers:
        if _pointer_source_cluster_id(pointer) == target.source_cluster_id:
            target_pointer = pointer
            break
    target_excerpt = (
        _pointer_text(target_pointer, parser_source_map=parser_source_map)
        if target_pointer is not None
        else ""
    )
    legal_cutpoints: list[_BoundaryRefinementCutpoint] = []
    if target_pointer is not None:
        start_char = _offset_value(_pointer_field(target_pointer, "start_char"), 0)
        for boundary in _legal_cutpoints_for_text(target_excerpt, max_points=128):
            legal_cutpoints.append(
                {
                    "cut_offset": start_char + boundary.cut_offset,
                    "boundary_kind": boundary.boundary_kind,
                    "confidence": float(boundary.confidence or 0.0),
                    "reason": str(boundary.reason or ""),
                }
            )
    return {
        "parent_node_id": target.parent_node_id,
        "source_cluster_id": target.source_cluster_id,
        "input_cut_offset": target.cut_offset,
        "text_before_cut": target.text_before_cut,
        "text_after_cut": target.text_after_cut,
        "cut_reason": str(target.cut_reason or target.reason or ""),
        "parent_ids": parent_ids,
        "pointer_excerpt": _trim_multiline_text(
            target_excerpt, max_lines=4, max_chars=700
        ),
        "legal_cutpoints": legal_cutpoints,
    }


def _boundary_decision_key(decision: BoundaryReviewDecision) -> tuple[str, str, int]:
    return (
        str(decision.parent_node_id),
        str(decision.source_cluster_id),
        int(
            decision.input_cut_offset
            if decision.input_cut_offset is not None
            else decision.cut_offset
        ),
    )


def _boundary_review_decision_summary(
    decisions: list[BoundaryReviewDecision],
) -> list[_BoundaryReviewDecisionSummary]:
    summaries: list[_BoundaryReviewDecisionSummary] = []
    for decision in decisions:
        summaries.append(
            {
                "candidate_id": decision.candidate_id,
                "decision": decision.decision,
                "parent_node_id": decision.parent_node_id,
                "source_cluster_id": decision.source_cluster_id,
                "input_cut_offset": decision.input_cut_offset,
                "resolved_cut_offset": decision.resolved_cut_offset,
                "boundary_kind": decision.boundary_kind,
                "anchor_match_mode": decision.anchor_match_mode,
                "anchor_match_score": decision.anchor_match_score,
                "reason": _trim_text(decision.reason or "", max_chars=180),
            }
        )
    return summaries


def _boundary_review_decision(
    *,
    cutpoint: BoundaryCutpoint,
    current_layer_context: CurrentLayerContext,
    parser_source_map: Mapping[str, Mapping[str, object]],
    max_repair_shift_chars: int = MAX_BOUNDARY_REPAIR_SHIFT_CHARS,
) -> BoundaryReviewDecision:
    parent_pointers = dict(current_layer_context.parent_content_pointers_by_id)
    pointer = None
    for parent_pointer in list(parent_pointers.get(cutpoint.parent_node_id) or []):
        if _pointer_source_cluster_id(parent_pointer) == cutpoint.source_cluster_id:
            pointer = parent_pointer
            break
    if pointer is None:
        return BoundaryReviewDecision(
            candidate_id=cutpoint.candidate_id,
            parent_node_id=cutpoint.parent_node_id,
            source_cluster_id=cutpoint.source_cluster_id,
            input_cut_offset=cutpoint.cut_offset,
            cut_offset=cutpoint.cut_offset,
            decision="reject",
            reason="source cluster not present in current parent context",
            text_before_cut=cutpoint.text_before_cut,
            text_after_cut=cutpoint.text_after_cut,
        )

    text = _pointer_text(pointer, parser_source_map=parser_source_map)
    start_char = _offset_value(_pointer_field(pointer, "start_char"), 0)
    _start_char, _end_char_exclusive = _pointer_span_bounds(
        pointer, parser_source_map=parser_source_map
    )
    local_cut = cutpoint.cut_offset - start_char
    if cutpoint.candidate_id:
        resolution = _BoundaryAnchorResolution(
            resolved_cut_offset=local_cut,
            match_mode="exact",
            match_score=1.0,
            reason="selected deterministic boundary candidate",
        )
    else:
        resolution = _resolve_boundary_anchor(
            text=text,
            cut_offset=local_cut,
            text_before_cut=cutpoint.text_before_cut,
            text_after_cut=cutpoint.text_after_cut,
        )
    if resolution.resolved_cut_offset is None:
        unresolved_reason = resolution.reason or "boundary anchor could not be resolved"
        unresolved_decision = (
            "needs_refinement"
            if "ambiguous" in unresolved_reason
            or "resolve" in unresolved_reason
            or "matched" in unresolved_reason
            else "reject"
        )
        return BoundaryReviewDecision(
            candidate_id=cutpoint.candidate_id,
            parent_node_id=cutpoint.parent_node_id,
            source_cluster_id=cutpoint.source_cluster_id,
            input_cut_offset=cutpoint.cut_offset,
            cut_offset=cutpoint.cut_offset,
            decision=unresolved_decision,
            reason=unresolved_reason,
            anchor_match_mode=cast(AnchorMatchMode, resolution.match_mode),
            anchor_match_score=resolution.match_score,
            text_before_cut=cutpoint.text_before_cut,
            text_after_cut=cutpoint.text_after_cut,
        )

    resolved_local_cut = resolution.resolved_cut_offset
    resolved_cut_offset = start_char + resolved_local_cut
    if resolved_local_cut <= 0 or resolved_local_cut >= len(text):
        return BoundaryReviewDecision(
            candidate_id=cutpoint.candidate_id,
            parent_node_id=cutpoint.parent_node_id,
            source_cluster_id=cutpoint.source_cluster_id,
            input_cut_offset=cutpoint.cut_offset,
            cut_offset=cutpoint.cut_offset,
            decision="reject",
            resolved_cut_offset=resolved_cut_offset,
            anchor_match_mode=cast(AnchorMatchMode, resolution.match_mode),
            anchor_match_score=resolution.match_score,
            text_before_cut=cutpoint.text_before_cut,
            text_after_cut=cutpoint.text_after_cut,
            reason="resolved boundary fell outside the parent span",
        )

    shift_distance = abs(resolved_cut_offset - cutpoint.cut_offset)
    if shift_distance > max_repair_shift_chars:
        return BoundaryReviewDecision(
            candidate_id=cutpoint.candidate_id,
            parent_node_id=cutpoint.parent_node_id,
            source_cluster_id=cutpoint.source_cluster_id,
            input_cut_offset=cutpoint.cut_offset,
            cut_offset=cutpoint.cut_offset,
            decision="reject",
            resolved_cut_offset=resolved_cut_offset,
            anchor_match_mode=cast(AnchorMatchMode, resolution.match_mode),
            anchor_match_score=resolution.match_score,
            text_before_cut=cutpoint.text_before_cut,
            text_after_cut=cutpoint.text_after_cut,
            reason=(
                f"boundary repair shift {shift_distance} chars exceeds "
                f"maximum {max_repair_shift_chars}"
            ),
        )

    legality_reason = _boundary_cutpoint_legality_reason(text, resolved_local_cut)
    if legality_reason:
        return BoundaryReviewDecision(
            candidate_id=cutpoint.candidate_id,
            parent_node_id=cutpoint.parent_node_id,
            source_cluster_id=cutpoint.source_cluster_id,
            input_cut_offset=cutpoint.cut_offset,
            cut_offset=cutpoint.cut_offset,
            decision="reject",
            resolved_cut_offset=resolved_cut_offset,
            anchor_match_mode=cast(AnchorMatchMode, resolution.match_mode),
            anchor_match_score=resolution.match_score,
            text_before_cut=cutpoint.text_before_cut,
            text_after_cut=cutpoint.text_after_cut,
            reason=legality_reason,
        )

    boundary_kind = _classify_boundary_kind(text, resolved_local_cut)
    if boundary_kind == "semantic":
        boundary_kind = cutpoint.boundary_kind
    normalized_kind = (
        boundary_kind
        if boundary_kind
        in {"section", "paragraph", "list_item", "table_row", "sentence", "word"}
        else "semantic"
    )
    if resolved_cut_offset == cutpoint.cut_offset:
        decision = "accept"
        reason = resolution.reason or "aligned with a unique anchored boundary"
    else:
        decision = (
            "shift_left" if resolved_cut_offset < cutpoint.cut_offset else "shift_right"
        )
        reason = resolution.reason or "shifted to a uniquely anchored boundary"
    return BoundaryReviewDecision(
        candidate_id=cutpoint.candidate_id,
        parent_node_id=cutpoint.parent_node_id,
        source_cluster_id=cutpoint.source_cluster_id,
        input_cut_offset=cutpoint.cut_offset,
        cut_offset=cutpoint.cut_offset,
        decision=decision,
        resolved_cut_offset=resolved_cut_offset,
        boundary_kind=cast(BoundaryKind, normalized_kind),
        anchor_match_mode=cast(AnchorMatchMode, resolution.match_mode),
        anchor_match_score=resolution.match_score,
        text_before_cut=cutpoint.text_before_cut,
        text_after_cut=cutpoint.text_after_cut,
        reason=reason,
    )


def _boundary_unit_summaries(
    *,
    current_layer_context: CurrentLayerContext,
    parser_source_map: Mapping[str, Mapping[str, object]],
    accepted_units: Sequence[_BoundaryAcceptedUnit],
) -> list[BoundaryUnitSummary]:
    summaries: list[BoundaryUnitSummary] = []
    for unit in accepted_units:
        summaries.append(
            _make_boundary_summary(
                parent_node_id=str(unit["parent_node_id"]),
                source_cluster_id=str(unit["source_cluster_id"]),
                start_char=unit["start_char"],
                end_char=unit["end_char"],
                parser_source_map=parser_source_map,
                boundary_kind=unit["boundary_kind"],
            )
        )
    return summaries


def _boundary_parent_coverage_report(
    *,
    parent_node_id: str,
    source_cluster_id: str,
    start_char: int,
    end_char_exclusive: int,
    segments: Sequence[_BoundarySegment],
) -> _BoundaryCoverageReport:
    covered_ranges: list[_BoundaryRange] = [
        {
            "start_char": segment["start_char"],
            "end_char": segment["end_char"],
        }
        for segment in segments
        if not segment.get("skipped", False)
    ]
    gap_ranges: list[_BoundaryGapRange] = [
        {
            "start_char": segment["start_char"],
            "end_char": segment["end_char"],
            "reason": segment.get("skip_reason", "unresolved"),
        }
        for segment in segments
        if segment.get("skipped", False)
    ]
    covered_start = covered_ranges[0]["start_char"] if covered_ranges else start_char
    covered_end = covered_ranges[-1]["end_char"] if covered_ranges else start_char - 1
    coverage_state = (
        "covered"
        if not gap_ranges
        and covered_start == start_char
        and covered_end >= end_char_exclusive - 1
        else "covered_with_gaps"
    )
    return {
        "parent_node_id": parent_node_id,
        "source_cluster_id": source_cluster_id,
        "parent_span": {
            "start_char": start_char,
            "end_char": end_char_exclusive - 1,
        },
        "covered_span": {
            "start_char": covered_start,
            "end_char": covered_end,
        },
        "coverage_state": coverage_state,
        "gap_count": len(gap_ranges),
        "gap_ranges": gap_ranges,
    }


def _assemble_layer_result_from_boundaries(
    *,
    current_layer_context: CurrentLayerContext,
    parser_source_map: Mapping[str, Mapping[str, object]],
    review_batch: BoundaryReviewBatch,
) -> tuple[
    CurrentLayerResult, list[BoundaryUnitSummary], list[_AcceptedBoundaryCutpoint]
]:
    accepted_cutpoints: list[_AcceptedBoundaryCutpoint] = []
    child_items: list[_BoundaryAcceptedUnit] = []
    coverage_reports: list[_BoundaryCoverageReport] = []
    unresolved_intervals = 0
    decisions_by_parent: dict[tuple[str, str], list[BoundaryReviewDecision]] = {}
    for decision in review_batch.decisions:
        key = (decision.parent_node_id, decision.source_cluster_id)
        decisions_by_parent.setdefault(key, []).append(decision)

    for parent_id in list(current_layer_context.parent_node_ids):
        parent_pointers = list(
            current_layer_context.parent_content_pointers_by_id.get(parent_id) or []
        )
        pointer_index = 0
        for pointer in parent_pointers:
            source_cluster_id = _pointer_source_cluster_id(pointer)
            text = _pointer_text(pointer, parser_source_map=parser_source_map)
            start_char, end_char_exclusive = _pointer_span_bounds(
                pointer, parser_source_map=parser_source_map
            )
            absolute_offsets = [start_char]
            pointer_segments: list[_BoundarySegment] = []
            pointer_decisions = decisions_by_parent.get(
                (parent_id, source_cluster_id), []
            )
            for decision in pointer_decisions:
                resolved_offset = (
                    decision.resolved_cut_offset
                    if decision.resolved_cut_offset is not None
                    else decision.cut_offset
                )
                if resolved_offset is None:
                    unresolved_intervals += 1
                    continue
                if (
                    resolved_offset <= start_char
                    or resolved_offset >= end_char_exclusive
                ):
                    unresolved_intervals += 1
                    continue
                if decision.decision in {"reject", "needs_refinement"}:
                    unresolved_intervals += 1
                    continue
                absolute_offsets.append(resolved_offset)
                accepted_cutpoints.append(
                    {
                        "parent_node_id": parent_id,
                        "source_cluster_id": source_cluster_id,
                        "cut_offset": resolved_offset,
                        "boundary_kind": decision.boundary_kind or "semantic",
                    }
                )
            absolute_offsets.append(end_char_exclusive)
            absolute_offsets = sorted(set(absolute_offsets))
            for segment_index, (segment_start, segment_end) in enumerate(
                pairwise(absolute_offsets),
                start=0,
            ):
                if segment_end <= segment_start:
                    pointer_segments.append(
                        {
                            "start_char": segment_start,
                            "end_char": segment_end - 1,
                            "skipped": True,
                            "skip_reason": "empty_segment",
                        }
                    )
                    continue
                segment_text = text[
                    (segment_start - start_char) : (segment_end - start_char)
                ]
                if not segment_text.strip():
                    pointer_segments.append(
                        {
                            "start_char": segment_start,
                            "end_char": segment_end - 1,
                            "skipped": True,
                            "skip_reason": "whitespace_only_segment",
                        }
                    )
                    continue
                pointer_segments.append(
                    {
                        "start_char": segment_start,
                        "end_char": segment_end - 1,
                        "skipped": False,
                    }
                )
                child_node_id = f"{parent_id}|{source_cluster_id}|segment-{pointer_index}-{segment_index}"
                boundary_kind = "semantic"
                for decision in pointer_decisions:
                    resolved_offset = (
                        decision.resolved_cut_offset
                        if decision.resolved_cut_offset is not None
                        else decision.cut_offset
                    )
                    if (
                        resolved_offset == segment_end
                        and decision.boundary_kind is not None
                    ):
                        boundary_kind = decision.boundary_kind
                        break
                child = LayerChildCandidate(
                    node_id=child_node_id,
                    parent_node_id=parent_id,
                    title=_trim_text(
                        segment_text.splitlines()[0]
                        if segment_text.splitlines()
                        else segment_text,
                        max_chars=160,
                    ),
                    node_type="TEXT_FLOW",
                    total_content_pointers=[
                        HydratedTextPointer(
                            source_cluster_id=source_cluster_id,
                            start_char=segment_start,
                            end_char=segment_end - 1,
                            verbatim_text=segment_text,
                        )
                    ],
                    expandable=bool(
                        _legal_cutpoints_for_text(segment_text, max_points=4)
                    ),
                    metadata={
                        "source": "boundary_first",
                        "boundary_kind": boundary_kind,
                    },
                )
                child_items.append(
                    {
                        "parent_node_id": parent_id,
                        "source_cluster_id": source_cluster_id,
                        "start_char": segment_start,
                        "end_char": segment_end - 1,
                        "boundary_kind": boundary_kind,
                        "verbatim_text": segment_text,
                        "summary_text": _trim_multiline_text(
                            segment_text, max_lines=2, max_chars=220
                        ),
                        "exact_text": _trim_text(segment_text, max_chars=500),
                        "expandable": child.expandable,
                    }
                )
                pointer_index += 1
            coverage_reports.append(
                _boundary_parent_coverage_report(
                    parent_node_id=parent_id,
                    source_cluster_id=source_cluster_id,
                    start_char=start_char,
                    end_char_exclusive=end_char_exclusive,
                    segments=pointer_segments,
                )
            )

        if not parent_pointers:
            unresolved_intervals += 1

    if not child_items:
        return (
            CurrentLayerResult(
                children=[],
                satisfied=True,
                atomic_retained=True,
                reasoning_history=[
                    LayerReasoningEntry(source="boundary_first_deterministic_empty")
                ],
                metadata={
                    "fallback": "boundary_empty",
                    "allow_empty_layer": True,
                },
            ),
            [],
            [],
        )

    children = [
        LayerChildCandidate(
            node_id=f"{item['parent_node_id']}|{item['source_cluster_id']}|unit-{index}",
            parent_node_id=str(item["parent_node_id"]),
            title=str(item["summary_text"] or item["exact_text"] or "Boundary Unit"),
            node_type="TEXT_FLOW",
            total_content_pointers=[
                HydratedTextPointer(
                    source_cluster_id=str(item["source_cluster_id"]),
                    start_char=_offset_value(item.get("start_char"), 0),
                    end_char=_offset_value(item.get("end_char"), 0),
                    verbatim_text=str(
                        item["verbatim_text"]
                        or item["exact_text"]
                        or item["summary_text"]
                        or ""
                    ),
                )
            ],
            expandable=bool(item["expandable"]),
            metadata=cast(
                dict[str, JsonValue],
                {
                    "source": "boundary_first",
                    "boundary_kind": item["boundary_kind"],
                    "summary_text": item["summary_text"],
                    "exact_text": item["exact_text"],
                },
            ),
        )
        for index, item in enumerate(child_items)
    ]

    summaries: list[BoundaryUnitSummary] = _boundary_unit_summaries(
        current_layer_context=current_layer_context,
        parser_source_map=parser_source_map,
        accepted_units=list(child_items),
    )
    result: CurrentLayerResult = CurrentLayerResult(
        children=children,
        satisfied=True,
        reasoning_history=[],
        metadata=cast(
            dict[str, JsonValue],
            {
                "proposal_mode": "boundaries",
                "boundary_unit_summaries": [
                    summary.model_dump() for summary in summaries
                ],
                "boundary_parent_coverage": coverage_reports,
                "boundary_review_notes": list(review_batch.review_notes),
                "unresolved_interval_count": unresolved_intervals,
            },
        ),
    )
    return result, summaries, accepted_cutpoints


def _boundary_identical_parent_child_ids(
    *,
    current_layer_context: CurrentLayerContext,
    current_layer_result: CurrentLayerResult,
) -> list[str]:
    """Find assembled children that reproduce a complete parent pointer."""

    parent_pointers_by_id = dict(current_layer_context.parent_content_pointers_by_id)
    identical_ids: list[str] = []
    for child in current_layer_result.children:
        child_pointers = list(child.total_content_pointers or [])
        if len(child_pointers) != 1:
            continue
        child_pointer = child_pointers[0]
        child_key = (
            _pointer_source_cluster_id(child_pointer),
            _offset_value(_pointer_field(child_pointer, "start_char"), 0),
            _offset_value(_pointer_field(child_pointer, "end_char"), -1),
        )
        for parent_pointer in parent_pointers_by_id.get(child.parent_node_id, []):
            parent_key = (
                _pointer_source_cluster_id(parent_pointer),
                _offset_value(_pointer_field(parent_pointer, "start_char"), 0),
                _offset_value(_pointer_field(parent_pointer, "end_char"), -1),
            )
            if child_key == parent_key:
                identical_ids.append(str(child.node_id))
                break
    return identical_ids


def _annotate_proposal_result(
    result: CurrentLayerResult,
    *,
    proposal_source: str,
    proposal_mode: str = "children",
    proposal_failure_reason: str | None = None,
    failure_type: FailureCategory | None = None,
    provider_child_count: int | None = None,
    boundary_count: int | None = None,
    accepted_boundary_count: int | None = None,
    shifted_boundary_count: int | None = None,
    rejected_boundary_count: int | None = None,
    refinement_count: int | None = None,
    unresolved_interval_count: int | None = None,
    summary_count: int | None = None,
) -> CurrentLayerResult:
    metadata = dict(result.metadata)
    metadata["proposal_source"] = proposal_source
    metadata["proposal_mode"] = proposal_mode
    if proposal_failure_reason:
        metadata["proposal_failure_reason"] = proposal_failure_reason
    if failure_type is not None:
        metadata["failure_type"] = failure_type
    if boundary_count is not None:
        metadata["boundary_count"] = boundary_count
    if accepted_boundary_count is not None:
        metadata["accepted_boundary_count"] = accepted_boundary_count
    if shifted_boundary_count is not None:
        metadata["shifted_boundary_count"] = shifted_boundary_count
    if rejected_boundary_count is not None:
        metadata["rejected_boundary_count"] = rejected_boundary_count
    if refinement_count is not None:
        metadata["refinement_count"] = refinement_count
    if unresolved_interval_count is not None:
        metadata["unresolved_interval_count"] = unresolved_interval_count
    if summary_count is not None:
        metadata["summary_count"] = summary_count
    reasoning_history = list(result.reasoning_history)
    marker_payload: dict[str, object] = {
        "source": "workflow_layered_proposal",
        "proposal_source": proposal_source,
        "proposal_mode": proposal_mode,
    }
    if provider_child_count is not None:
        marker_payload["provider_child_count"] = provider_child_count
    if boundary_count is not None:
        marker_payload["boundary_count"] = boundary_count
    if accepted_boundary_count is not None:
        marker_payload["accepted_boundary_count"] = accepted_boundary_count
    if shifted_boundary_count is not None:
        marker_payload["shifted_boundary_count"] = shifted_boundary_count
    if rejected_boundary_count is not None:
        marker_payload["rejected_boundary_count"] = rejected_boundary_count
    if refinement_count is not None:
        marker_payload["refinement_count"] = refinement_count
    if unresolved_interval_count is not None:
        marker_payload["unresolved_interval_count"] = unresolved_interval_count
    if summary_count is not None:
        marker_payload["summary_count"] = summary_count
    if proposal_failure_reason:
        marker_payload["proposal_failure_reason"] = proposal_failure_reason
    if failure_type is not None:
        marker_payload["failure_type"] = failure_type
    reasoning_history.append(LayerReasoningEntry.model_validate(marker_payload))
    payload = result.model_dump()
    payload["metadata"] = metadata
    payload["reasoning_history"] = [entry.model_dump() for entry in reasoning_history]
    return result.__class__.model_validate(payload)
