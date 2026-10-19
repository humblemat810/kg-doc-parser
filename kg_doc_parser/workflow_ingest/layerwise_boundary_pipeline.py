"""Boundary proposal validation and source-grounded layer assembly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Literal, NotRequired, TypedDict, cast

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
from .layerwise_payloads import _dump_model, _object_list
from .layerwise_source_spans import (
    _offset_value,
    _pointer_end_inclusive,
    _pointer_field,
    _pointer_source_cluster_id,
    _pointer_span_bounds,
    _pointer_span_text,
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
    ParseSessionState,
)
from .parser_core import SplitStrategy
from .semantics import HydratedTextPointer
from .serialization import JsonValue, ModelDumpLike


class _BoundaryPointerSpan(TypedDict):
    start_char: int
    end_char: int


class _BoundaryLegalCutpoint(TypedDict):
    candidate_id: str
    cut_offset: int
    boundary_kind: str
    confidence: float
    reason: str
    text_before_cut: str
    text_after_cut: str
    text_before_cut_preview: str
    text_after_cut_preview: str
    preview: str


class _BoundaryPromptCandidate(TypedDict):
    parent_node_id: str
    source_cluster_id: str
    pointer_excerpt: str
    pointer_span: _BoundaryPointerSpan
    legal_cutpoints: list[_BoundaryLegalCutpoint]


class _BoundaryLookupCandidate(_BoundaryLegalCutpoint):
    parent_node_id: str
    source_cluster_id: str


class _BoundaryRefinementCutpoint(TypedDict):
    cut_offset: int
    boundary_kind: str
    confidence: float
    reason: str


class _BoundaryRefinementPromptContext(TypedDict):
    parent_node_id: str
    source_cluster_id: str
    input_cut_offset: int | None
    text_before_cut: str
    text_after_cut: str
    cut_reason: str
    parent_ids: list[str]
    pointer_excerpt: str
    legal_cutpoints: list[_BoundaryRefinementCutpoint]


class _BoundarySegment(TypedDict):
    start_char: int
    end_char: int
    skipped: NotRequired[bool]
    skip_reason: NotRequired[str]


class _BoundaryRange(TypedDict):
    start_char: int
    end_char: int


class _BoundaryGapRange(_BoundaryRange):
    reason: str


class _BoundaryCoverageReport(TypedDict):
    parent_node_id: str
    source_cluster_id: str
    parent_span: _BoundaryRange
    covered_span: _BoundaryRange
    coverage_state: Literal["covered", "covered_with_gaps"]
    gap_count: int
    gap_ranges: list[_BoundaryGapRange]


class _BoundaryAcceptedUnit(TypedDict):
    parent_node_id: str
    source_cluster_id: str
    start_char: int
    end_char: int
    boundary_kind: str
    verbatim_text: str
    summary_text: str
    exact_text: str
    expandable: bool


class _AcceptedBoundaryCutpoint(TypedDict):
    parent_node_id: str
    source_cluster_id: str
    cut_offset: int
    boundary_kind: str


class _BoundaryReviewDecisionSummary(TypedDict):
    candidate_id: str | None
    decision: str
    parent_node_id: str
    source_cluster_id: str
    input_cut_offset: int | None
    resolved_cut_offset: int | None
    boundary_kind: str | None
    anchor_match_mode: str | None
    anchor_match_score: float | None
    reason: str


class _ProposalAttemptContext(TypedDict):
    attempt_number_for_depth: int
    attempts_by_depth: dict[str, int]
    strategy_history: list[SplitStrategy]
    fallback_split_strategy: SplitStrategy | None
    workflow_mode: Literal["workflow_layered", "legacy_compat"] | None
    prior_review: dict[str, JsonValue]


class _PointerExcerpt(TypedDict):
    source_cluster_id: str
    start_char: int | None
    end_char: int | None
    verbatim_text: str


class _ParentContextExcerpt(TypedDict):
    parent_node_id: str
    title: str | None
    pointer_count: int
    pointer_excerpt: list[_PointerExcerpt]


def _trim_multiline_text(
    value: object, *, max_lines: int = 3, max_chars: int = 400
) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) > max_lines:
        text = "\n".join(lines[:max_lines]) + "\n..."
    return _trim_text(text, max_chars=max_chars)


def _summarize_for_prompt(
    value: object,
    *,
    max_depth: int = 3,
    max_items: int = 6,
    max_string: int = 400,
) -> object:
    if isinstance(value, ModelDumpLike):
        value = _dump_model(value)
    if isinstance(value, dict):
        if max_depth <= 0:
            return {"summary": f"{len(value)} keys omitted"}
        items = list(value.items())
        summary: dict[str, object] = {}
        for key, item in items[:max_items]:
            summary[str(key)] = _summarize_for_prompt(
                item,
                max_depth=max_depth - 1,
                max_items=max_items,
                max_string=max_string,
            )
        if len(items) > max_items:
            summary["..."] = f"{len(items) - max_items} more keys omitted"
        return summary
    if isinstance(value, list):
        if max_depth <= 0:
            return [f"{len(value)} items omitted"]
        items = [
            _summarize_for_prompt(
                item,
                max_depth=max_depth - 1,
                max_items=max_items,
                max_string=max_string,
            )
            for item in value[:max_items]
        ]
        if len(value) > max_items:
            items.append(f"... {len(value) - max_items} more items omitted")
        return items
    if isinstance(value, str):
        return _trim_text(value, max_chars=max_string)
    return value


def _boundary_prompt_candidate_context(
    *,
    current_layer_context: CurrentLayerContext,
    parser_source_map: Mapping[str, Mapping[str, object]],
    max_points: int = 128,
) -> list[_BoundaryPromptCandidate]:
    parent_ids = list(current_layer_context.parent_node_ids)
    pointers_by_id = dict(current_layer_context.parent_content_pointers_by_id)
    candidates: list[_BoundaryPromptCandidate] = []
    for parent_id in parent_ids:
        for pointer in _object_list(pointers_by_id.get(parent_id)):
            text = _pointer_text(pointer, parser_source_map=parser_source_map)
            start_char = _offset_value(_pointer_field(pointer, "start_char"), 0)
            legal = _legal_cutpoints_for_text(text, max_points=max_points)
            legal_cutpoints: list[_BoundaryLegalCutpoint] = []
            for boundary in legal:
                legal_cutpoints.append(
                    {
                        "candidate_id": (
                            f"{parent_id}|{_pointer_field(pointer, 'source_cluster_id') or ''!s}|"
                            f"{start_char}|{boundary.candidate_id or boundary.cut_offset}"
                        ),
                        "cut_offset": start_char + boundary.cut_offset,
                        "boundary_kind": boundary.boundary_kind,
                        "confidence": float(boundary.confidence or 0.0),
                        "reason": str(boundary.reason or ""),
                        "text_before_cut": text[
                            max(0, boundary.cut_offset - 20) : boundary.cut_offset
                        ],
                        "text_after_cut": text[
                            boundary.cut_offset : min(
                                len(text), boundary.cut_offset + 20
                            )
                        ],
                        "text_before_cut_preview": _trim_text(
                            text[
                                max(0, boundary.cut_offset - 20) : boundary.cut_offset
                            ],
                            max_chars=60,
                        ),
                        "text_after_cut_preview": _trim_text(
                            text[
                                boundary.cut_offset : min(
                                    len(text), boundary.cut_offset + 20
                                )
                            ],
                            max_chars=60,
                        ),
                        "preview": _trim_text(
                            _pointer_span_text(
                                pointer,
                                parser_source_map=parser_source_map,
                                start_offset=max(
                                    start_char, start_char + boundary.cut_offset - 30
                                ),
                                end_offset=min(
                                    _pointer_span_bounds(
                                        pointer, parser_source_map=parser_source_map
                                    )[1],
                                    start_char + boundary.cut_offset + 30,
                                ),
                            ),
                            max_chars=120,
                        ),
                    }
                )
            candidates.append(
                {
                    "parent_node_id": parent_id,
                    "source_cluster_id": _pointer_source_cluster_id(pointer),
                    "pointer_excerpt": _trim_multiline_text(
                        text, max_lines=4, max_chars=700
                    ),
                    "pointer_span": {
                        "start_char": start_char,
                        "end_char": _pointer_end_inclusive(
                            pointer, parser_source_map=parser_source_map
                        ),
                    },
                    "legal_cutpoints": legal_cutpoints,
                }
            )
    return candidates


def _boundary_candidate_lookup(
    boundary_candidates: list[_BoundaryPromptCandidate],
) -> dict[str, _BoundaryLookupCandidate]:
    lookup: dict[str, _BoundaryLookupCandidate] = {}
    for parent in boundary_candidates:
        parent_node_id = str(parent.get("parent_node_id") or "")
        source_cluster_id = str(parent.get("source_cluster_id") or "")
        for candidate in _object_list(parent.get("legal_cutpoints")):
            if not isinstance(candidate, dict):
                continue
            candidate_id = str(candidate.get("candidate_id") or "")
            if not candidate_id:
                continue
            lookup[candidate_id] = {
                "candidate_id": candidate_id,
                "cut_offset": _offset_value(candidate.get("cut_offset"), 0),
                "boundary_kind": str(candidate.get("boundary_kind") or "semantic"),
                "confidence": float(candidate.get("confidence") or 0.0),
                "reason": str(candidate.get("reason") or ""),
                "text_before_cut": str(candidate.get("text_before_cut") or ""),
                "text_after_cut": str(candidate.get("text_after_cut") or ""),
                "text_before_cut_preview": str(
                    candidate.get("text_before_cut_preview") or ""
                ),
                "text_after_cut_preview": str(
                    candidate.get("text_after_cut_preview") or ""
                ),
                "preview": str(candidate.get("preview") or ""),
                "parent_node_id": parent_node_id,
                "source_cluster_id": source_cluster_id,
            }
    return lookup


def _normalize_boundary_cutpoints_from_candidates(
    parsed: LLMBoundaryProposalBatch,
    *,
    candidate_lookup: dict[str, _BoundaryLookupCandidate],
) -> LLMBoundaryProposalBatch:
    normalized: list[BoundaryCutpoint] = []
    for cutpoint in parsed.cutpoints:
        candidate_id = str(cutpoint.candidate_id or "")
        candidate = candidate_lookup.get(candidate_id)
        if candidate is None:
            parent_node_id = str(cutpoint.parent_node_id or "")
            source_cluster_id = str(cutpoint.source_cluster_id or "")
            cut_offset = (
                cutpoint.cut_offset if type(cutpoint.cut_offset) is int else None
            )
            if parent_node_id and source_cluster_id and cut_offset is not None:
                for lookup_candidate in candidate_lookup.values():
                    if (
                        str(lookup_candidate.get("parent_node_id") or "")
                        == parent_node_id
                        and str(lookup_candidate.get("source_cluster_id") or "")
                        == source_cluster_id
                        and _offset_value(lookup_candidate.get("cut_offset"), -1)
                        == cut_offset
                    ):
                        candidate = lookup_candidate
                        break
        if candidate is None:
            normalized.append(cutpoint)
            continue
        normalized.append(
            cutpoint.model_copy(
                update={
                    "candidate_id": candidate_id,
                    "parent_node_id": str(candidate["parent_node_id"]),
                    "source_cluster_id": str(candidate["source_cluster_id"]),
                    "cut_offset": _offset_value(
                        candidate.get("cut_offset"), cutpoint.cut_offset
                    ),
                    "boundary_kind": str(
                        candidate.get("boundary_kind") or cutpoint.boundary_kind
                    ),
                    "text_before_cut": str(
                        candidate.get("text_before_cut")
                        or candidate.get("text_before_cut_preview")
                        or cutpoint.text_before_cut
                    ),
                    "text_after_cut": str(
                        candidate.get("text_after_cut")
                        or candidate.get("text_after_cut_preview")
                        or cutpoint.text_after_cut
                    ),
                }
            )
        )
    return parsed.model_copy(update={"cutpoints": normalized})


def _make_boundary_summary(
    *,
    parent_node_id: str,
    source_cluster_id: str,
    start_char: int,
    end_char: int,
    parser_source_map: Mapping[str, Mapping[str, object]],
    boundary_kind: str,
) -> BoundaryUnitSummary:
    record = parser_source_map.get(source_cluster_id) or parser_source_map.get(
        str(source_cluster_id)
    )
    text = _record_text(record)
    exact_text = ""
    if text and 0 <= start_char <= end_char < len(text):
        exact_text = text[start_char : end_char + 1]
    if not exact_text:
        exact_text = _pointer_span_text(
            {
                "source_cluster_id": source_cluster_id,
                "start_char": start_char,
                "end_char": end_char,
            },
            parser_source_map=parser_source_map,
        )
    summary = _trim_multiline_text(exact_text, max_lines=2, max_chars=220)
    normalized_kind = (
        boundary_kind
        if boundary_kind
        in {"section", "paragraph", "list_item", "table_row", "sentence", "word"}
        else "semantic"
    )
    return BoundaryUnitSummary(
        parent_node_id=parent_node_id,
        source_cluster_id=source_cluster_id,
        start_char=start_char,
        end_char=end_char,
        boundary_kind=cast(BoundaryKind, normalized_kind),
        summary_text=summary,
        exact_text=_trim_text(exact_text, max_chars=600),
        expandable=False,
    )


def _pointer_excerpt(
    pointer: object,
    *,
    parser_source_map: Mapping[str, Mapping[str, object]],
) -> _PointerExcerpt:
    source_cluster_id = _pointer_source_cluster_id(pointer)
    raw_start_char = _pointer_field(pointer, "start_char")
    raw_end_char = _pointer_field(pointer, "end_char")
    start_char = raw_start_char if type(raw_start_char) is int else None
    end_char = raw_end_char if type(raw_end_char) is int else None
    excerpt = ""
    record = parser_source_map.get(source_cluster_id) or parser_source_map.get(
        str(source_cluster_id)
    )
    if record is not None and type(start_char) is int and type(end_char) is int:
        raw_text = _record_text(record)
        resolved_end = len(raw_text) - 1 if end_char == -1 else end_char
        if raw_text and 0 <= start_char <= resolved_end < len(raw_text):
            excerpt = raw_text[start_char : resolved_end + 1]
    return {
        "source_cluster_id": source_cluster_id,
        "start_char": start_char,
        "end_char": end_char,
        "verbatim_text": _trim_text(excerpt, max_chars=400),
    }


def _parent_context_excerpt(
    *,
    current_layer_context: CurrentLayerContext,
    parser_source_map: Mapping[str, Mapping[str, object]],
) -> list[_ParentContextExcerpt]:
    parent_ids = list(current_layer_context.parent_node_ids)
    parent_titles = list(current_layer_context.parent_titles)
    pointers_by_id = dict(current_layer_context.parent_content_pointers_by_id)
    parents: list[_ParentContextExcerpt] = []
    for index, parent_id in enumerate(parent_ids):
        title = parent_titles[index] if index < len(parent_titles) else None
        pointers = list(pointers_by_id.get(parent_id) or [])
        parents.append(
            {
                "parent_node_id": parent_id,
                "title": title,
                "pointer_count": len(pointers),
                "pointer_excerpt": [
                    _pointer_excerpt(pointer, parser_source_map=parser_source_map)
                    for pointer in pointers[:3]
                ],
            }
        )
    return parents


def _proposal_attempt_context(
    *,
    parse_session: ParseSessionState | None,
    current_layer_context: CurrentLayerContext,
) -> _ProposalAttemptContext:
    if parse_session is None:
        return {
            "attempt_number_for_depth": 1,
            "attempts_by_depth": {},
            "strategy_history": [],
            "fallback_split_strategy": None,
            "workflow_mode": None,
            "prior_review": {},
        }
    depth_key = str(current_layer_context.depth)
    return {
        "attempt_number_for_depth": parse_session.layer_attempts.get(depth_key, 0) + 1,
        "attempts_by_depth": dict(parse_session.layer_attempts),
        "strategy_history": list(parse_session.strategy_history),
        "fallback_split_strategy": parse_session.fallback_split_strategy,
        "workflow_mode": parse_session.mode,
        "prior_review": dict(parse_session.last_review),
    }


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
