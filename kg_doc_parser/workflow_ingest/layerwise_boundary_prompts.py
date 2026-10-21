"""Source-grounded prompt context and candidate normalization helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from .layerwise_boundaries import BoundaryKind, _legal_cutpoints_for_text
from .layerwise_boundary_contracts import (
    _BoundaryLegalCutpoint,
    _BoundaryLookupCandidate,
    _BoundaryPromptCandidate,
    _ParentContextExcerpt,
    _PointerExcerpt,
    _ProposalAttemptContext,
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
    BoundaryUnitSummary,
    CurrentLayerContext,
    LLMBoundaryProposalBatch,
    ParseSessionState,
)
from .serialization import ModelDumpLike


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
