from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence, TypeVar, TypedDict

from pydantic import BaseModel

from kg_doc_parser.llm_structured_output import build_structured_output_runnable
from kogwistar.fuzzy_offsets import find_fuzzy_spans, offset_repair_threshold

from .models import (
    BoundaryCutpoint,
    BoundaryReviewBatch,
    BoundaryReviewDecision,
    BoundaryUnitSummary,
    CurrentLayerResult,
    CurrentLayerReview,
    LLMBoundaryProposalBatch,
    LLMCurrentLayerResult,
    LLMCurrentLayerReview,
    LayerChildCandidate,
    LayerReasoningEntry,
)
from kogwistar.runtime import RetryExhaustedError, RetryResult, retry_with_context
from .providers import SupportsStructuredOutput, WorkflowProviderSettings, build_chat_model_for_role
from .semantics import HydratedTextPointer

TStructuredModel = TypeVar("TStructuredModel", bound=BaseModel)
LayerwiseProposeCallback = Callable[..., CurrentLayerResult]
LayerwiseReviewCallback = Callable[..., CurrentLayerReview]


class LayerwiseLLMCallbacks(TypedDict):
    propose_layer_fn: LayerwiseProposeCallback
    review_layer_fn: LayerwiseReviewCallback
    max_depth: int
    allow_review: bool


LayerwiseCallback = LayerwiseLLMCallbacks


MAX_BOUNDARY_REPAIR_SHIFT_CHARS = 8


@dataclass(frozen=True, slots=True)
class _BoundaryAnchorResolution:
    resolved_cut_offset: int | None
    match_mode: str | None
    match_score: float | None
    reason: str | None


def _trim_text(value: Any, *, max_chars: int = 400) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def _trim_multiline_text(value: Any, *, max_lines: int = 3, max_chars: int = 400) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) > max_lines:
        text = "\n".join(lines[:max_lines]) + "\n..."
    return _trim_text(text, max_chars=max_chars)


def _dump_model(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(field_mode="backend", dump_format="json")
        except TypeError:
            return value.model_dump()
    if isinstance(value, dict):
        return {str(key): _dump_model(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_dump_model(item) for item in value]
    return value


def _summarize_for_prompt(
    value: Any,
    *,
    max_depth: int = 3,
    max_items: int = 6,
    max_string: int = 400,
) -> Any:
    if hasattr(value, "model_dump"):
        value = _dump_model(value)
    if isinstance(value, dict):
        if max_depth <= 0:
            return {"summary": f"{len(value)} keys omitted"}
        items = list(value.items())
        summary: dict[str, Any] = {}
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


def _source_map_excerpt(
    parser_source_map: dict[str, dict[str, Any]],
    *,
    max_records: int = 12,
    max_text_chars: int = 1200,
) -> dict[str, dict[str, Any]]:
    excerpt: dict[str, dict[str, Any]] = {}
    for key, record in list(parser_source_map.items())[:max_records]:
        text = str(record.get("text") or "")
        excerpt[str(key)] = {
            "page_number": record.get("page_number"),
            "cluster_number": record.get("cluster_number"),
            "text": _trim_text(text, max_chars=max_text_chars),
        }
    return excerpt


def _pointer_field(pointer: Any, field_name: str) -> Any:
    if isinstance(pointer, dict):
        return pointer.get(field_name)
    return getattr(pointer, field_name, None)


def _pointer_signature(pointer: Any, *, parser_source_map: dict[str, dict[str, Any]]) -> tuple[str, int, int, str]:
    return (
        str(_pointer_field(pointer, "source_cluster_id") or ""),
        int(_pointer_field(pointer, "start_char") or 0),
        int(_pointer_field(pointer, "end_char") or -1),
        _pointer_text(pointer, parser_source_map=parser_source_map).strip(),
    )


def _pointer_end_inclusive(
    pointer: Any,
    *,
    parser_source_map: dict[str, dict[str, Any]],
) -> int:
    end_char = _pointer_field(pointer, "end_char")
    if isinstance(end_char, int) and end_char >= 0:
        return end_char
    source_cluster_id = str(_pointer_field(pointer, "source_cluster_id") or "")
    record = parser_source_map.get(source_cluster_id) or parser_source_map.get(str(source_cluster_id))
    text = str(record.get("text") or "") if record is not None else ""
    if text:
        return max(0, len(text) - 1)
    start_char = _pointer_field(pointer, "start_char")
    return int(start_char) if isinstance(start_char, int) else 0


def _pointer_span_bounds(
    pointer: Any,
    *,
    parser_source_map: dict[str, dict[str, Any]],
) -> tuple[int, int]:
    start_char = int(_pointer_field(pointer, "start_char") or 0)
    end_char_exclusive = _pointer_end_inclusive(pointer, parser_source_map=parser_source_map) + 1
    return start_char, max(start_char + 1, end_char_exclusive)


def _pointer_text(
    pointer: Any,
    *,
    parser_source_map: dict[str, dict[str, Any]],
) -> str:
    source_cluster_id = str(_pointer_field(pointer, "source_cluster_id") or "")
    start_char = _pointer_field(pointer, "start_char")
    end_char = _pointer_field(pointer, "end_char")
    verbatim_text = str(_pointer_field(pointer, "verbatim_text") or "")
    record = parser_source_map.get(source_cluster_id) or parser_source_map.get(str(source_cluster_id))
    if record is not None and isinstance(start_char, int) and isinstance(end_char, int):
        raw_text = str(record.get("text") or "")
        if 0 <= start_char <= end_char < len(raw_text):
            return raw_text[start_char : end_char + 1]
    return verbatim_text


def _pointer_span_text(
    pointer: Any,
    *,
    parser_source_map: dict[str, dict[str, Any]],
    start_offset: int | None = None,
    end_offset: int | None = None,
) -> str:
    source_cluster_id = str(_pointer_field(pointer, "source_cluster_id") or "")
    start_char, end_char_exclusive = _pointer_span_bounds(pointer, parser_source_map=parser_source_map)
    record = parser_source_map.get(source_cluster_id) or parser_source_map.get(str(source_cluster_id))
    raw_text = str(record.get("text") or "") if record is not None else ""
    if not raw_text:
        return _trim_text(_pointer_text(pointer, parser_source_map=parser_source_map))
    start = max(start_char, 0 if start_offset is None else start_offset)
    end = end_char_exclusive if end_offset is None else min(end_char_exclusive, end_offset)
    if end <= start:
        return ""
    if end >= len(raw_text):
        end = len(raw_text)
    if start >= len(raw_text):
        return ""
    return raw_text[start:end]


def _is_heading_line(line: str) -> bool:
    stripped = line.lstrip()
    return bool(re.match(r"^#{1,6}\s+\S", stripped))


def _is_list_item_line(line: str) -> bool:
    stripped = line.lstrip()
    return bool(re.match(r"^([-*+]\s+|\d+[.)]\s+|[a-zA-Z][.)]\s+)", stripped))


def _classify_boundary_kind(text: str, offset: int) -> str:
    if offset <= 0 or offset >= len(text):
        return "semantic"
    prev_char = text[offset - 1]
    next_char = text[offset]
    if text[max(0, offset - 2) : offset] == "\n\n":
        return "paragraph"
    line_start = text.rfind("\n", 0, offset - 1) + 1
    current_line = text[line_start:offset]
    if prev_char == "\n" and _is_heading_line(current_line):
        return "section"
    if prev_char == "\n" and _is_list_item_line(current_line):
        return "list_item"
    if prev_char in ".!?" and (next_char.isspace() or next_char.isupper() or next_char.isdigit()):
        return "sentence"
    if prev_char.isspace() or next_char.isspace():
        return "word"
    return "semantic"


def _boundary_kind_priority(kind: str) -> int:
    return {
        "section": 5,
        "paragraph": 4,
        "list_item": 3,
        "sentence": 2,
        "word": 1,
        "semantic": 0,
    }.get(kind, 0)


def _boundary_cutpoint_legality_reason(text: str, offset: int) -> str | None:
    if offset <= 0 or offset >= len(text):
        return "boundary proposal used a cut_offset outside the text span"
    left = text[offset - 1]
    right = text[offset]
    if left.isalnum() and right.isalnum():
        return "boundary proposal cut through a word or identifier"
    if left.isdigit() and right.isdigit():
        return "boundary proposal cut through a number"

    window_start = max(0, offset - 60)
    window_end = min(len(text), offset + 60)
    window = text[window_start:window_end]
    local_offset = offset - window_start
    url_match = re.search(r"(https?://|www\.)[^\s]+", window)
    if url_match and url_match.start() < local_offset < url_match.end():
        return "boundary proposal cut through a URL"
    if "`" in window and window.count("`") % 2 == 1:
        return "boundary proposal cut through a code span"
    citation_match = re.search(r"\[[0-9A-Za-z,\s;:-]+\]|\([^)]*\d{4}[^)]*\)", window)
    if citation_match and citation_match.start() < local_offset < citation_match.end():
        return "boundary proposal cut through a citation or reference"
    line_start = text.rfind("\n", 0, offset - 1) + 1
    line_prefix = text[line_start:offset]
    if re.match(r"^\s*(#{1,6}\s*|[-*+]\s*|\d+[.)]\s*|[a-zA-Z][.)]\s*)$", line_prefix):
        return "boundary proposal cut through a heading or list marker"
    return None


def _legal_cutpoints_for_text(
    text: str,
    *,
    max_points: int = 48,
) -> list[BoundaryCutpoint]:
    if not text:
        return []
    best: dict[int, str] = {}
    for offset in range(1, len(text)):
        kind = _classify_boundary_kind(text, offset)
        if kind == "semantic":
            continue
        existing = best.get(offset)
        if existing is None or _boundary_kind_priority(kind) > _boundary_kind_priority(existing):
            best[offset] = kind
    ordered: list[BoundaryCutpoint] = [
        BoundaryCutpoint(
            parent_node_id="",
            source_cluster_id="",
            cut_offset=offset,
            boundary_kind=kind,
            text_before_cut=text[max(0, offset - 80) : offset],
            text_after_cut=text[offset : offset + 80],
            cut_reason="deterministic legal boundary",
            confidence=1.0,
            reason="deterministic_legal_boundary",
        )
        for offset, kind in sorted(best.items())
    ]
    if len(ordered) <= max_points:
        return ordered
    sampled: list[BoundaryCutpoint] = []
    step = max(1, len(ordered) // max_points)
    for index, boundary in enumerate(ordered):
        if index % step == 0 or len(sampled) < 4:
            sampled.append(boundary)
    return sampled[:max_points]


def _boundary_candidates_for_pointer(
    pointer: Any,
    *,
    parser_source_map: dict[str, dict[str, Any]],
    max_points: int = 32,
) -> list[BoundaryCutpoint]:
    text = _pointer_text(pointer, parser_source_map=parser_source_map)
    source_cluster_id = str(_pointer_field(pointer, "source_cluster_id") or "")
    parent_node_id = ""
    candidates = _legal_cutpoints_for_text(text, max_points=max_points)
    if not candidates:
        return []
    start_char = int(_pointer_field(pointer, "start_char") or 0)
    return [
        candidate.model_copy(
            update={
                "parent_node_id": parent_node_id,
                "source_cluster_id": source_cluster_id,
                "cut_offset": start_char + candidate.cut_offset,
            }
        )
        for candidate in candidates
    ]


def _boundary_prompt_candidate_context(
    *,
    current_layer_context: Any,
    parser_source_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    parent_ids = list(getattr(current_layer_context, "parent_node_ids", []) or [])
    pointers_by_id = dict(getattr(current_layer_context, "parent_content_pointers_by_id", {}) or {})
    candidates: list[dict[str, Any]] = []
    for parent_id in parent_ids:
        for pointer in list(pointers_by_id.get(parent_id) or []):
            text = _pointer_text(pointer, parser_source_map=parser_source_map)
            legal = _legal_cutpoints_for_text(text)
            start_char = int(_pointer_field(pointer, "start_char") or 0)
            candidates.append(
                {
                    "parent_node_id": parent_id,
                    "source_cluster_id": str(_pointer_field(pointer, "source_cluster_id") or ""),
                    "pointer_excerpt": _trim_multiline_text(text, max_lines=4, max_chars=700),
                    "pointer_span": {
                        "start_char": start_char,
                        "end_char": _pointer_end_inclusive(pointer, parser_source_map=parser_source_map),
                    },
                    "legal_cutpoints": [
                        {
                            "cut_offset": start_char + boundary.cut_offset,
                            "boundary_kind": boundary.boundary_kind,
                            "confidence": boundary.confidence,
                            "reason": boundary.reason,
                            "text_before_cut_preview": _trim_text(
                                text[max(0, boundary.cut_offset - 20) : boundary.cut_offset],
                                max_chars=60,
                            ),
                            "text_after_cut_preview": _trim_text(
                                text[boundary.cut_offset : min(len(text), boundary.cut_offset + 20)],
                                max_chars=60,
                            ),
                            "preview": _trim_text(
                                _pointer_span_text(
                                    pointer,
                                    parser_source_map=parser_source_map,
                                    start_offset=max(start_char, start_char + boundary.cut_offset - 30),
                                    end_offset=min(
                                        _pointer_span_bounds(pointer, parser_source_map=parser_source_map)[1],
                                        start_char + boundary.cut_offset + 30,
                                    ),
                                ),
                                max_chars=120,
                            ),
                        }
                        for boundary in legal
                    ],
                }
            )
    return candidates


def _boundary_anchor_occurrences(text: str, needle: str) -> list[int]:
    if not needle:
        return []
    positions: list[int] = []
    start = 0
    while True:
        index = text.find(needle, start)
        if index < 0:
            break
        positions.append(index)
        start = index + 1
    return positions


def _resolve_boundary_anchor(
    *,
    text: str,
    cut_offset: int,
    text_before_cut: str,
    text_after_cut: str,
) -> _BoundaryAnchorResolution:
    before = str(text_before_cut or "")
    after = str(text_after_cut or "")
    if not before or not after:
        return _BoundaryAnchorResolution(
            resolved_cut_offset=None,
            match_mode=None,
            match_score=None,
            reason="boundary proposal missing text_before_cut or text_after_cut",
        )

    anchor = before + after
    exact_positions = _boundary_anchor_occurrences(text, anchor)
    if len(exact_positions) == 1:
        return _BoundaryAnchorResolution(
            resolved_cut_offset=exact_positions[0] + len(before),
            match_mode="exact",
            match_score=1.0,
            reason="unique exact boundary anchor match",
        )
    if len(exact_positions) > 1:
        return _BoundaryAnchorResolution(
            resolved_cut_offset=None,
            match_mode=None,
            match_score=None,
            reason="boundary anchor is ambiguous across the source text",
        )

    origin_start = max(0, cut_offset - len(before))
    fuzzy_hits = find_fuzzy_spans(
        content=text,
        excerpt=anchor,
        origin_start=origin_start,
        scan_band=max(50, len(anchor) * 4),
        max_hits=3,
    )
    if not fuzzy_hits:
        return _BoundaryAnchorResolution(
            resolved_cut_offset=None,
            match_mode=None,
            match_score=None,
            reason="boundary anchor could not be resolved against the source text",
        )
    if len(fuzzy_hits) > 1:
        return _BoundaryAnchorResolution(
            resolved_cut_offset=None,
            match_mode=None,
            match_score=None,
            reason="boundary anchor is ambiguous after fuzzy matching",
        )
    hit = fuzzy_hits[0]
    threshold = offset_repair_threshold(len(anchor))
    if hit.score < threshold:
        return _BoundaryAnchorResolution(
            resolved_cut_offset=None,
            match_mode=None,
            match_score=hit.score,
            reason=f"boundary anchor fuzzy score {hit.score:.2f} below threshold {threshold:.2f}",
        )
    return _BoundaryAnchorResolution(
        resolved_cut_offset=hit.start + len(before),
        match_mode="fuzzy",
        match_score=hit.score,
        reason=f"boundary anchor repaired fuzzily with score {hit.score:.2f}",
    )


def _nearest_legal_cutpoint(
    *,
    cut_offset: int,
    legal_cutpoints: list[int],
) -> tuple[int | None, int | None]:
    if not legal_cutpoints:
        return None, None
    nearest = min(legal_cutpoints, key=lambda candidate: (abs(candidate - cut_offset), candidate))
    return nearest, abs(nearest - cut_offset)


def _make_boundary_summary(
    *,
    parent_node_id: str,
    source_cluster_id: str,
    start_char: int,
    end_char: int,
    parser_source_map: dict[str, dict[str, Any]],
    boundary_kind: str,
) -> BoundaryUnitSummary:
    record = parser_source_map.get(source_cluster_id) or parser_source_map.get(str(source_cluster_id))
    text = str(record.get("text") or "") if record is not None else ""
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
    normalized_kind = boundary_kind if boundary_kind in {"section", "paragraph", "list_item", "sentence", "word"} else "semantic"
    return BoundaryUnitSummary(
        parent_node_id=parent_node_id,
        source_cluster_id=source_cluster_id,
        start_char=start_char,
        end_char=end_char,
        boundary_kind=normalized_kind,
        summary_text=summary,
        exact_text=_trim_text(exact_text, max_chars=600),
        expandable=False,
    )


def _pointer_excerpt(pointer: Any, *, parser_source_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source_cluster_id = str(_pointer_field(pointer, "source_cluster_id") or "")
    start_char = _pointer_field(pointer, "start_char")
    end_char = _pointer_field(pointer, "end_char")
    verbatim_text = _pointer_field(pointer, "verbatim_text")
    excerpt = str(verbatim_text or "")
    record = parser_source_map.get(source_cluster_id) or parser_source_map.get(str(source_cluster_id))
    if record is not None and isinstance(start_char, int) and isinstance(end_char, int):
        raw_text = str(record.get("text") or "")
        if 0 <= start_char <= end_char < len(raw_text):
            excerpt = raw_text[start_char : end_char + 1]
    return {
        "source_cluster_id": source_cluster_id,
        "start_char": start_char,
        "end_char": end_char,
        "verbatim_text": _trim_text(excerpt or verbatim_text or "", max_chars=400),
    }


def _parent_context_excerpt(
    *,
    current_layer_context: Any,
    parser_source_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    parent_ids = list(getattr(current_layer_context, "parent_node_ids", []) or [])
    parent_titles = list(getattr(current_layer_context, "parent_titles", []) or [])
    pointers_by_id = dict(getattr(current_layer_context, "parent_content_pointers_by_id", {}) or {})
    parents: list[dict[str, Any]] = []
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


def _proposal_attempt_context(*, parse_session: Any, current_layer_context: Any) -> dict[str, Any]:
    parse_session_dump = _dump_model(parse_session) if parse_session is not None else {}
    attempts = dict(parse_session_dump.get("layer_attempts") or {})
    depth_key = str(getattr(current_layer_context, "depth", 0))
    return {
        "attempt_number_for_depth": int(attempts.get(depth_key, 0)) + 1,
        "attempts_by_depth": attempts,
        "strategy_history": list(parse_session_dump.get("strategy_history") or []),
        "fallback_split_strategy": parse_session_dump.get("fallback_split_strategy"),
        "workflow_mode": parse_session_dump.get("mode"),
    }


def _proposal_validation_reason(
    *,
    parsed: Any,
    current_layer_context: Any,
    parser_source_map: dict[str, dict[str, Any]],
) -> str | None:
    children = list(getattr(parsed, "children", []) or [])
    if not children:
        if getattr(parsed, "satisfied", None) is True:
            return None
        return "proposal returned no children without satisfied=true"
    current_parent_ids = set(getattr(current_layer_context, "parent_node_ids", []) or [])
    child_parent_ids = {str(getattr(child, "parent_node_id", "") or "") for child in children}
    if current_parent_ids and not child_parent_ids.issubset(current_parent_ids):
        return "proposal referenced parent ids outside the current layer"
    children_by_parent: dict[str, list[Any]] = {}
    for child in children:
        pointers = list(getattr(child, "total_content_pointers", []) or [])
        if not pointers:
            return "proposal child missing total_content_pointers"
        parent_id = str(getattr(child, "parent_node_id", "") or "")
        if not parent_id:
            return "proposal child missing parent_node_id"
        children_by_parent.setdefault(parent_id, []).append(child)
        for pointer in pointers:
            source_cluster_id = str(_pointer_field(pointer, "source_cluster_id") or "")
            start_char = _pointer_field(pointer, "start_char")
            end_char = _pointer_field(pointer, "end_char")
            if not source_cluster_id or source_cluster_id not in parser_source_map:
                return "proposal pointer referenced source outside the supplied source map"
            if not isinstance(start_char, int) or not isinstance(end_char, int):
                return "proposal pointer missing character span integers"
            if start_char < 0 or end_char < start_char:
                return "proposal pointer used an invalid character span"
    parent_titles = dict(
        zip(
            list(getattr(current_layer_context, "parent_node_ids", []) or []),
            list(getattr(current_layer_context, "parent_titles", []) or []),
            strict=False,
        )
    )
    for parent_id, parent_children in children_by_parent.items():
        if len(parent_children) == 1:
            _ = parent_titles.get(parent_id)
            return "proposal collapsed a parent into a single child without a real breakdown"
    return None


def _boundary_validation_reason(
    *,
    parsed: Any,
    current_layer_context: Any,
    parser_source_map: dict[str, dict[str, Any]],
) -> str | None:
    cutpoints = list(getattr(parsed, "cutpoints", []) or [])
    if not cutpoints:
        if getattr(parsed, "satisfied", None) is True:
            return None
        return "boundary proposal returned no cutpoints without satisfied=true"
    current_parent_ids = set(getattr(current_layer_context, "parent_node_ids", []) or [])
    parent_pointers = dict(getattr(current_layer_context, "parent_content_pointers_by_id", {}) or {})
    seen: set[tuple[str, str, int]] = set()
    previous_key: tuple[str, str, int] | None = None
    for cutpoint in cutpoints:
        parent_node_id = str(getattr(cutpoint, "parent_node_id", "") or "")
        source_cluster_id = str(getattr(cutpoint, "source_cluster_id", "") or "")
        cut_offset = getattr(cutpoint, "cut_offset", None)
        text_before_cut = str(getattr(cutpoint, "text_before_cut", "") or "")
        text_after_cut = str(getattr(cutpoint, "text_after_cut", "") or "")
        cut_reason = str(getattr(cutpoint, "cut_reason", "") or getattr(cutpoint, "reason", "") or "")
        current_key = (parent_node_id, source_cluster_id, int(cut_offset) if isinstance(cut_offset, int) else -1)
        if previous_key is not None and current_key < previous_key:
            return "boundary proposal cutpoints must be sorted by parent, source cluster, then cut offset"
        previous_key = current_key
        if not parent_node_id or parent_node_id not in current_parent_ids:
            return "boundary proposal referenced parent ids outside the current layer"
        if not source_cluster_id or source_cluster_id not in parser_source_map:
            return "boundary proposal referenced source outside the supplied source map"
        if not isinstance(cut_offset, int):
            return "boundary proposal missing integer cut_offset"
        if not text_before_cut.strip() or not text_after_cut.strip():
            return "boundary proposal missing text_before_cut or text_after_cut evidence"
        if not cut_reason.strip():
            return "boundary proposal missing cut_reason"
        pointer_candidates = list(parent_pointers.get(parent_node_id) or [])
        pointer = None
        for parent_pointer in pointer_candidates:
            if str(_pointer_field(parent_pointer, "source_cluster_id") or "") == source_cluster_id:
                pointer = parent_pointer
                break
        if pointer is None:
            return "boundary proposal referenced a source cluster not present in the current parent context"
        start_char, end_char_exclusive = _pointer_span_bounds(pointer, parser_source_map=parser_source_map)
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
    current_layer_context: Any,
    parser_source_map: dict[str, dict[str, Any]],
    target: BoundaryCutpoint,
) -> dict[str, Any]:
    parent_ids = list(getattr(current_layer_context, "parent_node_ids", []) or [])
    parent_pointers = dict(getattr(current_layer_context, "parent_content_pointers_by_id", {}) or {})
    target_parent_pointers = list(parent_pointers.get(target.parent_node_id) or [])
    target_pointer = None
    for pointer in target_parent_pointers:
        if str(_pointer_field(pointer, "source_cluster_id") or "") == target.source_cluster_id:
            target_pointer = pointer
            break
    target_excerpt = _pointer_text(target_pointer, parser_source_map=parser_source_map) if target_pointer is not None else ""
    legal_cutpoints: list[dict[str, Any]] = []
    if target_pointer is not None:
        start_char = int(_pointer_field(target_pointer, "start_char") or 0)
        for boundary in _legal_cutpoints_for_text(target_excerpt):
            legal_cutpoints.append(
                {
                    "cut_offset": start_char + boundary.cut_offset,
                    "boundary_kind": boundary.boundary_kind,
                    "confidence": boundary.confidence,
                    "reason": boundary.reason,
                }
            )
    return {
        "parent_node_id": target.parent_node_id,
        "source_cluster_id": target.source_cluster_id,
        "input_cut_offset": target.cut_offset,
        "text_before_cut": target.text_before_cut,
        "text_after_cut": target.text_after_cut,
        "cut_reason": target.cut_reason or target.reason,
        "parent_ids": parent_ids,
        "pointer_excerpt": _trim_multiline_text(target_excerpt, max_lines=4, max_chars=700),
        "legal_cutpoints": legal_cutpoints,
    }


def _boundary_decision_key(decision: BoundaryReviewDecision) -> tuple[str, str, int]:
    return (
        str(decision.parent_node_id),
        str(decision.source_cluster_id),
        int(decision.input_cut_offset if decision.input_cut_offset is not None else decision.cut_offset),
    )


def _boundary_review_decision(
    *,
    cutpoint: BoundaryCutpoint,
    current_layer_context: Any,
    parser_source_map: dict[str, dict[str, Any]],
) -> BoundaryReviewDecision:
    parent_pointers = dict(getattr(current_layer_context, "parent_content_pointers_by_id", {}) or {})
    pointer = None
    for parent_pointer in list(parent_pointers.get(cutpoint.parent_node_id) or []):
        if str(_pointer_field(parent_pointer, "source_cluster_id") or "") == cutpoint.source_cluster_id:
            pointer = parent_pointer
            break
    if pointer is None:
        return BoundaryReviewDecision(
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
    start_char = int(_pointer_field(pointer, "start_char") or 0)
    _start_char, end_char_exclusive = _pointer_span_bounds(pointer, parser_source_map=parser_source_map)
    local_cut = cutpoint.cut_offset - start_char
    resolution = _resolve_boundary_anchor(
        text=text,
        cut_offset=local_cut,
        text_before_cut=cutpoint.text_before_cut,
        text_after_cut=cutpoint.text_after_cut,
    )
    if resolution.resolved_cut_offset is None:
        unresolved_reason = resolution.reason or "boundary anchor could not be resolved"
        unresolved_decision = "needs_refinement" if "ambiguous" in unresolved_reason or "resolve" in unresolved_reason or "matched" in unresolved_reason else "reject"
        return BoundaryReviewDecision(
            parent_node_id=cutpoint.parent_node_id,
            source_cluster_id=cutpoint.source_cluster_id,
            input_cut_offset=cutpoint.cut_offset,
            cut_offset=cutpoint.cut_offset,
            decision=unresolved_decision,
            reason=unresolved_reason,
            anchor_match_mode=resolution.match_mode,
            anchor_match_score=resolution.match_score,
            text_before_cut=cutpoint.text_before_cut,
            text_after_cut=cutpoint.text_after_cut,
        )

    resolved_local_cut = resolution.resolved_cut_offset
    resolved_cut_offset = start_char + resolved_local_cut
    if resolved_local_cut <= 0 or resolved_local_cut >= len(text):
        return BoundaryReviewDecision(
            parent_node_id=cutpoint.parent_node_id,
            source_cluster_id=cutpoint.source_cluster_id,
            input_cut_offset=cutpoint.cut_offset,
            cut_offset=cutpoint.cut_offset,
            decision="reject",
            resolved_cut_offset=resolved_cut_offset,
            anchor_match_mode=resolution.match_mode,
            anchor_match_score=resolution.match_score,
            text_before_cut=cutpoint.text_before_cut,
            text_after_cut=cutpoint.text_after_cut,
            reason="resolved boundary fell outside the parent span",
        )

    shift_distance = abs(resolved_cut_offset - cutpoint.cut_offset)
    if shift_distance > MAX_BOUNDARY_REPAIR_SHIFT_CHARS:
        return BoundaryReviewDecision(
            parent_node_id=cutpoint.parent_node_id,
            source_cluster_id=cutpoint.source_cluster_id,
            input_cut_offset=cutpoint.cut_offset,
            cut_offset=cutpoint.cut_offset,
            decision="reject",
            resolved_cut_offset=resolved_cut_offset,
            anchor_match_mode=resolution.match_mode,
            anchor_match_score=resolution.match_score,
            text_before_cut=cutpoint.text_before_cut,
            text_after_cut=cutpoint.text_after_cut,
            reason=(
                f"boundary repair shift {shift_distance} chars exceeds "
                f"maximum {MAX_BOUNDARY_REPAIR_SHIFT_CHARS}"
            ),
        )

    legality_reason = _boundary_cutpoint_legality_reason(text, resolved_local_cut)
    if legality_reason:
        return BoundaryReviewDecision(
            parent_node_id=cutpoint.parent_node_id,
            source_cluster_id=cutpoint.source_cluster_id,
            input_cut_offset=cutpoint.cut_offset,
            cut_offset=cutpoint.cut_offset,
            decision="reject",
            resolved_cut_offset=resolved_cut_offset,
            anchor_match_mode=resolution.match_mode,
            anchor_match_score=resolution.match_score,
            text_before_cut=cutpoint.text_before_cut,
            text_after_cut=cutpoint.text_after_cut,
            reason=legality_reason,
        )

    boundary_kind = _classify_boundary_kind(text, resolved_local_cut)
    if boundary_kind == "semantic":
        boundary_kind = cutpoint.boundary_kind
    normalized_kind = boundary_kind if boundary_kind in {"section", "paragraph", "list_item", "sentence", "word"} else "semantic"
    if resolved_cut_offset == cutpoint.cut_offset:
        decision = "accept"
        reason = resolution.reason or "aligned with a unique anchored boundary"
    else:
        decision = "shift_left" if resolved_cut_offset < cutpoint.cut_offset else "shift_right"
        reason = resolution.reason or "shifted to a uniquely anchored boundary"
    return BoundaryReviewDecision(
        parent_node_id=cutpoint.parent_node_id,
        source_cluster_id=cutpoint.source_cluster_id,
        input_cut_offset=cutpoint.cut_offset,
        cut_offset=cutpoint.cut_offset,
        decision=decision,
        resolved_cut_offset=resolved_cut_offset,
        boundary_kind=normalized_kind,
        anchor_match_mode=resolution.match_mode,
        anchor_match_score=resolution.match_score,
        text_before_cut=cutpoint.text_before_cut,
        text_after_cut=cutpoint.text_after_cut,
        reason=reason,
    )


def _boundary_unit_summaries(
    *,
    current_layer_context: Any,
    parser_source_map: dict[str, dict[str, Any]],
    accepted_units: list[dict[str, Any]],
) -> list[BoundaryUnitSummary]:
    summaries: list[BoundaryUnitSummary] = []
    for unit in accepted_units:
        summaries.append(
            _make_boundary_summary(
                parent_node_id=str(unit["parent_node_id"]),
                source_cluster_id=str(unit["source_cluster_id"]),
                start_char=int(unit["start_char"]),
                end_char=int(unit["end_char"]),
                parser_source_map=parser_source_map,
                boundary_kind=str(unit.get("boundary_kind") or "semantic"),
            )
        )
    return summaries


def _boundary_parent_coverage_report(
    *,
    parent_node_id: str,
    source_cluster_id: str,
    start_char: int,
    end_char_exclusive: int,
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    covered_ranges = [
        {
            "start_char": int(segment["start_char"]),
            "end_char": int(segment["end_char"]),
        }
        for segment in segments
        if not bool(segment.get("skipped"))
    ]
    gap_ranges = [
        {
            "start_char": int(segment["start_char"]),
            "end_char": int(segment["end_char"]),
            "reason": str(segment.get("skip_reason") or "unresolved"),
        }
        for segment in segments
        if bool(segment.get("skipped"))
    ]
    covered_start = covered_ranges[0]["start_char"] if covered_ranges else start_char
    covered_end = covered_ranges[-1]["end_char"] if covered_ranges else start_char - 1
    coverage_state = "covered" if not gap_ranges and covered_start == start_char and covered_end >= end_char_exclusive - 1 else "covered_with_gaps"
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
    current_layer_context: Any,
    parser_source_map: dict[str, dict[str, Any]],
    review_batch: BoundaryReviewBatch,
) -> tuple[CurrentLayerResult, list[BoundaryUnitSummary], list[dict[str, Any]]]:
    accepted_cutpoints: list[dict[str, Any]] = []
    child_items: list[dict[str, Any]] = []
    coverage_reports: list[dict[str, Any]] = []
    unresolved_intervals = 0
    decisions_by_parent: dict[tuple[str, str], list[BoundaryReviewDecision]] = {}
    for decision in review_batch.decisions:
        key = (decision.parent_node_id, decision.source_cluster_id)
        decisions_by_parent.setdefault(key, []).append(decision)

    for parent_id in list(getattr(current_layer_context, "parent_node_ids", []) or []):
        parent_pointers = list(
            dict(getattr(current_layer_context, "parent_content_pointers_by_id", {}) or {}).get(parent_id) or []
        )
        pointer_index = 0
        for pointer in parent_pointers:
            source_cluster_id = str(_pointer_field(pointer, "source_cluster_id") or "")
            text = _pointer_text(pointer, parser_source_map=parser_source_map)
            start_char, end_char_exclusive = _pointer_span_bounds(pointer, parser_source_map=parser_source_map)
            absolute_offsets = [start_char]
            pointer_segments: list[dict[str, Any]] = []
            pointer_decisions = decisions_by_parent.get((parent_id, source_cluster_id), [])
            for decision in pointer_decisions:
                resolved_offset = decision.resolved_cut_offset if decision.resolved_cut_offset is not None else decision.cut_offset
                if resolved_offset is None:
                    unresolved_intervals += 1
                    continue
                if resolved_offset <= start_char or resolved_offset >= end_char_exclusive:
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
                zip(absolute_offsets, absolute_offsets[1:], strict=False),
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
                segment_text = text[(segment_start - start_char) : (segment_end - start_char)]
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
                    resolved_offset = decision.resolved_cut_offset if decision.resolved_cut_offset is not None else decision.cut_offset
                    if resolved_offset == segment_end and decision.boundary_kind is not None:
                        boundary_kind = decision.boundary_kind
                        break
                child = LayerChildCandidate(
                    node_id=child_node_id,
                    parent_node_id=parent_id,
                    title=_trim_text(segment_text.splitlines()[0] if segment_text.splitlines() else segment_text, max_chars=160),
                    node_type="TEXT_FLOW",
                    total_content_pointers=[
                        HydratedTextPointer(
                            source_cluster_id=source_cluster_id,
                            start_char=segment_start,
                            end_char=segment_end - 1,
                            verbatim_text=segment_text,
                        )
                    ],
                    expandable=bool(_legal_cutpoints_for_text(segment_text, max_points=4)),
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
                        "summary_text": _trim_multiline_text(segment_text, max_lines=2, max_chars=220),
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
        return CurrentLayerResult(
            children=[],
            satisfied=True,
            reasoning_history=[{"source": "boundary_first_deterministic_empty"}],
            metadata={"fallback": "boundary_empty", "allow_empty_layer": True},
        ), [], []

    children = [
        LayerChildCandidate(
            node_id=f"{item['parent_node_id']}|{item['source_cluster_id']}|unit-{index}",
            parent_node_id=str(item["parent_node_id"]),
            title=str(item["summary_text"] or item["exact_text"] or "Boundary Unit"),
            node_type="TEXT_FLOW",
            total_content_pointers=[
                HydratedTextPointer(
                    source_cluster_id=str(item["source_cluster_id"]),
                    start_char=int(item["start_char"]),
                    end_char=int(item["end_char"]),
                    verbatim_text=str(item["exact_text"] or item["summary_text"] or ""),
                )
            ],
            expandable=bool(item["expandable"]),
            metadata={
                "source": "boundary_first",
                "boundary_kind": item["boundary_kind"],
                "summary_text": item["summary_text"],
                "exact_text": item["exact_text"],
            },
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
            metadata={
                "proposal_mode": "boundaries",
                "boundary_unit_summaries": [summary.model_dump() for summary in summaries],
                "boundary_parent_coverage": coverage_reports,
                "boundary_review_notes": list(review_batch.review_notes),
                "unresolved_interval_count": unresolved_intervals,
            },
        )
    return result, summaries, accepted_cutpoints


def _annotate_proposal_result(
    result: CurrentLayerResult,
    *,
    proposal_source: str,
    proposal_mode: str = "children",
    proposal_failure_reason: str | None = None,
    provider_child_count: int | None = None,
    boundary_count: int | None = None,
    accepted_boundary_count: int | None = None,
    shifted_boundary_count: int | None = None,
    rejected_boundary_count: int | None = None,
    refinement_count: int | None = None,
    unresolved_interval_count: int | None = None,
    summary_count: int | None = None,
) -> CurrentLayerResult:
    metadata = dict(getattr(result, "metadata", {}) or {})
    metadata["proposal_source"] = proposal_source
    metadata["proposal_mode"] = proposal_mode
    if proposal_failure_reason:
        metadata["proposal_failure_reason"] = proposal_failure_reason
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
    reasoning_history = list(getattr(result, "reasoning_history", []) or [])
    marker_payload: dict[str, Any] = {
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
    reasoning_history.append(LayerReasoningEntry.model_validate(marker_payload))
    payload = result.model_dump()
    payload["metadata"] = metadata
    payload["reasoning_history"] = [entry.model_dump() if hasattr(entry, "model_dump") else entry for entry in reasoning_history]
    return result.__class__.model_validate(payload)


def _structured_invoke(
    model: SupportsStructuredOutput,
    schema: type[TStructuredModel],
    messages: Sequence[tuple[str, str]],
) -> TStructuredModel:
    structured = build_structured_output_runnable(model, schema, include_raw=True)
    response = structured.invoke(list(messages))
    if isinstance(response, dict):
        parsed = response.get("parsed")
        if parsed is not None:
            if isinstance(parsed, schema):
                return parsed
            if hasattr(parsed, "model_dump"):
                return schema.model_validate(parsed.model_dump())
            return schema.model_validate(parsed)
        parsing_error = response.get("parsing_error")
        if parsing_error is not None:
            raise ValueError(str(parsing_error))
        return schema.model_validate(response)
    if isinstance(response, schema):
        return response
    if hasattr(response, "model_dump"):
        return schema.model_validate(response.model_dump())
    return schema.model_validate(response)


def _fallback_layer_result(
    *,
    current_layer_context: Any,
    parser_source_map: dict[str, dict[str, Any]],
) -> CurrentLayerResult:
    children: list[LayerChildCandidate] = []
    if int(getattr(current_layer_context, "depth", 0)) > 0:
        return CurrentLayerResult(
            children=[],
            satisfied=True,
            reasoning_history=[{"source": "deterministic_depth_stop"}],
            metadata={"fallback": "depth_stop", "allow_empty_layer": True},
        )
    parent_ids = list(getattr(current_layer_context, "parent_node_ids", []) or [])
    parent_id = parent_ids[0] if parent_ids else "root"
    records = list(parser_source_map.items())
    for index, (source_cluster_id, record) in enumerate(records[:8], start=1):
        text = str(record.get("text") or "").strip()
        if not text:
            continue
        title_match = re.search(r"(?m)^#{1,3}\s+(.+)$", text)
        title = title_match.group(1).strip() if title_match else f"Section {index}"
        excerpt = text
        children.append(
            LayerChildCandidate(
                node_id=f"{parent_id}|section-{index}",
                parent_node_id=parent_id,
                title=title[:160],
                node_type="TEXT_FLOW",
                total_content_pointers=[
                    HydratedTextPointer(
                        source_cluster_id=str(source_cluster_id),
                        start_char=0,
                        end_char=max(len(excerpt) - 1, 0),
                        verbatim_text=excerpt or " ",
                    )
                ],
                expandable=False,
                metadata={"source": "workflow_layered_fallback"},
            )
        )
    return CurrentLayerResult(
        children=children,
        satisfied=True,
        reasoning_history=[{"source": "deterministic_fallback"}],
        metadata={"fallback": "llm_empty_or_unavailable", "allow_empty_layer": False},
    )


def build_layerwise_llm_callbacks(
    provider_settings: WorkflowProviderSettings,
    *,
    event_sink: Callable[..., None] | None = None,
    fallback_layer_result_fn: Callable[..., CurrentLayerResult] | None = None,
    max_depth: int = 2,
    allow_review: bool = True,
    proposal_mode: str | None = None,
    boundary_refinement_rounds: int = 1,
) -> LayerwiseLLMCallbacks:
    chat_model = build_chat_model_for_role("parser", provider_settings)
    fallback_builder = fallback_layer_result_fn or _fallback_layer_result
    proposal_mode = str(proposal_mode or getattr(provider_settings, "proposal_mode", "children") or "children")
    if proposal_mode not in {"children", "boundaries"}:
        raise ValueError("proposal_mode must be either 'children' or 'boundaries'")
    boundary_refinement_rounds = max(0, int(boundary_refinement_rounds or 0))
    proposal_retry_rounds = max(0, int(getattr(provider_settings.parser, "max_retries", 0) or 0))

    def _emit(stage: str, **extra: Any) -> None:
        if callable(event_sink):
            event_sink(stage, **extra)

    def _proposal_attempt_payload(
        payload: dict[str, Any],
        *,
        attempt_index: int,
        prior_error: str | None,
    ) -> dict[str, Any]:
        payload = dict(payload)
        payload["proposal_attempt"] = attempt_index + 1
        payload["proposal_retry_rounds"] = proposal_retry_rounds
        if prior_error:
            payload["previous_attempt_error"] = prior_error
        return payload

    def _propose_layer_fn(
        *,
        parser_source_map,
        current_layer_context,
        semantic_tree,
        split_strategy,
        parser_input_dict,
        parse_session,
        **kwargs,
    ) -> CurrentLayerResult:
        if proposal_mode == "boundaries":
            boundary_candidates = _boundary_prompt_candidate_context(
                current_layer_context=current_layer_context,
                parser_source_map=parser_source_map,
            )
            def _build_boundary_messages(attempt_number: int, previous_error: str | None) -> list[tuple[str, str]]:
                prompt_payload = _proposal_attempt_payload(
                    {
                        "task": "Propose semantic cutpoints for the next layer. Return only boundary cutpoints, not child text.",
                        "output_contract": "Return an LLMBoundaryProposalBatch only.",
                        "proposal_mode": proposal_mode,
                        "split_strategy": split_strategy,
                        "attempt_context": _proposal_attempt_context(
                            parse_session=parse_session,
                            current_layer_context=current_layer_context,
                        ),
                        "current_layer_context": _dump_model(current_layer_context),
                        "parent_nodes": _parent_context_excerpt(
                            current_layer_context=current_layer_context,
                            parser_source_map=parser_source_map,
                        ),
                        "boundary_candidates": boundary_candidates,
                        "semantic_tree_snapshot": _summarize_for_prompt(
                            semantic_tree,
                            max_depth=4,
                            max_items=5,
                            max_string=280,
                        ),
                        "full_document_context": _summarize_for_prompt(
                            parser_input_dict,
                            max_depth=4,
                            max_items=5,
                            max_string=280,
                        ),
                        "source_map_excerpt": _source_map_excerpt(parser_source_map),
                        "rules": [
                            "Operate on ONE layer only: propose cutpoints for the current parent nodes.",
                            "Only choose cutpoints from the provided candidate lists when possible.",
                            "Do not emit child text, summaries, or prose.",
                            "For each cutpoint, emit cut_reason plus short exact text_before_cut and text_after_cut anchors.",
                            "The anchors must uniquely identify the cut within the parent text and must not be the full child verbatim.",
                            "Cutpoints must be parent-scoped and source-cluster scoped.",
                            "Prefer word, sentence, paragraph, list-item, or section boundaries over semantic guesses.",
                            "Keep the proposal sorted and unique per parent/source cluster.",
                            "If a parent is atomic, return no internal cutpoints for that parent and set satisfied=true only when the current layer is complete.",
                        ],
                    },
                    attempt_index=attempt_number - 1,
                    prior_error=previous_error,
                )
                return [
                    (
                        "system",
                    "You are revising ONE semantic layer in an iterative document parsing workflow. "
                    "Return only structured data matching LLMBoundaryProposalBatch. "
                    "Propose grounded cutpoints only; the host will review and assemble the children. "
                    "Each cutpoint must include cut_reason plus exact text_before_cut and text_after_cut anchors.",
                ),
                    ("human", json.dumps(prompt_payload, sort_keys=True)),
                ]

            def _emit_boundary_retry(record: Any) -> None:
                _emit(
                    "workflow_layered_proposal_retry",
                    proposal_source="llm",
                    proposal_mode="boundaries",
                    depth=int(getattr(current_layer_context, "depth", 0)),
                    retry_count=int(getattr(current_layer_context, "retry_count", 0)),
                    split_strategy=split_strategy,
                    attempt=record.attempt_number,
                    retry_budget=proposal_retry_rounds,
                    retry_reason=record.error_message,
                )

            try:
                boundary_result: RetryResult[LLMBoundaryProposalBatch] = retry_with_context(
                    max_attempts=proposal_retry_rounds + 1,
                    build_request=_build_boundary_messages,
                    invoke=lambda messages: _structured_invoke(chat_model, LLMBoundaryProposalBatch, messages),
                    validate=lambda parsed: _boundary_validation_reason(
                        parsed=parsed,
                        current_layer_context=current_layer_context,
                        parser_source_map=parser_source_map,
                    ),
                    on_retry=_emit_boundary_retry,
                )
                boundary_parsed: LLMBoundaryProposalBatch = boundary_result.value
            except RetryExhaustedError as exc:
                failure_reason = exc.last_error
                fallback = fallback_builder(
                    current_layer_context=current_layer_context,
                    parser_source_map=parser_source_map,
                )
                annotated = _annotate_proposal_result(
                    fallback,
                    proposal_source="fallback",
                    proposal_mode="boundaries",
                    proposal_failure_reason=failure_reason,
                    provider_child_count=0,
                )
                _emit(
                    "workflow_layered_proposal_result",
                    proposal_source="fallback",
                    proposal_mode="boundaries",
                    proposal_failure_reason=failure_reason,
                    depth=int(getattr(current_layer_context, "depth", 0)),
                    retry_count=int(getattr(current_layer_context, "retry_count", 0)),
                    split_strategy=split_strategy,
                    child_count=len(annotated.children),
                    satisfied=annotated.satisfied,
                )
                return annotated

            review_decisions: list[BoundaryReviewDecision] = [
                _boundary_review_decision(
                    cutpoint=cutpoint,
                    current_layer_context=current_layer_context,
                    parser_source_map=parser_source_map,
                )
                for cutpoint in boundary_parsed.cutpoints
            ]
            refinement_notes: list[str] = []
            refinement_attempts = 0
            if boundary_refinement_rounds > 0:
                unresolved_targets = _boundary_refinement_targets(
                    parsed=boundary_parsed,
                    review_decisions=review_decisions,
                )
                for target in unresolved_targets[:boundary_refinement_rounds]:
                    refinement_attempts += 1
                    refinement_prompt_payload = {
                        "task": "Refine a single ambiguous cutpoint using only nearby legal boundaries.",
                        "output_contract": "Return an LLMBoundaryProposalBatch only.",
                        "proposal_mode": proposal_mode,
                        "split_strategy": split_strategy,
                        "attempt_context": _proposal_attempt_context(
                            parse_session=parse_session,
                            current_layer_context=current_layer_context,
                        ),
                        "current_layer_context": _dump_model(current_layer_context),
                        "refinement_target": _boundary_refinement_prompt_context(
                            current_layer_context=current_layer_context,
                            parser_source_map=parser_source_map,
                            target=target,
                        ),
                        "rules": [
                            "Operate on one ambiguous cutpoint only.",
                            "Choose a nearby legal cutpoint or return no cutpoints with satisfied=true.",
                            "Do not emit prose or child text.",
                            "Preserve the original parent and source cluster scope.",
                        ],
                    }
                    refinement_messages = [
                        (
                            "system",
                            "You are refining one ambiguous semantic cutpoint in a layered document parser. "
                            "Return only structured data matching LLMBoundaryProposalBatch.",
                        ),
                        ("human", json.dumps(refinement_prompt_payload, sort_keys=True)),
                    ]
                    try:
                        refinement_result: LLMBoundaryProposalBatch = _structured_invoke(
                            chat_model,
                            LLMBoundaryProposalBatch,
                            refinement_messages,
                        )
                        refinement_parsed: LLMBoundaryProposalBatch = refinement_result
                        refinement_validation_reason = _boundary_validation_reason(
                            parsed=refinement_parsed,
                            current_layer_context=current_layer_context,
                            parser_source_map=parser_source_map,
                        )
                        if refinement_validation_reason:
                            raise ValueError(refinement_validation_reason)
                        refinement_decisions = [
                            _boundary_review_decision(
                                cutpoint=cutpoint,
                                current_layer_context=current_layer_context,
                                parser_source_map=parser_source_map,
                            )
                            for cutpoint in refinement_parsed.cutpoints
                        ]
                        if refinement_decisions:
                            replacement_map = {
                                _boundary_decision_key(decision): decision for decision in review_decisions
                            }
                            for decision in refinement_decisions:
                                if decision.decision not in {"accept", "shift_left", "shift_right"}:
                                    continue
                                replacement_map[_boundary_decision_key(decision)] = decision
                            review_decisions = sorted(replacement_map.values(), key=_boundary_decision_key)
                            refinement_notes.extend(
                                note
                                for note in (
                                    decision.reason for decision in refinement_decisions if decision.reason
                                )
                                if note not in refinement_notes
                            )
                    except Exception as refinement_exc:
                        refinement_notes.append(
                            f"boundary refinement skipped for {target.parent_node_id}:{target.source_cluster_id}:{target.cut_offset} "
                            f"due to {refinement_exc!r}"
                        )
            review_batch: BoundaryReviewBatch = BoundaryReviewBatch(
                decisions=review_decisions,
                satisfied=boundary_parsed.satisfied,
                coverage_ok=not any(
                    decision.decision in {"reject", "needs_refinement"}
                    for decision in review_decisions
                ),
                review_notes=[
                    decision.reason
                    for decision in review_decisions
                    if decision.reason
                ]
                + refinement_notes,
            )
            if boundary_parsed.satisfied is True and not boundary_parsed.cutpoints:
                runtime_result = CurrentLayerResult(
                    children=[],
                    satisfied=True,
                    reasoning_history=list(boundary_parsed.reasoning_history),
                    metadata={
                        "proposal_mode": "boundaries",
                        "proposal_source": "llm",
                        "proposal_retry_count": boundary_result.retry_count,
                        "boundary_proposed_count": 0,
                        "boundary_accepted_count": 0,
                        "boundary_shifted_count": 0,
                        "boundary_rejected_count": 0,
                        "boundary_refinement_count": 0,
                        "boundary_refinement_attempts": refinement_attempts,
                        "boundary_summary_count": 0,
                        "boundary_cutpoints": [],
                        "boundary_review_decisions": [],
                        "boundary_review_notes": list(review_batch.review_notes),
                        "boundary_atomic_decision": True,
                    },
                )
                annotated = _annotate_proposal_result(
                    runtime_result,
                    proposal_source="llm",
                    proposal_mode="boundaries",
                    boundary_count=0,
                    accepted_boundary_count=0,
                    shifted_boundary_count=0,
                    rejected_boundary_count=0,
                    refinement_count=0,
                    unresolved_interval_count=0,
                    summary_count=0,
                )
                _emit(
                    "workflow_layered_proposal_result",
                    proposal_source="llm",
                    proposal_mode="boundaries",
                    depth=int(getattr(current_layer_context, "depth", 0)),
                    retry_count=int(getattr(current_layer_context, "retry_count", 0)),
                    split_strategy=split_strategy,
                    boundary_count=0,
                    accepted_boundary_count=0,
                    shifted_boundary_count=0,
                    rejected_boundary_count=0,
                    refinement_count=0,
                    unresolved_interval_count=0,
                    child_count=0,
                    satisfied=True,
                )
                return annotated
            runtime_result: CurrentLayerResult
            accepted_cutpoints: list[dict[str, Any]]
            runtime_result, summaries, accepted_cutpoints = _assemble_layer_result_from_boundaries(
                current_layer_context=current_layer_context,
                parser_source_map=parser_source_map,
                review_batch=review_batch,
            )
            accepted_count = sum(1 for decision in review_decisions if decision.decision == "accept")
            shifted_count = sum(1 for decision in review_decisions if decision.decision in {"shift_left", "shift_right"})
            rejected_count = sum(1 for decision in review_decisions if decision.decision == "reject")
            refinement_count = sum(1 for decision in review_decisions if decision.decision == "needs_refinement")
            unresolved_interval_count = int(runtime_result.metadata.get("unresolved_interval_count", 0) or 0)
            if accepted_count + shifted_count == 0:
                failure_reason = "boundary proposal produced no accepted cutpoints"
                fallback = fallback_builder(
                    current_layer_context=current_layer_context,
                    parser_source_map=parser_source_map,
                )
                annotated = _annotate_proposal_result(
                    fallback,
                    proposal_source="fallback",
                    proposal_mode="boundaries",
                    proposal_failure_reason=failure_reason,
                    provider_child_count=0,
                )
                _emit(
                    "workflow_layered_proposal_result",
                    proposal_source="fallback",
                    proposal_mode="boundaries",
                    proposal_failure_reason=failure_reason,
                    depth=int(getattr(current_layer_context, "depth", 0)),
                    retry_count=int(getattr(current_layer_context, "retry_count", 0)),
                    split_strategy=split_strategy,
                    child_count=len(annotated.children),
                    satisfied=annotated.satisfied,
                )
                return annotated
            result_metadata = {
                **runtime_result.metadata,
                "proposal_mode": "boundaries",
                "proposal_retry_count": boundary_result.retry_count,
                "boundary_proposed_count": len(boundary_parsed.cutpoints),
                "boundary_accepted_count": accepted_count,
                "boundary_shifted_count": shifted_count,
                "boundary_rejected_count": rejected_count,
                "boundary_refinement_count": refinement_count,
                "boundary_refinement_attempts": refinement_attempts,
                "boundary_summary_count": len(summaries),
                "boundary_cutpoints": [cutpoint.model_dump() for cutpoint in boundary_parsed.cutpoints],
                "boundary_review_decisions": [decision.model_dump() for decision in review_decisions],
                "boundary_review_notes": list(review_batch.review_notes),
                "proposal_source": "llm",
            }
            runtime_result = runtime_result.model_copy(update={"metadata": result_metadata})
            annotated = _annotate_proposal_result(
                runtime_result,
                proposal_source="llm",
                proposal_mode="boundaries",
                boundary_count=len(boundary_parsed.cutpoints),
                accepted_boundary_count=accepted_count,
                shifted_boundary_count=shifted_count,
                rejected_boundary_count=rejected_count,
                refinement_count=refinement_count,
                unresolved_interval_count=unresolved_interval_count,
                summary_count=len(summaries),
            )
            _emit(
                "workflow_layered_proposal_result",
                proposal_source="llm",
                proposal_mode="boundaries",
                depth=int(getattr(current_layer_context, "depth", 0)),
                retry_count=int(getattr(current_layer_context, "retry_count", 0)),
                split_strategy=split_strategy,
                boundary_count=len(boundary_parsed.cutpoints),
                accepted_boundary_count=accepted_count,
                shifted_boundary_count=shifted_count,
                rejected_boundary_count=rejected_count,
                refinement_count=refinement_count,
                unresolved_interval_count=unresolved_interval_count,
                child_count=len(runtime_result.children),
                satisfied=runtime_result.satisfied,
            )
            return annotated

        def _build_child_messages(attempt_number: int, previous_error: str | None) -> list[tuple[str, str]]:
            prompt_payload = _proposal_attempt_payload(
                {
                    "task": "Propose the next semantic layer for the current parent nodes.",
                    "output_contract": "Return a CurrentLayerResult only. Do not emit prose.",
                    "split_strategy": split_strategy,
                    "attempt_context": _proposal_attempt_context(
                        parse_session=parse_session,
                        current_layer_context=current_layer_context,
                    ),
                    "current_layer_context": _dump_model(current_layer_context),
                    "parent_nodes": _parent_context_excerpt(
                        current_layer_context=current_layer_context,
                        parser_source_map=parser_source_map,
                    ),
                    "semantic_tree_snapshot": _summarize_for_prompt(
                        semantic_tree,
                        max_depth=4,
                        max_items=5,
                        max_string=280,
                    ),
                    "full_document_context": _summarize_for_prompt(
                        parser_input_dict,
                        max_depth=4,
                        max_items=5,
                        max_string=280,
                    ),
                    "source_map_excerpt": _source_map_excerpt(parser_source_map),
                    "rules": [
                        "Operate on ONE layer only: propose only the immediate children for the current parent nodes.",
                        "Do not edit ancestors or descendants and do not break down current children directly.",
                        "Children together must preserve the parent meaning collectively.",
                        "Each parent should end up with more than one child or no child at all.",
                        "If a parent is already atomic, return no children for that parent and use satisfied=true only when the current layer is complete.",
                        "Prefer a coarser layerwise breakdown over premature deep flattening so later iterations can refine substructure.",
                        "Do not recombine separated verbatim fragments into invented longer text; keep grounding exact and local.",
                        "Every pointer must use exact source_cluster_id values plus character spans from the supplied source map.",
                        "Set expandable=false for leaf nodes and expandable=true only when later subdivision is still appropriate.",
                    ],
                },
                attempt_index=attempt_number - 1,
                prior_error=previous_error,
            )
            return [
                (
                    "system",
                    "You are revising ONE semantic layer in an iterative document parsing workflow. "
                    "Return only structured data matching CurrentLayerResult. "
                    "Produce grounded immediate children for the supplied parents and preserve layerwise semantics.",
                ),
                ("human", json.dumps(prompt_payload, sort_keys=True)),
            ]

        def _emit_child_retry(record: Any) -> None:
            _emit(
                "workflow_layered_proposal_retry",
                proposal_source="llm",
                proposal_mode="children",
                depth=int(getattr(current_layer_context, "depth", 0)),
                retry_count=int(getattr(current_layer_context, "retry_count", 0)),
                split_strategy=split_strategy,
                attempt=record.attempt_number,
                retry_budget=proposal_retry_rounds,
                retry_reason=record.error_message,
            )

        try:
            child_result: RetryResult[LLMCurrentLayerResult] = retry_with_context(
                max_attempts=proposal_retry_rounds + 1,
                build_request=_build_child_messages,
                invoke=lambda messages: _structured_invoke(chat_model, LLMCurrentLayerResult, messages),
                validate=lambda parsed: _proposal_validation_reason(
                    parsed=parsed,
                    current_layer_context=current_layer_context,
                    parser_source_map=parser_source_map,
                ),
                on_retry=_emit_child_retry,
            )
            child_parsed: LLMCurrentLayerResult = child_result.value
        except RetryExhaustedError as exc:
            failure_reason = exc.last_error
            fallback = fallback_builder(
                current_layer_context=current_layer_context,
                parser_source_map=parser_source_map,
            )
            annotated = _annotate_proposal_result(
                fallback,
                proposal_source="fallback",
                proposal_mode="children",
                proposal_failure_reason=failure_reason,
                provider_child_count=0,
            )
            _emit(
                "workflow_layered_proposal_result",
                proposal_source="fallback",
                proposal_mode="children",
                proposal_failure_reason=failure_reason,
                depth=int(getattr(current_layer_context, "depth", 0)),
                retry_count=int(getattr(current_layer_context, "retry_count", 0)),
                split_strategy=split_strategy,
                child_count=len(annotated.children),
                satisfied=annotated.satisfied,
            )
            return annotated

        runtime_result: CurrentLayerResult = CurrentLayerResult.model_validate(child_parsed.model_dump())
        runtime_result = runtime_result.model_copy(
            update={"metadata": {**runtime_result.metadata, "proposal_retry_count": child_result.retry_count}}
        )
        annotated = _annotate_proposal_result(
            runtime_result,
            proposal_source="llm",
            proposal_mode="children",
            provider_child_count=len(runtime_result.children),
        )
        _emit(
            "workflow_layered_proposal_result",
            proposal_source="llm",
            proposal_mode="children",
            depth=int(getattr(current_layer_context, "depth", 0)),
            retry_count=int(getattr(current_layer_context, "retry_count", 0)),
            split_strategy=split_strategy,
            child_count=len(runtime_result.children),
            satisfied=runtime_result.satisfied,
        )
        return annotated

    def _review_layer_fn(
        *,
        current_layer_context,
        current_layer_result,
        split_strategy,
        parser_source_map=None,
        parse_session=None,
        **kwargs,
    ) -> CurrentLayerReview:
        review_payload = {
            "task": "Review the current semantic layer proposal for layerwise correctness and source-grounded coverage.",
            "split_strategy": split_strategy,
            "attempt_context": _proposal_attempt_context(
                parse_session=parse_session,
                current_layer_context=current_layer_context,
            ),
            "current_layer_context": _dump_model(current_layer_context),
            "current_layer_result": _summarize_for_prompt(
                current_layer_result,
                max_depth=4,
                max_items=6,
                max_string=280,
            ),
            "parent_nodes": _parent_context_excerpt(
                current_layer_context=current_layer_context,
                parser_source_map=parser_source_map or {},
            ),
            "source_map_excerpt": _source_map_excerpt(parser_source_map or {}),
            "review_rules": [
                "Review only the immediate children for this layer.",
                "Children should preserve each parent meaning collectively without inventing missing text.",
                "Coverage gaps, overlap conflicts, and duplicates should be called out explicitly.",
                "Prefer coarser layerwise structure over deep flattening when both are source-grounded.",
                "If the layer is already in a good layerwise sweet spot, satisfied=true is appropriate.",
            ],
        }
        messages = [
            (
                "system",
                "You review ONE semantic layer in an iterative document parsing workflow. "
                "Return only structured data matching CurrentLayerReview.",
            ),
            (
                "human",
                json.dumps(review_payload, sort_keys=True),
            ),
        ]
        try:
            review_result: LLMCurrentLayerReview = _structured_invoke(chat_model, LLMCurrentLayerReview, messages)
            reviewed: LLMCurrentLayerReview = review_result
            runtime_review: CurrentLayerReview = CurrentLayerReview.model_validate(reviewed.model_dump())
            _emit(
                "workflow_layered_review_result",
                review_source="llm",
                depth=int(getattr(current_layer_context, "depth", 0)),
                retry_count=int(getattr(current_layer_context, "retry_count", 0)),
                split_strategy=split_strategy,
                satisfied=runtime_review.satisfied,
                coverage_ok=runtime_review.coverage_ok,
            )
            return runtime_review
        except Exception as exc:
            failure_reason = _trim_text(repr(exc), max_chars=500)
            reviewed = CurrentLayerReview(
                updated_result=current_layer_result,
                coverage_ok=True,
                satisfied=True,
                strategy_used=split_strategy,
                review_notes=["deterministic review fallback after provider failure"],
            )
            _emit(
                "workflow_layered_review_result",
                review_source="fallback",
                review_failure_reason=failure_reason,
                depth=int(getattr(current_layer_context, "depth", 0)),
                retry_count=int(getattr(current_layer_context, "retry_count", 0)),
                split_strategy=split_strategy,
                satisfied=reviewed.satisfied,
                coverage_ok=reviewed.coverage_ok,
            )
            return reviewed

    return {
        "propose_layer_fn": _propose_layer_fn,
        "review_layer_fn": _review_layer_fn,
        "max_depth": max_depth,
        "allow_review": allow_review,
    }
