"""Deterministic legal cutpoint generation and source-anchor repair."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from kogwistar.fuzzy_offsets import find_fuzzy_spans, offset_repair_threshold

from .layerwise_source_spans import (
    _offset_value,
    _pointer_field,
    _pointer_source_cluster_id,
    _pointer_text,
)
from .models import BoundaryCutpoint, CurrentLayerContext
from .parser_core import ParserSourceMap

MAX_BOUNDARY_REPAIR_SHIFT_CHARS = 8
BoundaryKind = Literal[
    "section", "paragraph", "list_item", "table_row", "sentence", "word", "semantic"
]
AnchorMatchMode = Literal["exact", "fuzzy"]


@dataclass(frozen=True, slots=True)
class BoundaryAnchorResolution:
    resolved_cut_offset: int | None
    match_mode: AnchorMatchMode | None
    match_score: float | None
    reason: str | None


def is_heading_line(line: str) -> bool:
    return bool(re.match(r"^#{1,6}\s+\S", line.lstrip()))


def is_list_item_line(line: str) -> bool:
    return bool(re.match(r"^([-*+]\s+|\d+[.)]\s+|[a-zA-Z][.)]\s+)", line.lstrip()))


def is_table_row_line(line: str) -> bool:
    stripped = line.strip()
    if "|" not in stripped:
        return False
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    return len(cells) >= 2 and any(cells)


def classify_boundary_kind(text: str, offset: int) -> str:
    if offset <= 0 or offset >= len(text):
        return "semantic"
    prev_char = text[offset - 1]
    next_char = text[offset]
    line_end = text.find("\n", offset)
    if line_end == -1:
        line_end = len(text)
    next_line = text[offset:line_end]
    if text[max(0, offset - 2) : offset] == "\n\n" and is_heading_line(next_line):
        return "section"
    if text[max(0, offset - 2) : offset] == "\n\n":
        return "paragraph"
    line_start = text.rfind("\n", 0, offset - 1) + 1
    current_line = text[line_start:offset]
    if prev_char == "\n":
        previous_line_end = offset - 1
        previous_line_start = text.rfind("\n", 0, max(0, previous_line_end - 1)) + 1
        previous_line = text[previous_line_start:previous_line_end]
        next_line_end = text.find("\n", offset)
        if next_line_end == -1:
            next_line_end = len(text)
        next_line = text[offset:next_line_end]
        if is_table_row_line(previous_line) and is_table_row_line(next_line):
            return "table_row"
    if prev_char == "\n" and is_heading_line(current_line):
        return "section"
    if prev_char == "\n" and is_list_item_line(current_line):
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
        "table_row": 3,
        "sentence": 2,
        "word": 1,
        "semantic": 0,
    }.get(kind, 0)


def boundary_cutpoint_legality_reason(text: str, offset: int) -> str | None:
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


def legal_cutpoints_for_text(
    text: str,
    *,
    max_points: int = 48,
    include_word_boundaries: bool = False,
) -> list[BoundaryCutpoint]:
    if not text:
        return []
    best: dict[int, str] = {}
    for offset in range(1, len(text)):
        kind = classify_boundary_kind(text, offset)
        if kind == "semantic":
            continue
        if kind == "word" and not include_word_boundaries:
            continue
        if boundary_cutpoint_legality_reason(text, offset):
            continue
        existing = best.get(offset)
        if existing is None or _boundary_kind_priority(kind) > _boundary_kind_priority(existing):
            best[offset] = kind
    ordered = [
        BoundaryCutpoint(
            candidate_id=f"b{index:04d}-{kind}-{offset}",
            parent_node_id="",
            source_cluster_id="",
            cut_offset=offset,
            boundary_kind=cast(BoundaryKind, kind),
            text_before_cut=text[max(0, offset - 80) : offset],
            text_after_cut=text[offset : offset + 80],
            cut_reason="deterministic legal boundary",
            confidence=1.0,
            reason="deterministic_legal_boundary",
        )
        for index, (offset, kind) in enumerate(sorted(best.items()))
    ]
    if len(ordered) <= max_points:
        return ordered
    selected: list[BoundaryCutpoint] = []
    for kind in ("section", "paragraph", "list_item", "table_row", "sentence", "word"):
        kind_boundaries = [boundary for boundary in ordered if boundary.boundary_kind == kind]
        room = max_points - len(selected)
        if room <= 0:
            break
        selected.extend(kind_boundaries[:room])
    return sorted(selected, key=lambda boundary: boundary.cut_offset)


def boundary_candidates_for_pointer(
    pointer: object,
    *,
    parser_source_map: Mapping[str, Mapping[str, object]],
    max_points: int = 32,
) -> list[BoundaryCutpoint]:
    text = _pointer_text(pointer, parser_source_map=parser_source_map)
    source_cluster_id = _pointer_source_cluster_id(pointer)
    candidates = legal_cutpoints_for_text(text, max_points=max_points)
    if not candidates:
        return []
    start_char = _offset_value(_pointer_field(pointer, "start_char"), 0)
    return [
        candidate.model_copy(
            update={
                "parent_node_id": "",
                "source_cluster_id": source_cluster_id,
                "cut_offset": start_char + candidate.cut_offset,
            }
        )
        for candidate in candidates
    ]


def boundary_anchor_excerpt(
    text: str, cut_offset: int, *, window_chars: int = 20
) -> tuple[str, str]:
    start = max(0, cut_offset - window_chars)
    end = min(len(text), cut_offset + window_chars)
    return text[start:cut_offset], text[cut_offset:end]


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


def resolve_boundary_anchor(
    *,
    text: str,
    cut_offset: int,
    text_before_cut: str,
    text_after_cut: str,
) -> BoundaryAnchorResolution:
    before = str(text_before_cut or "")
    after = str(text_after_cut or "")
    if not before or not after:
        return BoundaryAnchorResolution(None, None, None, "boundary proposal missing text_before_cut or text_after_cut")
    anchor = before + after
    exact_positions = _boundary_anchor_occurrences(text, anchor)
    if len(exact_positions) == 1:
        return BoundaryAnchorResolution(
            exact_positions[0] + len(before), "exact", 1.0, "unique exact boundary anchor match"
        )
    if len(exact_positions) > 1:
        return BoundaryAnchorResolution(None, None, None, "boundary anchor is ambiguous across the source text")

    origin_start = max(0, cut_offset - len(before))
    fuzzy_hits = find_fuzzy_spans(
        content=text,
        excerpt=anchor,
        origin_start=origin_start,
        scan_band=max(50, len(anchor) * 4),
        max_hits=3,
    )
    if not fuzzy_hits:
        return BoundaryAnchorResolution(None, None, None, "boundary anchor could not be resolved against the source text")
    if len(fuzzy_hits) > 1:
        return BoundaryAnchorResolution(None, None, None, "boundary anchor is ambiguous after fuzzy matching")
    hit = fuzzy_hits[0]
    threshold = offset_repair_threshold(len(anchor))
    if hit.score < threshold:
        return BoundaryAnchorResolution(
            None,
            None,
            hit.score,
            f"boundary anchor fuzzy score {hit.score:.2f} below threshold {threshold:.2f}",
        )
    return BoundaryAnchorResolution(
        hit.start + len(before),
        "fuzzy",
        hit.score,
        f"boundary anchor repaired fuzzily with score {hit.score:.2f}",
    )


def repair_boundary_cutpoint_from_source(
    cutpoint: BoundaryCutpoint,
    *,
    current_layer_context: CurrentLayerContext,
    parser_source_map: ParserSourceMap,
) -> BoundaryCutpoint | None:
    parent_pointers = dict(current_layer_context.parent_content_pointers_by_id)
    pointer = next(
        (
            item
            for item in list(parent_pointers.get(cutpoint.parent_node_id) or [])
            if _pointer_source_cluster_id(item) == cutpoint.source_cluster_id
        ),
        None,
    )
    if pointer is None:
        return None
    text = _pointer_text(pointer, parser_source_map=parser_source_map)
    if not text:
        return None
    start_char = _offset_value(_pointer_field(pointer, "start_char"), 0)
    local_cut = cutpoint.cut_offset - start_char
    if local_cut < 0 or local_cut > len(text):
        return None
    text_before_cut, text_after_cut = boundary_anchor_excerpt(text, local_cut)
    if not text_before_cut.strip() or not text_after_cut.strip():
        return None
    return cutpoint.model_copy(
        update={
            "boundary_kind": classify_boundary_kind(text, local_cut),
            "text_before_cut": text_before_cut,
            "text_after_cut": text_after_cut,
            "cut_reason": cutpoint.cut_reason or cutpoint.reason or "boundary repaired from source excerpt",
            "reason": cutpoint.reason or "boundary repaired from source excerpt",
        }
    )


def nearest_legal_cutpoint(
    *, cut_offset: int, legal_cutpoints: list[int]
) -> tuple[int | None, int | None]:
    if not legal_cutpoints:
        return None, None
    nearest = min(legal_cutpoints, key=lambda candidate: (abs(candidate - cut_offset), candidate))
    return nearest, abs(nearest - cut_offset)


# Retain the old private names for callers that imported them from layerwise_llm.
_BoundaryAnchorResolution = BoundaryAnchorResolution
_is_heading_line = is_heading_line
_is_list_item_line = is_list_item_line
_is_table_row_line = is_table_row_line
_classify_boundary_kind = classify_boundary_kind
_boundary_cutpoint_legality_reason = boundary_cutpoint_legality_reason
_legal_cutpoints_for_text = legal_cutpoints_for_text
_boundary_candidates_for_pointer = boundary_candidates_for_pointer
_boundary_anchor_excerpt = boundary_anchor_excerpt
_resolve_boundary_anchor = resolve_boundary_anchor
_repair_boundary_cutpoint_from_source = repair_boundary_cutpoint_from_source
_nearest_legal_cutpoint = nearest_legal_cutpoint

__all__ = [
    "MAX_BOUNDARY_REPAIR_SHIFT_CHARS",
    "AnchorMatchMode",
    "BoundaryAnchorResolution",
    "BoundaryKind",
    "boundary_anchor_excerpt",
    "boundary_candidates_for_pointer",
    "boundary_cutpoint_legality_reason",
    "classify_boundary_kind",
    "is_heading_line",
    "is_list_item_line",
    "is_table_row_line",
    "legal_cutpoints_for_text",
    "nearest_legal_cutpoint",
    "repair_boundary_cutpoint_from_source",
    "resolve_boundary_anchor",
]
