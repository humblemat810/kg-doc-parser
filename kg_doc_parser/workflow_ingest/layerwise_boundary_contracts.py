"""Typed internal payloads used by boundary proposal and assembly."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

from .parser_core_contracts import SplitStrategy
from .serialization import JsonValue


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
