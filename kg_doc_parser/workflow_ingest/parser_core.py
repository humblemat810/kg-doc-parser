from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Mapping, Sequence
from typing import Literal, Protocol, cast

from pydantic import ValidationError

from .cache import WorkflowLLMCallCache
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
from .semantics import (
    HydratedTextPointer,
    SemanticNode,
    pointer_source_validation_error,
    validate_semantic_tree_integrity,
)
from .serialization import JsonObject, JsonValue, ModelDumpLike

_LOGGER = logging.getLogger(__name__)
_LEGACY_POINTER_ID_RE = re.compile(r"^p(?P<page>\d+)_c(?P<cluster>\d+)$")
SplitStrategy = Literal["excerpt_first", "boundary_first"]
ParserPayload = JsonObject
ParserSourceMap = dict[str, ParserPayload]


def source_map_fingerprint(parser_source_map: ParserSourceMap) -> str:
    """Return a stable identity for the source text addressed by pointers."""

    records = [
        {
            "source_cluster_id": source_cluster_id,
            "text": _source_text(record) if isinstance(record, Mapping) else None,
        }
        for source_cluster_id, record in sorted(parser_source_map.items())
    ]
    canonical = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _assert_source_map_matches_session(
    *,
    parse_session: ParseSessionState,
    parser_source_map: ParserSourceMap,
) -> None:
    expected = parse_session.metadata.get("source_map_fingerprint")
    if expected is None:
        # Sessions created before source binding remain readable. New sessions
        # always carry the fingerprint through initialize_parse_session.
        return
    if type(expected) is not str or expected != source_map_fingerprint(parser_source_map):
        raise ValueError("parser source map does not match the parse session source revision")


def _as_int(value: object, default: int = 0) -> int:
    if type(value) is int:
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    return default


def _source_text(record: object) -> str:
    if not isinstance(record, Mapping):
        return ""
    value = record.get("text", "")
    return value if isinstance(value, str) else ""


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _validation_diagnostics(
    *,
    phase: Literal["proposal", "review", "repair", "commit"],
    error: Exception,
) -> list[LayerValidationDiagnostic]:
    """Convert bounded Pydantic or adapter errors into retryable state."""

    details = error.errors(include_context=False) if isinstance(error, ValidationError) else []
    diagnostics: list[LayerValidationDiagnostic] = []
    for detail in details[:8]:
        raw_location = detail.get("loc", ())
        location = [str(item) for item in raw_location] if isinstance(raw_location, tuple) else []
        diagnostics.append(
            LayerValidationDiagnostic(
                phase=phase,
                error_type=str(detail.get("type", type(error).__name__)),
                message=str(detail.get("msg", str(error)))[:500],
                location=location,
            )
        )
    if not diagnostics:
        diagnostics.append(
            LayerValidationDiagnostic(
                phase=phase,
                error_type=type(error).__name__,
                message=str(error)[:500],
            )
        )
    return diagnostics


def _invalid_layer_result(
    *,
    phase: Literal["proposal", "review", "repair", "commit"],
    error: Exception,
) -> CurrentLayerResult:
    diagnostics = _validation_diagnostics(phase=phase, error=error)
    return CurrentLayerResult(
        children=[],
        satisfied=False,
        validation_diagnostics=diagnostics,
        metadata={
            "validation_failure": phase,
            "retryable": True,
            "validation_error": diagnostics[0].message,
        },
    )


class ParseSemanticFn(Protocol):
    """Provider callback for the legacy whole-document parse path."""

    def __call__(
        self,
        *,
        collection: SourceCollectionLike,
        parser_input_dict: ParserPayload,
        parser_source_map: ParserSourceMap,
        model_names: list[str] | None = None,
    ) -> object: ...


class ProposeLayerFn(Protocol):
    """Provider callback for one bounded semantic-layer proposal."""

    def __call__(
        self,
        *,
        collection: SourceCollectionLike,
        parser_input_dict: ParserPayload,
        parser_source_map: ParserSourceMap,
        parse_session: ParseSessionState,
        current_layer_context: CurrentLayerContext,
        semantic_tree: SemanticNode,
        split_strategy: SplitStrategy,
    ) -> CurrentLayerResult | Mapping[str, object] | Sequence[object]: ...


class ReviewLayerFn(Protocol):
    """Provider callback for reviewing one bounded semantic layer."""

    def __call__(
        self,
        *,
        parse_session: ParseSessionState,
        current_layer_context: CurrentLayerContext,
        current_layer_result: CurrentLayerResult,
        parser_source_map: ParserSourceMap,
        split_strategy: SplitStrategy,
    ) -> CurrentLayerReview | CurrentLayerResult | Mapping[str, object]: ...


class PointerCorrector(Protocol):
    """Repair one grounded pointer against the authoritative source map."""

    def __call__(
        self,
        pointer: HydratedTextPointer,
        parser_source_map: ParserSourceMap,
        /,
    ) -> HydratedTextPointer | None: ...


class SourceCollectionLike(Protocol):
    """Minimum collection identity needed by the parser core."""

    @property
    def collection_id(self) -> str: ...

    @property
    def title(self) -> str: ...

class SourceUnitLike(Protocol):
    """Minimum source-unit shape needed by deterministic fallback parsing."""

    @property
    def unit_id(self) -> str | None: ...

    @property
    def cluster_number(self) -> int | None: ...

    @property
    def text(self) -> str | None: ...


class SourcePageLike(Protocol):
    """Minimum page shape needed by deterministic fallback parsing."""

    @property
    def page_number(self) -> int: ...

    @property
    def units(self) -> Sequence[SourceUnitLike]: ...


class SourceCollectionWithPagesLike(SourceCollectionLike, Protocol):
    """Collection shape required by page-aware deterministic fallback parsing."""

    @property
    def pages(self) -> Sequence[SourcePageLike]: ...


class _NodeWithOptionalId(Protocol):
    @property
    def node_id(self) -> str | None: ...


def _required_node_id(node: _NodeWithOptionalId) -> str:
    if node.node_id is None:
        raise ValueError("semantic node must have a stable node_id")
    return node.node_id


def default_parse_semantic_fn(
    *,
    collection: SourceCollectionLike,
    parser_input_dict: ParserPayload,
    parser_source_map: ParserSourceMap,
    model_names: list[str] | None = None,
) -> object:
    from ..semantic_document_splitting_layerwise_edits import build_document_tree

    return build_document_tree(
        doc_id=collection.collection_id,
        llm_input_dict=parser_input_dict,
        source_map=parser_source_map,
        model_names=model_names,
    )


def _coerce_semantic_tree(tree: object) -> SemanticNode:
    if isinstance(tree, tuple):
        tree = tree[0]
    if isinstance(tree, ModelDumpLike):
        tree = tree.model_dump(mode="json")
    if isinstance(tree, dict):
        tree = SemanticNode.model_validate(tree)
    if isinstance(tree, SemanticNode):
        return tree
    raise TypeError(f"unsupported semantic tree result: {type(tree)!r}")


def _root_only(tree: SemanticNode) -> SemanticNode:
    payload = tree.model_dump(mode="json")
    payload["child_nodes"] = []
    return SemanticNode.model_validate(payload)


def _legacy_pointer_aliases(parser_source_map: ParserSourceMap) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for unit_id, record in parser_source_map.items():
        if not isinstance(record, Mapping):
            continue
        page_number = record.get("page_number")
        cluster_number = record.get("cluster_number")
        if page_number is None or cluster_number is None:
            continue
        alias = f"p{_as_int(page_number)}_c{_as_int(cluster_number)}"
        aliases.setdefault(alias, unit_id)
    return aliases


def _canonicalize_legacy_pointer_tree(
    tree: SemanticNode,
    *,
    parser_source_map: ParserSourceMap,
) -> SemanticNode:
    aliases = _legacy_pointer_aliases(parser_source_map)
    if not aliases:
        return tree

    def _canonicalize_source_cluster_id(source_cluster_id: str) -> str:
        alias_match = _LEGACY_POINTER_ID_RE.match(source_cluster_id)
        if alias_match is None:
            return source_cluster_id
        return aliases.get(
            source_cluster_id,
            source_cluster_id,
        )

    def _remap_pointer(pointer: HydratedTextPointer) -> HydratedTextPointer:
        canonical_id = _canonicalize_source_cluster_id(pointer.source_cluster_id)
        if canonical_id == pointer.source_cluster_id:
            return pointer
        return pointer.model_copy(update={"source_cluster_id": canonical_id})

    def _remap_node(node: SemanticNode) -> SemanticNode:
        return node.model_copy(
            update={
                "total_content_pointers": [_remap_pointer(pointer) for pointer in node.total_content_pointers],
                "aggregate_content_pointers": [
                    _remap_pointer(pointer) for pointer in node.aggregate_content_pointers
                ],
                "child_nodes": [_remap_node(child) for child in node.child_nodes],
            }
        )

    return _remap_node(tree)


def _pointer_end(pointer: HydratedTextPointer, source_map: ParserSourceMap | None = None) -> int:
    if pointer.end_char != -1:
        return pointer.end_char
    if source_map is not None:
        text = _source_text(source_map.get(pointer.source_cluster_id))
        if text:
            return max(0, len(text) - 1)
    return pointer.start_char


def _normalize_text(text: str) -> str:
    return "".join(text.split()).lower()


def _child_pointer_fingerprint(child: LayerChildCandidate) -> tuple[tuple[str, int, int, str], ...]:
    return tuple(
        (
            ptr.source_cluster_id,
            ptr.start_char,
            ptr.end_char,
            _normalize_text(ptr.verbatim_text),
        )
        for ptr in child.total_content_pointers
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


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged: list[tuple[int, int]] = []
    ordered_intervals = sorted(intervals)
    cur_s, cur_e = min(ordered_intervals)
    for s, e in ordered_intervals[1:]:
        if s <= cur_e + 1:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def _has_meaningful_gap_text(text: str) -> bool:
    return bool("".join(text.split()))


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
    overlap_conflicts: list[LayerSpanConflict] = []
    coverage_gaps: list[LayerCoverageGap] = []
    duplicate_notes: list[LayerDuplicateChildNote] = []
    review_notes: list[str] = []
    seen_overlap_pairs: set[tuple[str, str, str, str, int, int, str]] = set()
    seen_child_ids: dict[str, str] = {}

    # A repeated child ID is unsafe even when the spans differ: materializing
    # the layer indexes children by ID and would otherwise overwrite one
    # proposal nondeterministically.
    for child in current_layer_result.children:
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
        parent_children = [
            child for child in current_layer_result.children if child.parent_node_id == parent_id
        ]
        if not parent_children:
            explicitly_atomic = current_layer_result.atomic_retained and (
                not current_layer_result.atomic_parent_ids
                or parent_id in current_layer_result.atomic_parent_ids
            )
            if explicitly_atomic:
                parent_pointers_for_atomic = parent_pointers.get(parent_id, [])
                if not parent_pointers_for_atomic:
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
                    for parent_ptr in parent_pointers_for_atomic:
                        pointer_error = pointer_source_validation_error(parent_ptr, parser_source_map)
                        if pointer_error is not None:
                            coverage_gaps.append(
                                LayerCoverageGap(
                                    parent_node_id=parent_id,
                                    source_cluster_id=parent_ptr.source_cluster_id,
                                    gap_start=max(parent_ptr.start_char, 0),
                                    gap_end=_pointer_end(parent_ptr, parser_source_map),
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
                if not coverage_gaps or not any(
                    gap.parent_node_id == parent_id for gap in coverage_gaps
                ):
                    review_notes.append(f"parent {parent_id} retained as an explicit atomic unit")
            else:
                for parent_ptr in parent_pointers.get(parent_id, []):
                    parent_end = _pointer_end(parent_ptr, parser_source_map)
                    cluster_text = _source_text(
                        (parser_source_map or {}).get(parent_ptr.source_cluster_id)
                    )
                    expected_text = cluster_text[parent_ptr.start_char : parent_end + 1]
                    if _has_meaningful_gap_text(expected_text):
                        coverage_gaps.append(
                            LayerCoverageGap(
                                parent_node_id=parent_id,
                                source_cluster_id=parent_ptr.source_cluster_id,
                                gap_start=parent_ptr.start_char,
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

        seen_signatures: dict[tuple[str, tuple[tuple[str, int, int, str], ...]], str] = {}
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
                    f"duplicate child proposal under parent {parent_id}: {child.node_id} duplicates {duplicate_of}"
                )
            else:
                seen_signatures[signature] = child.node_id

        for left_index, left_child in enumerate(parent_children):
            for right_child in parent_children[left_index + 1 :]:
                for left_ptr in left_child.total_content_pointers:
                    for right_ptr in right_child.total_content_pointers:
                        if left_ptr.source_cluster_id != right_ptr.source_cluster_id:
                            continue
                        left_end = _pointer_end(left_ptr, parser_source_map)
                        right_end = _pointer_end(right_ptr, parser_source_map)
                        overlap_start = max(left_ptr.start_char, right_ptr.start_char)
                        overlap_end = min(left_end, right_end)
                        if overlap_start > overlap_end:
                            continue
                        pair_key = (
                            parent_id,
                            left_child.node_id,
                            right_child.node_id,
                            left_ptr.source_cluster_id,
                            overlap_start,
                            overlap_end,
                            "duplicate" if (
                                left_ptr.start_char == right_ptr.start_char
                                and left_end == right_end
                                and _normalize_text(left_ptr.verbatim_text) == _normalize_text(right_ptr.verbatim_text)
                            ) else "overlap",
                        )
                        if pair_key in seen_overlap_pairs:
                            continue
                        seen_overlap_pairs.add(pair_key)
                        conflict_kind = pair_key[-1]
                        overlap_conflicts.append(
                            LayerSpanConflict(
                                parent_node_id=parent_id,
                                left_child_id=_required_node_id(left_child),
                                right_child_id=_required_node_id(right_child),
                                source_cluster_id=left_ptr.source_cluster_id,
                                left_span=left_ptr,
                                right_span=right_ptr,
                                overlap_start=overlap_start,
                                overlap_end=overlap_end,
                                conflict_kind=conflict_kind,
                            )
                        )
                        review_notes.append(
                            f"{conflict_kind} between {_required_node_id(left_child)} and "
                            f"{_required_node_id(right_child)} "
                            f"on {left_ptr.source_cluster_id}:{overlap_start}-{overlap_end}"
                        )

        for parent_ptr in parent_pointers.get(parent_id, []):
            parent_pointer_error = (
                pointer_source_validation_error(parent_ptr, parser_source_map)
                if parser_source_map is not None
                else None
            )
            if parent_pointer_error is not None:
                coverage_gaps.append(
                    LayerCoverageGap(
                        parent_node_id=parent_id,
                        source_cluster_id=parent_ptr.source_cluster_id,
                        gap_start=max(parent_ptr.start_char, 0),
                        gap_end=_pointer_end(parent_ptr, parser_source_map),
                        expected_text=parent_pointer_error,
                    )
                )
                review_notes.append(
                    f"invalid parent pointer for {parent_id}: {parent_pointer_error}"
                )
                continue
            parent_end = _pointer_end(parent_ptr, parser_source_map)
            parent_start = max(parent_ptr.start_char, 0)
            if parent_end < parent_start:
                continue
            child_intervals = []
            for child in parent_children:
                if not child.total_content_pointers:
                    review_notes.append(
                        f"child {child.node_id} under {parent_id} has no content pointer"
                    )
                    continue
                for child_ptr in child.total_content_pointers:
                    pointer_error = (
                        pointer_source_validation_error(child_ptr, parser_source_map)
                        if parser_source_map is not None
                        else None
                    )
                    if pointer_error is not None:
                        coverage_gaps.append(
                            LayerCoverageGap(
                                parent_node_id=parent_id,
                                source_cluster_id=child_ptr.source_cluster_id,
                                gap_start=max(child_ptr.start_char, 0),
                                gap_end=_pointer_end(child_ptr, parser_source_map),
                                expected_text=pointer_error,
                            )
                        )
                        review_notes.append(
                            f"invalid child pointer {child.node_id} under {parent_id}: {pointer_error}"
                        )
                        continue
                    if child_ptr.source_cluster_id != parent_ptr.source_cluster_id:
                        continue
                    child_end = _pointer_end(child_ptr, parser_source_map)
                    if child_ptr.start_char < parent_start or child_end > parent_end:
                        coverage_gaps.append(
                            LayerCoverageGap(
                                parent_node_id=parent_id,
                                source_cluster_id=child_ptr.source_cluster_id,
                                gap_start=max(child_ptr.start_char, 0),
                                gap_end=child_end,
                                expected_text="child pointer is outside parent extent",
                            )
                        )
                        review_notes.append(
                            f"child pointer {child.node_id} is outside parent {parent_id}"
                        )
                        continue
                    child_intervals.append(
                        (max(child_ptr.start_char, 0), child_end)
                    )
            merged = _merge_intervals(child_intervals)
            cursor = parent_start
            cluster_text = _source_text(
                parser_source_map.get(parent_ptr.source_cluster_id)
            ) if parser_source_map else ""
            for start, end in merged:
                if start > cursor:
                    gap_start = cursor
                    gap_end = min(start - 1, parent_end)
                    if gap_start <= gap_end:
                        gap_text = cluster_text[gap_start : gap_end + 1] if cluster_text else ""
                        if not _has_meaningful_gap_text(gap_text):
                            cursor = max(cursor, gap_end + 1)
                        else:
                            coverage_gaps.append(
                                LayerCoverageGap(
                                    parent_node_id=parent_id,
                                    source_cluster_id=parent_ptr.source_cluster_id,
                                    gap_start=gap_start,
                                    gap_end=gap_end,
                                    expected_text=gap_text,
                                )
                            )
                            review_notes.append(
                                f"gap in parent {parent_id} for {parent_ptr.source_cluster_id}: {gap_start}-{gap_end}"
                            )
                cursor = max(cursor, end + 1)
                if cursor > parent_end:
                    break
            if cursor <= parent_end:
                gap_text = cluster_text[cursor : parent_end + 1] if cluster_text else ""
                if not _has_meaningful_gap_text(gap_text):
                    continue
                coverage_gaps.append(
                    LayerCoverageGap(
                        parent_node_id=parent_id,
                        source_cluster_id=parent_ptr.source_cluster_id,
                        gap_start=cursor,
                        gap_end=parent_end,
                        expected_text=gap_text,
                    )
                )
                review_notes.append(
                    f"gap in parent {parent_id} for {parent_ptr.source_cluster_id}: {cursor}-{parent_end}"
                )

    coverage_ok = not coverage_gaps
    branching_ok = all(
        len(
            [child for child in current_layer_result.children if child.parent_node_id == parent_id]
        ) != 1
        and all(len(child.child_candidates) != 1 for child in current_layer_result.children)
        for parent_id in current_layer_context.parent_node_ids
    )
    satisfied = coverage_ok and not overlap_conflicts and not duplicate_notes and branching_ok
    return coverage_ok, satisfied, overlap_conflicts, coverage_gaps, duplicate_notes, review_notes


def initialize_parse_session(
    *,
    collection: SourceCollectionLike,
    parser_input_dict: ParserPayload,
    parser_source_map: ParserSourceMap,
    max_depth: int = 10,
    allow_review: bool = True,
    split_strategy: SplitStrategy = "excerpt_first",
    fallback_split_strategy: SplitStrategy = "boundary_first",
    parse_semantic_fn: ParseSemanticFn | None = None,
) -> tuple[ParseSessionState, list[LayerFrontierItem], SemanticNode]:
    if parse_semantic_fn is not None:
        full_tree = _coerce_semantic_tree(
            parse_semantic_fn(
                collection=collection,
                parser_input_dict=parser_input_dict,
                parser_source_map=parser_source_map,
            )
        )
        full_tree = _canonicalize_legacy_pointer_tree(
            full_tree,
            parser_source_map=parser_source_map,
        )
        root = _root_only(full_tree)
        session = ParseSessionState(
            collection_id=collection.collection_id,
            root_node_id=_required_node_id(root),
            current_depth=0,
            max_depth=max_depth,
            allow_review=allow_review,
            split_strategy=split_strategy,
            fallback_split_strategy=fallback_split_strategy,
            strategy_history=[cast(SplitStrategy, split_strategy)],
            mode="legacy_compat",
            compat_full_tree=full_tree.model_dump(),
            metadata={"source_map_fingerprint": source_map_fingerprint(parser_source_map)},
        )
        frontier = [LayerFrontierItem(parent_node_id=_required_node_id(root), depth=0, order=0)]
        return session, frontier, root

    root = SemanticNode(
        node_id=f"{collection.collection_id}|root",
        title=collection.title,
        node_type="DOCUMENT_ROOT",
        total_content_pointers=[
            HydratedTextPointer(
                source_cluster_id=unit_id,
                start_char=0,
                end_char=-1,
                # The canonical persistence path validates excerpts against the
                # stored document content, so the root pointers need real text.
                verbatim_text=_source_text(record),
            )
            for unit_id, record in sorted(parser_source_map.items())
            if isinstance(record, Mapping)
            and record.get("participates_in_semantic_text", True)
        ],
        child_nodes=[],
        level_from_root=0,
    )
    session = ParseSessionState(
        collection_id=collection.collection_id,
        root_node_id=_required_node_id(root),
        current_depth=0,
        max_depth=max_depth,
        allow_review=allow_review,
        split_strategy=split_strategy,
        fallback_split_strategy=fallback_split_strategy,
        strategy_history=[cast(SplitStrategy, split_strategy)],
        mode="workflow_layered",
        metadata={
            "default_split_strategy": split_strategy,
            "default_fallback_split_strategy": fallback_split_strategy,
            "source_map_fingerprint": source_map_fingerprint(parser_source_map),
        },
    )
    frontier = [LayerFrontierItem(parent_node_id=_required_node_id(root), depth=0, order=0)]
    return session, frontier, root


def prepare_layer_frontier(
    *,
    parse_session: ParseSessionState,
    frontier_queue: list[LayerFrontierItem],
    semantic_tree: SemanticNode,
    max_retries: int = 3,
    max_items: int | None = None,
) -> tuple[CurrentLayerContext, list[LayerFrontierItem], ParseSessionState]:
    if not frontier_queue:
        raise ValueError("frontier queue is empty")
    sorted_queue = sorted(frontier_queue, key=lambda item: (item.depth, item.order))
    current_depth = sorted_queue[0].depth
    current_items = [item for item in sorted_queue if item.depth == current_depth]
    if max_items is not None:
        if max_items < 1:
            raise ValueError("max_items must be positive")
        selected_items = current_items[:max_items]
    else:
        selected_items = current_items
    # Preserve both same-depth items that did not fit in this batch and all
    # deeper work. Dropping the former loses durable parser work.
    remaining = [item for item in sorted_queue if item not in selected_items]
    parent_titles = []
    for parent_id in [item.parent_node_id for item in selected_items]:
        node = find_semantic_node(semantic_tree, parent_id)
        parent_titles.append(node.title if node is not None else parent_id)
    session = parse_session.model_copy(update={"current_depth": current_depth})
    frontier_identity = (
        f"depth={current_depth};parents={','.join(item.parent_node_id for item in selected_items)}"
    )
    def _pointers_for(parent_node_id: str) -> list[HydratedTextPointer]:
        parent = find_semantic_node(semantic_tree, parent_node_id)
        if parent is None:
            return []
        # Structural page-index containers own no leaf text themselves.  Their
        # aggregate span is the authoritative interval for the next refinement.
        return list(parent.total_content_pointers or parent.aggregate_content_pointers)

    default_split_strategy = cast(
        SplitStrategy,
        parse_session.metadata.get("default_split_strategy", parse_session.split_strategy),
    )

    context = CurrentLayerContext(
        depth=current_depth,
        parent_node_ids=[item.parent_node_id for item in selected_items],
        parent_titles=parent_titles,
        parent_content_pointers_by_id={
            item.parent_node_id: _pointers_for(item.parent_node_id) for item in selected_items
        },
        # Strategy selection is per layer.  Do not carry a prior layer's
        # triage decision into the next frontier depth.
        split_strategy=default_split_strategy,
        retry_count=int(
            parse_session.layer_attempts.get(
                frontier_identity,
                parse_session.layer_attempts.get(str(current_depth), 0),
            )
        ),
        max_retries=max_retries,
        metadata={"frontier_identity": frontier_identity},
    )
    return context, remaining, session


def legacy_children_for_context(
    *,
    parse_session: ParseSessionState,
    current_layer_context: CurrentLayerContext,
) -> CurrentLayerResult:
    if parse_session.compat_full_tree is None:
        raise ValueError("legacy compatibility tree is missing")
    full_tree = SemanticNode.model_validate(parse_session.compat_full_tree)
    children: list[LayerChildCandidate] = []
    atomic_parent_ids: set[str] = set()
    atomic_parent_pointers: dict[str, list[HydratedTextPointer]] = {}
    for parent_id in current_layer_context.parent_node_ids:
        parent = find_semantic_node(full_tree, parent_id)
        if parent is None:
            raise ValueError(
                f"legacy compatibility tree is missing frontier parent {parent_id!r}"
            )
        parent_children = list(parent.child_nodes)
        # A legacy callback has no separate proposal/review boundary.  Treat
        # a one-child result as an unsuccessful refinement and retain the
        # verified parent instead of allowing a unary tree into the workflow.
        if len(parent_children) < 2:
            atomic_parent_ids.add(parent_id)
            # The rejected refinement must retain the already verified parent
            # ownership, never a potentially narrower child transcription.
            parent_pointers = list(
                parent.total_content_pointers or parent.aggregate_content_pointers
            )
            atomic_parent_pointers[parent_id] = parent_pointers
            continue
        for child in parent_children:
            children.append(
                LayerChildCandidate(
                    node_id=_required_node_id(child),
                    parent_node_id=parent_id,
                    title=child.title,
                    node_type=child.node_type,
                    total_content_pointers=list(child.total_content_pointers),
                    # The legacy full tree is already authoritative.  Only
                    # nodes with children need another layer; treating every
                    # text node as expandable manufactures empty leaf layers.
                    expandable=bool(child.child_nodes),
                    metadata={"source": "legacy_compat"},
                )
            )
    return CurrentLayerResult(
        children=children,
        satisfied=True,
        reasoning_history=[],
        atomic_retained=bool(atomic_parent_ids),
        atomic_parent_ids=sorted(atomic_parent_ids),
        atomic_content_pointers_by_parent=atomic_parent_pointers,
        metadata={"source": "legacy_compat"},
    )


def propose_layer_breakdown(
    *,
    collection: SourceCollectionLike,
    parser_input_dict: ParserPayload,
    parser_source_map: ParserSourceMap,
    parse_session: ParseSessionState,
    current_layer_context: CurrentLayerContext,
    semantic_tree: SemanticNode,
    propose_layer_fn: ProposeLayerFn | None = None,
    llm_cache: WorkflowLLMCallCache | None = None,
    cache_namespace: Mapping[str, object] | None = None,
) -> CurrentLayerResult:
    _assert_source_map_matches_session(
        parse_session=parse_session,
        parser_source_map=parser_source_map,
    )
    if parse_session.mode == "legacy_compat":
        return legacy_children_for_context(
            parse_session=parse_session,
            current_layer_context=current_layer_context,
        )
    if propose_layer_fn is None:
        raise ValueError("workflow_layered mode requires propose_layer_fn")
    def call() -> CurrentLayerResult | Mapping[str, object] | Sequence[object]:
        return propose_layer_fn(
            collection=collection,
            parser_input_dict=parser_input_dict,
            parser_source_map=parser_source_map,
            parse_session=parse_session,
            current_layer_context=current_layer_context,
            semantic_tree=semantic_tree,
            split_strategy=current_layer_context.split_strategy,
        )
    try:
        if llm_cache is not None:
            proposed = llm_cache.cached_call(
                operation="propose_layer_breakdown",
                fingerprint={
                    "cache_namespace": dict(cache_namespace or {}),
                    "collection_id": collection.collection_id,
                    "parse_session": parse_session,
                    "current_layer_context": current_layer_context,
                    "semantic_tree": semantic_tree,
                    "parser_input_dict": parser_input_dict,
                    "parser_source_map": parser_source_map,
                },
                fn=call,
            )
        else:
            proposed = call()
        if isinstance(proposed, CurrentLayerResult):
            return proposed
        if isinstance(proposed, dict):
            return CurrentLayerResult.model_validate(proposed)
        if isinstance(proposed, list):
            return CurrentLayerResult(children=[_coerce_layer_child(child) for child in cast(list[object], proposed)])
        raise TypeError("unsupported proposed layer result")
    except (ValidationError, TypeError, ValueError) as exc:
        return _invalid_layer_result(phase="proposal", error=exc)


def review_layer(
    *,
    parse_session: ParseSessionState,
    current_layer_context: CurrentLayerContext,
    current_layer_result: CurrentLayerResult,
    parser_source_map: ParserSourceMap | None = None,
    review_layer_fn: ReviewLayerFn | None = None,
    llm_cache: WorkflowLLMCallCache | None = None,
    cache_namespace: Mapping[str, object] | None = None,
) -> tuple[CurrentLayerReview, ParseSessionState]:
    if parser_source_map is not None:
        _assert_source_map_matches_session(
            parse_session=parse_session,
            parser_source_map=parser_source_map,
        )
    if parse_session.mode == "legacy_compat":
        (
            coverage_ok,
            invariant_satisfied,
            overlap_conflicts,
            coverage_gaps,
            duplicate_notes,
            invariant_notes,
        ) = detect_layer_invariants(
            current_layer_context=current_layer_context,
            current_layer_result=current_layer_result,
            parser_source_map=parser_source_map,
        )
        structural_issues = _layer_structural_issues(
            current_layer_context=current_layer_context,
            current_layer_result=current_layer_result,
        )
        invariant_satisfied = invariant_satisfied and not structural_issues
        return (
            CurrentLayerReview(
                updated_result=current_layer_result.model_copy(update={"satisfied": invariant_satisfied}),
                coverage_ok=coverage_ok,
                satisfied=invariant_satisfied,
                strategy_used=current_layer_context.split_strategy,
                overlap_conflicts=overlap_conflicts,
                coverage_gap_notes=coverage_gaps,
                duplicate_child_notes=duplicate_notes,
                structural_issues=structural_issues,
                review_notes=["semantic review skipped for legacy compatibility", *invariant_notes[:20]],
                metadata={"review_skipped": True, "legacy_compat": True, "deterministic_validation": True},
            ),
            parse_session,
        )
    if not parse_session.allow_review:
        coverage_ok, invariant_satisfied, overlap_conflicts, coverage_gaps, duplicate_notes, invariant_notes = detect_layer_invariants(
            current_layer_context=current_layer_context,
            current_layer_result=current_layer_result,
            parser_source_map=parser_source_map,
        )
        return (
            CurrentLayerReview(
                updated_result=current_layer_result.model_copy(update={"satisfied": invariant_satisfied}),
                coverage_ok=coverage_ok,
                satisfied=invariant_satisfied,
                strategy_used=current_layer_context.split_strategy,
                overlap_conflicts=overlap_conflicts,
                coverage_gap_notes=coverage_gaps,
                duplicate_child_notes=duplicate_notes,
                structural_issues=_layer_structural_issues(
                    current_layer_context=current_layer_context,
                    current_layer_result=current_layer_result,
                ),
                review_notes=["semantic review disabled", *invariant_notes[:20]],
                metadata={"review_skipped": True, "deterministic_validation": True},
            ),
            parse_session,
        )
    if review_layer_fn is None:
        reviewed = current_layer_result
    else:
        def call() -> CurrentLayerReview | CurrentLayerResult | Mapping[str, object]:
            return review_layer_fn(
                parse_session=parse_session,
                current_layer_context=current_layer_context,
                current_layer_result=current_layer_result,
                parser_source_map=parser_source_map or {},
                split_strategy=current_layer_context.split_strategy,
            )
        try:
            if llm_cache is not None:
                reviewed = llm_cache.cached_call(
                    operation=f"review_cud_proposal:{current_layer_context.split_strategy}",
                    fingerprint={
                        "cache_namespace": dict(cache_namespace or {}),
                        "parse_session": parse_session,
                        "current_layer_context": current_layer_context,
                        "current_layer_result": current_layer_result,
                        "parser_source_map": parser_source_map or {},
                        "split_strategy": current_layer_context.split_strategy,
                    },
                    fn=call,
                )
            else:
                reviewed = call()
        except TimeoutError as exc:
            # A review timeout is different from a malformed or failed
            # provider response.  If the host-side invariants are complete,
            # deterministic validation can authorize this layer without
            # pretending that the semantic critic ran.
            coverage_ok, invariant_satisfied, overlap_conflicts, coverage_gaps, duplicate_notes, invariant_notes = detect_layer_invariants(
                current_layer_context=current_layer_context,
                current_layer_result=current_layer_result,
                parser_source_map=parser_source_map,
            )
            reviewed = CurrentLayerReview(
                updated_result=current_layer_result,
                coverage_ok=coverage_ok,
                satisfied=invariant_satisfied,
                strategy_used=current_layer_context.split_strategy,
                overlap_conflicts=overlap_conflicts,
                coverage_gap_notes=coverage_gaps,
                duplicate_child_notes=duplicate_notes,
                structural_issues=_layer_structural_issues(
                    current_layer_context=current_layer_context,
                    current_layer_result=current_layer_result,
                ),
                review_notes=[
                    "quality_unknown: semantic layer review timed out",
                    "deterministic invariants authorized the layer" if invariant_satisfied else "deterministic invariants rejected the layer",
                    *invariant_notes[:10],
                ],
                metadata={
                    "review_timeout": True,
                    "review_timeout_reason": repr(exc)[:500],
                },
            )
        except Exception as exc:
            reviewed = CurrentLayerReview(
                updated_result=current_layer_result,
                coverage_ok=None,
                satisfied=None,
                strategy_used=current_layer_context.split_strategy,
                review_notes=[
                    "quality_unknown: semantic layer review provider failed",
                    "deterministic checks did not authorize successful review",
                ],
                metadata={
                    "review_failure": "provider_failure",
                    "review_failure_reason": repr(exc)[:500],
                },
            )
    try:
        if isinstance(reviewed, CurrentLayerReview):
            result = reviewed
        elif isinstance(reviewed, CurrentLayerResult):
            result = CurrentLayerReview(
                updated_result=reviewed,
                coverage_ok=_optional_bool(reviewed.metadata.get("layer_coverage_ok")),
                satisfied=reviewed.satisfied,
                validation_diagnostics=list(reviewed.validation_diagnostics),
            )
        elif isinstance(reviewed, dict):
            if "updated_result" in reviewed or "coverage_ok" in reviewed or "review_notes" in reviewed:
                result = CurrentLayerReview.model_validate(reviewed)
            else:
                parsed = CurrentLayerResult.model_validate(reviewed)
                result = CurrentLayerReview(
                    updated_result=parsed,
                    coverage_ok=_optional_bool(parsed.metadata.get("layer_coverage_ok")),
                    satisfied=parsed.satisfied,
                    validation_diagnostics=list(parsed.validation_diagnostics),
                )
        else:
            raise TypeError("unsupported reviewed layer result")
    except (ValidationError, TypeError, ValueError) as exc:
        invalid_result = _invalid_layer_result(phase="review", error=exc)
        result = CurrentLayerReview(
            updated_result=invalid_result,
            coverage_ok=None,
            satisfied=None,
            validation_diagnostics=list(invalid_result.validation_diagnostics),
            review_notes=["quality_unknown: semantic layer review payload failed validation"],
            metadata={
                "review_failure": "structured_validation",
                "retryable": True,
            },
        )
    coverage_ok, invariant_satisfied, overlap_conflicts, coverage_gaps, duplicate_notes, invariant_notes = detect_layer_invariants(
        current_layer_context=current_layer_context,
        current_layer_result=result.updated_result or current_layer_result,
        parser_source_map=parser_source_map,
    )
    merged_notes = list(result.review_notes)
    for note in invariant_notes:
        if note not in merged_notes:
            merged_notes.append(note)
    provider_failure = bool(result.metadata.get("review_failure"))
    structural_issues = _layer_structural_issues(
        current_layer_context=current_layer_context,
        current_layer_result=result.updated_result or current_layer_result,
    )
    for issue in structural_issues:
        if issue.message not in merged_notes:
            merged_notes.append(issue.message)
    validation_diagnostics = list(result.validation_diagnostics)
    for diagnostic in (result.updated_result or current_layer_result).validation_diagnostics:
        if diagnostic not in validation_diagnostics:
            validation_diagnostics.append(diagnostic)
    base_satisfied = result.satisfied if result.satisfied is not None else invariant_satisfied
    satisfied = (
        None
        if provider_failure
        else bool(
            base_satisfied
            and not (overlap_conflicts or coverage_gaps or duplicate_notes or structural_issues)
            and not validation_diagnostics
        )
    )
    effective_coverage_ok = None if provider_failure else coverage_ok
    updated_result = (result.updated_result or current_layer_result).model_copy(
        update={
            # A malformed provider payload is quality-unknown at the review
            # layer, but its persisted result must still be explicitly
            # unsatisfied so routing cannot treat it as a usable result.
            "satisfied": False if validation_diagnostics else satisfied,
            "metadata": {
                **(result.updated_result or current_layer_result).metadata,
                **result.metadata,
                "split_strategy": current_layer_context.split_strategy,
                "overlap_conflicts": [
                    item.model_dump(field_mode="backend", dump_format="json") for item in overlap_conflicts
                ],
                "coverage_gaps": [
                    item.model_dump(field_mode="backend", dump_format="json") for item in coverage_gaps
                ],
                "duplicate_child_notes": [
                    item.model_dump(field_mode="backend", dump_format="json") for item in duplicate_notes
                ],
                "structural_issues": [
                    item.model_dump(field_mode="backend", dump_format="json")
                    for item in structural_issues
                ],
                "validation_diagnostics": [
                    item.model_dump(field_mode="backend", dump_format="json")
                    for item in validation_diagnostics
                ],
            },
        }
    )
    result = result.model_copy(
        update={
            "updated_result": updated_result,
            "coverage_ok": effective_coverage_ok,
            "satisfied": satisfied,
            "strategy_used": current_layer_context.split_strategy,
            "overlap_conflicts": overlap_conflicts,
            "coverage_gap_notes": coverage_gaps,
            "duplicate_child_notes": duplicate_notes,
            "structural_issues": structural_issues,
            "validation_diagnostics": validation_diagnostics,
            "review_notes": merged_notes,
        }
    )
    attempts = dict(parse_session.layer_attempts)
    frontier_identity = str(
        current_layer_context.metadata.get(
            "frontier_identity",
            f"depth={current_layer_context.depth};parents={','.join(current_layer_context.parent_node_ids)}",
        )
    )
    attempts[frontier_identity] = current_layer_context.retry_count + 1
    review_packet = {
        "strategy": result.strategy_used,
        "coverage_ok": result.coverage_ok,
        "satisfied": result.satisfied,
        "review_notes": list(result.review_notes),
        "overlap_conflicts": [
            item.model_dump(field_mode="backend", dump_format="json")
            for item in result.overlap_conflicts
        ],
        "coverage_gap_notes": [
            item.model_dump(field_mode="backend", dump_format="json")
            for item in result.coverage_gap_notes
        ],
        "duplicate_child_notes": [
            item.model_dump(field_mode="backend", dump_format="json")
            for item in result.duplicate_child_notes
        ],
        "structural_issues": [
            item.model_dump(field_mode="backend", dump_format="json")
            for item in result.structural_issues
        ],
        "validation_diagnostics": [
            item.model_dump(field_mode="backend", dump_format="json")
            for item in result.validation_diagnostics
        ],
        "metadata": dict(result.metadata),
    }
    return result, parse_session.model_copy(
        update={"layer_attempts": attempts, "last_review": review_packet}
    )


def apply_cud_update(
    *,
    current_layer_result: CurrentLayerResult,
    current_layer_review: CurrentLayerReview,
) -> CurrentLayerResult:
    updated = current_layer_review.updated_result or current_layer_result
    metadata = dict(updated.metadata)
    metadata.update(current_layer_review.metadata)
    metadata["split_strategy"] = current_layer_review.strategy_used
    if current_layer_review.coverage_ok is not None:
        metadata["layer_coverage_ok"] = current_layer_review.coverage_ok
    if current_layer_review.review_notes:
        metadata["review_notes"] = list(current_layer_review.review_notes)
    if current_layer_review.overlap_conflicts:
        metadata["overlap_conflicts"] = [
            item.model_dump(field_mode="backend", dump_format="json")
            for item in current_layer_review.overlap_conflicts
        ]
    if current_layer_review.coverage_gap_notes:
        metadata["coverage_gap_notes"] = [
            item.model_dump(field_mode="backend", dump_format="json")
            for item in current_layer_review.coverage_gap_notes
        ]
    if current_layer_review.duplicate_child_notes:
        metadata["duplicate_child_notes"] = [
            item.model_dump(field_mode="backend", dump_format="json")
            for item in current_layer_review.duplicate_child_notes
        ]
    if current_layer_review.structural_issues:
        metadata["structural_issues"] = [
            item.model_dump(field_mode="backend", dump_format="json")
            for item in current_layer_review.structural_issues
        ]
    if current_layer_review.validation_diagnostics:
        metadata["validation_diagnostics"] = [
            item.model_dump(field_mode="backend", dump_format="json")
            for item in current_layer_review.validation_diagnostics
        ]
    satisfied = (
        current_layer_review.satisfied
        if current_layer_review.satisfied is not None
        else updated.satisfied
    )
    return updated.model_copy(
        update={
            "satisfied": satisfied,
            "review_rounds": updated.review_rounds + 1,
            "metadata": metadata,
        }
    )


def check_layer_coverage(
    *,
    current_layer_context: CurrentLayerContext,
    current_layer_result: CurrentLayerResult,
    current_layer_review: CurrentLayerReview | None = None,
) -> tuple[bool, list[str]]:
    if current_layer_review is not None and current_layer_review.metadata.get("review_failure"):
        return False, ["layer review unavailable; quality is unknown"]
    if current_layer_review is not None and current_layer_review.coverage_ok is not None:
        notes = list(current_layer_review.review_notes)
        notes.extend(
            [
                f"overlap conflict: {item.left_child_id} vs {item.right_child_id} @ {item.source_cluster_id}:{item.overlap_start}-{item.overlap_end}"
                for item in current_layer_review.overlap_conflicts
            ]
        )
        notes.extend(
            [
                f"coverage gap: {item.parent_node_id} {item.source_cluster_id}:{item.gap_start}-{item.gap_end}"
                for item in current_layer_review.coverage_gap_notes
            ]
        )
        notes.extend(
            [
                f"duplicate child: {item.child_node_id} duplicates {item.duplicate_of_child_node_id}"
                for item in current_layer_review.duplicate_child_notes
            ]
        )
        return bool(current_layer_review.coverage_ok), notes

    metadata_flag = current_layer_result.metadata.get("layer_coverage_ok")
    if isinstance(metadata_flag, bool):
        raw_notes = current_layer_result.metadata.get("review_notes")
        notes = raw_notes if isinstance(raw_notes, list) else []
        return metadata_flag, [str(note) for note in notes]

    if current_layer_result.atomic_retained:
        return True, []

    parent_ids = set(current_layer_context.parent_node_ids)
    covered_parents = {
        child.parent_node_id for child in current_layer_result.children if child.parent_node_id in parent_ids
    }
    missing = sorted(parent_ids - covered_parents)
    if missing:
        return False, [f"missing children for parent ids: {', '.join(missing)}"]
    return True, []


def switch_split_strategy(
    *,
    parse_session: ParseSessionState,
    current_layer_context: CurrentLayerContext,
) -> tuple[ParseSessionState, CurrentLayerContext]:
    if current_layer_context.split_strategy == parse_session.fallback_split_strategy:
        raise ValueError("fallback split strategy already exhausted")
    next_strategy = parse_session.fallback_split_strategy
    history = list(parse_session.strategy_history)
    if not history or history[-1] != current_layer_context.split_strategy:
        history.append(current_layer_context.split_strategy)
    if history[-1] != next_strategy:
        history.append(next_strategy)
    updated_session = parse_session.model_copy(
        update={
            "split_strategy": next_strategy,
            "strategy_history": history,
            "strategy_switch_count": parse_session.strategy_switch_count + 1,
        }
    )
    updated_context = current_layer_context.model_copy(
        update={
            "split_strategy": next_strategy,
            "retry_count": 0,
            "metadata": {
                **current_layer_context.metadata,
                "split_strategy_switch_from": current_layer_context.split_strategy,
                "split_strategy_switch_to": next_strategy,
            },
        }
    )
    return updated_session, updated_context


def repair_layer_candidates(
    *,
    current_layer_result: CurrentLayerResult,
    parser_source_map: ParserSourceMap,
    correct_pointer_fn: PointerCorrector,
) -> tuple[CurrentLayerResult, int]:
    def _pointer_context(pointer: HydratedTextPointer) -> str:
        source = parser_source_map.get(pointer.source_cluster_id)
        text = _source_text(source)
        preview = text.replace("\n", "\\n").replace("\t", "\\t")
        if len(preview) > 120:
            preview = preview[:117] + "..."
        return (
            f"source_cluster_id={pointer.source_cluster_id!r}, "
            f"span=({pointer.start_char},{pointer.end_char}), "
            f"verbatim_text={pointer.verbatim_text!r}, "
            f"source_text={preview!r}"
        )

    repaired_children: list[LayerChildCandidate] = []
    repair_failures: list[str] = []
    repair_diagnostics: list[LayerValidationDiagnostic] = []
    repaired_count = 0
    for child in current_layer_result.children:
        repaired_ptrs = []
        child_failed = False
        for pointer in child.total_content_pointers:
            try:
                fixed = correct_pointer_fn(pointer, parser_source_map)
            except (ValidationError, TypeError, ValueError) as exc:
                repair_diagnostics.extend(_validation_diagnostics(phase="repair", error=exc))
                message = (
                    f"pointer repair validation failed for child {child.title!r} "
                    f"(parent={child.parent_node_id!r}, node_id={child.node_id!r}); "
                    f"{type(exc).__name__}: {exc}"
                )
                _LOGGER.warning("repair_layer_candidates failed: %s", message)
                repair_failures.append(message)
                child_failed = True
                break
            if fixed is None:
                message = (
                    f"unrecoverable pointer for child {child.title!r} "
                    f"(parent={child.parent_node_id!r}, node_id={child.node_id!r}); "
                    f"{_pointer_context(pointer)}"
                )
                _LOGGER.warning("repair_layer_candidates failed: %s", message)
                repair_failures.append(message)
                child_failed = True
                break
            if fixed.model_dump() != pointer.model_dump():
                repaired_count += 1
            repaired_ptrs.append(fixed)
        if child_failed:
            # A bad proposal must not erase verified siblings or abort an
            # unrelated parent.  Omit only this replacement; the invariant
            # checker will report the affected parent's coverage gap.
            continue
        repaired_children.append(
            child.model_copy(update={"total_content_pointers": repaired_ptrs})
        )
    metadata = dict(current_layer_result.metadata)
    if repair_failures:
        metadata["repair_failures"] = cast(JsonValue, repair_failures[:32])
        metadata["repair_failure_scope"] = "child_replacement"
        metadata["failure_type"] = "repair_failure"
        metadata["rollback"] = "verified_parent_retained"
        metadata["retryable"] = True
    validation_diagnostics = list(current_layer_result.validation_diagnostics)
    for diagnostic in repair_diagnostics:
        if diagnostic not in validation_diagnostics:
            validation_diagnostics.append(diagnostic)
    return current_layer_result.model_copy(
        update={
            "children": repaired_children,
            "metadata": metadata,
            "validation_diagnostics": validation_diagnostics,
            "satisfied": False if validation_diagnostics else current_layer_result.satisfied,
        }
    ), repaired_count


def dedupe_and_filter_layer(
    *,
    current_layer_context: CurrentLayerContext,
    current_layer_result: CurrentLayerResult,
) -> CurrentLayerResult:
    parent_ids = set(current_layer_context.parent_node_ids)
    seen: set[tuple[str, str, str, tuple[tuple[str, int, int, str], ...]]] = set()
    filtered: list[LayerChildCandidate] = []
    validation_diagnostics = list(current_layer_result.validation_diagnostics)
    for child in current_layer_result.children:
        if child.parent_node_id not in parent_ids:
            diagnostic = LayerValidationDiagnostic(
                phase="commit",
                error_type="foreign_parent",
                message=(
                    f"child {child.node_id!r} references parent "
                    f"{child.parent_node_id!r} outside the current layer"
                ),
                location=[child.node_id, child.parent_node_id],
            )
            if diagnostic not in validation_diagnostics:
                validation_diagnostics.append(diagnostic)
            continue
        key = (
            child.parent_node_id,
            child.node_type,
            child.title.strip().lower(),
            _child_pointer_fingerprint(child),
        )
        if key in seen:
            continue
        seen.add(key)
        filtered.append(child)
    return current_layer_result.model_copy(
        update={
            "children": filtered,
            "validation_diagnostics": validation_diagnostics,
            "satisfied": current_layer_result.satisfied and not validation_diagnostics,
        }
    )


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


def _coerce_layer_child(value: object) -> LayerChildCandidate:
    if isinstance(value, LayerChildCandidate):
        return value
    if isinstance(value, dict):
        return LayerChildCandidate.model_validate(value)
    raise TypeError("unsupported layer child candidate")
