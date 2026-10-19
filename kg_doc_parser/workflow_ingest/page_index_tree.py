"""Grounded semantic-tree assembly for validated PageIndex blocks."""

from __future__ import annotations

import json
from typing import Protocol

from kogwistar.id_provider import stable_id
from kogwistar.utils.fuzzy_offsets import FuzzySpanHit, find_best_fuzzy_span

from .models import GroundedSourceRecord
from .page_index_contracts import PageIndexBlockSpec
from .semantics import (
    HydratedTextPointer,
    SemanticNode,
    correct_and_validate_pointer,
)
from .serialization import JsonObject


class PageIndexTraceLogger(Protocol):
    """Receive bounded diagnostic messages from PageIndex tree building."""

    def __call__(self, message: str, /) -> None: ...


def stable_part(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))


def page_index_normalize_text(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def page_index_find_best_fuzzy_span(
    *,
    page_text: str,
    excerpt: str,
    origin_start: int,
    scan_band: int | None = None,
) -> FuzzySpanHit | None:
    page_text_norm = page_index_normalize_text(page_text)
    return find_best_fuzzy_span(
        content=page_text,
        excerpt=excerpt,
        origin_start=origin_start,
        scan_band=scan_band,
        candidate_filter=lambda candidate: page_index_normalize_text(candidate) != page_text_norm,
    )


def find_page_unit(
    authoritative_source_map: dict[str, GroundedSourceRecord], page_number: int
) -> tuple[str, str]:
    for unit_id, record in authoritative_source_map.items():
        if record.page_number == page_number and record.participates_in_semantic_text:
            return unit_id, record.text
    raise ValueError(f"missing text page for page number {page_number}")


def resolve_pointer(
    *,
    unit_id: str,
    page_text: str,
    excerpt: str,
    start_at: int = 0,
    repair_stats: dict[str, int] | None = None,
    trace_log: PageIndexTraceLogger | None = None,
) -> HydratedTextPointer:
    needle = excerpt.strip() or excerpt or page_text.strip()
    exact_start = page_text.find(needle, max(0, start_at)) if needle else -1
    if exact_start >= 0:
        return HydratedTextPointer(
            source_cluster_id=unit_id,
            start_char=exact_start,
            end_char=exact_start + len(needle) - 1,
            verbatim_text=page_text[exact_start : exact_start + len(needle)],
        )
    fuzzy_hit = page_index_find_best_fuzzy_span(
        page_text=page_text,
        excerpt=needle,
        origin_start=start_at,
    )
    if fuzzy_hit is None:
        if repair_stats is not None:
            repair_stats["pointer_fuzzy_failures"] = repair_stats.get("pointer_fuzzy_failures", 0) + 1
        raise ValueError(f"unable to resolve excerpt against page text for {unit_id!r}: {needle!r}")
    fuzzy_candidate = HydratedTextPointer(
        source_cluster_id=unit_id,
        start_char=fuzzy_hit.start,
        end_char=fuzzy_hit.end - 1,
        verbatim_text=page_text[fuzzy_hit.start : fuzzy_hit.end],
    )
    resolved = correct_and_validate_pointer(fuzzy_candidate, {unit_id: {"text": page_text}})
    if resolved is None:
        if repair_stats is not None:
            repair_stats["pointer_fuzzy_failures"] = repair_stats.get("pointer_fuzzy_failures", 0) + 1
        raise ValueError(f"unable to resolve excerpt against page text for {unit_id!r}: {needle!r}")
    if repair_stats is not None:
        repair_stats["pointer_fuzzy_repairs"] = repair_stats.get("pointer_fuzzy_repairs", 0) + 1
    if trace_log is not None:
        trace_log(
            "page_index_pointer_fuzzy_repair "
            f"unit_id={unit_id} start={fuzzy_hit.start} end={fuzzy_hit.end} score={fuzzy_hit.score:.2f}"
        )
    return resolved


def make_semantic_node(
    *,
    title: str,
    node_type: str,
    parent_id: str | None,
    level_from_root: int,
    pointers: list[HydratedTextPointer],
    summary: str = "",
    metadata: JsonObject | None = None,
) -> SemanticNode:
    return SemanticNode(
        title=title,
        summary=summary,
        node_type=node_type,
        parent_id=parent_id,
        level_from_root=level_from_root,
        total_content_pointers=pointers,
        child_nodes=[],
        metadata=dict(metadata or {}),
    )


def materialize_block_tree(
    *,
    block_specs: list[PageIndexBlockSpec],
    page_text: str,
    unit_id: str,
    parent_id: str,
    level_from_root: int,
    start_at: int = 0,
    repair_stats: dict[str, int] | None = None,
    trace_log: PageIndexTraceLogger | None = None,
    summary_enabled: bool = True,
) -> tuple[list[SemanticNode], int]:
    nodes: list[SemanticNode] = []
    cursor = start_at
    for spec in block_specs:
        pointer = resolve_pointer(
            unit_id=unit_id,
            page_text=page_text,
            excerpt=spec.excerpt,
            start_at=cursor,
            repair_stats=repair_stats,
            trace_log=trace_log,
        )
        cursor = pointer.end_char + 1
        is_heading = spec.source_role == "heading"
        node_id = (
            str(
                stable_id(
                    "workflow_ingest.page_index.heading_container",
                    stable_part(parent_id),
                    stable_part(spec.node_type),
                    stable_part(pointer.source_cluster_id),
                    stable_part(pointer.start_char),
                    stable_part(pointer.end_char),
                )
            )
            if is_heading
            else None
        )
        node = make_semantic_node(
            title=spec.title,
            node_type=spec.node_type,
            parent_id=parent_id,
            level_from_root=level_from_root,
            pointers=[] if is_heading else [pointer],
            summary=spec.summary.strip() if summary_enabled else "",
            metadata={
                "page_index_role": "heading_container" if is_heading else "content",
                "semantic_kind": "heading" if is_heading else "content",
                "legacy_node_type": spec.node_type,
                "source_role": spec.source_role,
                "display_excerpt": spec.display_excerpt,
                "summary_unavailable": bool(summary_enabled and not spec.summary.strip()),
            },
        )
        if node_id is not None:
            node = node.model_copy(update={"node_id": node_id})
        heading_leaf = None
        if is_heading:
            heading_leaf = make_semantic_node(
                title=spec.title,
                node_type="HEADING_TEXT",
                parent_id=node.node_id,
                level_from_root=level_from_root + 1,
                pointers=[pointer],
                summary=spec.summary.strip() if summary_enabled else "",
                metadata={
                    "page_index_role": "heading_text",
                    "semantic_kind": "heading_text",
                    "display_excerpt": spec.display_excerpt,
                    "summary_unavailable": bool(summary_enabled and not spec.summary.strip()),
                },
            )
            heading_leaf = heading_leaf.model_copy(
                update={
                    "node_id": str(
                        stable_id(
                            "workflow_ingest.page_index.heading_text",
                            stable_part(node.node_id),
                            stable_part(pointer.source_cluster_id),
                            stable_part(pointer.start_char),
                            stable_part(pointer.end_char),
                        )
                    )
                }
            )
        child_nodes, cursor = materialize_block_tree(
            block_specs=spec.child_nodes,
            page_text=page_text,
            unit_id=unit_id,
            parent_id=node.node_id or parent_id,
            level_from_root=level_from_root + 1,
            start_at=cursor,
            repair_stats=repair_stats,
            trace_log=trace_log,
            summary_enabled=summary_enabled,
        )
        if is_heading:
            aggregate_end = max(pointer.end_char, cursor - 1)
            node.aggregate_content_pointers = [
                HydratedTextPointer(
                    source_cluster_id=pointer.source_cluster_id,
                    start_char=pointer.start_char,
                    end_char=aggregate_end,
                    verbatim_text=page_text[pointer.start_char : aggregate_end + 1],
                )
            ]
        if heading_leaf is not None and child_nodes:
            node.child_nodes.append(heading_leaf)
        elif heading_leaf is not None:
            node.total_content_pointers = [pointer]
            node.metadata = {
                **dict(node.metadata),
                "page_index_role": "heading_leaf",
                "atomic_retained": True,
            }
        node.child_nodes.extend(child_nodes)
        nodes.append(node)
    return nodes, cursor


def rebase_page_index_subtree(
    children: list[SemanticNode], *, parent_id: str | None, level_from_root: int
) -> list[SemanticNode]:
    """Rebase every descendant after removing a structural wrapper."""

    pending: list[tuple[SemanticNode, str | None, int, bool]] = [
        (child, parent_id, level_from_root, False) for child in reversed(children)
    ]
    rebased: dict[int, SemanticNode] = {}
    while pending:
        current, expected_parent_id, current_level, exiting = pending.pop()
        marker = id(current)
        if exiting:
            rebased[marker] = current.model_copy(
                update={
                    "parent_id": expected_parent_id,
                    "level_from_root": current_level,
                    "child_nodes": [rebased[id(child)] for child in current.child_nodes],
                }
            )
            continue
        pending.append((current, expected_parent_id, current_level, True))
        for child in reversed(current.child_nodes):
            pending.append((child, current.node_id, current_level + 1, False))
    return [rebased[id(child)] for child in children]


def normalize_page_index_unary_nodes(node: SemanticNode) -> SemanticNode:
    """Normalize unary shapes without discarding source ownership."""

    pending: list[tuple[SemanticNode, bool]] = [(node, False)]
    normalized: dict[int, SemanticNode] = {}
    while pending:
        current, exiting = pending.pop()
        marker = id(current)
        if exiting:
            normalized_children = [normalized[id(child)] for child in current.child_nodes]
            current = current.model_copy(update={"child_nodes": normalized_children})
            if len(normalized_children) != 1:
                normalized[marker] = current
                continue
            child = normalized_children[0]
            role = str(current.metadata.get("page_index_role", ""))
            child_role = str(child.metadata.get("page_index_role", ""))
            is_document_wrapper = current.node_type == "DOCUMENT_ROOT" or role == "DOCUMENT_ROOT"
            is_section_wrapper = current.node_type == "PAGE" or role in {"page", "heading_container"}
            owns_content = not (is_document_wrapper or current.node_type == "PAGE" or role == "page")
            child_owns_content = bool(child.total_content_pointers) and not (
                child.node_type in {"DOCUMENT_ROOT", "PAGE"} or child_role == "page"
            )
            if (
                (is_document_wrapper or is_section_wrapper)
                and (not owns_content or not current.total_content_pointers)
                and not child_owns_content
                and len(child.child_nodes) >= 2
            ):
                normalized[marker] = current.model_copy(
                    update={
                        "child_nodes": rebase_page_index_subtree(
                            list(child.child_nodes),
                            parent_id=current.node_id,
                            level_from_root=current.level_from_root + 1,
                        )
                    }
                )
                continue
            if owns_content and current.total_content_pointers:
                own_content = SemanticNode(
                    node_id=str(stable_id("workflow_ingest.page_index.owned_content", stable_part(current.node_id))),
                    parent_id=current.node_id,
                    node_type="HEADING_TEXT" if current.node_type == "HEADING" else current.node_type,
                    title=current.title,
                    summary=current.summary,
                    total_content_pointers=list(current.total_content_pointers),
                    level_from_root=current.level_from_root + 1,
                    metadata={"page_index_role": "owned_content"},
                )
                normalized[marker] = current.model_copy(
                    update={"total_content_pointers": [], "child_nodes": [own_content, child]}
                )
                continue
            if child.child_nodes:
                raise ValueError(
                    f"cannot losslessly normalize unary PageIndex subtree at {current.node_id!r}"
                )
            metadata = dict(current.metadata)
            metadata["atomic_retained"] = True
            metadata["unary_normalization"] = "atomic_retained"
            total_pointers = (
                list(child.total_content_pointers)
                or list(current.total_content_pointers)
                or list(current.aggregate_content_pointers)
                or list(child.aggregate_content_pointers)
            )
            metadata["atomic_retained_child"] = {
                "node_id": child.node_id,
                "title": child.title,
                "node_type": child.node_type,
            }
            normalized[marker] = current.model_copy(
                update={"total_content_pointers": total_pointers, "child_nodes": [], "metadata": metadata}
            )
            continue
        pending.append((current, True))
        for child in reversed(current.child_nodes):
            pending.append((child, False))
    return normalized[id(node)]


_stable_part = stable_part
_page_index_normalize_text = page_index_normalize_text
_page_index_find_best_fuzzy_span = page_index_find_best_fuzzy_span
_find_page_unit = find_page_unit
_resolve_pointer = resolve_pointer
_make_semantic_node = make_semantic_node
_materialize_block_tree = materialize_block_tree
_rebase_page_index_subtree = rebase_page_index_subtree
_normalize_page_index_unary_nodes = normalize_page_index_unary_nodes

__all__ = [
    "PageIndexTraceLogger",
    "find_page_unit",
    "make_semantic_node",
    "materialize_block_tree",
    "normalize_page_index_unary_nodes",
    "page_index_find_best_fuzzy_span",
    "page_index_normalize_text",
    "rebase_page_index_subtree",
    "resolve_pointer",
    "stable_part",
]
