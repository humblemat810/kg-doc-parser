from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


class HydratedTextPointer(BaseModel):
    source_cluster_id: str
    start_char: int
    end_char: int
    verbatim_text: str


class SemanticNode(BaseModel):
    node_id: str = Field(default_factory=lambda: str(uuid4()))
    parent_id: str | None = None
    node_type: str = "TEXT_FLOW"
    title: str
    total_content_pointers: list[HydratedTextPointer] = Field(default_factory=list)
    child_nodes: list["SemanticNode"] = Field(default_factory=list)
    level_from_root: int = 0


SemanticNode.model_rebuild()


def correct_and_validate_pointer(
    pointer: HydratedTextPointer,
    source_map: dict[str, dict[str, Any]],
) -> HydratedTextPointer | None:
    source = source_map.get(pointer.source_cluster_id)
    if source is None:
        return None
    text = source.get("text", "")
    end_exclusive = len(text) if pointer.end_char == -1 else pointer.end_char + 1
    actual = text[max(pointer.start_char, 0):max(end_exclusive, 0)]
    if _normalize_text(actual) == _normalize_text(pointer.verbatim_text):
        return pointer
    if actual and (
        _normalize_text(actual) in _normalize_text(pointer.verbatim_text)
        or _normalize_text(pointer.verbatim_text) in _normalize_text(actual)
    ):
        return HydratedTextPointer(
            source_cluster_id=pointer.source_cluster_id,
            start_char=max(pointer.start_char, 0),
            end_char=max(end_exclusive - 1, -1),
            verbatim_text=actual,
        )

    occurrences = []
    start = 0
    needle = pointer.verbatim_text or ""
    while needle:
        idx = text.find(needle, start)
        if idx == -1:
            break
        occurrences.append((idx, idx + len(needle) - 1))
        start = idx + max(1, len(needle))
    if not occurrences:
        return None
    best_start, best_end = min(occurrences, key=lambda item: abs(item[0] - pointer.start_char))
    return HydratedTextPointer(
        source_cluster_id=pointer.source_cluster_id,
        start_char=best_start,
        end_char=best_end,
        verbatim_text=text[best_start:best_end + 1],
    )


def compute_pointer_coverage(
    root_node: SemanticNode,
    source_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ranges: dict[str, list[tuple[int, int]]] = {}

    def walk(node: SemanticNode) -> None:
        if node.node_type != "DOCUMENT_ROOT":
            for ptr in node.total_content_pointers:
                end = ptr.end_char
                if end == -1:
                    end = max(0, len(source_map.get(ptr.source_cluster_id, {}).get("text", "")) - 1)
                ranges.setdefault(ptr.source_cluster_id, []).append((ptr.start_char, end))
        for child in node.child_nodes:
            walk(child)

    walk(root_node)
    per_cluster: dict[str, float] = {}
    total_len = 0
    total_covered = 0
    for cluster_id, cluster_ranges in ranges.items():
        text = source_map.get(cluster_id, {}).get("text", "")
        if not text:
            continue
        cluster_ranges.sort()
        merged: list[tuple[int, int]] = []
        cur_s, cur_e = cluster_ranges[0]
        for s, e in cluster_ranges[1:]:
            if s <= cur_e + 1:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        covered = sum((e - s + 1) for s, e in merged)
        per_cluster[cluster_id] = covered / len(text)
        total_len += len(text)
        total_covered += covered
    overall = total_covered / total_len if total_len else 1.0
    return {"per_cluster": per_cluster, "overall": overall}


def semantic_tree_to_kge_payload(root: SemanticNode, *, doc_id: str) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    def spans(ptrs: list[HydratedTextPointer]) -> list[dict[str, Any]]:
        if not ptrs:
            return [
                {
                    "doc_id": doc_id,
                    "collection_page_url": f"doc://{doc_id}",
                    "document_page_url": f"doc://{doc_id}#synthetic",
                    "insertion_method": "workflow_ingest",
                    "page_number": 1,
                    "start_char": 0,
                    "end_char": 1,
                    "excerpt": " ",
                    "context_before": "",
                    "context_after": "",
                    "chunk_id": None,
                    "source_cluster_id": None,
                    "verification": None,
                }
            ]
        return [
            {
                "doc_id": doc_id,
                "collection_page_url": f"doc://{doc_id}",
                "document_page_url": f"doc://{doc_id}#{p.source_cluster_id}",
                "insertion_method": "workflow_ingest",
                "page_number": 1,
                "start_char": p.start_char,
                "end_char": p.end_char + 1,
                "excerpt": p.verbatim_text,
                "context_before": "",
                "context_after": "",
                "source_cluster_id": p.source_cluster_id,
                "verification": None,
            }
            for p in ptrs
        ]

    def walk(node: SemanticNode) -> None:
        nodes.append(
            {
                "id": node.node_id,
                "label": node.title,
                "type": "entity",
                "summary": node.title,
                "metadata": {
                    "semantic_node_type": node.node_type,
                    "doc_id": doc_id,
                    "parent_id": node.parent_id,
                    "level_from_root": node.level_from_root,
                },
                "mentions": [{"spans": spans(node.total_content_pointers)}],
            }
        )
        for child in node.child_nodes:
            edges.append(
                {
                    "id": str(uuid4()),
                    "label": "parent-child",
                    "type": "relationship",
                    "summary": f"{node.node_id}->{child.node_id}",
                    "relation": "HAS_CHILD",
                    "source_ids": [node.node_id],
                    "target_ids": [child.node_id],
                    "source_edge_ids": [],
                    "target_edge_ids": [],
                    "mentions": [
                        {
                            "spans": spans(
                                child.total_content_pointers or node.total_content_pointers
                            )
                        }
                    ],
                    "metadata": {"doc_id": doc_id, "insertion_method": "workflow_ingest"},
                }
            )
            walk(child)

    walk(root)
    return {"doc_id": doc_id, "insertion_method": "workflow_ingest", "nodes": nodes, "edges": edges}
