from __future__ import annotations

import re
from typing import Literal

from kogwistar.id_provider import stable_id
from kogwistar.json_types import JsonValue
from pydantic import BaseModel, Field, model_validator


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


class HydratedTextPointer(BaseModel):
    source_cluster_id: str
    start_char: int
    end_char: int
    verbatim_text: str


class SemanticNode(BaseModel):
    node_id: str | None = None
    parent_id: str | None = None
    node_type: str = "TEXT_FLOW"
    title: str
    summary: str = ""
    # Ownership spans belong to this node's own leaf content.  Aggregate spans
    # describe the structural section represented by a container node.
    total_content_pointers: list[HydratedTextPointer] = Field(default_factory=list)
    aggregate_content_pointers: list[HydratedTextPointer] = Field(default_factory=list)
    child_nodes: list[SemanticNode] = Field(default_factory=list)
    level_from_root: int = 0
    # Pydantic's recursive alias expansion is not stable across the supported
    # runtimes; keep the model field opaque while JSON boundaries stay typed.
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ensure_stable_node_id(self) -> SemanticNode:
        if self.node_id:
            return self
        pointer_fp = "|".join(
            f"{ptr.source_cluster_id}:{ptr.start_char}:{ptr.end_char}:{ptr.verbatim_text}"
            for ptr in self.total_content_pointers
        )
        self.node_id = str(
            stable_id(
                "workflow_ingest.semantic_node",
                str(self.parent_id or "root"),
                str(self.node_type),
                str(self.title),
                str(self.level_from_root),
                pointer_fp,
            )
        )
        return self


SemanticNode.model_rebuild()


def correct_and_validate_pointer(
    pointer: HydratedTextPointer,
    source_map: dict[str, dict[str, JsonValue]],
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


def pointer_source_validation_error(
    pointer: HydratedTextPointer,
    source_map: dict[str, dict[str, JsonValue]],
) -> str | None:
    """Return a deterministic error when a pointer is not source-grounded.

    Repair may use normalized or fuzzy matching while locating a pointer. Final
    validation is stricter: the persisted excerpt must agree with the exact
    authoritative source slice after the parser's documented whitespace
    normalization.
    """

    source = source_map.get(pointer.source_cluster_id)
    if source is None:
        return f"unknown source cluster {pointer.source_cluster_id!r}"
    text = str(source.get("text", "") or "")
    if pointer.start_char < 0:
        return "pointer start is negative"
    end = len(text) - 1 if pointer.end_char == -1 else pointer.end_char
    if end < pointer.start_char:
        return "pointer range is reversed"
    if end >= len(text):
        return "pointer exceeds source bounds"
    actual = text[pointer.start_char : end + 1]
    if _normalize_text(actual) != _normalize_text(pointer.verbatim_text):
        return "pointer excerpt does not match authoritative source slice"
    return None


def compute_pointer_coverage(
    root_node: SemanticNode,
    source_map: dict[str, dict[str, JsonValue]],
) -> dict[str, JsonValue]:
    def _meaningful_length(value: str) -> int:
        return sum(1 for char in value if not char.isspace())

    def _meaningful_slice_length(value: str, start: int, end: int) -> int:
        return _meaningful_length(value[max(0, start) : min(len(value), end + 1)])

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
    # Include every semantic source cluster in the denominator, including a
    # cluster with no pointers.  Otherwise an omitted document unit can make
    # an incomplete tree look fully covered.
    for cluster_id, record in source_map.items():
        text = str(record.get("text", "") or "")
        meaningful_total = _meaningful_length(text)
        if meaningful_total == 0:
            per_cluster[cluster_id] = 1.0
            continue
        cluster_ranges = ranges.get(cluster_id, [])
        if not cluster_ranges:
            per_cluster[cluster_id] = 0.0
            total_len += meaningful_total
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
        covered = sum(_meaningful_slice_length(text, s, e) for s, e in merged)
        per_cluster[cluster_id] = covered / meaningful_total
        total_len += meaningful_total
        total_covered += covered
    overall = total_covered / total_len if total_len else 1.0
    return {"per_cluster": per_cluster, "overall": overall}


def compute_terminal_content_coverage(
    root_node: SemanticNode,
    source_map: dict[str, dict[str, JsonValue]],
) -> dict[str, JsonValue]:
    """Measure exact ownership by terminal content nodes.

    This intentionally does not count document/page wrappers, aggregate
    pointers, or ancestor extents. It is the release-gate metric; the older
    ``compute_pointer_coverage`` remains available for compatibility.
    """

    def _ranges(values: list[int]) -> list[dict[str, int]]:
        if not values:
            return []
        result: list[dict[str, int]] = []
        start = previous = values[0]
        for value in values[1:]:
            if value != previous + 1:
                result.append({"start": start, "end": previous})
                start = value
            previous = value
        result.append({"start": start, "end": previous})
        return result

    ownership: dict[str, dict[int, list[str]]] = {}
    invalid_pointers: list[dict[str, str]] = []

    def walk(node: SemanticNode) -> None:
        if not node.child_nodes:
            node_id = str(node.node_id or "<missing-node-id>")
            for pointer in node.total_content_pointers:
                error = pointer_source_validation_error(pointer, source_map)
                if error is not None:
                    invalid_pointers.append({"node_id": node_id, "error": error})
                    continue
                source_text = str(source_map[pointer.source_cluster_id].get("text", "") or "")
                end = len(source_text) - 1 if pointer.end_char == -1 else pointer.end_char
                owners = ownership.setdefault(pointer.source_cluster_id, {})
                for position in range(pointer.start_char, end + 1):
                    if not source_text[position].isspace():
                        owners.setdefault(position, []).append(node_id)
        for child in node.child_nodes:
            walk(child)

    walk(root_node)
    per_cluster: dict[str, float] = {}
    missing_ranges: dict[str, list[dict[str, int]]] = {}
    multiply_owned_ranges: dict[str, list[dict[str, int]]] = {}
    total_nonws = 0
    covered_nonws = 0

    for cluster_id, record in source_map.items():
        text = str(record.get("text", "") or "")
        meaningful_positions = [index for index, char in enumerate(text) if not char.isspace()]
        total_nonws += len(meaningful_positions)
        owners = ownership.get(cluster_id, {})
        missing = [index for index in meaningful_positions if not owners.get(index)]
        multiply_owned = [index for index in meaningful_positions if len(owners.get(index, [])) > 1]
        covered = len(meaningful_positions) - len(missing)
        covered_nonws += covered
        per_cluster[cluster_id] = covered / len(meaningful_positions) if meaningful_positions else 1.0
        if missing:
            missing_ranges[cluster_id] = _ranges(missing)
        if multiply_owned:
            multiply_owned_ranges[cluster_id] = _ranges(multiply_owned)

    return {
        "total_nonws": total_nonws,
        "covered_nonws": covered_nonws,
        "missing_nonws_ranges": missing_ranges,
        "multiply_owned_nonws_ranges": multiply_owned_ranges,
        "per_cluster": per_cluster,
        "overall": covered_nonws / total_nonws if total_nonws else 1.0,
        "coverage_basis": "terminal_content_owners_exactly_once",
        "invalid_pointers": invalid_pointers,
        "valid": not invalid_pointers and not missing_ranges and not multiply_owned_ranges,
    }


def classify_terminal_coverage_status(
    root_node: SemanticNode,
    coverage: dict[str, JsonValue],
) -> Literal["complete", "atomic_valid", "partial_degraded", "failed"]:
    """Convert terminal ownership evidence into a truthful operator status."""
    if not coverage.get("valid", False):
        return "partial_degraded" if coverage.get("covered_nonws", 0) > 0 else "failed"
    pending = [root_node]
    while pending:
        node = pending.pop()
        if node.metadata.get("atomic_retained"):
            return "atomic_valid"
        pending.extend(node.child_nodes)
    return "complete"


def semantic_tree_to_kge_payload(root: SemanticNode, *, doc_id: str) -> dict[str, JsonValue]:
    nodes: list[dict[str, JsonValue]] = []
    edges: list[dict[str, JsonValue]] = []

    def spans(ptrs: list[HydratedTextPointer]) -> list[dict[str, JsonValue]]:
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
                "end_char": max(p.end_char + 1, p.start_char + max(len(p.verbatim_text), 1)),
                "excerpt": p.verbatim_text,
                "context_before": "",
                "context_after": "",
                "source_cluster_id": p.source_cluster_id,
                "verification": None,
            }
            for p in ptrs
        ]

    def walk(node: SemanticNode) -> None:
        aggregate_spans = spans(node.aggregate_content_pointers)
        # A structural page-index container is grounded by its aggregate span;
        # ordinary legacy structural nodes retain the existing synthetic fallback.
        mentions = [{"spans": spans(node.total_content_pointers or node.aggregate_content_pointers)}]
        node_metadata = {
            "semantic_node_type": node.node_type,
            "doc_id": doc_id,
            "parent_id": node.parent_id,
            "level_from_root": node.level_from_root,
            **dict(node.metadata or {}),
        }
        if not node.summary and "summary_unavailable" not in node_metadata:
            # Do not silently turn a title into an LLM summary.  Consumers can
            # decide whether to request a summary or display the title only.
            node_metadata["summary_unavailable"] = True
        nodes.append(
            {
                "id": node.node_id,
                "label": node.title,
                "type": "entity",
                "summary": node.summary,
                "metadata": node_metadata,
                "mentions": mentions,
                "aggregate_mentions": [{"spans": aggregate_spans}] if aggregate_spans else [],
            }
        )
        for child in node.child_nodes:
            edges.append(
                {
                    "id": str(
                        stable_id(
                            "workflow_ingest.edge",
                            "HAS_CHILD",
                            str(node.node_id),
                            str(child.node_id),
                            str(doc_id),
                        )
                    ),
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
                                child.total_content_pointers
                                or child.aggregate_content_pointers
                                or node.total_content_pointers
                                or node.aggregate_content_pointers
                            )
                        }
                    ],
                    "metadata": {"doc_id": doc_id, "insertion_method": "workflow_ingest"},
                }
            )
            walk(child)

    walk(root)
    return {"doc_id": doc_id, "insertion_method": "workflow_ingest", "nodes": nodes, "edges": edges}
