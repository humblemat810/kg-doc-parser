from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Literal, Protocol, cast, runtime_checkable

from kogwistar.id_provider import stable_id
from pydantic import BaseModel, Field, model_validator

from .serialization import JsonObject, JsonValue, json_safe


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _source_text(source: object) -> str:
    if not isinstance(source, Mapping):
        return ""
    value = source.get("text", "")
    return value if isinstance(value, str) else ""


def _span_bounds(
    text: str,
    start_char: object,
    end_char: object,
) -> tuple[int, int] | None:
    """Return validated inclusive/exclusive bounds without coercing input."""
    if type(start_char) is not int or type(end_char) is not int:
        return None
    end_exclusive = len(text) if end_char == -1 else end_char + 1
    if not 0 <= start_char < end_exclusive <= len(text):
        return None
    return start_char, end_exclusive


class HydratedTextPointer(BaseModel):
    source_cluster_id: str = Field(strict=True)
    start_char: int = Field(strict=True)
    end_char: int = Field(strict=True)
    verbatim_text: str = Field(strict=True)


class TextPointerLike(Protocol):
    @property
    def source_cluster_id(self) -> str: ...

    @property
    def start_char(self) -> int: ...

    @property
    def end_char(self) -> int: ...

    @property
    def verbatim_text(self) -> str | None: ...


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


@runtime_checkable
class SemanticTreeNodeLike(Protocol):
    """Minimum shape required by the shared tree-branching validator."""

    @property
    def node_id(self) -> object: ...

    @property
    def child_nodes(self) -> Sequence[object]: ...


@runtime_checkable
class SemanticIntegrityNodeLike(Protocol):
    """Fields shared by current and legacy semantic-node models."""

    @property
    def node_id(self) -> object: ...

    @property
    def parent_id(self) -> object: ...

    @property
    def child_nodes(self) -> Sequence[object]: ...

    @property
    def total_content_pointers(self) -> Sequence[TextPointerLike]: ...


@runtime_checkable
class _MetadataNodeLike(Protocol):
    @property
    def metadata(self) -> object: ...


@runtime_checkable
class _AggregatePointerNodeLike(Protocol):
    @property
    def aggregate_content_pointers(self) -> Sequence[TextPointerLike]: ...


def _attribute_sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _node_id_for_shape(node: object) -> str:
    value = node.node_id if isinstance(node, SemanticTreeNodeLike) else None
    return "<missing>" if value is None else str(value)


def _node_children_for_shape(node: object) -> tuple[object, ...]:
    if not isinstance(node, SemanticTreeNodeLike):
        return ()
    return _attribute_sequence(node.child_nodes)


def _node_metadata_for_shape(node: object) -> Mapping[str, object]:
    value = node.metadata if isinstance(node, _MetadataNodeLike) else {}
    return value if isinstance(value, Mapping) else {}


def _node_pointers_for_shape(node: object, attribute: str) -> tuple[TextPointerLike, ...]:
    if attribute == "total_content_pointers":
        values = (
            _attribute_sequence(node.total_content_pointers)
            if isinstance(node, SemanticIntegrityNodeLike)
            else ()
        )
    elif attribute == "aggregate_content_pointers":
        values = (
            _attribute_sequence(node.aggregate_content_pointers)
            if isinstance(node, _AggregatePointerNodeLike)
            else ()
        )
    else:
        values = ()
    return cast(tuple[TextPointerLike, ...], values)


def validate_semantic_tree_shape(
    root: SemanticTreeNodeLike,
    *,
    max_depth: int | None = None,
) -> None:
    """Reject unary semantic nodes without relying on Python recursion.

    A zero-child node is either a valid leaf or an explicitly retained atomic
    unit.  A node with one child is never a meaningful refinement.  This
    helper intentionally accepts the small structural protocol so the legacy
    parser can use the same invariant without importing the newer node model.
    """

    pending: list[tuple[SemanticTreeNodeLike, int, bool]] = [(root, 0, False)]
    active_object_ids: set[int] = set()
    while pending:
        node, depth, exiting = pending.pop()
        object_id = id(node)
        if exiting:
            active_object_ids.remove(object_id)
            continue
        if object_id in active_object_ids:
            raise ValueError("semantic tree contains a cycle while checking branching")
        active_object_ids.add(object_id)
        node_id = _node_id_for_shape(node)
        children = _node_children_for_shape(node)
        if len(children) == 1:
            raise ValueError(f"unary semantic node: {node_id}")
        if max_depth is not None and depth > max_depth:
            raise ValueError(f"semantic tree exceeds maximum depth {max_depth}: {node_id}")
        metadata = _node_metadata_for_shape(node)
        total_pointers = _node_pointers_for_shape(node, "total_content_pointers")
        aggregate_pointers = _node_pointers_for_shape(node, "aggregate_content_pointers")
        if (
            isinstance(metadata, Mapping)
            and metadata.get("atomic_retained") is True
            and not (total_pointers or aggregate_pointers)
        ):
            raise ValueError(
                f"atomic semantic node {node_id!r} has no verified source pointer"
            )
        pending.append((node, depth, True))
        for child in reversed(children):
            pending.append((cast(SemanticTreeNodeLike, child), depth + 1, False))


def hydrate_pointer_from_offsets(
    pointer: HydratedTextPointer,
    source_map: Mapping[str, Mapping[str, object]],
) -> HydratedTextPointer | None:
    """Hydrate a provider pointer from authoritative source offsets.

    This is intentionally separate from repair.  LLM review payloads may carry
    a lossy transcription, but valid offsets still identify the authoritative
    source span.
    """

    source = source_map.get(pointer.source_cluster_id)
    if source is None:
        return None
    text = _source_text(source)
    bounds = _span_bounds(text, pointer.start_char, pointer.end_char)
    if bounds is None:
        return None
    start_char, end_exclusive = bounds
    return pointer.model_copy(
        update={
            "end_char": end_exclusive - 1,
            "verbatim_text": text[start_char:end_exclusive],
        }
    )


def correct_and_validate_pointer(
    pointer: HydratedTextPointer,
    source_map: Mapping[str, Mapping[str, object]],
) -> HydratedTextPointer | None:
    """Repair a pointer only when the final evidence is source-authoritative.

    A valid range is authoritative even when the provider's transcription
    differs: the returned text is rehydrated from that source slice. Invalid
    ranges may be relocated only through one exact, unambiguous text
    occurrence. All successful results are rehydrated from the source slice.
    """

    source = source_map.get(pointer.source_cluster_id)
    if source is None:
        return None
    text = _source_text(source)
    bounds = _span_bounds(text, pointer.start_char, pointer.end_char)
    if bounds is not None:
        start_char, end_exclusive = bounds
        actual = text[start_char:end_exclusive]
        return pointer.model_copy(
            update={
                "end_char": end_exclusive - 1,
                "verbatim_text": actual,
            }
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
    if len(occurrences) != 1:
        return None
    best_start, best_end = occurrences[0]
    return HydratedTextPointer(
        source_cluster_id=pointer.source_cluster_id,
        start_char=best_start,
        end_char=best_end,
        verbatim_text=text[best_start:best_end + 1],
    )


def pointer_source_validation_error(
    pointer: TextPointerLike,
    source_map: Mapping[str, Mapping[str, object]],
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
    text = _source_text(source)
    if type(pointer.start_char) is not int or type(pointer.end_char) is not int:
        return "pointer bounds must be integers"
    if pointer.start_char < 0:
        return "pointer start is negative"
    end = len(text) - 1 if pointer.end_char == -1 else pointer.end_char
    if end < pointer.start_char:
        return "pointer range is reversed"
    if end >= len(text):
        return "pointer exceeds source bounds"
    actual = text[pointer.start_char : end + 1]
    if actual != (pointer.verbatim_text or ""):
        return "pointer excerpt does not match authoritative source slice"
    return None


def validate_semantic_tree_integrity(
    root: SemanticIntegrityNodeLike,
    *,
    parser_source_map: Mapping[str, Mapping[str, object]] | None = None,
    max_depth: int | None = None,
    validate_shape: bool = True,
) -> None:
    """Validate identity, ancestry, grounding, and branching in one pass.

    ``SemanticNode`` is intentionally constructible while a workflow is being
    assembled.  This validator is the finalized-tree boundary used before
    export, persistence, or a stable DTO result is returned.
    """

    pending: list[tuple[SemanticIntegrityNodeLike, str | None, int, bool]] = [
        (root, None, 0, False)
    ]
    seen_ids: set[str] = set()
    active_ids: set[str] = set()
    while pending:
        node, expected_parent_id, depth, exiting = pending.pop()
        node_id = str(node.node_id or "")
        if not node_id:
            raise ValueError("semantic tree contains a node without an id")
        if exiting:
            active_ids.remove(node_id)
            continue
        if node_id in active_ids:
            raise ValueError(f"semantic tree contains a cycle at {node_id!r}")
        if node_id in seen_ids:
            raise ValueError(f"semantic tree contains duplicate node id {node_id!r}")
        actual_parent_id = None if node.parent_id is None else str(node.parent_id)
        if actual_parent_id != expected_parent_id:
            raise ValueError(
                f"semantic tree parent mismatch for {node_id!r}: "
                f"expected {expected_parent_id!r}, got {actual_parent_id!r}"
            )
        if max_depth is not None and depth > max_depth:
            raise ValueError(f"semantic tree exceeds maximum depth {max_depth}: {node_id}")
        aggregate_pointers = _node_pointers_for_shape(node, "aggregate_content_pointers")
        pointers = [*node.total_content_pointers, *aggregate_pointers]
        for pointer in pointers:
            if pointer.start_char < 0 or pointer.end_char < -1:
                raise ValueError(f"invalid pointer bounds for node {node_id!r}")
            if pointer.end_char != -1 and pointer.end_char < pointer.start_char:
                raise ValueError(f"reversed pointer bounds for node {node_id!r}")
            if parser_source_map is not None:
                pointer_error = pointer_source_validation_error(pointer, parser_source_map)
                if pointer_error is not None:
                    raise ValueError(f"node {node_id!r}: {pointer_error}")
        seen_ids.add(node_id)
        active_ids.add(node_id)
        pending.append((node, expected_parent_id, depth, True))
        for child in reversed(node.child_nodes):
            pending.append((cast(SemanticIntegrityNodeLike, child), node_id, depth + 1, False))
    if validate_shape:
        validate_semantic_tree_shape(root, max_depth=max_depth)


def compute_pointer_coverage(
    root_node: SemanticNode,
    source_map: Mapping[str, Mapping[str, object]],
) -> dict[str, JsonValue]:
    def _meaningful_length(value: str) -> int:
        return sum(1 for char in value if not char.isspace())

    def _meaningful_slice_length(value: str, start: int, end: int) -> int:
        return _meaningful_length(value[max(0, start) : min(len(value), end + 1)])

    ranges: dict[str, list[tuple[int, int]]] = {}

    pending = [root_node]
    while pending:
        node = pending.pop()
        if node.node_type != "DOCUMENT_ROOT":
            for ptr in node.total_content_pointers:
                if pointer_source_validation_error(ptr, source_map) is not None:
                    continue
                end = ptr.end_char
                if end == -1:
                    end = max(
                        0,
                        len(_source_text(source_map.get(ptr.source_cluster_id))) - 1,
                    )
                ranges.setdefault(ptr.source_cluster_id, []).append((ptr.start_char, end))
        pending.extend(reversed(node.child_nodes))
    per_cluster: dict[str, float] = {}
    total_len = 0
    total_covered = 0
    # Include every semantic source cluster in the denominator, including a
    # cluster with no pointers.  Otherwise an omitted document unit can make
    # an incomplete tree look fully covered.
    for cluster_id, record in source_map.items():
        text = _source_text(record)
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
    return cast(dict[str, JsonValue], {"per_cluster": per_cluster, "overall": overall})


def compute_terminal_content_coverage(
    root_node: SemanticNode,
    source_map: Mapping[str, Mapping[str, object]],
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

    pending = [root_node]
    while pending:
        node = pending.pop()
        if not node.child_nodes:
            node_id = str(node.node_id or "<missing-node-id>")
            for pointer in node.total_content_pointers:
                error = pointer_source_validation_error(pointer, source_map)
                if error is not None:
                    invalid_pointers.append({"node_id": node_id, "error": error})
                    continue
                source_text = _source_text(source_map[pointer.source_cluster_id])
                end = len(source_text) - 1 if pointer.end_char == -1 else pointer.end_char
                owners = ownership.setdefault(pointer.source_cluster_id, {})
                for position in range(pointer.start_char, end + 1):
                    if not source_text[position].isspace():
                        owners.setdefault(position, []).append(node_id)
        pending.extend(reversed(node.child_nodes))
    per_cluster: dict[str, float] = {}
    missing_ranges: dict[str, list[dict[str, int]]] = {}
    multiply_owned_ranges: dict[str, list[dict[str, int]]] = {}
    total_nonws = 0
    covered_nonws = 0

    for cluster_id, record in source_map.items():
        text = _source_text(record)
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

    return cast(
        dict[str, JsonValue],
        {
            "total_nonws": total_nonws,
            "covered_nonws": covered_nonws,
            "missing_nonws_ranges": missing_ranges,
            "multiply_owned_nonws_ranges": multiply_owned_ranges,
            "per_cluster": per_cluster,
            "overall": covered_nonws / total_nonws if total_nonws else 1.0,
            "coverage_basis": "terminal_content_owners_exactly_once",
            "invalid_pointers": invalid_pointers,
            "valid": not invalid_pointers and not missing_ranges and not multiply_owned_ranges,
        },
    )


def classify_terminal_coverage_status(
    root_node: SemanticNode,
    coverage: dict[str, JsonValue],
) -> Literal["complete", "atomic_valid", "partial_degraded", "failed"]:
    """Convert terminal ownership evidence into a truthful operator status."""
    if not coverage.get("valid", False):
        covered = coverage.get("covered_nonws", 0)
        has_coverage = isinstance(covered, (int, float)) and not isinstance(covered, bool) and covered > 0
        return "partial_degraded" if has_coverage else "failed"
    pending = [root_node]
    while pending:
        node = pending.pop()
        if node.metadata.get("atomic_retained") is True:
            return "atomic_valid"
        pending.extend(node.child_nodes)
    return "complete"


def semantic_tree_to_kge_payload(
    root: SemanticNode,
    *,
    doc_id: str,
    parser_source_map: Mapping[str, Mapping[str, object]] | None = None,
    source_revision: str | None = None,
) -> dict[str, JsonValue]:
    # Graph export is a persistence boundary.  Validate before serializing so
    # callers cannot bypass the workflow commit gate with a hand-built tree.
    pending_nodes = [root]
    has_source_pointers = False
    while pending_nodes:
        node = pending_nodes.pop()
        if node.total_content_pointers or node.aggregate_content_pointers:
            has_source_pointers = True
            break
        pending_nodes.extend(node.child_nodes)
    if parser_source_map is None and has_source_pointers:
        raise ValueError("source-grounded graph export requires parser_source_map")
    validate_semantic_tree_integrity(root, parser_source_map=parser_source_map)
    nodes: list[JsonObject] = []
    edges: list[JsonObject] = []

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
                "end_char": (
                    p.start_char + len(p.verbatim_text)
                    if p.end_char == -1
                    else p.end_char + 1
                ),
                "excerpt": p.verbatim_text,
                "context_before": "",
                "context_after": "",
                "source_cluster_id": p.source_cluster_id,
                "verification": None,
            }
            for p in ptrs
        ]

    pending = [root]
    while pending:
        node = pending.pop()
        is_root = node.node_id == root.node_id
        aggregate_spans = spans(node.aggregate_content_pointers)
        # A structural page-index container is grounded by its aggregate span;
        # ordinary legacy structural nodes retain the existing synthetic fallback.
        mentions = [{"spans": spans(node.total_content_pointers or node.aggregate_content_pointers)}]
        raw_node_metadata = {
            "semantic_node_type": node.node_type,
            "doc_id": doc_id,
            "parent_id": node.parent_id,
            "level_from_root": node.level_from_root,
            **dict(node.metadata or {}),
        }
        node_metadata_value = json_safe(raw_node_metadata)
        if not isinstance(node_metadata_value, dict):
            raise TypeError("semantic node metadata must serialize to a JSON object")
        node_metadata: JsonObject = node_metadata_value
        if is_root and source_revision is not None:
            node_metadata["source_revision"] = source_revision
        if not node.summary and "summary_unavailable" not in node_metadata:
            # Do not silently turn a title into an LLM summary.  Consumers can
            # decide whether to request a summary or display the title only.
            node_metadata["summary_unavailable"] = True
        nodes.append(
            {
                "id": node.node_id,
                "label": node.title,
                "type": "entity",
                "doc_id": doc_id,
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
                    "doc_id": doc_id,
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
        pending.extend(reversed(node.child_nodes))
    return cast(
        dict[str, JsonValue],
        {
            "doc_id": doc_id,
            "insertion_method": "workflow_ingest",
            "source_revision": source_revision,
            "nodes": nodes,
            "edges": edges,
        },
    )
