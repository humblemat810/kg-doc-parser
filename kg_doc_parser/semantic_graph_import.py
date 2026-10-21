"""Typed decoding for legacy KGE graph payloads."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal
from uuid import UUID

if TYPE_CHECKING:
    from .semantic_document_splitting_layerwise_edits import (
        HydratedTextPointer,
        SemanticNode,
    )


def _legacy_object_list(value: object, *, label: str) -> list[Mapping[str, object]]:
    """Narrow a legacy JSON list before semantic reconstruction."""

    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise TypeError(f"legacy graph field {label!r} must be a list of objects")
    return [
        _legacy_mapping(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    ]


def _legacy_string_list(value: object, *, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"legacy graph field {label!r} must be a list of strings")
    return list(value)


def _legacy_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"legacy graph field {label!r} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"legacy graph field {label!r} must have string keys")
    return dict(value)


def _legacy_string(value: object, *, label: str, default: str | None = None) -> str:
    if value is None and default is not None:
        return default
    if not isinstance(value, str):
        raise TypeError(f"legacy graph field {label!r} must be a string")
    return value


def _legacy_int(value: object, *, label: str, default: int | None = None) -> int:
    if value is None and default is not None:
        return default
    if type(value) is not int:
        raise TypeError(f"legacy graph field {label!r} must be an integer")
    return value


def _legacy_node_type(
    value: object,
) -> Literal["DOCUMENT_ROOT", "TEXT_FLOW", "KEY_VALUE_PAIR", "TABLE"]:
    if value == "DOCUMENT_ROOT":
        return "DOCUMENT_ROOT"
    if value == "TEXT_FLOW":
        return "TEXT_FLOW"
    if value == "KEY_VALUE_PAIR":
        return "KEY_VALUE_PAIR"
    if value == "TABLE":
        return "TABLE"
    raise ValueError(f"unsupported legacy semantic node type: {value!r}")


def _extract_pointers_from_mentions(
    mentions: Sequence[Mapping[str, object]],
) -> list[HydratedTextPointer]:
    from .semantic_document_splitting_layerwise_edits import HydratedTextPointer

    if len(mentions) != 1:
        raise ValueError("legacy semantic nodes must contain exactly one mention group")
    spans = _legacy_object_list(mentions[0].get("spans"), label="mentions.spans")
    results: list[HydratedTextPointer] = []
    for span in spans:
        doc_page = _legacy_string(
            span.get("document_page_url"), label="document_page_url", default=""
        )
        source_cluster_id = doc_page.split("#", 1)[1] if "#" in doc_page else "p1_c0"
        if not source_cluster_id:
            source_cluster_id = "p1_c0"
        span_verification = span.get("verification")
        if span_verification is not None and not isinstance(span_verification, str):
            raise TypeError("legacy graph field 'verification' must be a string or null")
        results.append(
            HydratedTextPointer(
                source_cluster_id=source_cluster_id,
                start_char=_legacy_int(span.get("start_char"), label="start_char", default=0),
                end_char=(
                    -1
                    if span.get("end_char") == 10**9
                    else _legacy_int(span.get("end_char"), label="end_char", default=-1)
                ),
                verbatim_text=_legacy_string(
                    span.get("excerpt"), label="excerpt", default=""
                ),
                start_delimiter=None,
                end_delimiter=None,
                validation_method=span_verification,
            )
        )
    return results


def _extract_pointers_from_references(
    refs: Sequence[Mapping[str, object]],
) -> list[HydratedTextPointer]:
    from .semantic_document_splitting_layerwise_edits import HydratedTextPointer

    results: list[HydratedTextPointer] = []
    for reference in refs:
        doc_page = _legacy_string(
            reference.get("document_page_url"),
            label="document_page_url",
            default="",
        )
        source_cluster_id = doc_page.split("#", 1)[1] if "#" in doc_page else "p1_c0"
        if not source_cluster_id:
            source_cluster_id = "p1_c0"
        reference_mentions = _legacy_object_list(
            reference.get("mentions", []), label="references.mentions"
        )
        if len(reference_mentions) > 1:
            raise ValueError("legacy references support at most one mention group")
        verification = (
            reference_mentions[0].get("verification") if reference_mentions else None
        )
        if verification is not None and not isinstance(verification, str):
            raise TypeError("legacy graph field 'verification' must be a string or null")
        results.append(
            HydratedTextPointer(
                source_cluster_id=source_cluster_id,
                start_char=_legacy_int(
                    reference.get("start_char"), label="start_char", default=0
                ),
                end_char=(
                    -1
                    if reference.get("end_char") == 10**9
                    else _legacy_int(
                        reference.get("end_char"), label="end_char", default=-1
                    )
                ),
                verbatim_text=_legacy_string(
                    reference.get("snippet"), label="snippet", default=""
                ),
                start_delimiter=None,
                end_delimiter=None,
                validation_method=verification,
            )
        )
    return results


def kge_payload_to_semantic_tree(payload: Mapping[str, object]) -> SemanticNode:
    """Reconstruct and validate a semantic tree from legacy graph JSON."""

    from .semantic_document_splitting_layerwise_edits import SemanticNode
    from .workflow_ingest.semantics import validate_semantic_tree_integrity

    nodes_data = _legacy_object_list(payload.get("nodes", []), label="nodes")
    edges_data = _legacy_object_list(payload.get("edges", []), label="edges")
    sem_nodes: dict[str, SemanticNode] = {}
    for node_data in nodes_data:
        metadata = _legacy_mapping(node_data.get("metadata"), label="metadata")
        level_from_root = _legacy_int(
            metadata.get("level_from_root"), label="level_from_root"
        )
        mentions = _extract_pointers_from_mentions(
            _legacy_object_list(node_data.get("mentions"), label="mentions")
        )
        node_id = _legacy_string(node_data.get("id"), label="id")
        parent_id_raw = metadata.get("parent_id")
        parent_id = (
            UUID(_legacy_string(parent_id_raw, label="parent_id"))
            if parent_id_raw
            else None
        )
        semantic_node = SemanticNode(
            node_id=UUID(node_id),
            parent_id=parent_id,
            node_type=_legacy_node_type(metadata.get("semantic_node_type", "TEXT_FLOW")),
            title=_legacy_string(node_data.get("label"), label="label", default=""),
            total_content_pointers=mentions,
            child_nodes=[],
            level_from_root=level_from_root,
        )
        sem_nodes[node_id] = semantic_node

    children_by_parent: dict[str, list[str]] = defaultdict(list)
    for edge in edges_data:
        relation = edge.get("relation") or edge.get("predicate")
        if relation != "HAS_CHILD":
            continue
        if "source_ids" in edge or "target_ids" in edge:
            source_ids = _legacy_string_list(edge.get("source_ids", []), label="source_ids")
            target_ids = _legacy_string_list(edge.get("target_ids", []), label="target_ids")
            if not source_ids or not target_ids:
                continue
            parent_id, child_id = source_ids[0], target_ids[0]
        else:
            parent_id_value = edge.get("subject_id")
            child_id_value = edge.get("object_id")
            parent_id = (
                _legacy_string(parent_id_value, label="subject_id")
                if parent_id_value is not None
                else None
            )
            child_id = (
                _legacy_string(child_id_value, label="object_id")
                if child_id_value is not None
                else None
            )
            if not parent_id or not child_id:
                continue
        children_by_parent[parent_id].append(child_id)

    for parent_id, child_ids in children_by_parent.items():
        parent = sem_nodes.get(parent_id)
        if parent is None:
            continue
        for child_id in child_ids:
            child = sem_nodes.get(child_id)
            if child is not None:
                parent.child_nodes.append(child)

    root_candidates = [node for node in sem_nodes.values() if node.parent_id is None]
    if root_candidates:
        root = root_candidates[0]
    else:
        all_child_ids = {child_id for ids in children_by_parent.values() for child_id in ids}
        root_ids = list(set(sem_nodes) - all_child_ids)
        if root_ids:
            root = sem_nodes[root_ids[0]]
        elif sem_nodes:
            root = next(iter(sem_nodes.values()))
        else:
            raise IndexError("list index out of range")
    validate_semantic_tree_integrity(root)
    return root
