"""Compatibility adapters for legacy whole-document parser callbacks."""

from __future__ import annotations

import re
from collections.abc import Mapping

from .semantics import HydratedTextPointer, SemanticNode
from .serialization import JsonObject, ModelDumpLike

_LEGACY_POINTER_ID_RE = re.compile(r"^p(?P<page>\d+)_c(?P<cluster>\d+)$")
ParserSourceMap = dict[str, JsonObject]


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

    def canonicalize_source_cluster_id(source_cluster_id: str) -> str:
        if _LEGACY_POINTER_ID_RE.match(source_cluster_id) is None:
            return source_cluster_id
        return aliases.get(source_cluster_id, source_cluster_id)

    def remap_pointer(pointer: HydratedTextPointer) -> HydratedTextPointer:
        canonical_id = canonicalize_source_cluster_id(pointer.source_cluster_id)
        if canonical_id == pointer.source_cluster_id:
            return pointer
        return pointer.model_copy(update={"source_cluster_id": canonical_id})

    def remap_node(node: SemanticNode) -> SemanticNode:
        return node.model_copy(
            update={
                "total_content_pointers": [
                    remap_pointer(pointer) for pointer in node.total_content_pointers
                ],
                "aggregate_content_pointers": [
                    remap_pointer(pointer) for pointer in node.aggregate_content_pointers
                ],
                "child_nodes": [remap_node(child) for child in node.child_nodes],
            }
        )

    return remap_node(tree)
