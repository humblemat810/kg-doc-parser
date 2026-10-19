from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import cast

from .models import (
    CurrentLayerContext,
    CurrentLayerResult,
    LayerChildCandidate,
    LayerFrontierItem,
    ParseSessionState,
)
from .parser_core_commit import find_semantic_node
from .parser_core_contracts import (
    ParserPayload,
    ParserSourceMap,
    ParseSemanticFn,
    SourceCollectionLike,
    SplitStrategy,
    required_node_id,
)
from .parser_core_legacy import (
    _canonicalize_legacy_pointer_tree,
    _coerce_semantic_tree,
    _root_only,
    _source_text,
)
from .semantics import HydratedTextPointer, SemanticNode


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


def assert_source_map_matches_session(
    *,
    parse_session: ParseSessionState,
    parser_source_map: ParserSourceMap,
) -> None:
    expected = parse_session.metadata.get("source_map_fingerprint")
    if expected is None:
        # Sessions created before source binding remain readable.
        return
    if type(expected) is not str or expected != source_map_fingerprint(parser_source_map):
        raise ValueError("parser source map does not match the parse session source revision")


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
        root_id = required_node_id(root)
        session = ParseSessionState(
            collection_id=collection.collection_id,
            root_node_id=root_id,
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
        return session, [LayerFrontierItem(parent_node_id=root_id, depth=0, order=0)], root

    root = SemanticNode(
        node_id=f"{collection.collection_id}|root",
        title=collection.title,
        node_type="DOCUMENT_ROOT",
        total_content_pointers=[
            HydratedTextPointer(
                source_cluster_id=unit_id,
                start_char=0,
                end_char=-1,
                verbatim_text=_source_text(record),
            )
            for unit_id, record in sorted(parser_source_map.items())
            if isinstance(record, Mapping)
            and record.get("participates_in_semantic_text", True)
        ],
        child_nodes=[],
        level_from_root=0,
    )
    root_id = required_node_id(root)
    session = ParseSessionState(
        collection_id=collection.collection_id,
        root_node_id=root_id,
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
    return session, [LayerFrontierItem(parent_node_id=root_id, depth=0, order=0)], root


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
    remaining = [item for item in sorted_queue if item not in selected_items]
    parent_titles: list[str] = []
    for item in selected_items:
        node = find_semantic_node(semantic_tree, item.parent_node_id)
        parent_titles.append(node.title if node is not None else item.parent_node_id)
    session = parse_session.model_copy(update={"current_depth": current_depth})
    frontier_identity = (
        f"depth={current_depth};parents={','.join(item.parent_node_id for item in selected_items)}"
    )

    def pointers_for(parent_node_id: str) -> list[HydratedTextPointer]:
        parent = find_semantic_node(semantic_tree, parent_node_id)
        if parent is None:
            return []
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
            item.parent_node_id: pointers_for(item.parent_node_id) for item in selected_items
        },
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
            raise ValueError(f"legacy compatibility tree is missing frontier parent {parent_id!r}")
        parent_children = list(parent.child_nodes)
        if len(parent_children) < 2:
            atomic_parent_ids.add(parent_id)
            atomic_parent_pointers[parent_id] = list(
                parent.total_content_pointers or parent.aggregate_content_pointers
            )
            continue
        for child in parent_children:
            children.append(
                LayerChildCandidate(
                    node_id=required_node_id(child),
                    parent_node_id=parent_id,
                    title=child.title,
                    node_type=child.node_type,
                    total_content_pointers=list(child.total_content_pointers),
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
