from __future__ import annotations

from typing import Any, Callable

from .cache import WorkflowLLMCallCache
from .models import (
    CurrentLayerContext,
    CurrentLayerReview,
    CurrentLayerResult,
    LayerChildCandidate,
    LayerFrontierItem,
    ParseSessionState,
)
from .semantics import HydratedTextPointer, SemanticNode


def default_parse_semantic_fn(
    *,
    collection,
    parser_input_dict: dict[str, Any],
    parser_source_map: dict[str, dict[str, Any]],
):
    from ..semantic_document_splitting_layerwise_edits import build_document_tree

    return build_document_tree(
        doc_id=collection.collection_id,
        llm_input_dict=parser_input_dict,
        source_map=parser_source_map,
    )


def _coerce_semantic_tree(tree: Any) -> SemanticNode:
    if isinstance(tree, tuple):
        tree = tree[0]
    if isinstance(tree, dict):
        tree = SemanticNode.model_validate(tree)
    return tree


def _root_only(tree: SemanticNode) -> SemanticNode:
    payload = tree.model_dump(mode="json")
    payload["child_nodes"] = []
    return SemanticNode.model_validate(payload)


def initialize_parse_session(
    *,
    collection,
    parser_input_dict: dict[str, Any],
    parser_source_map: dict[str, dict[str, Any]],
    max_depth: int = 10,
    allow_review: bool = True,
    parse_semantic_fn: Callable[..., Any] | None = None,
) -> tuple[ParseSessionState, list[LayerFrontierItem], SemanticNode]:
    if parse_semantic_fn is not None:
        full_tree = _coerce_semantic_tree(
            parse_semantic_fn(
                collection=collection,
                parser_input_dict=parser_input_dict,
                parser_source_map=parser_source_map,
            )
        )
        root = _root_only(full_tree)
        session = ParseSessionState(
            collection_id=collection.collection_id,
            root_node_id=root.node_id,
            current_depth=0,
            max_depth=max_depth,
            allow_review=allow_review,
            mode="legacy_compat",
            compat_full_tree=full_tree.model_dump(mode="json"),
        )
        frontier = [LayerFrontierItem(parent_node_id=root.node_id, depth=0, order=0)]
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
                verbatim_text="",
            )
            for unit_id, record in sorted(parser_source_map.items())
            if record.get("participates_in_semantic_text", True)
        ],
        child_nodes=[],
        level_from_root=0,
    )
    session = ParseSessionState(
        collection_id=collection.collection_id,
        root_node_id=root.node_id,
        current_depth=0,
        max_depth=max_depth,
        allow_review=allow_review,
        mode="workflow_layered",
    )
    frontier = [LayerFrontierItem(parent_node_id=root.node_id, depth=0, order=0)]
    return session, frontier, root


def prepare_layer_frontier(
    *,
    parse_session: ParseSessionState,
    frontier_queue: list[LayerFrontierItem],
    semantic_tree: SemanticNode,
    max_retries: int = 3,
) -> tuple[CurrentLayerContext, list[LayerFrontierItem], ParseSessionState]:
    if not frontier_queue:
        raise ValueError("frontier queue is empty")
    sorted_queue = sorted(frontier_queue, key=lambda item: (item.depth, item.order))
    current_depth = sorted_queue[0].depth
    current_items = [item for item in sorted_queue if item.depth == current_depth]
    remaining = [item for item in sorted_queue if item.depth != current_depth]
    parent_titles = []
    for parent_id in [item.parent_node_id for item in current_items]:
        node = find_semantic_node(semantic_tree, parent_id)
        parent_titles.append(node.title if node is not None else parent_id)
    session = parse_session.model_copy(update={"current_depth": current_depth})
    context = CurrentLayerContext(
        depth=current_depth,
        parent_node_ids=[item.parent_node_id for item in current_items],
        parent_titles=parent_titles,
        retry_count=int(parse_session.layer_attempts.get(str(current_depth), 0)),
        max_retries=max_retries,
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
    for parent_id in current_layer_context.parent_node_ids:
        parent = find_semantic_node(full_tree, parent_id)
        if parent is None:
            continue
        for child in parent.child_nodes:
            children.append(
                LayerChildCandidate(
                    node_id=child.node_id,
                    parent_node_id=parent_id,
                    title=child.title,
                    node_type=child.node_type,
                    total_content_pointers=list(child.total_content_pointers),
                    expandable=child.node_type != "KEY_VALUE_PAIR",
                    metadata={"source": "legacy_compat"},
                )
            )
    return CurrentLayerResult(children=children, satisfied=True, reasoning_history=[])


def propose_layer_breakdown(
    *,
    collection,
    parser_input_dict: dict[str, Any],
    parser_source_map: dict[str, dict[str, Any]],
    parse_session: ParseSessionState,
    current_layer_context: CurrentLayerContext,
    semantic_tree: SemanticNode,
    propose_layer_fn: Callable[..., Any] | None = None,
    llm_cache: WorkflowLLMCallCache | None = None,
) -> CurrentLayerResult:
    if parse_session.mode == "legacy_compat":
        return legacy_children_for_context(
            parse_session=parse_session,
            current_layer_context=current_layer_context,
        )
    if propose_layer_fn is None:
        raise ValueError("workflow_layered mode requires propose_layer_fn")
    call = lambda: propose_layer_fn(
        collection=collection,
        parser_input_dict=parser_input_dict,
        parser_source_map=parser_source_map,
        parse_session=parse_session,
        current_layer_context=current_layer_context,
        semantic_tree=semantic_tree,
    )
    if llm_cache is not None:
        proposed = llm_cache.cached_call(
            operation="propose_layer_breakdown",
            fingerprint={
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
        return CurrentLayerResult(children=[_coerce_layer_child(child) for child in proposed])
    raise TypeError("unsupported proposed layer result")


def review_layer(
    *,
    parse_session: ParseSessionState,
    current_layer_context: CurrentLayerContext,
    current_layer_result: CurrentLayerResult,
    review_layer_fn: Callable[..., Any] | None = None,
    llm_cache: WorkflowLLMCallCache | None = None,
) -> tuple[CurrentLayerReview, ParseSessionState]:
    if parse_session.mode == "legacy_compat" or not parse_session.allow_review:
        return (
            CurrentLayerReview(
                updated_result=current_layer_result.model_copy(update={"satisfied": True}),
                coverage_ok=True,
                satisfied=True,
            ),
            parse_session,
        )
    if review_layer_fn is None:
        return (
            CurrentLayerReview(
                updated_result=current_layer_result.model_copy(update={"satisfied": True}),
                coverage_ok=True,
                satisfied=True,
            ),
            parse_session,
        )
    call = lambda: review_layer_fn(
        parse_session=parse_session,
        current_layer_context=current_layer_context,
        current_layer_result=current_layer_result,
    )
    if llm_cache is not None:
        reviewed = llm_cache.cached_call(
            operation="review_cud_proposal",
            fingerprint={
                "parse_session": parse_session,
                "current_layer_context": current_layer_context,
                "current_layer_result": current_layer_result,
            },
            fn=call,
        )
    else:
        reviewed = call()
    if isinstance(reviewed, CurrentLayerReview):
        result = reviewed
    elif isinstance(reviewed, CurrentLayerResult):
        result = CurrentLayerReview(
            updated_result=reviewed,
            coverage_ok=reviewed.metadata.get("layer_coverage_ok"),
            satisfied=reviewed.satisfied,
        )
    elif isinstance(reviewed, dict):
        if "updated_result" in reviewed or "coverage_ok" in reviewed or "review_notes" in reviewed:
            result = CurrentLayerReview.model_validate(reviewed)
        else:
            parsed = CurrentLayerResult.model_validate(reviewed)
            result = CurrentLayerReview(
                updated_result=parsed,
                coverage_ok=parsed.metadata.get("layer_coverage_ok"),
                satisfied=parsed.satisfied,
            )
    else:
        raise TypeError("unsupported reviewed layer result")
    attempts = dict(parse_session.layer_attempts)
    attempts[str(current_layer_context.depth)] = current_layer_context.retry_count + 1
    return result, parse_session.model_copy(update={"layer_attempts": attempts})


def apply_cud_update(
    *,
    current_layer_result: CurrentLayerResult,
    current_layer_review: CurrentLayerReview,
) -> CurrentLayerResult:
    updated = current_layer_review.updated_result or current_layer_result
    metadata = dict(updated.metadata)
    metadata.update(current_layer_review.metadata)
    if current_layer_review.coverage_ok is not None:
        metadata["layer_coverage_ok"] = current_layer_review.coverage_ok
    if current_layer_review.review_notes:
        metadata["review_notes"] = list(current_layer_review.review_notes)
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
    if current_layer_review is not None and current_layer_review.coverage_ok is not None:
        return bool(current_layer_review.coverage_ok), list(current_layer_review.review_notes)

    metadata_flag = current_layer_result.metadata.get("layer_coverage_ok")
    if isinstance(metadata_flag, bool):
        notes = current_layer_result.metadata.get("review_notes") or []
        return metadata_flag, [str(note) for note in notes]

    if current_layer_result.metadata.get("allow_empty_layer"):
        return True, []

    parent_ids = set(current_layer_context.parent_node_ids)
    covered_parents = {
        child.parent_node_id for child in current_layer_result.children if child.parent_node_id in parent_ids
    }
    missing = sorted(parent_ids - covered_parents)
    if missing:
        return False, [f"missing children for parent ids: {', '.join(missing)}"]
    return True, []


def repair_layer_candidates(
    *,
    current_layer_result: CurrentLayerResult,
    parser_source_map: dict[str, dict[str, Any]],
    correct_pointer_fn: Callable[[HydratedTextPointer, dict[str, dict[str, Any]]], HydratedTextPointer | None],
) -> tuple[CurrentLayerResult, int]:
    repaired_children: list[LayerChildCandidate] = []
    repaired_count = 0
    for child in current_layer_result.children:
        repaired_ptrs = []
        for pointer in child.total_content_pointers:
            fixed = correct_pointer_fn(pointer, parser_source_map)
            if fixed is None:
                raise ValueError(f"unrecoverable pointer for child {child.title!r}")
            if fixed.model_dump() != pointer.model_dump():
                repaired_count += 1
            repaired_ptrs.append(fixed)
        repaired_children.append(
            child.model_copy(update={"total_content_pointers": repaired_ptrs})
        )
    return current_layer_result.model_copy(update={"children": repaired_children}), repaired_count


def dedupe_and_filter_layer(
    *,
    current_layer_context: CurrentLayerContext,
    current_layer_result: CurrentLayerResult,
) -> CurrentLayerResult:
    parent_title_lookup = {
        node_id: title for node_id, title in zip(current_layer_context.parent_node_ids, current_layer_context.parent_titles)
    }
    seen: set[tuple[str, str, str]] = set()
    filtered: list[LayerChildCandidate] = []
    for child in current_layer_result.children:
        if child.parent_node_id not in parent_title_lookup:
            continue
        if child.title.strip() == parent_title_lookup[child.parent_node_id].strip():
            continue
        key = (child.parent_node_id, child.node_type, child.title.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        filtered.append(child)
    return current_layer_result.model_copy(update={"children": filtered})


def commit_layer_children(
    *,
    semantic_tree: SemanticNode,
    current_layer_result: CurrentLayerResult,
    current_depth: int,
) -> SemanticNode:
    tree = SemanticNode.model_validate(semantic_tree.model_dump(mode="json"))
    children_by_parent: dict[str, list[SemanticNode]] = {}
    for child in current_layer_result.children:
        children_by_parent.setdefault(child.parent_node_id, []).append(
            SemanticNode(
                node_id=child.node_id,
                parent_id=child.parent_node_id,
                node_type=child.node_type,
                title=child.title,
                total_content_pointers=list(child.total_content_pointers),
                child_nodes=[],
                level_from_root=current_depth + 1,
            )
        )

    def walk(node: SemanticNode) -> SemanticNode:
        updated_children = [walk(existing) for existing in node.child_nodes]
        if str(node.node_id) in children_by_parent:
            updated_children.extend(children_by_parent[str(node.node_id)])
        payload = node.model_dump(mode="json")
        payload["child_nodes"] = [child.model_dump(mode="json") for child in updated_children]
        return SemanticNode.model_validate(payload)

    return walk(tree)


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
    for child in current_layer_result.children:
        if child.expandable:
            queued.append(
                LayerFrontierItem(
                    parent_node_id=child.node_id,
                    depth=next_depth,
                    order=next_order,
                )
            )
            next_order += 1
    return queued


def find_semantic_node(root: SemanticNode, node_id: str) -> SemanticNode | None:
    if str(root.node_id) == str(node_id):
        return root
    for child in root.child_nodes:
        found = find_semantic_node(child, node_id)
        if found is not None:
            return found
    return None


def finalize_semantic_tree(semantic_tree: SemanticNode) -> SemanticNode:
    return semantic_tree


def _coerce_layer_child(value: Any) -> LayerChildCandidate:
    if isinstance(value, LayerChildCandidate):
        return value
    if isinstance(value, dict):
        return LayerChildCandidate.model_validate(value)
    raise TypeError("unsupported layer child candidate")
