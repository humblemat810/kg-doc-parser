from __future__ import annotations

from typing import Any, Callable

from .cache import WorkflowLLMCallCache
from .models import (
    CurrentLayerContext,
    CurrentLayerReview,
    CurrentLayerResult,
    LayerCoverageGap,
    LayerDuplicateChildNote,
    LayerChildCandidate,
    LayerFrontierItem,
    LayerSpanConflict,
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
    payload = tree.model_dump()
    payload["child_nodes"] = []
    return SemanticNode.model_validate(payload)


def _pointer_end(pointer: HydratedTextPointer, source_map: dict[str, dict[str, Any]] | None = None) -> int:
    if pointer.end_char != -1:
        return pointer.end_char
    if source_map is not None:
        text = source_map.get(pointer.source_cluster_id, {}).get("text", "")
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


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged: list[tuple[int, int]] = []
    cur_s, cur_e = sorted(intervals)[0]
    for s, e in sorted(intervals)[1:]:
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
    parser_source_map: dict[str, dict[str, Any]] | None = None,
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
    seen_overlap_pairs: set[tuple[str, str, str, int, int, str]] = set()

    parent_pointers = current_layer_context.parent_content_pointers_by_id or {}
    for parent_id in current_layer_context.parent_node_ids:
        parent_children = [
            child for child in current_layer_result.children if child.parent_node_id == parent_id
        ]
        if not parent_children:
            continue

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
                                left_child_id=left_child.node_id,
                                right_child_id=right_child.node_id,
                                source_cluster_id=left_ptr.source_cluster_id,
                                left_span=left_ptr,
                                right_span=right_ptr,
                                overlap_start=overlap_start,
                                overlap_end=overlap_end,
                                conflict_kind=conflict_kind,
                            )
                        )
                        review_notes.append(
                            f"{conflict_kind} between {left_child.node_id} and {right_child.node_id} "
                            f"on {left_ptr.source_cluster_id}:{overlap_start}-{overlap_end}"
                        )

        for parent_ptr in parent_pointers.get(parent_id, []):
            parent_end = _pointer_end(parent_ptr, parser_source_map)
            parent_start = max(parent_ptr.start_char, 0)
            if parent_end < parent_start:
                continue
            child_intervals = []
            for child in parent_children:
                for child_ptr in child.total_content_pointers:
                    if child_ptr.source_cluster_id != parent_ptr.source_cluster_id:
                        continue
                    child_intervals.append(
                        (max(child_ptr.start_char, 0), _pointer_end(child_ptr, parser_source_map))
                    )
            merged = _merge_intervals(child_intervals)
            cursor = parent_start
            cluster_text = parser_source_map.get(parent_ptr.source_cluster_id, {}).get("text", "") if parser_source_map else ""
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
    satisfied = coverage_ok and not overlap_conflicts and not duplicate_notes
    return coverage_ok, satisfied, overlap_conflicts, coverage_gaps, duplicate_notes, review_notes


def initialize_parse_session(
    *,
    collection,
    parser_input_dict: dict[str, Any],
    parser_source_map: dict[str, dict[str, Any]],
    max_depth: int = 10,
    allow_review: bool = True,
    split_strategy: str = "excerpt_first",
    fallback_split_strategy: str = "boundary_first",
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
            split_strategy=split_strategy,
            fallback_split_strategy=fallback_split_strategy,
            strategy_history=[split_strategy],
            mode="legacy_compat",
            compat_full_tree=full_tree.model_dump(),
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
                # The canonical persistence path validates excerpts against the
                # stored document content, so the root pointers need real text.
                verbatim_text=str(record.get("text") or ""),
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
        split_strategy=split_strategy,
        fallback_split_strategy=fallback_split_strategy,
        strategy_history=[split_strategy],
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
        parent_content_pointers_by_id={
            item.parent_node_id: list(find_semantic_node(semantic_tree, item.parent_node_id).total_content_pointers)
            if find_semantic_node(semantic_tree, item.parent_node_id) is not None
            else []
            for item in current_items
        },
        split_strategy=parse_session.split_strategy,
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
        split_strategy=current_layer_context.split_strategy,
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
    parser_source_map: dict[str, dict[str, Any]] | None = None,
    review_layer_fn: Callable[..., Any] | None = None,
    llm_cache: WorkflowLLMCallCache | None = None,
) -> tuple[CurrentLayerReview, ParseSessionState]:
    if parse_session.mode == "legacy_compat" or not parse_session.allow_review:
        return (
            CurrentLayerReview(
                updated_result=current_layer_result.model_copy(update={"satisfied": True}),
                coverage_ok=True,
                satisfied=True,
                strategy_used=current_layer_context.split_strategy,
            ),
            parse_session,
        )
    if review_layer_fn is None:
        reviewed = current_layer_result
    else:
        call = lambda: review_layer_fn(
            parse_session=parse_session,
            current_layer_context=current_layer_context,
            current_layer_result=current_layer_result,
            split_strategy=current_layer_context.split_strategy,
        )
        if llm_cache is not None:
            reviewed = llm_cache.cached_call(
                operation=f"review_cud_proposal:{current_layer_context.split_strategy}",
                fingerprint={
                    "parse_session": parse_session,
                    "current_layer_context": current_layer_context,
                    "current_layer_result": current_layer_result,
                    "split_strategy": current_layer_context.split_strategy,
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
    coverage_ok, invariant_satisfied, overlap_conflicts, coverage_gaps, duplicate_notes, invariant_notes = detect_layer_invariants(
        current_layer_context=current_layer_context,
        current_layer_result=result.updated_result or current_layer_result,
        parser_source_map=parser_source_map,
    )
    merged_notes = list(result.review_notes)
    for note in invariant_notes:
        if note not in merged_notes:
            merged_notes.append(note)
    base_satisfied = result.satisfied if result.satisfied is not None else invariant_satisfied
    satisfied = bool(base_satisfied and not (overlap_conflicts or coverage_gaps or duplicate_notes))
    updated_result = (result.updated_result or current_layer_result).model_copy(
        update={
            "satisfied": satisfied,
            "metadata": {
                **(result.updated_result or current_layer_result).metadata,
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
            },
        }
    )
    result = result.model_copy(
        update={
            "updated_result": updated_result,
            "coverage_ok": coverage_ok,
            "satisfied": satisfied,
            "strategy_used": current_layer_context.split_strategy,
            "overlap_conflicts": overlap_conflicts,
            "coverage_gap_notes": coverage_gaps,
            "duplicate_child_notes": duplicate_notes,
            "review_notes": merged_notes,
        }
    )
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
    tree = SemanticNode.model_validate(semantic_tree.model_dump())
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
        payload = node.model_dump()
        payload["child_nodes"] = [child.model_dump() for child in updated_children]
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
