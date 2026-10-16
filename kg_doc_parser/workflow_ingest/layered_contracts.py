"""Bounded, serializable adapter around the layered parser primitives.

The parser owns parsing mechanics only.  Callers own persistence, leases, and
recovery.  These DTOs make a single seed or frontier expansion safe to retry.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import (
    CurrentLayerContext,
    CurrentLayerResult,
    LayerChildCandidate,
    LayerFrontierItem,
    ParseSessionState,
)
from .parser_core import (
    ParserSourceMap,
    ProposeLayerFn,
    commit_layer_children,
    enqueue_next_layer_frontier,
    initialize_parse_session,
    legacy_children_for_context,
    prepare_layer_frontier,
    propose_layer_breakdown,
    requeue_failed_frontier_items,
    validate_layer_commit,
)
from .semantics import SemanticNode, validate_semantic_tree_integrity
from .serialization import JsonValue


@dataclass(frozen=True, slots=True)
class _LayeredCollection:
    collection_id: str
    title: str


LayeredPayload = dict[str, JsonValue]
LayeredSourceMap = dict[str, LayeredPayload]


class LayeredParseLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_depth: int = Field(default=10, strict=True, ge=1)
    max_frontier_items: int = Field(default=1, strict=True, ge=1)
    max_parser_calls: int = Field(default=1, strict=True, ge=1)
    token_budget: int | None = Field(default=None, strict=True, ge=1)
    wall_time_seconds: float | None = Field(default=None, strict=True, gt=0)


class LayeredParseUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parser_calls: int = Field(default=0, strict=True, ge=0)
    input_tokens: int = Field(default=0, strict=True, ge=0)
    output_tokens: int = Field(default=0, strict=True, ge=0)
    elapsed_ms: int = Field(default=0, strict=True, ge=0)


class LayeredParseSeedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    collection: LayeredPayload
    parser_input: LayeredPayload
    source_map: LayeredSourceMap
    limits: LayeredParseLimits = Field(default_factory=LayeredParseLimits)


class LayeredParseSeedResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session: ParseSessionState
    frontier: list[LayerFrontierItem]
    root: SemanticNode
    diagnostics: LayeredPayload = Field(default_factory=dict)
    usage: LayeredParseUsage = Field(default_factory=LayeredParseUsage)

    @model_validator(mode="after")
    def _validate_root(self) -> LayeredParseSeedResult:
        validate_semantic_tree_integrity(self.root, max_depth=self.session.max_depth)
        return self


class LayeredParseExpandRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    session: ParseSessionState
    frontier: list[LayerFrontierItem]
    semantic_tree: LayeredPayload
    collection: LayeredPayload
    parser_input: LayeredPayload
    source_map: LayeredSourceMap
    limits: LayeredParseLimits = Field(default_factory=LayeredParseLimits)

    @model_validator(mode="after")
    def _validate_input_tree(self) -> LayeredParseExpandRequest:
        validate_semantic_tree_integrity(
            SemanticNode.model_validate(self.semantic_tree),
            parser_source_map=self.source_map,
            max_depth=min(self.limits.max_depth, self.session.max_depth),
        )
        return self


class LayeredParseExpandResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session: ParseSessionState
    consumed_frontier: list[LayerFrontierItem] = Field(default_factory=list)
    remaining_frontier: list[LayerFrontierItem] = Field(default_factory=list)
    children: list[LayerChildCandidate] = Field(default_factory=list)
    semantic_tree: SemanticNode
    stable: bool
    diagnostics: LayeredPayload = Field(default_factory=dict)
    usage: LayeredParseUsage = Field(default_factory=LayeredParseUsage)

    @model_validator(mode="after")
    def _validate_result_tree(self) -> LayeredParseExpandResult:
        validate_semantic_tree_integrity(self.semantic_tree, max_depth=self.session.max_depth)
        return self


def initialize_layered_parse(request: LayeredParseSeedRequest) -> LayeredParseSeedResult:
    """Create a coarse root and one bounded frontier item."""

    collection = _model_or_mapping(request.collection, "collection")
    session, frontier, root = initialize_parse_session(
        collection=collection,
        parser_input_dict=dict(request.parser_input),
        parser_source_map=_parser_source_map(request.source_map),
        max_depth=request.limits.max_depth,
    )
    session = session.model_copy(
        update={
            "max_depth": request.limits.max_depth,
            "metadata": {
                **dict(session.metadata or {}),
                "max_frontier_items": request.limits.max_frontier_items,
                "max_parser_calls": request.limits.max_parser_calls,
                "token_budget": request.limits.token_budget,
                "wall_time_seconds": request.limits.wall_time_seconds,
            },
        }
    )
    return LayeredParseSeedResult(
        session=session,
        frontier=frontier[: request.limits.max_frontier_items],
        root=root,
        diagnostics={"phase": "parse_seeded", "frontier_bounded": True},
        usage=LayeredParseUsage(),
    )


def expand_layered_frontier(
    request: LayeredParseExpandRequest,
    *,
    propose_layer_fn: ProposeLayerFn | None = None,
) -> LayeredParseExpandResult:
    """Expand at most ``max_frontier_items`` items and return JSON-safe state."""

    raw_parser_calls = dict(request.session.metadata or {}).get("parser_calls")
    if raw_parser_calls is None:
        parser_calls = 0
    elif type(raw_parser_calls) is int and raw_parser_calls >= 0:
        parser_calls = raw_parser_calls
    elif isinstance(raw_parser_calls, str) and raw_parser_calls.isdecimal():
        parser_calls = int(raw_parser_calls)
    else:
        raise ValueError("layered parse parser_calls metadata must be a non-negative integer")
    if parser_calls >= request.limits.max_parser_calls:
        raise ValueError("layered parse parser-call budget is exhausted")
    started = time.monotonic()
    semantic_tree = SemanticNode.model_validate(request.semantic_tree)
    ordered_frontier = sorted(request.frontier, key=lambda item: (item.depth, item.order))
    if not ordered_frontier:
        return LayeredParseExpandResult(
            session=request.session,
            remaining_frontier=[],
            semantic_tree=semantic_tree,
            stable=True,
            diagnostics={"phase": "parsed_graph_persisted", "reason": "frontier_empty"},
        )
    context, remaining, session = prepare_layer_frontier(
        parse_session=request.session,
        frontier_queue=ordered_frontier,
        semantic_tree=semantic_tree,
        max_items=request.limits.max_frontier_items,
    )
    selected = [
        item for item in ordered_frontier
        if item not in remaining
    ]
    if request.limits.token_budget is not None:
        estimated_tokens = _estimate_context_tokens(context)
        if estimated_tokens > request.limits.token_budget:
            raise ValueError(
                "layered parse context exceeds token budget: "
                f"estimated={estimated_tokens} budget={request.limits.token_budget}"
            )
    if session.mode == "legacy_compat":
        layer_result = legacy_children_for_context(
            parse_session=session,
            current_layer_context=context,
        )
    else:
        if propose_layer_fn is None:
            raise ValueError("workflow_layered expansion requires propose_layer_fn")
        layer_result = propose_layer_breakdown(
            collection=_model_or_mapping(request.collection, "collection"),
            parser_input_dict=dict(request.parser_input),
            parser_source_map=_parser_source_map(request.source_map),
            parse_session=session,
            current_layer_context=context,
            semantic_tree=semantic_tree,
            propose_layer_fn=propose_layer_fn,
        )
    commit_review = validate_layer_commit(
        current_layer_context=context,
        current_layer_result=layer_result,
        parser_source_map=_parser_source_map(request.source_map),
    )
    if not commit_review.satisfied:
        reasons = "; ".join(commit_review.review_notes[:4])
        raise ValueError(f"layered parse refinement failed structural validation: {reasons}")
    committed_result = commit_review.updated_result or layer_result
    updated_tree = commit_layer_children(
        semantic_tree=semantic_tree,
        current_layer_result=committed_result,
        current_depth=context.depth,
        parent_node_ids=commit_review.committable_parent_node_ids,
        parser_source_map=_parser_source_map(request.source_map),
        max_depth=request.limits.max_depth,
    )
    next_frontier = enqueue_next_layer_frontier(
        frontier_queue=remaining,
        current_layer_context=context,
        current_layer_result=committed_result,
        parse_session=session,
    )
    next_frontier = requeue_failed_frontier_items(
        frontier_queue=next_frontier,
        parent_node_ids=commit_review.failed_parent_node_ids,
        depth=context.depth,
    )
    elapsed_ms = int((time.monotonic() - started) * 1000)
    if (
        request.limits.wall_time_seconds is not None
        and elapsed_ms > request.limits.wall_time_seconds * 1000
    ):
        raise TimeoutError("layered parse expansion exceeded wall-time budget")
    updated_session = session.model_copy(
        update={
            "current_depth": context.depth,
            "metadata": {
                **dict(session.metadata or {}),
                "parser_calls": parser_calls + 1,
                "last_progress_at": datetime.now(UTC).isoformat(),
                "last_expanded_parent_ids": list(context.parent_node_ids),
            },
        }
    )
    stable = not next_frontier
    return LayeredParseExpandResult(
        session=updated_session,
        consumed_frontier=selected,
        remaining_frontier=next_frontier,
        children=list(committed_result.children),
        semantic_tree=updated_tree,
        stable=stable,
        diagnostics={
            "phase": "parsed_graph_persisted" if stable else "parse_expanding",
            "frontier_batch_size": len(selected),
            "child_count": len(committed_result.children),
        },
        usage=LayeredParseUsage(
            parser_calls=0 if session.mode == "legacy_compat" else 1,
            input_tokens=_estimate_context_tokens(context),
            elapsed_ms=elapsed_ms,
        ),
    )


def _estimate_context_tokens(context: CurrentLayerContext) -> int:
    """Use a deterministic conservative estimate when no tokenizer is present."""

    text = " ".join(context.parent_titles)
    for pointer_list in context.parent_content_pointers_by_id.values():
        for pointer in pointer_list:
            text += " " + pointer.verbatim_text
    return max(1, (len(text) + 3) // 4)


def _parser_source_map(source_map: LayeredSourceMap) -> ParserSourceMap:
    """Return the DTO source map without widening its JSON value contract."""

    return dict(source_map)


def _model_or_mapping(value: Mapping[str, JsonValue], name: str) -> _LayeredCollection:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    collection_id = value.get("collection_id")
    title = value.get("title")
    if not isinstance(collection_id, str) or not isinstance(title, str):
        raise TypeError(f"{name} must include string collection_id and title")
    return _LayeredCollection(collection_id=collection_id, title=title)


def _attach_children(tree: SemanticNode, result: CurrentLayerResult) -> SemanticNode:
    children_by_parent: dict[str, list[SemanticNode]] = {}
    for child in result.children:
        children_by_parent.setdefault(str(child.parent_node_id), []).append(
            SemanticNode(
                node_id=child.node_id,
                parent_id=child.parent_node_id,
                title=child.title,
                node_type=child.node_type,
                total_content_pointers=[
                    pointer.model_copy()
                    for pointer in child.total_content_pointers
                ],
                metadata=dict(child.metadata or {}),
            )
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
        updated_children.extend(children_by_parent.get(str(node.node_id), []))
        rebuilt[id(node)] = node.model_copy(update={"child_nodes": updated_children})

    return rebuilt[id(tree)]


__all__ = [
    "LayeredParseExpandRequest",
    "LayeredParseExpandResult",
    "LayeredParseLimits",
    "LayeredParseSeedRequest",
    "LayeredParseSeedResult",
    "LayeredParseUsage",
    "LayeredPayload",
    "LayeredSourceMap",
    "expand_layered_frontier",
    "initialize_layered_parse",
]
