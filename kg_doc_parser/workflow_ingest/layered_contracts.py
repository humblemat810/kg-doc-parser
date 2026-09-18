"""Bounded, serializable adapter around the layered parser primitives.

The parser owns parsing mechanics only.  Callers own persistence, leases, and
recovery.  These DTOs make a single seed or frontier expansion safe to retry.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .models import (
    CurrentLayerResult,
    LayerChildCandidate,
    LayerFrontierItem,
    ParseSessionState,
)
from .parser_core import (
    enqueue_next_layer_frontier,
    initialize_parse_session,
    legacy_children_for_context,
    prepare_layer_frontier,
    propose_layer_breakdown,
)
from .semantics import SemanticNode


class LayeredParseLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_depth: int = Field(default=10, ge=1)
    max_frontier_items: int = Field(default=1, ge=1)
    max_parser_calls: int = Field(default=1, ge=1)
    token_budget: int | None = Field(default=None, ge=1)
    wall_time_seconds: float | None = Field(default=None, gt=0)


class LayeredParseUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parser_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    elapsed_ms: int = Field(default=0, ge=0)


class LayeredParseSeedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    collection: dict[str, Any]
    parser_input: dict[str, Any]
    source_map: dict[str, dict[str, Any]]
    limits: LayeredParseLimits = Field(default_factory=LayeredParseLimits)


class LayeredParseSeedResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session: ParseSessionState
    frontier: list[LayerFrontierItem]
    root: SemanticNode
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    usage: LayeredParseUsage = Field(default_factory=LayeredParseUsage)


class LayeredParseExpandRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    session: ParseSessionState
    frontier: list[LayerFrontierItem]
    semantic_tree: dict[str, Any]
    collection: dict[str, Any]
    parser_input: dict[str, Any]
    source_map: dict[str, dict[str, Any]]
    limits: LayeredParseLimits = Field(default_factory=LayeredParseLimits)


class LayeredParseExpandResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session: ParseSessionState
    consumed_frontier: list[LayerFrontierItem] = Field(default_factory=list)
    remaining_frontier: list[LayerFrontierItem] = Field(default_factory=list)
    children: list[LayerChildCandidate] = Field(default_factory=list)
    semantic_tree: SemanticNode
    stable: bool
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    usage: LayeredParseUsage = Field(default_factory=LayeredParseUsage)


def initialize_layered_parse(request: LayeredParseSeedRequest) -> LayeredParseSeedResult:
    """Create a coarse root and one bounded frontier item."""

    collection = _model_or_mapping(request.collection, "collection")
    session, frontier, root = initialize_parse_session(
        collection=collection,
        parser_input_dict=dict(request.parser_input),
        parser_source_map=dict(request.source_map),
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
    propose_layer_fn: Callable[..., CurrentLayerResult] | None = None,
) -> LayeredParseExpandResult:
    """Expand at most ``max_frontier_items`` items and return JSON-safe state."""

    parser_calls = int(dict(request.session.metadata or {}).get("parser_calls") or 0)
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
            parser_source_map=dict(request.source_map),
            parse_session=session,
            current_layer_context=context,
            semantic_tree=semantic_tree,
            propose_layer_fn=propose_layer_fn,
        )
    next_frontier = enqueue_next_layer_frontier(
        frontier_queue=remaining,
        current_layer_context=context,
        current_layer_result=layer_result,
        parse_session=session,
    )
    updated_tree = _attach_children(semantic_tree, layer_result)
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
        children=list(layer_result.children),
        semantic_tree=updated_tree,
        stable=stable,
        diagnostics={
            "phase": "parsed_graph_persisted" if stable else "parse_expanding",
            "frontier_batch_size": len(selected),
            "child_count": len(layer_result.children),
        },
        usage=LayeredParseUsage(
            parser_calls=0 if session.mode == "legacy_compat" else 1,
            input_tokens=_estimate_context_tokens(context),
            elapsed_ms=elapsed_ms,
        ),
    )


def _estimate_context_tokens(context: object) -> int:
    """Use a deterministic conservative estimate when no tokenizer is present."""

    parent_titles = getattr(context, "parent_titles", ()) or ()
    pointers = getattr(context, "parent_content_pointers_by_id", {}) or {}
    text = " ".join(str(title) for title in parent_titles)
    for pointer_list in pointers.values():
        for pointer in pointer_list:
            text += " " + str(getattr(pointer, "verbatim_text", "") or "")
    return max(1, (len(text) + 3) // 4)


def _model_or_mapping(value: Mapping[str, Any], name: str) -> Any:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return type("LayeredContractObject", (), dict(value))()


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
                    {
                        "source_cluster_id": pointer.source_cluster_id,
                        "start_char": pointer.start_char,
                        "end_char": pointer.end_char,
                        "verbatim_text": pointer.verbatim_text,
                    }
                    for pointer in child.total_content_pointers
                ],
                metadata=dict(child.metadata or {}),
            )
        )

    def walk(node: SemanticNode) -> SemanticNode:
        updated_children = [walk(existing) for existing in node.child_nodes]
        updated_children.extend(children_by_parent.get(str(node.node_id), []))
        return node.model_copy(update={"child_nodes": updated_children})

    return walk(tree)


__all__ = [
    "LayeredParseExpandRequest",
    "LayeredParseExpandResult",
    "LayeredParseLimits",
    "LayeredParseSeedRequest",
    "LayeredParseSeedResult",
    "LayeredParseUsage",
    "expand_layered_frontier",
    "initialize_layered_parse",
]
