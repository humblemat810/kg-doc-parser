"""Typed contracts and deterministic batching for semantic-node indexing."""

from __future__ import annotations

import json
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from contextvars import ContextVar
from enum import Enum
from typing import Self, TypeVar

from pydantic import BaseModel, Field, model_validator

from .workflow_ingest.serialization import JsonValue

TIndexItem = TypeVar("TIndexItem", bound=Mapping[str, object])

available_node_ids: ContextVar[set[str] | None] = ContextVar(
    "indexing_nodes", default=None
)


class IndexingResponse(BaseModel):
    """Search metadata generated for one semantic node."""

    node_id: str = Field(description="The node id this index result is representing")
    canonical_title: str = Field(description="best searchable title")
    keywords: list[str] = Field(description="5-12 short keywords")
    aliases: list[str] = Field(description="0-5 alternative phrasings")
    provision: str = Field(
        description="clauses, terms, sections, and schedules. Example: 'Schedule 3.1', 'Term 5a', 'Clause 3.12.2(a)'"
    )

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        node_set = available_node_ids.get()
        if node_set is None:
            raise ValueError("indexing response validation requires an active node-id context")
        try:
            node_set.remove(str(self.node_id))
        except KeyError:
            raise ValueError(f"node_id {self.node_id} duplicated or not exist") from None
        return self


class BatchIndexResponse(BaseModel):
    """Indexing results for one provider batch."""

    index: list[IndexingResponse] = Field(
        description="a list of indexing result. Each response member/element must have 1 and only 1 corresponding index. Each key is the input node's id"
    )


class IdMapping:
    """Stable short IDs used to reduce indexing prompt size."""

    def __init__(self) -> None:
        self.forward_map: dict[str, str] = {}
        self.backward_map: dict[str, str] = {}

    def to_uuid(self, short_id: str) -> str | None:
        return self.backward_map.get(short_id)

    def to_short_id(self, node_id: str, title: JsonValue) -> str:
        if node_id not in self.forward_map:
            short_id = f"nid:{len(self.forward_map)}:{title}"
            self.forward_map[node_id] = short_id
            self.backward_map[short_id] = node_id
        return self.forward_map[node_id]


class IndexMode(str, Enum):
    FLAT_VERBATIM = "flat_verbatim"
    BOTTOM_UP_DIGEST = "bottom_up_digest"
    BOTH = "both"


def default_token_estimate(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def json_compact(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def render_nodes_payload(
    nodes_payload: Sequence[Mapping[str, object]] | Mapping[str, object],
) -> str:
    return json_compact({"nodes": nodes_payload})


def batch_nodes(
    items: list[TIndexItem],
    *,
    max_nodes: int | None = None,
    max_input_tokens: int | None = None,
    token_estimator: Callable[[str], int] = default_token_estimate,
    base_prompt_tokens: int = 450,
    per_node_overhead_tokens: int = 15,
    render_item_for_estimate: Callable[[TIndexItem], str] = json_compact,
) -> Iterable[list[TIndexItem]]:
    """Split items in input order, allowing one oversized item per batch."""
    if max_nodes is None and max_input_tokens is None:
        max_nodes = 25

    batch: list[TIndexItem] = []
    batch_tokens = base_prompt_tokens

    def flush() -> Generator[list[TIndexItem], None, None]:
        nonlocal batch, batch_tokens
        if batch:
            yield batch
        batch = []
        batch_tokens = base_prompt_tokens

    for item in items:
        item_text = render_item_for_estimate(item)
        item_tokens = token_estimator(item_text) + per_node_overhead_tokens

        would_exceed_tokens = (
            max_input_tokens is not None
            and batch
            and (batch_tokens + item_tokens) > max_input_tokens
        )
        would_exceed_nodes = (
            max_nodes is not None
            and batch
            and (len(batch) + 1) > max_nodes
        )

        if would_exceed_tokens or would_exceed_nodes:
            yield from flush()

        batch.append(item)
        batch_tokens += item_tokens

    yield from flush()
