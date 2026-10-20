from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, Protocol

from .models import (
    CurrentLayerContext,
    CurrentLayerResult,
    CurrentLayerReview,
    ParseSessionState,
)
from .semantics import HydratedTextPointer, SemanticNode
from .serialization import JsonObject

SplitStrategy = Literal["excerpt_first", "boundary_first"]
ParserPayload = JsonObject
ParserSourceMap = dict[str, ParserPayload]


class ParseSemanticFn(Protocol):
    """Provider callback for the legacy whole-document parse path."""

    def __call__(
        self,
        *,
        collection: SourceCollectionLike,
        parser_input_dict: ParserPayload,
        parser_source_map: ParserSourceMap,
        model_names: list[str] | None = None,
    ) -> object: ...


class ProposeLayerFn(Protocol):
    """Provider callback for one bounded semantic-layer proposal."""

    def __call__(
        self,
        *,
        collection: SourceCollectionLike,
        parser_input_dict: ParserPayload,
        parser_source_map: ParserSourceMap,
        parse_session: ParseSessionState,
        current_layer_context: CurrentLayerContext,
        semantic_tree: SemanticNode,
        split_strategy: SplitStrategy,
    ) -> CurrentLayerResult | Mapping[str, object] | Sequence[object]: ...


class ReviewLayerFn(Protocol):
    """Provider callback for reviewing one bounded semantic layer."""

    def __call__(
        self,
        *,
        parse_session: ParseSessionState,
        current_layer_context: CurrentLayerContext,
        current_layer_result: CurrentLayerResult,
        parser_source_map: ParserSourceMap,
        split_strategy: SplitStrategy,
    ) -> CurrentLayerReview | CurrentLayerResult | Mapping[str, object]: ...


class PointerCorrector(Protocol):
    """Repair one grounded pointer against the authoritative source map."""

    def __call__(
        self,
        pointer: HydratedTextPointer,
        parser_source_map: ParserSourceMap,
        /,
    ) -> HydratedTextPointer | None: ...


class SourceCollectionLike(Protocol):
    """Minimum collection identity needed by the parser core."""

    @property
    def collection_id(self) -> str: ...

    @property
    def title(self) -> str: ...


class SourceUnitLike(Protocol):
    """Minimum source-unit shape needed by deterministic fallback parsing."""

    @property
    def unit_id(self) -> str | None: ...

    @property
    def cluster_number(self) -> int | None: ...

    @property
    def text(self) -> str | None: ...


class SourcePageLike(Protocol):
    """Minimum page shape needed by deterministic fallback parsing."""

    @property
    def page_number(self) -> int: ...

    @property
    def units(self) -> Sequence[SourceUnitLike]: ...


class SourceCollectionWithPagesLike(SourceCollectionLike, Protocol):
    """Collection shape required by page-aware deterministic fallback parsing."""

    @property
    def pages(self) -> Sequence[SourcePageLike]: ...


class NodeWithOptionalId(Protocol):
    @property
    def node_id(self) -> str | None: ...


def required_node_id(node: NodeWithOptionalId) -> str:
    if node.node_id is None:
        raise ValueError("semantic node must have a stable node_id")
    return node.node_id
