"""Typed callback contracts shared by layerwise orchestration modules."""

from __future__ import annotations

from typing import Protocol, TypedDict

from .models import (
    CurrentLayerContext,
    CurrentLayerResult,
    CurrentLayerReview,
    ParseSessionState,
)
from .parser_core import ParserPayload, ParserSourceMap, SplitStrategy
from .semantics import SemanticNode


class LayerwiseProposeCallback(Protocol):
    """Provider callback for a bounded layer proposal."""

    def __call__(
        self,
        *,
        parser_source_map: ParserSourceMap,
        current_layer_context: CurrentLayerContext,
        semantic_tree: SemanticNode,
        split_strategy: SplitStrategy,
        parser_input_dict: ParserPayload,
        parse_session: ParseSessionState,
        **kwargs: object,
    ) -> CurrentLayerResult: ...


class LayerwiseReviewCallback(Protocol):
    """Provider callback for reviewing a bounded layer proposal."""

    def __call__(
        self,
        *,
        current_layer_context: CurrentLayerContext,
        current_layer_result: CurrentLayerResult,
        split_strategy: SplitStrategy,
        parser_source_map: ParserSourceMap | None = None,
        parse_session: ParseSessionState | None = None,
        **kwargs: object,
    ) -> CurrentLayerReview: ...


class LayerwiseEventSink(Protocol):
    """Structured event callback used by parser orchestration."""

    def __call__(self, stage: str, **extra: object) -> None: ...


class LayerwiseFallbackBuilder(Protocol):
    """Deterministic fallback for unavailable or invalid provider output."""

    def __call__(
        self,
        *,
        current_layer_context: CurrentLayerContext,
        parser_source_map: ParserSourceMap,
    ) -> CurrentLayerResult: ...


class LayerwiseLLMCallbacks(TypedDict):
    propose_layer_fn: LayerwiseProposeCallback
    review_layer_fn: LayerwiseReviewCallback
    max_depth: int
    allow_review: bool
