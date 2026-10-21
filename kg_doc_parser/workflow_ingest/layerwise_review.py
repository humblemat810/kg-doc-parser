"""Review callback construction for layerwise LLM parsing."""

from __future__ import annotations

from collections.abc import Callable

from .layerwise_boundary_prompts import (
    _parent_context_excerpt,
    _proposal_attempt_context,
    _summarize_for_prompt,
)
from .layerwise_contracts import LayerwiseEventSink, LayerwiseReviewCallback
from .layerwise_invocation import structured_invoke
from .layerwise_payloads import _hydrate_llm_review
from .layerwise_source_spans import _source_map_excerpt, _trim_text
from .models import (
    CurrentLayerContext,
    CurrentLayerResult,
    CurrentLayerReview,
    FailureCategory,
    LLMCurrentLayerReview,
    ParseSessionState,
)
from .parser_core import ParserSourceMap, SplitStrategy
from .providers import (
    ProviderDiagnosticsSink,
    SupportsStructuredOutput,
    WorkflowProviderSettings,
)
from .serialization import safe_json_dumps


def build_layerwise_review_callback(
    *,
    chat_model: SupportsStructuredOutput,
    provider_settings: WorkflowProviderSettings,
    emit: LayerwiseEventSink,
    diagnostics_sink: ProviderDiagnosticsSink,
    normalized_failure_type: Callable[[], FailureCategory],
) -> LayerwiseReviewCallback:
    """Build the review callback with its runtime dependencies made explicit."""

    def review_layer(
        *,
        current_layer_context: CurrentLayerContext,
        current_layer_result: CurrentLayerResult,
        split_strategy: SplitStrategy,
        parser_source_map: ParserSourceMap | None = None,
        parse_session: ParseSessionState | None = None,
        **kwargs: object,
    ) -> CurrentLayerReview:
        source_map = parser_source_map or {}
        review_payload = {
            "task": "Review the current semantic layer proposal for layerwise correctness and source-grounded coverage.",
            "split_strategy": split_strategy,
            "attempt_context": _proposal_attempt_context(
                parse_session=parse_session,
                current_layer_context=current_layer_context,
            ),
            "current_layer_context": _summarize_for_prompt(
                current_layer_context,
                max_depth=4,
                max_items=6,
                max_string=280,
            ),
            "current_layer_result": _summarize_for_prompt(
                current_layer_result,
                max_depth=4,
                max_items=6,
                max_string=280,
            ),
            "parent_nodes": _parent_context_excerpt(
                current_layer_context=current_layer_context,
                parser_source_map=source_map,
            ),
            "source_map_excerpt": _source_map_excerpt(source_map),
            "review_rules": [
                "Review only the immediate children for this layer.",
                "Children should preserve each parent meaning collectively without inventing missing text.",
                "Coverage gaps, overlap conflicts, and duplicates should be called out explicitly.",
                "Prefer coarser layerwise structure over deep flattening when both are source-grounded.",
                "If the layer is already in a good layerwise sweet spot, satisfied=true is appropriate.",
            ],
        }
        messages = [
            (
                "system",
                (
                    "You review ONE semantic layer in an iterative document parsing workflow. "
                    "Return only structured data matching CurrentLayerReview."
                ),
            ),
            ("human", safe_json_dumps(review_payload, sort_keys=True)),
        ]
        emit(
            "workflow_layered_review_start",
            review_source="llm",
            depth=current_layer_context.depth,
            retry_count=current_layer_context.retry_count,
            split_strategy=split_strategy,
            child_count=len(current_layer_result.children),
        )
        try:
            review_result: LLMCurrentLayerReview = structured_invoke(
                chat_model,
                LLMCurrentLayerReview,
                messages,
                timeout_seconds=(
                    provider_settings.review_timeout_seconds
                    or provider_settings.parser.timeout_seconds
                ),
                diagnostics_sink=diagnostics_sink,
                operation="layer_review",
                max_in_flight=provider_settings.parser.max_in_flight_calls,
                attempt_index=1,
                call_role="review",
                strategy=split_strategy,
            )
            runtime_review = _hydrate_llm_review(
                review_result,
                parser_source_map=source_map,
            )
            emit(
                "workflow_layered_review_result",
                review_source="llm",
                depth=current_layer_context.depth,
                retry_count=current_layer_context.retry_count,
                split_strategy=split_strategy,
                satisfied=runtime_review.satisfied,
                coverage_ok=runtime_review.coverage_ok,
            )
            emit(
                "workflow_layered_review_completed",
                review_source="llm",
                depth=current_layer_context.depth,
                retry_count=current_layer_context.retry_count,
                split_strategy=split_strategy,
                satisfied=runtime_review.satisfied,
                coverage_ok=runtime_review.coverage_ok,
            )
            return runtime_review
        except Exception as exc:
            failure_reason = _trim_text(repr(exc), max_chars=500)
            failure_kind = (
                "timeout" if isinstance(exc, TimeoutError) else "provider_failure"
            )
            fallback_review = CurrentLayerReview(
                updated_result=current_layer_result,
                coverage_ok=None,
                satisfied=None,
                strategy_used=split_strategy,
                review_notes=[
                    f"quality_unknown: semantic layer review {failure_kind}",
                    "deterministic checks did not authorize successful review",
                ],
                metadata={
                    "review_failure": failure_kind,
                    "review_failure_reason": failure_reason,
                    "failure_type": (
                        "timeout"
                        if failure_kind == "timeout"
                        else normalized_failure_type()
                    ),
                },
            )
            emit(
                "workflow_layered_review_result",
                review_source="fallback",
                review_failure_reason=failure_reason,
                depth=current_layer_context.depth,
                retry_count=current_layer_context.retry_count,
                split_strategy=split_strategy,
                satisfied=fallback_review.satisfied,
                coverage_ok=fallback_review.coverage_ok,
            )
            emit(
                "workflow_layered_review_completed",
                review_source="fallback",
                review_failure_reason=failure_reason,
                depth=current_layer_context.depth,
                retry_count=current_layer_context.retry_count,
                split_strategy=split_strategy,
                satisfied=fallback_review.satisfied,
                coverage_ok=fallback_review.coverage_ok,
            )
            if failure_kind == "timeout":
                raise
            return fallback_review

    return review_layer
