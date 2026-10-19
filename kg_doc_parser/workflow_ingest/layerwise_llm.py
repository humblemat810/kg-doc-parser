from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from typing import Protocol, TypedDict, cast

from kogwistar.runtime import (
    RetryAttemptRecord,
    RetryExhaustedError,
    retry_with_context,
)

from . import layerwise_boundary_pipeline as boundary_pipeline
from .layerwise_boundaries import (
    _boundary_anchor_excerpt,  # noqa: F401 - compatibility re-export
    _boundary_candidates_for_pointer,  # noqa: F401 - compatibility re-export
    _BoundaryAnchorResolution,  # noqa: F401 - preserve test and caller patch seam
    _classify_boundary_kind,  # noqa: F401 - preserve callable module API
    _is_heading_line,  # noqa: F401 - compatibility re-export
    _is_list_item_line,  # noqa: F401 - compatibility re-export
    _is_table_row_line,  # noqa: F401 - compatibility re-export
    _nearest_legal_cutpoint,  # noqa: F401 - compatibility re-export
    _repair_boundary_cutpoint_from_source,
    _resolve_boundary_anchor,  # noqa: F401 - preserve test and caller patch seam
)
from .layerwise_boundary_pipeline import (
    _annotate_proposal_result,
    _assemble_layer_result_from_boundaries,
    _boundary_candidate_lookup,
    _boundary_decision_key,
    _boundary_identical_parent_child_ids,
    _boundary_refinement_prompt_context,
    _boundary_refinement_targets,
    _boundary_review_decision,
    _boundary_review_decision_summary,
    _boundary_validation_reason,
    _normalize_boundary_cutpoints_from_candidates,
    _parent_context_excerpt,
    _proposal_attempt_context,
    _proposal_validation_reason,
    _summarize_for_prompt,
)
from .layerwise_invocation import structured_invoke as _structured_invoke
from .layerwise_payloads import (
    _dump_model,
    _hydrate_llm_layer_result,
    _hydrate_llm_pointer_payload,  # noqa: F401 - compatibility re-export
    _hydrate_llm_review,
    _object_dict,  # noqa: F401 - compatibility re-export
    _object_list,
)
from .layerwise_source_spans import (
    _offset_value,
    _pointer_signature,  # noqa: F401 - compatibility re-export
    _record_text,
    _source_map_excerpt,
    _trim_text,
)
from .models import (
    BoundaryCutpoint,
    BoundaryReviewBatch,
    BoundaryReviewDecision,
    CurrentLayerContext,
    CurrentLayerResult,
    CurrentLayerReview,
    FailureCategory,
    LayerChildCandidate,
    LayerReasoningEntry,
    LLMBoundaryProposalBatch,
    LLMCurrentLayerResult,
    LLMCurrentLayerReview,
    ParseSessionState,
)
from .parser_core import ParserPayload, ParserSourceMap, SplitStrategy
from .providers import (
    ProviderDiagnosticValue,
    WorkflowProviderSettings,
    build_chat_model_for_role,
    provider_call_metrics_snapshot,
)
from .semantics import HydratedTextPointer, SemanticNode
from .serialization import JsonValue, json_safe, safe_json_dumps


def _bounded_retry_delay(
    *,
    attempt_number: int,
    retry_budget: int,
    provider_settings: WorkflowProviderSettings,
) -> float:
    """Return exponential backoff for a retry, never for the terminal attempt."""

    if attempt_number >= retry_budget:
        return 0.0
    base = float(provider_settings.parser.retry_backoff_seconds or 0.0)
    maximum = float(provider_settings.parser.retry_backoff_max_seconds or 0.0)
    return min(maximum, base * (2 ** max(0, attempt_number - 1))) if maximum else 0.0


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


LayerwiseCallback = LayerwiseLLMCallbacks


def _fallback_layer_result(
    *,
    current_layer_context: CurrentLayerContext,
    parser_source_map: ParserSourceMap,
) -> CurrentLayerResult:
    children: list[LayerChildCandidate] = []
    if current_layer_context.depth > 0:
        return CurrentLayerResult(
            children=[],
            satisfied=True,
            atomic_retained=True,
            reasoning_history=[LayerReasoningEntry(source="deterministic_depth_stop")],
            metadata={
                "fallback": "depth_stop",
                "allow_empty_layer": True,
            },
        )
    parent_ids = list(current_layer_context.parent_node_ids)
    parent_id = parent_ids[0] if parent_ids else "root"
    records = list(parser_source_map.items())
    for index, (source_cluster_id, record) in enumerate(records[:8], start=1):
        text = _record_text(record).strip()
        if not text:
            continue
        title_match = re.search(r"(?m)^#{1,3}\s+(.+)$", text)
        title = title_match.group(1).strip() if title_match else f"Section {index}"
        excerpt = text
        children.append(
            LayerChildCandidate(
                node_id=f"{parent_id}|section-{index}",
                parent_node_id=parent_id,
                title=title[:160],
                node_type="TEXT_FLOW",
                total_content_pointers=[
                    HydratedTextPointer(
                        source_cluster_id=str(source_cluster_id),
                        start_char=0,
                        end_char=max(len(excerpt) - 1, 0),
                        verbatim_text=excerpt or " ",
                    )
                ],
                expandable=False,
                metadata={"source": "workflow_layered_fallback"},
            )
        )
    return CurrentLayerResult(
        children=children,
        satisfied=True,
        reasoning_history=[LayerReasoningEntry(source="deterministic_fallback")],
        metadata={"fallback": "llm_empty_or_unavailable", "allow_empty_layer": False},
    )


def build_layerwise_llm_callbacks(
    provider_settings: WorkflowProviderSettings,
    *,
    event_sink: LayerwiseEventSink | None = None,
    model_callbacks: Sequence[object] | None = None,
    fallback_layer_result_fn: LayerwiseFallbackBuilder | None = None,
    max_depth: int = 2,
    allow_review: bool = True,
    proposal_mode: str | None = None,
    boundary_refinement_rounds: int = 1,
) -> LayerwiseLLMCallbacks:
    if model_callbacks:
        chat_model = build_chat_model_for_role(
            "parser",
            provider_settings,
            callbacks=list(model_callbacks),
        )
    else:
        # Preserve the narrow legacy factory shape used by lightweight callers
        # and test doubles that do not accept optional callback plumbing.
        chat_model = build_chat_model_for_role("parser", provider_settings)
    fallback_builder = fallback_layer_result_fn or _fallback_layer_result
    proposal_mode = str(proposal_mode or provider_settings.proposal_mode or "children")
    if proposal_mode not in {"children", "boundaries"}:
        raise ValueError("proposal_mode must be either 'children' or 'boundaries'")
    boundary_refinement_rounds = max(0, int(boundary_refinement_rounds or 0))
    proposal_retry_rounds = max(0, provider_settings.parser.max_retries)
    last_provider_failure_type: str | None = None

    def _emit(stage: str, **extra: object) -> None:
        if callable(event_sink):
            event_sink(stage, **extra)

    def _emit_provider_diagnostics(
        diagnostics: Mapping[str, ProviderDiagnosticValue],
    ) -> None:
        """Publish one structured call record without exposing provider payloads."""

        nonlocal last_provider_failure_type
        if diagnostics.get("success") is False:
            last_provider_failure_type = str(
                diagnostics.get("failure_type") or "transport/provider_exception"
            )
        _emit(
            "workflow_provider_call",
            **diagnostics,
            provider_metrics=provider_call_metrics_snapshot(),
        )

    def _normalized_provider_failure_type() -> FailureCategory:
        if last_provider_failure_type in {"timeout", "structured_output_parse_failure"}:
            return cast(FailureCategory, last_provider_failure_type)
        return "transport/provider_exception"

    def _proposal_attempt_payload(
        payload: Mapping[str, object],
        *,
        attempt_index: int,
        prior_error: str | None,
    ) -> ParserPayload:
        normalized = json_safe(payload)
        if not isinstance(normalized, dict):
            raise TypeError("proposal prompt payload must serialize to a JSON object")
        normalized["proposal_attempt"] = attempt_index + 1
        normalized["proposal_retry_rounds"] = proposal_retry_rounds
        if prior_error:
            normalized["previous_attempt_error"] = prior_error
        return normalized

    def _propose_layer_fn(
        *,
        parser_source_map: ParserSourceMap,
        current_layer_context: CurrentLayerContext,
        semantic_tree: SemanticNode,
        split_strategy: SplitStrategy,
        parser_input_dict: ParserPayload,
        parse_session: ParseSessionState,
        **kwargs: object,
    ) -> CurrentLayerResult:
        if proposal_mode == "boundaries":
            boundary_candidates = boundary_pipeline._boundary_prompt_candidate_context(
                current_layer_context=current_layer_context,
                parser_source_map=parser_source_map,
                max_points=provider_settings.boundary_max_points,
            )
            boundary_candidate_lookup = _boundary_candidate_lookup(boundary_candidates)
            boundary_dropped_count = 0
            boundary_repaired_count = 0
            _emit(
                "workflow_layered_boundary_proposal_start",
                proposal_mode="boundaries",
                depth=current_layer_context.depth,
                retry_count=current_layer_context.retry_count,
                split_strategy=split_strategy,
                candidate_count=len(boundary_candidates),
                refinement_budget=boundary_refinement_rounds,
            )

            def _build_boundary_messages(
                attempt_number: int, previous_error: str | None
            ) -> list[tuple[str, str]]:
                rules = [
                    "Operate on ONE layer only: propose cutpoints for the current parent nodes.",
                    "Prefer choosing candidate_id values from the provided legal_cutpoints list.",
                    "When choosing a candidate_id, copy its cut_offset, parent_node_id, and source_cluster_id exactly.",
                    "Do not emit child text, summaries, or prose.",
                    "For each cutpoint, emit cut_reason plus short exact text_before_cut and text_after_cut anchors.",
                    "The anchors must uniquely identify the cut within the parent text and must not be the full child verbatim.",
                    "Never omit either anchor: if using candidate_id, copy both anchors exactly from that candidate.",
                    "If a proposed cut cannot be grounded to a candidate, omit that cutpoint rather than guessing.",
                    "Cutpoints must be parent-scoped and source-cluster scoped.",
                    "Prefer section, paragraph, list-item, and sentence boundaries over semantic guesses.",
                    "Keep the proposal sorted and unique per parent/source cluster.",
                    "If a parent is atomic, return no internal cutpoints for that parent and set satisfied=true only when the current layer is complete.",
                ]
                if previous_error:
                    rules.extend(
                        [
                            "RECOVERY PASS: return only the smallest valid candidate set needed to make a real split.",
                            "RECOVERY PASS: do not invent offsets or anchors; omit an ambiguous cutpoint instead of retrying it.",
                        ]
                    )
                recovery_example: dict[str, object] | None = None
                if previous_error:
                    for boundary_candidate in boundary_candidates:
                        legal_cutpoints = _object_list(
                            boundary_candidate.get("legal_cutpoints")
                        )
                        if not legal_cutpoints:
                            continue
                        legal_cutpoint = legal_cutpoints[0]
                        if not isinstance(legal_cutpoint, dict):
                            continue
                        recovery_example = {
                            "cutpoints": [
                                {
                                    "candidate_id": legal_cutpoint.get("candidate_id"),
                                    "parent_node_id": boundary_candidate.get(
                                        "parent_node_id"
                                    ),
                                    "source_cluster_id": boundary_candidate.get(
                                        "source_cluster_id"
                                    ),
                                    "cut_offset": legal_cutpoint.get("cut_offset"),
                                    "boundary_kind": legal_cutpoint.get(
                                        "boundary_kind"
                                    ),
                                    "text_before_cut": legal_cutpoint.get(
                                        "text_before_cut"
                                    ),
                                    "text_after_cut": legal_cutpoint.get(
                                        "text_after_cut"
                                    ),
                                    "cut_reason": legal_cutpoint.get("reason")
                                    or "recovery example",
                                }
                            ],
                            "satisfied": False,
                        }
                        break
                prompt_payload = _proposal_attempt_payload(
                    {
                        "task": "Propose semantic cutpoints for the next layer. Return only boundary cutpoints, not child text.",
                        "output_contract": "Return an LLMBoundaryProposalBatch only.",
                        "proposal_mode": proposal_mode,
                        "split_strategy": split_strategy,
                        "attempt_context": _proposal_attempt_context(
                            parse_session=parse_session,
                            current_layer_context=current_layer_context,
                        ),
                        "current_layer_context": _dump_model(current_layer_context),
                        "parent_nodes": _parent_context_excerpt(
                            current_layer_context=current_layer_context,
                            parser_source_map=parser_source_map,
                        ),
                        "boundary_candidates": boundary_candidates,
                        "semantic_tree_snapshot": _summarize_for_prompt(
                            semantic_tree,
                            max_depth=4,
                            max_items=5,
                            max_string=280,
                        ),
                        "full_document_context": _summarize_for_prompt(
                            parser_input_dict,
                            max_depth=4,
                            max_items=5,
                            max_string=280,
                        ),
                        "source_map_excerpt": _source_map_excerpt(parser_source_map),
                        "rules": rules,
                        "recovery_example": recovery_example,
                    },
                    attempt_index=attempt_number - 1,
                    prior_error=previous_error,
                )
                return [
                    (
                        "system",
                        (
                            "You are revising ONE semantic layer in an iterative document parsing workflow. "
                            "Return only structured data matching LLMBoundaryProposalBatch. "
                            "Propose grounded cutpoints only; the host will review and assemble the children. "
                            "Each cutpoint must include cut_reason plus exact text_before_cut and text_after_cut anchors. "
                            "Never omit either anchor; copy both from the selected legal candidate when candidate_id is used."
                        ),
                    ),
                    ("human", safe_json_dumps(prompt_payload, sort_keys=True)),
                ]

            def _emit_boundary_retry(
                record: RetryAttemptRecord[list[tuple[str, str]]],
            ) -> None:
                retry_budget = proposal_retry_rounds + 1
                delay_seconds = _bounded_retry_delay(
                    attempt_number=record.attempt_number,
                    retry_budget=retry_budget,
                    provider_settings=provider_settings,
                )
                _emit(
                    "workflow_layered_proposal_retry",
                    proposal_source="llm",
                    proposal_mode="boundaries",
                    depth=current_layer_context.depth,
                    retry_count=current_layer_context.retry_count,
                    split_strategy=split_strategy,
                    attempt=record.attempt_number,
                    retry_budget=proposal_retry_rounds,
                    retry_reason=record.error_message,
                    retry_delay_seconds=delay_seconds,
                    failure_type="retry",
                )
                if delay_seconds > 0:
                    time.sleep(delay_seconds)

            provider_attempt_index = 0

            def _invoke_boundary(
                messages: list[tuple[str, str]],
            ) -> LLMBoundaryProposalBatch:
                nonlocal \
                    boundary_dropped_count, \
                    boundary_repaired_count, \
                    provider_attempt_index
                provider_attempt_index += 1
                raw_parsed: LLMBoundaryProposalBatch = _structured_invoke(
                    chat_model,
                    LLMBoundaryProposalBatch,
                    messages,
                    timeout_seconds=(
                        provider_settings.proposal_timeout_seconds
                        or provider_settings.parser.timeout_seconds
                    ),
                    diagnostics_sink=_emit_provider_diagnostics,
                    operation="layer_boundary_proposal",
                    max_in_flight=provider_settings.parser.max_in_flight_calls,
                    attempt_index=provider_attempt_index,
                    call_role="proposal",
                    strategy=split_strategy,
                )
                normalized_parsed: LLMBoundaryProposalBatch = (
                    _normalize_boundary_cutpoints_from_candidates(
                        raw_parsed,
                        candidate_lookup=boundary_candidate_lookup,
                    )
                )
                usable: list[BoundaryCutpoint] = []
                for raw_cutpoint, normalized_cutpoint in zip(
                    raw_parsed.cutpoints,
                    normalized_parsed.cutpoints,
                    strict=False,
                ):
                    # Candidate ids are host-issued coordinates. Use their
                    # authoritative anchors, while preserving provider anchors
                    # when the model supplied them without a candidate id so
                    # ambiguous repeated text remains detectable by review.
                    cutpoint = (
                        normalized_cutpoint
                        if raw_cutpoint.candidate_id
                        or not raw_cutpoint.text_before_cut.strip()
                        or not raw_cutpoint.text_after_cut.strip()
                        else raw_cutpoint
                    )
                    if (
                        not cutpoint.text_before_cut.strip()
                        or not cutpoint.text_after_cut.strip()
                    ) and (
                        repaired_cutpoint := _repair_boundary_cutpoint_from_source(
                            cutpoint,
                            current_layer_context=current_layer_context,
                            parser_source_map=parser_source_map,
                        )
                    ) is not None:
                        cutpoint = repaired_cutpoint
                        boundary_repaired_count += 1
                        _emit(
                            "workflow_layered_boundary_cutpoint_repaired",
                            proposal_mode="boundaries",
                            depth=current_layer_context.depth,
                            retry_count=current_layer_context.retry_count,
                            candidate_id=cutpoint.candidate_id,
                            parent_node_id=cutpoint.parent_node_id,
                            source_cluster_id=cutpoint.source_cluster_id,
                            cut_offset=cutpoint.cut_offset,
                            boundary_kind=cutpoint.boundary_kind,
                            repair_source="source_excerpt",
                            text_before_cut_preview=_trim_text(
                                cutpoint.text_before_cut, max_chars=60
                            ),
                            text_after_cut_preview=_trim_text(
                                cutpoint.text_after_cut, max_chars=60
                            ),
                        )
                    if (
                        cutpoint.text_before_cut.strip()
                        and cutpoint.text_after_cut.strip()
                    ):
                        usable.append(cutpoint)
                        continue
                    boundary_dropped_count += 1
                    _emit(
                        "workflow_layered_boundary_cutpoint_dropped",
                        proposal_mode="boundaries",
                        depth=current_layer_context.depth,
                        retry_count=current_layer_context.retry_count,
                        candidate_id=cutpoint.candidate_id,
                        parent_node_id=cutpoint.parent_node_id,
                        source_cluster_id=cutpoint.source_cluster_id,
                        cut_offset=cutpoint.cut_offset,
                        reason="missing source anchors after candidate normalization",
                    )
                return normalized_parsed.model_copy(
                    update={
                        "cutpoints": usable,
                        "review_rounds": normalized_parsed.review_rounds,
                        "satisfied": normalized_parsed.satisfied,
                        "reasoning_history": list(normalized_parsed.reasoning_history),
                    }
                )

            try:
                _emit(
                    "workflow_layered_boundary_proposal_attempt",
                    proposal_mode="boundaries",
                    depth=current_layer_context.depth,
                    retry_count=current_layer_context.retry_count,
                    split_strategy=split_strategy,
                    attempt=1,
                    retry_budget=proposal_retry_rounds,
                    candidate_count=len(boundary_candidates),
                )

                def _validate_boundary_parsed(
                    parsed: LLMBoundaryProposalBatch,
                ) -> str | None:
                    unknown_candidate_ids = [
                        str(cutpoint.candidate_id)
                        for cutpoint in parsed.cutpoints
                        if cutpoint.candidate_id
                        and str(cutpoint.candidate_id) not in boundary_candidate_lookup
                    ]
                    if unknown_candidate_ids:
                        return (
                            "boundary proposal referenced unknown candidate_id values: "
                            + ", ".join(unknown_candidate_ids[:3])
                        )
                    normalized = _normalize_boundary_cutpoints_from_candidates(
                        parsed,
                        candidate_lookup=boundary_candidate_lookup,
                    )
                    return _boundary_validation_reason(
                        parsed=normalized,
                        current_layer_context=current_layer_context,
                        parser_source_map=parser_source_map,
                    )

                boundary_result = retry_with_context(
                    max_attempts=proposal_retry_rounds + 1,
                    build_request=_build_boundary_messages,
                    invoke=_invoke_boundary,
                    validate=_validate_boundary_parsed,
                    on_retry=_emit_boundary_retry,
                )
                # The invoke wrapper already applies candidate authority only
                # where needed. Preserve provider anchors otherwise so repeated
                # text remains ambiguous and is sent through review/refinement.
                boundary_parsed: LLMBoundaryProposalBatch = boundary_result.value
                _emit(
                    "workflow_layered_boundary_proposal_completed",
                    proposal_mode="boundaries",
                    depth=current_layer_context.depth,
                    retry_count=current_layer_context.retry_count,
                    split_strategy=split_strategy,
                    attempt_count=boundary_result.retry_count + 1,
                    cutpoint_count=len(boundary_parsed.cutpoints),
                    satisfied=boundary_parsed.satisfied,
                )
            except RetryExhaustedError as exc:
                failure_reason = exc.last_error
                _emit(
                    "workflow_layered_boundary_proposal_failed",
                    proposal_mode="boundaries",
                    depth=current_layer_context.depth,
                    retry_count=current_layer_context.retry_count,
                    split_strategy=split_strategy,
                    failure_reason=failure_reason,
                    failure_type=_normalized_provider_failure_type(),
                )
                fallback = fallback_builder(
                    current_layer_context=current_layer_context,
                    parser_source_map=parser_source_map,
                )
                annotated = _annotate_proposal_result(
                    fallback,
                    proposal_source="fallback",
                    proposal_mode="boundaries",
                    proposal_failure_reason=failure_reason,
                    failure_type="fallback",
                    provider_child_count=0,
                )
                _emit(
                    "workflow_layered_proposal_result",
                    proposal_source="fallback",
                    proposal_mode="boundaries",
                    proposal_failure_reason=failure_reason,
                    failure_type="fallback",
                    depth=current_layer_context.depth,
                    retry_count=current_layer_context.retry_count,
                    split_strategy=split_strategy,
                    child_count=len(annotated.children),
                    satisfied=annotated.satisfied,
                )
                return annotated

            review_decisions: list[BoundaryReviewDecision] = [
                _boundary_review_decision(
                    cutpoint=cutpoint,
                    current_layer_context=current_layer_context,
                    parser_source_map=parser_source_map,
                    max_repair_shift_chars=provider_settings.boundary_max_repair_shift_chars,
                )
                for cutpoint in boundary_parsed.cutpoints
            ]
            _emit(
                "workflow_layered_boundary_review_completed",
                proposal_mode="boundaries",
                depth=current_layer_context.depth,
                retry_count=current_layer_context.retry_count,
                split_strategy=split_strategy,
                accepted_boundary_count=sum(
                    1 for decision in review_decisions if decision.decision == "accept"
                ),
                shifted_boundary_count=sum(
                    1
                    for decision in review_decisions
                    if decision.decision in {"shift_left", "shift_right"}
                ),
                rejected_boundary_count=sum(
                    1 for decision in review_decisions if decision.decision == "reject"
                ),
                failure_type=(
                    "anchor_ambiguity"
                    if any(
                        decision.decision == "reject" for decision in review_decisions
                    )
                    else None
                ),
                refinement_needed_count=sum(
                    1
                    for decision in review_decisions
                    if decision.decision == "needs_refinement"
                ),
                review_decisions=_boundary_review_decision_summary(review_decisions),
            )
            refinement_notes: list[str] = []
            refinement_attempts = 0
            if boundary_refinement_rounds > 0:
                unresolved_targets = _boundary_refinement_targets(
                    parsed=boundary_parsed,
                    review_decisions=review_decisions,
                )
                for target in unresolved_targets[:boundary_refinement_rounds]:
                    refinement_attempts += 1
                    _emit(
                        "workflow_layered_boundary_refinement_start",
                        proposal_mode="boundaries",
                        depth=current_layer_context.depth,
                        retry_count=current_layer_context.retry_count,
                        split_strategy=split_strategy,
                        target_parent_node_id=target.parent_node_id,
                        target_source_cluster_id=target.source_cluster_id,
                        target_cut_offset=target.cut_offset,
                    )
                    refinement_prompt_payload = {
                        "task": "Refine a single ambiguous cutpoint using only nearby legal boundaries.",
                        "output_contract": "Return an LLMBoundaryProposalBatch only.",
                        "proposal_mode": proposal_mode,
                        "split_strategy": split_strategy,
                        "attempt_context": _proposal_attempt_context(
                            parse_session=parse_session,
                            current_layer_context=current_layer_context,
                        ),
                        "current_layer_context": _dump_model(current_layer_context),
                        "refinement_target": _boundary_refinement_prompt_context(
                            current_layer_context=current_layer_context,
                            parser_source_map=parser_source_map,
                            target=target,
                        ),
                        "rules": [
                            "Operate on one ambiguous cutpoint only.",
                            "Choose a nearby legal cutpoint or return no cutpoints with satisfied=true.",
                            "Do not emit prose or child text.",
                            "Preserve the original parent and source cluster scope.",
                        ],
                    }
                    refinement_messages = [
                        (
                            "system",
                            (
                                "You are refining one ambiguous semantic cutpoint in a layered document parser. "
                                "Return only structured data matching LLMBoundaryProposalBatch."
                            ),
                        ),
                        (
                            "human",
                            safe_json_dumps(refinement_prompt_payload, sort_keys=True),
                        ),
                    ]
                    try:
                        refinement_result: LLMBoundaryProposalBatch = _structured_invoke(
                            chat_model,
                            LLMBoundaryProposalBatch,
                            refinement_messages,
                            timeout_seconds=(
                                provider_settings.proposal_timeout_seconds
                                or provider_settings.parser.timeout_seconds
                            ),
                            diagnostics_sink=_emit_provider_diagnostics,
                            operation="layer_boundary_refinement",
                            max_in_flight=provider_settings.parser.max_in_flight_calls,
                        )
                        refinement_parsed: LLMBoundaryProposalBatch = refinement_result
                        refinement_validation_reason = _boundary_validation_reason(
                            parsed=refinement_parsed,
                            current_layer_context=current_layer_context,
                            parser_source_map=parser_source_map,
                        )
                        if refinement_validation_reason:
                            raise ValueError(refinement_validation_reason)
                        refinement_decisions = [
                            _boundary_review_decision(
                                cutpoint=cutpoint,
                                current_layer_context=current_layer_context,
                                parser_source_map=parser_source_map,
                                max_repair_shift_chars=provider_settings.boundary_max_repair_shift_chars,
                            )
                            for cutpoint in refinement_parsed.cutpoints
                        ]
                        if refinement_decisions:
                            replacement_map = {
                                _boundary_decision_key(decision): decision
                                for decision in review_decisions
                            }
                            for decision in refinement_decisions:
                                if decision.decision not in {
                                    "accept",
                                    "shift_left",
                                    "shift_right",
                                }:
                                    continue
                                replacement_map[_boundary_decision_key(decision)] = (
                                    decision
                                )
                            review_decisions = sorted(
                                replacement_map.values(), key=_boundary_decision_key
                            )
                            refinement_notes.extend(
                                note
                                for note in (
                                    decision.reason
                                    for decision in refinement_decisions
                                    if decision.reason
                                )
                                if note not in refinement_notes
                            )
                            _emit(
                                "workflow_layered_boundary_refinement_completed",
                                proposal_mode="boundaries",
                                depth=current_layer_context.depth,
                                retry_count=current_layer_context.retry_count,
                                split_strategy=split_strategy,
                                target_parent_node_id=target.parent_node_id,
                                target_source_cluster_id=target.source_cluster_id,
                                target_cut_offset=target.cut_offset,
                                refined_cutpoint_count=len(refinement_decisions),
                            )
                    except Exception as refinement_exc:  # noqa: BLE001 - optional refinement must fail closed
                        refinement_notes.append(
                            f"boundary refinement skipped for {target.parent_node_id}:{target.source_cluster_id}:{target.cut_offset} "
                            f"due to {refinement_exc!r}"
                        )
                        _emit(
                            "workflow_layered_boundary_refinement_failed",
                            proposal_mode="boundaries",
                            depth=current_layer_context.depth,
                            retry_count=current_layer_context.retry_count,
                            split_strategy=split_strategy,
                            target_parent_node_id=target.parent_node_id,
                            target_source_cluster_id=target.source_cluster_id,
                            target_cut_offset=target.cut_offset,
                            failure_reason=_trim_text(
                                repr(refinement_exc), max_chars=280
                            ),
                        )
            review_batch: BoundaryReviewBatch = BoundaryReviewBatch(
                decisions=review_decisions,
                satisfied=boundary_parsed.satisfied,
                coverage_ok=not any(
                    decision.decision in {"reject", "needs_refinement"}
                    for decision in review_decisions
                ),
                review_notes=[
                    decision.reason for decision in review_decisions if decision.reason
                ]
                + refinement_notes,
            )
            if boundary_parsed.satisfied is True and not boundary_parsed.cutpoints:
                _emit(
                    "workflow_layered_boundary_assembly_skipped",
                    proposal_mode="boundaries",
                    depth=current_layer_context.depth,
                    retry_count=current_layer_context.retry_count,
                    split_strategy=split_strategy,
                    reason="atomic_no_cutpoints",
                )
                runtime_result = CurrentLayerResult(
                    children=[],
                    satisfied=True,
                    reasoning_history=list(boundary_parsed.reasoning_history),
                    metadata={
                        "proposal_mode": "boundaries",
                        "proposal_source": "llm",
                        "proposal_retry_count": boundary_result.retry_count,
                        "boundary_proposed_count": 0,
                        "boundary_dropped_count": boundary_dropped_count,
                        "boundary_repaired_count": boundary_repaired_count,
                        "boundary_accepted_count": 0,
                        "boundary_shifted_count": 0,
                        "boundary_rejected_count": 0,
                        "boundary_refinement_count": 0,
                        "boundary_refinement_attempts": refinement_attempts,
                        "boundary_summary_count": 0,
                        "boundary_cutpoints": [],
                        "boundary_review_decisions": [],
                        "boundary_review_notes": list(review_batch.review_notes),
                        "boundary_atomic_decision": True,
                    },
                )
                annotated = _annotate_proposal_result(
                    runtime_result,
                    proposal_source="llm",
                    proposal_mode="boundaries",
                    boundary_count=0,
                    accepted_boundary_count=0,
                    shifted_boundary_count=0,
                    rejected_boundary_count=0,
                    refinement_count=0,
                    unresolved_interval_count=0,
                    summary_count=0,
                )
                _emit(
                    "workflow_layered_proposal_result",
                    proposal_source="llm",
                    proposal_mode="boundaries",
                    depth=current_layer_context.depth,
                    retry_count=current_layer_context.retry_count,
                    split_strategy=split_strategy,
                    boundary_count=0,
                    accepted_boundary_count=0,
                    shifted_boundary_count=0,
                    rejected_boundary_count=0,
                    refinement_count=0,
                    unresolved_interval_count=0,
                    child_count=0,
                    satisfied=True,
                )
                return annotated
            runtime_result: CurrentLayerResult
            _emit(
                "workflow_layered_boundary_assembly_start",
                proposal_mode="boundaries",
                depth=current_layer_context.depth,
                retry_count=current_layer_context.retry_count,
                split_strategy=split_strategy,
                review_decision_count=len(review_decisions),
                cutpoint_count=len(boundary_parsed.cutpoints),
            )
            runtime_result, summaries, _accepted_cutpoints = (
                _assemble_layer_result_from_boundaries(
                    current_layer_context=current_layer_context,
                    parser_source_map=parser_source_map,
                    review_batch=review_batch,
                )
            )
            identical_parent_child_ids = _boundary_identical_parent_child_ids(
                current_layer_context=current_layer_context,
                current_layer_result=runtime_result,
            )
            if identical_parent_child_ids:
                failure_reason = "boundary proposal produced no accepted cutpoints: boundary assembly produced child span identical to parent"
                _emit(
                    "workflow_layered_boundary_assembly_rejected",
                    proposal_mode="boundaries",
                    depth=current_layer_context.depth,
                    retry_count=current_layer_context.retry_count,
                    split_strategy=split_strategy,
                    failure_reason=failure_reason,
                    failure_type="semantic_rejection",
                    identical_parent_child_ids=identical_parent_child_ids,
                )
                fallback = fallback_builder(
                    current_layer_context=current_layer_context,
                    parser_source_map=parser_source_map,
                )
                fallback_identical_ids = _boundary_identical_parent_child_ids(
                    current_layer_context=current_layer_context,
                    current_layer_result=fallback,
                )
                if fallback_identical_ids:
                    fallback = CurrentLayerResult(
                        children=[],
                        satisfied=True,
                        atomic_retained=True,
                        reasoning_history=list(fallback.reasoning_history),
                        metadata=cast(
                            dict[str, JsonValue],
                            {
                                **dict(fallback.metadata),
                                "allow_empty_layer": True,
                                "boundary_rejected_child_ids": fallback_identical_ids,
                            },
                        ),
                    )
                annotated = _annotate_proposal_result(
                    fallback,
                    proposal_source="fallback",
                    proposal_mode="boundaries",
                    proposal_failure_reason=failure_reason,
                    failure_type="fallback",
                    provider_child_count=0,
                )
                annotated = annotated.model_copy(
                    update={
                        "metadata": {
                            **dict(annotated.metadata),
                            "boundary_rejected_child_ids": identical_parent_child_ids,
                        }
                    }
                )
                _emit(
                    "workflow_layered_proposal_result",
                    proposal_source="fallback",
                    proposal_mode="boundaries",
                    proposal_failure_reason=failure_reason,
                    failure_type="fallback",
                    depth=current_layer_context.depth,
                    retry_count=current_layer_context.retry_count,
                    split_strategy=split_strategy,
                    child_count=len(annotated.children),
                    satisfied=annotated.satisfied,
                    identical_parent_child_count=len(identical_parent_child_ids),
                )
                return annotated
            accepted_count = sum(
                1 for decision in review_decisions if decision.decision == "accept"
            )
            shifted_count = sum(
                1
                for decision in review_decisions
                if decision.decision in {"shift_left", "shift_right"}
            )
            rejected_count = sum(
                1 for decision in review_decisions if decision.decision == "reject"
            )
            refinement_count = sum(
                1
                for decision in review_decisions
                if decision.decision == "needs_refinement"
            )
            unresolved_interval_count = _offset_value(
                runtime_result.metadata.get("unresolved_interval_count"), 0
            )
            if accepted_count + shifted_count == 0:
                failure_reason = "boundary proposal produced no accepted cutpoints"
                _emit(
                    "workflow_layered_boundary_assembly_failed",
                    proposal_mode="boundaries",
                    depth=current_layer_context.depth,
                    retry_count=current_layer_context.retry_count,
                    split_strategy=split_strategy,
                    failure_reason=failure_reason,
                    failure_type="semantic_rejection",
                )
                fallback = fallback_builder(
                    current_layer_context=current_layer_context,
                    parser_source_map=parser_source_map,
                )
                annotated = _annotate_proposal_result(
                    fallback,
                    proposal_source="fallback",
                    proposal_mode="boundaries",
                    proposal_failure_reason=failure_reason,
                    failure_type="fallback",
                    provider_child_count=0,
                )
                _emit(
                    "workflow_layered_proposal_result",
                    proposal_source="fallback",
                    proposal_mode="boundaries",
                    proposal_failure_reason=failure_reason,
                    failure_type="fallback",
                    depth=current_layer_context.depth,
                    retry_count=current_layer_context.retry_count,
                    split_strategy=split_strategy,
                    child_count=len(annotated.children),
                    satisfied=annotated.satisfied,
                )
                return annotated
            result_metadata = {
                **runtime_result.metadata,
                "proposal_mode": "boundaries",
                "proposal_retry_count": boundary_result.retry_count,
                "boundary_proposed_count": len(boundary_parsed.cutpoints),
                "boundary_dropped_count": boundary_dropped_count,
                "boundary_repaired_count": boundary_repaired_count,
                "boundary_accepted_count": accepted_count,
                "boundary_shifted_count": shifted_count,
                "boundary_rejected_count": rejected_count,
                "boundary_refinement_count": refinement_count,
                "boundary_refinement_attempts": refinement_attempts,
                "boundary_summary_count": len(summaries),
                "boundary_cutpoints": [
                    cutpoint.model_dump() for cutpoint in boundary_parsed.cutpoints
                ],
                "boundary_review_decisions": [
                    decision.model_dump() for decision in review_decisions
                ],
                "boundary_review_notes": list(review_batch.review_notes),
                "proposal_source": "llm",
            }
            runtime_result = runtime_result.model_copy(
                update={"metadata": result_metadata}
            )
            _emit(
                "workflow_layered_boundary_assembly_completed",
                proposal_mode="boundaries",
                depth=current_layer_context.depth,
                retry_count=current_layer_context.retry_count,
                split_strategy=split_strategy,
                child_count=len(runtime_result.children),
                accepted_boundary_count=accepted_count,
                shifted_boundary_count=shifted_count,
                rejected_boundary_count=rejected_count,
                refinement_count=refinement_count,
                unresolved_interval_count=unresolved_interval_count,
                summary_count=len(summaries),
            )
            annotated = _annotate_proposal_result(
                runtime_result,
                proposal_source="llm",
                proposal_mode="boundaries",
                boundary_count=len(boundary_parsed.cutpoints),
                accepted_boundary_count=accepted_count,
                shifted_boundary_count=shifted_count,
                rejected_boundary_count=rejected_count,
                refinement_count=refinement_count,
                unresolved_interval_count=unresolved_interval_count,
                summary_count=len(summaries),
            )
            _emit(
                "workflow_layered_proposal_result",
                proposal_source="llm",
                proposal_mode="boundaries",
                depth=current_layer_context.depth,
                retry_count=current_layer_context.retry_count,
                split_strategy=split_strategy,
                boundary_count=len(boundary_parsed.cutpoints),
                accepted_boundary_count=accepted_count,
                shifted_boundary_count=shifted_count,
                rejected_boundary_count=rejected_count,
                refinement_count=refinement_count,
                unresolved_interval_count=unresolved_interval_count,
                child_count=len(runtime_result.children),
                satisfied=runtime_result.satisfied,
            )
            return annotated

        def _build_child_messages(
            attempt_number: int, previous_error: str | None
        ) -> list[tuple[str, str]]:
            prompt_payload = _proposal_attempt_payload(
                {
                    "task": "Propose the next semantic layer for the current parent nodes.",
                    "output_contract": "Return a CurrentLayerResult only. Do not emit prose.",
                    "split_strategy": split_strategy,
                    "attempt_context": _proposal_attempt_context(
                        parse_session=parse_session,
                        current_layer_context=current_layer_context,
                    ),
                    "current_layer_context": _dump_model(current_layer_context),
                    "parent_nodes": _parent_context_excerpt(
                        current_layer_context=current_layer_context,
                        parser_source_map=parser_source_map,
                    ),
                    "semantic_tree_snapshot": _summarize_for_prompt(
                        semantic_tree,
                        max_depth=4,
                        max_items=5,
                        max_string=280,
                    ),
                    "full_document_context": _summarize_for_prompt(
                        parser_input_dict,
                        max_depth=4,
                        max_items=5,
                        max_string=280,
                    ),
                    "source_map_excerpt": _source_map_excerpt(parser_source_map),
                    "rules": [
                        "Operate on ONE layer only: propose only the immediate children for the current parent nodes.",
                        "Do not edit ancestors or descendants and do not break down current children directly.",
                        "Children together must preserve the parent meaning collectively.",
                        "Each parent should end up with more than one child or no child at all.",
                        "If a parent is already atomic, return no children for that parent and use satisfied=true only when the current layer is complete.",
                        "Prefer a coarser layerwise breakdown over premature deep flattening so later iterations can refine substructure.",
                        "Do not recombine separated verbatim fragments into invented longer text; keep grounding exact and local.",
                        "Every pointer must use exact source_cluster_id values plus character spans from the supplied source map.",
                        "Set expandable=false for leaf nodes and expandable=true only when later subdivision is still appropriate.",
                    ],
                },
                attempt_index=attempt_number - 1,
                prior_error=previous_error,
            )
            return [
                (
                    "system",
                    (
                        "You are revising ONE semantic layer in an iterative document parsing workflow. "
                        "Return only structured data matching CurrentLayerResult. "
                        "Produce grounded immediate children for the supplied parents and preserve layerwise semantics."
                    ),
                ),
                ("human", safe_json_dumps(prompt_payload, sort_keys=True)),
            ]

        def _emit_child_retry(
            record: RetryAttemptRecord[list[tuple[str, str]]],
        ) -> None:
            retry_budget = proposal_retry_rounds + 1
            delay_seconds = _bounded_retry_delay(
                attempt_number=record.attempt_number,
                retry_budget=retry_budget,
                provider_settings=provider_settings,
            )
            _emit(
                "workflow_layered_proposal_retry",
                proposal_source="llm",
                proposal_mode="children",
                depth=current_layer_context.depth,
                retry_count=current_layer_context.retry_count,
                split_strategy=split_strategy,
                attempt=record.attempt_number,
                retry_budget=proposal_retry_rounds,
                retry_reason=record.error_message,
                retry_delay_seconds=delay_seconds,
                failure_type="retry",
            )
            if delay_seconds > 0:
                time.sleep(delay_seconds)

        provider_attempt_index = 0

        def _invoke_child_provider(
            messages: list[tuple[str, str]],
        ) -> LLMCurrentLayerResult:
            nonlocal provider_attempt_index
            provider_attempt_index += 1
            return _structured_invoke(
                chat_model,
                LLMCurrentLayerResult,
                messages,
                timeout_seconds=(
                    provider_settings.proposal_timeout_seconds
                    or provider_settings.parser.timeout_seconds
                ),
                diagnostics_sink=_emit_provider_diagnostics,
                operation="layer_children_proposal",
                max_in_flight=provider_settings.parser.max_in_flight_calls,
                attempt_index=provider_attempt_index,
                call_role="proposal",
                strategy=split_strategy,
            )

        try:
            _emit(
                "workflow_layered_child_proposal_start",
                proposal_mode="children",
                depth=current_layer_context.depth,
                retry_count=current_layer_context.retry_count,
                split_strategy=split_strategy,
                parent_count=len(current_layer_context.parent_node_ids),
            )
            child_result = retry_with_context(
                max_attempts=proposal_retry_rounds + 1,
                build_request=_build_child_messages,
                invoke=_invoke_child_provider,
                validate=lambda parsed: _proposal_validation_reason(
                    parsed=parsed,
                    current_layer_context=current_layer_context,
                    parser_source_map=parser_source_map,
                ),
                on_retry=_emit_child_retry,
            )
            child_parsed: LLMCurrentLayerResult = child_result.value
            _emit(
                "workflow_layered_child_proposal_completed",
                proposal_mode="children",
                depth=current_layer_context.depth,
                retry_count=current_layer_context.retry_count,
                split_strategy=split_strategy,
                attempt_count=child_result.retry_count + 1,
                child_count=len(child_parsed.children),
                satisfied=child_parsed.satisfied,
            )
        except RetryExhaustedError as exc:
            failure_reason = exc.last_error
            _emit(
                "workflow_layered_child_proposal_failed",
                proposal_mode="children",
                depth=current_layer_context.depth,
                retry_count=current_layer_context.retry_count,
                split_strategy=split_strategy,
                failure_reason=failure_reason,
                failure_type=_normalized_provider_failure_type(),
            )
            fallback = fallback_builder(
                current_layer_context=current_layer_context,
                parser_source_map=parser_source_map,
            )
            annotated = _annotate_proposal_result(
                fallback,
                proposal_source="fallback",
                proposal_mode="children",
                proposal_failure_reason=failure_reason,
                failure_type="fallback",
                provider_child_count=0,
            )
            _emit(
                "workflow_layered_proposal_result",
                proposal_source="fallback",
                proposal_mode="children",
                proposal_failure_reason=failure_reason,
                failure_type="fallback",
                depth=current_layer_context.depth,
                retry_count=current_layer_context.retry_count,
                split_strategy=split_strategy,
                child_count=len(annotated.children),
                satisfied=annotated.satisfied,
            )
            return annotated

        runtime_result = _hydrate_llm_layer_result(
            child_parsed,
            parser_source_map=parser_source_map,
        )
        runtime_result = runtime_result.model_copy(
            update={
                "metadata": {
                    **runtime_result.metadata,
                    "proposal_retry_count": child_result.retry_count,
                }
            }
        )
        annotated = _annotate_proposal_result(
            runtime_result,
            proposal_source="llm",
            proposal_mode="children",
            provider_child_count=len(runtime_result.children),
        )
        _emit(
            "workflow_layered_child_proposal_assembled",
            proposal_mode="children",
            depth=current_layer_context.depth,
            retry_count=current_layer_context.retry_count,
            split_strategy=split_strategy,
            child_count=len(runtime_result.children),
            satisfied=runtime_result.satisfied,
        )
        _emit(
            "workflow_layered_proposal_result",
            proposal_source="llm",
            proposal_mode="children",
            depth=current_layer_context.depth,
            retry_count=current_layer_context.retry_count,
            split_strategy=split_strategy,
            child_count=len(runtime_result.children),
            satisfied=runtime_result.satisfied,
        )
        return annotated

    def _review_layer_fn(
        *,
        current_layer_context: CurrentLayerContext,
        current_layer_result: CurrentLayerResult,
        split_strategy: SplitStrategy,
        parser_source_map: ParserSourceMap | None = None,
        parse_session: ParseSessionState | None = None,
        **kwargs: object,
    ) -> CurrentLayerReview:
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
                parser_source_map=parser_source_map or {},
            ),
            "source_map_excerpt": _source_map_excerpt(parser_source_map or {}),
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
            (
                "human",
                safe_json_dumps(review_payload, sort_keys=True),
            ),
        ]
        _emit(
            "workflow_layered_review_start",
            review_source="llm",
            depth=current_layer_context.depth,
            retry_count=current_layer_context.retry_count,
            split_strategy=split_strategy,
            child_count=len(current_layer_result.children),
        )
        try:
            review_result: LLMCurrentLayerReview = _structured_invoke(
                chat_model,
                LLMCurrentLayerReview,
                messages,
                timeout_seconds=(
                    provider_settings.review_timeout_seconds
                    or provider_settings.parser.timeout_seconds
                ),
                diagnostics_sink=_emit_provider_diagnostics,
                operation="layer_review",
                max_in_flight=provider_settings.parser.max_in_flight_calls,
                attempt_index=1,
                call_role="review",
                strategy=split_strategy,
            )
            reviewed: LLMCurrentLayerReview = review_result
            runtime_review = _hydrate_llm_review(
                reviewed,
                parser_source_map=parser_source_map or {},
            )
            _emit(
                "workflow_layered_review_result",
                review_source="llm",
                depth=current_layer_context.depth,
                retry_count=current_layer_context.retry_count,
                split_strategy=split_strategy,
                satisfied=runtime_review.satisfied,
                coverage_ok=runtime_review.coverage_ok,
            )
            _emit(
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
                        else _normalized_provider_failure_type()
                    ),
                },
            )
            _emit(
                "workflow_layered_review_result",
                review_source="fallback",
                review_failure_reason=failure_reason,
                depth=current_layer_context.depth,
                retry_count=current_layer_context.retry_count,
                split_strategy=split_strategy,
                satisfied=fallback_review.satisfied,
                coverage_ok=fallback_review.coverage_ok,
            )
            _emit(
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

    return {
        "propose_layer_fn": _propose_layer_fn,
        "review_layer_fn": _review_layer_fn,
        "max_depth": max_depth,
        "allow_review": allow_review,
    }
