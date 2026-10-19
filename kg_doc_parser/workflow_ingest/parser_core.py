from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Literal, cast

from pydantic import ValidationError

from .cache import WorkflowLLMCallCache
from .models import (
    CurrentLayerContext,
    CurrentLayerResult,
    CurrentLayerReview,
    LayerChildCandidate,
    LayerValidationDiagnostic,
    ParseSessionState,
)
from .parser_core_commit import (
    coerce_layer_child,
    commit_layer_children,  # noqa: F401 - preserve callable module API
    enqueue_next_layer_frontier,  # noqa: F401 - preserve callable module API
    finalize_semantic_tree,  # noqa: F401 - preserve callable module API
    find_semantic_node,  # noqa: F401 - preserve callable module API
    requeue_failed_frontier_items,  # noqa: F401 - preserve callable module API
    validate_layer_commit,  # noqa: F401 - preserve callable module API
)
from .parser_core_contracts import (
    ParserPayload,
    ParserSourceMap,
    ParseSemanticFn,  # noqa: F401 - preserve callable module API
    PointerCorrector,
    ProposeLayerFn,
    ReviewLayerFn,
    SourceCollectionLike,
    SourceCollectionWithPagesLike,  # noqa: F401 - preserve callable module API
    SourcePageLike,  # noqa: F401 - preserve callable module API
    SourceUnitLike,  # noqa: F401 - preserve callable module API
    SplitStrategy,  # noqa: F401 - preserve callable module API
)
from .parser_core_contracts import (
    required_node_id as _required_node_id,  # noqa: F401 - preserve callable module API
)
from .parser_core_diagnostics import (
    _child_pointer_fingerprint,
    _layer_structural_issues,
)
from .parser_core_legacy import (
    _canonicalize_legacy_pointer_tree,  # noqa: F401 - preserve callable module API
    _coerce_semantic_tree,  # noqa: F401 - preserve callable module API
    _root_only,  # noqa: F401 - preserve callable module API
    _source_text,
)
from .parser_core_session import (
    assert_source_map_matches_session as _assert_source_map_matches_session,
)
from .parser_core_session import (
    initialize_parse_session,  # noqa: F401 - preserve callable module API
    legacy_children_for_context,
    prepare_layer_frontier,  # noqa: F401 - preserve callable module API
    source_map_fingerprint,  # noqa: F401 - preserve callable module API
)
from .parser_core_validation import detect_layer_invariants
from .semantics import (
    HydratedTextPointer,
    SemanticNode,
)
from .serialization import JsonValue, json_safe

_LOGGER = logging.getLogger(__name__)
def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _validation_diagnostics(
    *,
    phase: Literal["proposal", "review", "repair", "commit"],
    error: Exception,
) -> list[LayerValidationDiagnostic]:
    """Convert bounded Pydantic or adapter errors into retryable state."""

    details = error.errors(include_context=False) if isinstance(error, ValidationError) else []
    diagnostics: list[LayerValidationDiagnostic] = []
    for detail in details[:8]:
        raw_location = detail.get("loc", ())
        location = [str(item) for item in raw_location] if isinstance(raw_location, tuple) else []
        diagnostics.append(
            LayerValidationDiagnostic(
                phase=phase,
                error_type=str(detail.get("type", type(error).__name__)),
                message=str(detail.get("msg", str(error)))[:500],
                location=location,
            )
        )
    if not diagnostics:
        diagnostics.append(
            LayerValidationDiagnostic(
                phase=phase,
                error_type=type(error).__name__,
                message=str(error)[:500],
            )
        )
    return diagnostics


def _invalid_layer_result(
    *,
    phase: Literal["proposal", "review", "repair", "commit"],
    error: Exception,
) -> CurrentLayerResult:
    diagnostics = _validation_diagnostics(phase=phase, error=error)
    return CurrentLayerResult(
        children=[],
        satisfied=False,
        validation_diagnostics=diagnostics,
        metadata={
            "validation_failure": phase,
            "retryable": True,
            "validation_error": diagnostics[0].message,
        },
    )


def default_parse_semantic_fn(
    *,
    collection: SourceCollectionLike,
    parser_input_dict: ParserPayload,
    parser_source_map: ParserSourceMap,
    model_names: list[str] | None = None,
) -> object:
    from ..semantic_document_splitting_layerwise_edits import build_document_tree

    return build_document_tree(
        doc_id=collection.collection_id,
        llm_input_dict=parser_input_dict,
        source_map=parser_source_map,
        model_names=model_names,
    )


def propose_layer_breakdown(
    *,
    collection: SourceCollectionLike,
    parser_input_dict: ParserPayload,
    parser_source_map: ParserSourceMap,
    parse_session: ParseSessionState,
    current_layer_context: CurrentLayerContext,
    semantic_tree: SemanticNode,
    propose_layer_fn: ProposeLayerFn | None = None,
    llm_cache: WorkflowLLMCallCache | None = None,
    cache_namespace: Mapping[str, JsonValue] | None = None,
) -> CurrentLayerResult:
    _assert_source_map_matches_session(
        parse_session=parse_session,
        parser_source_map=parser_source_map,
    )
    if parse_session.mode == "legacy_compat":
        return legacy_children_for_context(
            parse_session=parse_session,
            current_layer_context=current_layer_context,
        )
    if propose_layer_fn is None:
        raise ValueError("workflow_layered mode requires propose_layer_fn")
    def call() -> CurrentLayerResult | Mapping[str, object] | Sequence[object]:
        return propose_layer_fn(
            collection=collection,
            parser_input_dict=parser_input_dict,
            parser_source_map=parser_source_map,
            parse_session=parse_session,
            current_layer_context=current_layer_context,
            semantic_tree=semantic_tree,
            split_strategy=current_layer_context.split_strategy,
        )
    try:
        if llm_cache is not None:
            proposed = llm_cache.cached_call(
                operation="propose_layer_breakdown",
                fingerprint={
                    "cache_namespace": dict(cache_namespace or {}),
                    "collection_id": collection.collection_id,
                    "parse_session": json_safe(parse_session),
                    "current_layer_context": json_safe(current_layer_context),
                    "semantic_tree": json_safe(semantic_tree),
                    "parser_input_dict": json_safe(parser_input_dict),
                    "parser_source_map": json_safe(parser_source_map),
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
            return CurrentLayerResult(children=[coerce_layer_child(child) for child in cast(list[object], proposed)])
        raise TypeError("unsupported proposed layer result")
    except (ValidationError, TypeError, ValueError) as exc:
        return _invalid_layer_result(phase="proposal", error=exc)


def review_layer(
    *,
    parse_session: ParseSessionState,
    current_layer_context: CurrentLayerContext,
    current_layer_result: CurrentLayerResult,
    parser_source_map: ParserSourceMap | None = None,
    review_layer_fn: ReviewLayerFn | None = None,
    llm_cache: WorkflowLLMCallCache | None = None,
    cache_namespace: Mapping[str, JsonValue] | None = None,
) -> tuple[CurrentLayerReview, ParseSessionState]:
    if parser_source_map is not None:
        _assert_source_map_matches_session(
            parse_session=parse_session,
            parser_source_map=parser_source_map,
        )
    if parse_session.mode == "legacy_compat":
        (
            coverage_ok,
            invariant_satisfied,
            overlap_conflicts,
            coverage_gaps,
            duplicate_notes,
            invariant_notes,
        ) = detect_layer_invariants(
            current_layer_context=current_layer_context,
            current_layer_result=current_layer_result,
            parser_source_map=parser_source_map,
        )
        structural_issues = _layer_structural_issues(
            current_layer_context=current_layer_context,
            current_layer_result=current_layer_result,
        )
        invariant_satisfied = invariant_satisfied and not structural_issues
        return (
            CurrentLayerReview(
                updated_result=current_layer_result.model_copy(update={"satisfied": invariant_satisfied}),
                coverage_ok=coverage_ok,
                satisfied=invariant_satisfied,
                strategy_used=current_layer_context.split_strategy,
                overlap_conflicts=overlap_conflicts,
                coverage_gap_notes=coverage_gaps,
                duplicate_child_notes=duplicate_notes,
                structural_issues=structural_issues,
                review_notes=["semantic review skipped for legacy compatibility", *invariant_notes[:20]],
                metadata={"review_skipped": True, "legacy_compat": True, "deterministic_validation": True},
            ),
            parse_session,
        )
    if not parse_session.allow_review:
        coverage_ok, invariant_satisfied, overlap_conflicts, coverage_gaps, duplicate_notes, invariant_notes = detect_layer_invariants(
            current_layer_context=current_layer_context,
            current_layer_result=current_layer_result,
            parser_source_map=parser_source_map,
        )
        return (
            CurrentLayerReview(
                updated_result=current_layer_result.model_copy(update={"satisfied": invariant_satisfied}),
                coverage_ok=coverage_ok,
                satisfied=invariant_satisfied,
                strategy_used=current_layer_context.split_strategy,
                overlap_conflicts=overlap_conflicts,
                coverage_gap_notes=coverage_gaps,
                duplicate_child_notes=duplicate_notes,
                structural_issues=_layer_structural_issues(
                    current_layer_context=current_layer_context,
                    current_layer_result=current_layer_result,
                ),
                review_notes=["semantic review disabled", *invariant_notes[:20]],
                metadata={"review_skipped": True, "deterministic_validation": True},
            ),
            parse_session,
        )
    if review_layer_fn is None:
        reviewed = current_layer_result
    else:
        def call() -> CurrentLayerReview | CurrentLayerResult | Mapping[str, object]:
            return review_layer_fn(
                parse_session=parse_session,
                current_layer_context=current_layer_context,
                current_layer_result=current_layer_result,
                parser_source_map=parser_source_map or {},
                split_strategy=current_layer_context.split_strategy,
            )
        try:
            if llm_cache is not None:
                reviewed = llm_cache.cached_call(
                    operation=f"review_cud_proposal:{current_layer_context.split_strategy}",
                    fingerprint={
                        "cache_namespace": dict(cache_namespace or {}),
                        "parse_session": json_safe(parse_session),
                        "current_layer_context": json_safe(current_layer_context),
                        "current_layer_result": json_safe(current_layer_result),
                        "parser_source_map": json_safe(parser_source_map or {}),
                        "split_strategy": current_layer_context.split_strategy,
                    },
                    fn=call,
                )
            else:
                reviewed = call()
        except TimeoutError as exc:
            # A review timeout is different from a malformed or failed
            # provider response.  If the host-side invariants are complete,
            # deterministic validation can authorize this layer without
            # pretending that the semantic critic ran.
            coverage_ok, invariant_satisfied, overlap_conflicts, coverage_gaps, duplicate_notes, invariant_notes = detect_layer_invariants(
                current_layer_context=current_layer_context,
                current_layer_result=current_layer_result,
                parser_source_map=parser_source_map,
            )
            reviewed = CurrentLayerReview(
                updated_result=current_layer_result,
                coverage_ok=coverage_ok,
                satisfied=invariant_satisfied,
                strategy_used=current_layer_context.split_strategy,
                overlap_conflicts=overlap_conflicts,
                coverage_gap_notes=coverage_gaps,
                duplicate_child_notes=duplicate_notes,
                structural_issues=_layer_structural_issues(
                    current_layer_context=current_layer_context,
                    current_layer_result=current_layer_result,
                ),
                review_notes=[
                    "quality_unknown: semantic layer review timed out",
                    "deterministic invariants authorized the layer" if invariant_satisfied else "deterministic invariants rejected the layer",
                    *invariant_notes[:10],
                ],
                metadata={
                    "review_timeout": True,
                    "review_timeout_reason": repr(exc)[:500],
                },
            )
        except Exception as exc:  # noqa: BLE001 - provider boundary must fail closed
            reviewed = CurrentLayerReview(
                updated_result=current_layer_result,
                coverage_ok=None,
                satisfied=None,
                strategy_used=current_layer_context.split_strategy,
                review_notes=[
                    "quality_unknown: semantic layer review provider failed",
                    "deterministic checks did not authorize successful review",
                ],
                metadata={
                    "review_failure": "provider_failure",
                    "review_failure_reason": repr(exc)[:500],
                },
            )
    try:
        if isinstance(reviewed, CurrentLayerReview):
            result = reviewed
        elif isinstance(reviewed, CurrentLayerResult):
            result = CurrentLayerReview(
                updated_result=reviewed,
                coverage_ok=_optional_bool(reviewed.metadata.get("layer_coverage_ok")),
                satisfied=reviewed.satisfied,
                validation_diagnostics=list(reviewed.validation_diagnostics),
            )
        elif isinstance(reviewed, dict):
            if "updated_result" in reviewed or "coverage_ok" in reviewed or "review_notes" in reviewed:
                result = CurrentLayerReview.model_validate(reviewed)
            else:
                parsed = CurrentLayerResult.model_validate(reviewed)
                result = CurrentLayerReview(
                    updated_result=parsed,
                    coverage_ok=_optional_bool(parsed.metadata.get("layer_coverage_ok")),
                    satisfied=parsed.satisfied,
                    validation_diagnostics=list(parsed.validation_diagnostics),
                )
        else:
            raise TypeError("unsupported reviewed layer result")
    except (ValidationError, TypeError, ValueError) as exc:
        invalid_result = _invalid_layer_result(phase="review", error=exc)
        result = CurrentLayerReview(
            updated_result=invalid_result,
            coverage_ok=None,
            satisfied=None,
            validation_diagnostics=list(invalid_result.validation_diagnostics),
            review_notes=["quality_unknown: semantic layer review payload failed validation"],
            metadata={
                "review_failure": "structured_validation",
                "retryable": True,
            },
        )
    coverage_ok, invariant_satisfied, overlap_conflicts, coverage_gaps, duplicate_notes, invariant_notes = detect_layer_invariants(
        current_layer_context=current_layer_context,
        current_layer_result=result.updated_result or current_layer_result,
        parser_source_map=parser_source_map,
    )
    merged_notes = list(result.review_notes)
    for note in invariant_notes:
        if note not in merged_notes:
            merged_notes.append(note)
    provider_failure = bool(result.metadata.get("review_failure"))
    structural_issues = _layer_structural_issues(
        current_layer_context=current_layer_context,
        current_layer_result=result.updated_result or current_layer_result,
    )
    for issue in structural_issues:
        if issue.message not in merged_notes:
            merged_notes.append(issue.message)
    validation_diagnostics = list(result.validation_diagnostics)
    for diagnostic in (result.updated_result or current_layer_result).validation_diagnostics:
        if diagnostic not in validation_diagnostics:
            validation_diagnostics.append(diagnostic)
    base_satisfied = result.satisfied if result.satisfied is not None else invariant_satisfied
    satisfied = (
        None
        if provider_failure
        else bool(
            base_satisfied
            and not (overlap_conflicts or coverage_gaps or duplicate_notes or structural_issues)
            and not validation_diagnostics
        )
    )
    effective_coverage_ok = None if provider_failure else coverage_ok
    updated_result = (result.updated_result or current_layer_result).model_copy(
        update={
            # A malformed provider payload is quality-unknown at the review
            # layer, but its persisted result must still be explicitly
            # unsatisfied so routing cannot treat it as a usable result.
            "satisfied": False if validation_diagnostics else satisfied,
            "metadata": {
                **(result.updated_result or current_layer_result).metadata,
                **result.metadata,
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
                "structural_issues": [
                    item.model_dump(field_mode="backend", dump_format="json")
                    for item in structural_issues
                ],
                "validation_diagnostics": [
                    item.model_dump(field_mode="backend", dump_format="json")
                    for item in validation_diagnostics
                ],
            },
        }
    )
    result = result.model_copy(
        update={
            "updated_result": updated_result,
            "coverage_ok": effective_coverage_ok,
            "satisfied": satisfied,
            "strategy_used": current_layer_context.split_strategy,
            "overlap_conflicts": overlap_conflicts,
            "coverage_gap_notes": coverage_gaps,
            "duplicate_child_notes": duplicate_notes,
            "structural_issues": structural_issues,
            "validation_diagnostics": validation_diagnostics,
            "review_notes": merged_notes,
        }
    )
    attempts = dict(parse_session.layer_attempts)
    frontier_identity = str(
        current_layer_context.metadata.get(
            "frontier_identity",
            f"depth={current_layer_context.depth};parents={','.join(current_layer_context.parent_node_ids)}",
        )
    )
    attempts[frontier_identity] = current_layer_context.retry_count + 1
    review_packet = {
        "strategy": result.strategy_used,
        "coverage_ok": result.coverage_ok,
        "satisfied": result.satisfied,
        "review_notes": list(result.review_notes),
        "overlap_conflicts": [
            item.model_dump(field_mode="backend", dump_format="json")
            for item in result.overlap_conflicts
        ],
        "coverage_gap_notes": [
            item.model_dump(field_mode="backend", dump_format="json")
            for item in result.coverage_gap_notes
        ],
        "duplicate_child_notes": [
            item.model_dump(field_mode="backend", dump_format="json")
            for item in result.duplicate_child_notes
        ],
        "structural_issues": [
            item.model_dump(field_mode="backend", dump_format="json")
            for item in result.structural_issues
        ],
        "validation_diagnostics": [
            item.model_dump(field_mode="backend", dump_format="json")
            for item in result.validation_diagnostics
        ],
        "metadata": dict(result.metadata),
    }
    return result, parse_session.model_copy(
        update={"layer_attempts": attempts, "last_review": review_packet}
    )


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
    if current_layer_review.structural_issues:
        metadata["structural_issues"] = [
            item.model_dump(field_mode="backend", dump_format="json")
            for item in current_layer_review.structural_issues
        ]
    if current_layer_review.validation_diagnostics:
        metadata["validation_diagnostics"] = [
            item.model_dump(field_mode="backend", dump_format="json")
            for item in current_layer_review.validation_diagnostics
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
    if current_layer_review is not None and current_layer_review.metadata.get("review_failure"):
        return False, ["layer review unavailable; quality is unknown"]
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
        raw_notes = current_layer_result.metadata.get("review_notes")
        notes = raw_notes if isinstance(raw_notes, list) else []
        return metadata_flag, [str(note) for note in notes]

    if current_layer_result.atomic_retained:
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
    parser_source_map: ParserSourceMap,
    correct_pointer_fn: PointerCorrector,
) -> tuple[CurrentLayerResult, int]:
    def _pointer_context(pointer: HydratedTextPointer) -> str:
        source = parser_source_map.get(pointer.source_cluster_id)
        text = _source_text(source)
        preview = text.replace("\n", "\\n").replace("\t", "\\t")
        if len(preview) > 120:
            preview = preview[:117] + "..."
        return (
            f"source_cluster_id={pointer.source_cluster_id!r}, "
            f"span=({pointer.start_char},{pointer.end_char}), "
            f"verbatim_text={pointer.verbatim_text!r}, "
            f"source_text={preview!r}"
        )

    repaired_children: list[LayerChildCandidate] = []
    repair_failures: list[str] = []
    repair_diagnostics: list[LayerValidationDiagnostic] = []
    repaired_count = 0
    for child in current_layer_result.children:
        repaired_ptrs = []
        child_failed = False
        for pointer in child.total_content_pointers:
            try:
                fixed = correct_pointer_fn(pointer, parser_source_map)
            except (ValidationError, TypeError, ValueError) as exc:
                repair_diagnostics.extend(_validation_diagnostics(phase="repair", error=exc))
                message = (
                    f"pointer repair validation failed for child {child.title!r} "
                    f"(parent={child.parent_node_id!r}, node_id={child.node_id!r}); "
                    f"{type(exc).__name__}: {exc}"
                )
                _LOGGER.warning("repair_layer_candidates failed: %s", message)
                repair_failures.append(message)
                child_failed = True
                break
            if fixed is None:
                message = (
                    f"unrecoverable pointer for child {child.title!r} "
                    f"(parent={child.parent_node_id!r}, node_id={child.node_id!r}); "
                    f"{_pointer_context(pointer)}"
                )
                _LOGGER.warning("repair_layer_candidates failed: %s", message)
                repair_failures.append(message)
                child_failed = True
                break
            if fixed.model_dump() != pointer.model_dump():
                repaired_count += 1
            repaired_ptrs.append(fixed)
        if child_failed:
            # A bad proposal must not erase verified siblings or abort an
            # unrelated parent.  Omit only this replacement; the invariant
            # checker will report the affected parent's coverage gap.
            continue
        repaired_children.append(
            child.model_copy(update={"total_content_pointers": repaired_ptrs})
        )
    metadata = dict(current_layer_result.metadata)
    if repair_failures:
        metadata["repair_failures"] = cast(JsonValue, repair_failures[:32])
        metadata["repair_failure_scope"] = "child_replacement"
        metadata["failure_type"] = "repair_failure"
        metadata["rollback"] = "verified_parent_retained"
        metadata["retryable"] = True
    validation_diagnostics = list(current_layer_result.validation_diagnostics)
    for diagnostic in repair_diagnostics:
        if diagnostic not in validation_diagnostics:
            validation_diagnostics.append(diagnostic)
    return current_layer_result.model_copy(
        update={
            "children": repaired_children,
            "metadata": metadata,
            "validation_diagnostics": validation_diagnostics,
            "satisfied": False if validation_diagnostics else current_layer_result.satisfied,
        }
    ), repaired_count


def dedupe_and_filter_layer(
    *,
    current_layer_context: CurrentLayerContext,
    current_layer_result: CurrentLayerResult,
) -> CurrentLayerResult:
    parent_ids = set(current_layer_context.parent_node_ids)
    seen: set[tuple[str, str, str, tuple[tuple[str, int, int, str], ...]]] = set()
    filtered: list[LayerChildCandidate] = []
    validation_diagnostics = list(current_layer_result.validation_diagnostics)
    for child in current_layer_result.children:
        if child.parent_node_id not in parent_ids:
            diagnostic = LayerValidationDiagnostic(
                phase="commit",
                error_type="foreign_parent",
                message=(
                    f"child {child.node_id!r} references parent "
                    f"{child.parent_node_id!r} outside the current layer"
                ),
                location=[child.node_id, child.parent_node_id],
            )
            if diagnostic not in validation_diagnostics:
                validation_diagnostics.append(diagnostic)
            continue
        key = (
            child.parent_node_id,
            child.node_type,
            child.title.strip().lower(),
            _child_pointer_fingerprint(child),
        )
        if key in seen:
            continue
        seen.add(key)
        filtered.append(child)
    return current_layer_result.model_copy(
        update={
            "children": filtered,
            "validation_diagnostics": validation_diagnostics,
            "satisfied": current_layer_result.satisfied and not validation_diagnostics,
        }
    )
