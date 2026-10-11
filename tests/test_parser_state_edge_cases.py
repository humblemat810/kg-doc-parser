from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from kg_doc_parser.workflow_ingest.handlers import (
    _coverage_map,
    _log_runtime_progress,
    _state_float,
    _state_int,
    _strategy_attempt,
)
from kg_doc_parser.workflow_ingest.models import (
    BoundaryCutpoint,
    BoundaryReviewDecision,
    BoundaryUnitSummary,
    CurrentLayerContext,
    CurrentLayerResult,
    LayerCoverageGap,
    LayerFrontierItem,
    LayerSpanConflict,
    LLMCurrentLayerResult,
    LLMTextPointer,
    ParseSessionState,
    StrategyExecutionRecord,
)
from kg_doc_parser.workflow_ingest.page_index import _increment_diagnostic
from kg_doc_parser.workflow_ingest.parser_core import (
    initialize_parse_session,
    propose_layer_breakdown,
    source_map_fingerprint,
)
from kg_doc_parser.workflow_ingest.semantics import HydratedTextPointer, SemanticNode

pytestmark = pytest.mark.ci


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("current_depth", -1),
        ("max_depth", -1),
        ("strategy_switch_count", -1),
    ],
)
def test_parse_session_rejects_negative_progress_state(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        ParseSessionState(
            collection_id="doc",
            root_node_id="doc|root",
            **{field: value},
        )


def test_parse_session_rejects_depth_past_configured_limit() -> None:
    with pytest.raises(ValidationError, match="current_depth"):
        ParseSessionState(
            collection_id="doc",
            root_node_id="doc|root",
            current_depth=4,
            max_depth=3,
        )


def test_new_parse_session_rejects_replacement_source_map() -> None:
    source_map = {"doc|p1_t0": {"text": "authoritative source"}}
    session, _frontier, root = initialize_parse_session(
        collection=SimpleNamespace(collection_id="doc", title="Document"),
        parser_input_dict={},
        parser_source_map=source_map,
    )

    with pytest.raises(ValueError, match="does not match the parse session source revision"):
        propose_layer_breakdown(
            collection=SimpleNamespace(collection_id="doc", title="Document"),
            parser_input_dict={},
            parser_source_map={"doc|p1_t0": {"text": "replacement source"}},
            parse_session=session,
            current_layer_context=CurrentLayerContext(depth=0),
            semantic_tree=root,
            propose_layer_fn=lambda **_kwargs: CurrentLayerResult(children=[]),
        )


def test_source_map_fingerprint_is_independent_of_mapping_order() -> None:
    first = {
        "doc|p2_t0": {"text": "second"},
        "doc|p1_t0": {"text": "first"},
    }
    second = {
        "doc|p1_t0": {"text": "first"},
        "doc|p2_t0": {"text": "second"},
    }

    assert source_map_fingerprint(first) == source_map_fingerprint(second)


def test_source_map_fingerprint_changes_when_authoritative_text_changes() -> None:
    original = {"doc|p1_t0": {"text": "authoritative source"}}
    changed = {"doc|p1_t0": {"text": "authoritative source!"}}

    assert source_map_fingerprint(original) != source_map_fingerprint(changed)


def test_legacy_parse_session_without_source_fingerprint_remains_readable() -> None:
    session = ParseSessionState(
        collection_id="doc",
        root_node_id="doc|root",
        mode="legacy_compat",
        compat_full_tree={"node_id": "doc|root", "title": "Document"},
    )
    result = propose_layer_breakdown(
        collection=SimpleNamespace(collection_id="doc", title="Document"),
        parser_input_dict={},
        parser_source_map={"doc|p1_t0": {"text": "source"}},
        parse_session=session,
        current_layer_context=CurrentLayerContext(depth=0),
        semantic_tree=SemanticNode(node_id="doc|root", title="Document"),
    )

    assert result.satisfied is True


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LayerFrontierItem(parent_node_id="root", depth=-1),
        lambda: LayerFrontierItem(parent_node_id="root", depth=0, order=-1),
        lambda: CurrentLayerContext(depth=-1),
        lambda: CurrentLayerContext(depth=0, retry_count=-1),
        lambda: CurrentLayerContext(depth=0, max_retries=-1),
        lambda: CurrentLayerResult(review_rounds=-1),
        lambda: LLMCurrentLayerResult(review_rounds=-1),
        lambda: BoundaryCutpoint(
            parent_node_id="root",
            source_cluster_id="doc|p1_t0",
            cut_offset=-1,
            boundary_kind="paragraph",
        ),
    ],
)
def test_progress_models_reject_negative_counters(factory) -> None:
    with pytest.raises(ValidationError):
        factory()


def test_layer_context_allows_initial_attempt_count_above_retry_budget() -> None:
    context = CurrentLayerContext(depth=0, retry_count=1, max_retries=0)
    assert context.retry_count == 1


@pytest.mark.parametrize(
    ("start_char", "end_char"),
    [(True, 0), (0, False), ("0", 0), (0, "0")],
)
def test_provider_pointer_rejects_coercible_non_integer_offsets(
    start_char: object,
    end_char: object,
) -> None:
    with pytest.raises(ValidationError):
        LLMTextPointer(
            source_cluster_id="doc|p1_t0",
            start_char=start_char,
            end_char=end_char,
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"parent_node_id": "root", "source_cluster_id": "doc|p1_t0", "depth": True},
        {"parent_node_id": "root", "source_cluster_id": "doc|p1_t0", "depth": 0, "order": "1"},
        {
            "parent_node_id": "root",
            "source_cluster_id": "doc|p1_t0",
            "cut_offset": True,
            "boundary_kind": "paragraph",
        },
    ],
)
def test_boundary_and_frontier_counters_reject_coercible_non_integers(payload: dict[str, object]) -> None:
    factory = BoundaryCutpoint if "cut_offset" in payload else LayerFrontierItem
    with pytest.raises(ValidationError):
        factory(**payload)


@pytest.mark.parametrize("value", [True, "1", 1.0, -1])
def test_persisted_layer_attempts_reject_non_strict_or_negative_values(value: object) -> None:
    with pytest.raises(ValidationError):
        ParseSessionState(
            collection_id="doc",
            root_node_id="doc|root",
            layer_attempts={"depth=0;parents=doc|root": value},
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: HydratedTextPointer(
            source_cluster_id=True,
            start_char=0,
            end_char=0,
            verbatim_text="x",
        ),
        lambda: HydratedTextPointer(
            source_cluster_id="doc|p1_t0",
            start_char=0,
            end_char=0,
            verbatim_text=1,
        ),
        lambda: StrategyExecutionRecord(
            strategy="layer_excerpt",
            depth=True,
            attempt=1,
            event="selected",
        ),
        lambda: StrategyExecutionRecord(
            strategy="layer_excerpt",
            depth=0,
            attempt="1",
            event="selected",
        ),
        lambda: BoundaryReviewDecision(
            parent_node_id="root",
            source_cluster_id="doc|p1_t0",
            cut_offset=True,
            decision="accept",
        ),
    ],
)
def test_authoritative_pointer_and_review_coordinates_reject_coercion(factory) -> None:
    with pytest.raises(ValidationError):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BoundaryUnitSummary(
            parent_node_id="root",
            source_cluster_id="doc|p1_t0",
            start_char=-1,
            end_char=0,
        ),
        lambda: BoundaryUnitSummary(
            parent_node_id="root",
            source_cluster_id="doc|p1_t0",
            start_char=2,
            end_char=1,
        ),
        lambda: LayerCoverageGap(
            parent_node_id="root",
            source_cluster_id="doc|p1_t0",
            gap_start=2,
            gap_end=1,
        ),
        lambda: LayerSpanConflict(
            parent_node_id="root",
            left_child_id="left",
            right_child_id="right",
            source_cluster_id="doc|p1_t0",
            left_span=HydratedTextPointer(
                source_cluster_id="doc|p1_t0",
                start_char=0,
                end_char=1,
                verbatim_text="ab",
            ),
            right_span=HydratedTextPointer(
                source_cluster_id="doc|p1_t0",
                start_char=1,
                end_char=2,
                verbatim_text="bc",
            ),
            overlap_start=2,
            overlap_end=1,
        ),
    ],
)
def test_interval_diagnostics_reject_invalid_ranges(factory) -> None:
    with pytest.raises(ValidationError):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BoundaryUnitSummary(
            parent_node_id="root",
            source_cluster_id="doc|p1_t0",
            start_char="0",
            end_char=1,
        ),
        lambda: LayerCoverageGap(
            parent_node_id="root",
            source_cluster_id="doc|p1_t0",
            gap_start=True,
            gap_end=1,
        ),
        lambda: LayerSpanConflict(
            parent_node_id="root",
            left_child_id="left",
            right_child_id="right",
            source_cluster_id="doc|p1_t0",
            left_span=HydratedTextPointer(
                source_cluster_id="doc|p1_t0",
                start_char=0,
                end_char=1,
                verbatim_text="ab",
            ),
            right_span=HydratedTextPointer(
                source_cluster_id="doc|p1_t0",
                start_char=1,
                end_char=2,
                verbatim_text="bc",
            ),
            overlap_start="1",
            overlap_end=1,
        ),
    ],
)
def test_interval_diagnostics_reject_coercible_offsets(factory) -> None:
    with pytest.raises(ValidationError):
        factory()


def test_progress_logging_is_fail_soft_for_malformed_checkpoint_state() -> None:
    _log_runtime_progress(
        step_name="review_layer",
        state_view={
            "current_layer_context": {
                "depth": "not-an-integer",
                "retry_count": True,
                "max_retries": object(),
            },
            "parse_session": {
                "max_depth": "not-an-integer",
            },
        },
    )


@pytest.mark.parametrize("value", [True, False, 1.5, -1, "1"])
def test_retry_state_does_not_truncate_malformed_numbers(value: object) -> None:
    assert _strategy_attempt({"strategy_attempt_counts": {"layer_excerpt": value}}, "layer_excerpt") == 1


@pytest.mark.parametrize("value", [True, -1, "1", 1.5])
def test_state_integer_reader_fails_closed(value: object) -> None:
    assert _state_int(value, default=4) == 4


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, "1"])
def test_state_float_reader_rejects_nonfinite_or_coercible_values(value: object) -> None:
    assert _state_float(value, default=4.0) == 4.0


def test_coverage_map_rejects_nonfinite_values_without_poisoning_state() -> None:
    assert _coverage_map({"cluster": math.nan, "other": math.inf, "ok": 0.5}) == {
        "cluster": 0.0,
        "other": 0.0,
        "ok": 0.5,
    }


def test_diagnostic_counter_does_not_truncate_malformed_existing_value() -> None:
    diagnostics: dict[str, object] = {"repaired": 1.5}
    _increment_diagnostic(diagnostics, "repaired")
    assert diagnostics["repaired"] == 1
