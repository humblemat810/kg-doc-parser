from __future__ import annotations

import logging

import pytest

from kg_doc_parser.workflow_ingest.models import (
    CurrentLayerContext,
    CurrentLayerResult,
    CurrentLayerReview,
    LayerChildCandidate,
    LayerCoverageGap,
    LayerDuplicateChildNote,
    LayerSpanConflict,
)
from kg_doc_parser.workflow_ingest.parser_core import (
    check_layer_coverage,
    dedupe_and_filter_layer,
    detect_layer_invariants,
    repair_layer_candidates,
)
from kg_doc_parser.workflow_ingest.semantics import HydratedTextPointer


pytestmark = [pytest.mark.workflow, pytest.mark.ci]


def _pointer(unit_id: str, text: str, start: int, end: int) -> HydratedTextPointer:
    return HydratedTextPointer(
        source_cluster_id=unit_id,
        start_char=start,
        end_char=end,
        verbatim_text=text[start : end + 1],
    )


def _child(
    *,
    node_id: str,
    parent_node_id: str,
    title: str,
    node_type: str,
    pointer: HydratedTextPointer,
    expandable: bool = False,
) -> LayerChildCandidate:
    return LayerChildCandidate(
        node_id=node_id,
        parent_node_id=parent_node_id,
        title=title,
        node_type=node_type,
        total_content_pointers=[pointer],
        expandable=expandable,
    )


def test_detect_layer_invariants_reports_overlap_gap_and_duplicate_notes():
    text = "AlphaXBetaYGammaZDelta"
    unit_id = "doc|p1_t0"
    parent_node_id = "doc|root"
    parent_ptr = _pointer(unit_id, text, 0, len(text) - 1)
    context = CurrentLayerContext(
        depth=0,
        parent_node_ids=[parent_node_id],
        parent_titles=["Doc"],
        parent_content_pointers_by_id={parent_node_id: [parent_ptr]},
    )
    result = CurrentLayerResult(
        children=[
            _child(
                node_id="child-a",
                parent_node_id=parent_node_id,
                title="Alpha",
                node_type="TEXT_FLOW",
                pointer=_pointer(unit_id, text, 0, 7),
            ),
            _child(
                node_id="child-b",
                parent_node_id=parent_node_id,
                title="Beta",
                node_type="TEXT_FLOW",
                pointer=_pointer(unit_id, text, 4, 12),
            ),
            _child(
                node_id="child-a-dup",
                parent_node_id=parent_node_id,
                title="Alpha",
                node_type="TEXT_FLOW",
                pointer=_pointer(unit_id, text, 0, 7),
            ),
            _child(
                node_id="child-c",
                parent_node_id=parent_node_id,
                title="Delta",
                node_type="TEXT_FLOW",
                pointer=_pointer(unit_id, text, 16, 21),
            ),
        ],
        satisfied=True,
        reasoning_history=[],
    )

    coverage_ok, satisfied, overlap_conflicts, coverage_gaps, duplicate_notes, notes = detect_layer_invariants(
        current_layer_context=context,
        current_layer_result=result,
        parser_source_map={unit_id: {"text": text}},
    )

    assert coverage_ok is False
    assert satisfied is False
    assert any(conflict.conflict_kind == "overlap" for conflict in overlap_conflicts)
    assert any(conflict.conflict_kind == "duplicate" for conflict in overlap_conflicts)
    assert duplicate_notes
    assert coverage_gaps
    assert any(gap.expected_text == text[13:16] for gap in coverage_gaps)
    assert any("duplicate child proposal" in note for note in notes)
    assert any("gap in parent" in note for note in notes)


def test_dedupe_and_filter_layer_keeps_unique_children_and_drops_parent_title():
    context = CurrentLayerContext(
        depth=1,
        parent_node_ids=["doc|root", "doc|section-a"],
        parent_titles=["Doc", "Section A"],
    )
    result = CurrentLayerResult(
        children=[
            _child(
                node_id="child-a-1",
                parent_node_id="doc|section-a",
                title="Section A",
                node_type="TEXT_FLOW",
                pointer=_pointer("doc|p1_t0", "Alpha", 0, 4),
            ),
            _child(
                node_id="child-a-2",
                parent_node_id="doc|section-a",
                title="Section A",
                node_type="TEXT_FLOW",
                pointer=_pointer("doc|p1_t0", "Alpha", 0, 4),
            ),
            _child(
                node_id="child-b",
                parent_node_id="doc|section-a",
                title="Section B",
                node_type="TEXT_FLOW",
                pointer=_pointer("doc|p1_t0", "Beta", 0, 3),
            ),
            _child(
                node_id="child-other",
                parent_node_id="doc|other",
                title="Other",
                node_type="TEXT_FLOW",
                pointer=_pointer("doc|p1_t0", "Other", 0, 4),
            ),
        ],
        satisfied=True,
        reasoning_history=[],
    )

    filtered = dedupe_and_filter_layer(
        current_layer_context=context,
        current_layer_result=result,
    )

    assert [child.node_id for child in filtered.children] == ["child-b"]
    assert filtered.children[0].title == "Section B"


def test_repair_layer_candidates_applies_fake_pointer_repair_and_counts_changes():
    text = "AlphaBetaGamma"
    unit_id = "doc|p1_t0"
    result = CurrentLayerResult(
        children=[
            _child(
                node_id="child-a",
                parent_node_id="doc|root",
                title="Alpha",
                node_type="TEXT_FLOW",
                pointer=_pointer(unit_id, text, 0, 5),
            ),
            _child(
                node_id="child-b",
                parent_node_id="doc|root",
                title="Beta",
                node_type="TEXT_FLOW",
                pointer=_pointer(unit_id, text, 5, 8),
            ),
        ],
        satisfied=True,
        reasoning_history=[],
    )

    def _correct(pointer, source_map):
        if pointer.verbatim_text == "AlphaB":
            return pointer.model_copy(
                update={
                    "end_char": 4,
                    "verbatim_text": "Alpha",
                }
            )
        return pointer

    repaired, repaired_count = repair_layer_candidates(
        current_layer_result=result,
        parser_source_map={unit_id: {"text": text}},
        correct_pointer_fn=_correct,
    )

    assert repaired_count == 1
    assert repaired.children[0].total_content_pointers[0].verbatim_text == "Alpha"
    assert repaired.children[0].total_content_pointers[0].end_char == 4
    assert repaired.children[1].total_content_pointers[0].verbatim_text == "Beta"


def test_repair_layer_candidates_raises_on_unrecoverable_pointer(caplog):
    result = CurrentLayerResult(
        children=[
            _child(
                node_id="child-a",
                parent_node_id="doc|root",
                title="Alpha",
                node_type="TEXT_FLOW",
                pointer=_pointer("doc|p1_t0", "Alpha", 0, 4),
            )
        ],
        satisfied=True,
        reasoning_history=[],
    )

    def _correct(pointer, source_map):
        return None

    caplog.set_level(logging.WARNING)
    with pytest.raises(ValueError, match="unrecoverable pointer"):
        repair_layer_candidates(
            current_layer_result=result,
            parser_source_map={"doc|p1_t0": {"text": "Alpha"}},
            correct_pointer_fn=_correct,
        )

    assert any("repair_layer_candidates failed" in record.message for record in caplog.records)
    assert any("source_cluster_id='doc|p1_t0'" in record.message for record in caplog.records)
    assert any("parent='doc|root'" in record.message for record in caplog.records)


def test_check_layer_coverage_emits_conflict_notes_from_review():
    context = CurrentLayerContext(
        depth=0,
        parent_node_ids=["doc|root"],
        parent_titles=["Doc"],
    )
    left_ptr = _pointer("doc|p1_t0", "AlphaBeta", 0, 4)
    right_ptr = _pointer("doc|p1_t0", "AlphaBeta", 3, 7)
    overlap = LayerSpanConflict(
        parent_node_id="doc|root",
        left_child_id="child-a",
        right_child_id="child-b",
        source_cluster_id="doc|p1_t0",
        left_span=left_ptr,
        right_span=right_ptr,
        overlap_start=3,
        overlap_end=4,
        conflict_kind="overlap",
    )
    gap = LayerCoverageGap(
        parent_node_id="doc|root",
        source_cluster_id="doc|p1_t0",
        gap_start=8,
        gap_end=10,
        expected_text="gap-text",
    )
    duplicate = LayerDuplicateChildNote(
        parent_node_id="doc|root",
        child_node_id="child-b",
        duplicate_of_child_node_id="child-a",
        reason="duplicate child proposal under the same parent",
    )
    review = CurrentLayerReview(
        updated_result=CurrentLayerResult(children=[], satisfied=False, reasoning_history=[]),
        coverage_ok=False,
        satisfied=False,
        overlap_conflicts=[overlap],
        coverage_gap_notes=[gap],
        duplicate_child_notes=[duplicate],
        review_notes=["conflict review"],
    )

    coverage_ok, notes = check_layer_coverage(
        current_layer_context=context,
        current_layer_result=review.updated_result,
        current_layer_review=review,
    )

    assert coverage_ok is False
    assert any("overlap conflict:" in note for note in notes)
    assert any("coverage gap:" in note for note in notes)
    assert any("duplicate child:" in note for note in notes)
