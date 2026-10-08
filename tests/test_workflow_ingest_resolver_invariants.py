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
    LayerFrontierItem,
    LayerSpanConflict,
    ParseSessionState,
)
from kg_doc_parser.workflow_ingest.parser_core import (
    check_layer_coverage,
    commit_layer_children,
    dedupe_and_filter_layer,
    detect_layer_invariants,
    enqueue_next_layer_frontier,
    finalize_semantic_tree,
    repair_layer_candidates,
    requeue_failed_frontier_items,
    review_layer,
    validate_layer_commit,
)
from kg_doc_parser.workflow_ingest.semantics import (
    HydratedTextPointer,
    SemanticNode,
    classify_terminal_coverage_status,
    compute_pointer_coverage,
    compute_terminal_content_coverage,
)

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


def test_dedupe_and_filter_layer_keeps_grounded_same_title_and_drops_true_duplicate():
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

    assert [child.node_id for child in filtered.children] == ["child-a-1", "child-b"]
    assert filtered.children[0].title == "Section A"


def test_detect_layer_invariants_rejects_empty_child_and_invalid_parent_source():
    parent_id = "doc|root"
    pointer = _pointer("doc|p1_t0", "Alpha", 0, 4)
    context = CurrentLayerContext(
        depth=0,
        parent_node_ids=[parent_id],
        parent_titles=["Doc"],
        parent_content_pointers_by_id={parent_id: [pointer]},
    )
    result = CurrentLayerResult(
        children=[
            LayerChildCandidate(
                node_id="empty-child",
                parent_node_id=parent_id,
                title="Empty",
                node_type="HEADING",
                total_content_pointers=[],
            )
        ],
        satisfied=True,
    )

    coverage_ok, satisfied, _overlaps, gaps, _duplicates, notes = detect_layer_invariants(
        current_layer_context=context,
        current_layer_result=result,
        parser_source_map={"doc|p1_t0": {"text": "Alpha"}},
    )

    assert coverage_ok is False
    assert satisfied is False
    assert gaps
    assert any("no content pointer" in note for note in notes)

    invalid_context = context.model_copy(
        update={
            "parent_content_pointers_by_id": {
                parent_id: [_pointer("missing", "Alpha", 0, 4)]
            }
        }
    )
    invalid = detect_layer_invariants(
        current_layer_context=invalid_context,
        current_layer_result=result,
        parser_source_map={"doc|p1_t0": {"text": "Alpha"}},
    )
    assert invalid[0] is False
    assert any("unknown source cluster" in note for note in invalid[-1])


def test_dedupe_and_filter_layer_keeps_same_label_at_distinct_source_spans():
    context = CurrentLayerContext(
        depth=1,
        parent_node_ids=["doc|root"],
        parent_titles=["Doc"],
    )
    result = CurrentLayerResult(
        children=[
            _child(
                node_id="child-a",
                parent_node_id="doc|root",
                title="Revenue",
                node_type="TEXT_FLOW",
                pointer=_pointer("doc|p1_t0", "Revenue", 0, 6),
            ),
            _child(
                node_id="child-b",
                parent_node_id="doc|root",
                title="Revenue",
                node_type="TEXT_FLOW",
                pointer=_pointer("doc|p1_t0", "Revenue", 20, 26),
            ),
        ],
        satisfied=True,
    )

    filtered = dedupe_and_filter_layer(
        current_layer_context=context,
        current_layer_result=result,
    )

    assert [child.node_id for child in filtered.children] == ["child-a", "child-b"]


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


def test_repair_layer_candidates_isolates_unrecoverable_pointer(caplog):
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
    repaired, repaired_count = repair_layer_candidates(
        current_layer_result=result,
        parser_source_map={"doc|p1_t0": {"text": "Alpha"}},
        correct_pointer_fn=_correct,
    )

    assert repaired_count == 0
    assert repaired.children == []
    assert repaired.metadata["repair_failure_scope"] == "child_replacement"
    assert repaired.metadata["failure_type"] == "repair_failure"
    assert repaired.metadata["rollback"] == "verified_parent_retained"
    assert "unrecoverable pointer" in repaired.metadata["repair_failures"][0]
    assert any("repair_layer_candidates failed" in record.message for record in caplog.records)
    assert any("source_cluster_id='doc|p1_t0'" in record.message for record in caplog.records)
    assert any("parent='doc|root'" in record.message for record in caplog.records)


def test_commit_and_frontier_enqueue_are_idempotent_on_replay():
    pointer = _pointer("doc|p1_t0", "Alpha", 0, 4)
    child = _child(
        node_id="child-a",
        parent_node_id="doc|root",
        title="Alpha",
        node_type="TEXT_FLOW",
        pointer=pointer,
        expandable=True,
    )
    result = CurrentLayerResult(children=[child], satisfied=True)
    tree = SemanticNode(
        node_id="doc|root",
        title="Doc",
        node_type="DOCUMENT_ROOT",
    )

    once = commit_layer_children(
        semantic_tree=tree,
        current_layer_result=result,
        current_depth=0,
    )
    twice = commit_layer_children(
        semantic_tree=once,
        current_layer_result=result,
        current_depth=0,
    )
    assert [node.node_id for node in twice.child_nodes] == ["child-a"]

    session = ParseSessionState(collection_id="doc", root_node_id="doc|root")
    first_queue = enqueue_next_layer_frontier(
        frontier_queue=[],
        current_layer_context=CurrentLayerContext(
            depth=0,
            parent_node_ids=["doc|root"],
            parent_titles=["Doc"],
        ),
        current_layer_result=result,
        parse_session=session,
    )
    second_queue = enqueue_next_layer_frontier(
        frontier_queue=first_queue,
        current_layer_context=CurrentLayerContext(
            depth=0,
            parent_node_ids=["doc|root"],
            parent_titles=["Doc"],
        ),
        current_layer_result=result,
        parse_session=session,
    )
    assert [(item.parent_node_id, item.depth) for item in second_queue] == [
        ("child-a", 1)
    ]


def test_atomic_noop_disposition_survives_commit_for_final_status():
    pointer = _pointer("doc|p1_t0", "Alpha", 0, 4)
    tree = SemanticNode(
        node_id="doc|root",
        title="Doc",
        node_type="DOCUMENT_ROOT",
        total_content_pointers=[pointer],
    )
    result = CurrentLayerResult(
        children=[],
        satisfied=True,
        metadata={"atomic_retained": True, "allow_empty_layer": True},
    )

    committed = commit_layer_children(
        semantic_tree=tree,
        current_layer_result=result,
        current_depth=0,
        parent_node_ids=["doc|root"],
    )

    coverage = compute_terminal_content_coverage(
        committed,
        {"doc|p1_t0": {"text": "Alpha"}},
    )
    assert classify_terminal_coverage_status(committed, coverage) == "atomic_valid"
    assert committed.metadata["atomic_retained"] is True


def test_pointer_coverage_counts_unreferenced_source_clusters():
    root = SemanticNode(
        node_id="doc|root",
        title="Doc",
        node_type="DOCUMENT_ROOT",
        child_nodes=[
            SemanticNode(
                node_id="child-a",
                parent_id="doc|root",
                title="Alpha",
                total_content_pointers=[_pointer("doc|p1_t0", "Alpha", 0, 4)],
            )
        ],
    )

    coverage = compute_pointer_coverage(
        root,
        {
            "doc|p1_t0": {"text": "Alpha"},
            "doc|p1_t1": {"text": "Missing meaningful content"},
            "doc|p1_t2": {"text": " \n\t"},
        },
    )

    assert coverage["per_cluster"]["doc|p1_t1"] == 0.0
    assert coverage["per_cluster"]["doc|p1_t2"] == 1.0
    assert coverage["overall"] < 1.0


def test_terminal_content_coverage_ignores_wrapper_and_detects_missing_content():
    root = SemanticNode(
        node_id="doc|root",
        title="Doc",
        node_type="DOCUMENT_ROOT",
        total_content_pointers=[_pointer("doc|p1_t0", "Alpha Beta", 0, 9)],
        child_nodes=[
            SemanticNode(
                node_id="page",
                parent_id="doc|root",
                title="Page 1",
                node_type="PAGE",
                total_content_pointers=[_pointer("doc|p1_t0", "Alpha Beta", 0, 9)],
                child_nodes=[
                    SemanticNode(
                        node_id="content",
                        parent_id="page",
                        title="Alpha",
                        total_content_pointers=[_pointer("doc|p1_t0", "Alpha Beta", 0, 4)],
                    )
                ],
            )
        ],
    )

    coverage = compute_terminal_content_coverage(
        root,
        {"doc|p1_t0": {"text": "Alpha Beta"}},
    )

    assert coverage["coverage_basis"] == "terminal_content_owners_exactly_once"
    assert coverage["overall"] < 1.0
    assert coverage["valid"] is False
    assert coverage["missing_nonws_ranges"]["doc|p1_t0"]


def test_terminal_content_coverage_rejects_duplicate_positions_but_allows_whitespace_gaps():
    text = "Alpha \n Beta"
    root = SemanticNode(
        node_id="doc|root",
        title="Doc",
        node_type="DOCUMENT_ROOT",
        child_nodes=[
            SemanticNode(
                node_id="left",
                parent_id="doc|root",
                title="Alpha",
                total_content_pointers=[_pointer("doc|p1_t0", text, 0, 4)],
            ),
            SemanticNode(
                node_id="right",
                parent_id="doc|root",
                title="Beta",
                total_content_pointers=[_pointer("doc|p1_t0", text, 8, 11)],
            ),
        ],
    )

    whitespace_gap = compute_terminal_content_coverage(root, {"doc|p1_t0": {"text": text}})
    assert whitespace_gap["valid"] is True
    assert whitespace_gap["overall"] == 1.0

    duplicate = root.model_copy(
        update={
            "child_nodes": [
                root.child_nodes[0],
                root.child_nodes[0].model_copy(update={"node_id": "duplicate"}),
                root.child_nodes[1],
            ]
        }
    )
    duplicate_coverage = compute_terminal_content_coverage(
        duplicate,
        {"doc|p1_t0": {"text": text}},
    )
    assert duplicate_coverage["valid"] is False
    assert duplicate_coverage["multiply_owned_nonws_ranges"]["doc|p1_t0"]

    assert classify_terminal_coverage_status(root, whitespace_gap) == "complete"
    assert classify_terminal_coverage_status(
        root.model_copy(update={"metadata": {"atomic_retained": True}}),
        whitespace_gap,
    ) == "atomic_valid"
    assert classify_terminal_coverage_status(root, duplicate_coverage) == "partial_degraded"
    assert classify_terminal_coverage_status(root, {"valid": False, "covered_nonws": 0}) == "failed"


def test_review_disabled_still_runs_deterministic_invariants():
    text = "Alpha Beta"
    context = CurrentLayerContext(
        depth=0,
        parent_node_ids=["doc|root"],
        parent_titles=["Doc"],
        parent_content_pointers_by_id={"doc|root": [_pointer("doc|p1_t0", text, 0, len(text) - 1)]},
    )
    result = CurrentLayerResult(
        children=[
            _child(
                node_id="child",
                parent_node_id="doc|root",
                title="Alpha",
                node_type="TEXT_FLOW",
                pointer=_pointer("doc|p1_t0", text, 0, 4),
            )
        ],
        satisfied=True,
    )
    review, _ = review_layer(
        parse_session=ParseSessionState(
            collection_id="doc",
            root_node_id="doc|root",
            allow_review=False,
        ),
        current_layer_context=context,
        current_layer_result=result,
        parser_source_map={"doc|p1_t0": {"text": text}},
    )
    assert review.metadata["review_skipped"] is True
    assert review.coverage_ok is False
    assert review.satisfied is False


def test_validate_layer_commit_rejects_changed_candidate_set():
    text = "Alpha Beta"
    context = CurrentLayerContext(
        depth=0,
        parent_node_ids=["doc|root"],
        parent_titles=["Doc"],
        parent_content_pointers_by_id={"doc|root": [_pointer("doc|p1_t0", text, 0, len(text) - 1)]},
    )
    review = validate_layer_commit(
        current_layer_context=context,
        current_layer_result=CurrentLayerResult(
            children=[
                _child(
                    node_id="child",
                    parent_node_id="doc|root",
                    title="Alpha",
                    node_type="TEXT_FLOW",
                    pointer=_pointer("doc|p1_t0", text, 0, 4),
                )
            ],
            satisfied=True,
        ),
        parser_source_map={"doc|p1_t0": {"text": text}},
    )
    assert review.coverage_ok is False
    assert review.metadata["commit_validation"] is True
    assert review.coverage_gap_notes


def test_validate_layer_commit_preserves_valid_parent_when_batch_sibling_fails():
    valid_text = "Alpha"
    invalid_text = "Beta"
    valid_parent = "doc|section-a"
    invalid_parent = "doc|section-b"
    context = CurrentLayerContext(
        depth=1,
        parent_node_ids=[valid_parent, invalid_parent],
        parent_titles=["A", "B"],
        parent_content_pointers_by_id={
            valid_parent: [_pointer("doc|p1_t0", valid_text, 0, 4)],
            invalid_parent: [_pointer("doc|p1_t1", invalid_text, 0, 3)],
        },
    )
    result = CurrentLayerResult(
        children=[
            _child(
                node_id="child-a",
                parent_node_id=valid_parent,
                title="Alpha",
                node_type="TEXT_FLOW",
                pointer=_pointer("doc|p1_t0", valid_text, 0, 4),
            )
        ],
        satisfied=True,
    )

    review = validate_layer_commit(
        current_layer_context=context,
        current_layer_result=result,
        parser_source_map={
            "doc|p1_t0": {"text": valid_text},
            "doc|p1_t1": {"text": invalid_text},
        },
    )

    assert review.satisfied is True
    assert review.coverage_ok is False
    assert review.committable_parent_node_ids == [valid_parent]
    assert review.failed_parent_node_ids == [invalid_parent]
    assert review.metadata["partial_commit"] is True
    assert [child.node_id for child in review.updated_result.children] == ["child-a"]

    tree = SemanticNode(
        node_id="doc|root",
        title="Doc",
        node_type="DOCUMENT_ROOT",
        child_nodes=[
            SemanticNode(
                node_id=valid_parent,
                parent_id="doc|root",
                title="A",
                node_type="SECTION",
                total_content_pointers=[_pointer("doc|p1_t0", valid_text, 0, 4)],
            ),
            SemanticNode(
                node_id=invalid_parent,
                parent_id="doc|root",
                title="B",
                node_type="SECTION",
                total_content_pointers=[_pointer("doc|p1_t1", invalid_text, 0, 3)],
            ),
        ],
    )
    committed = commit_layer_children(
        semantic_tree=tree,
        current_layer_result=review.updated_result,
        current_depth=1,
        parent_node_ids=review.committable_parent_node_ids,
    )
    assert [child.node_id for child in committed.child_nodes[0].child_nodes] == ["child-a"]
    assert committed.child_nodes[1].child_nodes == []


def test_failed_batch_parents_are_requeued_at_same_depth_once():
    queued = requeue_failed_frontier_items(
        frontier_queue=[
            LayerFrontierItem(parent_node_id="already-queued", depth=2, order=4),
        ],
        parent_node_ids=["failed-a", "already-queued", "failed-a"],
        depth=2,
    )
    assert [(item.parent_node_id, item.depth, item.order) for item in queued] == [
        ("already-queued", 2, 4),
        ("failed-a", 2, 5),
    ]


def test_finalize_semantic_tree_rejects_dangling_source_pointer():
    root = SemanticNode(
        node_id="doc|root",
        title="Doc",
        node_type="DOCUMENT_ROOT",
        child_nodes=[
            SemanticNode(
                node_id="child-a",
                parent_id="doc|root",
                title="Alpha",
                total_content_pointers=[_pointer("doc|missing", "Alpha", 0, 4)],
            )
        ],
    )

    with pytest.raises(ValueError, match="unknown source cluster"):
        finalize_semantic_tree(root, parser_source_map={"doc|p1_t0": {"text": "Alpha"}})


def test_finalize_semantic_tree_rejects_excerpt_that_disagrees_with_source():
    root = SemanticNode(
        node_id="doc|root",
        title="Doc",
        node_type="DOCUMENT_ROOT",
        child_nodes=[
            SemanticNode(
                node_id="child-a",
                parent_id="doc|root",
                title="Wrong",
                total_content_pointers=[
                    HydratedTextPointer(
                        source_cluster_id="doc|p1_t0",
                        start_char=0,
                        end_char=4,
                        verbatim_text="Wrong",
                    )
                ],
            )
        ],
    )

    with pytest.raises(ValueError, match="excerpt does not match"):
        finalize_semantic_tree(root, parser_source_map={"doc|p1_t0": {"text": "Alpha"}})


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
