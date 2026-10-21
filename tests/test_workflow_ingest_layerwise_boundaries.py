from __future__ import annotations

from layerwise_test_support import (
    _boundary_context,
    _boundary_cutpoint_payload,
    _parser_source_map,
)

from kg_doc_parser.workflow_ingest.layerwise_boundaries import _legal_cutpoints_for_text
from kg_doc_parser.workflow_ingest.layerwise_boundary_pipeline import (
    _boundary_parent_coverage_report,
    _boundary_review_decision,
    _boundary_validation_reason,
    _proposal_validation_reason,
)
from kg_doc_parser.workflow_ingest.layerwise_boundary_prompts import (
    _boundary_prompt_candidate_context,
)
from kg_doc_parser.workflow_ingest.models import (
    BoundaryCutpoint,
    BoundaryReviewBatch,
    BoundaryReviewDecision,
    CurrentLayerContext,
    CurrentLayerResult,
    LayerChildCandidate,
    LLMBoundaryProposal,
    LLMBoundaryProposalBatch,
)
from kg_doc_parser.workflow_ingest.semantics import (
    HydratedTextPointer,
)


def test_proposal_validation_does_not_stringify_malformed_source_text() -> None:
    context = CurrentLayerContext(
        depth=0,
        parent_node_ids=["root"],
        parent_titles=["Root"],
        parent_content_pointers_by_id={
            "root": [
                HydratedTextPointer(
                    source_cluster_id="bad",
                    start_char=0,
                    end_char=2,
                    verbatim_text="123",
                )
            ]
        },
    )
    result = CurrentLayerResult(
        children=[
            LayerChildCandidate(
                node_id="child-a",
                parent_node_id="root",
                title="A",
                node_type="TEXT_FLOW",
                total_content_pointers=[
                    HydratedTextPointer(
                        source_cluster_id="bad",
                        start_char=0,
                        end_char=0,
                        verbatim_text="1",
                    )
                ],
            ),
            LayerChildCandidate(
                node_id="child-b",
                parent_node_id="root",
                title="B",
                node_type="TEXT_FLOW",
                total_content_pointers=[
                    HydratedTextPointer(
                        source_cluster_id="bad",
                        start_char=1,
                        end_char=2,
                        verbatim_text="23",
                    )
                ],
            ),
        ]
    )

    assert (
        _proposal_validation_reason(
            parsed=result,
            current_layer_context=context,
            parser_source_map={"bad": {"text": 123}},
        )
        == "proposal pointer used a character span outside the source text"
    )


def test_boundary_assembly_keeps_cross_cluster_parent_pointers_separate() -> None:
    from kg_doc_parser.workflow_ingest.layerwise_llm import (
        _assemble_layer_result_from_boundaries,
    )

    cluster_one = "A1 first.\nA1 second."
    cluster_two = "A2 first.\nA2 second."
    context = CurrentLayerContext(
        depth=0,
        parent_node_ids=["doc|root"],
        parent_titles=["Demo Doc"],
        parent_content_pointers_by_id={
            "doc|root": [
                HydratedTextPointer(
                    source_cluster_id="cluster-1",
                    start_char=0,
                    end_char=len(cluster_one) - 1,
                    verbatim_text=cluster_one,
                ),
                HydratedTextPointer(
                    source_cluster_id="cluster-2",
                    start_char=0,
                    end_char=len(cluster_two) - 1,
                    verbatim_text=cluster_two,
                ),
            ]
        },
        split_strategy="boundary_first",
        retry_count=0,
        max_retries=2,
    )
    review = BoundaryReviewBatch(
        decisions=[
            BoundaryReviewDecision(
                parent_node_id="doc|root",
                source_cluster_id="cluster-1",
                input_cut_offset=10,
                cut_offset=10,
                resolved_cut_offset=10,
                decision="accept",
                boundary_kind="paragraph",
            ),
            BoundaryReviewDecision(
                parent_node_id="doc|root",
                source_cluster_id="cluster-2",
                input_cut_offset=10,
                cut_offset=10,
                resolved_cut_offset=10,
                decision="accept",
                boundary_kind="paragraph",
            ),
        ]
    )

    result, _summaries, _accepted = _assemble_layer_result_from_boundaries(
        current_layer_context=context,
        parser_source_map={
            "cluster-1": {"text": cluster_one},
            "cluster-2": {"text": cluster_two},
        },
        review_batch=review,
    )

    assert result.children
    assert {
        pointer.source_cluster_id
        for child in result.children
        for pointer in child.total_content_pointers
    } == {
        "cluster-1",
        "cluster-2",
    }
    assert all(
        len({pointer.source_cluster_id for pointer in child.total_content_pointers})
        == 1
        for child in result.children
    )


def test_boundary_helpers_classify_and_snap_cutpoints():

    text = "Alpha clause.\n\nBeta clause."
    legal_cutpoints = _legal_cutpoints_for_text(text)
    assert any(cutpoint.boundary_kind == "paragraph" for cutpoint in legal_cutpoints)

    decision = _boundary_review_decision(
        cutpoint=BoundaryCutpoint(
            **_boundary_cutpoint_payload(
                "Alpha clause. Beta clause.",
                12,
                anchor_offset=13,
                boundary_kind="semantic",
                confidence=0.5,
                reason="near the paragraph boundary",
            )
        ),
        current_layer_context=_boundary_context(),
        parser_source_map=_parser_source_map(),
    )

    assert decision.decision == "shift_right"
    assert decision.resolved_cut_offset == 13

    reject_decision = _boundary_review_decision(
        cutpoint=BoundaryCutpoint(
            parent_node_id="doc|root",
            source_cluster_id="cluster-1",
            cut_offset=16,
            boundary_kind="semantic",
            text_before_cut="Alpha clause. Be",
            text_after_cut="ta clause.",
            cut_reason="inside token",
            confidence=0.5,
            reason="inside token",
        ),
        current_layer_context=_boundary_context(),
        parser_source_map=_parser_source_map(),
    )
    assert reject_decision.decision == "reject"
    assert "word" in (reject_decision.reason or "")

    fuzzy_decision = _boundary_review_decision(
        cutpoint=BoundaryCutpoint(
            parent_node_id="doc|root",
            source_cluster_id="cluster-1",
            cut_offset=12,
            boundary_kind="sentence",
            text_before_cut="Alpha clausx.",
            text_after_cut=" Beta",
            cut_reason="near the sentence boundary",
            confidence=0.8,
            reason="near the sentence boundary",
        ),
        current_layer_context=_boundary_context(),
        parser_source_map=_parser_source_map(),
    )
    assert fuzzy_decision.anchor_match_mode == "fuzzy"
    assert fuzzy_decision.resolved_cut_offset == 13
    assert fuzzy_decision.decision == "shift_right"

    far_shift_decision = _boundary_review_decision(
        cutpoint=BoundaryCutpoint(
            parent_node_id="doc|root",
            source_cluster_id="cluster-1",
            cut_offset=0,
            boundary_kind="sentence",
            text_before_cut="Alpha clausx.",
            text_after_cut=" Beta",
            cut_reason="far off boundary guess",
            confidence=0.8,
            reason="far off boundary guess",
        ),
        current_layer_context=_boundary_context(),
        parser_source_map=_parser_source_map(),
    )
    assert far_shift_decision.decision == "reject"
    assert "exceeds maximum" in (far_shift_decision.reason or "")

    reversed_batch = LLMBoundaryProposalBatch(
        cutpoints=[
            BoundaryCutpoint(
                **_boundary_cutpoint_payload(
                    "Alpha clause. Beta clause.",
                    20,
                    boundary_kind="sentence",
                    confidence=0.8,
                    reason="later boundary",
                )
            ),
            BoundaryCutpoint(
                **_boundary_cutpoint_payload(
                    "Alpha clause. Beta clause.",
                    10,
                    boundary_kind="sentence",
                    confidence=0.8,
                    reason="earlier boundary",
                )
            ),
        ],
        satisfied=True,
        reasoning_history=[],
    )
    assert "sorted" in (
        _boundary_validation_reason(
            parsed=reversed_batch,
            current_layer_context=_boundary_context(),
            parser_source_map=_parser_source_map(),
        )
        or ""
    )


def test_boundary_candidate_generation_prefers_structural_boundaries_over_words():
    from collections import Counter

    text = (
        "# Demo\n\n"
        "Intro sentence. Another sentence.\n\n"
        "## Finding 1\n\n"
        "Alpha clause. Beta clause.\n\n"
        "## Finding 2\n\n"
        "Alpha clause. Beta clause."
    )

    cutpoints = _legal_cutpoints_for_text(text, max_points=8)
    counts = Counter(cutpoint.boundary_kind for cutpoint in cutpoints)

    assert counts["word"] == 0
    assert counts["section"] >= 2
    assert all(cutpoint.candidate_id for cutpoint in cutpoints)
    assert [cutpoint.cut_offset for cutpoint in cutpoints] == sorted(
        cutpoint.cut_offset for cutpoint in cutpoints
    )


def test_boundary_candidate_ids_include_pointer_span_to_avoid_multispan_collisions():
    text = "Alpha clause. Beta clause.\nGamma clause. Delta clause."
    context = CurrentLayerContext(
        depth=0,
        parent_node_ids=["doc|root"],
        parent_titles=["Demo Doc"],
        parent_content_pointers_by_id={
            "doc|root": [
                HydratedTextPointer(
                    source_cluster_id="cluster-1",
                    start_char=0,
                    end_char=25,
                    verbatim_text="Alpha clause. Beta clause.",
                ),
                HydratedTextPointer(
                    source_cluster_id="cluster-1",
                    start_char=27,
                    end_char=len(text) - 1,
                    verbatim_text="Gamma clause. Delta clause.",
                ),
            ]
        },
        split_strategy="boundary_first",
        retry_count=0,
        max_retries=2,
    )

    candidates = _boundary_prompt_candidate_context(
        current_layer_context=context,
        parser_source_map={"cluster-1": {"text": text}},
    )
    candidate_ids = [
        str(cutpoint["candidate_id"])
        for parent in candidates
        for cutpoint in parent["legal_cutpoints"]
    ]

    assert len(candidate_ids) == len(set(candidate_ids))
    assert any("|0|" in candidate_id for candidate_id in candidate_ids)
    assert any("|27|" in candidate_id for candidate_id in candidate_ids)


def test_boundary_parent_coverage_report_marks_gaps_and_spans():
    report = _boundary_parent_coverage_report(
        parent_node_id="doc|root",
        source_cluster_id="cluster-1",
        start_char=0,
        end_char_exclusive=20,
        segments=[
            {"start_char": 0, "end_char": 9},
            {
                "start_char": 10,
                "end_char": 19,
                "skipped": True,
                "skip_reason": "ambiguous boundary",
            },
        ],
    )

    assert report["parent_node_id"] == "doc|root"
    assert report["coverage_state"] == "covered_with_gaps"
    assert report["parent_span"] == {"start_char": 0, "end_char": 19}
    assert report["covered_span"] == {"start_char": 0, "end_char": 9}
    assert report["gap_count"] == 1
    assert report["gap_ranges"][0]["reason"] == "ambiguous boundary"


def test_llm_boundary_proposal_round_trip():
    proposal = LLMBoundaryProposal(
        parent_node_id="doc|root",
        source_cluster_id="cluster-1",
        cutpoints=[
            BoundaryCutpoint(
                **_boundary_cutpoint_payload(
                    "Alpha clause. Beta clause.",
                    13,
                    boundary_kind="sentence",
                    confidence=0.9,
                    reason="sentence boundary",
                )
            )
        ],
        satisfied=True,
        reasoning_history=[],
        review_rounds=1,
    )

    dumped = proposal.model_dump()
    restored = LLMBoundaryProposal.model_validate(dumped)

    assert restored.parent_node_id == "doc|root"
    assert restored.source_cluster_id == "cluster-1"
    assert restored.cutpoints[0].cut_offset == 13
    assert restored.review_rounds == 1
