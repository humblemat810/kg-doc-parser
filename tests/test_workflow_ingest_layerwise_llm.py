from __future__ import annotations

from typing import Any

import pytest

from kg_doc_parser.workflow_ingest import ProviderEndpointConfig, WorkflowProviderSettings
from kg_doc_parser.workflow_ingest.layerwise_llm import build_layerwise_llm_callbacks
from kg_doc_parser.workflow_ingest.models import (
    BoundaryCutpoint,
    LLMBoundaryProposal,
    LLMBoundaryProposalBatch,
    CurrentLayerContext,
    CurrentLayerResult,
    LayerChildCandidate,
    ParseSessionState,
)
from kg_doc_parser.workflow_ingest.semantics import HydratedTextPointer, SemanticNode, semantic_tree_to_kge_payload
from kg_doc_parser.workflow_ingest.parser_core import commit_layer_children


def _provider_settings() -> WorkflowProviderSettings:
    return WorkflowProviderSettings(
        parser=ProviderEndpointConfig(provider="fake", model="fake-model"),
    )


def _context() -> CurrentLayerContext:
    return CurrentLayerContext(
        depth=0,
        parent_node_ids=["doc|root"],
        parent_titles=["Demo Doc"],
        parent_content_pointers_by_id={
            "doc|root": [
                HydratedTextPointer(
                    source_cluster_id="cluster-1",
                    start_char=0,
                    end_char=20,
                    verbatim_text="Alpha clause. Beta",
                )
            ]
        },
        split_strategy="excerpt_first",
        retry_count=0,
        max_retries=2,
    )


def _boundary_context() -> CurrentLayerContext:
    return CurrentLayerContext(
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
                )
            ]
        },
        split_strategy="boundary_first",
        retry_count=0,
        max_retries=2,
    )


def _parse_session() -> ParseSessionState:
    return ParseSessionState(
        collection_id="doc",
        root_node_id="doc|root",
        split_strategy="excerpt_first",
        fallback_split_strategy="boundary_first",
        strategy_history=["excerpt_first"],
        mode="workflow_layered",
        layer_attempts={},
    )


def _semantic_tree() -> SemanticNode:
    return SemanticNode(
        node_id="doc|root",
        title="Demo Doc",
        node_type="DOCUMENT_ROOT",
        total_content_pointers=[],
        child_nodes=[],
        level_from_root=0,
    )


def _parser_input_dict() -> dict[str, Any]:
    return {
        "document_id": "doc",
        "title": "Demo Doc",
        "collections": [
            {
                "title": "Demo Doc",
                "pages": [
                    {
                        "page_number": 1,
                        "units": [{"id": "cluster-1", "text": "Alpha clause. Beta clause."}],
                    }
                ],
            }
        ],
    }


def _parser_source_map() -> dict[str, dict[str, Any]]:
    return {
        "cluster-1": {
            "page_number": 1,
            "cluster_number": 1,
            "text": "Alpha clause. Beta clause.",
        }
    }


def _boundary_cutpoint_payload(
    text: str,
    offset: int,
    *,
    anchor_offset: int | None = None,
    parent_node_id: str = "doc|root",
    source_cluster_id: str = "cluster-1",
    boundary_kind: str = "sentence",
    confidence: float | None = None,
    reason: str = "boundary anchor",
) -> dict[str, Any]:
    anchor_offset = offset if anchor_offset is None else anchor_offset
    window = 8
    start = max(0, min(len(text), anchor_offset - window))
    end = max(0, min(len(text), anchor_offset + window))
    payload: dict[str, Any] = {
        "parent_node_id": parent_node_id,
        "source_cluster_id": source_cluster_id,
        "cut_offset": offset,
        "boundary_kind": boundary_kind,
        "text_before_cut": text[start:anchor_offset],
        "text_after_cut": text[anchor_offset:end],
        "cut_reason": reason,
        "reason": reason,
    }
    if confidence is not None:
        payload["confidence"] = confidence
    return payload


def test_boundary_helpers_classify_and_snap_cutpoints():
    from kg_doc_parser.workflow_ingest.layerwise_llm import (
        _boundary_review_decision,
        _boundary_validation_reason,
        _legal_cutpoints_for_text,
    )

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
            BoundaryCutpoint(**_boundary_cutpoint_payload("Alpha clause. Beta clause.", 20, boundary_kind="sentence", confidence=0.8, reason="later boundary")),
            BoundaryCutpoint(**_boundary_cutpoint_payload("Alpha clause. Beta clause.", 10, boundary_kind="sentence", confidence=0.8, reason="earlier boundary")),
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

    from kg_doc_parser.workflow_ingest.layerwise_llm import _legal_cutpoints_for_text

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
    from kg_doc_parser.workflow_ingest.layerwise_llm import _boundary_prompt_candidate_context

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
    from kg_doc_parser.workflow_ingest.layerwise_llm import _boundary_parent_coverage_report

    report = _boundary_parent_coverage_report(
        parent_node_id="doc|root",
        source_cluster_id="cluster-1",
        start_char=0,
        end_char_exclusive=20,
        segments=[
            {"start_char": 0, "end_char": 9},
            {"start_char": 10, "end_char": 19, "skipped": True, "skip_reason": "ambiguous boundary"},
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
            BoundaryCutpoint(**_boundary_cutpoint_payload("Alpha clause. Beta clause.", 13, boundary_kind="sentence", confidence=0.9, reason="sentence boundary"))
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


def test_workflow_provider_settings_from_env_enables_boundary_mode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KG_DOC_PARSER_PROPOSAL_MODE", "boundaries")
    settings = WorkflowProviderSettings.from_env()
    assert settings.proposal_mode == "boundaries"

    fake_model = _FakeChatModel(
        {
            "parsed": {
                "cutpoints": [
                    _boundary_cutpoint_payload("Alpha clause. Beta clause.", 14, boundary_kind="sentence", confidence=0.92, reason="sentence break"),
                ],
                "satisfied": True,
                "reasoning_history": [],
                "review_rounds": 0,
            }
        }
    )
    monkeypatch.setattr(
        "kg_doc_parser.workflow_ingest.layerwise_llm.build_chat_model_for_role",
        lambda role, settings: fake_model,
    )

    callbacks = build_layerwise_llm_callbacks(settings)
    result = callbacks["propose_layer_fn"](
        parser_source_map=_parser_source_map(),
        current_layer_context=_boundary_context(),
        semantic_tree=_semantic_tree(),
        split_strategy="boundary_first",
        parser_input_dict=_parser_input_dict(),
        parse_session=_parse_session(),
    )

    assert result.metadata["proposal_mode"] == "boundaries"
    assert fake_model.structured_output_kwargs and fake_model.structured_output_kwargs.get("method") == "json_schema"


def test_structured_invoke_returns_typed_pydantic_model():
    from pydantic import BaseModel

    from kg_doc_parser.workflow_ingest.layerwise_llm import _structured_invoke

    class _Schema(BaseModel):
        value: int
        label: str

    fake_model = _FakeChatModel({"parsed": {"value": 7, "label": "demo"}})
    result = _structured_invoke(fake_model, _Schema, [("human", "hello")])

    assert isinstance(result, _Schema)
    assert result.value == 7
    assert result.label == "demo"
    assert fake_model.structured_output_kwargs and fake_model.structured_output_kwargs.get("method") == "json_schema"


def test_boundary_mode_proposes_cutpoints_and_assembles_children(monkeypatch: pytest.MonkeyPatch):
    fake_model = _FakeChatModel(
        {
            "parsed": {
                "cutpoints": [
                    _boundary_cutpoint_payload("Alpha clause. Beta clause.", 12, anchor_offset=13, boundary_kind="semantic", confidence=0.4, reason="near sentence boundary"),
                    _boundary_cutpoint_payload("Alpha clause. Beta clause.", 16, anchor_offset=16, boundary_kind="semantic", confidence=0.92, reason="inside word and should be rejected"),
                ],
                "satisfied": True,
                "reasoning_history": [],
                "review_rounds": 0,
            }
        }
    )
    layer_events: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "kg_doc_parser.workflow_ingest.layerwise_llm.build_chat_model_for_role",
        lambda role, settings: fake_model,
    )

    callbacks = build_layerwise_llm_callbacks(
        _provider_settings(),
        proposal_mode="boundaries",
        event_sink=lambda stage, **extra: layer_events.append({"stage": stage, **extra}),
    )

    result = callbacks["propose_layer_fn"](
        parser_source_map=_parser_source_map(),
        current_layer_context=_boundary_context(),
        semantic_tree=_semantic_tree(),
        split_strategy="boundary_first",
        parser_input_dict=_parser_input_dict(),
        parse_session=_parse_session(),
    )

    assert result.metadata["proposal_mode"] == "boundaries"
    assert result.metadata["boundary_proposed_count"] == 2
    assert result.metadata["boundary_summary_count"] >= 1
    assert result.metadata["boundary_shifted_count"] >= 1
    assert result.metadata["boundary_rejected_count"] >= 1
    assert result.metadata["unresolved_interval_count"] >= 1
    assert result.metadata["boundary_parent_coverage"]
    assert result.metadata["boundary_parent_coverage"][0]["coverage_state"] in {"covered", "covered_with_gaps"}
    assert result.metadata["boundary_refinement_attempts"] == 0
    assert result.reasoning_history[-1].proposal_mode == "boundaries"
    assert fake_model.structured_output_kwargs and fake_model.structured_output_kwargs.get("method") == "json_schema"
    assert any(event["stage"] == "workflow_layered_boundary_proposal_start" for event in layer_events)
    assert any(event["stage"] == "workflow_layered_boundary_review_completed" for event in layer_events)
    assert any(event["stage"] == "workflow_layered_boundary_assembly_start" for event in layer_events)
    assert any(event["stage"] == "workflow_layered_boundary_assembly_completed" for event in layer_events)
    assert layer_events[-1]["stage"] == "workflow_layered_proposal_result"
    assert layer_events[-1]["proposal_mode"] == "boundaries"
    assert result.children
    assert [child.parent_node_id for child in result.children] == ["doc|root", "doc|root"]


def test_boundary_mode_drops_one_unrecoverable_cutpoint_but_keeps_valid_split(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_model = _FakeChatModel(
        {
            "parsed": {
                "cutpoints": [
                    _boundary_cutpoint_payload(
                        "Alpha clause. Beta clause.",
                        13,
                        boundary_kind="sentence",
                        reason="sentence boundary",
                    ),
                        {
                            "parent_node_id": "doc|root",
                            "source_cluster_id": "cluster-1",
                            "cut_offset": 118,
                            "boundary_kind": "semantic",
                        },
                    ],
                "satisfied": True,
                "reasoning_history": [],
                "review_rounds": 0,
            }
        }
    )
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "kg_doc_parser.workflow_ingest.layerwise_llm.build_chat_model_for_role",
        lambda role, settings: fake_model,
    )
    callbacks = build_layerwise_llm_callbacks(
        _provider_settings(),
        proposal_mode="boundaries",
        event_sink=lambda stage, **extra: events.append({"stage": stage, **extra}),
    )

    result = callbacks["propose_layer_fn"](
        parser_source_map=_parser_source_map(),
        current_layer_context=_boundary_context(),
        semantic_tree=_semantic_tree(),
        split_strategy="boundary_first",
        parser_input_dict=_parser_input_dict(),
        parse_session=_parse_session(),
    )

    assert result.metadata["boundary_proposed_count"] == 1
    assert result.metadata["boundary_dropped_count"] == 1
    assert result.metadata["boundary_repaired_count"] == 0
    assert result.children
    assert any(event["stage"] == "workflow_layered_boundary_cutpoint_dropped" for event in events)


def test_boundary_mode_accepts_candidate_id_even_when_copied_offset_is_wrong(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_model = _FakeChatModel(
        {
            "parsed": {
                "cutpoints": [
                    {
                        "candidate_id": "doc|root|cluster-1|0|b0000-sentence-13",
                        "parent_node_id": "doc|root",
                        "source_cluster_id": "cluster-1",
                        "cut_offset": 1,
                        "boundary_kind": "word",
                        "text_before_cut": "#",
                        "text_after_cut": " Alpha",
                        "cut_reason": "sentence boundary selected from candidate list",
                        "confidence": 0.95,
                        "reason": "candidate-selected sentence boundary",
                    }
                ],
                "satisfied": True,
                "reasoning_history": [],
                "review_rounds": 0,
            }
        }
    )
    monkeypatch.setattr(
        "kg_doc_parser.workflow_ingest.layerwise_llm.build_chat_model_for_role",
        lambda role, settings: fake_model,
    )

    callbacks = build_layerwise_llm_callbacks(
        _provider_settings(),
        proposal_mode="boundaries",
    )
    result = callbacks["propose_layer_fn"](
        parser_source_map=_parser_source_map(),
        current_layer_context=_boundary_context(),
        semantic_tree=_semantic_tree(),
        split_strategy="boundary_first",
        parser_input_dict=_parser_input_dict(),
        parse_session=_parse_session(),
    )

    assert result.metadata["boundary_proposed_count"] == 1
    assert result.metadata["boundary_accepted_count"] == 1
    assert result.metadata["boundary_rejected_count"] == 0
    assert len(result.children) == 2
    assert result.children[0].total_content_pointers[0].end_char == 12
    assert result.metadata["boundary_review_decisions"][0]["decision"] == "accept"
    assert result.metadata["boundary_review_decisions"][0]["resolved_cut_offset"] == 13


def test_boundary_mode_backfills_missing_candidate_id_from_exact_offset(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_model = _FakeChatModel(
        {
            "parsed": {
                "cutpoints": [
                    BoundaryCutpoint(
                        candidate_id=None,
                        parent_node_id="doc|root",
                        source_cluster_id="cluster-1",
                        cut_offset=13,
                        boundary_kind="sentence",
                        text_before_cut="",
                        text_after_cut="",
                        cut_reason="sentence boundary selected from candidate list",
                        confidence=0.95,
                        reason="candidate-selected sentence boundary",
                    )
                ],
                "satisfied": True,
                "reasoning_history": [],
                "review_rounds": 0,
            }
        }
    )
    monkeypatch.setattr(
        "kg_doc_parser.workflow_ingest.layerwise_llm.build_chat_model_for_role",
        lambda role, settings: fake_model,
    )

    callbacks = build_layerwise_llm_callbacks(
        _provider_settings(),
        proposal_mode="boundaries",
    )
    result = callbacks["propose_layer_fn"](
        parser_source_map=_parser_source_map(),
        current_layer_context=_boundary_context(),
        semantic_tree=_semantic_tree(),
        split_strategy="boundary_first",
        parser_input_dict=_parser_input_dict(),
        parse_session=_parse_session(),
    )

    assert result.metadata["boundary_proposed_count"] == 1
    assert result.metadata["boundary_accepted_count"] == 1
    assert result.metadata["boundary_rejected_count"] == 0
    assert len(result.children) == 2
    assert result.children[0].total_content_pointers[0].end_char == 12
    assert result.metadata["boundary_review_decisions"][0]["decision"] == "accept"
    assert result.metadata["boundary_review_decisions"][0]["resolved_cut_offset"] == 13


def test_boundary_mode_repairs_missing_anchors_from_source_excerpt(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_model = _FakeChatModel(
        {
            "parsed": {
                "cutpoints": [
                    {
                        "parent_node_id": "doc|root",
                        "source_cluster_id": "cluster-1",
                        "cut_offset": 13,
                        "boundary_kind": "sentence",
                        "cut_reason": "sentence boundary selected without anchors",
                        "reason": "sentence boundary selected without anchors",
                        "confidence": 0.9,
                    }
                ],
                "satisfied": True,
                "reasoning_history": [],
                "review_rounds": 0,
            }
        }
    )
    monkeypatch.setattr(
        "kg_doc_parser.workflow_ingest.layerwise_llm.build_chat_model_for_role",
        lambda role, settings: fake_model,
    )
    monkeypatch.setattr(
        "kg_doc_parser.workflow_ingest.layerwise_llm._boundary_prompt_candidate_context",
        lambda **kwargs: [],
    )

    events: list[dict[str, Any]] = []
    callbacks = build_layerwise_llm_callbacks(
        _provider_settings(),
        proposal_mode="boundaries",
        event_sink=lambda stage, **extra: events.append({"stage": stage, **extra}),
    )
    result = callbacks["propose_layer_fn"](
        parser_source_map=_parser_source_map(),
        current_layer_context=_boundary_context(),
        semantic_tree=_semantic_tree(),
        split_strategy="boundary_first",
        parser_input_dict=_parser_input_dict(),
        parse_session=_parse_session(),
    )

    assert result.metadata["boundary_proposed_count"] == 1
    assert result.metadata["boundary_repaired_count"] == 1
    assert result.metadata["boundary_dropped_count"] == 0
    assert result.metadata["boundary_accepted_count"] == 1
    assert len(result.children) == 2
    assert result.metadata["boundary_review_decisions"][0]["decision"] == "accept"
    assert result.metadata["boundary_review_decisions"][0]["anchor_match_mode"] in {"exact", "fuzzy"}
    assert any(event["stage"] == "workflow_layered_boundary_cutpoint_repaired" for event in events)


def test_boundary_mode_accepts_atomic_no_split_layer(monkeypatch: pytest.MonkeyPatch):
    fake_model = _FakeChatModel(
        {
            "parsed": {
                "cutpoints": [],
                "satisfied": True,
                "reasoning_history": [],
                "review_rounds": 0,
            }
        }
    )
    monkeypatch.setattr(
        "kg_doc_parser.workflow_ingest.layerwise_llm.build_chat_model_for_role",
        lambda role, settings: fake_model,
    )

    layer_events: list[dict[str, Any]] = []
    callbacks = build_layerwise_llm_callbacks(
        _provider_settings(),
        proposal_mode="boundaries",
        event_sink=lambda stage, **extra: layer_events.append({"stage": stage, **extra}),
    )
    result = callbacks["propose_layer_fn"](
        parser_source_map=_parser_source_map(),
        current_layer_context=_boundary_context(),
        semantic_tree=_semantic_tree(),
        split_strategy="boundary_first",
        parser_input_dict=_parser_input_dict(),
        parse_session=_parse_session(),
    )

    assert result.metadata["proposal_source"] == "llm"
    assert result.metadata["proposal_mode"] == "boundaries"
    assert result.metadata["boundary_atomic_decision"] is True
    assert result.metadata["boundary_proposed_count"] == 0
    assert result.metadata["boundary_accepted_count"] == 0
    assert result.metadata["boundary_shifted_count"] == 0
    assert result.metadata["boundary_rejected_count"] == 0
    assert result.metadata["boundary_summary_count"] == 0
    assert "proposal_failure_reason" not in result.metadata
    assert result.satisfied is True
    assert result.children == []
    assert any(event["stage"] == "workflow_layered_boundary_proposal_start" for event in layer_events)
    assert any(event["stage"] == "workflow_layered_boundary_assembly_skipped" for event in layer_events)
    assert layer_events[-1]["stage"] == "workflow_layered_proposal_result"
    assert layer_events[-1]["proposal_source"] == "llm"
    assert layer_events[-1]["proposal_mode"] == "boundaries"
    assert layer_events[-1]["child_count"] == 0
    assert layer_events[-1]["satisfied"] is True


class _FakeStructuredInvoker:
    def __init__(self, owner: "_FakeChatModel"):
        self._owner = owner

    def invoke(self, messages):
        self._owner.messages = messages
        response = self._owner.response
        if isinstance(response, Exception):
            raise response
        return response


class _FakeChatModel:
    def __init__(self, response: Any):
        self.response = response
        self.messages = None
        self.structured_output_kwargs: dict[str, Any] | None = None

    def with_structured_output(self, schema, include_raw: bool = True, **kwargs: Any):
        self.structured_output_kwargs = {"include_raw": include_raw, **kwargs}
        return _FakeStructuredInvoker(self)


class _SequencedStructuredInvoker:
    def __init__(self, owner: "_SequencedFakeChatModel"):
        self._owner = owner

    def invoke(self, messages):
        self._owner.messages.append(messages)
        if not self._owner.responses:
            raise RuntimeError("no more fake responses configured")
        response = self._owner.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _SequencedFakeChatModel:
    def __init__(self, responses: list[Any]):
        self.responses = list(responses)
        self.messages: list[Any] = []
        self.structured_output_kwargs: dict[str, Any] | None = None

    def with_structured_output(self, schema, include_raw: bool = True, **kwargs: Any):
        self.structured_output_kwargs = {"include_raw": include_raw, **kwargs}
        return _SequencedStructuredInvoker(self)


def test_propose_layer_fn_returns_llm_result_and_logs_source(monkeypatch: pytest.MonkeyPatch):
    parsed = CurrentLayerResult(
        children=[
            LayerChildCandidate(
                node_id="doc|root|alpha",
                parent_node_id="doc|root",
                title="Alpha",
                node_type="TEXT_FLOW",
                total_content_pointers=[
                    HydratedTextPointer(
                        source_cluster_id="cluster-1",
                        start_char=0,
                        end_char=11,
                        verbatim_text="Alpha clause",
                    )
                ],
                expandable=False,
            ),
            LayerChildCandidate(
                node_id="doc|root|beta",
                parent_node_id="doc|root",
                title="Beta",
                node_type="TEXT_FLOW",
                total_content_pointers=[
                    HydratedTextPointer(
                        source_cluster_id="cluster-1",
                        start_char=14,
                        end_char=24,
                        verbatim_text="Beta clause",
                    )
                ],
                expandable=False,
            ),
        ],
        satisfied=True,
        reasoning_history=[],
    )
    fake_model = _FakeChatModel({"parsed": parsed})
    layer_events: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "kg_doc_parser.workflow_ingest.layerwise_llm.build_chat_model_for_role",
        lambda role, settings: fake_model,
    )

    callbacks = build_layerwise_llm_callbacks(
        _provider_settings(),
        event_sink=lambda stage, **extra: layer_events.append({"stage": stage, **extra}),
    )

    result = callbacks["propose_layer_fn"](
        parser_source_map=_parser_source_map(),
        current_layer_context=_boundary_context(),
        semantic_tree=_semantic_tree(),
        split_strategy="excerpt_first",
        parser_input_dict=_parser_input_dict(),
        parse_session=_parse_session(),
    )

    assert [child.node_id for child in result.children] == ["doc|root|alpha", "doc|root|beta"]
    assert result.metadata["proposal_source"] == "llm"
    assert "proposal_failure_reason" not in result.metadata
    assert result.reasoning_history[-1].proposal_source == "llm"
    assert layer_events[-1]["stage"] == "workflow_layered_proposal_result"
    assert layer_events[-1]["proposal_source"] == "llm"
    assert any(event["stage"] == "workflow_layered_child_proposal_start" for event in layer_events)
    assert any(event["stage"] == "workflow_layered_child_proposal_completed" for event in layer_events)
    assert any(event["stage"] == "workflow_layered_child_proposal_assembled" for event in layer_events)
    assert fake_model.structured_output_kwargs and fake_model.structured_output_kwargs.get("method") == "json_schema"
    prompt_body = fake_model.messages[1][1].lower()
    assert "coarser layerwise breakdown" in prompt_body
    assert "do not recombine separated verbatim fragments" in prompt_body


def test_review_layer_fn_emits_start_and_completed_traces(monkeypatch: pytest.MonkeyPatch):
    fake_model = _FakeChatModel(
        {
            "parsed": {
                "updated_result": None,
                "coverage_ok": True,
                "satisfied": True,
                "strategy_used": "excerpt_first",
                "review_notes": ["ok"],
            }
        }
    )
    layer_events: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "kg_doc_parser.workflow_ingest.layerwise_llm.build_chat_model_for_role",
        lambda role, settings: fake_model,
    )

    callbacks = build_layerwise_llm_callbacks(
        _provider_settings(),
        event_sink=lambda stage, **extra: layer_events.append({"stage": stage, **extra}),
    )
    review = callbacks["review_layer_fn"](
        current_layer_context=_context(),
        current_layer_result=CurrentLayerResult(children=[], satisfied=True, reasoning_history=[]),
        split_strategy="excerpt_first",
        parser_source_map=_parser_source_map(),
        parse_session=_parse_session(),
    )

    assert review.satisfied is True
    assert any(event["stage"] == "workflow_layered_review_start" for event in layer_events)
    assert any(event["stage"] == "workflow_layered_review_completed" for event in layer_events)


def test_propose_layer_fn_retries_child_mode_with_previous_error_context(monkeypatch: pytest.MonkeyPatch):
    parsed = CurrentLayerResult(
        children=[
            LayerChildCandidate(
                node_id="doc|root|alpha",
                parent_node_id="doc|root",
                title="Alpha",
                node_type="TEXT_FLOW",
                total_content_pointers=[
                    HydratedTextPointer(
                        source_cluster_id="cluster-1",
                        start_char=0,
                        end_char=11,
                        verbatim_text="Alpha clause",
                    )
                ],
                expandable=False,
            ),
            LayerChildCandidate(
                node_id="doc|root|beta",
                parent_node_id="doc|root",
                title="Beta",
                node_type="TEXT_FLOW",
                total_content_pointers=[
                    HydratedTextPointer(
                        source_cluster_id="cluster-1",
                        start_char=14,
                        end_char=24,
                        verbatim_text="Beta clause",
                    )
                ],
                expandable=False,
            ),
        ],
        satisfied=True,
        reasoning_history=[],
    )
    fake_model = _SequencedFakeChatModel(
        [
            RuntimeError("transient structured output failure"),
            {"parsed": parsed},
        ]
    )
    monkeypatch.setattr(
        "kg_doc_parser.workflow_ingest.layerwise_llm.build_chat_model_for_role",
        lambda role, settings: fake_model,
    )
    settings = _provider_settings()
    settings.parser.max_retries = 1

    callbacks = build_layerwise_llm_callbacks(settings)
    result = callbacks["propose_layer_fn"](
        parser_source_map=_parser_source_map(),
        current_layer_context=_boundary_context(),
        semantic_tree=_semantic_tree(),
        split_strategy="excerpt_first",
        parser_input_dict=_parser_input_dict(),
        parse_session=_parse_session(),
    )

    assert [child.node_id for child in result.children] == ["doc|root|alpha", "doc|root|beta"]
    assert len(fake_model.messages) == 2
    assert "transient structured output failure" in fake_model.messages[1][1][1]
    assert fake_model.structured_output_kwargs and fake_model.structured_output_kwargs.get("method") == "json_schema"


def test_boundary_mode_recurses_through_refinement_for_ambiguous_cutpoints(monkeypatch: pytest.MonkeyPatch):
    ambiguous_text = "Alpha clause. Beta clause. Alpha clause. Beta clause."
    ambiguous_source_map = {
        "cluster-1": {
            "page_number": 1,
            "cluster_number": 1,
            "text": ambiguous_text,
        }
    }
    ambiguous_context = CurrentLayerContext(
        depth=0,
        parent_node_ids=["doc|root"],
        parent_titles=["Demo Doc"],
        parent_content_pointers_by_id={
            "doc|root": [
                HydratedTextPointer(
                    source_cluster_id="cluster-1",
                    start_char=0,
                    end_char=len(ambiguous_text) - 1,
                    verbatim_text=ambiguous_text,
                )
            ]
        },
        split_strategy="boundary_first",
        retry_count=0,
        max_retries=2,
    )
    fake_model = _FakeChatModel(
        {
            "parsed": {
                "cutpoints": [
                    _boundary_cutpoint_payload(ambiguous_text, 13, boundary_kind="semantic", confidence=0.2, reason="ambiguous start"),
                ],
                "satisfied": True,
                "reasoning_history": [],
                "review_rounds": 0,
            }
        }
    )
    monkeypatch.setattr(
        "kg_doc_parser.workflow_ingest.layerwise_llm.build_chat_model_for_role",
        lambda role, settings: fake_model,
    )

    callbacks = build_layerwise_llm_callbacks(
        _provider_settings(),
        proposal_mode="boundaries",
        boundary_refinement_rounds=1,
    )

    result = callbacks["propose_layer_fn"](
        parser_source_map=ambiguous_source_map,
        current_layer_context=ambiguous_context,
        semantic_tree=_semantic_tree(),
        split_strategy="boundary_first",
        parser_input_dict=_parser_input_dict(),
        parse_session=_parse_session(),
    )

    assert result.metadata["proposal_mode"] == "boundaries"
    assert result.metadata["proposal_source"] == "fallback"
    assert "no accepted cutpoints" in result.metadata["proposal_failure_reason"]


def test_boundary_and_child_modes_produce_equivalent_labels_for_same_fixture(monkeypatch: pytest.MonkeyPatch):
    child_model = _FakeChatModel(
        {
            "parsed": CurrentLayerResult(
                children=[
                    LayerChildCandidate(
                        node_id="doc|root|alpha",
                        parent_node_id="doc|root",
                        title="Alpha clause.",
                        node_type="TEXT_FLOW",
                        total_content_pointers=[
                            HydratedTextPointer(
                                source_cluster_id="cluster-1",
                                start_char=0,
                                end_char=12,
                                verbatim_text="Alpha clause.",
                            )
                        ],
                        expandable=False,
                    ),
                    LayerChildCandidate(
                        node_id="doc|root|beta",
                        parent_node_id="doc|root",
                        title="Beta clause.",
                        node_type="TEXT_FLOW",
                        total_content_pointers=[
                            HydratedTextPointer(
                                source_cluster_id="cluster-1",
                                start_char=14,
                                end_char=25,
                                verbatim_text="Beta clause.",
                            )
                        ],
                        expandable=False,
                    ),
                ],
                satisfied=True,
                reasoning_history=[],
            )
        }
    )
    boundary_model = _FakeChatModel(
        {
            "parsed": {
                "cutpoints": [
                    _boundary_cutpoint_payload("Alpha clause. Beta clause.", 13, boundary_kind="sentence", confidence=0.95, reason="sentence boundary"),
                ],
                "satisfied": True,
                "reasoning_history": [],
                "review_rounds": 0,
            }
        }
    )
    build_calls = {"count": 0}
    def _build_chat_model_for_role(role, settings):
        if build_calls["count"] == 0:
            build_calls["count"] = 1
            return child_model
        return boundary_model

    monkeypatch.setattr(
        "kg_doc_parser.workflow_ingest.layerwise_llm.build_chat_model_for_role",
        _build_chat_model_for_role,
    )

    child_callbacks = build_layerwise_llm_callbacks(_provider_settings(), proposal_mode="children")
    boundary_callbacks = build_layerwise_llm_callbacks(_provider_settings(), proposal_mode="boundaries")

    child_result = child_callbacks["propose_layer_fn"](
        parser_source_map=_parser_source_map(),
        current_layer_context=_boundary_context(),
        semantic_tree=_semantic_tree(),
        split_strategy="excerpt_first",
        parser_input_dict=_parser_input_dict(),
        parse_session=_parse_session(),
    )
    boundary_result = boundary_callbacks["propose_layer_fn"](
        parser_source_map=_parser_source_map(),
        current_layer_context=_boundary_context(),
        semantic_tree=_semantic_tree(),
        split_strategy="boundary_first",
        parser_input_dict=_parser_input_dict(),
        parse_session=_parse_session(),
    )

    assert [child.title for child in child_result.children] == [child.title for child in boundary_result.children]
    assert child_result.metadata["proposal_mode"] == "children"
    assert boundary_result.metadata["proposal_mode"] == "boundaries"
    assert boundary_result.metadata["boundary_proposed_count"] == 1


def test_boundary_summaries_survive_tree_commit_and_prompt_summary():
    root = SemanticNode(
        node_id="doc|root",
        title="Demo Doc",
        node_type="DOCUMENT_ROOT",
        total_content_pointers=[],
        child_nodes=[],
        level_from_root=0,
    )
    result = CurrentLayerResult(
        children=[
            LayerChildCandidate(
                node_id="doc|root|alpha",
                parent_node_id="doc|root",
                title="Alpha clause.",
                node_type="TEXT_FLOW",
                total_content_pointers=[
                    HydratedTextPointer(
                        source_cluster_id="cluster-1",
                        start_char=0,
                        end_char=12,
                        verbatim_text="Alpha clause.",
                    )
                ],
                expandable=False,
                metadata={
                    "source": "boundary_first",
                    "summary_text": "Alpha clause.",
                    "exact_text": "Alpha clause.",
                },
            )
        ],
        satisfied=True,
        reasoning_history=[],
    )

    committed = commit_layer_children(
        semantic_tree=root,
        current_layer_result=result,
        current_depth=0,
    )
    payload = semantic_tree_to_kge_payload(committed, doc_id="doc")
    node_payload = next(node for node in payload["nodes"] if node["id"] == "doc|root|alpha")

    assert committed.child_nodes[0].metadata["summary_text"] == "Alpha clause."
    assert committed.child_nodes[0].metadata["exact_text"] == "Alpha clause."
    assert node_payload["metadata"]["summary_text"] == "Alpha clause."
    assert node_payload["metadata"]["exact_text"] == "Alpha clause."


def test_boundary_mode_preserves_full_verbatim_text_for_returned_child_pointers(
    monkeypatch: pytest.MonkeyPatch,
):
    long_left = "Alpha " * 120
    long_right = "Beta " * 40
    text = f"{long_left}\n\n{long_right}".strip()
    cut_offset = text.index("\n\n") + 2
    fake_model = _FakeChatModel(
        {
            "parsed": {
                "cutpoints": [
                    _boundary_cutpoint_payload(
                        text,
                        cut_offset,
                        boundary_kind="paragraph",
                        confidence=0.95,
                        reason="paragraph break",
                    )
                ],
                "satisfied": True,
                "reasoning_history": [],
                "review_rounds": 0,
            }
        }
    )
    monkeypatch.setattr(
        "kg_doc_parser.workflow_ingest.layerwise_llm.build_chat_model_for_role",
        lambda role, settings: fake_model,
    )

    callbacks = build_layerwise_llm_callbacks(
        _provider_settings(),
        proposal_mode="boundaries",
    )
    parser_source_map = {
        "cluster-1": {
            "page_number": 1,
            "cluster_number": 1,
            "text": text,
        }
    }
    current_layer_context = CurrentLayerContext(
        depth=0,
        parent_node_ids=["doc|root"],
        parent_titles=["Demo Doc"],
        parent_content_pointers_by_id={
            "doc|root": [
                HydratedTextPointer(
                    source_cluster_id="cluster-1",
                    start_char=0,
                    end_char=len(text) - 1,
                    verbatim_text=text,
                )
            ]
        },
        split_strategy="boundary_first",
        retry_count=0,
        max_retries=2,
    )

    result = callbacks["propose_layer_fn"](
        parser_source_map=parser_source_map,
        current_layer_context=current_layer_context,
        semantic_tree=_semantic_tree(),
        split_strategy="boundary_first",
        parser_input_dict=_parser_input_dict(),
        parse_session=_parse_session(),
    )

    assert len(result.children) == 2
    first_pointer = result.children[0].total_content_pointers[0]
    expected_text = text[first_pointer.start_char : first_pointer.end_char + 1]
    assert first_pointer.verbatim_text == expected_text
    assert len(first_pointer.verbatim_text) > 500


@pytest.mark.parametrize(
    ("response", "reason_fragment"),
    [
        (RuntimeError("provider exploded"), "provider exploded"),
        (
            {
                "parsed": {
                    "children": [
                        {
                            "node_id": "broken-child",
                        }
                    ]
                }
            },
            "validation",
        ),
    ],
)
def test_propose_layer_fn_falls_back_and_marks_reason(
    monkeypatch: pytest.MonkeyPatch,
    response: Any,
    reason_fragment: str,
):
    fake_model = _FakeChatModel(response)
    layer_events: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "kg_doc_parser.workflow_ingest.layerwise_llm.build_chat_model_for_role",
        lambda role, settings: fake_model,
    )

    callbacks = build_layerwise_llm_callbacks(
        _provider_settings(),
        event_sink=lambda stage, **extra: layer_events.append({"stage": stage, **extra}),
    )

    result = callbacks["propose_layer_fn"](
        parser_source_map=_parser_source_map(),
        current_layer_context=_context(),
        semantic_tree=_semantic_tree(),
        split_strategy="excerpt_first",
        parser_input_dict=_parser_input_dict(),
        parse_session=_parse_session(),
    )

    assert result.metadata["proposal_source"] == "fallback"
    assert result.metadata["fallback"] == "llm_empty_or_unavailable"
    assert "proposal_failure_reason" in result.metadata
    if reason_fragment == "validation":
        assert "validation" in result.metadata["proposal_failure_reason"].lower()
    else:
        assert reason_fragment in result.metadata["proposal_failure_reason"]
    assert result.reasoning_history[-1].proposal_source == "fallback"
    assert layer_events[-1]["stage"] == "workflow_layered_proposal_result"
    assert layer_events[-1]["proposal_source"] == "fallback"
    assert "proposal_failure_reason" in layer_events[-1]


def test_propose_layer_fn_falls_back_for_structurally_empty_response(monkeypatch: pytest.MonkeyPatch):
    fake_model = _FakeChatModel({"parsed": {"children": [], "satisfied": None, "reasoning_history": []}})
    layer_events: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "kg_doc_parser.workflow_ingest.layerwise_llm.build_chat_model_for_role",
        lambda role, settings: fake_model,
    )

    callbacks = build_layerwise_llm_callbacks(
        _provider_settings(),
        event_sink=lambda stage, **extra: layer_events.append({"stage": stage, **extra}),
    )

    result = callbacks["propose_layer_fn"](
        parser_source_map=_parser_source_map(),
        current_layer_context=_context(),
        semantic_tree=_semantic_tree(),
        split_strategy="excerpt_first",
        parser_input_dict=_parser_input_dict(),
        parse_session=_parse_session(),
    )

    assert result.metadata["proposal_source"] == "fallback"
    assert "no children without satisfied=true" in result.metadata["proposal_failure_reason"]
    assert layer_events[-1]["proposal_source"] == "fallback"


def test_propose_layer_fn_rejects_single_child_fake_split(monkeypatch: pytest.MonkeyPatch):
    fake_model = _FakeChatModel(
        {
            "parsed": {
                "children": [
                    {
                        "node_id": "doc|root|only",
                        "parent_node_id": "doc|root",
                        "title": "Different title",
                        "node_type": "TEXT_FLOW",
                        "total_content_pointers": [
                            {
                                "source_cluster_id": "cluster-1",
                                "start_char": 0,
                                "end_char": 20,
                                "verbatim_text": "Alpha clause. Beta cl",
                            }
                        ],
                        "expandable": True,
                    }
                ],
                "satisfied": True,
                "reasoning_history": [],
            }
        }
    )
    monkeypatch.setattr(
        "kg_doc_parser.workflow_ingest.layerwise_llm.build_chat_model_for_role",
        lambda role, settings: fake_model,
    )

    callbacks = build_layerwise_llm_callbacks(_provider_settings())
    result = callbacks["propose_layer_fn"](
        parser_source_map=_parser_source_map(),
        current_layer_context=_context(),
        semantic_tree=_semantic_tree(),
        split_strategy="excerpt_first",
        parser_input_dict=_parser_input_dict(),
        parse_session=_parse_session(),
    )

    assert result.metadata["proposal_source"] == "fallback"
    assert "single child" in result.metadata["proposal_failure_reason"]


def test_propose_layer_fn_rejects_pointer_outside_source_map(monkeypatch: pytest.MonkeyPatch):
    fake_model = _FakeChatModel(
        {
            "parsed": {
                "children": [
                    {
                        "node_id": "doc|root|alpha",
                        "parent_node_id": "doc|root",
                        "title": "Alpha",
                        "node_type": "TEXT_FLOW",
                        "total_content_pointers": [
                            {
                                "source_cluster_id": "missing-cluster",
                                "start_char": 0,
                                "end_char": 5,
                                "verbatim_text": "Alpha",
                            }
                        ],
                        "expandable": False,
                    },
                    {
                        "node_id": "doc|root|beta",
                        "parent_node_id": "doc|root",
                        "title": "Beta",
                        "node_type": "TEXT_FLOW",
                        "total_content_pointers": [
                            {
                                "source_cluster_id": "cluster-1",
                                "start_char": 14,
                                "end_char": 24,
                                "verbatim_text": "Beta clause",
                            }
                        ],
                        "expandable": False,
                    },
                ],
                "satisfied": True,
                "reasoning_history": [],
            }
        }
    )
    monkeypatch.setattr(
        "kg_doc_parser.workflow_ingest.layerwise_llm.build_chat_model_for_role",
        lambda role, settings: fake_model,
    )

    callbacks = build_layerwise_llm_callbacks(_provider_settings())
    result = callbacks["propose_layer_fn"](
        parser_source_map=_parser_source_map(),
        current_layer_context=_context(),
        semantic_tree=_semantic_tree(),
        split_strategy="excerpt_first",
        parser_input_dict=_parser_input_dict(),
        parse_session=_parse_session(),
    )

    assert result.metadata["proposal_source"] == "fallback"
    assert "source map" in result.metadata["proposal_failure_reason"]


def test_propose_layer_fn_rejects_child_pointer_outside_parent_span(monkeypatch: pytest.MonkeyPatch):
    fake_model = _FakeChatModel(
        {
            "parsed": {
                "children": [
                    {
                        "node_id": "doc|root|alpha",
                        "parent_node_id": "doc|root",
                        "title": "Alpha",
                        "node_type": "TEXT_FLOW",
                        "total_content_pointers": [
                            {
                                "source_cluster_id": "cluster-1",
                                "start_char": 0,
                                "end_char": 5,
                                "verbatim_text": "Alpha ",
                            }
                        ],
                        "expandable": False,
                    },
                    {
                        "node_id": "doc|root|outside",
                        "parent_node_id": "doc|root",
                        "title": "Outside",
                        "node_type": "TEXT_FLOW",
                        "total_content_pointers": [
                            {
                                "source_cluster_id": "cluster-1",
                                "start_char": 21,
                                "end_char": 24,
                                "verbatim_text": "ause",
                            }
                        ],
                        "expandable": False,
                    },
                ],
                "satisfied": True,
                "reasoning_history": [],
            }
        }
    )
    monkeypatch.setattr(
        "kg_doc_parser.workflow_ingest.layerwise_llm.build_chat_model_for_role",
        lambda role, settings: fake_model,
    )

    callbacks = build_layerwise_llm_callbacks(_provider_settings())
    result = callbacks["propose_layer_fn"](
        parser_source_map=_parser_source_map(),
        current_layer_context=_context(),
        semantic_tree=_semantic_tree(),
        split_strategy="excerpt_first",
        parser_input_dict=_parser_input_dict(),
        parse_session=_parse_session(),
    )

    assert result.metadata["proposal_source"] == "fallback"
    assert "parent span" in result.metadata["proposal_failure_reason"]


def test_propose_layer_fn_rejects_child_pointer_verbatim_mismatch(monkeypatch: pytest.MonkeyPatch):
    fake_model = _FakeChatModel(
        {
            "parsed": {
                "children": [
                    {
                        "node_id": "doc|root|alpha",
                        "parent_node_id": "doc|root",
                        "title": "Alpha",
                        "node_type": "TEXT_FLOW",
                        "total_content_pointers": [
                            {
                                "source_cluster_id": "cluster-1",
                                "start_char": 0,
                                "end_char": 5,
                                "verbatim_text": "Alpha ",
                            }
                        ],
                        "expandable": False,
                    },
                    {
                        "node_id": "doc|root|beta",
                        "parent_node_id": "doc|root",
                        "title": "Beta",
                        "node_type": "TEXT_FLOW",
                        "total_content_pointers": [
                            {
                                "source_cluster_id": "cluster-1",
                                "start_char": 14,
                                "end_char": 17,
                                "verbatim_text": "WRONG",
                            }
                        ],
                        "expandable": False,
                    },
                ],
                "satisfied": True,
                "reasoning_history": [],
            }
        }
    )
    monkeypatch.setattr(
        "kg_doc_parser.workflow_ingest.layerwise_llm.build_chat_model_for_role",
        lambda role, settings: fake_model,
    )

    callbacks = build_layerwise_llm_callbacks(_provider_settings())
    result = callbacks["propose_layer_fn"](
        parser_source_map=_parser_source_map(),
        current_layer_context=_context(),
        semantic_tree=_semantic_tree(),
        split_strategy="excerpt_first",
        parser_input_dict=_parser_input_dict(),
        parse_session=_parse_session(),
    )

    assert result.metadata["proposal_source"] == "fallback"
    assert "verbatim_text" in result.metadata["proposal_failure_reason"]
