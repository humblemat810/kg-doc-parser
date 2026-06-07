from __future__ import annotations

from typing import Any

import pytest

from kg_doc_parser.workflow_ingest import ProviderEndpointConfig, WorkflowProviderSettings
from kg_doc_parser.workflow_ingest.layerwise_llm import build_layerwise_llm_callbacks
from kg_doc_parser.workflow_ingest.models import (
    BoundaryCutpoint,
    LLMBoundaryProposal,
    LLMBoundaryProposalBatch,
    BoundaryReviewBatch,
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
            parent_node_id="doc|root",
            source_cluster_id="cluster-1",
            cut_offset=12,
            boundary_kind="semantic",
            confidence=0.5,
            reason="near the paragraph boundary",
        ),
        current_layer_context=_context(),
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
            confidence=0.5,
            reason="inside token",
        ),
        current_layer_context=_context(),
        parser_source_map=_parser_source_map(),
    )
    assert reject_decision.decision == "reject"
    assert "word" in (reject_decision.reason or "")

    reversed_batch = LLMBoundaryProposalBatch(
        cutpoints=[
            BoundaryCutpoint(
                parent_node_id="doc|root",
                source_cluster_id="cluster-1",
                cut_offset=20,
                boundary_kind="sentence",
                confidence=0.8,
                reason="later boundary",
            ),
            BoundaryCutpoint(
                parent_node_id="doc|root",
                source_cluster_id="cluster-1",
                cut_offset=10,
                boundary_kind="sentence",
                confidence=0.8,
                reason="earlier boundary",
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
            BoundaryCutpoint(
                parent_node_id="doc|root",
                source_cluster_id="cluster-1",
                cut_offset=13,
                boundary_kind="sentence",
                confidence=0.9,
                reason="sentence boundary",
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


def test_workflow_provider_settings_from_env_enables_boundary_mode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KG_DOC_PARSER_PROPOSAL_MODE", "boundaries")
    settings = WorkflowProviderSettings.from_env()
    assert settings.proposal_mode == "boundaries"

    fake_model = _FakeChatModel(
        {
            "parsed": {
                "cutpoints": [
                    {
                        "parent_node_id": "doc|root",
                        "source_cluster_id": "cluster-1",
                        "cut_offset": 14,
                        "boundary_kind": "sentence",
                        "confidence": 0.92,
                        "reason": "sentence break",
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


def test_boundary_mode_proposes_cutpoints_and_assembles_children(monkeypatch: pytest.MonkeyPatch):
    fake_model = _FakeChatModel(
        {
            "parsed": {
                "cutpoints": [
                    {
                        "parent_node_id": "doc|root",
                        "source_cluster_id": "cluster-1",
                        "cut_offset": 12,
                        "boundary_kind": "semantic",
                        "confidence": 0.4,
                        "reason": "near sentence boundary",
                    },
                    {
                        "parent_node_id": "doc|root",
                        "source_cluster_id": "cluster-1",
                        "cut_offset": 16,
                        "boundary_kind": "semantic",
                        "confidence": 0.92,
                        "reason": "inside word and should be rejected",
                    },
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
    assert layer_events[-1]["stage"] == "workflow_layered_proposal_result"
    assert layer_events[-1]["proposal_mode"] == "boundaries"
    assert result.children
    assert [child.parent_node_id for child in result.children] == ["doc|root", "doc|root"]


def test_boundary_mode_rejects_no_split_layer(monkeypatch: pytest.MonkeyPatch):
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

    callbacks = build_layerwise_llm_callbacks(_provider_settings(), proposal_mode="boundaries")
    result = callbacks["propose_layer_fn"](
        parser_source_map=_parser_source_map(),
        current_layer_context=_boundary_context(),
        semantic_tree=_semantic_tree(),
        split_strategy="boundary_first",
        parser_input_dict=_parser_input_dict(),
        parse_session=_parse_session(),
    )

    assert result.metadata["proposal_source"] == "fallback"
    assert "no accepted cutpoints" in result.metadata["proposal_failure_reason"]


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
        current_layer_context=_context(),
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
    assert fake_model.structured_output_kwargs and fake_model.structured_output_kwargs.get("method") == "json_schema"
    prompt_body = fake_model.messages[1][1].lower()
    assert "coarser layerwise breakdown" in prompt_body
    assert "do not recombine separated verbatim fragments" in prompt_body


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
        current_layer_context=_context(),
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
    ambiguous_text = "Alpha clause;more text"
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
                    {
                        "parent_node_id": "doc|root",
                        "source_cluster_id": "cluster-1",
                        "cut_offset": 13,
                        "boundary_kind": "semantic",
                        "confidence": 0.2,
                        "reason": "ambiguous start",
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
                    {
                        "parent_node_id": "doc|root",
                        "source_cluster_id": "cluster-1",
                        "cut_offset": 13,
                        "boundary_kind": "sentence",
                        "confidence": 0.95,
                        "reason": "sentence boundary",
                    }
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
        current_layer_context=_context(),
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
                                "verbatim_text": "Alpha clause. Beta",
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
