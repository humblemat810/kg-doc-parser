from __future__ import annotations

from typing import Any

import pytest
from layerwise_test_support import (
    _boundary_context,
    _boundary_cutpoint_payload,
    _context,
    _parse_session,
    _parser_input_dict,
    _parser_source_map,
    _provider_settings,
    _semantic_tree,
)

from kg_doc_parser.workflow_ingest import (
    WorkflowProviderSettings,
)
from kg_doc_parser.workflow_ingest.layerwise_boundary_pipeline import (
    _boundary_identical_parent_child_ids,
    _make_boundary_summary,
    _pointer_excerpt,
)
from kg_doc_parser.workflow_ingest.layerwise_llm import (
    _fallback_layer_result,
    build_layerwise_llm_callbacks,
)
from kg_doc_parser.workflow_ingest.layerwise_payloads import _dump_model
from kg_doc_parser.workflow_ingest.layerwise_source_spans import (
    _pointer_field,
    _pointer_text,
    _source_map_excerpt,
)
from kg_doc_parser.workflow_ingest.models import (
    BoundaryCutpoint,
    CurrentLayerContext,
    CurrentLayerResult,
    LayerChildCandidate,
    LLMBoundaryProposalBatch,
    LLMCurrentLayerReview,
    ParseSessionState,
)
from kg_doc_parser.workflow_ingest.parser_core import (
    commit_layer_children,
    review_layer,
    source_map_fingerprint,
)
from kg_doc_parser.workflow_ingest.semantics import (
    HydratedTextPointer,
    SemanticNode,
    correct_and_validate_pointer,
    hydrate_pointer_from_offsets,
    pointer_source_validation_error,
    semantic_tree_to_kge_payload,
)


@pytest.mark.ci
def test_layerwise_internal_model_and_pointer_seams_are_typed() -> None:
    class _ModelDump:
        def model_dump(self, **kwargs: object) -> dict[str, object]:
            assert kwargs == {"field_mode": "backend", "dump_format": "json"}
            return {"value": "model"}

    pointer = HydratedTextPointer(
        source_cluster_id="cluster-1",
        start_char=2,
        end_char=4,
        verbatim_text="abc",
    )

    assert _dump_model(_ModelDump()) == {"value": "model"}
    assert _pointer_field(pointer, "start_char") == 2
    assert _pointer_field({"start_char": 3}, "start_char") == 3
    assert _pointer_field(object(), "start_char") is None


@pytest.mark.ci
def test_source_prompt_helpers_fail_closed_for_malformed_records() -> None:
    source_map = {
        "missing": None,
        "non_text": {"text": 123},
        "valid": {"text": "Authoritative"},
    }
    pointer = {
        "source_cluster_id": "non_text",
        "start_char": 0,
        "end_char": -1,
        "verbatim_text": "provider supplied text",
    }

    assert _source_map_excerpt(source_map)["missing"]["text"] == ""
    assert _source_map_excerpt(source_map)["non_text"]["text"] == ""
    assert _pointer_text(pointer, parser_source_map=source_map) == ""
    assert (
        _pointer_excerpt(pointer, parser_source_map=source_map)["verbatim_text"] == ""
    )
    summary = _make_boundary_summary(
        parent_node_id="root",
        source_cluster_id="non_text",
        start_char=0,
        end_char=4,
        parser_source_map=source_map,
        boundary_kind="paragraph",
    )
    assert summary.exact_text == ""
    assert summary.summary_text == ""


@pytest.mark.ci
def test_pointer_prompt_text_resolves_full_source_sentinel_from_authority() -> None:
    pointer = {
        "source_cluster_id": "valid",
        "start_char": 3,
        "end_char": -1,
        "verbatim_text": "wrong provider transcription",
    }

    assert (
        _pointer_text(
            pointer, parser_source_map={"valid": {"text": ":: Authoritative"}}
        )
        == "Authoritative"
    )


@pytest.mark.ci
def test_fallback_layer_skips_malformed_source_records() -> None:
    result = _fallback_layer_result(
        current_layer_context=_context(),
        parser_source_map={
            "bad-none": None,
            "bad-value": {"text": 42},
            "valid": {"text": "# Valid heading\nbody"},
        },
    )

    assert len(result.children) == 1
    assert result.children[0].total_content_pointers[0].source_cluster_id == "valid"
    assert (
        result.children[0].total_content_pointers[0].verbatim_text
        == "# Valid heading\nbody"
    )


@pytest.mark.ci
@pytest.mark.ci
def test_workflow_provider_settings_from_env_enables_boundary_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("KG_DOC_PARSER_PROPOSAL_MODE", "boundaries")
    settings = WorkflowProviderSettings.from_env()
    assert settings.proposal_mode == "boundaries"

    fake_model = _FakeChatModel(
        {
            "parsed": {
                "cutpoints": [
                    _boundary_cutpoint_payload(
                        "Alpha clause. Beta clause.",
                        14,
                        boundary_kind="sentence",
                        confidence=0.92,
                        reason="sentence break",
                    ),
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
    assert (
        fake_model.structured_output_kwargs
        and fake_model.structured_output_kwargs.get("method") == "json_schema"
    )


def test_structured_invoke_returns_typed_pydantic_model():
    from pydantic import BaseModel

    from kg_doc_parser.workflow_ingest.layerwise_invocation import structured_invoke

    class _Schema(BaseModel):
        value: int
        label: str

    fake_model = _FakeChatModel({"parsed": {"value": 7, "label": "demo"}})
    result = structured_invoke(fake_model, _Schema, [("human", "hello")])

    assert isinstance(result, _Schema)
    assert result.value == 7
    assert result.label == "demo"
    assert (
        fake_model.structured_output_kwargs
        and fake_model.structured_output_kwargs.get("method") == "json_schema"
    )


def test_structured_invoke_classifies_parse_failures_for_observability():
    from pydantic import BaseModel

    from kg_doc_parser.workflow_ingest.layerwise_invocation import structured_invoke

    class _Schema(BaseModel):
        value: int

    events: list[dict[str, object]] = []
    with pytest.raises(ValueError):
        structured_invoke(
            _FakeChatModel({"parsed": {"value": "not-an-int"}}),
            _Schema,
            [("human", "hello")],
            diagnostics_sink=events.append,
        )

    assert events[0]["failure_type"] == "structured_output_parse_failure"
    assert events[0]["error_type"]


def test_structured_invoke_rejects_free_form_boundary_payload() -> None:
    """A provider's unrelated JSON shape must not bypass schema validation."""
    from pydantic import ValidationError

    from kg_doc_parser.workflow_ingest.layerwise_invocation import structured_invoke

    events: list[dict[str, object]] = []
    response = {
        "parsed": {
            "semantic_boundaries": {"blocks": []},
        },
        "raw": None,
        "parsing_error": None,
    }

    with pytest.raises(ValidationError):
        structured_invoke(
            _FakeChatModel(response),
            LLMBoundaryProposalBatch,
            [("human", "return boundary proposals")],
            diagnostics_sink=events.append,
        )

    assert events[0]["failure_type"] == "structured_output_parse_failure"
    assert events[0]["error_type"] == "ValidationError"


def test_boundary_mode_proposes_cutpoints_and_assembles_children(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_model = _FakeChatModel(
        {
            "parsed": {
                "cutpoints": [
                    _boundary_cutpoint_payload(
                        "Alpha clause. Beta clause.",
                        12,
                        anchor_offset=13,
                        boundary_kind="semantic",
                        confidence=0.4,
                        reason="near sentence boundary",
                    ),
                    _boundary_cutpoint_payload(
                        "Alpha clause. Beta clause.",
                        16,
                        anchor_offset=16,
                        boundary_kind="semantic",
                        confidence=0.92,
                        reason="inside word and should be rejected",
                    ),
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
        event_sink=lambda stage, **extra: layer_events.append(
            {"stage": stage, **extra}
        ),
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
    assert result.metadata["boundary_parent_coverage"][0]["coverage_state"] in {
        "covered",
        "covered_with_gaps",
    }
    assert result.metadata["boundary_refinement_attempts"] == 0
    assert result.reasoning_history[-1].proposal_mode == "boundaries"
    assert (
        fake_model.structured_output_kwargs
        and fake_model.structured_output_kwargs.get("method") == "json_schema"
    )
    assert any(
        event["stage"] == "workflow_layered_boundary_proposal_start"
        for event in layer_events
    )
    assert any(
        event["stage"] == "workflow_layered_boundary_review_completed"
        for event in layer_events
    )
    review_events = [
        event
        for event in layer_events
        if event["stage"] == "workflow_layered_boundary_review_completed"
    ]
    assert review_events[0]["failure_type"] == "anchor_ambiguity"
    assert any(
        event["stage"] == "workflow_layered_boundary_assembly_start"
        for event in layer_events
    )
    assert any(
        event["stage"] == "workflow_layered_boundary_assembly_completed"
        for event in layer_events
    )
    provider_events = [
        event for event in layer_events if event["stage"] == "workflow_provider_call"
    ]
    assert provider_events
    assert provider_events[0]["operation"] == "layer_boundary_proposal"
    assert provider_events[0]["success"] is True
    assert "provider_metrics" in provider_events[0]
    assert layer_events[-1]["stage"] == "workflow_layered_proposal_result"
    assert layer_events[-1]["proposal_mode"] == "boundaries"
    assert result.children
    assert [child.parent_node_id for child in result.children] == [
        "doc|root",
        "doc|root",
    ]


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
    assert any(
        event["stage"] == "workflow_layered_boundary_cutpoint_dropped"
        for event in events
    )


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
        "kg_doc_parser.workflow_ingest.layerwise_boundary_pipeline._boundary_prompt_candidate_context",
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
    assert result.metadata["boundary_review_decisions"][0]["anchor_match_mode"] in {
        "exact",
        "fuzzy",
    }
    assert any(
        event["stage"] == "workflow_layered_boundary_cutpoint_repaired"
        for event in events
    )


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
        event_sink=lambda stage, **extra: layer_events.append(
            {"stage": stage, **extra}
        ),
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
    assert any(
        event["stage"] == "workflow_layered_boundary_proposal_start"
        for event in layer_events
    )
    assert any(
        event["stage"] == "workflow_layered_boundary_assembly_skipped"
        for event in layer_events
    )
    assert layer_events[-1]["stage"] == "workflow_layered_proposal_result"
    assert layer_events[-1]["proposal_source"] == "llm"
    assert layer_events[-1]["proposal_mode"] == "boundaries"
    assert layer_events[-1]["child_count"] == 0
    assert layer_events[-1]["satisfied"] is True


def test_boundary_mode_detects_child_with_identical_parent_span() -> None:
    child = LayerChildCandidate(
        node_id="doc|root|cluster-1|unit-0",
        parent_node_id="doc|root",
        title="same span",
        node_type="TEXT_FLOW",
        total_content_pointers=[
            HydratedTextPointer(
                source_cluster_id="cluster-1",
                start_char=0,
                end_char=25,
                verbatim_text="Alpha clause. Beta clause.",
            )
        ],
    )

    assert _boundary_identical_parent_child_ids(
        current_layer_context=_boundary_context(),
        current_layer_result=CurrentLayerResult(children=[child], satisfied=True),
    ) == ["doc|root|cluster-1|unit-0"]


class _FakeStructuredInvoker:
    def __init__(self, owner: _FakeChatModel):
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
    def __init__(self, owner: _SequencedFakeChatModel):
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


def test_propose_layer_fn_returns_llm_result_and_logs_source(
    monkeypatch: pytest.MonkeyPatch,
):
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
        event_sink=lambda stage, **extra: layer_events.append(
            {"stage": stage, **extra}
        ),
    )

    result = callbacks["propose_layer_fn"](
        parser_source_map=_parser_source_map(),
        current_layer_context=_boundary_context(),
        semantic_tree=_semantic_tree(),
        split_strategy="excerpt_first",
        parser_input_dict=_parser_input_dict(),
        parse_session=_parse_session(),
    )

    assert [child.node_id for child in result.children] == [
        "doc|root|alpha",
        "doc|root|beta",
    ]
    assert result.metadata["proposal_source"] == "llm"
    assert "proposal_failure_reason" not in result.metadata
    assert result.reasoning_history[-1].proposal_source == "llm"
    assert layer_events[-1]["stage"] == "workflow_layered_proposal_result"
    assert layer_events[-1]["proposal_source"] == "llm"
    assert any(
        event["stage"] == "workflow_layered_child_proposal_start"
        for event in layer_events
    )
    assert any(
        event["stage"] == "workflow_layered_child_proposal_completed"
        for event in layer_events
    )
    assert any(
        event["stage"] == "workflow_layered_child_proposal_assembled"
        for event in layer_events
    )
    assert (
        fake_model.structured_output_kwargs
        and fake_model.structured_output_kwargs.get("method") == "json_schema"
    )
    prompt_body = fake_model.messages[1][1].lower()
    assert "coarser layerwise breakdown" in prompt_body
    assert "do not recombine separated verbatim fragments" in prompt_body


def test_review_layer_fn_emits_start_and_completed_traces(
    monkeypatch: pytest.MonkeyPatch,
):
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
        event_sink=lambda stage, **extra: layer_events.append(
            {"stage": stage, **extra}
        ),
    )
    review = callbacks["review_layer_fn"](
        current_layer_context=_context(),
        current_layer_result=CurrentLayerResult(
            children=[], satisfied=True, reasoning_history=[]
        ),
        split_strategy="excerpt_first",
        parser_source_map=_parser_source_map(),
        parse_session=_parse_session(),
    )

    assert review.satisfied is True
    assert any(
        event["stage"] == "workflow_layered_review_start" for event in layer_events
    )
    assert any(
        event["stage"] == "workflow_layered_review_completed" for event in layer_events
    )


@pytest.mark.ci
def test_review_layer_hydrates_offset_only_payload_through_core_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "# SDK\n\nStatus: SDK\\_UNINITIALIZED | See \\[the docs\\] for details.\n"
    split_at = source.index("Status")
    source_map = {"cluster-1": {"text": source}}
    fake_model = _FakeChatModel(
        {
            "parsed": {
                "updated_result": {
                    "children": [
                        {
                            "node_id": "doc|root|source",
                            "parent_node_id": "doc|root",
                            "title": "Source",
                            "node_type": "TEXT_FLOW",
                            "total_content_pointers": [
                                {
                                    "source_cluster_id": "cluster-1",
                                    "start_char": 0,
                                    "end_char": split_at - 1,
                                    "verbatim_text": "model transcription with a dropped \\[ escape",
                                }
                            ],
                            "expandable": False,
                        },
                        {
                            "node_id": "doc|root|details",
                            "parent_node_id": "doc|root",
                            "title": "Details",
                            "node_type": "TEXT_FLOW",
                            "total_content_pointers": [
                                {
                                    "source_cluster_id": "cluster-1",
                                    "start_char": split_at,
                                    "end_char": len(source) - 1,
                                    "verbatim_text": "another model transcription",
                                }
                            ],
                            "expandable": False,
                        },
                    ],
                    "satisfied": True,
                    "reasoning_history": [],
                    "review_rounds": 0,
                },
                "coverage_ok": True,
                "satisfied": True,
                "strategy_used": "excerpt_first",
                "review_notes": ["offset-only review"],
            }
        }
    )
    monkeypatch.setattr(
        "kg_doc_parser.workflow_ingest.layerwise_llm.build_chat_model_for_role",
        lambda role, settings: fake_model,
    )
    callbacks = build_layerwise_llm_callbacks(_provider_settings())
    context = CurrentLayerContext(
        depth=0,
        parent_node_ids=["doc|root"],
        parent_content_pointers_by_id={
            "doc|root": [
                HydratedTextPointer(
                    source_cluster_id="cluster-1",
                    start_char=0,
                    end_char=len(source) - 1,
                    verbatim_text=source,
                )
            ]
        },
        split_strategy="excerpt_first",
    )
    session = ParseSessionState(
        collection_id="doc",
        root_node_id="doc|root",
        mode="workflow_layered",
        metadata={"source_map_fingerprint": source_map_fingerprint(source_map)},
    )

    review, _ = review_layer(
        parse_session=session,
        current_layer_context=context,
        current_layer_result=CurrentLayerResult(children=[], satisfied=True),
        parser_source_map=source_map,
        review_layer_fn=callbacks["review_layer_fn"],
    )

    assert review.satisfied is True
    assert review.coverage_ok is True
    assert review.updated_result is not None
    pointer = review.updated_result.children[0].total_content_pointers[0]
    assert pointer.start_char == 0
    assert pointer.end_char == split_at - 1
    assert pointer.verbatim_text == source[:split_at]


@pytest.mark.ci
def test_recursive_reviews_rehydrate_each_depth_from_the_same_source_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "Section heading\nChild detail\nGrandchild detail"
    source_map = {"cluster-1": {"text": source}}
    section_start = source.index("Section heading")
    grandchild_start = source.index("Grandchild detail")
    fake_model = _SequencedFakeChatModel(
        [
            {
                "parsed": {
                    "updated_result": {
                        "children": [
                            {
                                "node_id": "doc|root|section",
                                "parent_node_id": "doc|root",
                                "title": "Section",
                                "node_type": "SECTION",
                                "total_content_pointers": [
                                    {
                                        "source_cluster_id": "cluster-1",
                                        "start_char": section_start,
                                        "end_char": len(source) - 1,
                                        "verbatim_text": "lossy section transcription",
                                    }
                                ],
                                "expandable": True,
                            }
                        ],
                        "satisfied": True,
                        "reasoning_history": [],
                    },
                    "coverage_ok": True,
                    "satisfied": True,
                    "strategy_used": "excerpt_first",
                }
            },
            {
                "parsed": {
                    "updated_result": {
                        "children": [
                            {
                                "node_id": "doc|root|section|grandchild",
                                "parent_node_id": "doc|root|section",
                                "title": "Grandchild",
                                "node_type": "TEXT_FLOW",
                                "total_content_pointers": [
                                    {
                                        "source_cluster_id": "cluster-1",
                                        "start_char": grandchild_start,
                                        "end_char": len(source) - 1,
                                        "verbatim_text": "another lossy transcription",
                                    }
                                ],
                                "expandable": False,
                            }
                        ],
                        "satisfied": True,
                        "reasoning_history": [],
                    },
                    "coverage_ok": True,
                    "satisfied": True,
                    "strategy_used": "excerpt_first",
                }
            },
        ]
    )
    monkeypatch.setattr(
        "kg_doc_parser.workflow_ingest.layerwise_llm.build_chat_model_for_role",
        lambda role, settings: fake_model,
    )
    callbacks = build_layerwise_llm_callbacks(_provider_settings())
    session = _parse_session().model_copy(
        update={
            "metadata": {"source_map_fingerprint": source_map_fingerprint(source_map)}
        }
    )

    first_context = _context()
    first_review, _ = review_layer(
        parse_session=session,
        current_layer_context=first_context,
        current_layer_result=CurrentLayerResult(children=[], satisfied=True),
        parser_source_map=source_map,
        review_layer_fn=callbacks["review_layer_fn"],
    )
    second_context = first_context.model_copy(
        update={
            "depth": 1,
            "parent_node_ids": ["doc|root|section"],
            "parent_content_pointers_by_id": {
                "doc|root|section": [
                    HydratedTextPointer(
                        source_cluster_id="cluster-1",
                        start_char=section_start,
                        end_char=len(source) - 1,
                        verbatim_text=source[section_start:],
                    )
                ]
            },
        }
    )
    second_review, _ = review_layer(
        parse_session=session,
        current_layer_context=second_context,
        current_layer_result=first_review.updated_result or CurrentLayerResult(),
        parser_source_map=source_map,
        review_layer_fn=callbacks["review_layer_fn"],
    )

    assert first_review.updated_result is not None
    assert second_review.updated_result is not None
    assert (
        first_review.updated_result.children[0].total_content_pointers[0].verbatim_text
        == source[section_start:]
    )
    assert (
        second_review.updated_result.children[0].total_content_pointers[0].verbatim_text
        == source[grandchild_start:]
    )
    assert len(fake_model.messages) == 2


@pytest.mark.ci
def test_malformed_structured_review_payload_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = _FakeChatModel(
        {
            "parsed": {
                "updated_result": {
                    "children": [
                        {
                            "node_id": "doc|root|bad",
                            "parent_node_id": "doc|root",
                            "title": "Bad",
                            "node_type": "TEXT_FLOW",
                            "total_content_pointers": [
                                {
                                    "source_cluster_id": "cluster-1",
                                    "start_char": "zero",
                                    "end_char": 2,
                                    "verbatim_text": "bad",
                                }
                            ],
                        }
                    ],
                    "satisfied": True,
                },
                "coverage_ok": True,
                "satisfied": True,
                "strategy_used": "excerpt_first",
            }
        }
    )
    monkeypatch.setattr(
        "kg_doc_parser.workflow_ingest.layerwise_llm.build_chat_model_for_role",
        lambda role, settings: fake_model,
    )
    callbacks = build_layerwise_llm_callbacks(_provider_settings())
    review, _ = review_layer(
        parse_session=_parse_session(),
        current_layer_context=_context(),
        current_layer_result=CurrentLayerResult(children=[], satisfied=True),
        parser_source_map=_parser_source_map(),
        review_layer_fn=callbacks["review_layer_fn"],
    )

    assert review.satisfied is None
    assert review.coverage_ok is None
    assert review.metadata["review_failure"] == "provider_failure"


def test_propose_layer_fn_retries_child_mode_with_previous_error_context(
    monkeypatch: pytest.MonkeyPatch,
):
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

    assert [child.node_id for child in result.children] == [
        "doc|root|alpha",
        "doc|root|beta",
    ]
    assert len(fake_model.messages) == 2
    assert "transient structured output failure" in fake_model.messages[1][1][1]
    assert (
        fake_model.structured_output_kwargs
        and fake_model.structured_output_kwargs.get("method") == "json_schema"
    )


def test_boundary_mode_recurses_through_refinement_for_ambiguous_cutpoints(
    monkeypatch: pytest.MonkeyPatch,
):
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
                    _boundary_cutpoint_payload(
                        ambiguous_text,
                        13,
                        boundary_kind="semantic",
                        confidence=0.2,
                        reason="ambiguous start",
                    ),
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

    events: list[dict[str, Any]] = []
    callbacks = build_layerwise_llm_callbacks(
        _provider_settings(),
        proposal_mode="boundaries",
        boundary_refinement_rounds=1,
        event_sink=lambda stage, **extra: events.append({"stage": stage, **extra}),
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
    assert result.metadata["failure_type"] == "fallback"
    assert "no accepted cutpoints" in result.metadata["proposal_failure_reason"]
    assert any(event.get("failure_type") == "semantic_rejection" for event in events)


def test_boundary_and_child_modes_produce_equivalent_labels_for_same_fixture(
    monkeypatch: pytest.MonkeyPatch,
):
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
                    _boundary_cutpoint_payload(
                        "Alpha clause. Beta clause.",
                        13,
                        boundary_kind="sentence",
                        confidence=0.95,
                        reason="sentence boundary",
                    ),
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

    child_callbacks = build_layerwise_llm_callbacks(
        _provider_settings(), proposal_mode="children"
    )
    boundary_callbacks = build_layerwise_llm_callbacks(
        _provider_settings(), proposal_mode="boundaries"
    )

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

    assert [child.title for child in child_result.children] == [
        child.title for child in boundary_result.children
    ]
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
                metadata={
                    "source": "boundary_first",
                    "summary_text": "Beta clause.",
                    "exact_text": "Beta clause.",
                },
            ),
        ],
        satisfied=True,
        reasoning_history=[],
    )

    committed = commit_layer_children(
        semantic_tree=root,
        current_layer_result=result,
        current_depth=0,
    )
    payload = semantic_tree_to_kge_payload(
        committed,
        doc_id="doc",
        parser_source_map={"cluster-1": {"text": "Alpha clause. Beta clause."}},
    )
    node_payload = next(
        node for node in payload["nodes"] if node["id"] == "doc|root|alpha"
    )

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
        event_sink=lambda stage, **extra: layer_events.append(
            {"stage": stage, **extra}
        ),
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
    assert result.metadata["failure_type"] == "fallback"
    assert "proposal_failure_reason" in result.metadata
    if reason_fragment == "validation":
        assert "validation" in result.metadata["proposal_failure_reason"].lower()
    else:
        assert reason_fragment in result.metadata["proposal_failure_reason"]
    assert result.reasoning_history[-1].proposal_source == "fallback"
    assert layer_events[-1]["stage"] == "workflow_layered_proposal_result"
    assert layer_events[-1]["proposal_source"] == "fallback"
    assert layer_events[-1]["failure_type"] == "fallback"
    assert "proposal_failure_reason" in layer_events[-1]


def test_propose_layer_fn_falls_back_for_structurally_empty_response(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_model = _FakeChatModel(
        {"parsed": {"children": [], "satisfied": None, "reasoning_history": []}}
    )
    layer_events: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "kg_doc_parser.workflow_ingest.layerwise_llm.build_chat_model_for_role",
        lambda role, settings: fake_model,
    )

    callbacks = build_layerwise_llm_callbacks(
        _provider_settings(),
        event_sink=lambda stage, **extra: layer_events.append(
            {"stage": stage, **extra}
        ),
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
    assert result.metadata["failure_type"] == "fallback"
    assert (
        "no children without satisfied=true"
        in result.metadata["proposal_failure_reason"]
    )
    assert layer_events[-1]["proposal_source"] == "fallback"
    assert layer_events[-1]["failure_type"] == "fallback"


def test_propose_layer_fn_rejects_single_child_fake_split(
    monkeypatch: pytest.MonkeyPatch,
):
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


def test_propose_layer_fn_rejects_pointer_outside_source_map(
    monkeypatch: pytest.MonkeyPatch,
):
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


def test_propose_layer_fn_rejects_child_pointer_outside_parent_span(
    monkeypatch: pytest.MonkeyPatch,
):
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


def test_propose_layer_fn_rehydrates_child_pointer_from_offsets(
    monkeypatch: pytest.MonkeyPatch,
):
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

    assert result.metadata["proposal_source"] == "llm"
    pointers = [
        pointer for child in result.children for pointer in child.total_content_pointers
    ]
    assert [pointer.verbatim_text for pointer in pointers] == ["Alpha ", "Beta"]


@pytest.mark.parametrize(
    "model_text",
    [
        "Status: SDK\\_UNINITIALIZED | See [the docs] for details.",
        "Status: SDK\\_UNINITIALIZED | See the docs for details.",
        "Status: SDK\\_UNINITIALIZED | S'ee \\[the docs\\] for details.",
    ],
)
def test_pointer_offsets_rehydrate_authoritative_source_text(model_text: str):
    source = "# SDK\n\nStatus: SDK\\_UNINITIALIZED | See \\[the docs\\] for details.\n"
    target = "Status: SDK\\_UNINITIALIZED | See \\[the docs\\] for details."
    start = source.index(target)
    pointer = HydratedTextPointer(
        source_cluster_id="doc|p1_t0",
        start_char=start,
        end_char=start + len(target) - 1,
        verbatim_text=model_text,
    )

    hydrated = hydrate_pointer_from_offsets(
        pointer,
        {"doc|p1_t0": {"text": source}},
    )

    assert hydrated is not None
    assert hydrated.verbatim_text == target
    assert (
        pointer_source_validation_error(
            hydrated,
            {"doc|p1_t0": {"text": source}},
        )
        is None
    )


def test_pointer_repair_rehydrates_conflicting_valid_offsets():
    source = "Alpha clause."
    pointer = HydratedTextPointer(
        source_cluster_id="doc|p1_t0",
        start_char=0,
        end_char=len(source) - 1,
        verbatim_text="Beta clause.",
    )

    repaired = correct_and_validate_pointer(pointer, {"doc|p1_t0": {"text": source}})

    assert repaired is not None
    assert repaired.start_char == 0
    assert repaired.end_char == len(source) - 1
    assert repaired.verbatim_text == source


def test_pointer_repair_accepts_only_unique_exact_relocation():
    source = "Prefix. Alpha clause. Suffix."
    pointer = HydratedTextPointer(
        source_cluster_id="doc|p1_t0",
        start_char=999,
        end_char=1000,
        verbatim_text="Alpha clause.",
    )

    repaired = correct_and_validate_pointer(pointer, {"doc|p1_t0": {"text": source}})

    assert repaired is not None
    assert repaired.verbatim_text == "Alpha clause."
    assert source[repaired.start_char : repaired.end_char + 1] == repaired.verbatim_text


def test_llm_layer_pointer_schema_is_offset_only():
    schema_text = str(LLMCurrentLayerReview.model_json_schema())

    assert "LLMTextPointer" in schema_text
    assert "verbatim_text" not in schema_text
