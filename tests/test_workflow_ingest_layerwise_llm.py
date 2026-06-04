from __future__ import annotations

from typing import Any

import pytest

from kg_doc_parser.workflow_ingest import ProviderEndpointConfig, WorkflowProviderSettings
from kg_doc_parser.workflow_ingest.layerwise_llm import build_layerwise_llm_callbacks
from kg_doc_parser.workflow_ingest.models import (
    CurrentLayerContext,
    CurrentLayerResult,
    LayerChildCandidate,
    ParseSessionState,
)
from kg_doc_parser.workflow_ingest.semantics import HydratedTextPointer, SemanticNode


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

    def with_structured_output(self, schema, include_raw: bool = True):
        return _FakeStructuredInvoker(self)


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
    assert result.reasoning_history[-1]["proposal_source"] == "llm"
    assert layer_events[-1]["stage"] == "workflow_layered_proposal_result"
    assert layer_events[-1]["proposal_source"] == "llm"
    prompt_body = fake_model.messages[1][1].lower()
    assert "coarser layerwise breakdown" in prompt_body
    assert "do not recombine separated verbatim fragments" in prompt_body


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
    assert result.reasoning_history[-1]["proposal_source"] == "fallback"
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
                        "title": "Demo Doc",
                        "node_type": "TEXT_FLOW",
                        "total_content_pointers": [
                            {
                                "source_cluster_id": "cluster-1",
                                "start_char": 0,
                                "end_char": 24,
                                "verbatim_text": "Alpha clause. Beta clause.",
                            }
                        ],
                        "expandable": False,
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
