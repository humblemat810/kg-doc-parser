"""Shared typed fixtures for layerwise workflow tests."""

from __future__ import annotations

from typing import Any

from kg_doc_parser.workflow_ingest import (
    ProviderEndpointConfig,
    WorkflowProviderSettings,
)
from kg_doc_parser.workflow_ingest.models import CurrentLayerContext, ParseSessionState
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
                        "units": [
                            {"id": "cluster-1", "text": "Alpha clause. Beta clause."}
                        ],
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
