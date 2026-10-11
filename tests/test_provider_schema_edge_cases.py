from __future__ import annotations

import pytest
from pydantic import ValidationError

from kg_doc_parser.workflow_ingest.models import (
    BoundaryCutpoint,
    BoundaryReviewDecision,
    LLMBoundaryProposalBatch,
    LLMLayerChildCandidate,
    LLMTextPointer,
)


def test_boundary_payload_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        BoundaryCutpoint(
            parent_node_id="parent",
            source_cluster_id="cluster",
            cut_offset=4,
            boundary_kind="paragraph",
            model_instruction="write outside the source span",
        )

    with pytest.raises(ValidationError):
        LLMBoundaryProposalBatch(cutpoints=[], hidden_decision="accept")


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BoundaryCutpoint(
            parent_node_id="parent",
            source_cluster_id="cluster",
            cut_offset="4",
            boundary_kind="paragraph",
        ),
        lambda: BoundaryCutpoint(
            parent_node_id="parent",
            source_cluster_id="cluster",
            cut_offset=4,
            boundary_kind="paragraph",
            confidence="0.9",
        ),
        lambda: BoundaryReviewDecision(
            parent_node_id="parent",
            source_cluster_id="cluster",
            cut_offset=4,
            decision="accept",
            coverage_ok=True,
        ),
        lambda: LLMTextPointer(source_cluster_id="cluster", start_char=True, end_char=4),
        lambda: LLMLayerChildCandidate(
            node_id="child",
            parent_node_id="parent",
            title="Title",
            node_type="PARAGRAPH",
            expandable="false",
        ),
    ],
)
def test_provider_payload_rejects_coercible_scalars(factory) -> None:
    with pytest.raises(ValidationError):
        factory()
