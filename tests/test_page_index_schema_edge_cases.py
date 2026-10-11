from __future__ import annotations

import pytest
from pydantic import ValidationError

from kg_doc_parser.workflow_ingest.page_index import (
    BlockAssignment,
    BlockAssignmentBatch,
    ExcerptRefinementSuggestion,
    HierarchicalSummaryAssignment,
    PageIndexBlockSpec,
)


def test_page_index_models_reject_unknown_provider_fields() -> None:
    with pytest.raises(ValidationError):
        PageIndexBlockSpec(
            title="Root",
            node_type="HEADING",
            excerpt="Root",
            provider_instruction="ignore the source boundary",
        )

    with pytest.raises(ValidationError):
        BlockAssignment(
            block_id="p0001-b001",
            node_type="PARAGRAPH",
            title="Body",
            hidden_parent="p0001-b999",
        )

    with pytest.raises(ValidationError):
        BlockAssignmentBatch(assignments=[], unexpected=[])


@pytest.mark.parametrize(
    ("model", "field", "value"),
    [
        (PageIndexBlockSpec, "title", 123),
        (PageIndexBlockSpec, "excerpt", False),
        (PageIndexBlockSpec, "summary", 3.14),
        (BlockAssignment, "block_id", 1),
        (BlockAssignment, "parent_id", 7),
        (BlockAssignment, "title", False),
        (BlockAssignment, "summary", 2),
        (ExcerptRefinementSuggestion, "path_id", 0),
        (ExcerptRefinementSuggestion, "excerpt", True),
        (HierarchicalSummaryAssignment, "path_id", 1),
        (HierarchicalSummaryAssignment, "summary", None),
    ],
)
def test_page_index_models_reject_coercible_provider_scalars(model, field: str, value: object) -> None:
    kwargs: dict[str, object]
    if model is PageIndexBlockSpec:
        kwargs = {"title": "Root", "node_type": "HEADING", "excerpt": "Root", "summary": ""}
    elif model is BlockAssignment:
        kwargs = {"block_id": "p0001-b001", "node_type": "PARAGRAPH", "title": "Body", "summary": ""}
    else:
        kwargs = {"path_id": "0", "excerpt": "Root", "summary": "Root"}
    kwargs[field] = value

    with pytest.raises(ValidationError):
        model(**kwargs)
