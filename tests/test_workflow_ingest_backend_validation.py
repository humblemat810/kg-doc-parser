from __future__ import annotations

import pydantic_extension.model_slicing.mixin as model_slicing_mixin
import pytest

from kg_doc_parser.workflow_ingest.model_validation import validate_backend_payload
from kg_doc_parser.workflow_ingest.models import (
    CurrentLayerResult,
    LayerChildCandidate,
    WorkflowIngestInput,
)

pytestmark = [pytest.mark.workflow, pytest.mark.ci]


def test_internal_backend_validation_avoids_stack_inference_and_preserves_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "children": [
            {
                "node_id": "child",
                "parent_node_id": "root",
                "title": "A section",
                "node_type": "TEXT_FLOW",
                "total_content_pointers": [],
                "metadata": {"internal": True},
                "child_candidates": [],
            }
        ],
        "metadata": {"origin": "workflow_state"},
    }

    def unexpected_stack_inference(**_: object) -> str:
        raise AssertionError("backend state should not infer its mode from the stack")

    monkeypatch.setattr(
        model_slicing_mixin, "_infer_mode_from_stack", unexpected_stack_inference
    )
    validated = validate_backend_payload(CurrentLayerResult, payload)
    assert validated.children[0].metadata == {"internal": True}
    assert validated.children[0].child_candidates == []
    assert validated.metadata == {"origin": "workflow_state"}
    assert isinstance(validated.children[0], LayerChildCandidate)


def test_backend_validation_handles_nested_normalized_input_without_stack_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = WorkflowIngestInput.from_text(
        document_id="backend-input",
        text="persisted normalized source",
    ).model_dump(field_mode="backend", dump_format="json")

    def unexpected_stack_inference(**_: object) -> str:
        raise AssertionError("backend state should not infer its mode from the stack")

    monkeypatch.setattr(
        model_slicing_mixin, "_infer_mode_from_stack", unexpected_stack_inference
    )
    validated = validate_backend_payload(WorkflowIngestInput, payload)
    assert validated.collections[0].pages[0].units[0].text == (
        "persisted normalized source"
    )
