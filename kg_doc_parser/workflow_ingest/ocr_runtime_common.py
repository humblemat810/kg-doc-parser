"""Shared state/result helpers for persisted OCR workflow steps."""

from __future__ import annotations

from kogwistar.runtime.models import RunSuccess, StateUpdate

from .document_models import DocumentTree
from .ocr_runtime_contracts import OCRWorkflowState


def workflow_success(
    next_step: str | None,
    *,
    state_update: list[StateUpdate] | None = None,
) -> RunSuccess:
    """Return the standard Kogwistar success result and optional route."""
    return RunSuccess(
        conversation_node_id=None,
        state_update=state_update or [],
        _route_next=[] if next_step is None else [next_step],
    )


def document_tree_from_state(state: OCRWorkflowState) -> DocumentTree:
    """Decode the validated tree payload from persisted workflow state."""
    payload = state.get("tree")
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) for key in payload
    ):
        raise TypeError("OCR grouping runtime state has no tree")
    return DocumentTree.model_validate(payload)
