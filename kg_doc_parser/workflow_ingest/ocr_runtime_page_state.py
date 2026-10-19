"""Checkpoint and artifact validation for persisted OCR page workflows."""

from __future__ import annotations

import hashlib
import json

from kogwistar.engine_core.engine import GraphKnowledgeEngine

from .ocr_pipeline import OCRPageProcessingContext, OCRSourcePlan
from .ocr_runtime_contracts import (
    OCR_PAGE_STATE_VERSION,
    OCRCheckpointPatch,
    OCRWorkflowState,
)
from .ocr_runtime_state import int_list, json_pages, page_cursor, page_sources


def reused_pages_from_trace(
    conversation_engine: GraphKnowledgeEngine, run_id: str
) -> list[int]:
    """Rebuild the compatibility reuse summary from compact per-step deltas."""
    nodes = list(
        conversation_engine.read.get_nodes(
            where={
                "$and": [
                    {"entity_type": "workflow_step_exec"},
                    {"run_id": run_id},
                ]
            },
            limit=None,
        )
    )
    outcomes: dict[int, tuple[int, int, int, bool]] = {}
    for node in nodes:
        metadata = node.metadata or {}
        result_raw = metadata.get("result_json")
        if not isinstance(result_raw, str):
            continue
        try:
            result = json.loads(result_raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(result, dict):
            continue
        updates = result.get("state_update", [])
        if not isinstance(updates, list):
            continue
        step_sequence = metadata.get("step_seq", 0)
        if type(step_sequence) is not int:
            step_sequence = 0
        attempt = metadata.get("attempt", 0)
        if type(attempt) is not int:
            attempt = 0
        for update_index, update in enumerate(updates):
            if not isinstance(update, (list, tuple)) or len(update) != 2:
                continue
            operation, payload = update
            if not isinstance(payload, dict):
                continue
            page_cursor_value = payload.get("page_cursor")
            if (
                type(page_cursor_value) is int
                and page_cursor_value >= 1
                and "last_reused_page" in payload
            ):
                reused_page = payload.get("last_reused_page")
                outcomes[page_cursor_value] = (
                    step_sequence,
                    attempt,
                    update_index,
                    reused_page == page_cursor_value,
                )
            reused_page = payload.get("last_reused_page")
            if type(reused_page) is int and reused_page >= 1:
                outcomes[reused_page] = (step_sequence, attempt, update_index, True)
            if operation == "a":
                legacy_page = payload.get("reused_pages")
                if type(legacy_page) is int and legacy_page >= 1:
                    outcomes[legacy_page] = (step_sequence, attempt, update_index, True)
    return [
        page_number
        for page_number, (_step, _attempt, _update, was_reused) in sorted(
            outcomes.items(), key=lambda item: item[1][:3]
        )
        if was_reused
    ]


def page_workflow_initial_state(source_plan: OCRSourcePlan) -> OCRWorkflowState:
    return {
        "ocr_page_state_version": OCR_PAGE_STATE_VERSION,
        "source_fingerprint": source_plan.input_fingerprint,
        "legacy_dir": str(source_plan.legacy_dir),
        "page_cursor": 0,
    }


def reconcile_page_checkpoint(
    *,
    state: OCRWorkflowState,
    context: OCRPageProcessingContext,
    source_plan: OCRSourcePlan,
    start_index: int = 0,
) -> OCRCheckpointPatch | None:
    """Return a rewind patch at the first page without an accepted artifact."""
    cursor = page_cursor(state)
    sources = (
        page_sources(state) if "page_sources" in state else source_plan.page_sources
    )
    completed = int_list(state, "completed_pages") if "completed_pages" in state else []
    reused = int_list(state, "reused_pages") if "reused_pages" in state else []
    if start_index < 0 or start_index > cursor:
        raise ValueError("OCR checkpoint reconciliation start index is out of range")
    for index in range(start_index, cursor):
        page_number, image_path = sources[index]
        image_digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        page_path = source_plan.legacy_dir / f"page_{page_number}.json"
        if context.state_store.reconcile_orphan_artifact(
            document_id=context.document_id,
            page_number=page_number,
            content_hash=image_digest,
            artifact_path=page_path,
            config_fingerprints=context.configuration_fingerprints,
        ):
            continue
        if context.state_store.should_skip_page(
            document_id=context.document_id,
            page_number=page_number,
            stage="ocr",
            content_hash=image_digest,
            artifact_path=page_path,
            config_fingerprint=context.configuration_fingerprints,
        ):
            continue
        repair: OCRCheckpointPatch = {"page_cursor": index}
        if "completed_pages" in state:
            repair["completed_pages"] = completed[:index]
        if "reused_pages" in state:
            repair["reused_pages"] = [
                number for number in reused[:index] if number in completed[:index]
            ]
        if "raw_pages" in state:
            repair["raw_pages"] = json_pages(state, source_plan=source_plan)[:index]
        return repair
    return None


def validate_root_page_result(
    *,
    page_result: OCRWorkflowState,
    context: OCRPageProcessingContext,
    source_plan: OCRSourcePlan,
) -> None:
    """Fail closed if a terminal page child no longer owns valid artifacts."""
    sources = source_plan.page_sources
    cursor = page_cursor(page_result)
    expected_pages = [page_number for page_number, _image_path in sources]
    legacy_completed = (
        int_list(page_result, "completed_pages")
        if "completed_pages" in page_result
        else []
    )
    completed = legacy_completed or expected_pages[:cursor]
    if cursor != len(sources) or completed != expected_pages:
        raise RuntimeError("OCR root page child completed with an incomplete page checkpoint")
    if reconcile_page_checkpoint(
        state=page_result,
        context=context,
        source_plan=source_plan,
    ) is not None:
        raise RuntimeError(
            "OCR root page checkpoint is no longer backed by accepted artifacts; "
            "resume must retry the page child"
        )


def page_result_needs_recovery(
    *,
    page_result: OCRWorkflowState,
    context: OCRPageProcessingContext,
    source_plan: OCRSourcePlan,
) -> bool:
    sources = source_plan.page_sources
    expected_pages = [page_number for page_number, _image_path in sources]
    legacy_completed = (
        int_list(page_result, "completed_pages")
        if "completed_pages" in page_result
        else []
    )
    completed = legacy_completed or expected_pages[: page_cursor(page_result)]
    if page_cursor(page_result) != len(sources) or completed != expected_pages:
        return True
    return reconcile_page_checkpoint(
        state=page_result,
        context=context,
        source_plan=source_plan,
    ) is not None


def root_page_artifacts_need_recovery(
    *,
    context: OCRPageProcessingContext,
    source_plan: OCRSourcePlan,
) -> bool:
    """Detect a broken accepted page set before resuming a terminal root run."""
    for page_number, image_path in source_plan.page_sources:
        image_digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        page_path = source_plan.legacy_dir / f"page_{page_number}.json"
        context.state_store.reconcile_orphan_artifact(
            document_id=context.document_id,
            page_number=page_number,
            content_hash=image_digest,
            artifact_path=page_path,
            config_fingerprints=context.configuration_fingerprints,
        )
        if not context.state_store.should_skip_page(
            document_id=context.document_id,
            page_number=page_number,
            stage="ocr",
            content_hash=image_digest,
            artifact_path=page_path,
            config_fingerprint=context.configuration_fingerprints,
        ):
            return True
    return False


def validate_existing_page_identities(
    *,
    context: OCRPageProcessingContext,
    source_plan: OCRSourcePlan,
) -> None:
    """Reject provider changes before root recovery replays any page."""
    expected = set(context.configuration_fingerprints)
    for page_number, _image_path in source_plan.page_sources:
        state = context.state_store.get_page_state(
            document_id=context.document_id,
            page_number=page_number,
            stage="ocr",
        )
        if (
            state is not None
            and state.config_fingerprint is not None
            and state.config_fingerprint not in expected
        ):
            raise RuntimeError(
                "OCR root checkpoint identity mismatch; refusing to mix persisted "
                f"OCR configuration for page {page_number}"
            )
