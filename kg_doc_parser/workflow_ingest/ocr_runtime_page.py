"""Persisted page-level OCR workflow orchestration."""

from __future__ import annotations

import hashlib
from pathlib import Path

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.runtime import MappingStepResolver
from kogwistar.runtime.models import RunFailure, StateUpdate, StepRunResult
from kogwistar.runtime.runtime import StepContext, WorkflowRuntime

from .design import DEFAULT_OCR_PAGE_WORKFLOW_ID, ensure_ocr_page_workflow_design
from .ocr_pipeline import (
    OCRPageJSON,
    OCRPageProcessingContext,
    OCRSourcePlan,
    _process_ocr_page,
    _sync_progress_from_state,
)
from .ocr_runtime import (
    _checkpoint_repair_cursor,
    _page_workflow_initial_state,
    _success,
    _typed_state,
)
from .ocr_runtime_page_state import (
    reconcile_page_checkpoint as _reconcile_page_checkpoint,
)
from .ocr_runtime_page_state import (
    reused_pages_from_trace as _reused_pages_from_trace,
)
from .ocr_runtime_state import (
    json_pages,
    validate_page_resume_identity,
)
from .ocr_runtime_state import (
    page_cursor as _page_cursor,
)


def build_page_resolver(
    *,
    context: OCRPageProcessingContext,
    source_plan: OCRSourcePlan,
    progress_path: Path,
) -> MappingStepResolver:
    """Build the page child resolver for both standalone and nested runs."""
    resolver = MappingStepResolver()
    verified_prefix_cursor = 0
    resolver.set_state_schema(
        {
            "ocr_page_state_version": "u",
            "document_id": "u",
            "source_fingerprint": "u",
            "ocr_configuration_fingerprints": "u",
            "page_sources": "u",
            "legacy_dir": "u",
            "page_cursor": "u",
            "raw_pages": "u",
            "completed_pages": "u",
            "reused_pages": "u",
            "last_failed_page": "u",
            "last_reused_page": "u",
        }
    )

    @resolver.register("start")
    def _start(ctx: StepContext) -> StepRunResult:
        validate_page_resume_identity(
            _typed_state(ctx.state_view),
            document_id=context.document_id,
            source_fingerprint=source_plan.input_fingerprint,
            configuration_fingerprints=context.configuration_fingerprints,
        )
        return _success("process_ocr_page")

    @resolver.register("process_ocr_page")
    def _process(ctx: StepContext) -> StepRunResult:
        nonlocal verified_prefix_cursor
        state = _typed_state(ctx.state_view)
        repair = _reconcile_page_checkpoint(
            state=state,
            context=context,
            source_plan=source_plan,
            start_index=verified_prefix_cursor,
        )
        if repair is not None:
            verified_prefix_cursor = _checkpoint_repair_cursor(repair)
            repair_update: StateUpdate = ("u", dict(repair))
            return _success("process_ocr_page", state_update=[repair_update])
        sources = source_plan.page_sources
        cursor = _page_cursor(state)
        verified_prefix_cursor = cursor
        if cursor >= len(sources):
            return _success("finalize_ocr_pages")
        page_number, image_path = sources[cursor]
        outcome = _process_ocr_page(
            context=context,
            done_count=cursor + 1,
            page_number=page_number,
            image_path=image_path,
            image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
            page_json_path=source_plan.legacy_dir / f"page_{page_number}.json",
            page_image_path=source_plan.legacy_dir
            / f"page_{page_number}{image_path.suffix or '.png'}",
        )
        if outcome.status == "failed":
            return RunFailure(
                conversation_node_id=None,
                state_update=[("u", {"last_failed_page": page_number})],
                errors=[f"OCR page {page_number} failed after all candidates"],
            )
        if outcome.page_number != page_number or outcome.page is None:
            raise RuntimeError("OCR page processing produced an inconsistent outcome")

        updates: list[StateUpdate] = [
            (
                "u",
                {
                    "page_cursor": cursor + 1,
                    "last_reused_page": (
                        page_number if outcome.status == "reused" else None
                    ),
                },
            ),
        ]
        if "completed_pages" in state or "reused_pages" in state:
            updates.append(("u", {"completed_pages": [], "reused_pages": []}))
        if state.get("raw_pages") is not None:
            updates.append(("u", {"raw_pages": None}))
        verified_prefix_cursor = cursor + 1
        return _success("process_ocr_page", state_update=updates)

    @resolver.register("finalize_ocr_pages")
    def _finalize(ctx: StepContext) -> StepRunResult:
        nonlocal verified_prefix_cursor
        state = _typed_state(ctx.state_view)
        repair = _reconcile_page_checkpoint(
            state=state,
            context=context,
            source_plan=source_plan,
        )
        if repair is not None:
            verified_prefix_cursor = _checkpoint_repair_cursor(repair)
            return _success("process_ocr_page", state_update=[("u", dict(repair))])
        context.state_store.refresh_document_completion(document_id=context.document_id)
        _sync_progress_from_state(
            state_store=context.state_store,
            document_id=context.document_id,
            title=context.title,
            source_kind=context.source_kind,
            total_pages=context.total_pages,
            progress_path=progress_path,
        )
        return _success("end")

    @resolver.register("end")
    def _end(ctx: StepContext) -> StepRunResult:
        del ctx
        return _success(None)

    return resolver


def run_ocr_page_workflow(
    *,
    workflow_engine: GraphKnowledgeEngine,
    conversation_engine: GraphKnowledgeEngine,
    context: OCRPageProcessingContext,
    source_plan: OCRSourcePlan,
    progress_path: Path,
    run_id: str,
    resume_from_checkpoint: bool = False,
) -> tuple[list[OCRPageJSON], list[int], list[int], str]:
    """Process OCR pages as persisted Kogwistar steps."""
    ensure_ocr_page_workflow_design(workflow_engine)
    resolver = build_page_resolver(
        context=context,
        source_plan=source_plan,
        progress_path=progress_path,
    )
    runtime = WorkflowRuntime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=resolver.resolve,
        predicate_registry={},
        trace=True,
    )
    initial_state = _page_workflow_initial_state(source_plan)
    initial_state.update(
        {
            "document_id": context.document_id,
            "ocr_configuration_fingerprints": list(context.configuration_fingerprints),
        }
    )
    result = (
        runtime.resume_from_latest_checkpoint(
            run_id=run_id,
            workflow_id=DEFAULT_OCR_PAGE_WORKFLOW_ID,
            conversation_id=f"ocr:{context.document_id}",
            turn_node_id=f"ocr:{context.document_id}:pages",
        )
        if resume_from_checkpoint
        else runtime.run(
            workflow_id=DEFAULT_OCR_PAGE_WORKFLOW_ID,
            conversation_id=f"ocr:{context.document_id}",
            turn_node_id=f"ocr:{context.document_id}:pages",
            initial_state=dict(initial_state),
            run_id=run_id,
        )
    )
    if result.status != "succeeded":
        raise RuntimeError(f"OCR page workflow {result.status}: {result.errors}")
    final_state = _typed_state(result.final_state)
    return (
        json_pages(final_state, source_plan=source_plan),
        [page_number for page_number, _path in source_plan.page_sources],
        _reused_pages_from_trace(conversation_engine, result.run_id),
        result.run_id,
    )
