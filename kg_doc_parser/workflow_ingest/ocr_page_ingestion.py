"""Execute OCR pages and normalize their accepted results for ingest."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from kogwistar.engine_core.engine import GraphKnowledgeEngine

from .adapters import OCRPageJSON
from .document_models import SummaryPolicy
from .ocr_finalization import _finalize_ocr_workflow_artifacts
from .ocr_inputs import sha256_file
from .ocr_page_processing import _emit_ocr_event, _process_ocr_page
from .ocr_pipeline_contracts import (
    GroundedSummaryGenerator,
    OCRImagePayload,
    OCRRunner,
    OCRWorkflowArtifacts,
    PDFRasterizer,
    SummaryGenerator,
)
from .ocr_preparation import OCRPagePreparation, build_ocr_page_preparation
from .ocr_progress import sync_progress_from_state
from .ocr_quality_policy import OCRQualityPolicy
from .probe import WorkflowProbe
from .providers import WorkflowProviderSettings


def prepare_ocr_workflow_input(
    *,
    document_id: str,
    title: str,
    output_dir: str | Path,
    image_payloads: Sequence[OCRImagePayload] | None = None,
    pdf_path: str | Path | None = None,
    provider_settings: WorkflowProviderSettings | None = None,
    ocr_runner: OCRRunner | None = None,
    pdf_rasterizer: PDFRasterizer | None = None,
    ocr_candidate_models: Sequence[str] | None = None,
    probe: WorkflowProbe | None = None,
    restore_grouping_checkpoint: bool = True,
    persist_grouping_checkpoint: bool = True,
    workflow_engine: GraphKnowledgeEngine | None = None,
    conversation_engine: GraphKnowledgeEngine | None = None,
    ocr_runtime_run_id: str | None = None,
    resume_from_checkpoint: bool = False,
    grouping_policy_version: str = "baseline-v1",
    summary_policy: SummaryPolicy | None = None,
    summary_generator: SummaryGenerator | None = None,
    grounded_summary_generator: GroundedSummaryGenerator | None = None,
    quality_policy: OCRQualityPolicy | None = None,
) -> OCRWorkflowArtifacts:
    """Run page OCR and normalize its validated results into ingest input."""
    if bool(image_payloads) == bool(pdf_path):
        raise ValueError("provide exactly one of image_payloads or pdf_path")

    effective_settings = provider_settings or WorkflowProviderSettings.from_env()
    _emit_ocr_event(
        probe,
        "ocr.prepare_started",
        document_id=document_id,
        title=title,
        output_dir=str(output_dir),
        source_kind="pdf" if pdf_path is not None else "image",
    )
    preparation: OCRPagePreparation = build_ocr_page_preparation(
        document_id=document_id,
        title=title,
        output_dir=Path(output_dir),
        image_payloads=image_payloads,
        pdf_path=Path(pdf_path) if pdf_path is not None else None,
        provider_settings=effective_settings,
        ocr_runner=ocr_runner,
        pdf_rasterizer=pdf_rasterizer,
        ocr_candidate_models=ocr_candidate_models,
        probe=probe,
        quality_policy=quality_policy,
    )
    source_plan = preparation.source_plan
    state_store = preparation.state_store
    page_context = preparation.page_context
    _emit_ocr_event(
        probe,
        "ocr.state_ready",
        document_id=document_id,
        state_db_path=str(preparation.state_db_path),
        progress_path=str(preparation.progress_path),
        rendered_dir=str(source_plan.rendered_dir),
        legacy_dir=str(source_plan.legacy_dir),
    )

    raw_pages: list[OCRPageJSON] = []
    completed_pages: list[int] = []
    reused_pages: list[int] = []
    page_failures: list[int] = []

    if workflow_engine is not None and conversation_engine is not None:
        from .ocr_runtime import run_ocr_page_workflow

        raw_pages, completed_pages, reused_pages, _runtime_run_id = (
            run_ocr_page_workflow(
                workflow_engine=workflow_engine,
                conversation_engine=conversation_engine,
                context=page_context,
                source_plan=source_plan,
                progress_path=preparation.progress_path,
                run_id=ocr_runtime_run_id or f"ocr-pages|{document_id}",
                resume_from_checkpoint=resume_from_checkpoint,
            )
        )
    else:
        for done_count, (page_number, image_path) in enumerate(
            source_plan.page_sources, start=1
        ):
            image_path = Path(image_path)
            image_sha256 = sha256_file(image_path)
            page_json_path = source_plan.legacy_dir / f"page_{page_number}.json"
            page_image_path = (
                source_plan.legacy_dir
                / f"page_{page_number}{image_path.suffix or '.png'}"
            )
            outcome = _process_ocr_page(
                context=page_context,
                done_count=done_count,
                page_number=page_number,
                image_path=image_path,
                image_sha256=image_sha256,
                page_json_path=page_json_path,
                page_image_path=page_image_path,
            )
            if outcome.status == "failed":
                page_failures.append(outcome.page_number)
                continue
            if outcome.page is None:
                raise RuntimeError("accepted OCR page outcome has no page payload")
            raw_pages.append(outcome.page)
            completed_pages.append(outcome.page_number)
            if outcome.status == "reused":
                reused_pages.append(outcome.page_number)

    state_store.refresh_document_completion(document_id=document_id)
    sync_progress_from_state(
        state_store=state_store,
        document_id=document_id,
        title=title,
        source_kind=source_plan.source_kind,
        total_pages=len(source_plan.page_sources),
        progress_path=preparation.progress_path,
    )
    _emit_ocr_event(
        probe,
        "ocr.prepare_finished",
        document_id=document_id,
        status="failed" if page_failures else "succeeded",
        completed_pages=completed_pages,
        reused_pages=reused_pages,
        failed_pages=page_failures,
        state_db_path=str(preparation.state_db_path),
    )
    if page_failures:
        raise RuntimeError(
            f"OCR preparation incomplete for pages {page_failures}; "
            "rerun will retry failed pages."
        )

    return _finalize_ocr_workflow_artifacts(
        document_id=document_id,
        title=title,
        raw_pages=raw_pages,
        completed_pages=completed_pages,
        reused_pages=reused_pages,
        rendered_dir=source_plan.rendered_dir,
        legacy_dir=source_plan.legacy_dir,
        state_db_path=preparation.state_db_path,
        progress_path=preparation.progress_path,
        summary_path=preparation.summary_path,
        provider_settings=effective_settings,
        state_store=state_store,
        restore_grouping_checkpoint=restore_grouping_checkpoint,
        persist_grouping_checkpoint=persist_grouping_checkpoint,
        grouping_policy_version=grouping_policy_version,
        summary_policy=summary_policy,
        summary_generator=summary_generator,
        grounded_summary_generator=grounded_summary_generator,
    )
