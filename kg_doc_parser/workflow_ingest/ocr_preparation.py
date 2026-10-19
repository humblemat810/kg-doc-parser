"""Build validated source and provider context for OCR page execution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..models import OCRClusterResponse
from .ocr_page_processing import (
    OCRPageProcessingContext,
    _emit_ocr_event,
    _ocr_configuration_fingerprint,
    _ocr_page_metadata,
)
from .ocr_page_runner import _ocr_direct_pdf_image_regions, _run_live_ocr_page
from .ocr_pipeline_contracts import (
    OCRImagePayload,
    OCRRunner,
    PDFRasterizer,
)
from .ocr_quality_policy import OCRQualityPolicy
from .ocr_source_plan import OCRSourcePlan, _resolve_ocr_source_plan
from .ocr_state_store import OCRWorkflowStateStore
from .probe import WorkflowProbe
from .providers import WorkflowProviderSettings


@dataclass(slots=True)
class OCRPagePreparation:
    """Durable inputs and injected dependencies for page execution."""

    source_plan: OCRSourcePlan
    state_store: OCRWorkflowStateStore
    page_context: OCRPageProcessingContext
    state_db_path: Path
    progress_path: Path
    summary_path: Path


def build_ocr_page_preparation(
    *,
    document_id: str,
    title: str,
    output_dir: Path,
    image_payloads: Sequence[OCRImagePayload] | None,
    pdf_path: Path | None,
    provider_settings: WorkflowProviderSettings,
    ocr_runner: OCRRunner | None,
    pdf_rasterizer: PDFRasterizer | None,
    ocr_candidate_models: Sequence[str] | None,
    probe: WorkflowProbe | None,
    quality_policy: OCRQualityPolicy | None = None,
) -> OCRPagePreparation:
    """Build page-run context without invoking an OCR provider."""

    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "ocr-progress.json"
    summary_path = output_dir / "ocr-summary.json"
    state_db_path = output_dir / "ocr-state.sqlite"
    source_plan = _resolve_ocr_source_plan(
        document_id=document_id,
        output_dir=output_dir,
        image_payloads=image_payloads,
        pdf_path=pdf_path,
        pdf_rasterizer=pdf_rasterizer,
    )
    state_store = OCRWorkflowStateStore.open_or_rebuild(
        db_path=state_db_path,
        document_id=document_id,
        title=title,
        source_kind=source_plan.source_kind,
        total_pages=len(source_plan.page_sources),
        input_fingerprint=source_plan.input_fingerprint,
        rendered_dir=source_plan.rendered_dir,
        legacy_dir=source_plan.legacy_dir,
        progress_path=progress_path,
        render_configuration_fingerprint=source_plan.render_configuration_fingerprint,
    )
    candidate_models = list(ocr_candidate_models or [provider_settings.ocr.model])
    retry_feedback_by_page: dict[int, tuple[str, ...]] = {}
    if ocr_runner is None:

        def instrumented_ocr_runner(
            image_path: Path,
            page_number: int,
            settings: WorkflowProviderSettings,
        ) -> OCRClusterResponse:
            return _run_live_ocr_page(
                image_path,
                page_number,
                settings,
                provider_diagnostics_sink=(
                    lambda diagnostics: (
                        _emit_ocr_event(probe, "ocr.provider_call", **diagnostics)
                        if probe is not None
                        else None
                    )
                ),
                retry_feedback=retry_feedback_by_page.get(page_number, ()),
            )

        effective_runner: OCRRunner = instrumented_ocr_runner
    else:
        effective_runner = ocr_runner

    native_pdf_path = source_plan.pdf_source
    if (
        native_pdf_path is not None
        and provider_settings.ocr.pdf_text_mode != "disabled"
    ):
        configured_runner = effective_runner

        def pdf_aware_runner(
            image_path: Path,
            page_number: int,
            settings: WorkflowProviderSettings,
        ) -> OCRClusterResponse:
            from .pdf_native import (
                direct_pdf_image_regions,
                extract_native_pdf_page,
                merge_native_and_ocr_page,
                pdf_page_has_embedded_images,
            )

            native = extract_native_pdf_page(native_pdf_path, page_number)
            if settings.ocr.pdf_text_mode == "native_only":
                has_images = pdf_page_has_embedded_images(native_pdf_path, page_number)
                if native is None or has_images is not False:
                    reason = (
                        "native PDF text is unavailable"
                        if native is None
                        else "native PDF image inventory is incomplete"
                    )
                    raise ValueError(
                        f"{reason} for page {page_number}; "
                        "native_only mode refuses an ungrounded fallback"
                    )
                return native

            image_regions = direct_pdf_image_regions(native_pdf_path, page_number)
            if image_regions is not None:
                if image_regions:
                    region_ocr = _ocr_direct_pdf_image_regions(
                        pdf_path=native_pdf_path,
                        image_regions=image_regions,
                        source_image_path=image_path,
                        page_number=page_number,
                        runner=configured_runner,
                        settings=settings,
                    )
                    if region_ocr is not None:
                        return (
                            merge_native_and_ocr_page(native, region_ocr)
                            if native is not None
                            else region_ocr
                        )
                elif native is not None:
                    return native

            ocr = configured_runner(image_path, page_number, settings)
            if native is not None:
                return merge_native_and_ocr_page(native, ocr)
            return ocr

        effective_runner = pdf_aware_runner

    effective_quality_policy = quality_policy or OCRQualityPolicy()
    page_context = OCRPageProcessingContext(
        document_id=document_id,
        title=title,
        source_kind=source_plan.source_kind,
        total_pages=len(source_plan.page_sources),
        input_fingerprint=source_plan.input_fingerprint,
        candidate_models=candidate_models,
        provider_settings=provider_settings,
        ocr_page_runner=effective_runner,
        state_store=state_store,
        probe=probe,
        configuration_fingerprints=tuple(
            _ocr_configuration_fingerprint(
                provider_settings.model_copy(
                    update={
                        "ocr": provider_settings.ocr.model_copy(update={"model": model})
                    }
                ),
                candidate_models=candidate_models,
                quality_policy=effective_quality_policy,
            )
            for model in candidate_models
        )
        + (
            _ocr_page_metadata(
                provider_settings,
                quality_policy=effective_quality_policy,
            )["ocr_configuration_fingerprint"],
        ),
        retry_feedback_by_page=retry_feedback_by_page,
        render_configuration_fingerprint=source_plan.render_configuration_fingerprint,
        quality_policy=effective_quality_policy,
    )
    return OCRPagePreparation(
        source_plan=source_plan,
        state_store=state_store,
        page_context=page_context,
        state_db_path=state_db_path,
        progress_path=progress_path,
        summary_path=summary_path,
    )
