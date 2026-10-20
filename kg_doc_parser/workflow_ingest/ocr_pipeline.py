"""Public OCR workflow facade and compatibility re-exports.

Page planning, page execution, persistence, rendering, quality checks, and
finalization live in focused modules. This module keeps the stable OCR ingest
entrypoints available while composing them with the persisted root workflow.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from kogwistar.engine_core.engine import GraphKnowledgeEngine

from .adapters import OCRPageJSON  # noqa: F401 - compatibility re-export
from .document_grouping import (
    GroupProposal,
    GroupProposalGenerator,
)
from .document_models import (
    AdaptiveGroupingPolicy,
    SummaryPolicy,
)
from .models import WorkflowExportBundle
from .ocr_file_io import (
    write_json_atomic as _write_json_atomic,  # noqa: F401 - compatibility import
)
from .ocr_finalization import (
    _finalize_ocr_workflow_artifacts,  # noqa: F401 - compatibility import
    _write_document_tree,  # noqa: F401 - runtime compatibility import
    _write_summary,  # noqa: F401 - runtime compatibility import
)
from .ocr_generators import (
    GroundedChatGroupProposalGenerator,  # noqa: F401 - compatibility re-export
    GroundedChatSummaryGenerator,  # noqa: F401 - compatibility re-export
    _effective_summary_policy,  # noqa: F401 - compatibility import
    _provider_endpoint_configuration_identity,  # noqa: F401 - compatibility import
    _summary_generator_identity,  # noqa: F401 - runtime compatibility import
    build_production_grounded_summary_generator,
)
from .ocr_inputs import sha256_file as _sha256_file  # noqa: F401 - compatibility import
from .ocr_page_processing import (
    OCRPageOutcome,  # noqa: F401 - compatibility re-export
    OCRPageProcessingContext,  # noqa: F401 - runtime compatibility import
    _emit_ocr_event,  # noqa: F401 - compatibility import
    _ocr_configuration_fingerprint,  # noqa: F401 - compatibility import
    _ocr_page_metadata,  # noqa: F401 - compatibility import
    _process_ocr_page,  # noqa: F401 - compatibility import
)
from .ocr_page_runner import (
    _ocr_direct_pdf_image_regions,  # noqa: F401 - compatibility import
    _run_live_ocr_page,  # noqa: F401 - compatibility import
)
from .ocr_pdf_rendering import (
    _pdf_render_manifest_payload,  # noqa: F401 - compatibility re-export
    _render_pdf_to_images,  # noqa: F401 - compatibility re-export
    _render_pdf_with_identity,  # noqa: F401 - compatibility re-export
)
from .ocr_pipeline_contracts import (
    GroundedSummaryGenerator,
    OCRImagePayload,
    OCRRunner,
    OCRWorkflowArtifacts,
    OCRWorkflowProgress,  # noqa: F401 - compatibility re-export
    PDFRasterizer,
    SummaryGenerator,
)
from .ocr_progress import (
    sync_progress_from_state as _sync_progress_from_state,  # noqa: F401 - compatibility import
)
from .ocr_progress import (
    write_progress as _write_progress,  # noqa: F401 - compatibility import
)
from .ocr_quality_policy import (
    OCRQualityDiagnostic,  # noqa: F401 - compatibility re-export
    OCRQualityPolicy,
)
from .ocr_quality_policy import (
    evaluate_ocr_quality as _evaluate_ocr_quality,  # noqa: F401 - compatibility re-export
)
from .ocr_source_plan import (
    OCRSourcePlan,  # noqa: F401 - runtime compatibility import
    _compute_input_fingerprint,  # noqa: F401 - compatibility re-export
    _materialize_image_payloads,  # noqa: F401 - compatibility re-export
    _resolve_ocr_source_plan,  # noqa: F401 - compatibility import
)
from .ocr_state_store import (
    OCRPageStateSnapshot,  # noqa: F401 - compatibility re-export
    OCRWorkflowStateStore,  # noqa: F401 - compatibility re-export
)
from .ocr_summary_workflow import (
    _generate_tree_summaries,  # noqa: F401 - runtime compatibility import
)
from .probe import WorkflowProbe
from .providers import (
    WorkflowProviderSettings,
)
from .runtime_contracts import WorkflowRuntimeDeps
from .serialization import JsonValue
from .service import _RunCompat

JsonObject = dict[str, JsonValue]


from .ocr_page_ingestion import prepare_ocr_workflow_input  # noqa: F401 - public facade
from .ocr_preparation import (
    OCRPagePreparation,  # noqa: F401 - compatibility re-export
)
from .ocr_preparation import (
    build_ocr_page_preparation as _build_ocr_page_preparation,
)


def run_ocr_ingest_workflow(
    *,
    document_id: str,
    title: str,
    output_dir: str | Path,
    workflow_engine: GraphKnowledgeEngine,
    conversation_engine: GraphKnowledgeEngine,
    knowledge_engine: GraphKnowledgeEngine | None = None,
    image_payloads: Sequence[OCRImagePayload] | None = None,
    pdf_path: str | Path | None = None,
    provider_settings: WorkflowProviderSettings | None = None,
    ocr_runner: OCRRunner | None = None,
    pdf_rasterizer: PDFRasterizer | None = None,
    ocr_candidate_models: Sequence[str] | None = None,
    deps: WorkflowRuntimeDeps | None = None,
    probe: WorkflowProbe | None = None,
    grouping_proposals: Sequence[GroupProposal] | None = None,
    grouping_passes: Sequence[Sequence[GroupProposal]] | None = None,
    grouping_policy_version: str = "baseline-v1",
    adaptive_grouping_policy: AdaptiveGroupingPolicy | None = None,
    grouping_proposer: GroupProposalGenerator | None = None,
    summary_policy: SummaryPolicy | None = None,
    summary_generator: SummaryGenerator | None = None,
    grounded_summary_generator: GroundedSummaryGenerator | None = None,
    quality_policy: OCRQualityPolicy | None = None,
    source_namespace: str | None = None,
    run_id: str | None = None,
    resume_from_checkpoint: bool = False,
    run_ownership_lease_ms: int | None = None,
) -> tuple[_RunCompat, WorkflowExportBundle | None, OCRWorkflowArtifacts]:
    """Run OCR, grouping, and semantic ingest under one persisted root run."""

    if grouping_proposals is not None and grouping_passes is not None:
        raise ValueError("provide grouping_proposals or grouping_passes, not both")

    if bool(image_payloads) == bool(pdf_path):
        raise ValueError("provide exactly one of image_payloads or pdf_path")
    effective_settings = provider_settings or WorkflowProviderSettings.from_env()
    if (
        summary_policy is not None
        and summary_generator is None
        and grounded_summary_generator is None
    ):
        grounded_summary_generator = build_production_grounded_summary_generator(
            effective_settings
        )
    preparation = _build_ocr_page_preparation(
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
    from .ocr_runtime import run_ocr_root_workflow

    root_result, bundle, artifacts = run_ocr_root_workflow(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        preparation=preparation,
        provider_settings=effective_settings,
        document_id=document_id,
        title=title,
        grouping_policy_version=grouping_policy_version,
        source_namespace=source_namespace,
        grouping_proposals=tuple(grouping_proposals or ()),
        grouping_passes=tuple(grouping_passes or ()),
        adaptive_grouping_policy=adaptive_grouping_policy,
        grouping_proposer=grouping_proposer,
        summary_policy=summary_policy,
        summary_generator=summary_generator,
        grounded_summary_generator=grounded_summary_generator,
        deps=deps,
        run_id=run_id,
        resume_from_checkpoint=resume_from_checkpoint,
        run_ownership_lease_ms=run_ownership_lease_ms,
    )
    return (
        _RunCompat(
            run_id=str(root_result.run_id),
            final_state=root_result.final_state,
            status=str(root_result.status),
        ),
        bundle,
        artifacts,
    )
