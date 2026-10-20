"""Validate and publish accepted results from the OCR root workflow."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.runtime.runtime import RunResult

from .document_models import DocumentTree
from .models import WorkflowExportBundle, WorkflowIngestInput
from .ocr_pipeline import (
    OCRPageJSON,
    OCRPagePreparation,
    OCRWorkflowArtifacts,
    _finalize_ocr_workflow_artifacts,
    _write_document_tree,
    _write_summary,
)
from .ocr_runtime_common import document_tree_from_state
from .ocr_runtime_lineage import (
    recovery_generation,
    semantic_parse_generation,
    validate_root_invocation_lineage,
)
from .ocr_runtime_page_state import (
    reused_pages_from_trace,
    validate_root_page_result,
)
from .ocr_runtime_state import (
    int_list,
    json_pages,
    typed_child_state,
    typed_state,
    validate_root_resume_identity,
)

if TYPE_CHECKING:
    from .providers import WorkflowProviderSettings


def finalize_ocr_root_result(
    *,
    result: RunResult,
    effective_run_id: str,
    logical_run_id: str,
    resume_from_checkpoint: bool,
    conversation_engine: GraphKnowledgeEngine,
    preparation: OCRPagePreparation,
    provider_settings: WorkflowProviderSettings,
    document_id: str,
    title: str,
    source_namespace_identity: str,
    grouping_policy_version: str,
    prepared_page_payloads: Sequence[OCRPageJSON] | None,
    prepared_reused_pages: Sequence[int] | None,
) -> tuple[RunResult, WorkflowExportBundle | None, OCRWorkflowArtifacts]:
    if result.status != "succeeded":
        raise RuntimeError(f"OCR root workflow {result.status}: {result.errors}")
    if effective_run_id != logical_run_id:
        # Recovery uses a fresh persisted incarnation because released runtimes
        # treat a terminal run as immutable. Keep that storage identity internal.
        result = replace(result, run_id=logical_run_id)

    final_state = typed_state(result.final_state)
    validate_root_resume_identity(
        final_state,
        document_id=document_id,
        source_fingerprint=preparation.source_plan.input_fingerprint,
        source_namespace=source_namespace_identity,
        configuration_fingerprints=preparation.page_context.configuration_fingerprints,
        grouping_policy_version=grouping_policy_version,
        parse_generation=semantic_parse_generation(recovery_generation(final_state)),
        require_match=True,
    )
    validate_root_invocation_lineage(
        conversation_engine,
        root_run_id=effective_run_id,
        recovery_generation=recovery_generation(final_state),
    )
    page_result = typed_child_state(
        final_state.get("ocr_page_result"),
        label="OCR root page child",
    )
    if prepared_page_payloads is None:
        # A terminal root resume skips its resolver steps. Revalidate and
        # materialize that accepted generation once; normal execution reuses
        # the exact page snapshot already consumed by grouping.
        validate_root_page_result(
            page_result=page_result,
            context=preparation.page_context,
            source_plan=preparation.source_plan,
        )
        raw_pages = json_pages(page_result, source_plan=preparation.source_plan)
    else:
        raw_pages = list(prepared_page_payloads)

    completed_pages = [
        page_number for page_number, _image_path in preparation.source_plan.page_sources
    ]
    if "completed_pages" in page_result:
        legacy_completed = int_list(page_result, "completed_pages")
        if legacy_completed:
            completed_pages = legacy_completed

    generation = recovery_generation(final_state)
    recovery_suffix = f"|recovery-{generation}" if generation else ""
    if prepared_page_payloads is not None:
        reused_pages = list(prepared_reused_pages or ())
    else:
        page_run_id = f"{effective_run_id}|ocr-pages{recovery_suffix}"
        reused_pages = reused_pages_from_trace(conversation_engine, page_run_id)
    if "reused_pages" in page_result:
        legacy_reused = int_list(page_result, "reused_pages")
        if legacy_reused:
            reused_pages = legacy_reused
    # A terminal child invocation is not rerun during parent resume. Reconcile
    # the reporting list from the accepted page set without weakening validation.
    if resume_from_checkpoint and completed_pages and not reused_pages:
        reused_pages = list(completed_pages)

    grouped_tree: DocumentTree = document_tree_from_state(final_state)
    artifacts = _finalize_ocr_workflow_artifacts(
        document_id=document_id,
        title=title,
        raw_pages=raw_pages,
        completed_pages=completed_pages,
        reused_pages=reused_pages,
        rendered_dir=preparation.source_plan.rendered_dir,
        legacy_dir=preparation.source_plan.legacy_dir,
        state_db_path=preparation.state_db_path,
        progress_path=preparation.progress_path,
        summary_path=preparation.summary_path,
        provider_settings=provider_settings,
        state_store=preparation.state_store,
        restore_grouping_checkpoint=False,
        persist_grouping_checkpoint=False,
        publish_artifacts=False,
        grouping_policy_version=grouping_policy_version,
    )
    workflow_input_payload = final_state.get("workflow_input")
    if not isinstance(workflow_input_payload, dict):
        raise TypeError("OCR root completed without workflow input")
    artifacts.workflow_input = WorkflowIngestInput.model_validate(
        workflow_input_payload
    )
    artifacts.document_tree = grouped_tree
    _write_document_tree(
        preparation.source_plan.legacy_dir / "document-tree.json", grouped_tree
    )
    _write_summary(
        artifacts,
        document_id=document_id,
        provider_settings=provider_settings,
        state_store=preparation.state_store,
    )
    raw_bundle = final_state.get("export_bundle")
    bundle = (
        WorkflowExportBundle.model_validate(raw_bundle)
        if isinstance(raw_bundle, dict)
        else None
    )
    return result, bundle, artifacts
