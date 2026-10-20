"""Finalize accepted OCR pages into durable document artifacts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from .adapters import OCRPageJSON, normalize_ocr_pages
from .document_adapters import (
    apply_grouping_proposals,
    attach_document_tree_metadata,
    build_document_tree,
    suggest_cross_page_continuations,
)
from .document_grouping import GroupProposal
from .document_models import DocumentTree, SummaryPolicy
from .models import WorkflowIngestInput
from .ocr_file_io import (
    write_bytes_atomic as _write_bytes_atomic,
)
from .ocr_file_io import (
    write_json_atomic as _write_json_atomic,
)
from .ocr_pipeline_contracts import (
    GroundedSummaryGenerator,
    OCRWorkflowArtifacts,
    SummaryGenerator,
)
from .ocr_state_store import OCRWorkflowStateStore
from .ocr_summary_workflow import _generate_tree_summaries
from .providers import WorkflowProviderSettings
from .serialization import JsonObject, JsonValue, json_safe


def _write_document_tree(path: Path, tree: DocumentTree) -> None:
    """Persist the derived tree with replace semantics for crash-safe replay."""
    _write_bytes_atomic(
        path,
        json.dumps(tree.model_dump(mode="json"), indent=2, ensure_ascii=False).encode("utf-8"),
    )


def _coalesce_cross_page_continuations(
    proposals: Sequence[GroupProposal],
) -> tuple[GroupProposal, ...]:
    """Turn adjacent pair suggestions into disjoint continuation chains."""
    if not proposals:
        return ()
    chains: list[GroupProposal] = []
    current = proposals[0]
    for proposal in proposals[1:]:
        if (
            proposal.kind == current.kind
            and proposal.members[0] == current.members[-1]
        ):
            current = GroupProposal(
                kind=current.kind,
                members=current.members + (proposal.members[-1],),
            )
        else:
            chains.append(current)
            current = proposal
    chains.append(current)
    return tuple(chains)


def _apply_cross_page_continuations(
    tree: DocumentTree,
    *,
    grouping_policy_version: str,
) -> DocumentTree:
    """Apply only provider-signaled, adjacent cross-page continuations."""
    proposals = _coalesce_cross_page_continuations(suggest_cross_page_continuations(tree))
    if not proposals:
        return tree
    return apply_grouping_proposals(
        tree,
        proposals,
        grouping_policy_version=grouping_policy_version,
    )




def _finalize_ocr_workflow_artifacts(
    *,
    document_id: str,
    title: str,
    raw_pages: list[OCRPageJSON],
    completed_pages: list[int],
    reused_pages: list[int],
    rendered_dir: Path,
    legacy_dir: Path,
    state_db_path: Path,
    progress_path: Path,
    summary_path: Path,
    provider_settings: WorkflowProviderSettings,
    state_store: OCRWorkflowStateStore,
    restore_grouping_checkpoint: bool = True,
    persist_grouping_checkpoint: bool = True,
    publish_artifacts: bool = True,
    grouping_policy_version: str = "baseline-v1",
    summary_policy: SummaryPolicy | None = None,
    summary_generator: SummaryGenerator | None = None,
    grounded_summary_generator: GroundedSummaryGenerator | None = None,
) -> OCRWorkflowArtifacts:
    workflow_input: WorkflowIngestInput = normalize_ocr_pages(
        document_id=document_id,
        title=title,
        pages=raw_pages,
    )
    input_fingerprint = state_store.document_input_fingerprint(document_id=document_id)
    source_revision = (
        f"ocr-v1:{input_fingerprint}"
        if input_fingerprint
        else "ocr-v1"
    )
    document_tree = build_document_tree(workflow_input, source_revision=source_revision)
    if document_tree is not None:
        checkpointed_tree = (
            state_store.load_document_tree(
                document_id=document_id,
                source_fingerprint=document_tree.source_fingerprint(),
                grouping_policy_version=grouping_policy_version,
            )
            if restore_grouping_checkpoint and persist_grouping_checkpoint
            else None
        )
        if checkpointed_tree is not None:
            document_tree = checkpointed_tree
        else:
            document_tree = _apply_cross_page_continuations(
                document_tree,
                grouping_policy_version=grouping_policy_version,
            )
            if persist_grouping_checkpoint:
                state_store.save_document_tree(
                    document_id=document_id,
                    tree=document_tree,
                    grouping_policy_version=grouping_policy_version,
                    last_runtime_step="finalize_grouping_checkpoint",
                )
        if persist_grouping_checkpoint:
            existing_tree_revision = state_store.document_tree_revision(document_id=document_id)
            document_tree = _generate_tree_summaries(
                document_tree,
                document_id=document_id,
                state_store=state_store,
                policy=summary_policy,
                generator=summary_generator,
                grounded_generator=grounded_summary_generator,
                grouping_policy_version=grouping_policy_version,
                expected_tree_revision=existing_tree_revision,
                require_no_existing_tree=existing_tree_revision is None,
            )
        attach_document_tree_metadata(workflow_input, document_tree)
    artifacts = OCRWorkflowArtifacts(
        workflow_input=workflow_input,
        ocr_pages=raw_pages,
        legacy_dir=legacy_dir,
        rendered_dir=rendered_dir,
        state_db_path=state_db_path,
        progress_path=progress_path,
        summary_path=summary_path,
        completed_pages=completed_pages,
        reused_pages=reused_pages,
        document_tree=document_tree,
    )
    if publish_artifacts:
        if document_tree is not None:
            _write_document_tree(legacy_dir / "document-tree.json", document_tree)
        _write_summary(
            artifacts,
            document_id=document_id,
            provider_settings=provider_settings,
            state_store=state_store,
        )
    return artifacts


def _write_summary(
    artifacts: OCRWorkflowArtifacts,
    *,
    document_id: str,
    provider_settings: WorkflowProviderSettings,
    state_store: OCRWorkflowStateStore,
) -> None:
    payload: JsonObject = {
        "document_id": document_id,
        "ocr_provider": provider_settings.ocr.provider,
        "ocr_model": provider_settings.ocr.model,
        "legacy_dir": str(artifacts.legacy_dir),
        "rendered_dir": str(artifacts.rendered_dir),
        "state_db_path": str(artifacts.state_db_path),
        "progress_path": str(artifacts.progress_path),
        "document_completed": state_store.read_document_completed(document_id=document_id),
        "completed_pages": cast(list[JsonValue], list(artifacts.completed_pages)),
        "reused_pages": cast(list[JsonValue], list(artifacts.reused_pages)),
        "model_attempts": json_safe(
            {
                str(page_number): state_store.list_model_attempts(
                    document_id=document_id,
                    page_number=page_number,
                )
                for page_number in artifacts.completed_pages
            }
        ),
        "summary_attempts": json_safe(
            state_store.list_summary_attempts(document_id=document_id)
        ),
        "page_count": len(artifacts.ocr_pages),
        "document_tree_path": str(artifacts.document_tree_path) if artifacts.document_tree is not None else None,
    }
    _write_json_atomic(artifacts.summary_path, payload)

