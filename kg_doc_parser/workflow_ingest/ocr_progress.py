"""Shared progress-mirror updates for standalone and runtime OCR execution."""

from __future__ import annotations

from pathlib import Path

from .ocr_file_io import write_json_atomic
from .ocr_pipeline_contracts import OCRWorkflowProgress
from .ocr_state_store import OCRWorkflowStateStore


def write_progress(progress: OCRWorkflowProgress, progress_path: Path) -> None:
    write_json_atomic(progress_path, progress.model_dump(mode="json"))


def sync_progress_from_state(
    *,
    state_store: OCRWorkflowStateStore,
    document_id: str,
    title: str,
    source_kind: str,
    total_pages: int,
    progress_path: Path,
) -> OCRWorkflowProgress:
    progress = state_store.export_progress_payload(
        document_id=document_id,
        title=title,
        source_kind=source_kind,
        total_pages=total_pages,
    )
    write_progress(progress, progress_path)
    return progress
