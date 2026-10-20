"""Typed decoding and projection of persisted OCR progress state."""

from __future__ import annotations

import sqlite3

import pytest
from ocr_workflow_support import scratch_path

from kg_doc_parser.workflow_ingest.ocr_state_store import OCRWorkflowStateStore
from kg_doc_parser.workflow_ingest.ocr_state_types import (
    decode_ocr_page_status,
    decode_ocr_stage,
)

pytestmark = [pytest.mark.workflow]


def _store(name: str) -> OCRWorkflowStateStore:
    store = OCRWorkflowStateStore(scratch_path(name) / "ocr-state.sqlite")
    store.ensure_document(
        document_id="progress-doc",
        title="Progress",
        source_kind="image",
        total_pages=1,
        input_fingerprint="source-v1",
    )
    return store


@pytest.mark.ci
def test_progress_export_decodes_sqlite_values_into_typed_model() -> None:
    store = _store("ocr_progress_export")
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            """
            INSERT INTO page_state (
                document_id, page_number, stage, attempt_count, status,
                content_hash, artifact_path
            ) VALUES (?, 1, 'ocr', 1, 'completed', ?, ?)
            """,
            ("progress-doc", "image-digest", "page-1.json"),
        )

    progress = store.export_progress_payload(
        document_id="progress-doc",
        title="Progress",
        source_kind="image",
        total_pages=1,
    )

    assert progress.pages["1"].page_number == 1
    assert progress.pages["1"].image_sha256 == "image-digest"
    assert progress.pages["1"].json_path == "page-1.json"
    assert progress.pages["1"].status == "completed"


@pytest.mark.ci
def test_progress_export_rejects_non_text_checkpoint_fields() -> None:
    store = _store("ocr_progress_malformed_text")
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            """
            INSERT INTO page_state (
                document_id, page_number, stage, attempt_count, status,
                content_hash, artifact_path
            ) VALUES (?, 1, 'ocr', 1, ?, ?, ?)
            """,
            (
                "progress-doc",
                sqlite3.Binary(b"completed"),
                "image-digest",
                "page-1.json",
            ),
        )

    with pytest.raises(TypeError, match="status.*text"):
        store.export_progress_payload(
            document_id="progress-doc",
            title="Progress",
            source_kind="image",
            total_pages=1,
        )


@pytest.mark.ci
def test_completion_read_rejects_non_boolean_integer_state() -> None:
    store = _store("ocr_progress_invalid_completion")
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE document_state SET is_completed = 2 WHERE document_id = ?",
            ("progress-doc",),
        )

    with pytest.raises(ValueError, match="is_completed must be 0 or 1"):
        store.read_document_completed(document_id="progress-doc")


@pytest.mark.ci
def test_ocr_state_decoders_accept_only_persisted_stage_and_status_values() -> None:
    assert decode_ocr_stage("render") == "render"
    assert decode_ocr_stage("ocr") == "ocr"
    assert decode_ocr_page_status("pending") == "pending"
    assert decode_ocr_page_status("completed") == "completed"
    assert decode_ocr_page_status("failed") == "failed"
    with pytest.raises(ValueError, match="stage is invalid"):
        decode_ocr_stage("summary")
    with pytest.raises(ValueError, match="page status is invalid"):
        decode_ocr_page_status("running")


@pytest.mark.ci
def test_page_state_snapshot_rejects_unknown_persisted_status() -> None:
    store = _store("ocr_page_state_unknown_status")
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            """
            INSERT INTO page_state (
                document_id, page_number, stage, attempt_count, status
            ) VALUES (?, 1, 'ocr', 1, 'running')
            """,
            ("progress-doc",),
        )

    with pytest.raises(ValueError, match="page status is invalid"):
        store.get_page_state(document_id="progress-doc", page_number=1, stage="ocr")


@pytest.mark.ci
def test_summary_journal_rejects_status_outside_its_typed_contract() -> None:
    store = _store("ocr_summary_invalid_status")

    with pytest.raises(ValueError, match="unsupported summary status"):
        store.record_summary_attempt(
            document_id="progress-doc",
            node_id="root",
            input_fingerprint="input-v1",
            status="running",  # type: ignore[arg-type]
        )
