"""Compare-and-swap guarantees for accepted OCR page state."""

from __future__ import annotations

import pytest
from ocr_workflow_support import (
    scratch_path as _scratch,
)
from ocr_workflow_support import (
    write_valid_ocr_artifact as _write_valid_ocr_artifact,
)

from kg_doc_parser.workflow_ingest.ocr_state_store import OCRWorkflowStateStore

pytestmark = [pytest.mark.workflow]

@pytest.mark.ci
def test_ocr_page_completion_rejects_a_late_writer_after_acceptance() -> None:
    scratch = _scratch("ocr_page_completion_cas")
    artifact = scratch / "page-1.json"
    artifact.write_text('{"page": 1}', encoding="utf-8")
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="page-completion-cas-doc",
        title="Page Completion CAS",
        source_kind="image",
        total_pages=1,
        input_fingerprint="fingerprint",
    )
    attempt = store.record_attempt(
        document_id="page-completion-cas-doc",
        page_number=1,
        stage="render",
        content_hash="image-a",
        model_name="renderer-a",
        artifact_path=artifact,
        config_fingerprint="render-a",
    )
    assert store.record_page_completed(
        document_id="page-completion-cas-doc",
        page_number=1,
        stage="render",
        content_hash="image-a",
        artifact_path=artifact,
        model_name="renderer-a",
        attempt_index=attempt,
        config_fingerprint="render-a",
    )

    assert not store.record_page_completed(
        document_id="page-completion-cas-doc",
        page_number=1,
        stage="render",
        content_hash="image-b",
        artifact_path=artifact,
        model_name="renderer-b",
        config_fingerprint="render-b",
    )
    state = store.get_page_state(
        document_id="page-completion-cas-doc", page_number=1, stage="render"
    )
    assert state is not None
    assert state.content_hash == "image-a"
    assert state.config_fingerprint == "render-a"


@pytest.mark.ci
def test_legacy_state_upsert_cannot_replace_completed_page() -> None:
    """Compatibility rebuilds must not downgrade an accepted page state."""
    scratch = _scratch("ocr_legacy_upsert_cas")
    artifact = scratch / "page-1.json"
    artifact.write_text('{"page": 1}', encoding="utf-8")
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="legacy-upsert-cas-doc",
        title="Legacy Upsert CAS",
        source_kind="image",
        total_pages=1,
        input_fingerprint="fingerprint",
    )
    store._upsert_page_state(
        document_id="legacy-upsert-cas-doc",
        page_number=1,
        stage="ocr",
        attempt_count=1,
        status="completed",
        content_hash="image-a",
        artifact_path=str(artifact),
        last_error=None,
        last_model="model-a",
        config_fingerprint="provider-a/model-a",
        artifact_sha256="artifact-a",
    )
    store._upsert_page_state(
        document_id="legacy-upsert-cas-doc",
        page_number=1,
        stage="ocr",
        attempt_count=2,
        status="pending",
        content_hash="image-b",
        artifact_path=str(artifact),
        last_error="stale rebuild",
        last_model="model-b",
        config_fingerprint="provider-b/model-b",
        artifact_sha256=None,
    )

    state = store.get_page_state(
        document_id="legacy-upsert-cas-doc", page_number=1, stage="ocr"
    )
    assert state is not None
    assert state.status == "completed"
    assert state.content_hash == "image-a"
    assert state.config_fingerprint == "provider-a/model-a"


@pytest.mark.ci
def test_ocr_page_failure_cannot_downgrade_an_accepted_result() -> None:
    scratch = _scratch("ocr_page_failure_cas")
    artifact = scratch / "page-1.json"
    _write_valid_ocr_artifact(
        artifact,
        page_number=1,
        source_image_sha256="image-a",
        ocr_configuration_fingerprint="provider-a/model-a",
    )
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="page-failure-cas-doc",
        title="Page Failure CAS",
        source_kind="image",
        total_pages=1,
        input_fingerprint="fingerprint",
    )
    attempt = store.record_attempt(
        document_id="page-failure-cas-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-a",
        model_name="model-a",
        artifact_path=artifact,
        config_fingerprint="provider-a/model-a",
    )
    assert store.record_page_completed(
        document_id="page-failure-cas-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-a",
        artifact_path=artifact,
        model_name="model-a",
        attempt_index=attempt,
        config_fingerprint="provider-a/model-a",
    )

    assert not store.record_page_failed(
        document_id="page-failure-cas-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-a",
        config_fingerprint="provider-a/model-a",
        error_message="late provider failure",
        model_name="model-a",
        attempt_index=attempt,
    )
    state = store.get_page_state(
        document_id="page-failure-cas-doc", page_number=1, stage="ocr"
    )
    assert state is not None
    assert state.status == "completed"
    assert state.last_error is None


@pytest.mark.ci
def test_ocr_page_failure_cannot_replace_a_newer_pending_attempt() -> None:
    scratch = _scratch("ocr_page_failure_attempt_cas")
    artifact = scratch / "page-1.json"
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="page-failure-attempt-cas-doc",
        title="Page Failure Attempt CAS",
        source_kind="image",
        total_pages=1,
        input_fingerprint="fingerprint",
    )
    first = store.record_attempt(
        document_id="page-failure-attempt-cas-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-a",
        model_name="model-a",
        artifact_path=artifact,
        config_fingerprint="provider-a/model-a",
    )
    second = store.record_attempt(
        document_id="page-failure-attempt-cas-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-a",
        model_name="model-b",
        artifact_path=artifact,
        config_fingerprint="provider-b/model-b",
    )
    assert second == first + 1
    assert not store.record_page_failed(
        document_id="page-failure-attempt-cas-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-a",
        config_fingerprint="provider-a/model-a",
        error_message="late first-attempt failure",
        model_name="model-a",
        attempt_index=first,
    )
    state = store.get_page_state(
        document_id="page-failure-attempt-cas-doc", page_number=1, stage="ocr"
    )
    assert state is not None
    assert state.status == "pending"
    assert state.attempt_count == second
    assert state.config_fingerprint == "provider-b/model-b"


@pytest.mark.ci
def test_ocr_page_completion_cannot_replace_a_newer_pending_attempt() -> None:
    scratch = _scratch("ocr_page_completion_attempt_cas")
    artifact = scratch / "page-1.json"
    artifact.write_text('{"page": 1}', encoding="utf-8")
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="page-completion-attempt-cas-doc",
        title="Page Completion Attempt CAS",
        source_kind="image",
        total_pages=1,
        input_fingerprint="fingerprint",
    )
    first = store.record_attempt(
        document_id="page-completion-attempt-cas-doc",
        page_number=1,
        stage="render",
        content_hash="source-a",
        model_name="renderer",
        artifact_path=artifact,
        config_fingerprint="renderer-v1",
    )
    second = store.record_attempt(
        document_id="page-completion-attempt-cas-doc",
        page_number=1,
        stage="render",
        content_hash="source-a",
        model_name="renderer",
        artifact_path=artifact,
        config_fingerprint="renderer-v1",
    )
    assert second == first + 1
    assert not store.record_page_completed(
        document_id="page-completion-attempt-cas-doc",
        page_number=1,
        stage="render",
        content_hash="source-a",
        artifact_path=artifact,
        model_name="renderer",
        attempt_index=first,
        config_fingerprint="renderer-v1",
    )
    state = store.get_page_state(
        document_id="page-completion-attempt-cas-doc", page_number=1, stage="render"
    )
    assert state is not None
    assert state.status == "pending"
    assert state.attempt_count == second
