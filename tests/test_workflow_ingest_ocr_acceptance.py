"""Tests for validating and accepting persisted OCR page artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from ocr_workflow_support import scratch_path, write_valid_ocr_artifact

import kg_doc_parser.workflow_ingest.ocr_state_store as ocr_state_store_module
from kg_doc_parser.workflow_ingest.ocr_state_store import OCRWorkflowStateStore

pytestmark = [pytest.mark.workflow]


@pytest.mark.ci
def test_legacy_ocr_completion_rejects_malformed_artifact() -> None:
    """The compatibility writer must enforce the canonical OCR schema."""
    scratch = scratch_path("ocr_legacy_malformed_completion")
    artifact = scratch / "page-1.json"
    artifact.write_text("{}", encoding="utf-8")
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="legacy-malformed-completion-doc",
        title="Legacy Malformed Completion",
        source_kind="image",
        total_pages=1,
        input_fingerprint="fingerprint",
    )
    attempt = store.record_attempt(
        document_id="legacy-malformed-completion-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-hash",
        model_name="model",
        artifact_path=artifact,
    )

    assert not store.record_page_completed(
        document_id="legacy-malformed-completion-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-hash",
        artifact_path=artifact,
        model_name="model",
        attempt_index=attempt,
    )
    state = store.get_page_state(
        document_id="legacy-malformed-completion-doc", page_number=1, stage="ocr"
    )
    assert state is not None and state.status == "pending"


@pytest.mark.ci
def test_legacy_ocr_completion_rejects_quality_rejected_artifact() -> None:
    """Compatibility completion cannot accept a strict-policy rejection."""
    scratch = scratch_path("ocr_legacy_quality_rejected_completion")
    artifact = scratch / "page-1.json"
    source_image_sha256 = "image-hash"
    config_fingerprint = "strict-quality-config"
    write_valid_ocr_artifact(
        artifact,
        source_image_sha256=source_image_sha256,
        ocr_configuration_fingerprint=config_fingerprint,
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["metadata"]["ocr_quality_status"] = "rejected"
    payload["metadata"]["ocr_quality_diagnostics"] = [
        {
            "code": "unresolved_items",
            "severity": "error",
            "message": "provider could not ground one observed item",
        }
    ]
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="legacy-quality-rejected-doc",
        title="Legacy Quality Rejected",
        source_kind="image",
        total_pages=1,
        input_fingerprint="fingerprint",
    )
    attempt = store.record_attempt(
        document_id="legacy-quality-rejected-doc",
        page_number=1,
        stage="ocr",
        content_hash=source_image_sha256,
        model_name="model",
        artifact_path=artifact,
        config_fingerprint=config_fingerprint,
    )

    assert not store.record_page_completed(
        document_id="legacy-quality-rejected-doc",
        page_number=1,
        stage="ocr",
        content_hash=source_image_sha256,
        artifact_path=artifact,
        model_name="model",
        attempt_index=attempt,
        config_fingerprint=config_fingerprint,
    )
    state = store.get_page_state(
        document_id="legacy-quality-rejected-doc", page_number=1, stage="ocr"
    )
    assert state is not None and state.status == "pending"
    assert not store.should_skip_page(
        document_id="legacy-quality-rejected-doc",
        page_number=1,
        stage="ocr",
        content_hash=source_image_sha256,
        artifact_path=artifact,
        config_fingerprint=config_fingerprint,
    )
    # Older state databases may already claim completion for a payload that
    # predates the quality gate. Reuse and orphan recovery must revalidate it.
    store._upsert_page_state(
        document_id="legacy-quality-rejected-doc",
        page_number=1,
        stage="ocr",
        attempt_count=attempt,
        status="completed",
        content_hash=source_image_sha256,
        artifact_path=str(artifact),
        last_error=None,
        last_model="model",
        config_fingerprint=config_fingerprint,
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
    )
    assert not store.should_skip_page(
        document_id="legacy-quality-rejected-doc",
        page_number=1,
        stage="ocr",
        content_hash=source_image_sha256,
        artifact_path=artifact,
        config_fingerprint=config_fingerprint,
    )
    store._upsert_page_state(
        document_id="legacy-quality-rejected-doc",
        page_number=1,
        stage="ocr",
        attempt_count=attempt,
        status="pending",
        content_hash=source_image_sha256,
        artifact_path=str(artifact),
        last_error=None,
        last_model="model",
        config_fingerprint=config_fingerprint,
    )
    assert not store.reconcile_orphan_artifact(
        document_id="legacy-quality-rejected-doc",
        page_number=1,
        content_hash=source_image_sha256,
        artifact_path=artifact,
        config_fingerprints=[config_fingerprint],
    )


@pytest.mark.ci
def test_direct_ocr_publication_rejects_quality_rejected_artifact() -> None:
    """The state-store publication boundary rejects errored diagnostics."""
    scratch = scratch_path("ocr_direct_quality_rejected_publish")
    artifact = scratch / "page-1.json"
    source_image_sha256 = "image-hash"
    config_fingerprint = "strict-quality-config"
    write_valid_ocr_artifact(
        artifact,
        source_image_sha256=source_image_sha256,
        ocr_configuration_fingerprint=config_fingerprint,
    )
    original_artifact = artifact.read_bytes()
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["metadata"]["ocr_quality_status"] = "flagged"
    payload["metadata"]["ocr_quality_diagnostics"] = [
        {
            "code": "unresolved_items",
            "severity": "error",
            "message": "provider could not ground one observed item",
        }
    ]

    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="direct-quality-rejected-doc",
        title="Direct Quality Rejected",
        source_kind="image",
        total_pages=1,
        input_fingerprint="fingerprint",
    )
    attempt = store.record_attempt(
        document_id="direct-quality-rejected-doc",
        page_number=1,
        stage="ocr",
        content_hash=source_image_sha256,
        model_name="model",
        artifact_path=artifact,
        config_fingerprint=config_fingerprint,
    )

    assert not store.publish_ocr_page(
        document_id="direct-quality-rejected-doc",
        page_number=1,
        content_hash=source_image_sha256,
        artifact_path=artifact,
        payload=payload,
        model_name="model",
        attempt_index=attempt,
        config_fingerprint=config_fingerprint,
    )
    state = store.get_page_state(
        document_id="direct-quality-rejected-doc", page_number=1, stage="ocr"
    )
    assert state is not None and state.status == "pending"
    assert artifact.read_bytes() == original_artifact


@pytest.mark.ci
def test_direct_ocr_publication_commits_the_validated_payload_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller mutation after validation cannot change the committed bytes."""
    scratch = scratch_path("ocr_publish_validated_payload_snapshot")
    artifact = scratch / "page-1.json"
    source_image_sha256 = "image-hash"
    config_fingerprint = "provider-config"
    write_valid_ocr_artifact(
        artifact,
        source_image_sha256=source_image_sha256,
        ocr_configuration_fingerprint=config_fingerprint,
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    artifact.unlink()

    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="publish-snapshot-doc",
        title="Publish Snapshot",
        source_kind="image",
        total_pages=1,
        input_fingerprint="fingerprint",
    )
    attempt = store.record_attempt(
        document_id="publish-snapshot-doc",
        page_number=1,
        stage="ocr",
        content_hash=source_image_sha256,
        model_name="model",
        artifact_path=artifact,
        config_fingerprint=config_fingerprint,
    )
    write_bytes_atomic = ocr_state_store_module._write_bytes_atomic

    def mutate_then_write(path: Path, artifact_bytes: bytes) -> None:
        payload["metadata"]["ocr_quality_status"] = "rejected"
        payload["metadata"]["ocr_quality_diagnostics"] = [
            {
                "code": "unresolved_items",
                "severity": "error",
                "message": "late caller mutation",
            }
        ]
        write_bytes_atomic(path, artifact_bytes)

    monkeypatch.setattr(ocr_state_store_module, "_write_bytes_atomic", mutate_then_write)
    assert store.publish_ocr_page(
        document_id="publish-snapshot-doc",
        page_number=1,
        content_hash=source_image_sha256,
        artifact_path=artifact,
        payload=payload,
        model_name="model",
        attempt_index=attempt,
        config_fingerprint=config_fingerprint,
    )
    persisted = json.loads(artifact.read_text(encoding="utf-8"))
    assert "ocr_quality_status" not in persisted["metadata"]
    state = store.get_page_state(
        document_id="publish-snapshot-doc", page_number=1, stage="ocr"
    )
    assert state is not None and state.status == "completed"


@pytest.mark.ci
def test_legacy_ocr_completion_rejects_wrong_page_artifact() -> None:
    """A valid artifact for another page cannot satisfy this page's row."""
    scratch = scratch_path("ocr_legacy_wrong_page_completion")
    artifact = scratch / "page-1.json"
    write_valid_ocr_artifact(
        artifact,
        page_number=2,
        source_image_sha256="image-hash",
    )
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="legacy-wrong-page-completion-doc",
        title="Legacy Wrong Page Completion",
        source_kind="image",
        total_pages=1,
        input_fingerprint="fingerprint",
    )
    attempt = store.record_attempt(
        document_id="legacy-wrong-page-completion-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-hash",
        model_name="model",
        artifact_path=artifact,
    )

    assert not store.record_page_completed(
        document_id="legacy-wrong-page-completion-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-hash",
        artifact_path=artifact,
        model_name="model",
        attempt_index=attempt,
    )
