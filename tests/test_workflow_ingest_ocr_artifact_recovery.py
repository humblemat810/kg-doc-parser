"""OCR artifact identity, rebuild, schema, and recovery race tests."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from ocr_workflow_support import (
    fake_ocr_response as _fake_ocr_response,
)
from ocr_workflow_support import (
    scratch_path as _scratch,
)
from ocr_workflow_support import (
    write_valid_ocr_artifact as _write_valid_ocr_artifact,
)

import kg_doc_parser.workflow_ingest.ocr_state_store as ocr_state_store_module
from kg_doc_parser.models import SplitPage, SplitPageMeta
from kg_doc_parser.workflow_ingest.ocr_state_store import OCRWorkflowStateStore

pytestmark = [pytest.mark.workflow]

@pytest.mark.ci
def test_ocr_state_does_not_skip_when_matching_artifact_was_removed() -> None:
    scratch = _scratch("ocr_missing_artifact")
    artifact = scratch / "page-1.json"
    _write_valid_ocr_artifact(
        artifact,
        source_image_sha256="image-hash",
        ocr_configuration_fingerprint="provider-a/model-a",
    )
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="missing-artifact-doc",
        title="Missing Artifact",
        source_kind="image",
        total_pages=1,
        input_fingerprint="fingerprint",
    )
    attempt = store.record_attempt(
        document_id="missing-artifact-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-hash",
        model_name="model",
        artifact_path=artifact,
        config_fingerprint="provider-a/model-a",
    )
    store.record_page_completed(
        document_id="missing-artifact-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-hash",
        artifact_path=artifact,
        model_name="model",
        attempt_index=attempt,
        config_fingerprint="provider-a/model-a",
    )
    assert store.should_skip_page(
        document_id="missing-artifact-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-hash",
        artifact_path=artifact,
        config_fingerprint="provider-a/model-a",
    )

    artifact.unlink()

    assert not store.should_skip_page(
        document_id="missing-artifact-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-hash",
        artifact_path=artifact,
        config_fingerprint="provider-a/model-a",
    )


@pytest.mark.ci
def test_rebuild_ignores_artifacts_removed_by_a_shorter_source_revision() -> None:
    """A shorter replacement document cannot resurrect old trailing pages."""
    scratch = _scratch("ocr_removed_trailing_pages")
    rendered_dir = scratch / "rendered"
    legacy_dir = scratch / "legacy"
    rendered_dir.mkdir(parents=True)
    legacy_dir.mkdir(parents=True)

    for page_number in (1, 2):
        rendered = rendered_dir / f"page_{page_number}.png"
        rendered.write_bytes(f"source-page-{page_number}".encode("ascii"))
        split_page = SplitPage(
            pdf_page_num=page_number,
            metadata=SplitPageMeta(
                ocr_model_name="model-a",
                ocr_datetime=0.0,
                ocr_json_version="workflow_ingest_v1",
            ),
            **_fake_ocr_response(page_number, f"page {page_number}").model_dump(
                dump_format="python"
            ),
        )
        rendered_digest = hashlib.sha256(rendered.read_bytes()).hexdigest()
        page_payload = split_page.dump_raw(dump_format="json")
        page_payload["metadata"] = {
            **split_page.metadata.model_dump(mode="json"),
            "source_image_sha256": rendered_digest,
        }
        (legacy_dir / f"page_{page_number}.json").write_text(
            json.dumps(page_payload),
            encoding="utf-8",
        )

    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    assert store.rebuild_from_artifacts(
        document_id="shorter-source-revision",
        title="Shorter Source Revision",
        source_kind="image",
        total_pages=1,
        input_fingerprint="new-input-fingerprint",
        rendered_dir=rendered_dir,
        legacy_dir=legacy_dir,
        progress_path=scratch / "missing-progress.json",
    )

    page_one = store.get_page_state(
        document_id="shorter-source-revision", page_number=1, stage="ocr"
    )
    assert page_one is not None and page_one.status == "completed"
    assert (
        store.get_page_state(
            document_id="shorter-source-revision", page_number=2, stage="ocr"
        )
        is None
    )
    assert store.read_document_completed(document_id="shorter-source-revision")


@pytest.mark.ci
def test_rebuild_rejects_legacy_ocr_when_page_40_changes_in_new_source_revision() -> (
    None
):
    """A changed page 40 in revision two invalidates revision one's OCR."""
    scratch = _scratch("ocr_replaced_same_page")
    rendered_dir = scratch / "rendered"
    legacy_dir = scratch / "legacy"
    rendered_dir.mkdir(parents=True)
    legacy_dir.mkdir(parents=True)
    rendered = rendered_dir / "page_40.png"
    rendered.write_bytes(b"revision-one-page-40")
    old_digest = hashlib.sha256(rendered.read_bytes()).hexdigest()
    _write_valid_ocr_artifact(
        legacy_dir / "page_40.json", source_image_sha256=old_digest
    )
    (scratch / "progress.json").write_text(
        json.dumps({"pages": {"40": {"image_sha256": old_digest}}}),
        encoding="utf-8",
    )

    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    assert store.rebuild_from_artifacts(
        document_id="source-revision-page-40-doc",
        title="Source Revision Page 40",
        source_kind="pdf",
        total_pages=100,
        input_fingerprint="source-revision-one",
        rendered_dir=rendered_dir,
        legacy_dir=legacy_dir,
        progress_path=scratch / "progress.json",
    )

    initial_state = store.get_page_state(
        document_id="source-revision-page-40-doc", page_number=40, stage="ocr"
    )
    assert initial_state is not None
    assert initial_state.status == "completed"

    rendered.write_bytes(b"revision-two-changed-page-40")
    assert store.rebuild_from_artifacts(
        document_id="source-revision-page-40-doc",
        title="Source Revision Page 40",
        source_kind="pdf",
        total_pages=100,
        input_fingerprint="source-revision-two",
        rendered_dir=rendered_dir,
        legacy_dir=legacy_dir,
        progress_path=scratch / "progress.json",
    )
    state = store.get_page_state(
        document_id="source-revision-page-40-doc", page_number=40, stage="ocr"
    )
    assert state is not None
    assert state.status == "pending"
    assert state.content_hash == hashlib.sha256(rendered.read_bytes()).hexdigest()
    assert state.last_error == (
        "OCR artifact source-image digest does not match the current rendered page"
    )


@pytest.mark.ci
def test_rebuild_does_not_trust_legacy_image_as_current_source() -> None:
    scratch = _scratch("ocr_legacy_image_not_authoritative")
    rendered_dir = scratch / "rendered"
    legacy_dir = scratch / "legacy"
    rendered_dir.mkdir(parents=True)
    legacy_dir.mkdir(parents=True)
    stale_legacy_image = legacy_dir / "page_1.png"
    stale_legacy_image.write_bytes(b"stale legacy image")
    artifact = legacy_dir / "page_1.json"
    _write_valid_ocr_artifact(
        artifact,
        source_image_sha256=hashlib.sha256(stale_legacy_image.read_bytes()).hexdigest(),
        ocr_configuration_fingerprint="provider/model",
    )
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")

    assert store.rebuild_from_artifacts(
        document_id="ocr-legacy-source-not-authoritative-doc",
        title="Legacy Source Not Authoritative",
        source_kind="image",
        total_pages=1,
        input_fingerprint="current-source-without-render",
        rendered_dir=rendered_dir,
        legacy_dir=legacy_dir,
        progress_path=scratch / "progress.json",
    )

    state = store.get_page_state(
        document_id="ocr-legacy-source-not-authoritative-doc",
        page_number=1,
        stage="ocr",
    )
    assert state is not None
    assert state.status == "pending"
    assert state.content_hash is None
    assert state.last_error == "OCR artifact has no authoritative source-image digest"


@pytest.mark.ci
def test_ocr_state_rejects_stale_provider_config_or_tampered_artifact() -> None:
    scratch = _scratch("ocr_config_identity")
    artifact = scratch / "page-1.json"
    _write_valid_ocr_artifact(
        artifact,
        source_image_sha256="image-hash",
        ocr_configuration_fingerprint="provider-a/model-a",
    )
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="config-identity-doc",
        title="Config Identity",
        source_kind="image",
        total_pages=1,
        input_fingerprint="fingerprint",
    )
    attempt = store.record_attempt(
        document_id="config-identity-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-hash",
        model_name="model-a",
        artifact_path=artifact,
        config_fingerprint="provider-a/model-a",
    )
    store.record_page_completed(
        document_id="config-identity-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-hash",
        artifact_path=artifact,
        model_name="model-a",
        attempt_index=attempt,
        config_fingerprint="provider-a/model-a",
    )

    assert store.should_skip_page(
        document_id="config-identity-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-hash",
        artifact_path=artifact,
        config_fingerprint=("provider-a/model-a", "provider-a/model-b"),
    )
    assert not store.should_skip_page(
        document_id="config-identity-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-hash",
        artifact_path=artifact,
        config_fingerprint="provider-b/model-a",
    )

    child_script = f"""
from pathlib import Path
from kg_doc_parser.workflow_ingest.ocr_pipeline import OCRWorkflowStateStore
store = OCRWorkflowStateStore(Path({str(scratch / "ocr-state.sqlite")!r}))
assert not store.should_skip_page(
    document_id='config-identity-doc', page_number=1, stage='ocr',
    content_hash='image-hash', artifact_path=Path({str(artifact)!r}),
    config_fingerprint='provider-b/model-a'
)
assert store.should_skip_page(
    document_id='config-identity-doc', page_number=1, stage='ocr',
    content_hash='image-hash', artifact_path=Path({str(artifact)!r}),
    config_fingerprint='provider-a/model-a'
)
"""
    child = subprocess.run(
        [sys.executable, "-c", child_script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0

    artifact.write_text('{"page": 1, "tampered": true}', encoding="utf-8")
    assert not store.should_skip_page(
        document_id="config-identity-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-hash",
        artifact_path=artifact,
        config_fingerprint="provider-a/model-a",
    )


@pytest.mark.ci
@pytest.mark.parametrize(
    ("schema_version", "expected_status"),
    [
        ("0.1", "completed"),
        ("workflow_ingest_v1", "completed"),
        ("workflow_ingest_v99", "pending"),
    ],
)
def test_artifact_rebuild_accepts_known_ocr_schemas_only(
    schema_version: str, expected_status: str
) -> None:
    scratch = _scratch("ocr_schema_rebuild")
    rendered_dir = scratch / "rendered"
    legacy_dir = scratch / "legacy"
    rendered_dir.mkdir(parents=True)
    legacy_dir.mkdir(parents=True)
    image = rendered_dir / "page_1.png"
    image.write_bytes(b"source-image")
    artifact = legacy_dir / "page_1.json"
    source_digest = hashlib.sha256(image.read_bytes()).hexdigest()
    _write_valid_ocr_artifact(
        artifact,
        source_image_sha256=source_digest,
        ocr_configuration_fingerprint="provider/model",
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["metadata"]["ocr_json_version"] = schema_version
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    assert store.rebuild_from_artifacts(
        document_id="ocr-schema-rebuild-doc",
        title="OCR Schema Rebuild",
        source_kind="image",
        total_pages=1,
        input_fingerprint="source-revision",
        rendered_dir=rendered_dir,
        legacy_dir=legacy_dir,
        progress_path=scratch / "progress.json",
    )

    state = store.get_page_state(
        document_id="ocr-schema-rebuild-doc", page_number=1, stage="ocr"
    )
    assert state is not None
    assert state.status == expected_status
    if schema_version == "workflow_ingest_v99":
        assert state.last_error == (
            "unsupported OCR artifact schema version: 'workflow_ingest_v99'"
        )


@pytest.mark.ci
def test_artifact_rebuild_does_not_resurrect_quality_rejected_ocr() -> None:
    scratch = _scratch("ocr_quality_rejected_rebuild")
    rendered_dir = scratch / "rendered"
    legacy_dir = scratch / "legacy"
    rendered_dir.mkdir(parents=True)
    legacy_dir.mkdir(parents=True)
    image = rendered_dir / "page_1.png"
    image.write_bytes(b"source-image")
    artifact = legacy_dir / "page_1.json"
    source_digest = hashlib.sha256(image.read_bytes()).hexdigest()
    _write_valid_ocr_artifact(
        artifact,
        source_image_sha256=source_digest,
        ocr_configuration_fingerprint="provider/model",
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
    assert store.rebuild_from_artifacts(
        document_id="ocr-quality-rejected-rebuild-doc",
        title="OCR Quality Rejected Rebuild",
        source_kind="image",
        total_pages=1,
        input_fingerprint="source-revision",
        rendered_dir=rendered_dir,
        legacy_dir=legacy_dir,
        progress_path=scratch / "progress.json",
    )
    state = store.get_page_state(
        document_id="ocr-quality-rejected-rebuild-doc",
        page_number=1,
        stage="ocr",
    )
    assert state is not None and state.status == "pending"
    assert state.last_error == "OCR quality status is rejected"


@pytest.mark.ci
def test_ocr_page_reuse_requires_artifact_path_and_supported_schema() -> None:
    scratch = _scratch("ocr_schema_reuse")
    artifact = scratch / "page-1.json"
    source_digest = hashlib.sha256(b"source-image").hexdigest()
    config_fingerprint = "provider/model"
    _write_valid_ocr_artifact(
        artifact,
        source_image_sha256=source_digest,
        ocr_configuration_fingerprint=config_fingerprint,
    )
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="ocr-schema-reuse-doc",
        title="OCR Schema Reuse",
        source_kind="image",
        total_pages=1,
        input_fingerprint="source-revision",
    )
    attempt = store.record_attempt(
        document_id="ocr-schema-reuse-doc",
        page_number=1,
        stage="ocr",
        content_hash=source_digest,
        model_name="model",
        artifact_path=artifact,
        config_fingerprint=config_fingerprint,
    )
    assert store.record_page_completed(
        document_id="ocr-schema-reuse-doc",
        page_number=1,
        stage="ocr",
        content_hash=source_digest,
        artifact_path=artifact,
        model_name="model",
        attempt_index=attempt,
        config_fingerprint=config_fingerprint,
    )
    identity = {
        "document_id": "ocr-schema-reuse-doc",
        "page_number": 1,
        "stage": "ocr",
        "content_hash": source_digest,
        "config_fingerprint": config_fingerprint,
    }
    assert not store.should_skip_page(**identity)

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["metadata"]["ocr_json_version"] = "workflow_ingest_v99"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            "UPDATE page_state SET artifact_sha256 = ? WHERE document_id = ? AND page_number = 1 AND stage = 'ocr'",
            (hashlib.sha256(artifact.read_bytes()).hexdigest(), "ocr-schema-reuse-doc"),
        )

    assert not store.should_skip_page(**identity, artifact_path=artifact)


@pytest.mark.ci
def test_legacy_completion_rejects_unsupported_ocr_schema_version() -> None:
    scratch = _scratch("ocr_unsupported_schema_completion")
    artifact = scratch / "page-1.json"
    source_digest = "source-image-hash"
    config_fingerprint = "provider/model"
    _write_valid_ocr_artifact(
        artifact,
        source_image_sha256=source_digest,
        ocr_configuration_fingerprint=config_fingerprint,
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["metadata"]["ocr_json_version"] = "workflow_ingest_v99"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="ocr-unsupported-schema-completion-doc",
        title="OCR Unsupported Schema Completion",
        source_kind="image",
        total_pages=1,
        input_fingerprint="source-revision",
    )
    attempt = store.record_attempt(
        document_id="ocr-unsupported-schema-completion-doc",
        page_number=1,
        stage="ocr",
        content_hash=source_digest,
        model_name="model",
        artifact_path=artifact,
        config_fingerprint=config_fingerprint,
    )

    assert not store.record_page_completed(
        document_id="ocr-unsupported-schema-completion-doc",
        page_number=1,
        stage="ocr",
        content_hash=source_digest,
        artifact_path=artifact,
        model_name="model",
        attempt_index=attempt,
        config_fingerprint=config_fingerprint,
    )


@pytest.mark.ci
def test_orphan_reconciliation_rejects_unsupported_ocr_schema_version() -> None:
    scratch = _scratch("ocr_unsupported_schema_orphan")
    artifact = scratch / "page-1.json"
    source_digest = "source-image-hash"
    config_fingerprint = "provider/model"
    _write_valid_ocr_artifact(
        artifact,
        source_image_sha256=source_digest,
        ocr_configuration_fingerprint=config_fingerprint,
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["metadata"]["ocr_json_version"] = "workflow_ingest_v99"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="ocr-unsupported-schema-orphan-doc",
        title="OCR Unsupported Schema Orphan",
        source_kind="image",
        total_pages=1,
        input_fingerprint="source-revision",
    )

    assert not store.reconcile_orphan_artifact(
        document_id="ocr-unsupported-schema-orphan-doc",
        page_number=1,
        content_hash=source_digest,
        artifact_path=artifact,
        config_fingerprints=(config_fingerprint,),
    )
    assert (
        store.get_page_state(
            document_id="ocr-unsupported-schema-orphan-doc", page_number=1, stage="ocr"
        )
        is None
    )


@pytest.mark.ci
def test_ocr_artifact_disappearing_during_recovery_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = _scratch("ocr_artifact_disappears_during_recovery")
    artifact = scratch / "page-1.json"
    source_digest = "source-image-hash"
    config_fingerprint = "provider/model"
    _write_valid_ocr_artifact(
        artifact,
        source_image_sha256=source_digest,
        ocr_configuration_fingerprint=config_fingerprint,
    )
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    completed_document_id = "ocr-artifact-disappears-completed-doc"
    store.ensure_document(
        document_id=completed_document_id,
        title="OCR Artifact Disappears Completed",
        source_kind="image",
        total_pages=1,
        input_fingerprint="source-revision",
    )
    attempt = store.record_attempt(
        document_id=completed_document_id,
        page_number=1,
        stage="ocr",
        content_hash=source_digest,
        model_name="model",
        artifact_path=artifact,
        config_fingerprint=config_fingerprint,
    )
    assert store.record_page_completed(
        document_id=completed_document_id,
        page_number=1,
        stage="ocr",
        content_hash=source_digest,
        artifact_path=artifact,
        model_name="model",
        attempt_index=attempt,
        config_fingerprint=config_fingerprint,
    )
    orphan_document_id = "ocr-artifact-disappears-orphan-doc"
    store.ensure_document(
        document_id=orphan_document_id,
        title="OCR Artifact Disappears Orphan",
        source_kind="image",
        total_pages=1,
        input_fingerprint="source-revision",
    )

    def _raise_missing_file(_path: Path) -> dict[str, object]:
        raise FileNotFoundError("artifact removed after the existence check")

    def _raise_hash_error(_path: Path) -> str:
        raise FileNotFoundError("artifact removed after the existence check")

    with monkeypatch.context() as patcher:
        patcher.setattr(ocr_state_store_module, "_sha256_file", _raise_hash_error)
        assert not store.should_skip_page(
            document_id=completed_document_id,
            page_number=1,
            stage="ocr",
            content_hash=source_digest,
            artifact_path=artifact,
            config_fingerprint=config_fingerprint,
        )
        assert not store.record_page_completed(
            document_id=completed_document_id,
            page_number=1,
            stage="ocr",
            content_hash=source_digest,
            artifact_path=artifact,
            model_name="model",
            attempt_index=attempt,
            config_fingerprint=config_fingerprint,
        )
        assert not store.reconcile_orphan_artifact(
            document_id=orphan_document_id,
            page_number=1,
            content_hash=source_digest,
            artifact_path=artifact,
            config_fingerprints=(config_fingerprint,),
        )

    monkeypatch.setattr(
        ocr_state_store_module, "_read_json_object", _raise_missing_file
    )
    assert not store.should_skip_page(
        document_id=completed_document_id,
        page_number=1,
        stage="ocr",
        content_hash=source_digest,
        artifact_path=artifact,
        config_fingerprint=config_fingerprint,
    )
    assert not store.record_page_completed(
        document_id=completed_document_id,
        page_number=1,
        stage="ocr",
        content_hash=source_digest,
        artifact_path=artifact,
        model_name="model",
        attempt_index=attempt,
        config_fingerprint=config_fingerprint,
    )
    assert not store.reconcile_orphan_artifact(
        document_id=orphan_document_id,
        page_number=1,
        content_hash=source_digest,
        artifact_path=artifact,
        config_fingerprints=(config_fingerprint,),
    )
