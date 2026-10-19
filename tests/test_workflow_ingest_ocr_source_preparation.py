"""OCR input preparation and source/configuration identity tests."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from ocr_workflow_support import (
    draw_test_image as _draw_test_image,
)
from ocr_workflow_support import (
    fake_ocr_response as _fake_ocr_response,
)
from ocr_workflow_support import (
    scratch_path as _scratch,
)
from ocr_workflow_support import (
    write_two_page_pdf as _write_two_page_pdf,
)
from ocr_workflow_support import (
    write_valid_ocr_artifact as _write_valid_ocr_artifact,
)
from PIL import Image

from kg_doc_parser.models import OCRClusterResponse
from kg_doc_parser.ocr import regen_doc
from kg_doc_parser.workflow_ingest import (
    EmbeddingProviderConfig,
    OCRImagePayload,
    ProviderEndpointConfig,
    WorkflowProviderSettings,
    prepare_ocr_workflow_input,
)
from kg_doc_parser.workflow_ingest.ocr_pipeline import (
    OCRQualityPolicy,
    _compute_input_fingerprint,
    _ocr_configuration_fingerprint,
    _resolve_ocr_source_plan,
)
from kg_doc_parser.workflow_ingest.ocr_state_store import OCRWorkflowStateStore

pytestmark = [pytest.mark.workflow]

@pytest.mark.ci
def test_ocr_completion_requires_exact_expected_page_numbers() -> None:
    scratch = _scratch("ocr_completion_page_set")
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="page-set-doc",
        title="Page Set",
        source_kind="image",
        total_pages=1,
        input_fingerprint="fingerprint",
    )
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            """
            INSERT INTO page_state (
                document_id, page_number, stage, attempt_count, status,
                content_hash, artifact_path, last_error, last_model, last_attempted_ts
            ) VALUES (?, ?, 'ocr', 1, 'completed', ?, NULL, NULL, NULL, ?)
            """,
            ("page-set-doc", 999, "hash", 0.0),
        )

    assert not store.refresh_document_completion(document_id="page-set-doc")
    assert not store.read_document_completed(document_id="page-set-doc")


@pytest.mark.ci
def test_prepare_ocr_workflow_input_writes_legacy_compatible_artifacts() -> None:
    scratch = _scratch("ocr_legacy_compat")
    image_path = scratch / "images" / "page_1.png"
    _draw_test_image(image_path, lines=["Compatibility page"])
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        embedding=EmbeddingProviderConfig(
            provider="fake", model="fake-embed", dimension=1
        ),
    )
    artifacts = prepare_ocr_workflow_input(
        document_id="ocr-compat-doc",
        title="OCR Compatibility Doc",
        output_dir=scratch / "artifacts",
        image_payloads=[OCRImagePayload(page_number=1, image_path=str(image_path))],
        provider_settings=settings,
        ocr_runner=lambda image_path, page_number, provider_settings: (
            _fake_ocr_response(page_number, "Compatibility page")
        ),
    )

    regenerated = regen_doc(str(artifacts.legacy_dir), use_raw=True)

    assert artifacts.legacy_dir.exists()
    assert len(regenerated) == 1
    assert regenerated[0]["OCR_text_clusters"][0]["text"] == "Compatibility page"
    assert (
        artifacts.workflow_input.collections[0].pages[0].units[0].text
        == "Compatibility page"
    )
    persisted_page = json.loads(
        (artifacts.legacy_dir / "page_1.json").read_text(encoding="utf-8")
    )
    assert persisted_page["metadata"]["ocr_provider"] == "fake"
    assert len(persisted_page["metadata"]["ocr_configuration_fingerprint"]) == 64
    assert artifacts.document_tree is not None
    assert artifacts.document_tree.source_items[0].ocr_metadata is not None


@pytest.mark.ci
def test_prepare_ocr_workflow_input_accepts_pdf_with_injected_rasterizer() -> None:
    scratch = _scratch("ocr_pdf")
    pdf_path = scratch / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
    rendered_image = scratch / "rendered" / "page_1.png"
    _draw_test_image(rendered_image, lines=["Injected raster page"])
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        embedding=EmbeddingProviderConfig(
            provider="fake", model="fake-embed", dimension=1
        ),
    )

    def _rasterizer(pdf_path: Path, rendered_dir: Path) -> list[Path]:
        rendered_dir.mkdir(parents=True, exist_ok=True)
        target = rendered_dir / "page_1.png"
        shutil_path = rendered_image
        if shutil_path.resolve() != target.resolve():
            import shutil

            shutil.copy2(shutil_path, target)
        return [target]

    artifacts = prepare_ocr_workflow_input(
        document_id="ocr-pdf-doc",
        title="OCR PDF Doc",
        output_dir=scratch / "artifacts",
        pdf_path=pdf_path,
        provider_settings=settings,
        ocr_runner=lambda image_path, page_number, provider_settings: (
            _fake_ocr_response(page_number, "Injected raster page")
        ),
        pdf_rasterizer=_rasterizer,
        quality_policy=OCRQualityPolicy(mode="permissive"),
    )

    assert len(artifacts.ocr_pages) == 1
    assert (
        artifacts.workflow_input.collections[0].pages[0].units[0].text
        == "Injected raster page"
    )
@pytest.mark.ci
def test_prepare_ocr_workflow_input_retries_failed_pages_and_tracks_attempts() -> None:
    scratch = _scratch("ocr_retry")
    images_dir = scratch / "images"
    image_paths = [images_dir / "page_1.png", images_dir / "page_2.png"]
    _draw_test_image(image_paths[0], lines=["Retry test page one"])
    _draw_test_image(image_paths[1], lines=["Retry test page two"])
    payloads = [
        OCRImagePayload(page_number=1, image_path=str(image_paths[0])),
        OCRImagePayload(page_number=2, image_path=str(image_paths[1])),
    ]
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        embedding=EmbeddingProviderConfig(
            provider="fake", model="fake-embed", dimension=1
        ),
    )
    calls = {"page1": 0, "page2": 0}

    def _runner(
        image_path: Path, page_number: int, provider_settings: WorkflowProviderSettings
    ):
        if page_number == 1:
            calls["page1"] += 1
            if calls["page1"] == 1:
                raise RuntimeError("temporary ocr failure")
        else:
            calls["page2"] += 1
        return _fake_ocr_response(page_number, f"Retry page {page_number}")

    with pytest.raises(RuntimeError, match="OCR preparation incomplete"):
        prepare_ocr_workflow_input(
            document_id="ocr-retry-doc",
            title="OCR Retry Doc",
            output_dir=scratch / "artifacts",
            image_payloads=payloads,
            provider_settings=settings,
            ocr_runner=_runner,
            quality_policy=OCRQualityPolicy(mode="permissive"),
        )

    second = prepare_ocr_workflow_input(
        document_id="ocr-retry-doc",
        title="OCR Retry Doc",
        output_dir=scratch / "artifacts",
        image_payloads=payloads,
        provider_settings=settings,
        ocr_runner=_runner,
        quality_policy=OCRQualityPolicy(mode="permissive"),
    )

    with sqlite3.connect(second.state_db_path) as conn:
        row = conn.execute(
            """
            SELECT attempt_count, status
            FROM page_state
            WHERE document_id = ? AND page_number = ? AND stage = 'ocr'
            """,
            ("ocr-retry-doc", 1),
        ).fetchone()
    assert row is not None
    assert int(row[0]) == 2
    assert row[1] == "completed"
    assert second.reused_pages == [2]


@pytest.mark.ci
def test_prepare_ocr_workflow_input_exhausts_candidate_models_until_success() -> None:
    scratch = _scratch("ocr_candidates")
    image_path = scratch / "images" / "page_1.png"
    _draw_test_image(image_path, lines=["Candidate page"])
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="primary-model"),
        embedding=EmbeddingProviderConfig(
            provider="fake", model="fake-embed", dimension=1
        ),
    )

    def _runner(
        image_path: Path, page_number: int, provider_settings: WorkflowProviderSettings
    ):
        if provider_settings.ocr.model == "primary-model":
            raise RuntimeError("primary failed")
        return _fake_ocr_response(page_number, f"winner:{provider_settings.ocr.model}")

    artifacts = prepare_ocr_workflow_input(
        document_id="ocr-candidates-doc",
        title="OCR Candidate Doc",
        output_dir=scratch / "artifacts",
        image_payloads=[OCRImagePayload(page_number=1, image_path=str(image_path))],
        provider_settings=settings,
        ocr_runner=_runner,
        ocr_candidate_models=["primary-model", "backup-model"],
    )

    with sqlite3.connect(artifacts.state_db_path) as conn:
        rows = conn.execute(
            """
            SELECT model_name, status
            FROM model_attempts
            WHERE document_id = ? AND page_number = ? AND stage = 'ocr'
            ORDER BY attempt_index
            """,
            ("ocr-candidates-doc", 1),
        ).fetchall()
    assert [tuple(row) for row in rows] == [
        ("primary-model", "failed"),
        ("backup-model", "completed"),
    ]
    assert (
        artifacts.workflow_input.collections[0].pages[0].units[0].text
        == "winner:backup-model"
    )


@pytest.mark.ci
def test_strict_empty_ocr_result_uses_next_candidate_model() -> None:
    scratch = _scratch("ocr_strict_empty_fallback")
    image_path = scratch / "images" / "page_1.png"
    _draw_test_image(image_path, lines=["Visible content omitted by primary"])
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="primary-model"),
        embedding=EmbeddingProviderConfig(
            provider="fake", model="fake-embed", dimension=1
        ),
    )
    empty = _fake_ocr_response(1, "placeholder").model_copy(
        update={
            "OCR_text_clusters": [],
            "non_text_objects": [],
            "is_empty_page": True,
            "meaningful_ordering": [],
            "source_item_count": 0,
        }
    )
    calls: list[str] = []

    def runner(
        image_path: Path, page_number: int, provider_settings: WorkflowProviderSettings
    ) -> OCRClusterResponse:
        del image_path
        calls.append(provider_settings.ocr.model)
        if provider_settings.ocr.model == "primary-model":
            return empty.model_copy(update={"printed_page_number": str(page_number)})
        backup = _fake_ocr_response(page_number, "backup recognized the page")
        return backup.model_copy(
            update={
                "OCR_text_clusters": [
                    backup.OCR_text_clusters[0].model_copy(
                        update={
                            "bb_x_min": 0,
                            "bb_x_max": 1200,
                            "bb_y_min": 0,
                            "bb_y_max": 800,
                        }
                    )
                ]
            }
        )

    artifacts = prepare_ocr_workflow_input(
        document_id="strict-empty-fallback-doc",
        title="Strict Empty Fallback",
        output_dir=scratch / "artifacts",
        image_payloads=[OCRImagePayload(page_number=1, image_path=str(image_path))],
        provider_settings=settings,
        ocr_runner=runner,
        ocr_candidate_models=["primary-model", "backup-model"],
    )

    assert calls == ["primary-model", "backup-model"]
    assert artifacts.workflow_input.collections[0].pages[0].units[0].text == (
        "backup recognized the page"
    )


@pytest.mark.ci
def test_resolve_ocr_source_plan_uses_document_fingerprint_for_pdf_source() -> None:
    scratch = _scratch("ocr_fingerprint_plan")
    pdf_path = scratch / "sample.pdf"
    page_1 = scratch / "page_1.png"
    page_2 = scratch / "page_2.png"
    _draw_test_image(page_1, lines=["Fingerprint page one"])
    _draw_test_image(page_2, lines=["Fingerprint page two"])
    _write_two_page_pdf([page_1, page_2], pdf_path)

    output_dir = scratch / "artifacts"
    plan = _resolve_ocr_source_plan(
        document_id="ocr-fingerprint-doc",
        output_dir=output_dir,
        image_payloads=None,
        pdf_path=pdf_path,
        pdf_rasterizer=lambda source_pdf, rendered_dir: [page_1, page_2],
    )

    assert plan.source_kind == "pdf"
    assert plan.pdf_source == pdf_path
    assert plan.input_fingerprint == _compute_input_fingerprint(pdf_path=pdf_path)
    assert plan.input_fingerprint != _compute_input_fingerprint(
        page_sources=[(1, page_1)]
    )


@pytest.mark.ci
def test_pdf_artifact_directories_do_not_collide_for_same_basename() -> None:
    scratch = _scratch("ocr_same_basename")
    first_pdf = scratch / "first" / "report.pdf"
    second_pdf = scratch / "second" / "report.pdf"
    first_pdf.parent.mkdir(parents=True)
    second_pdf.parent.mkdir(parents=True)
    first_pdf.write_bytes(b"%PDF-1.4\nfirst source\n")
    second_pdf.write_bytes(b"%PDF-1.4\nsecond source\n")

    def _rasterizer(source_pdf: Path, rendered_dir: Path) -> list[Path]:
        rendered_dir.mkdir(parents=True, exist_ok=True)
        page = rendered_dir / "page_1.png"
        color = "red" if b"first source" in source_pdf.read_bytes() else "blue"
        Image.new("RGB", (11, 7), color).save(page, format="PNG")
        return [page]

    output_dir = scratch / "artifacts"
    first = _resolve_ocr_source_plan(
        document_id="same-document",
        output_dir=output_dir,
        image_payloads=None,
        pdf_path=first_pdf,
        pdf_rasterizer=_rasterizer,
    )
    second = _resolve_ocr_source_plan(
        document_id="same-document",
        output_dir=output_dir,
        image_payloads=None,
        pdf_path=second_pdf,
        pdf_rasterizer=_rasterizer,
    )

    assert first.rendered_dir != second.rendered_dir
    assert first.legacy_dir != second.legacy_dir
    assert first.rendered_dir.name.startswith("same-document-report-")
    assert second.rendered_dir.name.startswith("same-document-report-")
    assert (
        first.page_sources[0][1].read_bytes() != second.page_sources[0][1].read_bytes()
    )


@pytest.mark.ci
def test_pdf_artifact_directories_share_content_identity_across_paths() -> None:
    scratch = _scratch("ocr_same_content_cross_path")
    first_pdf = scratch / "first" / "report.pdf"
    second_pdf = scratch / "second" / "report.pdf"
    first_pdf.parent.mkdir(parents=True)
    second_pdf.parent.mkdir(parents=True)
    payload = b"%PDF-1.4\nshared source\n"
    first_pdf.write_bytes(payload)
    second_pdf.write_bytes(payload)

    def _rasterizer(_source: Path, rendered_dir: Path) -> list[Path]:
        rendered_dir.mkdir(parents=True, exist_ok=True)
        page = rendered_dir / "page_1.png"
        Image.new("RGB", (11, 7), "green").save(page, format="PNG")
        return [page]

    output_dir = scratch / "artifacts"
    first = _resolve_ocr_source_plan(
        document_id="shared-content-document",
        output_dir=output_dir,
        image_payloads=None,
        pdf_path=first_pdf,
        pdf_rasterizer=_rasterizer,
    )
    second = _resolve_ocr_source_plan(
        document_id="shared-content-document",
        output_dir=output_dir,
        image_payloads=None,
        pdf_path=second_pdf,
        pdf_rasterizer=_rasterizer,
    )

    assert first.rendered_dir == second.rendered_dir
    assert first.legacy_dir == second.legacy_dir
    assert first.input_fingerprint == second.input_fingerprint


@pytest.mark.ci
def test_prepare_ocr_workflow_input_keeps_document_fingerprint_stable_in_state_db() -> (
    None
):
    scratch = _scratch("ocr_fingerprint_state")
    page_1 = scratch / "images" / "page_1.png"
    page_2 = scratch / "images" / "page_2.png"
    _draw_test_image(page_1, lines=["Fingerprint state page one"])
    _draw_test_image(page_2, lines=["Fingerprint state page two"])
    payloads = [
        OCRImagePayload(page_number=1, image_path=str(page_1)),
        OCRImagePayload(page_number=2, image_path=str(page_2)),
    ]
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        embedding=EmbeddingProviderConfig(
            provider="fake", model="fake-embed", dimension=1
        ),
    )

    def _runner(
        image_path: Path, page_number: int, provider_settings: WorkflowProviderSettings
    ):
        return _fake_ocr_response(page_number, f"Fingerprint page {page_number}")

    artifacts = prepare_ocr_workflow_input(
        document_id="ocr-fingerprint-doc",
        title="OCR Fingerprint Doc",
        output_dir=scratch / "artifacts",
        image_payloads=payloads,
        provider_settings=settings,
        ocr_runner=_runner,
        quality_policy=OCRQualityPolicy(mode="permissive"),
    )

    expected_fingerprint = _compute_input_fingerprint(
        page_sources=[(1, page_1), (2, page_2)]
    )
    with sqlite3.connect(artifacts.state_db_path) as conn:
        row = conn.execute(
            """
            SELECT input_fingerprint, total_pages, source_kind, is_completed
            FROM document_state
            WHERE document_id = ?
            """,
            ("ocr-fingerprint-doc",),
        ).fetchone()
    assert row is not None
    assert row[0] == expected_fingerprint
    assert int(row[1]) == 2
    assert row[2] == "image"
    assert int(row[3]) == 1

    second = prepare_ocr_workflow_input(
        document_id="ocr-fingerprint-doc",
        title="OCR Fingerprint Doc",
        output_dir=scratch / "artifacts",
        image_payloads=payloads,
        provider_settings=settings,
        ocr_runner=_runner,
        quality_policy=OCRQualityPolicy(mode="permissive"),
    )
    assert second.reused_pages == [1, 2]
    with sqlite3.connect(second.state_db_path) as conn:
        row2 = conn.execute(
            """
            SELECT input_fingerprint
            FROM document_state
            WHERE document_id = ?
            """,
            ("ocr-fingerprint-doc",),
        ).fetchone()
    assert row2 is not None
    assert row2[0] == expected_fingerprint


@pytest.mark.ci
def test_ocr_state_fingerprint_change_discards_stale_page_and_model_state() -> None:
    scratch = _scratch("ocr_fingerprint_invalidation")
    artifact = scratch / "page-1.json"
    _write_valid_ocr_artifact(artifact)
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="fingerprint-invalidation-doc",
        title="Fingerprint Invalidation",
        source_kind="image",
        total_pages=1,
        input_fingerprint="fingerprint-a",
    )
    attempt = store.record_attempt(
        document_id="fingerprint-invalidation-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-a",
        model_name="model-a",
        artifact_path=artifact,
    )
    store.record_page_completed(
        document_id="fingerprint-invalidation-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-a",
        artifact_path=artifact,
        model_name="model-a",
        attempt_index=attempt,
    )
    assert (
        store.get_page_state(
            document_id="fingerprint-invalidation-doc", page_number=1, stage="ocr"
        )
        is not None
    )
    assert store.list_model_attempts(
        document_id="fingerprint-invalidation-doc", page_number=1
    )

    store.ensure_document(
        document_id="fingerprint-invalidation-doc",
        title="Fingerprint Invalidation",
        source_kind="image",
        total_pages=1,
        input_fingerprint="fingerprint-b",
    )

    assert (
        store.get_page_state(
            document_id="fingerprint-invalidation-doc", page_number=1, stage="ocr"
        )
        is None
    )
    assert (
        store.list_model_attempts(
            document_id="fingerprint-invalidation-doc", page_number=1
        )
        == []
    )
    assert not store.read_document_completed(document_id="fingerprint-invalidation-doc")


@pytest.mark.ci
def test_ocr_render_reuse_requires_render_configuration_identity() -> None:
    scratch = _scratch("ocr_render_config_identity")
    artifact = scratch / "page-1.png"
    artifact.write_bytes(b"rendered-page")
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="render-config-doc",
        title="Render Config",
        source_kind="pdf",
        total_pages=1,
        input_fingerprint="pdf-fingerprint",
    )
    attempt = store.record_attempt(
        document_id="render-config-doc",
        page_number=1,
        stage="render",
        content_hash="source-image",
        model_name="rasterizer-a",
        artifact_path=artifact,
        config_fingerprint="renderer-a",
    )
    store.record_page_completed(
        document_id="render-config-doc",
        page_number=1,
        stage="render",
        content_hash="source-image",
        artifact_path=artifact,
        model_name="rasterizer-a",
        attempt_index=attempt,
        config_fingerprint="renderer-a",
    )

    assert store.should_skip_page(
        document_id="render-config-doc",
        page_number=1,
        stage="render",
        content_hash="source-image",
        artifact_path=artifact,
        config_fingerprint="renderer-a",
    )
    assert not store.should_skip_page(
        document_id="render-config-doc",
        page_number=1,
        stage="render",
        content_hash="source-image",
        artifact_path=artifact,
        config_fingerprint="renderer-b",
    )


@pytest.mark.ci
def test_ocr_configuration_fingerprint_covers_provider_output_settings() -> None:
    scratch = _scratch("ocr_output_fingerprint_identity")
    base = ProviderEndpointConfig(
        provider="openai",
        model="vision-model",
        api_key_env="OCR_API_KEY",
        base_url="https://ocr.example.test/v1",
        temperature=0.0,
        max_output_tokens=1024,
    )
    settings = WorkflowProviderSettings(ocr=base)
    original = _ocr_configuration_fingerprint(settings)
    variants = (
        base.model_copy(update={"temperature": 0.2}),
        base.model_copy(update={"reasoning_effort": "high"}),
        base.model_copy(update={"location": "region-b"}),
        base.model_copy(update={"max_output_tokens": 2048}),
        base.model_copy(update={"ocr_engine": "easyocr"}),
        base.model_copy(update={"ocr_language": "fr"}),
        base.model_copy(update={"max_retries": base.max_retries + 1}),
        base.model_copy(update={"timeout_seconds": base.timeout_seconds + 1}),
        base.model_copy(
            update={
                "fallback_specs": [
                    ProviderEndpointConfig(
                        provider="openai",
                        model="fallback-vision-model",
                        api_key_env="OCR_API_KEY",
                    )
                ]
            }
        ),
    )
    changed_fingerprints = tuple(
        _ocr_configuration_fingerprint(settings.model_copy(update={"ocr": variant}))
        for variant in variants
    )

    assert all(fingerprint != original for fingerprint in changed_fingerprints)

    image_digest = "a" * 64
    artifact = scratch / "page-1.json"
    _write_valid_ocr_artifact(
        artifact,
        source_image_sha256=image_digest,
        ocr_configuration_fingerprint=original,
    )
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="provider-fingerprint-doc",
        title="Provider Fingerprint",
        source_kind="image",
        total_pages=1,
        input_fingerprint="stable-source",
    )
    attempt = store.record_attempt(
        document_id="provider-fingerprint-doc",
        page_number=1,
        stage="ocr",
        content_hash=image_digest,
        model_name="vision-model",
        artifact_path=artifact,
        config_fingerprint=original,
    )
    store.record_page_completed(
        document_id="provider-fingerprint-doc",
        page_number=1,
        stage="ocr",
        content_hash=image_digest,
        artifact_path=artifact,
        model_name="vision-model",
        attempt_index=attempt,
        config_fingerprint=original,
    )

    assert store.should_skip_page(
        document_id="provider-fingerprint-doc",
        page_number=1,
        stage="ocr",
        content_hash=image_digest,
        artifact_path=artifact,
        config_fingerprint=original,
    )
    assert all(
        not store.should_skip_page(
            document_id="provider-fingerprint-doc",
            page_number=1,
            stage="ocr",
            content_hash=image_digest,
            artifact_path=artifact,
            config_fingerprint=fingerprint,
        )
        for fingerprint in changed_fingerprints
    )



