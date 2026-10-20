"""Strict OCR quality rejection and recovery behavior."""

from __future__ import annotations

import json
import os
import subprocess
import sys
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

from kg_doc_parser.models import NonTextCluster, OCRClusterResponse, TextCluster
from kg_doc_parser.workflow_ingest import (
    EmbeddingProviderConfig,
    OCRImagePayload,
    ProviderEndpointConfig,
    WorkflowProviderSettings,
    prepare_ocr_workflow_input,
)
from kg_doc_parser.workflow_ingest.ocr_pipeline import OCRQualityPolicy
from kg_doc_parser.workflow_ingest.ocr_state_store import OCRWorkflowStateStore

pytestmark = [pytest.mark.workflow]

@pytest.mark.ci
def test_ocr_quality_diagnostics_are_persisted_and_strict_policy_fails_closed() -> None:
    scratch = _scratch("ocr_quality_persistence")
    image_path = scratch / "images" / "page_1.png"
    _draw_test_image(image_path, lines=["quality persistence"])
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        embedding=EmbeddingProviderConfig(
            provider="fake", model="fake-embed", dimension=1
        ),
    )

    def _incomplete_runner(
        image_path: Path,
        page_number: int,
        provider_settings: WorkflowProviderSettings,
    ) -> OCRClusterResponse:
        return _fake_ocr_response(page_number, "quality persistence").model_copy(
            update={"incomplete_text": True, "data_loss_likelihood": 0.25}
        )

    accepted = prepare_ocr_workflow_input(
        document_id="ocr-quality-warning-doc",
        title="OCR Quality Warning",
        output_dir=scratch / "warning",
        image_payloads=[OCRImagePayload(page_number=1, image_path=str(image_path))],
        provider_settings=settings,
        ocr_runner=_incomplete_runner,
    )
    persisted = json.loads(
        (accepted.legacy_dir / "page_1.json").read_text(encoding="utf-8")
    )
    metadata = persisted["metadata"]
    assert metadata["ocr_quality_status"] == "flagged"
    assert metadata["ocr_quality_diagnostics"] == [
        {
            "code": "incomplete_text",
            "severity": "warning",
            "message": "provider marked the OCR text as incomplete",
        }
    ]

    with pytest.raises(RuntimeError, match="OCR preparation incomplete"):
        prepare_ocr_workflow_input(
            document_id="ocr-quality-strict-doc",
            title="OCR Quality Strict",
            output_dir=scratch / "strict",
            image_payloads=[OCRImagePayload(page_number=1, image_path=str(image_path))],
            provider_settings=settings,
            ocr_runner=_incomplete_runner,
            quality_policy=OCRQualityPolicy(reject_incomplete_text=True),
        )
    strict_store = OCRWorkflowStateStore(scratch / "strict" / "ocr-state.sqlite")
    strict_state = strict_store.get_page_state(
        document_id="ocr-quality-strict-doc", page_number=1, stage="ocr"
    )
    assert strict_state is not None
    assert strict_state.status == "failed"
    assert not (scratch / "strict" / "legacy" / "page_1.json").exists()

    def _invalid_coordinates_runner(
        image_path: Path,
        page_number: int,
        provider_settings: WorkflowProviderSettings,
    ) -> OCRClusterResponse:
        del image_path, provider_settings
        return _fake_ocr_response(page_number, "invalid coordinates").model_copy(
            update={
                "spatial_accuracy": "region",
                "OCR_text_clusters": [
                    TextCluster(
                        text="invalid coordinates",
                        bb_x_min=10.0,
                        bb_x_max=10.0,
                        bb_y_min=10.0,
                        bb_y_max=30.0,
                        cluster_number=0,
                    )
                ],
            }
        )

    with pytest.raises(RuntimeError, match="OCR preparation incomplete"):
        prepare_ocr_workflow_input(
            document_id="ocr-quality-coordinate-doc",
            title="OCR Coordinate Quality",
            output_dir=scratch / "coordinates",
            image_payloads=[OCRImagePayload(page_number=1, image_path=str(image_path))],
            provider_settings=settings,
            ocr_runner=_invalid_coordinates_runner,
        )
    coordinate_store = OCRWorkflowStateStore(
        scratch / "coordinates" / "ocr-state.sqlite"
    )
    coordinate_state = coordinate_store.get_page_state(
        document_id="ocr-quality-coordinate-doc", page_number=1, stage="ocr"
    )
    assert coordinate_state is not None
    assert coordinate_state.status == "failed"
    assert not (scratch / "coordinates" / "legacy" / "page_1.json").exists()


@pytest.mark.ci
def test_ocr_strict_quality_rejection_recovers_with_valid_provider_result() -> None:
    """A strict rejection must reopen and retry instead of blocking the page."""
    scratch = _scratch("ocr_strict_quality_recovery")
    image_path = scratch / "images" / "page_1.png"
    _draw_test_image(image_path, lines=["strict recovery"])
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        embedding=EmbeddingProviderConfig(
            provider="fake", model="fake-embed", dimension=1
        ),
    )
    policy = OCRQualityPolicy(reject_unresolved_items=True)
    artifact = (
        scratch
        / "legacy_split_pages"
        / "ocr-strict-quality-recovery-doc"
        / "page_1.json"
    )
    calls = 0

    def _recovering_runner(
        image_path: Path,
        page_number: int,
        provider_settings: WorkflowProviderSettings,
    ) -> OCRClusterResponse:
        nonlocal calls
        del image_path, provider_settings
        calls += 1
        if calls == 1:
            return _fake_ocr_response(page_number, "strict recovery").model_copy(
                update={
                    "unresolved_item_count": 1,
                    "source_item_count": 2,
                }
            )
        return _fake_ocr_response(page_number, "strict recovery").model_copy(
            update={
                "meaningful_ordering": [0, 1],
                "contains_table": True,
                "non_text_objects": [
                    NonTextCluster(
                        description="figure: recovered chart",
                        bb_x_min=520.0,
                        bb_x_max=900.0,
                        bb_y_min=100.0,
                        bb_y_max=420.0,
                        cluster_number=1,
                    )
                ],
                "source_item_count": 2,
            }
        )

    with pytest.raises(RuntimeError, match="OCR preparation incomplete"):
        prepare_ocr_workflow_input(
            document_id="ocr-strict-quality-recovery-doc",
            title="OCR Strict Quality Recovery",
            output_dir=scratch,
            image_payloads=[OCRImagePayload(page_number=1, image_path=str(image_path))],
            provider_settings=settings,
            ocr_runner=_recovering_runner,
            quality_policy=policy,
        )
    assert not artifact.exists()
    failed_store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    failed_state = failed_store.get_page_state(
        document_id="ocr-strict-quality-recovery-doc", page_number=1, stage="ocr"
    )
    assert failed_state is not None
    assert failed_state.status == "failed"

    prepare_ocr_workflow_input(
        document_id="ocr-strict-quality-recovery-doc",
        title="OCR Strict Quality Recovery",
        output_dir=scratch,
        image_payloads=[OCRImagePayload(page_number=1, image_path=str(image_path))],
        provider_settings=settings,
        ocr_runner=_recovering_runner,
        quality_policy=policy,
    )

    assert calls == 2
    assert artifact.is_file()
    persisted = json.loads(artifact.read_text(encoding="utf-8"))
    assert len(persisted["non_text_objects"]) == 1
    assert persisted["non_text_objects"][0]["description"] == "figure: recovered chart"
    assert persisted["source_item_count"] == 2
    assert persisted["unresolved_item_count"] == 0
    recovered_store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    recovered_state = recovered_store.get_page_state(
        document_id="ocr-strict-quality-recovery-doc", page_number=1, stage="ocr"
    )
    assert recovered_state is not None
    assert recovered_state.status == "completed"


@pytest.mark.ci
def test_strict_quality_rejection_recovers_in_a_fresh_process() -> None:
    """A persisted rejected attempt is retried, not accepted, after restart."""
    scratch = _scratch("ocr_strict_quality_fresh_process")
    _draw_test_image(scratch / "source" / "page-1.png", lines=["strict restart"])
    worker = Path(__file__).with_name("_ocr_strict_quality_fresh_process_worker.py")
    repo_root = Path(__file__).resolve().parents[1]
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = os.pathsep.join(
        path
        for path in (
            str(repo_root),
            str(repo_root.parent / "kogwistar"),
            child_env.get("PYTHONPATH", ""),
        )
        if path
    )

    results: dict[str, dict[str, object]] = {}
    for mode in ("reject", "resume"):
        process = subprocess.run(
            [sys.executable, str(worker), str(scratch), mode],
            env=child_env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert process.returncode == 0, process.stdout + process.stderr
        results[mode] = json.loads(
            (scratch / f"{mode}-result.json").read_text(encoding="utf-8")
        )
        if mode == "reject":
            assert results[mode]["calls"] == 1
            assert results[mode]["status"] == "rejected"
            assert results[mode]["page_status"] == "failed"
            rejected_artifact = Path(str(results[mode]["artifact_path"]))
            assert not rejected_artifact.exists()

    assert results["resume"]["calls"] == 1
    assert results["resume"]["status"] == "completed"
    assert results["resume"]["page_status"] == "completed"
    accepted_artifact = Path(str(results["resume"]["artifact_path"]))
    assert accepted_artifact.is_file()
    persisted = json.loads(accepted_artifact.read_text(encoding="utf-8"))
    assert persisted["source_item_count"] == 2
    assert persisted["unresolved_item_count"] == 0
    assert persisted["non_text_objects"][0]["description"] == (
        "figure: recovered diagram"
    )
