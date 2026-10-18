from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import pytest

from kg_doc_parser.models import NonTextCluster, OCRClusterResponse, TextCluster
from kg_doc_parser.workflow_ingest.ocr_quality import (
    OCRGroundTruth,
    OCRGroundTruthItem,
    evaluate_ocr_ground_truth,
)


def _response(*, include_beta: bool = True, beta_text: str = "beta") -> OCRClusterResponse:
    clusters = [
        TextCluster(
            text="alpha",
            bb_x_min=0.0,
            bb_x_max=50.0,
            bb_y_min=0.0,
            bb_y_max=20.0,
            cluster_number=1,
        )
    ]
    if include_beta:
        clusters.append(
            TextCluster(
                text=beta_text,
                bb_x_min=50.0,
                bb_x_max=100.0,
                bb_y_min=0.0,
                bb_y_max=20.0,
                cluster_number=2,
            )
        )
    return OCRClusterResponse(
        OCR_text_clusters=clusters,
        non_text_objects=[],
        is_empty_page=False,
        printed_page_number=None,
        meaningful_ordering=[cluster.cluster_number for cluster in clusters],
        page_x_min=0.0,
        page_x_max=100.0,
        page_y_min=0.0,
        page_y_max=20.0,
        estimated_rotation_degrees=0.0,
        incomplete_words_on_edge=False,
        incomplete_text=False,
        data_loss_likelihood=0.0,
        scan_quality="high",
        contains_table=False,
        source_item_count=len(clusters),
    )


def _ground_truth() -> OCRGroundTruth:
    return OCRGroundTruth(
        items=(
            OCRGroundTruthItem(
                item_id="alpha",
                modality="text",
                x_min=0.0,
                y_min=0.0,
                x_max=50.0,
                y_max=20.0,
                text="alpha",
            ),
            OCRGroundTruthItem(
                item_id="beta",
                modality="text",
                x_min=50.0,
                y_min=0.0,
                x_max=100.0,
                y_max=20.0,
                text="beta",
            ),
        )
    )


@pytest.mark.parametrize(
    ("ordering", "diagnostic"),
    [
        ([1, 1], "missing_cluster_ids=[2]"),
        ([1, 2, 9], "unknown_cluster_ids=[9]"),
        ([1, 2, 2], "duplicate_ordering_ids=[2]"),
    ],
)
def test_invalid_reading_order_reports_retry_action_details(
    ordering: list[int], diagnostic: str
) -> None:
    payload = _response().model_dump(dump_format="python")
    payload["meaningful_ordering"] = ordering

    with pytest.raises(ValueError, match=diagnostic.replace("[", r"\[").replace("]", r"\]")):
        OCRClusterResponse.model_validate(payload)


def test_ground_truth_report_detects_recall_and_text_loss_independently() -> None:
    report = evaluate_ocr_ground_truth(
        _response(include_beta=False),
        _ground_truth(),
    )

    assert report.item_recall == pytest.approx(0.5)
    assert report.item_recall_by_modality == {
        "text": 0.5,
        "non_text": None,
    }
    assert report.missing_item_ids == ("beta",)
    assert report.text_mismatch_item_ids == ()


def test_ground_truth_report_detects_spatial_and_transcription_errors() -> None:
    response = _response(beta_text="gamma")
    response.OCR_text_clusters[1].bb_x_min = 75.0
    report = evaluate_ocr_ground_truth(response, _ground_truth(), minimum_iou=0.2)

    assert report.item_recall == pytest.approx(1.0)
    assert report.mean_spatial_iou < 1.0
    assert report.text_mismatch_item_ids == ("beta",)


def test_zero_iou_threshold_does_not_match_spatially_disjoint_items() -> None:
    response = _response(include_beta=False)
    response.OCR_text_clusters[0].bb_x_min = 200.0
    response.OCR_text_clusters[0].bb_x_max = 250.0

    report = evaluate_ocr_ground_truth(response, _ground_truth(), minimum_iou=0.0)

    assert report.item_recall == pytest.approx(0.0)
    assert report.missing_item_ids == ("alpha", "beta")


@pytest.mark.parametrize(
    ("x_min", "x_max"),
    [(float("nan"), 50.0), (10.0, 10.0)],
)
def test_ground_truth_evaluator_rejects_invalid_provider_boxes(
    x_min: float, x_max: float
) -> None:
    response = _response(include_beta=False)
    response.OCR_text_clusters[0].bb_x_min = x_min
    response.OCR_text_clusters[0].bb_x_max = x_max

    with pytest.raises(ValueError, match="finite, positive-area bounds"):
        evaluate_ocr_ground_truth(response, _ground_truth())


def test_ground_truth_matching_can_reassign_an_overlapping_candidate() -> None:
    """A valid one-to-one assignment must not depend on greedy item order."""
    ground_truth = OCRGroundTruth(
        items=(
            OCRGroundTruthItem(
                item_id="whole-line",
                modality="text",
                x_min=0.0,
                y_min=0.0,
                x_max=100.0,
                y_max=20.0,
                text="beta",
            ),
            OCRGroundTruthItem(
                item_id="alpha-line",
                modality="text",
                x_min=0.0,
                y_min=0.0,
                x_max=50.0,
                y_max=20.0,
                text="alpha",
            ),
        ),
    )

    report = evaluate_ocr_ground_truth(_response(), ground_truth)

    assert report.item_recall == pytest.approx(1.0)
    assert report.missing_item_ids == ()


def test_ground_truth_report_matches_non_text_modalities_without_text() -> None:
    response = _response(include_beta=False).model_copy(
        update={
            "non_text_objects": [
                NonTextCluster(
                    description="table",
                    bb_x_min=50.0,
                    bb_x_max=100.0,
                    bb_y_min=0.0,
                    bb_y_max=20.0,
                    cluster_number=3,
                )
            ],
            "meaningful_ordering": [1, 3],
            "source_item_count": 2,
        }
    )
    ground_truth = OCRGroundTruth(
        items=(
            _ground_truth().items[0],
            OCRGroundTruthItem(
                item_id="table",
                modality="non_text",
                x_min=50.0,
                y_min=0.0,
                x_max=100.0,
                y_max=20.0,
            ),
        )
    )

    report = evaluate_ocr_ground_truth(response, ground_truth)

    assert report.item_recall == pytest.approx(1.0)
    assert report.mean_spatial_iou == pytest.approx(1.0)
    assert report.text_mismatch_item_ids == ()


def test_ground_truth_report_exposes_missing_non_text_recall_separately() -> None:
    ground_truth = OCRGroundTruth(
        items=(
            *_ground_truth().items,
            OCRGroundTruthItem(
                item_id="figure",
                modality="non_text",
                x_min=25.0,
                y_min=0.0,
                x_max=75.0,
                y_max=20.0,
            ),
        )
    )

    report = evaluate_ocr_ground_truth(_response(), ground_truth)

    assert report.item_recall == pytest.approx(2 / 3)
    assert report.item_recall_by_modality == {
        "text": 1.0,
        "non_text": 0.0,
    }
    assert report.missing_item_ids == ("figure",)


def test_ground_truth_contract_rejects_invalid_annotations() -> None:
    with pytest.raises(ValueError, match="positive area"):
        OCRGroundTruthItem(
            item_id="invalid",
            modality="non_text",
            x_min=1.0,
            y_min=1.0,
            x_max=1.0,
            y_max=2.0,
        )
    with pytest.raises(ValueError, match="finite"):
        OCRGroundTruthItem(
            item_id="infinite",
            modality="non_text",
            x_min=0.0,
            y_min=0.0,
            x_max=float("inf"),
            y_max=1.0,
        )


def test_recorded_codex_layout_fixture_has_independent_spatial_coverage() -> None:
    fixture_dir = Path(__file__).parent / "fixtures" / "ocr_ground_truth"
    source_image = fixture_dir / "codex_ocr_layout_page.png"
    image_bytes = source_image.read_bytes()
    assert hashlib.sha256(image_bytes).hexdigest() == (
        "5f55174cf5cfddd227c5833c07de0c56563fd5850817f7ae518f25cbceecfc3c"
    )
    assert image_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", image_bytes[16:24]) == (1600, 1200)

    response = OCRClusterResponse.model_validate_json(
        (fixture_dir / "codex_ocr_layout_page_response.json").read_text(
            encoding="utf-8"
        )
    )
    ground_truth = OCRGroundTruth.model_validate_json(
        (fixture_dir / "codex_ocr_layout_page_ground_truth.json").read_text(
            encoding="utf-8"
        )
    )

    report = evaluate_ocr_ground_truth(response, ground_truth)

    assert len(ground_truth.items) == 17
    assert report.item_recall == pytest.approx(16 / 17)
    assert report.item_recall_by_modality == {"text": 1.0, "non_text": 0.0}
    assert report.missing_item_ids == ("table-region",)
    assert report.mean_spatial_iou >= report.minimum_iou
    assert report.text_mismatch_item_ids == ()
    assert report.extra_cluster_numbers == ()


@pytest.mark.parametrize(
    "response_name",
    [
        "docling_easyocr_layout_page_response.json",
        "docling_rapidocr_layout_page_response.json",
    ],
)
def test_recorded_docling_full_page_fallback_does_not_claim_region_grounding(
    response_name: str,
) -> None:
    fixture_dir = Path(__file__).parent / "fixtures" / "ocr_ground_truth"
    ground_truth = OCRGroundTruth.model_validate_json(
        (fixture_dir / "codex_ocr_layout_page_ground_truth.json").read_text(
            encoding="utf-8"
        )
    )
    response = OCRClusterResponse.model_validate_json(
        (fixture_dir / response_name).read_text(encoding="utf-8")
    )

    report = evaluate_ocr_ground_truth(response, ground_truth)

    assert report.expected_item_count == 17
    assert report.matched_item_count == 0
    assert report.item_recall == 0.0
    assert report.item_recall_by_modality == {"text": 0.0, "non_text": 0.0}
    assert report.missing_item_ids == tuple(item.item_id for item in ground_truth.items)
    assert report.extra_cluster_numbers == (0,)


@pytest.mark.ci
def test_recorded_live_docling_provenance_exposes_text_loss_and_region_recall() -> None:
    fixture_dir = Path(__file__).parent / "fixtures" / "ocr_ground_truth"
    ground_truth = OCRGroundTruth.model_validate_json(
        (fixture_dir / "codex_ocr_layout_page_ground_truth.json").read_text(
            encoding="utf-8"
        )
    )
    response = OCRClusterResponse.model_validate_json(
        (fixture_dir / "docling_rapidocr_provenance_response.json").read_text(
            encoding="utf-8"
        )
    )

    report = evaluate_ocr_ground_truth(response, ground_truth)

    assert report.matched_item_count == 5
    assert report.item_recall == pytest.approx(5 / 17)
    assert report.item_recall_by_modality == {"text": 0.25, "non_text": 1.0}
    assert report.missing_item_ids == (
        "header-name",
        "header-value",
        "header-status",
        "alpha",
        "42",
        "ready",
        "beta",
        "17",
        "review",
        "gamma",
        "8",
        "done",
    )
    assert report.text_mismatch_item_ids == ("list-1",)
