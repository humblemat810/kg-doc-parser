from __future__ import annotations

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


def test_ground_truth_report_detects_recall_and_text_loss_independently() -> None:
    report = evaluate_ocr_ground_truth(
        _response(include_beta=False),
        _ground_truth(),
    )

    assert report.item_recall == pytest.approx(0.5)
    assert report.missing_item_ids == ("beta",)
    assert report.text_mismatch_item_ids == ()


def test_ground_truth_report_detects_spatial_and_transcription_errors() -> None:
    response = _response(beta_text="gamma")
    response.OCR_text_clusters[1].bb_x_min = 75.0
    report = evaluate_ocr_ground_truth(response, _ground_truth(), minimum_iou=0.2)

    assert report.item_recall == pytest.approx(1.0)
    assert report.mean_spatial_iou < 1.0
    assert report.text_mismatch_item_ids == ("beta",)


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
