"""Focused unit tests for OCR quality diagnostics and policy severity."""

from __future__ import annotations

import pytest
from ocr_workflow_support import fake_ocr_response, scratch_path
from PIL import Image, ImageDraw

from kg_doc_parser.models import OCRClusterResponse, TextCluster
from kg_doc_parser.workflow_ingest.ocr_page_runner import (
    _minimal_ocr_response_from_text,
)
from kg_doc_parser.workflow_ingest.ocr_pipeline import (
    OCRQualityPolicy,
    _evaluate_ocr_quality,
)

pytestmark = [pytest.mark.workflow]


@pytest.mark.ci
@pytest.mark.parametrize(
    "update",
    [
        {"is_empty_page": True},
        {"meaningful_ordering": []},
        {"meaningful_ordering": [99]},
        {"meaningful_ordering": [0, 0]},
    ],
)
def test_ocr_cluster_response_rejects_inconsistent_empty_or_ordering_state(
    update: dict[str, object],
) -> None:
    payload = fake_ocr_response(1, "Alpha").model_dump()
    payload.update(update)

    with pytest.raises(ValueError, match="(is_empty_page|meaningful_ordering)"):
        OCRClusterResponse.model_validate(payload)


@pytest.mark.ci
def test_ocr_quality_policy_flags_and_can_reject_provider_loss_signals() -> None:
    response = fake_ocr_response(1, "quality signal").model_copy(
        update={
            "incomplete_text": True,
            "incomplete_words_on_edge": True,
            "data_loss_likelihood": 0.75,
        }
    )
    flagged = _evaluate_ocr_quality(response, OCRQualityPolicy())
    assert {item.code for item in flagged} == {
        "incomplete_text",
        "incomplete_words_on_edge",
    }
    assert all(item.severity == "warning" for item in flagged)

    rejected = _evaluate_ocr_quality(
        response,
        OCRQualityPolicy(
            max_data_loss_likelihood=0.5,
            reject_incomplete_text=True,
            reject_incomplete_words_on_edge=True,
        ),
    )
    assert {item.code for item in rejected} == {
        "incomplete_text",
        "incomplete_words_on_edge",
        "data_loss_likelihood",
    }
    assert all(item.severity == "error" for item in rejected)


@pytest.mark.ci
def test_ocr_quality_policy_can_reject_empty_provider_output() -> None:
    payload = fake_ocr_response(1, "quality signal").model_dump()
    payload.update(
        {"OCR_text_clusters": [], "is_empty_page": True, "meaningful_ordering": []}
    )
    response = OCRClusterResponse.model_validate(payload)

    flagged = _evaluate_ocr_quality(response, OCRQualityPolicy())
    assert [(item.code, item.severity) for item in flagged] == [
        ("empty_ocr_output", "warning")
    ]

    rejected = _evaluate_ocr_quality(response, OCRQualityPolicy(reject_empty_page=True))
    assert [(item.code, item.severity) for item in rejected] == [
        ("empty_ocr_output", "error")
    ]


@pytest.mark.ci
def test_ocr_quality_independently_flags_empty_output_on_visible_source_ink() -> None:
    scratch = scratch_path("ocr_quality_visible_source")
    source_image = Image.new("RGB", (240, 80), "white")
    ImageDraw.Draw(source_image).text((12, 24), "Visible source text", fill="black")
    image_path = scratch / "text-heavy.png"
    source_image.save(image_path)
    payload = fake_ocr_response(1, "placeholder").model_dump()
    payload.update(
        {
            "OCR_text_clusters": [],
            "non_text_objects": [],
            "is_empty_page": True,
            "meaningful_ordering": [],
            "source_item_count": 0,
        }
    )
    response = OCRClusterResponse.model_validate(payload)

    warning = _evaluate_ocr_quality(
        response,
        OCRQualityPolicy(mode="permissive"),
        source_image_path=image_path,
    )
    warning_by_code = {item.code: item for item in warning}
    assert warning_by_code["empty_ocr_output_on_nonblank_image"].severity == "warning"

    rejected = _evaluate_ocr_quality(
        response,
        OCRQualityPolicy(),
        source_image_path=image_path,
    )
    rejected_by_code = {item.code: item for item in rejected}
    assert rejected_by_code["empty_ocr_output_on_nonblank_image"].severity == "error"

    blank_path = scratch / "blank.png"
    Image.new("RGB", (240, 80), "white").save(blank_path)
    blank_diagnostics = _evaluate_ocr_quality(
        response, OCRQualityPolicy(), source_image_path=blank_path
    )
    assert "empty_ocr_output_on_nonblank_image" not in {
        item.code for item in blank_diagnostics
    }


@pytest.mark.ci
def test_ocr_quality_strict_mode_flags_unexplained_image_regions() -> None:
    scratch = scratch_path("ocr_quality_unexplained_regions")
    image = Image.new("RGB", (240, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.text((12, 12), "recognized", fill="black")
    draw.text((12, 92), "omitted", fill="black")
    image_path = scratch / "partial-ocr.png"
    image.save(image_path)
    response = OCRClusterResponse.model_validate(
        {
            **fake_ocr_response(1, "recognized").model_dump(),
            "page_x_min": 0,
            "page_x_max": 240,
            "page_y_min": 0,
            "page_y_max": 120,
            "OCR_text_clusters": [
                {
                    "text": "recognized",
                    "bb_x_min": 10,
                    "bb_x_max": 100,
                    "bb_y_min": 8,
                    "bb_y_max": 32,
                    "cluster_number": 0,
                }
            ],
            "non_text_objects": [],
            "is_empty_page": False,
            "meaningful_ordering": [0],
            "source_item_count": 1,
        }
    )

    diagnostics = _evaluate_ocr_quality(
        response, OCRQualityPolicy(), source_image_path=image_path
    )
    diagnostic = next(
        item for item in diagnostics if item.code == "unexplained_visual_ink_regions"
    )
    assert diagnostic.severity == "error"
    assert "not proof of recognition completeness" in diagnostic.message


@pytest.mark.ci
def test_ocr_quality_accepts_boxes_aligned_with_each_rendered_text_line() -> None:
    scratch = scratch_path("ocr_quality_aligned_text_lines")
    image_path = scratch / "aligned-lines.png"
    image = Image.new("RGB", (1200, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.text((60, 60), "durable OCR alpha", fill="black")
    draw.text((60, 140), "durable OCR beta", fill="black")
    image.save(image_path)
    image.close()

    response = OCRClusterResponse.model_validate(
        {
            **fake_ocr_response(1, "durable OCR alpha").model_dump(),
            "OCR_text_clusters": [
                {
                    "text": "durable OCR alpha",
                    "bb_x_min": 55,
                    "bb_x_max": 300,
                    "bb_y_min": 55,
                    "bb_y_max": 80,
                    "cluster_number": 0,
                },
                {
                    "text": "durable OCR beta",
                    "bb_x_min": 55,
                    "bb_x_max": 300,
                    "bb_y_min": 135,
                    "bb_y_max": 160,
                    "cluster_number": 1,
                },
            ],
            "meaningful_ordering": [0, 1],
        }
    )

    diagnostics = _evaluate_ocr_quality(
        response, OCRQualityPolicy(), source_image_path=image_path
    )
    assert "unexplained_visual_ink_regions" not in {
        item.code for item in diagnostics
    }


@pytest.mark.ci
def test_ocr_quality_policy_records_unresolved_provider_items() -> None:
    response = fake_ocr_response(1, "quality signal").model_copy(
        update={"unresolved_item_count": 2}
    )

    flagged = _evaluate_ocr_quality(response, OCRQualityPolicy())
    assert [(item.code, item.severity) for item in flagged] == [
        ("unresolved_items", "warning")
    ]

    rejected = _evaluate_ocr_quality(
        response, OCRQualityPolicy(reject_unresolved_items=True)
    )
    assert [(item.code, item.severity) for item in rejected] == [
        ("unresolved_items", "error")
    ]


@pytest.mark.ci
def test_ocr_quality_policy_can_reject_unaccounted_source_items() -> None:
    response = fake_ocr_response(1, "source accounting").model_copy(
        update={"source_item_count": 2}
    )

    flagged = _evaluate_ocr_quality(response, OCRQualityPolicy())
    assert [(item.code, item.severity) for item in flagged] == [
        ("source_item_loss", "warning")
    ]

    rejected = _evaluate_ocr_quality(
        response, OCRQualityPolicy(reject_source_item_loss=True)
    )
    assert [(item.code, item.severity) for item in rejected] == [
        ("source_item_loss", "error")
    ]


@pytest.mark.ci
def test_raw_text_ocr_fallback_declares_source_item_accounting() -> None:
    image_path = scratch_path("raw_text_ocr_fallback_accounting") / "page.png"
    Image.new("RGB", (80, 40), "white").save(image_path)

    empty = _minimal_ocr_response_from_text(
        text="{}", page_number=1, image_path=image_path
    )
    assert empty.source_item_count == 0
    assert empty.unresolved_item_count == 0

    fallback = _minimal_ocr_response_from_text(
        text="raw provider text", page_number=1, image_path=image_path
    )
    assert fallback.source_item_count == 1
    assert fallback.unresolved_item_count == 0
    assert len(fallback.OCR_text_clusters) + len(fallback.non_text_objects) == 1


@pytest.mark.ci
def test_ocr_quality_policy_rejects_invalid_coordinates_for_item_precision() -> None:
    response = fake_ocr_response(1, "quality signal").model_copy(
        update={
            "spatial_accuracy": "region",
            "OCR_text_clusters": [
                TextCluster(
                    text="quality signal",
                    bb_x_min=20.0,
                    bb_x_max=20.0,
                    bb_y_min=10.0,
                    bb_y_max=30.0,
                    cluster_number=0,
                )
            ],
        }
    )

    rejected = _evaluate_ocr_quality(response, OCRQualityPolicy())
    assert [(item.code, item.severity) for item in rejected] == [
        ("invalid_spatial_coordinates", "error")
    ]

    warned = _evaluate_ocr_quality(
        response, OCRQualityPolicy(reject_invalid_spatial_coordinates=False)
    )
    assert [(item.code, item.severity) for item in warned] == [
        ("invalid_spatial_coordinates", "warning")
    ]
