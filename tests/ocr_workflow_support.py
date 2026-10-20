"""Shared data builders for OCR workflow test modules."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageDraw

from kg_doc_parser.models import (
    OCRClusterResponse,
    SplitPage,
    SplitPageMeta,
    TextCluster,
)


def scratch_path(name: str) -> Path:
    root = Path("tests") / ".tmp_workflow_ingest_ocr"
    path = root / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def manual_case_dir(*, input_kind: str, provider: str, model: str) -> Path:
    safe_model = model.replace(":", "_").replace("/", "_")
    path = (
        Path("tests")
        / ".tmp_workflow_ingest_ocr"
        / "manual_cases"
        / provider
        / safe_model
        / input_kind
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def draw_test_image(path: Path, *, lines: list[str]) -> None:
    image = Image.new("RGB", (1200, 800), "white")
    draw = ImageDraw.Draw(image)
    y = 60
    for line in lines:
        draw.text((60, y), line, fill="black")
        y += 80
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG")


def write_two_page_pdf(image_paths: list[Path], pdf_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    first, rest = images[0], images[1:]
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    first.save(pdf_path, "PDF", save_all=True, append_images=rest)
    for image in images:
        image.close()


def fake_ocr_response(page_number: int, text: str) -> OCRClusterResponse:
    return OCRClusterResponse(
        OCR_text_clusters=[
            TextCluster(
                text=text,
                bb_x_min=10.0,
                bb_x_max=500.0,
                bb_y_min=10.0,
                bb_y_max=50.0,
                cluster_number=0,
            )
        ],
        non_text_objects=[],
        is_empty_page=False,
        printed_page_number=str(page_number),
        meaningful_ordering=[0],
        page_x_min=0.0,
        page_x_max=1200.0,
        page_y_min=0.0,
        page_y_max=800.0,
        estimated_rotation_degrees=0.0,
        incomplete_words_on_edge=False,
        incomplete_text=False,
        data_loss_likelihood=0.0,
        scan_quality="high",
        contains_table=False,
    )


def write_valid_ocr_artifact(
    path: Path,
    *,
    page_number: int = 1,
    source_image_sha256: str | None = None,
    ocr_configuration_fingerprint: str | None = None,
) -> None:
    split_page = SplitPage(
        pdf_page_num=page_number,
        metadata=SplitPageMeta(
            ocr_model_name="test-model",
            ocr_datetime=0.0,
            ocr_json_version="workflow_ingest_v1",
        ),
        **fake_ocr_response(page_number, f"page {page_number}").model_dump(
            dump_format="python"
        ),
    )
    payload = split_page.dump_raw(dump_format="json")
    if source_image_sha256 is not None or ocr_configuration_fingerprint is not None:
        payload["metadata"] = {
            **split_page.metadata.model_dump(mode="json"),
            **(
                {"source_image_sha256": source_image_sha256}
                if source_image_sha256 is not None
                else {}
            ),
            **(
                {"ocr_configuration_fingerprint": ocr_configuration_fingerprint}
                if ocr_configuration_fingerprint is not None
                else {}
            ),
        }
    path.write_text(json.dumps(payload), encoding="utf-8")
