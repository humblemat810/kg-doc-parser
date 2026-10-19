"""Optional native-text extraction for hybrid PDF OCR."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from pydantic_extension.model_slicing import DtoType
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from ..models import NonTextCluster, OCRClusterResponse, TextCluster

_TextVisitor = Callable[[str, Sequence[float], Sequence[float], object, float], None]


def extract_native_pdf_page(
    pdf_path: Path,
    page_number: int,
) -> OCRClusterResponse | None:
    """Extract positioned native text, or return ``None`` when unsafe.

    pypdf reports text coordinates in PDF points. They are line-level
    approximations, so the response declares ``spatial_accuracy="line"``;
    callers must not promote these boxes to character-level grounding.
    """
    if page_number < 1:
        raise ValueError("page_number must be positive")
    reader = PdfReader(str(pdf_path))
    if page_number > len(reader.pages):
        raise ValueError(f"page_number {page_number} exceeds PDF page count")
    page = reader.pages[page_number - 1]
    box = page.mediabox
    crop_box = page.cropbox
    if int(page.rotation) % 360 != 0 or any(
        abs(float(media_coordinate) - float(crop_coordinate)) > 1e-6
        for media_coordinate, crop_coordinate in zip(box, crop_box, strict=True)
    ):
        return None
    width = float(box.width)
    height = float(box.height)
    if not all(math.isfinite(value) and value > 0 for value in (width, height)):
        return None
    runs: list[tuple[str, float, float, float]] = []

    def visitor(
        text: str,
        _cm: Sequence[float],
        tm: Sequence[float],
        _font: object,
        font_size: float,
    ) -> None:
        if not text.strip() or len(tm) < 6:
            return
        try:
            x = float(tm[4])
            y = float(tm[5])
            size = max(abs(float(font_size)), 1.0)
        except (TypeError, ValueError):
            return
        if not all(math.isfinite(value) for value in (x, y, size)):
            return
        runs.append((text.strip(), x, y, size))

    page.extract_text(visitor_text=cast(_TextVisitor, visitor))
    if not runs:
        return None

    ordered = sorted(runs, key=lambda item: (-item[2], item[1]))
    clusters = [
        TextCluster(
            text=text,
            bb_x_min=max(0.0, x),
            bb_x_max=min(width, max(x + max(size * len(text) * 0.5, 1.0), x)),
            bb_y_min=max(0.0, y),
            bb_y_max=min(height, max(y + size, y)),
            cluster_number=index,
        )
        for index, (text, x, y, size) in enumerate(ordered)
    ]
    return OCRClusterResponse(
        OCR_text_clusters=clusters,
        non_text_objects=[],
        is_empty_page=False,
        printed_page_number=str(page_number),
        meaningful_ordering=list(range(len(clusters))),
        page_x_min=0.0,
        page_x_max=width,
        page_y_min=0.0,
        page_y_max=height,
        estimated_rotation_degrees=0.0,
        incomplete_words_on_edge=False,
        incomplete_text=False,
        data_loss_likelihood=0.0,
        scan_quality="high",
        contains_table=False,
        unresolved_item_count=0,
        source_item_count=len(clusters),
        spatial_accuracy="line",
    )


def pdf_page_has_embedded_images(pdf_path: Path, page_number: int) -> bool | None:
    """Return whether a page has embedded images, or ``None`` if uncertain."""
    if page_number < 1:
        raise ValueError("page_number must be positive")
    try:
        reader = PdfReader(str(pdf_path))
        if page_number > len(reader.pages):
            raise ValueError(f"page_number {page_number} exceeds PDF page count")
        return bool(reader.pages[page_number - 1].images)
    except (AttributeError, OSError, PdfReadError, ValueError):
        return None


def merge_native_and_ocr_page(
    native: OCRClusterResponse,
    ocr: OCRClusterResponse,
) -> OCRClusterResponse:
    """Merge native PDF text with OCR regions without dropping OCR-only evidence.

    OCR text is removed only when normalized content and geometry both match a
    native line. Conflicting or image-derived text remains visible as separate
    evidence for downstream review.
    """
    native_width = native.page_x_max - native.page_x_min
    native_height = native.page_y_max - native.page_y_min
    ocr_width = ocr.page_x_max - ocr.page_x_min
    ocr_height = ocr.page_y_max - ocr.page_y_min
    if min(native_width, native_height, ocr_width, ocr_height) <= 0:
        return ocr

    def normalized_text(text: str) -> str:
        normalized = "".join(
            character if character.isalnum() else " " for character in text.casefold()
        )
        return " ".join(normalized.split())

    def project_native(cluster: TextCluster) -> TextCluster:
        x_min = ocr.page_x_min + (cluster.bb_x_min - native.page_x_min) * ocr_width / native_width
        x_max = ocr.page_x_min + (cluster.bb_x_max - native.page_x_min) * ocr_width / native_width
        # PDF coordinates have an upward origin; raster coordinates have a downward origin.
        y_min = ocr.page_y_min + (native.page_y_max - cluster.bb_y_max) * ocr_height / native_height
        y_max = ocr.page_y_min + (native.page_y_max - cluster.bb_y_min) * ocr_height / native_height
        return cluster.model_copy(
            update={
                "bb_x_min": max(ocr.page_x_min, min(x_min, ocr.page_x_max)),
                "bb_x_max": max(ocr.page_x_min, min(x_max, ocr.page_x_max)),
                "bb_y_min": max(ocr.page_y_min, min(y_min, ocr.page_y_max)),
                "bb_y_max": max(ocr.page_y_min, min(y_max, ocr.page_y_max)),
            }
        )

    projected_native = [project_native(item) for item in native.OCR_text_clusters]

    def is_duplicate(ocr_cluster: TextCluster) -> bool:
        ocr_text = normalized_text(ocr_cluster.text)
        if not ocr_text:
            return False
        for native_cluster in projected_native:
            native_text = normalized_text(native_cluster.text)
            if not native_text or not (
                native_text == ocr_text
                or f" {native_text} " in f" {ocr_text} "
                or f" {ocr_text} " in f" {native_text} "
            ):
                continue
            x_overlap = max(
                0.0,
                min(ocr_cluster.bb_x_max, native_cluster.bb_x_max)
                - max(ocr_cluster.bb_x_min, native_cluster.bb_x_min),
            )
            y_overlap = max(
                0.0,
                min(ocr_cluster.bb_y_max, native_cluster.bb_y_max)
                - max(ocr_cluster.bb_y_min, native_cluster.bb_y_min),
            )
            smaller_area = min(
                max(0.0, ocr_cluster.bb_x_max - ocr_cluster.bb_x_min)
                * max(0.0, ocr_cluster.bb_y_max - ocr_cluster.bb_y_min),
                max(0.0, native_cluster.bb_x_max - native_cluster.bb_x_min)
                * max(0.0, native_cluster.bb_y_max - native_cluster.bb_y_min),
            )
            if smaller_area and x_overlap * y_overlap / smaller_area >= 0.5:
                return True
        return False

    retained_ocr = [item for item in ocr.OCR_text_clusters if not is_duplicate(item)]
    text_items = [*projected_native, *retained_ocr]
    text_items.sort(key=lambda item: (item.bb_y_min, item.bb_x_min, item.cluster_number))
    non_text_items = sorted(
        ocr.non_text_objects,
        key=lambda item: (item.bb_y_min, item.bb_x_min, item.cluster_number),
    )
    ordered_items: Sequence[
        tuple[DtoType[float], DtoType[float], int, TextCluster | NonTextCluster]
    ] = [
        (item.bb_y_min, item.bb_x_min, item.cluster_number, item)
        for item in text_items
    ] + [
        (item.bb_y_min, item.bb_x_min, item.cluster_number, item)
        for item in non_text_items
    ]
    ordered_items.sort(key=lambda item: item[:3])
    merged_text: list[TextCluster] = []
    merged_non_text: list[NonTextCluster] = []
    ordering: list[int] = []
    for index, (_y, _x, _old_id, item) in enumerate(ordered_items):
        if isinstance(item, TextCluster):
            merged_text.append(item.model_copy(update={"cluster_number": index}))
        else:
            merged_non_text.append(item.model_copy(update={"cluster_number": index}))
        ordering.append(index)

    payload = ocr.model_dump()
    payload.update(
        {
            "OCR_text_clusters": merged_text,
            "non_text_objects": merged_non_text,
            "meaningful_ordering": ordering,
            "is_empty_page": not ordered_items,
            "spatial_accuracy": "line",
            "source_item_count": len(ordered_items) + ocr.unresolved_item_count,
        }
    )
    return OCRClusterResponse.model_validate(payload)


__all__ = [
    "extract_native_pdf_page",
    "merge_native_and_ocr_page",
    "pdf_page_has_embedded_images",
]
