"""Optional native-text extraction for hybrid PDF OCR."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from pypdf import PdfReader

from ..models import OCRClusterResponse, TextCluster

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
    try:
        if page.images:
            return None
    except (AttributeError, OSError, ValueError):
        return None

    box = page.mediabox
    width = float(box.width)
    height = float(box.height)
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
        spatial_accuracy="line",
    )


__all__ = ["extract_native_pdf_page"]
