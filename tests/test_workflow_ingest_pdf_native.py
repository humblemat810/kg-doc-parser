from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from kg_doc_parser.workflow_ingest import ocr_pipeline, pdf_native
from kg_doc_parser.workflow_ingest.providers import (
    ProviderEndpointConfig,
    WorkflowProviderSettings,
)


class _Box:
    width = 600
    height = 800


class _TextPage:
    images: ClassVar[list[object]] = []
    mediabox = _Box()

    def extract_text(self, *, visitor_text) -> str:
        visitor_text("lower line", [], [1, 0, 0, 1, 40, 100], None, 12)
        visitor_text("upper line", [], [1, 0, 0, 1, 20, 300], None, 12)
        return "upper line\nlower line"


class _ImagePage(_TextPage):
    images: ClassVar[list[object]] = [object()]

    def extract_text(self, *, visitor_text) -> str:
        raise AssertionError("image pages must not use native text")


class _Reader:
    def __init__(self, page: object) -> None:
        self.pages = [page]


def test_native_pdf_text_preserves_line_order_and_declares_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pdf_native, "PdfReader", lambda _path: _Reader(_TextPage()))

    result = pdf_native.extract_native_pdf_page(Path("report.pdf"), 1)

    assert result is not None
    assert [cluster.text for cluster in result.OCR_text_clusters] == [
        "upper line",
        "lower line",
    ]
    assert result.meaningful_ordering == [0, 1]
    assert result.spatial_accuracy == "line"
    assert result.page_x_max == 600.0
    assert result.page_y_max == 800.0


def test_native_pdf_text_defers_pages_with_embedded_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pdf_native, "PdfReader", lambda _path: _Reader(_ImagePage()))

    assert pdf_native.extract_native_pdf_page(Path("scan.pdf"), 1) is None


def test_pdf_text_mode_is_part_of_ocr_cache_identity() -> None:
    native = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(pdf_text_mode="prefer_native")
    )
    raster = native.model_copy(
        update={"ocr": native.ocr.model_copy(update={"pdf_text_mode": "disabled"})}
    )

    assert ocr_pipeline._ocr_configuration_fingerprint(native) != ocr_pipeline._ocr_configuration_fingerprint(
        raster
    )
