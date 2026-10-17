from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest
from PIL import Image
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from kg_doc_parser.models import OCRClusterResponse, TextCluster
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


def test_native_pdf_text_reads_a_real_selectable_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "selectable.pdf"
    writer = PdfWriter()
    page = writer.add_blank_page(width=600, height=800)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    resources = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}
            )
        }
    )
    page[NameObject("/Resources")] = resources
    content = DecodedStreamObject()
    content.set_data(
        b"BT /F1 12 Tf 40 700 Td (upper line) Tj "
        b"0 -40 Td (lower line) Tj ET"
    )
    page[NameObject("/Contents")] = writer._add_object(content)
    with pdf_path.open("wb") as stream:
        writer.write(stream)

    result = pdf_native.extract_native_pdf_page(pdf_path, 1)

    assert result is not None
    assert [cluster.text for cluster in result.OCR_text_clusters] == [
        "upper line",
        "lower line",
    ]
    assert result.meaningful_ordering == [0, 1]
    assert result.spatial_accuracy == "line"
    assert result.OCR_text_clusters[0].bb_y_min > result.OCR_text_clusters[1].bb_y_min


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


def _native_response() -> OCRClusterResponse:
    return OCRClusterResponse(
        OCR_text_clusters=[
            TextCluster(
                text="native text",
                bb_x_min=0,
                bb_x_max=20,
                bb_y_min=0,
                bb_y_max=10,
                cluster_number=0,
            )
        ],
        non_text_objects=[],
        is_empty_page=False,
        printed_page_number="1",
        meaningful_ordering=[0],
        page_x_min=0,
        page_x_max=100,
        page_y_min=0,
        page_y_max=100,
        estimated_rotation_degrees=0,
        incomplete_words_on_edge=False,
        incomplete_text=False,
        data_loss_likelihood=0,
        scan_quality="high",
        contains_table=False,
    )


def test_pdf_preparation_prefers_native_text_at_the_production_runner_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "report.pdf"
    pdf_path.write_bytes(b"placeholder pdf")
    image_path = tmp_path / "page-1.png"
    Image.new("RGB", (10, 10), "white").save(image_path)
    native = _native_response()
    fallback_calls = 0

    monkeypatch.setattr(pdf_native, "extract_native_pdf_page", lambda _path, _page: native)

    def fallback(_image: Path, _page: int, _settings: WorkflowProviderSettings) -> OCRClusterResponse:
        nonlocal fallback_calls
        fallback_calls += 1
        return _native_response()

    preparation = ocr_pipeline._build_ocr_page_preparation(
        document_id="native-boundary-doc",
        title="Native boundary",
        output_dir=tmp_path / "artifacts",
        image_payloads=None,
        pdf_path=pdf_path,
        provider_settings=WorkflowProviderSettings(
            ocr=ProviderEndpointConfig(pdf_text_mode="prefer_native")
        ),
        ocr_runner=fallback,
        pdf_rasterizer=lambda _source, _rendered: [image_path],
        ocr_candidate_models=None,
        probe=None,
    )

    result = preparation.page_context.ocr_page_runner(
        image_path, 1, preparation.page_context.provider_settings
    )

    assert result is native
    assert fallback_calls == 0


def test_pdf_preparation_native_only_fails_closed_without_native_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"placeholder pdf")
    image_path = tmp_path / "page-1.png"
    Image.new("RGB", (10, 10), "white").save(image_path)
    monkeypatch.setattr(pdf_native, "extract_native_pdf_page", lambda _path, _page: None)

    preparation = ocr_pipeline._build_ocr_page_preparation(
        document_id="native-only-doc",
        title="Native only",
        output_dir=tmp_path / "artifacts",
        image_payloads=None,
        pdf_path=pdf_path,
        provider_settings=WorkflowProviderSettings(
            ocr=ProviderEndpointConfig(pdf_text_mode="native_only")
        ),
        ocr_runner=lambda *_args: _native_response(),
        pdf_rasterizer=lambda _source, _rendered: [image_path],
        ocr_candidate_models=None,
        probe=None,
    )

    with pytest.raises(ValueError, match="native PDF text is unavailable"):
        preparation.page_context.ocr_page_runner(
            image_path, 1, preparation.page_context.provider_settings
        )
