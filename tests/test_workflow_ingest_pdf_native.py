from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import pytest
from PIL import Image, ImageDraw
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
)

from kg_doc_parser.models import NonTextCluster, OCRClusterResponse, TextCluster
from kg_doc_parser.workflow_ingest import ocr_pipeline, pdf_native
from kg_doc_parser.workflow_ingest.providers import (
    ProviderEndpointConfig,
    WorkflowProviderSettings,
)


class _Box:
    width = 600
    height = 800

    def __iter__(self) -> Iterator[int]:
        return iter((0, 0, self.width, self.height))


class _TextPage:
    images: ClassVar[list[object]] = []
    mediabox = _Box()
    cropbox = _Box()
    rotation = 0

    def extract_text(self, *, visitor_text) -> str:
        visitor_text("lower line", [], [1, 0, 0, 1, 40, 100], None, 12)
        visitor_text("upper line", [], [1, 0, 0, 1, 20, 300], None, 12)
        return "upper line\nlower line"


class _ImagePage(_TextPage):
    images: ClassVar[list[object]] = [object()]


class _RotatedTextPage(_TextPage):
    rotation = 90


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


def test_native_pdf_text_is_retained_for_hybrid_pages_with_embedded_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pdf_native, "PdfReader", lambda _path: _Reader(_ImagePage()))

    result = pdf_native.extract_native_pdf_page(Path("scan.pdf"), 1)

    assert result is not None
    assert pdf_native.pdf_page_has_embedded_images(Path("scan.pdf"), 1) is True
    assert [cluster.text for cluster in result.OCR_text_clusters] == [
        "upper line",
        "lower line",
    ]


def test_native_pdf_text_defers_rotated_page_to_raster_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pdf_native, "PdfReader", lambda _path: _Reader(_RotatedTextPage())
    )

    assert pdf_native.extract_native_pdf_page(Path("rotated.pdf"), 1) is None


def test_hybrid_pdf_merge_deduplicates_matching_text_and_keeps_ocr_regions() -> None:
    native = _native_response()
    ocr = OCRClusterResponse(
        OCR_text_clusters=[
            TextCluster(
                text="native text",
                bb_x_min=0,
                bb_x_max=20,
                bb_y_min=90,
                bb_y_max=100,
                cluster_number=0,
            ),
            TextCluster(
                text="text inside scanned figure",
                bb_x_min=40,
                bb_x_max=90,
                bb_y_min=10,
                bb_y_max=20,
                cluster_number=1,
            ),
            TextCluster(
                text="textbook",
                bb_x_min=0,
                bb_x_max=20,
                bb_y_min=90,
                bb_y_max=100,
                cluster_number=3,
            ),
        ],
        non_text_objects=[
            NonTextCluster(
                description="chart",
                bb_x_min=30,
                bb_x_max=90,
                bb_y_min=25,
                bb_y_max=80,
                cluster_number=2,
            )
        ],
        is_empty_page=False,
        printed_page_number="1",
        meaningful_ordering=[0, 1, 2, 3],
        page_x_min=0,
        page_x_max=100,
        page_y_min=0,
        page_y_max=100,
        estimated_rotation_degrees=0,
        incomplete_words_on_edge=False,
        incomplete_text=False,
        data_loss_likelihood=0,
        scan_quality="high",
        contains_table=True,
        source_item_count=4,
    )

    merged = pdf_native.merge_native_and_ocr_page(native, ocr)

    assert {item.text for item in merged.OCR_text_clusters} == {
        "text inside scanned figure",
        "native text",
        "textbook",
    }
    assert [item.description for item in merged.non_text_objects] == ["chart"]
    assert merged.contains_table is True
    assert merged.spatial_accuracy == "line"
    assert sorted(merged.meaningful_ordering) == [0, 1, 2, 3]
    assert merged.source_item_count == 4


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
    assert result.unresolved_item_count == 0
    assert result.source_item_count == 2
    assert pdf_native.direct_pdf_image_regions(pdf_path, 1) == ()


@pytest.mark.ci
def test_pdf_text_only_page_uses_native_text_without_page_ocr(tmp_path: Path) -> None:
    pdf_path = tmp_path / "text-only.pdf"
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=400)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}
            )
        }
    )
    content = DecodedStreamObject()
    content.set_data(b"BT /F1 12 Tf 50 300 Td (native body text) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(content)
    with pdf_path.open("wb") as stream:
        writer.write(stream)
    rendered_page = tmp_path / "text-only.png"
    Image.new("RGB", (300, 400), "white").save(rendered_page)
    calls = 0

    def unexpected_ocr(
        _image: Path, _page: int, _settings: WorkflowProviderSettings
    ) -> OCRClusterResponse:
        nonlocal calls
        calls += 1
        return _native_response()

    preparation = ocr_pipeline._build_ocr_page_preparation(
        document_id="text-only-pdf-doc",
        title="Text-only PDF",
        output_dir=tmp_path / "artifacts",
        image_payloads=None,
        pdf_path=pdf_path,
        provider_settings=WorkflowProviderSettings(
            ocr=ProviderEndpointConfig(pdf_text_mode="prefer_native")
        ),
        ocr_runner=unexpected_ocr,
        pdf_rasterizer=lambda _source, _rendered: [rendered_page],
        ocr_candidate_models=None,
        probe=None,
    )

    result = preparation.page_context.ocr_page_runner(
        rendered_page, 1, preparation.page_context.provider_settings
    )

    assert calls == 0
    assert [cluster.text for cluster in result.OCR_text_clusters] == ["native body text"]


@pytest.mark.ci
def test_pdf_image_region_detection_fails_closed_for_vector_graphics(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "vector-figure.pdf"
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=400)
    content = DecodedStreamObject()
    content.set_data(b"0 0 200 100 re f")
    page[NameObject("/Contents")] = writer._add_object(content)
    with pdf_path.open("wb") as stream:
        writer.write(stream)

    assert pdf_native.direct_pdf_image_regions(pdf_path, 1) is None


@pytest.mark.ci
def test_hybrid_pdf_ocr_crops_a_scanned_figure_and_table_region(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "mixed-page.pdf"
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=400)
    scan = Image.new("RGB", (120, 100), "white")
    draw = ImageDraw.Draw(scan)
    draw.rectangle((5, 5, 50, 55), outline="black", width=2)
    for y in (65, 78, 91):
        draw.line((60, y, 114, y), fill="black", width=1)
    image_stream = DecodedStreamObject()
    image_stream.set_data(scan.tobytes())
    image_stream.update(
        {
            NameObject("/Type"): NameObject("/XObject"),
            NameObject("/Subtype"): NameObject("/Image"),
            NameObject("/Width"): NumberObject(120),
            NameObject("/Height"): NumberObject(100),
            NameObject("/ColorSpace"): NameObject("/DeviceRGB"),
            NameObject("/BitsPerComponent"): NumberObject(8),
        }
    )
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}
            ),
            NameObject("/XObject"): DictionaryObject(
                {NameObject("/Im0"): writer._add_object(image_stream)}
            ),
        }
    )
    content = DecodedStreamObject()
    content.set_data(
        b"q 120 0 0 100 50 160 cm /Im0 Do Q "
        b"BT /F1 12 Tf 50 120 Td (Selectable caption) Tj ET"
    )
    page[NameObject("/Contents")] = writer._add_object(content)
    with pdf_path.open("wb") as stream:
        writer.write(stream)

    assert pdf_native.direct_pdf_image_regions(pdf_path, 1) == ((50.0, 160.0, 170.0, 260.0),)

    rendered_page = Image.new("RGB", (300, 400), "white")
    rendered_page.paste(scan.resize((120, 100)), (50, 140))
    rendered_page_path = tmp_path / "rendered-page.png"
    rendered_page.save(rendered_page_path)
    calls: list[tuple[int, int]] = []

    def region_ocr(
        image_path: Path, page_number: int, _settings: WorkflowProviderSettings
    ) -> OCRClusterResponse:
        assert page_number == 1
        with Image.open(image_path) as crop:
            calls.append(crop.size)
        return OCRClusterResponse(
            OCR_text_clusters=[
                TextCluster(
                    text="scanned table row",
                    bb_x_min=2,
                    bb_x_max=90,
                    bb_y_min=65,
                    bb_y_max=80,
                    cluster_number=0,
                )
            ],
            non_text_objects=[
                NonTextCluster(
                    description="scanned figure and table",
                    bb_x_min=3,
                    bb_x_max=115,
                    bb_y_min=3,
                    bb_y_max=95,
                    cluster_number=1,
                )
            ],
            is_empty_page=False,
            printed_page_number=None,
            meaningful_ordering=[0, 1],
            page_x_min=0,
            page_x_max=120,
            page_y_min=0,
            page_y_max=100,
            estimated_rotation_degrees=0,
            incomplete_words_on_edge=False,
            incomplete_text=False,
            data_loss_likelihood=0,
            scan_quality="high",
            contains_table=True,
            source_item_count=2,
            spatial_accuracy="region",
        )

    preparation = ocr_pipeline._build_ocr_page_preparation(
        document_id="mixed-pdf-doc",
        title="Mixed PDF",
        output_dir=tmp_path / "artifacts",
        image_payloads=None,
        pdf_path=pdf_path,
        provider_settings=WorkflowProviderSettings(
            ocr=ProviderEndpointConfig(pdf_text_mode="prefer_native")
        ),
        ocr_runner=region_ocr,
        pdf_rasterizer=lambda _source, _rendered: [rendered_page_path],
        ocr_candidate_models=None,
        probe=None,
    )
    result = preparation.page_context.ocr_page_runner(
        rendered_page_path, 1, preparation.page_context.provider_settings
    )

    assert calls == [(120, 100)]
    assert {cluster.text for cluster in result.OCR_text_clusters} == {
        "scanned table row",
        "Selectable caption",
    }
    assert [item.description for item in result.non_text_objects] == [
        "scanned figure and table"
    ]
    assert result.page_x_max == 300
    assert result.page_y_max == 400
    assert result.contains_table is True


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
    raster_ocr = native.model_copy(
        update={
            "OCR_text_clusters": [
                native.OCR_text_clusters[0].model_copy(
                    update={"bb_y_min": 90, "bb_y_max": 100}
                )
            ]
        }
    )
    fallback_calls = 0

    monkeypatch.setattr(pdf_native, "extract_native_pdf_page", lambda _path, _page: native)
    monkeypatch.setattr(pdf_native, "pdf_page_has_embedded_images", lambda _path, _page: False)

    def fallback(_image: Path, _page: int, _settings: WorkflowProviderSettings) -> OCRClusterResponse:
        nonlocal fallback_calls
        fallback_calls += 1
        return raster_ocr

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

    assert result is not native
    assert fallback_calls == 1
    assert [cluster.text for cluster in result.OCR_text_clusters] == ["native text"]


def test_pdf_preparation_native_only_fails_closed_without_native_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"placeholder pdf")
    image_path = tmp_path / "page-1.png"
    Image.new("RGB", (10, 10), "white").save(image_path)
    monkeypatch.setattr(pdf_native, "extract_native_pdf_page", lambda _path, _page: None)
    monkeypatch.setattr(pdf_native, "pdf_page_has_embedded_images", lambda _path, _page: None)

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


def test_pdf_native_only_rejects_image_bearing_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "hybrid.pdf"
    pdf_path.write_bytes(b"placeholder pdf")
    image_path = tmp_path / "page-1.png"
    Image.new("RGB", (10, 10), "white").save(image_path)
    monkeypatch.setattr(pdf_native, "PdfReader", lambda _path: _Reader(_ImagePage()))

    preparation = ocr_pipeline._build_ocr_page_preparation(
        document_id="native-only-hybrid-doc",
        title="Native only hybrid",
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

    with pytest.raises(ValueError, match="native PDF image inventory is incomplete"):
        preparation.page_context.ocr_page_runner(
            image_path, 1, preparation.page_context.provider_settings
        )
