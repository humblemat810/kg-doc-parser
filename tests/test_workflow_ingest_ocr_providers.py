from __future__ import annotations

import json
import sys
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType
from typing import ClassVar, Self

import pytest

from kg_doc_parser.models import NonTextCluster, OCRClusterResponse, TextCluster
from kg_doc_parser.workflow_ingest import (
    ProviderEndpointConfig,
    WorkflowProviderSettings,
    normalize_ocr_pages,
    ocr_provider_smoke,
)
from kg_doc_parser.workflow_ingest.ocr_providers import (
    list_ocr_provider_capabilities,
    normalize_ocr_provider,
    register_ocr_provider,
    run_docling_ocr_page,
    run_easyocr_page,
    run_kserve_v2_page,
    run_rapidocr_page,
    run_registered_ocr_page,
    unregister_ocr_provider,
)
from kg_doc_parser.workflow_ingest.providers import build_chat_model


def test_capability_inventory_contains_all_docling_engines_and_codex_boundary() -> None:
    capabilities = list_ocr_provider_capabilities()
    engines = {item.engine for item in capabilities if item.provider == "docling"}
    assert engines == {"rapidocr", "easyocr", "tesseract", "tesserocr", "ocrmac", "nemotron-ocr"}
    codex = next(item for item in capabilities if item.provider == "codex")
    assert codex.vision is True
    direct = {item.provider for item in capabilities if item.provider in {"rapidocr", "easyocr"}}
    assert direct == {"rapidocr", "easyocr"}
    assert all(item.coordinates for item in capabilities if item.provider in direct)
    assert all(
        item.coordinate_precision == "region"
        for item in capabilities
        if item.provider in direct or item.provider == "docling"
    )
    assert all(
        item.coordinate_precision == "unknown"
        for item in capabilities
        if item.provider in {"codex", "openai", "azure", "gemini", "ollama"}
    )
    assert {item.provider for item in capabilities} >= {"custom", "kserve"}


def test_ocr_provider_smoke_writes_typed_json_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = Path("tests") / ".tmp_ocr_provider_smoke_report"
    output_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        sys,
        "argv",
        ["ocr-provider-smoke", "--output-dir", str(output_dir)],
    )

    assert ocr_provider_smoke.main() == 0
    report = json.loads(
        (output_dir / "capability-report.json").read_text(encoding="utf-8")
    )
    assert isinstance(report, dict)
    assert report["live_status"] == "not_requested"
    assert isinstance(report["providers"], list)


def test_ocr_response_accepts_unified_text_and_nontext_reading_order() -> None:
    response = OCRClusterResponse(
        OCR_text_clusters=[
            TextCluster(
                text="caption",
                bb_x_min=0,
                bb_x_max=10,
                bb_y_min=0,
                bb_y_max=10,
                cluster_number=0,
            )
        ],
        non_text_objects=[
            NonTextCluster(
                description="figure",
                bb_x_min=0,
                bb_x_max=10,
                bb_y_min=10,
                bb_y_max=20,
                cluster_number=1,
            )
        ],
        is_empty_page=False,
        printed_page_number=None,
        meaningful_ordering=[0, 1],
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
    assert response.meaningful_ordering == [0, 1]


def test_normalize_ocr_pages_preserves_caption_and_nontext_permutation() -> None:
    """Keep a declared figure/caption/table/text traversal as one sequence."""
    page = {
        "pdf_page_num": 1,
        "OCR_text_clusters": [
            {
                "text": "caption",
                "cluster_number": 1,
                "bb_x_min": 10,
                "bb_x_max": 90,
                "bb_y_min": 20,
                "bb_y_max": 30,
            },
            {
                "text": "body text",
                "cluster_number": 3,
                "bb_x_min": 10,
                "bb_x_max": 90,
                "bb_y_min": 80,
                "bb_y_max": 100,
            },
        ],
        "non_text_objects": [
            {
                "description": "figure",
                "cluster_number": 0,
                "bb_x_min": 10,
                "bb_x_max": 90,
                "bb_y_min": 0,
                "bb_y_max": 20,
            },
            {
                "description": "table",
                "cluster_number": 2,
                "bb_x_min": 10,
                "bb_x_max": 90,
                "bb_y_min": 30,
                "bb_y_max": 80,
            },
        ],
        "meaningful_ordering": [0, 1, 2, 3],
    }

    normalized = normalize_ocr_pages(
        document_id="doc",
        title="doc",
        pages=[page],
    )

    units = normalized.collections[0].pages[0].units
    assert [unit.cluster_number for unit in units] == [0, 1, 2, 3]
    assert [unit.modality for unit in units] == [
        "image_region",
        "ocr_text",
        "image_region",
        "ocr_text",
    ]


@pytest.mark.parametrize(
    ("raw", "engine"),
    [
        ("docling:easyocr", "easyocr"),
        ("rapidocr", None),
        ("easyocr", None),
        ("bonsai2", None),
        ("openai", None),
        ("custom", None),
        ("kserve", None),
    ],
)
def test_provider_names_are_normalized(raw: str, engine: str | None) -> None:
    provider, selected_engine = normalize_ocr_provider(raw)
    assert provider in {"docling", "rapidocr", "easyocr", "bonsai", "openai", "custom", "kserve"}
    assert selected_engine == engine


def test_unknown_docling_engine_fails_before_optional_import() -> None:
    with pytest.raises(ValueError, match="unsupported Docling OCR engine"):
        normalize_ocr_provider("docling", engine="missing")


def test_bonsai_reuses_openai_compatible_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeChat:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    module = ModuleType("langchain_openai")
    module.ChatOpenAI = FakeChat  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_openai", module)
    monkeypatch.setenv("BONSAI_API_KEY", "test-token")
    model = build_chat_model(
        ProviderEndpointConfig(
            provider="bonsai",
            model="Ternary-Bonsai-2-27B-PTQ1_0",
            base_url="http://127.0.0.1:8001/v1",
            api_key_env="BONSAI_API_KEY",
        )
    )
    assert model is not None
    assert captured["model"] == "Ternary-Bonsai-2-27B-PTQ1_0"
    assert captured["base_url"] == "http://127.0.0.1:8001/v1"
    assert captured["api_key"] == "test-token"


def test_docling_is_configurable_without_importing_docling() -> None:
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="docling", model="docling", ocr_engine="rapidocr")
    )
    assert settings.ocr.ocr_engine == "rapidocr"


def test_direct_rapidocr_adapter_normalizes_boxes_without_docling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from PIL import Image

    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 80), "white").save(image_path)

    class Output:
        boxes: ClassVar[list[list[tuple[int, int]]]] = [[(10, 20), (50, 20), (50, 30), (10, 30)]]
        txts: ClassVar[list[str]] = ["hello"]

    class FakeRapidOCR:
        def __call__(self, path: str) -> Output:
            assert path == str(image_path)
            return Output()

    module = ModuleType("rapidocr")
    module.RapidOCR = FakeRapidOCR  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rapidocr", module)
    response = run_rapidocr_page(image_path, 1, ProviderEndpointConfig(provider="rapidocr"))
    assert response.OCR_text_clusters[0].text == "hello"
    assert response.OCR_text_clusters[0].bb_x_min == 10
    assert response.OCR_text_clusters[0].bb_y_max == 30


def test_direct_rapidocr_adapter_rejects_boolean_polygon_coordinates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from PIL import Image

    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 80), "white").save(image_path)

    class Output:
        boxes: ClassVar[list[list[tuple[object, object]]]] = [
            [(True, 20), (50, 20), (50, 30), (10, 30)]
        ]
        txts: ClassVar[list[str]] = ["malformed"]

    class FakeRapidOCR:
        def __call__(self, path: str) -> Output:
            assert path == str(image_path)
            return Output()

    module = ModuleType("rapidocr")
    module.RapidOCR = FakeRapidOCR  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rapidocr", module)

    with pytest.raises(RuntimeError, match="invalid quadrilateral"):
        run_rapidocr_page(image_path, 1, ProviderEndpointConfig(provider="rapidocr"))


def test_direct_easyocr_adapter_normalizes_boxes_without_docling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from PIL import Image

    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 80), "white").save(image_path)

    class FakeReader:
        def __init__(self, languages: list[str], *, gpu: bool, verbose: bool) -> None:
            assert languages == ["en"]
            assert gpu is False and verbose is False

        def readtext(self, path: str, *, detail: int) -> list[object]:
            assert path == str(image_path) and detail == 1
            return [[[(10, 20), (50, 20), (50, 30), (10, 30)], "hello", 0.99]]

    module = ModuleType("easyocr")
    module.Reader = FakeReader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "easyocr", module)
    response = run_easyocr_page(image_path, 1, ProviderEndpointConfig(provider="easyocr"))
    assert response.OCR_text_clusters[0].text == "hello"
    assert response.OCR_text_clusters[0].bb_x_max == 50


def test_docling_missing_runtime_fails_at_selection_boundary() -> None:
    if find_spec("docling") is not None:
        pytest.skip("Docling is installed; missing-runtime branch is not applicable")
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="docling", model="docling", ocr_engine="rapidocr")
    )
    with pytest.raises(RuntimeError, match="optional docling package"):
        run_docling_ocr_page(Path("missing-page.png"), 1, settings, engine="rapidocr")


def test_docling_adapter_preserves_text_table_figure_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Docling item provenance remains distinct and is clipped to page bounds."""
    from PIL import Image

    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 80), "white").save(image_path)

    class BBox:
        def __init__(self, left: float, right: float, top: float, bottom: float) -> None:
            self.l = left
            self.r = right
            self.t = top
            self.b = bottom

    class Provenance:
        def __init__(self, bbox: BBox) -> None:
            self.bbox = bbox

    class Item:
        def __init__(self, label: str, text: str, bbox: BBox | None) -> None:
            self.label = label
            self.text = text
            self.prov = [Provenance(bbox)] if bbox is not None else []

    class Document:
        def iterate_items(self) -> list[Item]:
            return [
                Item("text", "heading", BBox(-10, 40, 5, 20)),
                Item("table", "<!-- table --> | a | b |", BBox(20, 120, 25, 90)),
                Item("picture", "", BBox(45, 70, 30, 60)),
                Item("figure", "missing provenance", None),
            ]

    class Conversion:
        document = Document()

    class InputFormat:
        IMAGE = "image"

    class ImageFormatOption:
        def __init__(self, **_kwargs: object) -> None:
            pass

    class DocumentConverter:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def convert(self, _path: str) -> Conversion:
            return Conversion()

    docling = ModuleType("docling")
    datamodel = ModuleType("docling.datamodel")
    base_models = ModuleType("docling.datamodel.base_models")
    base_models.InputFormat = InputFormat  # type: ignore[attr-defined]
    converter = ModuleType("docling.document_converter")
    converter.DocumentConverter = DocumentConverter  # type: ignore[attr-defined]
    converter.ImageFormatOption = ImageFormatOption  # type: ignore[attr-defined]
    pipeline = ModuleType("docling.pipeline")
    standard = ModuleType("docling.pipeline.standard_pdf_pipeline")
    standard.StandardPdfPipeline = object  # type: ignore[attr-defined]
    monkeypatch.setattr(
        "kg_doc_parser.workflow_ingest.ocr_providers._docling_pipeline_options",
        lambda _engine: object(),
    )
    for name, module in {
        "docling": docling,
        "docling.datamodel": datamodel,
        "docling.datamodel.base_models": base_models,
        "docling.document_converter": converter,
        "docling.pipeline": pipeline,
        "docling.pipeline.standard_pdf_pipeline": standard,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    response = run_docling_ocr_page(
        image_path,
        1,
        WorkflowProviderSettings(ocr=ProviderEndpointConfig(provider="docling", model="docling")),
        engine="rapidocr",
    )

    assert [cluster.text for cluster in response.OCR_text_clusters] == ["heading"]
    assert response.OCR_text_clusters[0].bb_x_min == 0
    assert response.OCR_text_clusters[0].bb_x_max == 40
    assert [item.description for item in response.non_text_objects] == ["| a | b |", "picture"]
    assert response.non_text_objects[0].bb_x_max == 100
    assert response.non_text_objects[0].bb_y_max == 80
    assert response.meaningful_ordering == [0, 1, 2]
    assert response.contains_table is True
    assert response.unresolved_item_count == 1
    assert response.source_item_count == 4

    class PlaceholderDocument:
        def iterate_items(self) -> list[Item]:
            return [Item("text", "heading\n\n<!-- image -->", BBox(0, 40, 5, 20))]

    Conversion.document = PlaceholderDocument()
    placeholder_response = run_docling_ocr_page(
        image_path,
        1,
        WorkflowProviderSettings(ocr=ProviderEndpointConfig(provider="docling", model="docling")),
        engine="rapidocr",
    )
    assert [cluster.text for cluster in placeholder_response.OCR_text_clusters] == ["heading"]
    assert placeholder_response.non_text_objects == []
    assert placeholder_response.unresolved_item_count == 1
    assert placeholder_response.source_item_count == 2

    class FallbackDocument:
        def iterate_items(self) -> list[Item]:
            return []

        def export_to_markdown(self) -> str:
            return "heading\n\n<!-- image -->"

    Conversion.document = FallbackDocument()
    fallback_response = run_docling_ocr_page(
        image_path,
        1,
        WorkflowProviderSettings(ocr=ProviderEndpointConfig(provider="docling", model="docling")),
        engine="rapidocr",
    )
    assert [cluster.text for cluster in fallback_response.OCR_text_clusters] == ["heading"]
    assert fallback_response.unresolved_item_count == 1
    assert fallback_response.source_item_count == 2


def test_registered_provider_is_explicit_and_process_local(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"image")
    expected = OCRClusterResponse(
        OCR_text_clusters=[],
        non_text_objects=[],
        is_empty_page=True,
        printed_page_number=None,
        meaningful_ordering=[],
        page_x_min=0.0,
        page_x_max=1.0,
        page_y_min=0.0,
        page_y_max=1.0,
        estimated_rotation_degrees=0.0,
        incomplete_words_on_edge=False,
        incomplete_text=False,
        data_loss_likelihood=0.0,
        scan_quality="high",
        contains_table=False,
    )

    def provider(
        path: Path, page_number: int, provider_settings: WorkflowProviderSettings, /
    ) -> OCRClusterResponse:
        assert path == image_path and page_number == 1
        assert provider_settings.ocr.provider == "custom"
        return expected

    register_ocr_provider("test-vlm", provider)
    try:
        result = run_registered_ocr_page(
            image_path,
            1,
            WorkflowProviderSettings(ocr=ProviderEndpointConfig(provider="custom", model="test-vlm")),
            provider_name="test-vlm",
        )
        assert result == expected
    finally:
        unregister_ocr_provider("test-vlm")
    with pytest.raises(RuntimeError, match="no OCR provider is registered"):
        run_registered_ocr_page(
            image_path,
            1,
            WorkflowProviderSettings(ocr=ProviderEndpointConfig(provider="custom", model="test-vlm")),
            provider_name="test-vlm",
        )


def test_kserve_v2_adapter_posts_image_and_decodes_canonical_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"image")
    response_model = OCRClusterResponse(
        OCR_text_clusters=[],
        non_text_objects=[],
        is_empty_page=True,
        printed_page_number=None,
        meaningful_ordering=[],
        page_x_min=0.0,
        page_x_max=1.0,
        page_y_min=0.0,
        page_y_max=1.0,
        estimated_rotation_degrees=0.0,
        incomplete_words_on_edge=False,
        incomplete_text=False,
        data_loss_likelihood=0.0,
        scan_quality="high",
        contains_table=False,
    )
    captured: dict[str, object] = {}

    class Response:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"outputs": [{"data": [json.dumps(response_model.model_dump(dump_format="json"))]}]}).encode()

    def fake_urlopen(request: object, *, timeout: float) -> Response:
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(
            provider="kserve",
            model="ocr-model",
            base_url="http://kserve.test",
            timeout_seconds=7.0,
        )
    )
    result = run_kserve_v2_page(image_path, 1, settings)
    request = captured["request"]
    assert result == response_model
    assert request.full_url == "http://kserve.test/v2/models/ocr-model/infer"
    body = json.loads(request.data.decode())
    assert body["inputs"][0]["datatype"] == "BYTES"


def test_kserve_v2_adapter_rejects_unknown_output_shape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"image")

    class Response:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"outputs": [{"data": ["{\\"blocks\\": []}"]}]}'

    monkeypatch.setattr("urllib.request.urlopen", lambda request, *, timeout: Response())
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="kserve", model="ocr-model", base_url="http://kserve.test")
    )
    with pytest.raises(ValueError, match="validation error|OCRClusterResponse"):
        run_kserve_v2_page(image_path, 1, settings)
