"""Optional OCR provider contracts and OCR engine adapters.

The workflow still owns resume state, artifacts, and semantic ingestion.  This
module only answers two questions: which OCR providers are available, and how
to invoke an optional OCR engine for one image.  All vendor imports remain
lazy so OCR is deliberately a runtime dependency, not a dependency of the
parser's base installation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

from ..models import OCRClusterResponse, TextCluster
from .providers import ProviderEndpointConfig, WorkflowProviderSettings

DoclingEngine = Literal["rapidocr", "easyocr", "tesseract", "tesserocr", "ocrmac", "nemotron-ocr"]
OCRProviderName = Literal[
    "anthropic",
    "codex",
    "docling",
    "rapidocr",
    "easyocr",
    "fake",
    "gemini",
    "ollama",
    "openai",
    "bonsai",
    "azure",
    "vertex",
]


@dataclass(frozen=True, slots=True)
class OCRProviderCapabilities:
    """Stable, descriptive capabilities; these do not grant authority."""

    provider: OCRProviderName
    engine: str | None
    vision: bool
    local: bool
    coordinates: bool
    structured_output: bool
    optional_dependency: str | None
    platforms: tuple[str, ...]


class OCRPageProvider(Protocol):
    def __call__(
        self,
        image_path: Path,
        page_number: int,
        provider_settings: WorkflowProviderSettings,
        /,
    ) -> OCRClusterResponse: ...


_DOCLING_CAPABILITIES: tuple[OCRProviderCapabilities, ...] = (
    OCRProviderCapabilities("docling", "rapidocr", True, True, False, False, "docling", ("linux", "windows", "macos")),
    OCRProviderCapabilities("docling", "easyocr", True, True, False, False, "docling[easyocr]", ("linux", "windows", "macos")),
    OCRProviderCapabilities("docling", "tesseract", True, True, False, False, "docling + tesseract", ("linux", "windows", "macos")),
    OCRProviderCapabilities("docling", "tesserocr", True, True, False, False, "docling[tesserocr]", ("linux", "windows", "macos")),
    OCRProviderCapabilities("docling", "ocrmac", True, True, False, False, "docling[ocrmac]", ("macos",)),
    OCRProviderCapabilities("docling", "nemotron-ocr", True, True, False, False, "docling[nemotron]", ("linux",)),
)

_DIRECT_CAPABILITIES: tuple[OCRProviderCapabilities, ...] = (
    OCRProviderCapabilities("rapidocr", None, True, True, True, False, "rapidocr-onnxruntime", ("linux", "windows", "macos")),
    OCRProviderCapabilities("easyocr", None, True, True, True, False, "easyocr", ("linux", "windows", "macos")),
)

_CHAT_CAPABILITIES: tuple[OCRProviderCapabilities, ...] = (
    OCRProviderCapabilities("bonsai", None, True, True, False, True, "langchain-openai", ("linux", "windows", "macos")),
    OCRProviderCapabilities("gemini", None, True, False, False, True, "langchain-google-genai", ("linux", "windows", "macos")),
    OCRProviderCapabilities("ollama", None, True, True, False, True, "langchain-ollama", ("linux", "windows", "macos")),
    OCRProviderCapabilities("openai", None, True, False, False, True, "langchain-openai", ("linux", "windows", "macos")),
    OCRProviderCapabilities("azure", None, True, False, False, True, "langchain-openai", ("linux", "windows", "macos")),
    OCRProviderCapabilities("vertex", None, True, False, False, True, "langchain-google-vertexai", ("linux", "windows", "macos")),
    OCRProviderCapabilities("anthropic", None, True, False, False, True, "langchain-anthropic", ("linux", "windows", "macos")),
    OCRProviderCapabilities("codex", None, True, False, False, True, None, ("linux", "windows", "macos")),
    OCRProviderCapabilities("fake", None, False, True, False, True, None, ("linux", "windows", "macos")),
)


def list_ocr_provider_capabilities() -> tuple[OCRProviderCapabilities, ...]:
    """Return the complete provider inventory without importing optional SDKs."""

    return _DOCLING_CAPABILITIES + _DIRECT_CAPABILITIES + _CHAT_CAPABILITIES


def normalize_ocr_provider(value: str, *, engine: str | None = None) -> tuple[OCRProviderName, str | None]:
    """Normalize provider aliases and reject unknown Docling engines early."""

    provider = value.strip().lower()
    if provider.startswith("docling:"):
        provider, engine = provider.split(":", 1)
    if provider == "bonsai2":
        provider = "bonsai"
    valid = {item.provider for item in _CHAT_CAPABILITIES} | {"docling"} | {
        item.provider for item in _DIRECT_CAPABILITIES
    }
    if provider not in valid:
        raise ValueError(f"unsupported OCR provider: {value!r}")
    if provider == "docling":
        selected = (engine or "rapidocr").strip().lower()
        if selected not in {item.engine for item in _DOCLING_CAPABILITIES}:
            raise ValueError(f"unsupported Docling OCR engine: {selected!r}")
        return "docling", selected
    if provider in {"rapidocr", "easyocr"}:
        if engine is not None:
            raise ValueError(f"direct OCR provider does not accept an engine: {provider!r}")
        return cast(OCRProviderName, provider), None
    return cast(OCRProviderName, provider), None


def _image_size(image_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            return image.size
    except (ImportError, OSError, ValueError) as exc:
        raise RuntimeError(f"cannot inspect OCR image {image_path}") from exc


def _box_to_bounds(box: object) -> tuple[float, float, float, float]:
    """Convert a four-point OCR polygon into an axis-aligned pixel box."""

    try:
        points = tuple(tuple(float(value) for value in point) for point in box)  # type: ignore[union-attr]
        coordinates = tuple(value for point in points for value in point)
        if len(points) < 4 or len(coordinates) < 8:
            raise ValueError
        xs = coordinates[0::2]
        ys = coordinates[1::2]
        return min(xs), max(xs), min(ys), max(ys)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("OCR provider returned an invalid quadrilateral") from exc


def _direct_response(
    image_path: Path,
    detections: list[tuple[object, str]],
) -> OCRClusterResponse:
    width, height = _image_size(image_path)
    normalized: list[tuple[float, float, float, float, str]] = []
    for box, text in detections:
        value = text.strip()
        if not value:
            continue
        x_min, x_max, y_min, y_max = _box_to_bounds(box)
        normalized.append((x_min, y_min, x_max, y_max, value))
    normalized.sort(key=lambda item: (item[1], item[0]))
    clusters = [
        TextCluster(
            text=text,
            bb_x_min=max(0.0, min(float(width), x_min)),
            bb_x_max=max(0.0, min(float(width), x_max)),
            bb_y_min=max(0.0, min(float(height), y_min)),
            bb_y_max=max(0.0, min(float(height), y_max)),
            cluster_number=index,
        )
        for index, (x_min, y_min, x_max, y_max, text) in enumerate(normalized)
    ]
    return OCRClusterResponse(
        OCR_text_clusters=clusters,
        non_text_objects=[],
        is_empty_page=not bool(clusters),
        printed_page_number=None,
        meaningful_ordering=list(range(len(clusters))),
        page_x_min=0.0,
        page_x_max=float(width),
        page_y_min=0.0,
        page_y_max=float(height),
        estimated_rotation_degrees=0.0,
        incomplete_words_on_edge=False,
        incomplete_text=False,
        data_loss_likelihood=0.0,
        scan_quality="medium",
        contains_table=False,
    )


def run_rapidocr_page(image_path: Path, page_number: int, provider_settings: ProviderEndpointConfig) -> OCRClusterResponse:
    """Run RapidOCR directly, without importing or invoking Docling."""

    del page_number, provider_settings
    try:
        from rapidocr import RapidOCR  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError("RapidOCR selected but rapidocr-onnxruntime is not installed") from exc
    result = RapidOCR()(str(image_path))
    texts = getattr(result, "txts", ())
    boxes = getattr(result, "boxes", ())
    return _direct_response(image_path, list(zip(boxes, texts, strict=False)))


def run_easyocr_page(image_path: Path, page_number: int, provider_settings: ProviderEndpointConfig) -> OCRClusterResponse:
    """Run EasyOCR directly, without importing or invoking Docling."""

    del page_number
    try:
        from easyocr import Reader  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError("EasyOCR selected but easyocr is not installed") from exc
    languages = tuple(
        language.strip()
        for language in getattr(provider_settings, "ocr_language", "en").split(",")
        if language.strip()
    ) or ("en",)
    reader = Reader(list(languages), gpu=False, verbose=False)
    raw_results = reader.readtext(str(image_path), detail=1)
    detections = [(item[0], str(item[1])) for item in raw_results]
    return _direct_response(image_path, detections)


def _docling_pipeline_options(engine: str) -> object:
    """Build one official Docling pipeline option object lazily."""

    try:
        from docling.datamodel.pipeline_options import (  # pyright: ignore[reportMissingImports]
            EasyOcrOptions,
            NemotronOcrOptions,
            OcrMacOptions,
            OcrMode,
            PdfPipelineOptions,
            RapidOcrOptions,
            TesseractCliOcrOptions,
            TesseractOcrOptions,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Docling OCR selected but Docling is not installed; install docling "
            "and the selected engine's optional runtime"
        ) from exc
    option_types: Mapping[str, type[object]] = {
        "rapidocr": RapidOcrOptions,
        "easyocr": EasyOcrOptions,
        "tesseract": TesseractCliOcrOptions,
        "tesserocr": TesseractOcrOptions,
        "ocrmac": OcrMacOptions,
        "nemotron-ocr": NemotronOcrOptions,
    }
    options = PdfPipelineOptions()
    options.do_ocr = True
    options.ocr_options = option_types[engine](mode=OcrMode.FULL_PAGE)
    return options


def run_docling_ocr_page(
    image_path: Path,
    page_number: int,
    provider_settings: WorkflowProviderSettings,
    *,
    engine: str,
) -> OCRClusterResponse:
    """OCR one image through a selected Docling engine.

    Docling's document text is authoritative here, while the current parser
    contract only promises page-level geometry.  We therefore use the image's
    full-page bounds rather than inventing word coordinates.
    """

    try:
        from docling.datamodel.base_models import (  # pyright: ignore[reportMissingImports]
            InputFormat,
        )
        from docling.document_converter import (  # pyright: ignore[reportMissingImports]
            DocumentConverter,
            ImageFormatOption,
        )
        from docling.pipeline.standard_pdf_pipeline import (  # pyright: ignore[reportMissingImports]
            StandardPdfPipeline,
        )
    except ImportError as exc:
        raise RuntimeError("Docling OCR selected but the optional docling package is unavailable") from exc
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot inspect OCR image {image_path}") from exc
    pipeline_options = _docling_pipeline_options(engine)
    converter = DocumentConverter(
        format_options={
            InputFormat.IMAGE: ImageFormatOption(
                pipeline_cls=StandardPdfPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )
    result = converter.convert(str(image_path))
    document = result.document
    export = getattr(document, "export_to_markdown", None)
    text = str(export() if callable(export) else getattr(document, "text", "")).strip()
    clusters: list[TextCluster] = []
    if text:
        clusters.append(
            TextCluster(
                text=text,
                bb_x_min=0.0,
                bb_x_max=float(width),
                bb_y_min=0.0,
                bb_y_max=float(height),
                cluster_number=0,
            )
        )
    return OCRClusterResponse(
        OCR_text_clusters=clusters,
        non_text_objects=[],
        is_empty_page=not bool(clusters),
        printed_page_number=None,
        meaningful_ordering=[0] if clusters else [],
        page_x_min=0.0,
        page_x_max=float(width),
        page_y_min=0.0,
        page_y_max=float(height),
        estimated_rotation_degrees=0.0,
        incomplete_words_on_edge=False,
        incomplete_text=False,
        data_loss_likelihood=0.0,
        scan_quality="medium",
        contains_table=False,
    )

