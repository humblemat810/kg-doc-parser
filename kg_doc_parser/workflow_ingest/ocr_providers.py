"""Optional OCR provider contracts and Docling engine adapters.

The workflow still owns resume state, artifacts, and semantic ingestion.  This
module only answers two questions: which OCR providers are available, and how
to invoke an optional Docling engine for one image.  Docling is deliberately a
runtime dependency, not a dependency of the parser's base installation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

from ..models import OCRClusterResponse, TextCluster
from .providers import WorkflowProviderSettings

DoclingEngine = Literal["rapidocr", "easyocr", "tesseract", "tesserocr", "ocrmac", "nemotron-ocr"]
OCRProviderName = Literal[
    "anthropic",
    "codex",
    "docling",
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

_CHAT_CAPABILITIES: tuple[OCRProviderCapabilities, ...] = (
    OCRProviderCapabilities("bonsai", None, True, True, False, True, "langchain-openai", ("linux", "windows", "macos")),
    OCRProviderCapabilities("gemini", None, True, False, False, True, "langchain-google-genai", ("linux", "windows", "macos")),
    OCRProviderCapabilities("ollama", None, True, True, False, True, "langchain-ollama", ("linux", "windows", "macos")),
    OCRProviderCapabilities("openai", None, True, False, False, True, "langchain-openai", ("linux", "windows", "macos")),
    OCRProviderCapabilities("azure", None, True, False, False, True, "langchain-openai", ("linux", "windows", "macos")),
    OCRProviderCapabilities("vertex", None, True, False, False, True, "langchain-google-vertexai", ("linux", "windows", "macos")),
    OCRProviderCapabilities("anthropic", None, True, False, False, True, "langchain-anthropic", ("linux", "windows", "macos")),
    OCRProviderCapabilities("codex", None, False, False, False, True, None, ("linux", "windows", "macos")),
    OCRProviderCapabilities("fake", None, False, True, False, True, None, ("linux", "windows", "macos")),
)


def list_ocr_provider_capabilities() -> tuple[OCRProviderCapabilities, ...]:
    """Return the complete provider inventory without importing optional SDKs."""

    return _DOCLING_CAPABILITIES + _CHAT_CAPABILITIES


def normalize_ocr_provider(value: str, *, engine: str | None = None) -> tuple[OCRProviderName, str | None]:
    """Normalize provider aliases and reject unknown Docling engines early."""

    provider = value.strip().lower()
    if provider.startswith("docling:"):
        provider, engine = provider.split(":", 1)
    if provider == "bonsai2":
        provider = "bonsai"
    valid = {item.provider for item in _CHAT_CAPABILITIES} | {"docling"}
    if provider not in valid:
        raise ValueError(f"unsupported OCR provider: {value!r}")
    if provider == "docling":
        selected = (engine or "rapidocr").strip().lower()
        if selected not in {item.engine for item in _DOCLING_CAPABILITIES}:
            raise ValueError(f"unsupported Docling OCR engine: {selected!r}")
        return "docling", selected
    return cast(OCRProviderName, provider), None


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
        from docling.datamodel.pipeline_options import (  # pyright: ignore[reportMissingImports]
            ImageFormatOption,
        )
        from docling.document_converter import (  # pyright: ignore[reportMissingImports]
            DocumentConverter,
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

