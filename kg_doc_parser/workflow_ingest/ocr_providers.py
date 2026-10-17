"""Optional OCR provider contracts and OCR engine adapters.

The workflow still owns resume state, artifacts, and semantic ingestion.  This
module only answers two questions: which OCR providers are available, and how
to invoke an optional OCR engine for one image.  All vendor imports remain
lazy so OCR is deliberately a runtime dependency, not a dependency of the
parser's base installation.
"""

from __future__ import annotations

import base64
import json
import os
import platform
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

from ..models import NonTextCluster, OCRClusterResponse, TextCluster
from .providers import ProviderEndpointConfig, WorkflowProviderSettings

DoclingEngine = Literal["rapidocr", "easyocr", "tesseract", "tesserocr", "ocrmac", "nemotron-ocr"]
CoordinatePrecision = Literal["page", "region", "token", "table_cell", "unknown"]
OCRProviderName = Literal[
    "anthropic",
    "codex",
    "custom",
    "kserve",
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
    coordinate_precision: CoordinatePrecision = "unknown"


class OCRPageProvider(Protocol):
    def __call__(
        self,
        image_path: Path,
        page_number: int,
        provider_settings: WorkflowProviderSettings,
        /,
    ) -> OCRClusterResponse: ...


OCRResponseDecoder = Callable[[Mapping[str, object]], OCRClusterResponse]

_REGISTERED_PROVIDERS: dict[str, OCRPageProvider] = {}


def _external_attribute(value: object, name: str) -> object | None:
    """Read an optional vendor attribute at the untyped SDK boundary."""

    return getattr(value, name, None)


def _as_external_sequence(value: object) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return ()
    return value


def _external_sequence(value: object, name: str) -> Sequence[object]:
    return _as_external_sequence(_external_attribute(value, name))


def _external_text(value: object, name: str, default: str = "") -> str:
    candidate = _external_attribute(value, name)
    return candidate if isinstance(candidate, str) else default


def _external_callable(value: object, name: str) -> Callable[[], object] | None:
    candidate = _external_attribute(value, name)
    return candidate if callable(candidate) else None


def _external_number(value: object, name: str) -> int | float | None:
    candidate = _external_attribute(value, name)
    return candidate if isinstance(candidate, (int, float)) else None


def register_ocr_provider(name: str, provider: OCRPageProvider) -> None:
    """Register a named user provider for explicit ``custom`` OCR calls.

    Registration only affects the current process.  The provider still has to
    return the existing ``OCRClusterResponse`` contract; it does not bypass
    workflow validation or checkpointing.
    """

    normalized = name.strip()
    if not normalized or normalized in {"custom", "kserve"}:
        raise ValueError("custom OCR provider names must be nonempty and non-reserved")
    _REGISTERED_PROVIDERS[normalized] = provider


def unregister_ocr_provider(name: str) -> None:
    """Remove a process-local provider registration if present."""

    _REGISTERED_PROVIDERS.pop(name.strip(), None)


def _registered_ocr_provider(name: str) -> OCRPageProvider:
    try:
        return _REGISTERED_PROVIDERS[name]
    except KeyError as exc:
        raise RuntimeError(f"no OCR provider is registered as {name!r}") from exc


_DOCLING_CAPABILITIES: tuple[OCRProviderCapabilities, ...] = (
    OCRProviderCapabilities("docling", "rapidocr", True, True, True, False, "docling", ("linux", "windows", "macos"), "region"),
    OCRProviderCapabilities("docling", "easyocr", True, True, True, False, "docling[easyocr]", ("linux", "windows", "macos"), "region"),
    OCRProviderCapabilities("docling", "tesseract", True, True, True, False, "docling + tesseract", ("linux", "windows", "macos"), "region"),
    OCRProviderCapabilities("docling", "tesserocr", True, True, True, False, "docling[tesserocr]", ("linux", "windows", "macos"), "region"),
    OCRProviderCapabilities("docling", "ocrmac", True, True, True, False, "docling[ocrmac]", ("macos",), "region"),
    OCRProviderCapabilities("docling", "nemotron-ocr", True, True, True, False, "docling[nemotron]", ("linux",), "region"),
)

_DIRECT_CAPABILITIES: tuple[OCRProviderCapabilities, ...] = (
    OCRProviderCapabilities("rapidocr", None, True, True, True, False, "rapidocr-onnxruntime", ("linux", "windows", "macos"), "region"),
    OCRProviderCapabilities("easyocr", None, True, True, True, False, "easyocr", ("linux", "windows", "macos"), "region"),
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
    OCRProviderCapabilities("custom", None, True, False, False, True, None, ("linux", "windows", "macos")),
    OCRProviderCapabilities("kserve", None, True, False, False, True, "urllib", ("linux", "windows", "macos")),
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


def run_registered_ocr_page(
    image_path: Path,
    page_number: int,
    provider_settings: WorkflowProviderSettings,
    *,
    provider_name: str,
) -> OCRClusterResponse:
    """Invoke one explicitly registered user OCR provider."""

    return _registered_ocr_provider(provider_name)(image_path, page_number, provider_settings)


def _default_kserve_response_decoder(payload: Mapping[str, object]) -> OCRClusterResponse:
    """Decode a KServe response whose output contains the public OCR model.

    KServe does not prescribe an OCR tensor schema.  This decoder accepts the
    common ``outputs[*].data`` JSON-string/object form and then validates the
    result as the parser's stable response model.  Model-specific tensor
    layouts should use a registered provider instead of being guessed here.
    """

    outputs = payload.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("KServe OCR response must contain a non-empty outputs list")
    first = outputs[0]
    if not isinstance(first, Mapping):
        raise TypeError("KServe OCR response output must be an object")
    data = first.get("data")
    if isinstance(data, list) and len(data) == 1:
        data = data[0]
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValueError("KServe OCR output data is not JSON") from exc
    if not isinstance(data, Mapping):
        raise TypeError("KServe OCR output data must contain OCRClusterResponse JSON")
    return OCRClusterResponse.model_validate(data)


def run_kserve_v2_page(
    image_path: Path,
    page_number: int,
    provider_settings: WorkflowProviderSettings,
    *,
    response_decoder: OCRResponseDecoder = _default_kserve_response_decoder,
) -> OCRClusterResponse:
    """Call a KServe v2 ``/v2/models/{model}/infer`` OCR endpoint.

    The transport is dependency-free and deliberately does not infer a
    provider-specific tensor layout.  A custom decoder may be supplied for a
    model whose output is not the canonical OCR response object.
    """

    del page_number
    endpoint = provider_settings.ocr.base_url
    model = provider_settings.ocr.model.strip()
    if not endpoint or not model:
        raise ValueError("KServe OCR requires base_url and model")
    image_bytes = image_path.read_bytes()
    payload = {
        "inputs": [
            {
                "name": "image",
                "shape": [1],
                "datatype": "BYTES",
                "data": [base64.b64encode(image_bytes).decode("ascii")],
            }
        ]
    }
    body = json.dumps(payload).encode("utf-8")
    url = endpoint.rstrip("/") + f"/v2/models/{model}/infer"
    headers = {"Content-Type": "application/json"}
    api_key_env = provider_settings.ocr.api_key_env
    if api_key_env:
        token = os.getenv(api_key_env)
        if token:
            headers["Authorization"] = f"Bearer {token}"
    from urllib.request import Request, urlopen

    request = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=provider_settings.ocr.timeout_seconds) as response:
            raw = response.read()
    except OSError as exc:
        raise RuntimeError(f"KServe OCR request failed for model {model!r}") from exc
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("KServe OCR response was not valid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise TypeError("KServe OCR response must be a JSON object")
    return response_decoder(decoded)


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
        if isinstance(box, (str, bytes)) or not isinstance(box, Sequence):
            raise TypeError
        points: list[tuple[float, float]] = []
        for point in box:
            if isinstance(point, (str, bytes)) or not isinstance(point, Sequence):
                raise TypeError
            if len(point) < 2:
                raise ValueError
            x, y = point[0], point[1]
            if isinstance(x, bool) or isinstance(y, bool):
                raise TypeError
            points.append((float(x), float(y)))
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
    detections: Sequence[tuple[object, str]],
) -> OCRClusterResponse:
    width, height = _image_size(image_path)
    normalized: list[tuple[float, float, float, float, str]] = []
    unresolved_item_count = 0
    for box, text in detections:
        value = text.strip()
        if not value:
            unresolved_item_count += 1
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
        unresolved_item_count=unresolved_item_count,
        source_item_count=len(clusters) + unresolved_item_count,
        spatial_accuracy="region",
    )


def run_rapidocr_page(image_path: Path, page_number: int, provider_settings: ProviderEndpointConfig) -> OCRClusterResponse:
    """Run RapidOCR directly, without importing or invoking Docling."""

    del page_number, provider_settings
    try:
        from rapidocr import RapidOCR  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError("RapidOCR selected but rapidocr-onnxruntime is not installed") from exc
    result = RapidOCR()(str(image_path))
    texts = _external_sequence(result, "txts")
    boxes = _external_sequence(result, "boxes")
    detections = [(box, str(text)) for box, text in zip(boxes, texts, strict=False)]
    return _direct_response(image_path, detections)


def run_easyocr_page(image_path: Path, page_number: int, provider_settings: ProviderEndpointConfig) -> OCRClusterResponse:
    """Run EasyOCR directly, without importing or invoking Docling."""

    del page_number
    try:
        from easyocr import Reader  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError("EasyOCR selected but easyocr is not installed") from exc
    languages = tuple(
        language.strip()
        for language in provider_settings.ocr_language.split(",")
        if language.strip()
    ) or ("en",)
    reader = Reader(list(languages), gpu=False, verbose=False)
    raw_results = reader.readtext(str(image_path), detail=1)
    detections: list[tuple[object, str]] = []
    for item in _as_external_sequence(raw_results):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            detections.append((item[0], str(item[1])))
    return _direct_response(image_path, detections)


def _docling_pipeline_options(engine: str) -> object:
    """Build one official Docling pipeline option object lazily."""

    system = platform.system().lower()
    if engine == "ocrmac" and system != "darwin":
        raise RuntimeError("Docling OcrMac is only available on macOS")
    if engine == "nemotron-ocr" and system != "linux":
        raise RuntimeError("Docling Nemotron-OCR is only supported on Linux")
    if engine == "tesseract" and shutil.which("tesseract") is None:
        raise RuntimeError("Docling Tesseract CLI selected but the tesseract executable is unavailable")

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

    Prefer Docling provenance boxes for each item.  The Markdown export is a
    fallback only for older/limited Docling adapters that expose no item
    provenance; it must not erase positional information when that information
    is available.
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
                pipeline_options=pipeline_options,  # pyright: ignore[reportArgumentType]
            )
        }
    )
    result = converter.convert(str(image_path))
    document = result.document
    clusters: list[TextCluster] = []
    non_text_objects: list[NonTextCluster] = []
    ordering: list[int] = []
    unresolved_item_count = 0

    def _box(item: object) -> tuple[float, float, float, float] | None:
        provenance = _external_attribute(item, "prov")
        records: Sequence[object] = _as_external_sequence(provenance) or (provenance,)
        for record in records:
            bbox = _external_attribute(record, "bbox")
            if bbox is None:
                continue
            left_value = _external_number(bbox, "l")
            left_value = left_value if left_value is not None else _external_number(bbox, "left")
            right_value = _external_number(bbox, "r")
            right_value = right_value if right_value is not None else _external_number(bbox, "right")
            top_value = _external_number(bbox, "t")
            top_value = top_value if top_value is not None else _external_number(bbox, "top")
            bottom_value = _external_number(bbox, "b")
            bottom_value = bottom_value if bottom_value is not None else _external_number(bbox, "bottom")
            raw_values = (left_value, right_value, top_value, bottom_value)
            numeric_values: list[float] = []
            for value in raw_values:
                if not isinstance(value, (int, float)):
                    break
                numeric_values.append(float(value))
            if len(numeric_values) != 4:
                continue
            left_value, right_value, top_value, bottom_value = numeric_values
            left = max(0.0, left_value)
            right = min(float(width), right_value)
            top = max(0.0, top_value)
            bottom = min(float(height), bottom_value)
            if right > left and bottom > top:
                return left, right, top, bottom
        return None

    iterate_items = _external_callable(document, "iterate_items")
    raw_items = iterate_items() if iterate_items is not None else ()
    items = list(_as_external_sequence(raw_items))
    unresolved_markers = 0
    represented_visual_items = 0
    for item_record in items:
        item = item_record[0] if isinstance(item_record, tuple) and item_record else item_record
        box = _box(item)
        if box is None:
            unresolved_item_count += 1
            continue
        text_value = _external_text(item, "text")
        if not text_value:
            export_item = _external_callable(item, "export_to_markdown")
            text_value = export_item() if export_item is not None else ""
        text = str(text_value or "").strip()
        label = _external_text(item, "label").lower()
        for marker in ("<!-- image -->", "<!-- table -->"):
            marker_count = text.count(marker)
            unresolved_markers += marker_count
            text = text.replace(marker, "").strip()
        if any(kind in label for kind in ("picture", "figure", "image", "table")):
            represented_visual_items += 1
        cluster_number = len(ordering)
        if text and "table" not in label and "picture" not in label and "figure" not in label:
            clusters.append(
                TextCluster(
                    text=text,
                    bb_x_min=box[0],
                    bb_x_max=box[1],
                    bb_y_min=box[2],
                    bb_y_max=box[3],
                    cluster_number=cluster_number,
                )
            )
        else:
            description = text or label or "Docling non-text item"
            non_text_objects.append(
                NonTextCluster(
                    description=description,
                    bb_x_min=box[0],
                    bb_x_max=box[1],
                    bb_y_min=box[2],
                    bb_y_max=box[3],
                    cluster_number=cluster_number,
                )
            )
        ordering.append(cluster_number)

    # Docling can expose a page-wide text item whose Markdown contains a
    # placeholder for a picture/table but omit the corresponding item from
    # ``iterate_items``. Do not silently claim that placeholder is grounded.
    unresolved_item_count += max(0, unresolved_markers - represented_visual_items)

    if not ordering:
        export = _external_callable(document, "export_to_markdown")
        text = str(export() if export is not None else _external_text(document, "text")).strip()
        for marker in ("<!-- image -->", "<!-- table -->"):
            marker_count = text.count(marker)
            unresolved_item_count += marker_count
            text = text.replace(marker, "").strip()
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
            ordering = [0]
    return OCRClusterResponse(
        OCR_text_clusters=clusters,
        non_text_objects=non_text_objects,
        is_empty_page=not bool(clusters or non_text_objects),
        printed_page_number=None,
        meaningful_ordering=ordering,
        page_x_min=0.0,
        page_x_max=float(width),
        page_y_min=0.0,
        page_y_max=float(height),
        estimated_rotation_degrees=0.0,
        incomplete_words_on_edge=False,
        incomplete_text=False,
        data_loss_likelihood=0.0,
        scan_quality="medium",
        contains_table=any("table" in _external_text(item, "label").lower() for item in items),
        unresolved_item_count=unresolved_item_count,
        source_item_count=len(clusters) + len(non_text_objects) + unresolved_item_count,
        spatial_accuracy="region" if ordering else "page",
    )

