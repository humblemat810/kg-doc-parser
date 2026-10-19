"""Single-page OCR provider execution and response normalization."""

from __future__ import annotations

import json
import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image
from pydantic import ValidationError
from pypdf import PdfReader

from ..llm_structured_output import build_structured_output_runnable
from ..models import NonTextCluster, OCRClusterResponse, TextCluster
from .ocr_inputs import image_to_data_url
from .ocr_pipeline_contracts import OCRRunner
from .ocr_quality_policy import OCRQualityPolicyRejection
from .providers import (
    ProviderDiagnostics,
    ProviderDiagnosticsSink,
    WorkflowProviderSettings,
    build_chat_model_for_role,
    invoke_with_timeout,
)
from .sdk_boundary import raw_content
from .serialization import JsonObject

_LOGGER = logging.getLogger(__name__)


def _coerce_ocr_response(payload: object) -> OCRClusterResponse:
    """Coerce a structured-output payload into the OCR model."""
    if isinstance(payload, OCRClusterResponse):
        return payload
    if isinstance(payload, dict):
        parsed = payload.get("parsed", payload)
        if parsed is not None:
            return OCRClusterResponse.model_validate(parsed)
        raw = payload.get("raw")
        if raw is not None:
            content = raw_content(raw)
            if isinstance(content, str):
                return OCRClusterResponse.model_validate(json.loads(content))
            if isinstance(content, list):
                text = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
                if text.strip():
                    return OCRClusterResponse.model_validate(json.loads(text))
    return OCRClusterResponse.model_validate(payload)


def _extract_message_text(raw: object) -> str:
    """Extract plain text from LangChain raw message content."""
    content = raw_content(raw)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def _minimal_ocr_response_from_text(
    *, text: str, page_number: int, image_path: Path
) -> OCRClusterResponse:
    """Create a coarse page-wide OCR result when only raw text is available."""
    normalized_text = text.strip()
    with Image.open(image_path) as image:
        width, height = image.size
    if not normalized_text or normalized_text == "{}":
        return OCRClusterResponse(
            OCR_text_clusters=[],
            non_text_objects=[],
            is_empty_page=True,
            printed_page_number=str(page_number),
            meaningful_ordering=[],
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
            source_item_count=0,
            spatial_accuracy="page",
        )
    return OCRClusterResponse(
        OCR_text_clusters=[
            TextCluster(
                text=normalized_text,
                bb_x_min=0.0,
                bb_x_max=float(width),
                bb_y_min=0.0,
                bb_y_max=float(height),
                cluster_number=0,
            )
        ],
        non_text_objects=[],
        is_empty_page=False,
        printed_page_number=str(page_number),
        meaningful_ordering=[0],
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
        source_item_count=1,
        spatial_accuracy="page",
    )


def _retry_feedback_from_attempts(
    attempts: Sequence[JsonObject], *, max_items: int = 2, max_chars: int = 1200
) -> tuple[str, ...]:
    """Return bounded prior OCR failures suitable for the next prompt."""
    feedback: list[str] = []
    for attempt in reversed(attempts):
        message = attempt.get("retry_feedback")
        if not isinstance(message, str) or not message.strip():
            continue
        model = attempt.get("model_name")
        label = model if isinstance(model, str) and model else "previous provider"
        feedback.append(f"{label}: {message.strip()[:max_chars]}")
        if len(feedback) >= max_items:
            break
    return tuple(reversed(feedback))


def _safe_ocr_retry_feedback(error: Exception, *, max_chars: int = 1200) -> str | None:
    """Extract bounded OCR validation details, excluding provider errors."""
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, ValidationError) and current.title in {
            "OCRClusterResponse",
            "RawOCRResponse",
            "OCRClusterResponseMetaless",
        }:
            details: list[str] = []
            for item in current.errors(include_input=False, include_context=False):
                location = item.get("loc", ())
                field_path = ".".join(str(part) for part in location)
                message = str(item.get("msg", "OCR output failed schema validation"))
                if not field_path and "meaningful_ordering must be" in message:
                    field_path = "meaningful_ordering"
                    message = message.removeprefix("Value error, ")
                details.append(f"{field_path}: {message}" if field_path else message)
            return "; ".join(details)[:max_chars] or "OCR output failed schema validation"
        if isinstance(current, OCRQualityPolicyRejection):
            safe_codes = [code for code in current.codes if code.replace("_", "").isalnum()]
            if safe_codes:
                return (
                    "OCR quality checks rejected the response; address these "
                    f"diagnostic codes: {', '.join(safe_codes)}"
                )[:max_chars]
        current = current.__cause__ or current.__context__
    return None


def _ocr_direct_pdf_image_regions(
    *,
    pdf_path: Path,
    image_regions: tuple[tuple[float, float, float, float], ...],
    source_image_path: Path,
    page_number: int,
    runner: OCRRunner,
    settings: WorkflowProviderSettings,
) -> OCRClusterResponse | None:
    """OCR placed PDF image regions when page content is fully accounted for."""
    if not image_regions:
        return None
    reader = PdfReader(str(pdf_path))
    page = reader.pages[page_number - 1]
    pdf_width = float(page.mediabox.width)
    pdf_height = float(page.mediabox.height)
    if pdf_width <= 0 or pdf_height <= 0:
        return None
    with Image.open(source_image_path) as source_image:
        page_width, page_height = source_image.size
        text_clusters: list[TextCluster] = []
        non_text_objects: list[NonTextCluster] = []
        responses: list[OCRClusterResponse] = []
        with tempfile.TemporaryDirectory(prefix="kgdoc-pdf-image-region-") as temp_dir:
            for index, (x_min, y_min, x_max, y_max) in enumerate(image_regions):
                left = max(0, min(page_width, round(x_min * page_width / pdf_width)))
                right = max(0, min(page_width, round(x_max * page_width / pdf_width)))
                top = max(0, min(page_height, round((pdf_height - y_max) * page_height / pdf_height)))
                bottom = max(0, min(page_height, round((pdf_height - y_min) * page_height / pdf_height)))
                if right <= left or bottom <= top:
                    return None
                crop_path = Path(temp_dir) / f"region-{index}.png"
                source_image.crop((left, top, right, bottom)).save(crop_path, "PNG")
                response = runner(crop_path, page_number, settings)
                local_width = response.page_x_max - response.page_x_min
                local_height = response.page_y_max - response.page_y_min
                if local_width <= 0 or local_height <= 0:
                    return None
                responses.append(response)

                def rebase(
                    item: TextCluster | NonTextCluster,
                    *,
                    page_left: int = left,
                    page_top: int = top,
                    page_right: int = right,
                    page_bottom: int = bottom,
                    local_page: OCRClusterResponse = response,
                    local_width_px: float = local_width,
                    local_height_px: float = local_height,
                ) -> dict[str, float | int]:
                    return {
                        "bb_x_min": page_left + (item.bb_x_min - local_page.page_x_min) * (page_right - page_left) / local_width_px,
                        "bb_x_max": page_left + (item.bb_x_max - local_page.page_x_min) * (page_right - page_left) / local_width_px,
                        "bb_y_min": page_top + (item.bb_y_min - local_page.page_y_min) * (page_bottom - page_top) / local_height_px,
                        "bb_y_max": page_top + (item.bb_y_max - local_page.page_y_min) * (page_bottom - page_top) / local_height_px,
                        "cluster_number": len(text_clusters) + len(non_text_objects),
                    }

                text_clusters.extend(item.model_copy(update=rebase(item)) for item in response.OCR_text_clusters)
                non_text_objects.extend(item.model_copy(update=rebase(item)) for item in response.non_text_objects)
    clusters: list[TextCluster | NonTextCluster] = [*text_clusters, *non_text_objects]
    clusters.sort(key=lambda item: (item.bb_y_min, item.bb_x_min, item.cluster_number))
    text_clusters = []
    non_text_objects = []
    for cluster_number, item in enumerate(clusters):
        item = item.model_copy(update={"cluster_number": cluster_number})
        if isinstance(item, TextCluster):
            text_clusters.append(item)
        else:
            non_text_objects.append(item)
    if not responses:
        return None
    first = responses[0]
    accuracy_rank = {"page": 0, "region": 1, "line": 2, "word": 3, "character": 4}
    accuracy = min((response.spatial_accuracy for response in responses), key=lambda value: accuracy_rank[value])
    sorted_items = sorted([*text_clusters, *non_text_objects], key=lambda item: (item.bb_y_min, item.bb_x_min, item.cluster_number))
    payload = first.model_dump()
    payload.update(
        {
            "OCR_text_clusters": text_clusters,
            "non_text_objects": non_text_objects,
            "is_empty_page": not sorted_items,
            "meaningful_ordering": [item.cluster_number for item in sorted_items],
            "page_x_min": 0,
            "page_x_max": page_width,
            "page_y_min": 0,
            "page_y_max": page_height,
            "estimated_rotation_degrees": 0,
            "incomplete_words_on_edge": any(response.incomplete_words_on_edge for response in responses),
            "incomplete_text": any(response.incomplete_text for response in responses),
            "data_loss_likelihood": max(response.data_loss_likelihood for response in responses),
            "scan_quality": min((response.scan_quality for response in responses), key=lambda value: {"low": 0, "medium": 1, "high": 2}[value]),
            "contains_table": any(response.contains_table for response in responses),
            "unresolved_item_count": sum(response.unresolved_item_count for response in responses),
            "source_item_count": len(sorted_items) + sum(response.unresolved_item_count for response in responses),
            "spatial_accuracy": accuracy,
        }
    )
    return OCRClusterResponse.model_validate(payload)


def _run_live_ocr_page(
    image_path: Path,
    page_number: int,
    provider_settings: WorkflowProviderSettings,
    *,
    provider_diagnostics_sink: ProviderDiagnosticsSink | None = None,
    retry_feedback: Sequence[str] = (),
) -> OCRClusterResponse:
    """Run one page image through the configured OCR provider."""
    if provider_settings.ocr.provider == "docling":
        from .ocr_providers import run_docling_ocr_page

        return run_docling_ocr_page(image_path, page_number, provider_settings, engine=provider_settings.ocr.ocr_engine or "rapidocr")
    if provider_settings.ocr.provider == "rapidocr":
        from .ocr_providers import run_rapidocr_page

        return run_rapidocr_page(image_path, page_number, provider_settings.ocr)
    if provider_settings.ocr.provider == "easyocr":
        from .ocr_providers import run_easyocr_page

        return run_easyocr_page(image_path, page_number, provider_settings.ocr)
    if provider_settings.ocr.provider == "kserve":
        from .ocr_providers import run_kserve_v2_page

        return run_kserve_v2_page(image_path, page_number, provider_settings)
    if provider_settings.ocr.provider == "custom":
        from .ocr_providers import run_registered_ocr_page

        return run_registered_ocr_page(image_path, page_number, provider_settings, provider_name=provider_settings.ocr.model)
    chat = build_chat_model_for_role("ocr", provider_settings)
    structured = build_structured_output_runnable(chat, OCRClusterResponse, include_raw=True)
    prompt = (
        "You are an OCR model for workflow ingest.\n"
        "Return structured OCR for one page image.\n"
        "Set meaningful_ordering to the authoritative reading order from first "
        "to last meaningful text or non-text region; include every cluster "
        "number exactly once and no others. Preserve cluster numbering and "
        "non-text regions.\n"
        "Do not invent missing text. If the page is empty, mark it as empty."
    )
    if retry_feedback:
        prompt += (
            "\nA previous attempt was rejected. Correct these bounded validation "
            "diagnostics without omitting any source region:\n"
            + "\n".join(f"- {message}" for message in retry_feedback)
        )
    call_diagnostics: ProviderDiagnostics = {}
    try:
        response = invoke_with_timeout(
            lambda: structured.invoke(
                [
                    SystemMessage(content=prompt),
                    HumanMessage(content=[
                        {"type": "text", "text": f"OCR page {page_number} and return the structured schema."},
                        {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                    ]),
                ]
            ),
            timeout_seconds=provider_settings.ocr.timeout_seconds,
            diagnostics=call_diagnostics,
            operation="ocr_page",
            max_in_flight=provider_settings.ocr.max_in_flight_calls,
            attempt_index=1,
            call_role="ocr",
            strategy="ocr",
        )
    except Exception as exc:
        call_diagnostics.update({"success": False, "failure_type": call_diagnostics.get("failure_type", "transport/provider_exception"), "error_type": type(exc).__name__, "error": str(exc)})
        if provider_diagnostics_sink is not None:
            provider_diagnostics_sink(dict(call_diagnostics))
        raise
    try:
        result = _coerce_ocr_response(response)
    except Exception as exc:
        if isinstance(response, dict):
            raw = response.get("raw")
            raw_text = _extract_message_text(raw) if raw is not None else ""
            if raw_text:
                _LOGGER.info("ocr structured parse fallback | provider=%s model=%s page=%s | parsing_error=%s", provider_settings.ocr.provider, provider_settings.ocr.model, page_number, response.get("parsing_error"))
                result = _minimal_ocr_response_from_text(text=raw_text, page_number=page_number, image_path=image_path)
                call_diagnostics.update({"success": False, "failure_type": "structured_output_parse_failure", "fallback": "raw_text_ocr", "error_type": type(exc).__name__})
                if provider_diagnostics_sink is not None:
                    provider_diagnostics_sink(dict(call_diagnostics))
                return result
        call_diagnostics.update({"success": False, "failure_type": "structured_output_parse_failure", "error_type": type(exc).__name__, "error": str(exc)})
        if provider_diagnostics_sink is not None:
            provider_diagnostics_sink(dict(call_diagnostics))
        raise RuntimeError(f"ocr provider returned an unreadable response for page {page_number}: {exc}") from exc
    if provider_diagnostics_sink is not None:
        provider_diagnostics_sink(dict(call_diagnostics))
    return result
