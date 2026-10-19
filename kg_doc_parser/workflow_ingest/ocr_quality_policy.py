"""OCR provider-output quality policy and source-image screening."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from ..models import OCRClusterResponse


class OCRQualityPolicy(BaseModel):
    """Policy for turning provider and source-image checks into acceptance decisions.

    Strict mode rejects visible source ink with no OCR evidence and substantial
    ink regions outside reported boxes. These are conservative screening gates,
    not guarantees that the provider recognized every semantic item.
    """

    mode: Literal["strict", "permissive"] = Field(
        default="strict",
        description="Strict rejects source-image evidence loss; permissive records warnings.",
    )
    max_data_loss_likelihood: float = Field(default=1.0, ge=0.0, le=1.0)
    reject_incomplete_text: bool = False
    reject_incomplete_words_on_edge: bool = False
    reject_empty_page: bool = False
    reject_empty_output_on_nonblank_image: bool = False
    minimum_uncovered_ink_tiles: int = Field(
        default=3,
        ge=1,
        description="Minimum coarse ink tiles outside all provider boxes before flagging.",
    )
    minimum_uncovered_ink_tile_ratio: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Minimum fraction of ink-bearing tiles outside provider boxes.",
    )
    reject_unresolved_items: bool = False
    reject_source_item_loss: bool = False
    reject_invalid_spatial_coordinates: bool = True


class OCRQualityDiagnostic(BaseModel):
    """A bounded, persisted explanation for a flagged OCR page."""

    code: str
    severity: Literal["warning", "error"]
    message: str


def ocr_artifact_quality_error(payload: Mapping[str, object]) -> str | None:
    """Reject persisted quality metadata that does not describe an accepted page."""
    raw_metadata = payload.get("metadata")
    if not isinstance(raw_metadata, Mapping):
        return None
    has_status = "ocr_quality_status" in raw_metadata
    has_diagnostics = "ocr_quality_diagnostics" in raw_metadata
    if not has_status and not has_diagnostics:
        return None
    status = raw_metadata.get("ocr_quality_status")
    if has_status and (
        not isinstance(status, str) or status not in {"accepted", "flagged", "rejected"}
    ):
        return "invalid OCR quality status"
    if status == "rejected":
        return "OCR quality status is rejected"
    if not has_diagnostics:
        return None
    raw_diagnostics = raw_metadata.get("ocr_quality_diagnostics")
    if not isinstance(raw_diagnostics, list):
        return "OCR quality diagnostics must be a list"
    try:
        diagnostics = [
            OCRQualityDiagnostic.model_validate(item) for item in raw_diagnostics
        ]
    except (TypeError, ValidationError):
        return "OCR quality diagnostics are malformed"
    if any(item.severity == "error" for item in diagnostics):
        return "OCR artifact contains error-severity quality diagnostics"
    if status == "accepted" and diagnostics:
        return "accepted OCR quality status cannot include diagnostics"
    if status == "flagged" and not diagnostics:
        return "flagged OCR quality status requires diagnostics"
    return None


class OCRQualityPolicyRejection(Exception):
    """Internal failure carrying only parser-generated quality diagnostic codes."""

    def __init__(self, page_number: int, codes: Sequence[str]) -> None:
        self.page_number = page_number
        self.codes = tuple(codes)
        super().__init__(
            f"OCR quality policy rejected page {page_number}: " + "; ".join(self.codes)
        )


def evaluate_ocr_quality(
    response: OCRClusterResponse,
    policy: OCRQualityPolicy,
    *,
    source_image_path: Path | None = None,
) -> tuple[OCRQualityDiagnostic, ...]:
    """Return policy diagnostics for provider output and visible source evidence."""
    diagnostics: list[OCRQualityDiagnostic] = []
    has_any_evidence = bool(response.OCR_text_clusters or response.non_text_objects)
    if source_image_path is not None:
        tile_columns = 16
        tile_rows = 16
        ink_tiles: set[tuple[int, int]] = set()
        with Image.open(source_image_path) as source_image:
            grayscale = source_image.convert("L")
            grayscale.thumbnail((256, 256))
            histogram = grayscale.histogram()
            sample_width, sample_height = grayscale.size
            for tile_y in range(tile_rows):
                top = tile_y * sample_height // tile_rows
                bottom = (tile_y + 1) * sample_height // tile_rows
                for tile_x in range(tile_columns):
                    left = tile_x * sample_width // tile_columns
                    right = (tile_x + 1) * sample_width // tile_columns
                    if left >= right or top >= bottom:
                        continue
                    tile_histogram = grayscale.crop((left, top, right, bottom)).histogram()
                    tile_area = (right - left) * (bottom - top)
                    if sum(tile_histogram[:252]) >= max(1, int(tile_area * 0.0002)):
                        ink_tiles.add((tile_x, tile_y))
        sampled_pixel_count = sample_width * sample_height
        dark_pixel_count = sum(histogram[:252])
        minimum_ink_pixels = max(5, int(sampled_pixel_count * 0.0002))
        if dark_pixel_count >= minimum_ink_pixels and not has_any_evidence:
            diagnostics.append(
                OCRQualityDiagnostic(
                    code="empty_ocr_output_on_nonblank_image",
                    severity=(
                        "error"
                        if policy.mode == "strict"
                        or policy.reject_empty_output_on_nonblank_image
                        else "warning"
                    ),
                    message=(
                        f"the source image contains {dark_pixel_count} of {sampled_pixel_count} "
                        "sampled pixels below the ink threshold but OCR returned no "
                        "text or non-text evidence; this screening signal is not a "
                        "completeness measurement"
                    ),
                )
            )
        elif dark_pixel_count >= minimum_ink_pixels and has_any_evidence:
            page_width = response.page_x_max - response.page_x_min
            page_height = response.page_y_max - response.page_y_min
            if page_width > 0 and page_height > 0 and sample_width and sample_height:
                boxes = [
                    (item.bb_x_min, item.bb_y_min, item.bb_x_max, item.bb_y_max)
                    for item in (
                        *response.OCR_text_clusters,
                        *response.non_text_objects,
                    )
                ]
                covered_tiles: set[tuple[int, int]] = set()
                for tile_x, tile_y in ink_tiles:
                    image_x = (tile_x + 0.5) * sample_width / tile_columns
                    image_y = (tile_y + 0.5) * sample_height / tile_rows
                    page_x = response.page_x_min + image_x * page_width / sample_width
                    page_y = response.page_y_min + image_y * page_height / sample_height
                    if any(
                        x_min - page_width * 0.015
                        <= page_x
                        <= x_max + page_width * 0.015
                        and y_min - page_height * 0.015
                        <= page_y
                        <= y_max + page_height * 0.015
                        for x_min, y_min, x_max, y_max in boxes
                    ):
                        covered_tiles.add((tile_x, tile_y))
                uncovered_tiles = len(ink_tiles - covered_tiles)
                uncovered_ratio = uncovered_tiles / max(1, len(ink_tiles))
                if (
                    uncovered_tiles >= policy.minimum_uncovered_ink_tiles
                    and uncovered_ratio >= policy.minimum_uncovered_ink_tile_ratio
                ):
                    diagnostics.append(
                        OCRQualityDiagnostic(
                            code="unexplained_visual_ink_regions",
                            severity="error" if policy.mode == "strict" else "warning",
                            message=(
                                f"{uncovered_tiles} of {len(ink_tiles)} coarse ink tiles "
                                f"({uncovered_ratio:.1%}) do not overlap reported OCR "
                                "evidence boxes; this is a conservative review signal, "
                                "not proof of recognition completeness"
                            ),
                        )
                    )
    if response.is_empty_page:
        diagnostics.append(
            OCRQualityDiagnostic(
                code="empty_ocr_output",
                severity="error" if policy.reject_empty_page else "warning",
                message="provider reported no OCR or non-text evidence for the page",
            )
        )
    if response.incomplete_text:
        diagnostics.append(
            OCRQualityDiagnostic(
                code="incomplete_text",
                severity="error" if policy.reject_incomplete_text else "warning",
                message="provider marked the OCR text as incomplete",
            )
        )
    if response.incomplete_words_on_edge:
        diagnostics.append(
            OCRQualityDiagnostic(
                code="incomplete_words_on_edge",
                severity="error"
                if policy.reject_incomplete_words_on_edge
                else "warning",
                message="provider marked words at a page edge as incomplete",
            )
        )
    if response.data_loss_likelihood > policy.max_data_loss_likelihood:
        diagnostics.append(
            OCRQualityDiagnostic(
                code="data_loss_likelihood",
                severity="error",
                message=(
                    "provider data-loss likelihood exceeds the configured "
                    f"limit ({response.data_loss_likelihood:.3f} > "
                    f"{policy.max_data_loss_likelihood:.3f})"
                ),
            )
        )
    if response.unresolved_item_count:
        diagnostics.append(
            OCRQualityDiagnostic(
                code="unresolved_items",
                severity="error" if policy.reject_unresolved_items else "warning",
                message=(
                    "provider reported items that could not be represented with "
                    f"grounded records ({response.unresolved_item_count})"
                ),
            )
        )
    if response.source_item_count is not None:
        represented_item_count = len(response.OCR_text_clusters) + len(
            response.non_text_objects
        )
        accounted_item_count = represented_item_count + response.unresolved_item_count
        if response.source_item_count != accounted_item_count:
            diagnostics.append(
                OCRQualityDiagnostic(
                    code="source_item_loss",
                    severity="error" if policy.reject_source_item_loss else "warning",
                    message=(
                        "provider source item accounting does not reconcile "
                        f"({response.source_item_count} observed versus "
                        f"{accounted_item_count} represented or unresolved)"
                    ),
                )
            )
    if response.spatial_accuracy != "page":
        invalid_coordinates = [
            cluster.cluster_number
            for cluster in (*response.OCR_text_clusters, *response.non_text_objects)
            if not all(
                math.isfinite(value)
                for value in (
                    cluster.bb_x_min,
                    cluster.bb_y_min,
                    cluster.bb_x_max,
                    cluster.bb_y_max,
                )
            )
            or cluster.bb_x_min >= cluster.bb_x_max
            or cluster.bb_y_min >= cluster.bb_y_max
        ]
        if invalid_coordinates:
            diagnostics.append(
                OCRQualityDiagnostic(
                    code="invalid_spatial_coordinates",
                    severity=(
                        "error"
                        if policy.reject_invalid_spatial_coordinates
                        else "warning"
                    ),
                    message=(
                        "provider declared item-level spatial precision but emitted "
                        "invalid bounding boxes for clusters "
                        + ", ".join(str(number) for number in invalid_coordinates)
                    ),
                )
            )
    return tuple(diagnostics)
