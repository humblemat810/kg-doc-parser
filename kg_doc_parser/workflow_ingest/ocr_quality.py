"""Independent OCR quality evaluation for controlled ground-truth fixtures.

This module is deliberately separate from provider acceptance.  A provider
cannot establish its own recall or spatial accuracy merely by reporting those
properties, so experiments may compare a response with an independently
authored page fixture.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from kg_doc_parser.models import OCRClusterResponse

OCRGroundTruthModality = Literal["text", "non_text"]


class OCRGroundTruthItem(BaseModel):
    """One independently annotated source item on an OCR page."""

    item_id: str = Field(min_length=1)
    modality: OCRGroundTruthModality
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    text: str | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> OCRGroundTruthItem:
        if not all(
            math.isfinite(value)
            for value in (self.x_min, self.y_min, self.x_max, self.y_max)
        ):
            raise ValueError("ground-truth item bounds must be finite")
        if not self.x_min < self.x_max or not self.y_min < self.y_max:
            raise ValueError("ground-truth item bounds must have positive area")
        if self.modality == "text" and not self.text:
            raise ValueError("text ground-truth items require text")
        if self.modality == "non_text" and self.text is not None:
            raise ValueError("non-text ground-truth items cannot carry text")
        return self


class OCRGroundTruth(BaseModel):
    """Independent annotations used to measure one OCR response."""

    items: tuple[OCRGroundTruthItem, ...] = ()

    @model_validator(mode="after")
    def validate_unique_ids(self) -> OCRGroundTruth:
        ids = [item.item_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("ground-truth item IDs must be unique")
        return self


class OCRGroundTruthReport(BaseModel):
    """Deterministic recall, spatial, and text comparison results."""

    expected_item_count: int = Field(ge=0)
    matched_item_count: int = Field(ge=0)
    item_recall: float = Field(ge=0.0, le=1.0)
    mean_spatial_iou: float = Field(ge=0.0, le=1.0)
    missing_item_ids: tuple[str, ...] = ()
    extra_cluster_numbers: tuple[int, ...] = ()
    text_mismatch_item_ids: tuple[str, ...] = ()
    minimum_iou: float = Field(ge=0.0, le=1.0)
    item_recall_by_modality: dict[OCRGroundTruthModality, float] = Field(
        default_factory=dict
    )


def _normalise_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _intersection_over_union(
    left: OCRGroundTruthItem,
    right: tuple[float, float, float, float],
) -> float:
    x_min = max(left.x_min, right[0])
    y_min = max(left.y_min, right[1])
    x_max = min(left.x_max, right[2])
    y_max = min(left.y_max, right[3])
    intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    left_area = (left.x_max - left.x_min) * (left.y_max - left.y_min)
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _assign_matches(
    expected_items: tuple[OCRGroundTruthItem, ...],
    candidates: Sequence[
        tuple[str, int, tuple[float, float, float, float], str | None]
    ],
    minimum_iou: float,
) -> dict[int, tuple[int, float]]:
    """Find a deterministic maximum-cardinality one-to-one assignment."""

    eligible: dict[int, list[tuple[int, float]]] = {}
    for expected_index, expected in enumerate(expected_items):
        options = [
            (candidate_index, iou)
            for candidate_index, candidate in enumerate(candidates)
            if candidate[0] == expected.modality
            for iou in (_intersection_over_union(expected, candidate[2]),)
            if iou >= minimum_iou
        ]
        eligible[expected_index] = sorted(options, key=lambda value: (-value[1], value[0]))

    candidate_owner: dict[int, int] = {}

    def visit(expected_index: int, seen: set[int]) -> bool:
        for candidate_index, _iou in eligible[expected_index]:
            if candidate_index in seen:
                continue
            seen.add(candidate_index)
            owner = candidate_owner.get(candidate_index)
            if owner is None or visit(owner, seen):
                candidate_owner[candidate_index] = expected_index
                return True
        return False

    for expected_index in range(len(expected_items)):
        visit(expected_index, set())

    assignments: dict[int, tuple[int, float]] = {
        expected_index: next(
            (candidate_index, iou)
            for candidate_index, owner in candidate_owner.items()
            if owner == expected_index
            for candidate, iou in eligible[expected_index]
            if candidate == candidate_index
        )
        for expected_index in range(len(expected_items))
        if any(owner == expected_index for owner in candidate_owner.values())
    }
    return assignments


def evaluate_ocr_ground_truth(
    response: OCRClusterResponse,
    ground_truth: OCRGroundTruth,
    *,
    minimum_iou: float = 0.5,
) -> OCRGroundTruthReport:
    """Compare a provider response against independent page annotations.

    Matching is one-to-one and uses modality plus bounding-box IoU.  Text is
    reported separately so a positionally correct OCR item is not hidden by a
    transcription error.  The function does not alter provider acceptance or
    persisted workflow state.
    """

    if not 0.0 <= minimum_iou <= 1.0:
        raise ValueError("minimum_iou must be between 0.0 and 1.0")

    candidates = [
        (
            "text",
            cluster.cluster_number,
            (
                float(cluster.bb_x_min),
                float(cluster.bb_y_min),
                float(cluster.bb_x_max),
                float(cluster.bb_y_max),
            ),
            str(cluster.text),
        )
        for cluster in response.OCR_text_clusters
    ] + [
        (
            "non_text",
            cluster.cluster_number,
            (
                float(cluster.bb_x_min),
                float(cluster.bb_y_min),
                float(cluster.bb_x_max),
                float(cluster.bb_y_max),
            ),
            None,
        )
        for cluster in response.non_text_objects
    ]
    assignments = _assign_matches(ground_truth.items, candidates, minimum_iou)
    recall_by_modality: dict[OCRGroundTruthModality, float] = {}
    for modality in ("text", "non_text"):
        expected_indices = [
            index
            for index, item in enumerate(ground_truth.items)
            if item.modality == modality
        ]
        matched_count = sum(index in assignments for index in expected_indices)
        recall_by_modality[modality] = (
            matched_count / len(expected_indices) if expected_indices else 1.0
        )
    missing: list[str] = []
    mismatches: list[str] = []
    spatial_ious: list[float] = []
    matched = 0
    for expected_index, expected in enumerate(ground_truth.items):
        assignment = assignments.get(expected_index)
        if assignment is None:
            missing.append(expected.item_id)
            continue
        index, iou = assignment
        matched += 1
        spatial_ious.append(iou)
        candidate_text = candidates[index][3]
        if expected.text is not None and _normalise_text(candidate_text or "") != _normalise_text(expected.text):
            mismatches.append(expected.item_id)

    assigned_candidates = {assignment[0] for assignment in assignments.values()}
    return OCRGroundTruthReport(
        expected_item_count=len(ground_truth.items),
        matched_item_count=matched,
        item_recall=(matched / len(ground_truth.items)) if ground_truth.items else 1.0,
        mean_spatial_iou=(sum(spatial_ious) / len(spatial_ious)) if spatial_ious else 0.0,
        missing_item_ids=tuple(missing),
        extra_cluster_numbers=tuple(
            candidate[1]
            for index, candidate in enumerate(candidates)
            if index not in assigned_candidates
        ),
        text_mismatch_item_ids=tuple(mismatches),
        minimum_iou=minimum_iou,
        item_recall_by_modality=recall_by_modality,
    )
