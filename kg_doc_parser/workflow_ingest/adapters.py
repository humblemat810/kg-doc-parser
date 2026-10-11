from __future__ import annotations

import math
from collections.abc import Mapping
from typing import NotRequired, TypedDict

from .models import (
    BoundingBox,
    GroundedSourceRecord,
    NormalizedPage,
    NormalizedSourceCollection,
    SourceUnit,
    WorkflowIngestInput,
)


class OCRTextClusterJSON(TypedDict, total=False):
    bb_y_min: float
    bb_x_min: float
    bb_y_max: float
    bb_x_max: float
    cluster_number: int
    text: str


class OCRNonTextObjectJSON(TypedDict, total=False):
    bb_y_min: float
    bb_x_min: float
    bb_y_max: float
    bb_x_max: float
    cluster_number: int
    description: str


class OCRPageJSON(TypedDict):
    """Serialized OCR page payload produced by the OCR preparation layer."""

    pdf_page_num: int
    printed_page_number: NotRequired[str]
    contains_table: NotRequired[bool]
    OCR_text_clusters: NotRequired[list[OCRTextClusterJSON]]
    non_text_objects: NotRequired[list[OCRNonTextObjectJSON]]
    text: NotRequired[str]


def _ocr_page_number(value: object) -> int:
    """Return a positive page number without permissive bool/float coercion."""
    if type(value) is int:
        page_number = value
    elif isinstance(value, str) and value.strip().isdigit():
        page_number = int(value.strip())
    else:
        raise ValueError("OCR pdf_page_num must be a positive integer")
    if page_number < 1:
        raise ValueError("OCR pdf_page_num must be a positive integer")
    return page_number


def _ocr_cluster_number(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is int:
        cluster_number = value
    elif isinstance(value, str) and value.strip().isdigit():
        cluster_number = int(value.strip())
    else:
        raise ValueError("OCR cluster_number must be a non-negative integer")
    if cluster_number < 0:
        raise ValueError("OCR cluster_number must be a non-negative integer")
    return cluster_number


def _ocr_coordinate(value: object, *, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        raise TypeError("OCR bounding-box coordinates must be finite numbers")
    if not isinstance(value, (int, float, str)):
        raise TypeError("OCR bounding-box coordinates must be finite numbers")
    try:
        coordinate = float(value)
    except ValueError:
        raise ValueError("OCR bounding-box coordinates must be finite numbers") from None
    if not math.isfinite(coordinate):
        raise ValueError("OCR bounding-box coordinates must be finite numbers")
    return coordinate


def _ocr_bbox(raw: Mapping[str, object]) -> BoundingBox:
    y_min = _ocr_coordinate(raw.get("bb_y_min"))
    x_min = _ocr_coordinate(raw.get("bb_x_min"))
    y_max = _ocr_coordinate(raw.get("bb_y_max"))
    x_max = _ocr_coordinate(raw.get("bb_x_max"))
    if y_min > y_max or x_min > x_max:
        raise ValueError("OCR bounding-box minimums must not exceed maximums")
    return BoundingBox(y_min=y_min, x_min=x_min, y_max=y_max, x_max=x_max)


def normalize_ocr_pages(
    *,
    document_id: str,
    title: str,
    pages: list[OCRPageJSON],
) -> WorkflowIngestInput:
    """Normalize raw OCR JSON into workflow ingest models.

    OCR text clusters and non-text regions are both preserved with page and
    cluster metadata. `embedding_space="image"` is an intent label here; it
    does not imply a separate image embedder is already wired at runtime.
    """
    if not isinstance(pages, list):
        raise TypeError("OCR pages must be a list")
    normalized_pages: list[NormalizedPage] = []
    seen_page_numbers: set[int] = set()
    for raw_page in pages:
        if not isinstance(raw_page, dict):
            raise TypeError("each OCR page must be an object")
        if "pdf_page_num" not in raw_page:
            raise ValueError("each OCR page requires pdf_page_num")
        page_number = _ocr_page_number(raw_page["pdf_page_num"])
        if page_number in seen_page_numbers:
            raise ValueError(f"duplicate OCR pdf_page_num: {page_number}")
        seen_page_numbers.add(page_number)
        seen_clusters: dict[str, set[int]] = {"text": set(), "image": set()}
        units: list[SourceUnit] = []
        text_clusters = raw_page.get("OCR_text_clusters", [])
        non_text_objects = raw_page.get("non_text_objects", [])
        if not isinstance(text_clusters, list) or not isinstance(non_text_objects, list):
            raise TypeError("OCR cluster collections must be lists")
        for cluster in text_clusters:
            if not isinstance(cluster, dict):
                raise TypeError("each OCR text cluster must be an object")
            cluster_number = _ocr_cluster_number(cluster.get("cluster_number"))
            if cluster_number is not None and cluster_number in seen_clusters["text"]:
                raise ValueError(f"duplicate OCR text cluster_number: {cluster_number}")
            if cluster_number is not None:
                seen_clusters["text"].add(cluster_number)
            units.append(
                SourceUnit(
                    modality="ocr_text",
                    page_number=page_number,
                    cluster_number=cluster_number,
                    text=cluster.get("text"),
                    bbox=_ocr_bbox(cluster),
                    metadata={},
                )
            )
        for obj in non_text_objects:
            if not isinstance(obj, dict):
                raise TypeError("each OCR non-text object must be an object")
            cluster_number = _ocr_cluster_number(obj.get("cluster_number"))
            if cluster_number is not None and cluster_number in seen_clusters["image"]:
                raise ValueError(f"duplicate OCR image cluster_number: {cluster_number}")
            if cluster_number is not None:
                seen_clusters["image"].add(cluster_number)
            units.append(
                SourceUnit(
                    modality="image_region",
                    page_number=page_number,
                    cluster_number=cluster_number,
                    description=obj.get("description"),
                    bbox=_ocr_bbox(obj),
                    # Keep the "image" space label for future routing; the
                    # current engine still embeds through a single function.
                    embedding_space="image",
                    metadata={"participates_in_semantic_text": False},
                )
            )
        normalized_pages.append(
            NormalizedPage(
                page_number=page_number,
                units=units,
                metadata={
                    "printed_page_number": raw_page.get("printed_page_number"),
                    "contains_table": raw_page.get("contains_table"),
                },
            )
        )
    embedding_spaces = ["default_text"]
    if any(
        unit.embedding_space == "image"
        for page in normalized_pages
        for unit in page.units
    ):
        embedding_spaces.append("image")
    return WorkflowIngestInput(
        request_id=document_id,
        collections=[
            NormalizedSourceCollection(
                collection_id=document_id,
                title=title,
                modality="ocr",
                pages=normalized_pages,
                embedding_spaces=embedding_spaces,
            )
        ],
    )


def _allocated_cluster_numbers(units: list[SourceUnit]) -> list[int]:
    """Allocate missing cluster numbers exactly as the source-map builder does."""
    reserved: dict[str, set[int]] = {"t": set(), "i": set()}
    for unit in units:
        prefix = "t" if unit.modality in {"text", "ocr_text"} else "i"
        if unit.cluster_number is not None:
            reserved[prefix].add(unit.cluster_number)

    next_ordinal: dict[str, int] = {"t": 0, "i": 0}
    used: dict[str, set[int]] = {"t": set(), "i": set()}
    allocated: list[int] = []
    for unit in units:
        prefix = "t" if unit.modality in {"text", "ocr_text"} else "i"
        cluster_number = unit.cluster_number
        if cluster_number is not None and cluster_number in used[prefix]:
            raise ValueError(f"duplicate cluster number {cluster_number} in page")
        if cluster_number is None:
            cluster_number = next_ordinal[prefix]
            while cluster_number in reserved[prefix] or cluster_number in used[prefix]:
                cluster_number += 1
        used[prefix].add(cluster_number)
        next_ordinal[prefix] = cluster_number + 1
        allocated.append(cluster_number)
    return allocated


def build_authoritative_source_map(
    inp: WorkflowIngestInput,
) -> dict[str, GroundedSourceRecord]:
    source_map: dict[str, GroundedSourceRecord] = {}
    for collection in inp.collections:
        for page in collection.pages:
            allocated_clusters = _allocated_cluster_numbers(page.units)
            for unit, allocated_cluster in zip(page.units, allocated_clusters, strict=True):
                modality_prefix = "t" if unit.modality in {"text", "ocr_text"} else "i"
                unit_id = unit.unit_id or f"{collection.collection_id}|p{page.page_number}_{modality_prefix}{allocated_cluster}"
                record = GroundedSourceRecord(
                    unit_id=unit_id,
                    collection_id=collection.collection_id,
                    modality=unit.modality,
                    page_number=page.page_number,
                    # Persist the allocated identity so the record agrees with
                    # its source-map key when the input omitted a cluster number.
                    cluster_number=allocated_cluster,
                    text=unit.text or unit.description or "",
                    parser_text=unit.parser_text,
                    source_uri=unit.source_uri,
                    embedding_space=unit.embedding_space,
                    participates_in_semantic_text=unit.modality in {"text", "ocr_text"},
                    bbox=unit.bbox,
                    metadata={
                        **unit.metadata,
                        "original_cluster_number": unit.cluster_number,
                    },
                )
                if unit_id in source_map:
                    raise ValueError(f"source map collision detected for {unit_id}")
                source_map[unit_id] = record
    return source_map


def select_primary_collection(inp: WorkflowIngestInput) -> NormalizedSourceCollection:
    return inp.collections[0]


def build_parser_input_dict(
    collection: NormalizedSourceCollection,
) -> dict[str, object]:
    pages: list[dict[str, object]] = []
    for page in collection.pages:
        text_clusters = []
        non_text_objects = []
        allocated_clusters = _allocated_cluster_numbers(page.units)
        for unit, cluster_number in zip(page.units, allocated_clusters, strict=True):
            bbox = unit.bbox
            if unit.modality in {"text", "ocr_text"}:
                text_clusters.append(
                    {
                        "text": unit.text or "",
                        "bb_x_min": bbox.x_min if bbox else 0.0,
                        "bb_x_max": bbox.x_max if bbox else 0.0,
                        "bb_y_min": bbox.y_min if bbox else 0.0,
                        "bb_y_max": bbox.y_max if bbox else 0.0,
                        "cluster_number": cluster_number,
                    }
                )
            else:
                non_text_objects.append(
                    {
                        "description": unit.description or unit.source_uri or "image-region",
                        "bb_x_min": bbox.x_min if bbox else 0.0,
                        "bb_x_max": bbox.x_max if bbox else 0.0,
                        "bb_y_min": bbox.y_min if bbox else 0.0,
                        "bb_y_max": bbox.y_max if bbox else 0.0,
                        "cluster_number": cluster_number,
                    }
                )
        pages.append(
            {
                "pdf_page_num": page.page_number,
                "printed_page_number": page.metadata.get("printed_page_number"),
                "contains_table": page.metadata.get("contains_table", False),
                "OCR_text_clusters": text_clusters,
                "non_text_objects": non_text_objects,
            }
        )
    return {"document_filename": collection.title, "pages": pages}


def build_parser_source_map(
    source_map: dict[str, GroundedSourceRecord],
) -> dict[str, dict[str, object]]:
    return {
        unit_id: {
            "id": record.unit_id,
            "text": record.parser_text,
            "modality": record.modality,
            "page_number": record.page_number,
            "cluster_number": record.cluster_number,
            "embedding_space": record.embedding_space,
            "participates_in_semantic_text": record.participates_in_semantic_text,
            "metadata": record.metadata,
        }
        for unit_id, record in source_map.items()
    }
