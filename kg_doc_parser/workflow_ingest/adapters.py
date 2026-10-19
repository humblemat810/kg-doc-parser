from __future__ import annotations

import math
from collections.abc import Mapping
from typing import NotRequired, TypedDict, cast

from ..models import _meaningful_ordering_error
from .models import (
    BoundingBox,
    GroundedSourceRecord,
    NormalizedPage,
    NormalizedSourceCollection,
    SourceUnit,
    WorkflowIngestInput,
)
from .serialization import JsonObject, JsonValue


class OCRTextClusterJSON(TypedDict, total=False):
    bb_y_min: float | None
    bb_x_min: float | None
    bb_y_max: float | None
    bb_x_max: float | None
    cluster_number: int | None
    text: str | None
    coordinate_system: str | None
    spatial_accuracy: str | None


class OCRNonTextObjectJSON(TypedDict, total=False):
    bb_y_min: float | None
    bb_x_min: float | None
    bb_y_max: float | None
    bb_x_max: float | None
    cluster_number: int | None
    description: str | None
    source_uri: str | None
    mime_type: str | None
    media_sha256: str | None
    media_byte_length: int | None
    coordinate_system: str | None
    spatial_accuracy: str | None


class OCRPageJSON(TypedDict):
    """Serialized OCR page payload produced by the OCR preparation layer."""

    pdf_page_num: int | str
    printed_page_number: NotRequired[str | None]
    contains_table: NotRequired[bool | None]
    unresolved_item_count: NotRequired[int | None]
    is_blank_page: NotRequired[bool | None]
    meaningful_ordering: NotRequired[list[int] | None]
    source_item_count: NotRequired[int | None]
    page_x_min: NotRequired[float | None]
    page_x_max: NotRequired[float | None]
    page_y_min: NotRequired[float | None]
    page_y_max: NotRequired[float | None]
    metadata: NotRequired[dict[str, JsonValue] | None]
    OCR_text_clusters: NotRequired[list[OCRTextClusterJSON] | None]
    non_text_objects: NotRequired[list[OCRNonTextObjectJSON] | None]
    text: NotRequired[str | None]
    spatial_accuracy: NotRequired[str | None]


def _validated_ocr_page_json(
    value: object,
    *,
    default_page_number: int | None = None,
    strict_page_number_types: bool = False,
) -> OCRPageJSON:
    """Validate the top-level shape before narrowing persisted OCR JSON."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError("OCR page payload must be a JSON object")
    page_number = value.get("pdf_page_num", default_page_number)
    if type(page_number) is int:
        if page_number < 1:
            raise ValueError("OCR page pdf_page_num must be a positive integer")
    elif isinstance(page_number, str):
        if not page_number.strip().isdigit():
            error_type = TypeError if strict_page_number_types else ValueError
            raise error_type("OCR page pdf_page_num must be a positive integer")
    else:
        error_type = TypeError if strict_page_number_types else ValueError
        raise error_type("OCR page pdf_page_num must be a positive integer")
    scalar_types: dict[str, type[object] | tuple[type[object], ...]] = {
        "printed_page_number": str,
        "contains_table": bool,
        "is_blank_page": bool,
        "spatial_accuracy": str,
        "text": str,
    }
    for key, expected_type in scalar_types.items():
        field = value.get(key)
        if field is not None and not isinstance(field, expected_type):
            raise TypeError(f"OCR page field {key!r} has an invalid type")
    unresolved_item_count = value.get("unresolved_item_count")
    if unresolved_item_count is not None and (
        type(unresolved_item_count) is not int or unresolved_item_count < 0
    ):
        raise TypeError("OCR page field 'unresolved_item_count' must be a non-negative integer")
    source_item_count = value.get("source_item_count")
    if source_item_count is not None and (
        type(source_item_count) is not int or source_item_count < 0
    ):
        raise TypeError("OCR page field 'source_item_count' must be a non-negative integer")
    for key in ("page_x_min", "page_x_max", "page_y_min", "page_y_max"):
        coordinate = value.get(key)
        if coordinate is not None and (
            isinstance(coordinate, bool)
            or not isinstance(coordinate, (int, float))
            or not math.isfinite(float(coordinate))
        ):
            raise TypeError(f"OCR page field {key!r} must be a finite number")
    for key in ("meaningful_ordering",):
        field = value.get(key)
        if field is not None and (
            not isinstance(field, list) or any(type(item) is not int for item in field)
        ):
            raise TypeError(f"OCR page field {key!r} must be a list of integers")
    metadata = value.get("metadata")
    if metadata is not None and (
        not isinstance(metadata, dict) or not all(isinstance(key, str) for key in metadata)
    ):
        raise TypeError("OCR page metadata must be a JSON object")
    for key in ("OCR_text_clusters", "non_text_objects"):
        field = value.get(key)
        if field is not None and (
            not isinstance(field, list)
            or any(
                not isinstance(item, dict)
                or not all(isinstance(item_key, str) for item_key in item)
                for item in field
            )
        ):
            if not isinstance(field, list):
                raise TypeError("OCR cluster collections must be lists")
            label = "text cluster" if key == "OCR_text_clusters" else "non-text object"
            raise TypeError(f"each OCR {label} must be an object")
        if isinstance(field, list):
            for index, item in enumerate(field):
                if not isinstance(item, dict):
                    continue
                for item_key in (
                    "text",
                    "description",
                    "source_uri",
                    "mime_type",
                    "media_sha256",
                    "coordinate_system",
                    "spatial_accuracy",
                ):
                    item_value = item.get(item_key)
                    if item_value is not None and not isinstance(item_value, str):
                        raise TypeError(
                            f"OCR page field {key}[{index}].{item_key!r} must be a string"
                        )
                media_length = item.get("media_byte_length")
                if media_length is not None and type(media_length) is not int:
                    raise TypeError(
                        f"OCR page field {key}[{index}].'media_byte_length' must be an integer"
                    )
    if "pdf_page_num" not in value:
        return cast(OCRPageJSON, {**value, "pdf_page_num": page_number})
    return cast(OCRPageJSON, value)


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
        raw_page = _validated_ocr_page_json(raw_page)
        page_number = _ocr_page_number(raw_page["pdf_page_num"])
        if page_number in seen_page_numbers:
            raise ValueError(f"duplicate OCR pdf_page_num: {page_number}")
        seen_page_numbers.add(page_number)
        seen_clusters: set[int] = set()
        units: list[SourceUnit] = []
        text_clusters = raw_page.get("OCR_text_clusters", [])
        non_text_objects = raw_page.get("non_text_objects", [])
        if not isinstance(text_clusters, list) or not isinstance(non_text_objects, list):
            raise TypeError("OCR cluster collections must be lists")
        source_item_count = raw_page.get("source_item_count")
        unresolved_item_count = raw_page.get("unresolved_item_count", 0)
        if unresolved_item_count is None:
            unresolved_item_count = 0
        if source_item_count is not None:
            represented_item_count = len(text_clusters) + len(non_text_objects)
            accounted_item_count = represented_item_count + unresolved_item_count
            if source_item_count != accounted_item_count:
                raise ValueError(
                    "OCR source item accounting does not reconcile "
                    f"({source_item_count} observed versus "
                    f"{accounted_item_count} represented or unresolved)"
                )
        page_metadata = raw_page.get("metadata", {})
        if not isinstance(page_metadata, dict):
            raise TypeError("OCR page metadata must be an object")
        declared_page_accuracy = raw_page.get("spatial_accuracy")
        if declared_page_accuracy is not None:
            if not isinstance(declared_page_accuracy, str):
                raise TypeError("OCR spatial_accuracy must be a string")
            page_metadata = {**page_metadata, "spatial_accuracy": declared_page_accuracy}
        for key in (
            "incomplete_text",
            "incomplete_words_on_edge",
            "data_loss_likelihood",
            "scan_quality",
            "unresolved_item_count",
            "source_item_count",
            "page_x_min",
            "page_x_max",
            "page_y_min",
            "page_y_max",
        ):
            value = raw_page.get(key)
            if value is not None:
                page_metadata = {**page_metadata, key: value}
        ordering = raw_page.get("meaningful_ordering")
        if ordering is not None:
            if not isinstance(ordering, list) or any(type(value) is not int for value in ordering):
                raise TypeError("meaningful_ordering must be a list of integer cluster numbers")
            cluster_by_number: dict[
                int, OCRTextClusterJSON | OCRNonTextObjectJSON
            ] = {}
            expected_cluster_count = len(text_clusters) + len(non_text_objects)
            text_numbers: set[int] = set()
            non_text_numbers: set[int] = set()
            for clusters, numbers in (
                (text_clusters, text_numbers),
                (non_text_objects, non_text_numbers),
            ):
                for cluster in clusters:
                    cluster_number = cluster.get("cluster_number")
                    if type(cluster_number) is not int:
                        raise ValueError(
                            "meaningful_ordering requires every OCR cluster to have "
                            "an integer number"
                        )
                    if cluster_number in cluster_by_number:
                        raise ValueError(
                            "meaningful_ordering requires every OCR cluster to have "
                            "a unique number"
                        )
                    cluster_by_number[cluster_number] = cluster
                    numbers.add(cluster_number)
            if (
                len(ordering) != expected_cluster_count
                or set(ordering) != set(cluster_by_number)
            ):
                raise ValueError(_meaningful_ordering_error(set(cluster_by_number), ordering))
            ordered_text_clusters = [
                cluster_by_number[cluster_number]
                for cluster_number in ordering
                if cluster_number in text_numbers
            ]
            ordered_non_text_objects = [
                cluster_by_number[cluster_number]
                for cluster_number in ordering
                if cluster_number in non_text_numbers
            ]
            ordering_status = "declared"
        else:
            ordered_text_clusters = text_clusters
            ordered_non_text_objects = non_text_objects
            ordering_status = (
                "ambiguous_provider_order"
                if text_clusters and non_text_objects
                else "single_modality_order"
            )
        for cluster in ordered_text_clusters:
            if not isinstance(cluster, dict):
                raise TypeError("each OCR text cluster must be an object")
            cluster_number = _ocr_cluster_number(cluster.get("cluster_number"))
            if cluster_number is not None and cluster_number in seen_clusters:
                raise ValueError(f"duplicate OCR text cluster_number: {cluster_number}")
            if cluster_number is not None:
                seen_clusters.add(cluster_number)
            units.append(
                SourceUnit(
                    modality="ocr_text",
                    page_number=page_number,
                    cluster_number=cluster_number,
                    text=cluster.get("text"),
                    bbox=_ocr_bbox(cluster),
                    metadata={
                        **page_metadata,
                        "coordinate_system": cluster.get("coordinate_system", "pixels"),
                        "spatial_accuracy": cluster.get(
                            "spatial_accuracy", page_metadata.get("spatial_accuracy", "page")
                        ),
                    },
                )
            )
        for obj in ordered_non_text_objects:
            if not isinstance(obj, dict):
                raise TypeError("each OCR non-text object must be an object")
            cluster_number = _ocr_cluster_number(obj.get("cluster_number"))
            if cluster_number is not None and cluster_number in seen_clusters:
                raise ValueError(f"duplicate OCR image cluster_number: {cluster_number}")
            if cluster_number is not None:
                seen_clusters.add(cluster_number)
            units.append(
                SourceUnit(
                    modality="image_region",
                    page_number=page_number,
                    cluster_number=cluster_number,
                    description=obj.get("description"),
                    source_uri=obj.get("source_uri"),
                    bbox=_ocr_bbox(obj),
                    # Keep the "image" space label for future routing; the
                    # current engine still embeds through a single function.
                    embedding_space="image",
                    metadata={
                        **page_metadata,
                        "coordinate_system": obj.get("coordinate_system", "pixels"),
                        "spatial_accuracy": obj.get(
                            "spatial_accuracy", page_metadata.get("spatial_accuracy", "page")
                        ),
                        "participates_in_semantic_text": False,
                        "mime_type": obj.get("mime_type"),
                        "media_sha256": obj.get("media_sha256"),
                        "media_byte_length": obj.get("media_byte_length"),
                    },
                )
            )
        if ordering is not None:
            unit_by_cluster = {
                unit.cluster_number: unit
                for unit in units
                if unit.cluster_number is not None
            }
            units = [unit_by_cluster[cluster_number] for cluster_number in ordering]
        normalized_pages.append(
            NormalizedPage(
                page_number=page_number,
                units=units,
                metadata={
                    **page_metadata,
                    "ordering_status": ordering_status,
                    "printed_page_number": raw_page.get("printed_page_number"),
                    "contains_table": raw_page.get("contains_table"),
                    "is_blank_page": not units,
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
) -> JsonObject:
    pages: list[JsonObject] = []
    for page in collection.pages:
        text_clusters: list[JsonValue] = []
        non_text_objects: list[JsonValue] = []
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
                "is_blank_page": page.metadata.get("is_blank_page", not page.units),
                "OCR_text_clusters": text_clusters,
                "non_text_objects": non_text_objects,
            }
        )
    payload: JsonObject = {
        "document_filename": collection.title,
        "pages": cast(JsonValue, pages),
    }
    if "document_tree" in collection.metadata:
        payload["document_tree"] = collection.metadata["document_tree"]
    return payload


def build_parser_source_map(
    source_map: dict[str, GroundedSourceRecord],
) -> dict[str, JsonObject]:
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
