from __future__ import annotations

from typing import Any

from .models import (
    BoundingBox,
    GroundedSourceRecord,
    NormalizedPage,
    NormalizedSourceCollection,
    SourceUnit,
    WorkflowIngestInput,
)


def normalize_ocr_pages(
    *,
    document_id: str,
    title: str,
    pages: list[dict[str, Any]],
) -> WorkflowIngestInput:
    """Normalize raw OCR JSON into workflow ingest models.

    OCR text clusters and non-text regions are both preserved with page and
    cluster metadata. `embedding_space="image"` is an intent label here; it
    does not imply a separate image embedder is already wired at runtime.
    """
    normalized_pages: list[NormalizedPage] = []
    for raw_page in pages:
        page_number = int(raw_page["pdf_page_num"])
        units: list[SourceUnit] = []
        for cluster in raw_page.get("OCR_text_clusters", []):
            bbox = BoundingBox(
                y_min=float(cluster.get("bb_y_min", 0.0)),
                x_min=float(cluster.get("bb_x_min", 0.0)),
                y_max=float(cluster.get("bb_y_max", 0.0)),
                x_max=float(cluster.get("bb_x_max", 0.0)),
            )
            units.append(
                SourceUnit(
                    modality="ocr_text",
                    page_number=page_number,
                    cluster_number=cluster.get("cluster_number"),
                    text=cluster.get("text"),
                    bbox=bbox,
                    metadata={},
                )
            )
        for obj in raw_page.get("non_text_objects", []):
            bbox = BoundingBox(
                y_min=float(obj.get("bb_y_min", 0.0)),
                x_min=float(obj.get("bb_x_min", 0.0)),
                y_max=float(obj.get("bb_y_max", 0.0)),
                x_max=float(obj.get("bb_x_max", 0.0)),
            )
            units.append(
                SourceUnit(
                    modality="image_region",
                    page_number=page_number,
                    cluster_number=obj.get("cluster_number"),
                    description=obj.get("description"),
                    bbox=bbox,
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


def build_authoritative_source_map(
    inp: WorkflowIngestInput,
) -> dict[str, GroundedSourceRecord]:
    source_map: dict[str, GroundedSourceRecord] = {}
    for collection in inp.collections:
        for page in collection.pages:
            ordinal_by_modality: dict[str, int] = {}
            for unit in page.units:
                modality_prefix = "t" if unit.modality in {"text", "ocr_text"} else "i"
                original_cluster = unit.cluster_number
                if original_cluster is None:
                    original_cluster = ordinal_by_modality.get(modality_prefix, 0)
                ordinal_by_modality[modality_prefix] = ordinal_by_modality.get(modality_prefix, 0) + 1
                unit_id = unit.unit_id or f"{collection.collection_id}|p{page.page_number}_{modality_prefix}{original_cluster}"
                record = GroundedSourceRecord(
                    unit_id=unit_id,
                    collection_id=collection.collection_id,
                    modality=unit.modality,
                    page_number=page.page_number,
                    cluster_number=unit.cluster_number,
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
) -> dict[str, Any]:
    pages: list[dict[str, Any]] = []
    for page in collection.pages:
        text_clusters = []
        non_text_objects = []
        next_cluster = 0
        for unit in page.units:
            bbox = unit.bbox
            cluster_number = unit.cluster_number if unit.cluster_number is not None else next_cluster
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
            next_cluster += 1
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
) -> dict[str, dict[str, Any]]:
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
