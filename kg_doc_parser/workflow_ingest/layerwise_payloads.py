"""Validated conversion of structured provider payloads into parser DTOs."""

from __future__ import annotations

from typing import cast

from .model_validation import validate_backend_payload
from .models import (
    CurrentLayerResult,
    CurrentLayerReview,
    LLMCurrentLayerResult,
    LLMCurrentLayerReview,
)
from .parser_core import ParserSourceMap
from .semantics import HydratedTextPointer, hydrate_pointer_from_offsets
from .serialization import ModelDumpLike


def dump_model(value: object) -> object:
    """Normalize supported model-like values without assuming SDK payloads."""

    if isinstance(value, ModelDumpLike):
        try:
            return value.model_dump(field_mode="backend", dump_format="json")
        except TypeError:
            return value.model_dump()
    if isinstance(value, dict):
        return {str(key): dump_model(item) for key, item in value.items()}
    if isinstance(value, list):
        return [dump_model(item) for item in value]
    return value


def object_list(value: object) -> list[object]:
    return list(value) if isinstance(value, (list, tuple)) else []


def object_dict(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}


def hydrate_pointer_payload(
    pointer: object,
    *,
    parser_source_map: ParserSourceMap,
) -> dict[str, object]:
    """Hydrate offsets from authoritative source text, not provider echoes."""

    payload = object_dict(dump_model(pointer))
    raw_source_cluster_id = payload.get("source_cluster_id")
    source_cluster_id = (
        raw_source_cluster_id if isinstance(raw_source_cluster_id, str) else ""
    )
    start_char = payload.get("start_char")
    end_char = payload.get("end_char")
    if type(start_char) is int and type(end_char) is int:
        raw_verbatim_text = payload.get("verbatim_text")
        verbatim_text = raw_verbatim_text if isinstance(raw_verbatim_text, str) else ""
        hydrated = hydrate_pointer_from_offsets(
            HydratedTextPointer(
                source_cluster_id=source_cluster_id,
                start_char=cast(int, start_char),
                end_char=cast(int, end_char),
                verbatim_text=verbatim_text,
            ),
            parser_source_map,
        )
        if hydrated is not None:
            payload.update(hydrated.model_dump())
            return payload
    raw_verbatim_text = payload.get("verbatim_text")
    payload["verbatim_text"] = (
        raw_verbatim_text if isinstance(raw_verbatim_text, str) else ""
    )
    return payload


def hydrate_layer_result(
    result: LLMCurrentLayerResult,
    *,
    parser_source_map: ParserSourceMap,
) -> CurrentLayerResult:
    payload = object_dict(result.model_dump())
    children = []
    for child in object_list(payload.get("children")):
        child_payload = object_dict(child)
        child_payload["total_content_pointers"] = [
            hydrate_pointer_payload(pointer, parser_source_map=parser_source_map)
            for pointer in object_list(child_payload.get("total_content_pointers"))
        ]
        children.append(child_payload)
    payload["children"] = children
    return validate_backend_payload(CurrentLayerResult, payload)


def hydrate_review(
    review: LLMCurrentLayerReview,
    *,
    parser_source_map: ParserSourceMap,
) -> CurrentLayerReview:
    payload = object_dict(review.model_dump())
    updated_result = payload.get("updated_result")
    if isinstance(updated_result, dict):
        children = []
        for child in object_list(updated_result.get("children")):
            child_payload = object_dict(child)
            child_payload["total_content_pointers"] = [
                hydrate_pointer_payload(pointer, parser_source_map=parser_source_map)
                for pointer in object_list(child_payload.get("total_content_pointers"))
            ]
            children.append(child_payload)
        updated_result["children"] = children
    for conflict in object_list(payload.get("overlap_conflicts")):
        conflict_payload = object_dict(conflict)
        conflict_payload["left_span"] = hydrate_pointer_payload(
            conflict_payload.get("left_span"), parser_source_map=parser_source_map
        )
        conflict_payload["right_span"] = hydrate_pointer_payload(
            conflict_payload.get("right_span"), parser_source_map=parser_source_map
        )
    return validate_backend_payload(CurrentLayerReview, payload)


# Keep the compatibility helpers imported by existing internal callers.
_dump_model = dump_model
_object_list = object_list
_object_dict = object_dict
_hydrate_llm_pointer_payload = hydrate_pointer_payload
_hydrate_llm_layer_result = hydrate_layer_result
_hydrate_llm_review = hydrate_review

__all__ = [
    "dump_model",
    "hydrate_layer_result",
    "hydrate_pointer_payload",
    "hydrate_review",
    "object_dict",
    "object_list",
]
