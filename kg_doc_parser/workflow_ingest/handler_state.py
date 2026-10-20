"""Typed decoding helpers for persisted parser workflow state."""

from __future__ import annotations

import math
from collections.abc import Mapping

from .parser_core import ParserPayload, ParserSourceMap
from .semantics import (
    HydratedTextPointer,
    correct_and_validate_pointer,
)
from .serialization import JsonValue, decode_json_object


def strategy_attempt(state_view: Mapping[str, object], strategy: str) -> int:
    value = state_mapping(state_view.get("strategy_attempt_counts")).get(strategy)
    if type(value) is int and value >= 0:
        return value
    return 1


def parser_payload(value: object) -> ParserPayload:
    """Decode one JSON object from runtime state into parser payload shape."""

    return decode_json_object(value, label="parser payload")


def parser_source_map(value: object | None) -> ParserSourceMap:
    """Decode the nested parser source map carried through checkpoint state."""

    if value is None:
        return {}
    source_map: ParserSourceMap = {}
    decoded = decode_json_object(value, label="parser source map")
    for cluster_id, payload in decoded.items():
        source_map[cluster_id] = decode_json_object(
            payload, label=f"parser source cluster {cluster_id!r}"
        )
    return source_map


def correct_parser_pointer(
    pointer: HydratedTextPointer,
    parser_source_map: ParserSourceMap,
) -> HydratedTextPointer | None:
    """Adapt rich parser payloads to the text-only pointer validator."""

    text_source_map: dict[str, dict[str, JsonValue]] = {}
    for cluster_id, payload in parser_source_map.items():
        text = payload.get("text", "")
        text_source_map[cluster_id] = {"text": text if isinstance(text, str) else ""}
    return correct_and_validate_pointer(pointer, text_source_map)


def state_object_map(value: object) -> dict[str, dict[str, JsonValue]]:
    if not isinstance(value, dict):
        return {}
    objects: dict[str, dict[str, JsonValue]] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("workflow state object map contains a non-string key")
        try:
            objects[key] = decode_json_object(
                item, label=f"workflow state object map entry {key!r}"
            )
        except TypeError as exc:
            if not isinstance(item, dict) or not all(
                isinstance(item_key, str) for item_key in item
            ):
                raise TypeError(
                    f"workflow state object map entry {key!r} is malformed"
                ) from exc
            raise
    return objects


def state_object_list(value: object) -> list[dict[str, JsonValue]]:
    """Decode a persisted list of JSON objects without widening its items."""

    if not isinstance(value, list):
        return []
    objects: list[dict[str, JsonValue]] = []
    for item in value:
        if not isinstance(item, dict) or not all(isinstance(key, str) for key in item):
            raise TypeError("workflow state object list contains a malformed item")
        objects.append(
            decode_json_object(item, label="workflow state object list item")
        )
    return objects


def state_string_list(value: object) -> list[str]:
    """Decode a persisted list whose values are required to be strings."""

    if not isinstance(value, list):
        return []
    if any(not isinstance(item, str) for item in value):
        raise TypeError("workflow state string list contains a non-string item")
    return list(value)


def state_mapping(value: object) -> Mapping[str, JsonValue]:
    """Narrow a persisted JSON object before reading named fields."""

    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return decode_json_object(value, label="workflow state mapping")
    return {}


def state_scalar(value: object) -> JsonValue | None:
    """Return only values safe to emit in a JSON probe snapshot."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return None


def state_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if type(value) is int and value >= 0:
        return value
    return default


def state_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        converted = float(value)
        if math.isfinite(converted):
            return converted
    return default


def coverage_map(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {str(key): state_float(item) for key, item in value.items()}
