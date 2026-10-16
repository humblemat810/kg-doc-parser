"""Bounded JSON conversion for provider diagnostics and prompt payloads."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable
from typing_extensions import TypeAliasType
from uuid import UUID

JsonScalar = TypeAliasType("JsonScalar", None | bool | int | float | str)
JsonValue = TypeAliasType(
    "JsonValue",
    JsonScalar | list["JsonValue"] | dict[str, "JsonValue"],
)
JsonObject = dict[str, JsonValue]


def decode_json_value(value: object, *, label: str = "value") -> JsonValue:
    """Validate and copy a JSON value at an internal persistence boundary."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [
            decode_json_value(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        decoded: JsonObject = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{label} contains a non-string object key")
            decoded[key] = decode_json_value(item, label=f"{label}.{key}")
        return decoded
    raise TypeError(f"{label} is not JSON serializable: {type(value).__name__}")


def decode_json_object(value: object, *, label: str = "object") -> JsonObject:
    """Validate and copy a JSON object without an unchecked cast."""

    decoded = decode_json_value(value, label=label)
    if not isinstance(decoded, dict):
        raise TypeError(f"{label} must be JSON objects")
    return decoded


@runtime_checkable
class ModelDumpLike(Protocol):
    """Minimal serialization seam for Pydantic-like provider payloads."""

    def model_dump(self, **kwargs: object) -> object: ...


def json_safe(value: object, *, _seen: set[int] | None = None) -> JsonValue:
    """Convert structured and third-party values into JSON-safe primitives.

    This is intentionally conservative at provider boundaries: unknown objects
    become bounded strings rather than escaping into a callback/logger failure.
    Cycles are represented explicitly instead of recursing forever.
    """

    seen = _seen if _seen is not None else set()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (UUID, Path)):
        return str(value)
    identity = id(value)
    if identity in seen:
        return "<cycle>"
    seen.add(identity)
    try:
        if isinstance(value, ModelDumpLike):
            try:
                return json_safe(value.model_dump(mode="python"), _seen=seen)
            except TypeError:
                return json_safe(value.model_dump(), _seen=seen)
        if is_dataclass(value) and not isinstance(value, type):
            return json_safe(asdict(value), _seen=seen)
        if isinstance(value, Mapping):
            return {
                str(key): json_safe(item, _seen=seen)
                for key, item in value.items()
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [json_safe(item, _seen=seen) for item in value]
        if isinstance(value, (set, frozenset)):
            return [json_safe(item, _seen=seen) for item in sorted(value, key=str)]
        return str(value)
    finally:
        seen.discard(identity)


def safe_json_dumps(
    value: object,
    *,
    ensure_ascii: bool = True,
    sort_keys: bool = False,
) -> str:
    """Serialize a value after applying :func:`json_safe`."""

    return json.dumps(
        json_safe(value),
        ensure_ascii=ensure_ascii,
        sort_keys=sort_keys,
    )


__all__ = [
    "JsonObject",
    "JsonValue",
    "decode_json_object",
    "decode_json_value",
    "json_safe",
    "safe_json_dumps",
]
