"""Bounded JSON conversion for provider diagnostics and prompt payloads."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Protocol, cast
from uuid import UUID

try:
    from typing import TypeAliasType
except ImportError:  # PyPy 3.11 does not expose the Python 3.12 alias API.
    from typing_extensions import TypeAliasType  # noqa: UP035

JsonScalar = TypeAliasType("JsonScalar", None | bool | int | float | str)  # noqa: UP040
JsonValue = TypeAliasType(  # noqa: UP040
    "JsonValue",
    JsonScalar | list["JsonValue"] | dict[str, "JsonValue"],
)


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
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            dumper = cast(ModelDumpLike, value)
            try:
                return json_safe(dumper.model_dump(mode="python"), _seen=seen)
            except TypeError:
                return json_safe(dumper.model_dump(), _seen=seen)
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


__all__ = ["JsonValue", "json_safe", "safe_json_dumps"]
