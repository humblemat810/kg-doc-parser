"""Source-map text and pointer-span operations used by layerwise parsing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

from .semantics import HydratedTextPointer


class SourceMapExcerptRecord(TypedDict):
    page_number: int | None
    cluster_number: int | None
    text: str


def trim_text(value: object, *, max_chars: int = 400) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def source_map_excerpt(
    parser_source_map: Mapping[str, object],
    *,
    max_records: int = 12,
    max_text_chars: int = 1200,
) -> dict[str, SourceMapExcerptRecord]:
    excerpt: dict[str, SourceMapExcerptRecord] = {}
    for key, record in list(parser_source_map.items())[:max_records]:
        raw_page_number = record.get("page_number") if isinstance(record, Mapping) else None
        raw_cluster_number = record.get("cluster_number") if isinstance(record, Mapping) else None
        excerpt[str(key)] = {
            "page_number": raw_page_number if type(raw_page_number) is int else None,
            "cluster_number": raw_cluster_number if type(raw_cluster_number) is int else None,
            "text": trim_text(record_text(record), max_chars=max_text_chars),
        }
    return excerpt


def record_text(record: object) -> str:
    """Read source text without converting malformed records into evidence."""

    if not isinstance(record, Mapping):
        return ""
    value = record.get("text", "")
    return value if isinstance(value, str) else ""


def pointer_field(pointer: object, field_name: str) -> object:
    if isinstance(pointer, Mapping):
        return pointer.get(field_name)
    if isinstance(pointer, HydratedTextPointer):
        fields: dict[str, str | int] = {
            "source_cluster_id": pointer.source_cluster_id,
            "start_char": pointer.start_char,
            "end_char": pointer.end_char,
            "verbatim_text": pointer.verbatim_text,
        }
        return fields.get(field_name)
    return None


def pointer_source_cluster_id(pointer: object) -> str:
    value = pointer_field(pointer, "source_cluster_id")
    return value if isinstance(value, str) else ""


def offset_value(value: object, default: int) -> int:
    """Decode a persisted character offset without treating booleans as ints."""

    if type(value) is int:
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def pointer_signature(
    pointer: object,
    *,
    parser_source_map: Mapping[str, Mapping[str, object]],
) -> tuple[str, int, int, str]:
    return (
        pointer_source_cluster_id(pointer),
        offset_value(pointer_field(pointer, "start_char"), 0),
        offset_value(pointer_field(pointer, "end_char"), -1),
        pointer_text(pointer, parser_source_map=parser_source_map).strip(),
    )


def pointer_end_inclusive(
    pointer: object,
    *,
    parser_source_map: Mapping[str, Mapping[str, object]],
) -> int:
    end_char = pointer_field(pointer, "end_char")
    if type(end_char) is int and end_char >= 0:
        return end_char
    source_cluster_id = pointer_source_cluster_id(pointer)
    record = parser_source_map.get(source_cluster_id) or parser_source_map.get(
        str(source_cluster_id)
    )
    text = record_text(record)
    if text:
        return max(0, len(text) - 1)
    return offset_value(pointer_field(pointer, "start_char"), 0)


def pointer_span_bounds(
    pointer: object,
    *,
    parser_source_map: Mapping[str, Mapping[str, object]],
) -> tuple[int, int]:
    start_char = offset_value(pointer_field(pointer, "start_char"), 0)
    end_char_exclusive = pointer_end_inclusive(
        pointer, parser_source_map=parser_source_map
    ) + 1
    return start_char, max(start_char + 1, end_char_exclusive)


def pointer_text(
    pointer: object,
    *,
    parser_source_map: Mapping[str, Mapping[str, object]],
) -> str:
    source_cluster_id = pointer_source_cluster_id(pointer)
    start_char = pointer_field(pointer, "start_char")
    end_char = pointer_field(pointer, "end_char")
    record = parser_source_map.get(source_cluster_id) or parser_source_map.get(
        str(source_cluster_id)
    )
    raw_text = record_text(record)
    if not raw_text or type(start_char) is not int or type(end_char) is not int:
        return ""
    resolved_end = len(raw_text) - 1 if end_char == -1 else end_char
    if 0 <= start_char <= resolved_end < len(raw_text):
        return raw_text[start_char : resolved_end + 1]
    return ""


def pointer_span_text(
    pointer: object,
    *,
    parser_source_map: Mapping[str, Mapping[str, object]],
    start_offset: int | None = None,
    end_offset: int | None = None,
) -> str:
    source_cluster_id = pointer_source_cluster_id(pointer)
    start_char, end_char_exclusive = pointer_span_bounds(
        pointer, parser_source_map=parser_source_map
    )
    record = parser_source_map.get(source_cluster_id) or parser_source_map.get(
        str(source_cluster_id)
    )
    raw_text = record_text(record)
    if not raw_text:
        return trim_text(pointer_text(pointer, parser_source_map=parser_source_map))
    start = max(start_char, 0 if start_offset is None else start_offset)
    end = end_char_exclusive if end_offset is None else min(end_char_exclusive, end_offset)
    if end <= start:
        return ""
    end = min(len(raw_text), end)
    if start >= len(raw_text):
        return ""
    return raw_text[start:end]


_SourceMapExcerptRecord = SourceMapExcerptRecord
_trim_text = trim_text
_source_map_excerpt = source_map_excerpt
_record_text = record_text
_pointer_field = pointer_field
_pointer_source_cluster_id = pointer_source_cluster_id
_offset_value = offset_value
_pointer_signature = pointer_signature
_pointer_end_inclusive = pointer_end_inclusive
_pointer_span_bounds = pointer_span_bounds
_pointer_text = pointer_text
_pointer_span_text = pointer_span_text

__all__ = [
    "SourceMapExcerptRecord",
    "offset_value",
    "pointer_end_inclusive",
    "pointer_field",
    "pointer_signature",
    "pointer_source_cluster_id",
    "pointer_span_bounds",
    "pointer_span_text",
    "pointer_text",
    "record_text",
    "source_map_excerpt",
    "trim_text",
]
