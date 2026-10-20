"""Strict extraction of text evidence from legacy source-map records."""

from __future__ import annotations

from collections.abc import Mapping


def _string_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            return None
        result[key] = item
    return result


def _first_text(record: dict[str, object], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def source_map_entry_text(entry: object) -> str:
    """Return actual string evidence, never a stringified arbitrary value."""
    record = _string_mapping(entry)
    if record is None:
        return ""

    direct_text = _first_text(
        record, ("text", "content", "verbatim_text", "snippet", "summary")
    )
    if direct_text is not None:
        return direct_text

    raw_clusters = record.get("OCR_text_clusters")
    if isinstance(raw_clusters, list):
        parts: list[str] = []
        for raw_cluster in raw_clusters:
            cluster = _string_mapping(raw_cluster)
            if cluster is None:
                continue
            text = _first_text(
                cluster, ("text", "content", "verbatim_text", "snippet")
            )
            if text:
                parts.append(text)
        if parts:
            return "\n".join(parts)

    raw_objects = record.get("non_text_objects")
    if isinstance(raw_objects, list):
        parts = []
        for raw_object in raw_objects:
            item = _string_mapping(raw_object)
            if item is None:
                continue
            text = _first_text(item, ("description", "text"))
            if text:
                parts.append(text)
        if parts:
            return "\n".join(parts)

    return ""
