"""Typed decoding helpers for persisted OCR page artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import NotRequired, TypedDict

from .serialization import JsonObject, decode_json_object


class OCRArtifactMetadata(TypedDict, total=False):
    """Known metadata fields read from persisted OCR artifact JSON."""

    ocr_provider: NotRequired[str]
    ocr_model_name: NotRequired[str]
    ocr_configuration_fingerprint: NotRequired[str]
    source_image_sha256: NotRequired[str]
    ocr_json_version: NotRequired[str]
    ocr_datetime: NotRequired[float]


SUPPORTED_OCR_JSON_VERSIONS = frozenset({"0.1", "workflow_ingest_v1"})


def json_object(value: object) -> JsonObject:
    """Decode a JSON-compatible mapping without trusting nested values."""
    try:
        return decode_json_object(value, label="OCR artifact")
    except (TypeError, ValueError):
        return {}


def read_json_object(path: Path) -> JsonObject:
    """Read optional resume metadata without making it authoritative."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return json_object(value)


def artifact_metadata(payload: Mapping[str, object]) -> OCRArtifactMetadata:
    """Narrow fields used by artifact reuse and recovery decisions."""
    raw_metadata = payload.get("metadata")
    if not isinstance(raw_metadata, Mapping):
        return {}
    metadata: OCRArtifactMetadata = {}
    for key in (
        "ocr_provider",
        "ocr_model_name",
        "ocr_configuration_fingerprint",
        "source_image_sha256",
        "ocr_json_version",
    ):
        value = raw_metadata.get(key)
        if isinstance(value, str):
            metadata[key] = value
    timestamp = raw_metadata.get("ocr_datetime")
    if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool):
        metadata["ocr_datetime"] = float(timestamp)
    return metadata


def artifact_schema_error(payload: Mapping[str, object]) -> str | None:
    """Reject unknown serialized OCR contracts while retaining legacy data."""
    version = artifact_metadata(payload).get("ocr_json_version")
    if version not in SUPPORTED_OCR_JSON_VERSIONS:
        received = repr(version) if version is not None else "missing"
        return f"unsupported OCR artifact schema version: {received}"
    return None
