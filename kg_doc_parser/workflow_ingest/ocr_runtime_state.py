"""Validation and decoding for persisted OCR workflow state."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from .adapters import _validated_ocr_page_json
from .ocr_pipeline import OCRPageJSON, OCRSourcePlan
from .ocr_runtime_contracts import (
    OCR_PAGE_STATE_VERSION,
    OCR_ROOT_STATE_VERSION,
    OCRWorkflowState,
    PageListKey,
    PageSourceState,
    ResumeIdentityKey,
    ResumeIdentityValue,
    StateBoolKey,
    StateObjectKey,
    StateStringKey,
)
from .serialization import JsonObject, decode_json_object, decode_json_value


def page_source_state(value: object) -> PageSourceState:
    """Decode one persisted page source without widening known fields."""

    if not isinstance(value, Mapping):
        raise TypeError("OCR workflow state page_sources contains a malformed entry")
    page_number = value.get("page_number")
    image_path = value.get("image_path")
    if type(page_number) is not int or not isinstance(image_path, str) or not image_path:
        raise TypeError("OCR workflow state page_sources contains a malformed entry")
    return {"page_number": page_number, "image_path": image_path}


def _json_object(value: object, *, label: str) -> JsonObject:
    return decode_json_object(value, label=f"OCR workflow state field {label!r}")


def _json_object_passes(value: object, *, label: str) -> list[list[JsonObject]]:
    if not isinstance(value, list):
        raise TypeError(f"OCR workflow state field {label!r} must be a list of passes")
    passes: list[list[JsonObject]] = []
    for pass_index, raw_pass in enumerate(value):
        if not isinstance(raw_pass, list):
            raise TypeError(
                f"OCR workflow state field {label!r} pass {pass_index} must be a list"
            )
        passes.append(
            [
                _json_object(item, label=f"{label}[{pass_index}]")
                for item in raw_pass
            ]
        )
    return passes


def _checkpoint_ocr_page(value: object, *, index: int) -> OCRPageJSON:
    try:
        return _validated_ocr_page_json(value, strict_page_number_types=True)
    except TypeError as exc:
        raise TypeError(f"OCR workflow state raw_pages[{index}]: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"OCR workflow state raw_pages[{index}]: {exc}") from exc


def typed_state(value: Mapping[str, object]) -> OCRWorkflowState:
    """Validate checkpoint JSON and narrow known fields to their contracts."""
    if not all(isinstance(key, str) for key in value):
        raise TypeError("OCR workflow state keys must be strings")

    # Kogwistar may inject opaque underscore-prefixed runtime plumbing values.
    decoded_state: dict[str, object] = {}
    for key, field in value.items():
        decoded_state[key] = (
            field
            if key.startswith("_")
            else decode_json_value(field, label=f"OCR workflow state.{key}")
        )
    state = cast(OCRWorkflowState, decoded_state)

    string_keys = (
        "ocr_page_state_version",
        "ocr_root_state_version",
        "document_id",
        "source_fingerprint",
        "source_namespace",
        "grouping_policy_version",
        "parse_generation",
        "legacy_dir",
        "checkpoint_revision",
    )
    for key in string_keys:
        field = value.get(key)
        if field is not None and not isinstance(field, str):
            raise TypeError(f"OCR workflow state field {key!r} must be a string")

    integer_keys = (
        "pass_index",
        "adaptive_grouping_round",
        "adaptive_grouping_window",
        "ocr_page_recovery_generation",
        "page_cursor",
        "last_failed_page",
    )
    for key in integer_keys:
        field = value.get(key)
        if field is not None and type(field) is not int:
            raise TypeError(f"OCR workflow state field {key!r} must be an integer")

    last_reused_page = value.get("last_reused_page")
    if last_reused_page is not None and type(last_reused_page) is not int:
        raise TypeError(
            "OCR workflow state field 'last_reused_page' must be an integer or null"
        )

    boolean_keys = (
        "checkpoint_restored",
        "checkpoint_absent",
        "summaries_generated",
        "adaptive_grouping_complete",
    )
    for key in boolean_keys:
        field = value.get(key)
        if field is not None and type(field) is not bool:
            raise TypeError(f"OCR workflow state field {key!r} must be a boolean")

    object_keys: tuple[StateObjectKey, ...] = (
        "tree",
        "workflow_input",
        "grouping_result",
        "semantic_result",
        "semantic_tree",
        "ocr_page_result",
        "export_bundle",
    )
    for key in object_keys:
        field = value.get(key)
        if field is not None:
            state[key] = _json_object(field, label=key)

    grouping_passes = value.get("grouping_passes")
    if grouping_passes is not None:
        state["grouping_passes"] = _json_object_passes(
            grouping_passes, label="grouping_passes"
        )

    for key in ("adaptive_grouping_history", "adaptive_grouping_window_proposals"):
        field = value.get(key)
        if field is not None:
            if not isinstance(field, list):
                raise TypeError(f"OCR workflow state {key} must be a list")
            state[key] = [
                decode_json_object(item, label=f"{key}[{index}]")
                for index, item in enumerate(field)
            ]

    for key in ("completed_pages", "reused_pages"):
        field = value.get(key)
        if field is not None and (
            not isinstance(field, list) or any(type(item) is not int for item in field)
        ):
            raise TypeError(f"OCR workflow state field {key!r} must be a list of integers")
        if field is not None:
            normalized_pages = cast(list[int], list(field))
            if key == "completed_pages":
                state["completed_pages"] = normalized_pages
            else:
                state["reused_pages"] = normalized_pages

    raw_pages = value.get("raw_pages")
    if raw_pages is not None:
        if not isinstance(raw_pages, list):
            raise TypeError("OCR workflow state field 'raw_pages' must be a list or null")
        state["raw_pages"] = [
            _checkpoint_ocr_page(page, index=index)
            for index, page in enumerate(raw_pages)
        ]

    page_sources = value.get("page_sources")
    if page_sources is not None:
        if not isinstance(page_sources, list):
            raise TypeError("OCR workflow state field 'page_sources' must be a list")
        state["page_sources"] = [page_source_state(item) for item in page_sources]

    return state


def typed_child_state(value: object, *, label: str) -> OCRWorkflowState:
    """Validate and narrow a JSON child-state payload at a workflow seam."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{label} did not return a JSON object state")
    try:
        return typed_state(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} returned invalid typed state: {exc}") from exc


def resume_identity_value(
    state: OCRWorkflowState,
    key: ResumeIdentityKey,
) -> ResumeIdentityValue | None:
    """Read an immutable resume identity field without widening it to object."""
    value = state.get(key)
    if value is None or isinstance(value, str):
        return value
    if key == "ocr_configuration_fingerprints" and isinstance(value, list) and all(
        isinstance(item, str) for item in value
    ):
        return list(value)
    raise TypeError(f"OCR checkpoint identity field {key!r} has an invalid type")


def _validate_resume_identity(
    state: OCRWorkflowState,
    expected: Mapping[ResumeIdentityKey, ResumeIdentityValue],
    *,
    workflow_kind: str,
) -> None:
    missing = [key for key in expected if key not in state]
    if missing:
        raise RuntimeError(
            f"cannot resume OCR {workflow_kind} checkpoint without immutable identity fields: "
            + ", ".join(sorted(missing))
        )
    mismatches = [
        key
        for key, expected_value in expected.items()
        if resume_identity_value(state, key) != expected_value
    ]
    if mismatches:
        raise RuntimeError(
            f"OCR {workflow_kind} checkpoint identity mismatch; refusing to mix persisted state for: "
            + ", ".join(sorted(mismatches))
        )


def validate_root_resume_identity(
    state: OCRWorkflowState,
    *,
    document_id: str,
    source_fingerprint: str,
    source_namespace: str,
    configuration_fingerprints: Sequence[str],
    grouping_policy_version: str,
    parse_generation: str,
    require_match: bool,
) -> None:
    """Reject a root checkpoint built from a different immutable input set."""
    if not require_match:
        return
    expected: dict[ResumeIdentityKey, ResumeIdentityValue] = {
        "ocr_root_state_version": OCR_ROOT_STATE_VERSION,
        "document_id": document_id,
        "source_fingerprint": source_fingerprint,
        "source_namespace": source_namespace,
        "ocr_configuration_fingerprints": list(configuration_fingerprints),
        "grouping_policy_version": grouping_policy_version,
        "parse_generation": parse_generation,
    }
    _validate_resume_identity(state, expected, workflow_kind="root")


def validate_page_resume_identity(
    state: OCRWorkflowState,
    *,
    document_id: str,
    source_fingerprint: str,
    configuration_fingerprints: Sequence[str],
) -> None:
    """Reject a page checkpoint built from a different immutable input set."""
    expected: dict[ResumeIdentityKey, ResumeIdentityValue] = {
        "ocr_page_state_version": OCR_PAGE_STATE_VERSION,
        "document_id": document_id,
        "source_fingerprint": source_fingerprint,
        "ocr_configuration_fingerprints": list(configuration_fingerprints),
    }
    _validate_resume_identity(state, expected, workflow_kind="page")


def state_string(state: OCRWorkflowState, key: StateStringKey, default: str) -> str:
    value = state.get(key)
    return value if isinstance(value, str) else default


def state_optional_string(
    state: OCRWorkflowState,
    key: StateStringKey,
) -> str | None:
    value = state.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"OCR workflow state field {key!r} must be a string")
    return value


def state_bool(
    state: OCRWorkflowState,
    key: StateBoolKey,
    default: bool = False,
) -> bool:
    value = state.get(key)
    return value if type(value) is bool else default


def page_cursor(state: OCRWorkflowState) -> int:
    value = state.get("page_cursor", 0)
    return value if type(value) is int and value >= 0 else 0


def page_sources(state: OCRWorkflowState) -> list[tuple[int, Path]]:
    value = state.get("page_sources", [])
    if not isinstance(value, list):
        raise TypeError("OCR page runtime state has no page source list")
    sources: list[tuple[int, Path]] = []
    for item in value:
        source = page_source_state(item)
        page_number = source["page_number"]
        image_path = source["image_path"]
        if page_number < 1:
            raise ValueError("OCR page runtime source page number is malformed")
        sources.append((page_number, Path(image_path)))
    return sources


def json_pages(
    state: OCRWorkflowState,
    *,
    source_plan: OCRSourcePlan | None = None,
) -> list[OCRPageJSON]:
    """Load accepted pages from legacy checkpoints or their artifact files."""
    value = state.get("raw_pages")
    if value is not None:
        if not isinstance(value, list):
            raise ValueError("OCR page runtime raw page state is malformed")
        return [
            _validated_ocr_page_json(item, strict_page_number_types=True)
            for item in value
        ]

    legacy_dir_value = state.get("legacy_dir")
    if not isinstance(legacy_dir_value, str) or not legacy_dir_value:
        raise ValueError("OCR page runtime state has no legacy artifact directory")
    cursor = page_cursor(state)
    pages: list[OCRPageJSON] = []
    accepted_sources = (
        source_plan.page_sources if source_plan is not None else page_sources(state)
    )
    for page_number, _image_path in accepted_sources[:cursor]:
        page_path = Path(legacy_dir_value) / f"page_{page_number}.json"
        try:
            payload = json.loads(page_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"OCR page artifact is not readable: {page_path}") from exc
        if not isinstance(payload, dict):
            raise TypeError(f"OCR page artifact is not a JSON object: {page_path}")
        pages.append(
            _validated_ocr_page_json(payload, default_page_number=page_number)
        )
    return pages


def int_list(state: OCRWorkflowState, key: PageListKey) -> list[int]:
    value = state.get(key, [])
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ValueError(f"OCR page runtime state field {key!r} is malformed")
    result: list[int] = [item for item in value if type(item) is int]
    if len(result) != len(value):
        raise ValueError(f"OCR page runtime state field {key!r} is malformed")
    return result
