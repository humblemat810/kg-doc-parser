"""Typed checkpoint contracts shared by the OCR workflow runtime."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

from .ocr_pipeline import OCRPageJSON
from .serialization import JsonObject


class PageSourceState(TypedDict):
    page_number: int
    image_path: str


class OCRCheckpointPatch(TypedDict):
    """Validated partial state used to rewind an accepted page prefix."""

    page_cursor: int
    completed_pages: NotRequired[list[int]]
    reused_pages: NotRequired[list[int]]
    raw_pages: NotRequired[list[OCRPageJSON] | None]


PageListKey = Literal["completed_pages", "reused_pages"]
StateStringKey = Literal[
    "document_id",
    "source_fingerprint",
    "source_namespace",
    "grouping_policy_version",
    "parse_generation",
    "legacy_dir",
    "checkpoint_revision",
]
StateBoolKey = Literal[
    "adaptive_grouping_complete",
    "checkpoint_absent",
    "checkpoint_restored",
    "summaries_generated",
]
StateObjectKey = Literal[
    "tree",
    "workflow_input",
    "grouping_result",
    "semantic_result",
    "semantic_tree",
    "ocr_page_result",
    "export_bundle",
]
ResumeIdentityValue = str | list[str]
ResumeIdentityKey = Literal[
    "ocr_root_state_version",
    "ocr_page_state_version",
    "document_id",
    "source_fingerprint",
    "source_namespace",
    "ocr_configuration_fingerprints",
    "grouping_policy_version",
    "parse_generation",
]


class OCRWorkflowState(TypedDict, total=False):
    ocr_page_state_version: str
    ocr_root_state_version: str
    document_id: str
    source_fingerprint: str
    source_namespace: str
    ocr_configuration_fingerprints: list[str]
    grouping_policy_version: str
    parse_generation: str
    tree: JsonObject
    pass_index: int
    ocr_page_recovery_generation: int
    page_cursor: int
    last_reused_page: int | None
    page_sources: list[PageSourceState]
    legacy_dir: str
    raw_pages: list[OCRPageJSON] | None
    completed_pages: list[int]
    reused_pages: list[int]
    last_failed_page: int
    checkpoint_revision: str
    grouping_passes: list[list[JsonObject]]
    adaptive_grouping_round: int
    adaptive_grouping_history: list[JsonObject]
    adaptive_grouping_complete: bool
    adaptive_grouping_window: int
    adaptive_grouping_window_proposals: list[JsonObject]
    workflow_input: JsonObject
    grouping_result: JsonObject
    semantic_result: JsonObject
    semantic_tree: JsonObject
    ocr_page_result: JsonObject
    export_bundle: JsonObject
    checkpoint_restored: bool
    checkpoint_absent: bool
    summaries_generated: bool
