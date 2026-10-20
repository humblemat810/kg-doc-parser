"""Closed type contracts for OCR page-state persistence."""

from __future__ import annotations

from typing import Literal, TypeAlias

OCRStage: TypeAlias = Literal["render", "ocr"]  # noqa: UP040 - Python 3.11 support
OCRPageStatus: TypeAlias = Literal[  # noqa: UP040 - Python 3.11 support
    "pending", "completed", "failed"
]


def decode_ocr_stage(value: object) -> OCRStage:
    """Validate a stage read from the durable OCR state store."""

    if value == "render":
        return "render"
    if value == "ocr":
        return "ocr"
    raise ValueError(f"persisted OCR stage is invalid: {value!r}")


def decode_ocr_page_status(value: object) -> OCRPageStatus:
    """Validate a page status read from the durable OCR state store."""

    if value == "pending":
        return "pending"
    if value == "completed":
        return "completed"
    if value == "failed":
        return "failed"
    raise ValueError(f"persisted OCR page status is invalid: {value!r}")
