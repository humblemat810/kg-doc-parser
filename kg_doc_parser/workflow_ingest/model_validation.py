"""Validation helpers for trusted, internally persisted workflow state."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

_ModelT = TypeVar("_ModelT", bound=BaseModel)


def validate_backend_payload(model_type: type[_ModelT], payload: object) -> _ModelT:
    """Validate a known backend payload without stack-based mode inference."""

    return model_type.model_validate(
        payload,
        context={getattr(model_type, "context_mode_key", "modeslice:mode"): "backend"},
    )
