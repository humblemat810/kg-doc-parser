"""Typed helpers for optional third-party SDK response attributes."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ContentCarrier(Protocol):
    @property
    def content(self) -> object: ...


def raw_content(value: object) -> object | None:
    """Read SDK message content without treating arbitrary objects as messages."""

    if isinstance(value, ContentCarrier):
        content = value.content
        return content if content is not None else value
    return value
