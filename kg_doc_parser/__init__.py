"""Public import surface for the document parser package."""

from __future__ import annotations

from . import workflow_ingest
from .workflow_ingest import *  # noqa: F401,F403
from .workflow_ingest import __all__ as _workflow_ingest_all

# The parser package intentionally mirrors the workflow package's public API.
__all__ = ["workflow_ingest", *_workflow_ingest_all]  # pyright: ignore[reportUnsupportedDunderAll]
