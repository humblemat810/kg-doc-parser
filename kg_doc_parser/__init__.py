from __future__ import annotations

"""Public import facade for the document parser package.

The repo historically exposed its implementation under ``src``. This package
provides the stable import name ``kg_doc_parser`` while keeping the existing
module layout intact.
"""

import sys

from src.workflow_ingest import *  # noqa: F401,F403
from src.workflow_ingest import __all__ as _workflow_ingest_all

__all__ = list(_workflow_ingest_all)

import src.workflow_ingest as _workflow_ingest

sys.modules.setdefault(__name__ + ".workflow_ingest", _workflow_ingest)
