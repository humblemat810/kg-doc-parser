from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

import pytest


pytestmark = [pytest.mark.ci]


def test_kg_doc_parser_import_surface_is_available() -> None:
    import kg_doc_parser
    import kg_doc_parser.workflow_ingest as workflow_ingest

    assert hasattr(kg_doc_parser, "parse_document")
    assert kg_doc_parser.parse_document is workflow_ingest.parse_document
    assert hasattr(workflow_ingest, "parse_ocr_document")
    assert hasattr(workflow_ingest, "parse_page_index_document")
    assert hasattr(workflow_ingest, "parse_tree_document")
    assert hasattr(kg_doc_parser, "workflow_ingest")


def test_src_package_is_not_importable() -> None:
    parser_src = (Path(__file__).resolve().parents[1] / "src").resolve()
    assert str(parser_src) not in {str(Path(entry).resolve()) for entry in sys.path if entry}


def test_package_modules_import_cleanly() -> None:
    import kg_doc_parser.ocr
    import kg_doc_parser.workflow_ingest

    assert kg_doc_parser.workflow_ingest.parse_document is not None
    assert kg_doc_parser.ocr.regen_doc is not None


def test_ci_checks_out_the_declared_kogwistar_revision() -> None:
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    declared_revision = metadata["tool"]["poetry"]["dependencies"]["kogwistar"]["rev"]
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    checkout_revision = re.search(r"ref:\s*([0-9a-f]{40})", workflow)

    assert checkout_revision is not None
    assert checkout_revision.group(1) == declared_revision
