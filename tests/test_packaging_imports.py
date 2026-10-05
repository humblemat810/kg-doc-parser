from __future__ import annotations

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


def test_ci_uses_the_released_kogwistar_package_and_pinned_pypy_source_revision() -> None:
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    declared_version = metadata["tool"]["poetry"]["dependencies"]["kogwistar"]
    lock = tomllib.loads((root / "poetry.lock").read_text(encoding="utf-8"))
    locked_package = next(
        package for package in lock["package"] if package["name"] == "kogwistar"
    )
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert declared_version == "0.6.3"
    assert locked_package["version"] == declared_version
    assert "source" not in locked_package
    assert "ref: v0.6.3" in workflow
