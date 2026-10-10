from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

pytestmark = [pytest.mark.ci]


def test_kg_doc_parser_import_surface_is_available() -> None:
    import kg_doc_parser
    from kg_doc_parser import workflow_ingest

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


def test_ci_uses_the_pinned_kogwistar_commit_and_pinned_pypy_source_revision() -> None:
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    declared_dependency = metadata["tool"]["poetry"]["dependencies"]["kogwistar"]
    lock = tomllib.loads((root / "poetry.lock").read_text(encoding="utf-8"))
    locked_package = next(
        package for package in lock["package"] if package["name"] == "kogwistar"
    )
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert declared_dependency["rev"] == "78b88c44d7d61bf1d9b2b7e3fdaf435df4687d09"
    assert locked_package["version"] == "0.6.6"
    assert locked_package["source"]["reference"] == declared_dependency["rev"]
    assert "ref: v0.6.6" in workflow


def test_cloud_adapter_extras_are_declared_without_changing_base_install() -> None:
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = metadata["tool"]["poetry"]["dependencies"]
    extras = metadata["tool"]["poetry"]["extras"]

    assert dependencies["langchain-google-genai"]["optional"] is True
    assert dependencies["langchain-openai"]["optional"] is True
    assert dependencies["langchain-google-vertexai"]["optional"] is True
    assert extras["openai"] == ["langchain-openai"]
    assert extras["azure"] == ["langchain-openai"]
    assert extras["gemini"] == ["langchain-google-genai"]
    assert extras["vertex"] == ["langchain-google-vertexai"]
    assert extras["cloud"] == [
        "langchain-google-genai",
        "langchain-openai",
        "langchain-google-vertexai",
    ]
