from __future__ import annotations

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
    with pytest.raises(ModuleNotFoundError):
        __import__("src")


def test_package_modules_import_cleanly() -> None:
    import kg_doc_parser.ocr
    import kg_doc_parser.workflow_ingest

    assert kg_doc_parser.workflow_ingest.parse_document is not None
    assert kg_doc_parser.ocr.regen_doc is not None
