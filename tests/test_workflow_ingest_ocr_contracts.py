"""Focused tests for public OCR workflow value contracts."""

from __future__ import annotations

import pytest

from kg_doc_parser.workflow_ingest.adapters import OCRPageJSON
from kg_doc_parser.workflow_ingest.ocr_page_processing import OCRPageOutcome
from kg_doc_parser.workflow_ingest.ocr_runtime import _checkpoint_repair_cursor
from kg_doc_parser.workflow_ingest.ocr_runtime_contracts import OCRCheckpointPatch

pytestmark = [pytest.mark.workflow]


@pytest.mark.ci
def test_ocr_page_outcome_enforces_accepted_and_failed_shapes() -> None:
    page: OCRPageJSON = {"pdf_page_num": 1}
    accepted = OCRPageOutcome(page_number=1, status="completed", page=page)
    assert accepted.page_number == 1 and accepted.page == page
    assert OCRPageOutcome(page_number=1, status="failed", error="provider failed").error
    with pytest.raises(ValueError, match="accepted page outcomes require a page"):
        OCRPageOutcome(page_number=1, status="reused")
    with pytest.raises(ValueError, match="failed page outcomes require an error"):
        OCRPageOutcome(page_number=1, status="failed")


@pytest.mark.ci
def test_ocr_checkpoint_repair_contract_requires_cursor() -> None:
    assert "page_cursor" in OCRCheckpointPatch.__required_keys__
    assert _checkpoint_repair_cursor({"page_cursor": 3}) == 3
    with pytest.raises(ValueError, match="requires a non-negative page_cursor"):
        _checkpoint_repair_cursor({})
    with pytest.raises(ValueError, match="requires a non-negative page_cursor"):
        _checkpoint_repair_cursor({"page_cursor": -1})
