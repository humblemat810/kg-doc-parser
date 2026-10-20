from __future__ import annotations

from typing import cast

import pytest

from kg_doc_parser.workflow_ingest.ocr_runtime_contracts import (
    OCRWorkflowState as _OCRWorkflowState,
)
from kg_doc_parser.workflow_ingest.ocr_runtime_state import (
    page_source_state as _page_source_state,
    resume_identity_value as _resume_identity_value,
)


def test_resume_identity_reader_rejects_non_string_fingerprint_values() -> None:
    state: _OCRWorkflowState = {
        "ocr_configuration_fingerprints": ["ocr-v1"],
    }

    assert _resume_identity_value(state, "ocr_configuration_fingerprints") == ["ocr-v1"]

    malformed = cast(
        _OCRWorkflowState,
        {"ocr_configuration_fingerprints": ["ocr-v1", 7]},
    )
    with pytest.raises(TypeError, match="invalid type"):
        _resume_identity_value(malformed, "ocr_configuration_fingerprints")


@pytest.mark.parametrize(
    "payload",
    [
        {"page_number": "1", "image_path": "page-1.png"},
        {"page_number": 1, "image_path": ""},
        {"page_number": 1},
    ],
)
def test_page_source_decoder_rejects_malformed_persisted_entries(payload: object) -> None:
    with pytest.raises(TypeError, match="malformed entry"):
        _page_source_state(payload)
