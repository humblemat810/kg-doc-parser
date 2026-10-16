from __future__ import annotations

import pytest
from typing import cast

from kg_doc_parser.workflow_ingest.ocr_runtime import (
    _resume_identity_value,
    _OCRWorkflowState,
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
