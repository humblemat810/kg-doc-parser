"""Tests for opaque OCR SDK response decoding boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kg_doc_parser.ocr import _raw_content, _raw_usage_metadata
from kg_doc_parser.workflow_ingest.sdk_boundary import ContentCarrier, raw_content

pytestmark = [pytest.mark.workflow]


def test_raw_ocr_content_narrows_sdk_content_without_stringifying_other_values() -> None:
    sdk_payload = {"boxes_2d": [], "non_text_objects": []}
    sdk_response = SimpleNamespace(content=sdk_payload)

    assert _raw_content(sdk_response) is sdk_payload
    mapping_payload = {"content": "already decoded"}
    assert _raw_content(mapping_payload) is mapping_payload
    assert _raw_content(None) is None


def test_shared_sdk_content_boundary_preserves_mapping_and_empty_content_rules() -> None:
    provider_message = SimpleNamespace(content="ocr text")
    empty_message = SimpleNamespace(content=None)
    mapping_payload = {"content": "already decoded"}

    assert isinstance(provider_message, ContentCarrier)
    assert raw_content(provider_message) == "ocr text"
    assert raw_content(empty_message) is empty_message
    assert raw_content(mapping_payload) is mapping_payload


def test_ocr_usage_metadata_boundary_rejects_mapping_and_missing_payloads() -> None:
    sdk_response = SimpleNamespace(usage_metadata={"input_tokens": 4})
    mapping_payload = {"usage_metadata": {"input_tokens": 9}}

    assert _raw_usage_metadata(sdk_response) == {"input_tokens": 4}
    assert _raw_usage_metadata(mapping_payload) is None
    assert _raw_usage_metadata(None) is None
