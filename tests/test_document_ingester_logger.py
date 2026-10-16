from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from kg_doc_parser.document_ingester_logger import _message_usage_metadata


def test_message_usage_metadata_narrows_optional_provider_extension() -> None:
    usage = {"input_tokens": 7, "output_tokens": 3}
    provider_message = SimpleNamespace(usage_metadata=usage)
    ordinary_message = HumanMessage(content="hello")
    malformed_message = SimpleNamespace(usage_metadata=["not", "a", "mapping"])

    assert _message_usage_metadata(provider_message) == usage
    assert _message_usage_metadata(ordinary_message) == {}
    assert _message_usage_metadata(malformed_message) == {}
