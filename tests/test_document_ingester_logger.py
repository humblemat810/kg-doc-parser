from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from kg_doc_parser.document_ingester_logger import _message_usage_metadata
from kg_doc_parser.utils.langchain import (
    _generation_message,
    _model_name,
    _response_metadata,
    _usage_metadata,
)


def test_message_usage_metadata_narrows_optional_provider_extension() -> None:
    usage = {"input_tokens": 7, "output_tokens": 3}
    provider_message = SimpleNamespace(usage_metadata=usage)
    ordinary_message = HumanMessage(content="hello")
    malformed_message = SimpleNamespace(usage_metadata=["not", "a", "mapping"])

    assert _message_usage_metadata(provider_message) == usage
    assert _message_usage_metadata(ordinary_message) == {}
    assert _message_usage_metadata(malformed_message) == {}


def test_langchain_accounting_seams_narrow_optional_sdk_attributes() -> None:
    generation = SimpleNamespace(
        message=SimpleNamespace(
            usage_metadata={"input_tokens": 4},
            response_metadata={"model_name": "provider-model"},
        ),
        generation_info={"model_name": "generation-model"},
    )
    malformed = SimpleNamespace(
        message=SimpleNamespace(usage_metadata=["bad"], response_metadata=[]),
        generation_info=[],
    )

    assert _generation_message(generation) is generation.message
    assert _usage_metadata(generation.message) == {"input_tokens": 4}
    assert _response_metadata(generation.message) == {"model_name": "provider-model"}
    assert _model_name(generation) == "generation-model"
    assert _generation_message(malformed) is malformed.message
    assert _usage_metadata(malformed.message) is None
    assert _response_metadata(malformed.message) == []
    assert _model_name(malformed) == "unknown"
