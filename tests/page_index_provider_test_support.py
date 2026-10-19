"""Shared fake structured-output provider for PageIndex workflow tests."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import BaseModel

import kg_doc_parser.workflow_ingest.page_index as page_index_module
import kg_doc_parser.workflow_ingest.page_index_contracts as page_index_contracts_module
import kg_doc_parser.workflow_ingest.page_index_provider as page_index_provider_module
import kg_doc_parser.workflow_ingest.page_index_refinement as page_index_refinement_module
from kg_doc_parser.workflow_ingest import BlockAssignmentBatch


def install_fake_page_index_chat(
    monkeypatch: pytest.MonkeyPatch,
    *,
    assignment_payload: BlockAssignmentBatch | None,
    assignment_parsing_error: str | None = None,
    refinement_payload: BaseModel | None = None,
    refinement_parsing_error: str | None = None,
    capture_messages: list[object] | None = None,
) -> None:
    class _FakeStructured:
        def __init__(self, schema: object) -> None:
            self.schema = schema

        def invoke(self, messages: Sequence[object]) -> dict[str, object]:
            if capture_messages is not None:
                capture_messages.extend(messages)
            if self.schema is page_index_module.BlockAssignmentBatch:
                if assignment_parsing_error is not None:
                    return {"parsed": None, "parsing_error": assignment_parsing_error}
                return {"parsed": assignment_payload}
            if self.schema is page_index_contracts_module.ExcerptRefinementBatch:
                if refinement_parsing_error is not None:
                    return {"parsed": None, "parsing_error": refinement_parsing_error}
                return {"parsed": refinement_payload}
            raise AssertionError(f"unexpected structured schema: {self.schema!r}")

    class _FakeChat:
        def with_structured_output(self, schema: object, include_raw: bool = True) -> _FakeStructured:
            del include_raw
            return _FakeStructured(schema)

    monkeypatch.setattr(
        page_index_provider_module,
        "build_chat_model_for_role",
        lambda *args, **kwargs: _FakeChat(),
    )
    monkeypatch.setattr(
        page_index_refinement_module,
        "build_chat_model_for_role",
        lambda *args, **kwargs: _FakeChat(),
    )
