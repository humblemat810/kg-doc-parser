from __future__ import annotations

import pytest

from kg_doc_parser.workflow_ingest.parser_core import _coerce_semantic_tree
from kg_doc_parser.workflow_ingest.semantics import SemanticNode

pytestmark = pytest.mark.ci


def test_semantic_tree_coercion_uses_typed_model_dump_boundary() -> None:
    expected = SemanticNode(node_id="doc|root", title="Document")

    class _TreeCarrier:
        def model_dump(self, **kwargs: object) -> dict[str, object]:
            assert kwargs == {"mode": "json"}
            return expected.model_dump(mode="json")

    assert _coerce_semantic_tree(_TreeCarrier()).node_id == "doc|root"
