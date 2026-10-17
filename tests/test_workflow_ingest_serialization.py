from __future__ import annotations

from uuid import UUID

from kg_doc_parser.workflow_ingest.serialization import json_safe


class _Dumpable:
    def model_dump(self, **kwargs: object) -> object:
        del kwargs
        return {"items": [1, "two", None]}


def test_json_safe_preserves_recursive_json_sum_types() -> None:
    value = json_safe(
        {
            "nested": [{"uuid": UUID("00000000-0000-0000-0000-000000000001")}],
            "dumpable": _Dumpable(),
        }
    )

    assert value == {
        "nested": [{"uuid": "00000000-0000-0000-0000-000000000001"}],
        "dumpable": {"items": [1, "two", None]},
    }
