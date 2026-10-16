from __future__ import annotations

import json
from pathlib import Path

from kg_doc_parser.workflow_ingest.probe import WorkflowProbe


class _BackendDump:
    def model_dump(self, **kwargs: object) -> object:
        if kwargs == {"field_mode": "backend", "dump_format": "json"}:
            return {"value": "backend"}
        return {"value": "legacy"}


def test_probe_serializes_protocol_models_without_dynamic_attribute_lookup(
    tmp_path: Path,
) -> None:
    path = tmp_path / "probe.jsonl"
    probe = WorkflowProbe(path)
    probe.emit("model", value=_BackendDump())

    event = json.loads(path.read_text(encoding="utf-8"))
    assert event["value"] == {"value": "backend"}
