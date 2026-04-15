from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, TypeVar

from kogwistar.id_provider import stable_id

from .probe import emit_probe_event

T = TypeVar("T")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(field_mode="backend", dump_format="json")
        except TypeError:
            return value.model_dump()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class WorkflowLLMCallCache:
    def __init__(self, cache_dir: str | Path, *, probe=None) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.probe = probe

    def _cache_path(self, operation: str, fingerprint: dict[str, Any]) -> Path:
        payload = json.dumps(_jsonable(fingerprint), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        cache_id = stable_id("workflow_ingest.llm_call", operation, payload)
        return self.cache_dir / f"{cache_id}.json"

    def cached_call(
        self,
        *,
        operation: str,
        fingerprint: dict[str, Any],
        fn: Callable[[], T],
    ) -> Any:
        path = self._cache_path(operation, fingerprint)
        if path.exists():
            emit_probe_event(
                self.probe,
                "workflow.llm_cache_hit",
                operation=operation,
                cache_path=str(path),
            )
            return json.loads(path.read_text(encoding="utf-8"))
        result = fn()
        payload = _jsonable(result)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        emit_probe_event(
            self.probe,
            "workflow.llm_cache_miss",
            operation=operation,
            cache_path=str(path),
        )
        return payload
