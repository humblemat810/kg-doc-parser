from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeVar, cast

from kogwistar.id_provider import stable_id

from .probe import WorkflowProbe, emit_probe_event

T = TypeVar("T")


def _jsonable(value: object) -> object:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dump = cast(Callable[..., object], model_dump)
        try:
            return dump(field_mode="backend", dump_format="json")
        except TypeError:
            return dump()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class WorkflowLLMCallCache:
    def __init__(self, cache_dir: str | Path, *, probe: WorkflowProbe | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.probe = probe

    def _cache_path(self, operation: str, fingerprint: Mapping[str, object]) -> Path:
        payload = json.dumps(_jsonable(fingerprint), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        cache_id = stable_id("workflow_ingest.llm_call", operation, payload)
        return self.cache_dir / f"{cache_id}.json"

    def cached_call(
        self,
        *,
        operation: str,
        fingerprint: Mapping[str, object],
        fn: Callable[[], T],
    ) -> T:
        path = self._cache_path(operation, fingerprint)
        if path.exists():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                # A cache entry is an optimization, not parser state. A
                # truncated or unreadable entry must not abort ingestion.
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            else:
                emit_probe_event(
                    self.probe,
                    "workflow.llm_cache_hit",
                    operation=operation,
                    cache_path=str(path),
                )
                return cast(T, cached)
        result = fn()
        payload = _jsonable(result)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        emit_probe_event(
            self.probe,
            "workflow.llm_cache_miss",
            operation=operation,
            cache_path=str(path),
        )
        return cast(T, payload)
