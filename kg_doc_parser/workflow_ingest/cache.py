from __future__ import annotations

import json
import os
import platform
import tempfile
import threading
from collections.abc import Callable, Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TypeVar, cast

from kogwistar.id_provider import stable_id

from .probe import WorkflowProbe, emit_probe_event
from .serialization import JsonValue, ModelDumpLike, json_safe

T = TypeVar("T")
CACHE_CONTRACT_VERSION = 2
_PACKAGE_NAME = "graph-knowledge-doc-parser"
_CACHE_WRITE_LOCK = threading.Lock()


def workflow_cache_namespace(provider_settings: object | None = None) -> dict[str, JsonValue]:
    """Return stable, non-secret identity for workflow LLM cache entries.

    The request fingerprint remains responsible for source and prompt inputs.
    This namespace prevents an entry produced by a different parser release,
    runtime, or provider configuration from being reused accidentally.
    """

    try:
        package_version = version(_PACKAGE_NAME)
    except PackageNotFoundError:
        package_version = "uninstalled"
    settings: object = provider_settings
    if isinstance(settings, ModelDumpLike):
        try:
            settings = settings.model_dump(mode="json", exclude_none=True)
        except TypeError:
            settings = settings.model_dump()
    return {
        "cache_contract_version": CACHE_CONTRACT_VERSION,
        "parser_package": _PACKAGE_NAME,
        "parser_package_version": package_version,
        "python_implementation": platform.python_implementation(),
        "python_version": list(platform.python_version_tuple()[:2]),
        "provider_settings": _jsonable(settings) if provider_settings is not None else None,
    }


def _jsonable(value: object) -> JsonValue:
    if isinstance(value, ModelDumpLike):
        try:
            dumped = value.model_dump(field_mode="backend", dump_format="json")
        except TypeError:
            dumped = value.model_dump()
        return _jsonable(dumped)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return json_safe(value)


class WorkflowLLMCallCache:
    def __init__(
        self,
        cache_dir: str | Path,
        *,
        probe: WorkflowProbe | None = None,
        namespace: Mapping[str, object] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.probe = probe
        self.namespace = dict(namespace or workflow_cache_namespace())

    def _cache_path(self, operation: str, fingerprint: Mapping[str, object]) -> Path:
        payload = json.dumps(
            _jsonable({"namespace": self.namespace, "request": fingerprint}),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
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
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        temporary_path: Path | None = None
        try:
            # Windows can reject simultaneous replacements of the same target
            # even when each writer uses a distinct temporary file. Serialize
            # writers in-process, then publish with one atomic replacement.
            with _CACHE_WRITE_LOCK:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=self.cache_dir,
                    prefix=f".{path.stem}.",
                    suffix=".tmp",
                    delete=False,
                ) as temporary:
                    temporary.write(serialized)
                    temporary.flush()
                    os.fsync(temporary.fileno())
                    temporary_path = Path(temporary.name)
                os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
        emit_probe_event(
            self.probe,
            "workflow.llm_cache_miss",
            operation=operation,
            cache_path=str(path),
        )
        return cast(T, payload)
