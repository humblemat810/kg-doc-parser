from __future__ import annotations

import json
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass
class _SysMonitoringState:
    tool_id: int
    code_name_allowlist: set[str]
    file_substrings: tuple[str, ...]


class WorkflowProbe:
    """Small JSONL probe sink for demo-friendly workflow event trails."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._sys_monitoring: _SysMonitoringState | None = None

    def emit(self, kind: str, /, **payload: Any) -> None:
        event = {"ts": _utc_now(), "kind": str(kind), **_jsonable(payload)}
        line = json.dumps(event, ensure_ascii=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.write("\n")

    def enable_sys_monitoring(
        self,
        *,
        code_name_allowlist: set[str] | None = None,
        file_substrings: tuple[str, ...] = ("workflow_ingest",),
    ) -> bool:
        monitoring = getattr(sys, "monitoring", None)
        if monitoring is None:
            self.emit("probe.sys_monitoring_unavailable", reason="sys.monitoring missing")
            return False
        try:
            tool_id = 5
            monitoring.use_tool_id(tool_id, "kg_doc_parser_demo_probe")
            allowlist = set(code_name_allowlist or {
                "_normalize_input",
                "_build_source_map",
                "_init_parse_session",
                "_prepare_layer_frontier",
                "_propose_layer_breakdown",
                "_review_cud_proposal",
                "_apply_cud_update",
                "_check_layer_coverage",
                "_check_layer_satisfaction",
                "_repair_layer_pointers",
                "_dedupe_and_filter_layer",
                "_commit_layer_children",
                "_enqueue_next_layer_frontier",
                "_finalize_semantic_tree",
                "_validate_tree",
                "_export_graph",
                "_persist_canonical_graph",
            })
            state = _SysMonitoringState(
                tool_id=tool_id,
                code_name_allowlist=allowlist,
                file_substrings=file_substrings,
            )
            self._sys_monitoring = state

            def _should_log(code: Any) -> bool:
                filename = str(getattr(code, "co_filename", "") or "")
                if state.code_name_allowlist and getattr(code, "co_name", "") not in state.code_name_allowlist:
                    return False
                return any(part in filename for part in state.file_substrings)

            def _on_start(code: Any, offset: int) -> None:
                if _should_log(code):
                    self.emit(
                        "probe.sys_monitoring.start",
                        code_name=getattr(code, "co_name", None),
                        filename=getattr(code, "co_filename", None),
                        offset=int(offset),
                    )

            def _on_return(code: Any, offset: int, value: Any) -> None:
                if _should_log(code):
                    self.emit(
                        "probe.sys_monitoring.return",
                        code_name=getattr(code, "co_name", None),
                        filename=getattr(code, "co_filename", None),
                        offset=int(offset),
                    )

            def _on_raise(code: Any, offset: int, exc: Any) -> None:
                if _should_log(code):
                    self.emit(
                        "probe.sys_monitoring.raise",
                        code_name=getattr(code, "co_name", None),
                        filename=getattr(code, "co_filename", None),
                        offset=int(offset),
                        error=repr(exc),
                    )

            monitoring.register_callback(tool_id, monitoring.events.PY_START, _on_start)
            monitoring.register_callback(tool_id, monitoring.events.PY_RETURN, _on_return)
            monitoring.register_callback(tool_id, monitoring.events.RAISE, _on_raise)
            monitoring.set_events(
                tool_id,
                monitoring.events.PY_START | monitoring.events.PY_RETURN | monitoring.events.RAISE,
            )
            self.emit("probe.sys_monitoring_enabled", tool_id=tool_id)
            return True
        except Exception as exc:  # noqa: BLE001
            self.emit("probe.sys_monitoring_enable_failed", error=repr(exc))
            self._sys_monitoring = None
            return False

    def disable_sys_monitoring(self) -> None:
        monitoring = getattr(sys, "monitoring", None)
        if monitoring is None or self._sys_monitoring is None:
            return
        state = self._sys_monitoring
        try:
            monitoring.set_events(state.tool_id, 0)
            monitoring.free_tool_id(state.tool_id)
            self.emit("probe.sys_monitoring_disabled", tool_id=state.tool_id)
        except Exception as exc:  # noqa: BLE001
            self.emit("probe.sys_monitoring_disable_failed", error=repr(exc))
        finally:
            self._sys_monitoring = None

    def close(self) -> None:
        self.disable_sys_monitoring()
        self.emit("probe.closed", path=str(self.path))


def emit_probe_event(probe: WorkflowProbe | None, kind: str, /, **payload: Any) -> None:
    if probe is None:
        return
    probe.emit(kind, **payload)
