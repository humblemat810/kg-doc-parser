from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Protocol, cast

from kogwistar.runtime import MappingStepResolver, StepContext
from kogwistar.runtime.models import (
    RunFailure,
    RunSuccess,
    RunSuspended,
    StateUpdate,
    StepRunResult,
    get_route_next_names,
)

from .handler_state import state_int, state_mapping, state_scalar
from .probe import emit_probe_event
from .runtime_contracts import WorkflowRuntimeDeps
from .serialization import JsonValue

_LOGGER = logging.getLogger(__name__)


class StepHandler(Protocol):
    """Execute one parser workflow step against the runtime context."""

    def __call__(self, context: StepContext, /) -> StepRunResult: ...


def success(
    next_step: str | None = None,
    *,
    state_update: list[StateUpdate] | None = None,
) -> RunSuccess:
    return RunSuccess(
        conversation_node_id=None,
        state_update=state_update or [],
        _route_next=[] if next_step is None else [next_step],
    )


def _probe_snapshot(state_view: Mapping[str, object]) -> dict[str, JsonValue]:
    snapshot: dict[str, JsonValue] = {
        "state_keys": cast(
            JsonValue,
            sorted(k for k, v in state_view.items() if v is not None),
        ),
    }
    frontier = state_view.get("layer_frontier_queue")
    if isinstance(frontier, list):
        snapshot["frontier_size"] = len(frontier)
    current_layer_context = state_mapping(state_view.get("current_layer_context"))
    for field_name, output_name in (
        ("depth", "current_depth"),
        ("retry_count", "retry_count"),
        ("split_strategy", "split_strategy"),
    ):
        value = state_scalar(current_layer_context.get(field_name))
        if value is not None:
            snapshot[output_name] = value
    parse_session = state_mapping(state_view.get("parse_session"))
    for field_name, output_name in (
        ("mode", "parse_mode"),
        ("current_depth", "session_depth"),
        ("split_strategy", "session_strategy"),
        ("strategy_switch_count", "strategy_switch_count"),
    ):
        value = state_scalar(parse_session.get(field_name))
        if value is not None:
            snapshot[output_name] = value
    return snapshot


def _progress_bar(done: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return "?" * width
    filled = min(width, max(0, round((done / total) * width)))
    return ("█" * filled) + ("░" * (width - filled))


def log_runtime_progress(*, step_name: str, state_view: Mapping[str, object]) -> None:
    try:
        current_layer_context = state_mapping(state_view.get("current_layer_context"))
        parse_session = state_mapping(state_view.get("parse_session"))
    except (TypeError, ValueError):
        # Progress reporting must not turn malformed diagnostic state into a
        # workflow failure; strict decoding remains enforced by state consumers.
        return
    if not current_layer_context or not parse_session:
        return
    depth = state_int(current_layer_context.get("depth"), 0)
    max_depth = state_int(parse_session.get("max_depth"), 10)
    layer_num = depth + 1
    retry_count = state_int(current_layer_context.get("retry_count"), 0)
    max_retries = state_int(current_layer_context.get("max_retries"), 3)
    strategy = str(current_layer_context.get("split_strategy", "excerpt_first"))
    fallback_strategy = str(
        parse_session.get("fallback_split_strategy", "boundary_first")
    )
    strategy_idx = 2 if strategy == fallback_strategy else 1
    strategy_cap = 2
    phase = step_name.replace("_", " ")
    pct = round((layer_num / max_depth) * 100) if max_depth > 0 else 0
    _LOGGER.info(
        "⏳ layer %s/%s | %3s%% | %s | iter %s/%s | strategy %s/%s %s | %s",
        layer_num,
        max_depth,
        pct,
        _progress_bar(layer_num, max_depth),
        retry_count + 1,
        max_retries,
        strategy_idx,
        strategy_cap,
        strategy,
        phase,
    )


def register_step(
    resolver: MappingStepResolver,
    *,
    step_name: str,
    runtime_deps: WorkflowRuntimeDeps,
) -> Callable[[StepHandler], StepHandler]:
    probe = runtime_deps.get("probe")

    def decorator(fn: StepHandler) -> StepHandler:
        @resolver.register(step_name)
        def _wrapped(ctx: StepContext) -> StepRunResult:
            log_runtime_progress(step_name=step_name, state_view=dict(ctx.state_view))
            emit_probe_event(
                probe,
                "workflow.step_started",
                step=step_name,
                **_probe_snapshot(dict(ctx.state_view)),
            )
            try:
                result = fn(ctx)
            except Exception as exc:
                emit_probe_event(
                    probe,
                    "workflow.step_exception",
                    step=step_name,
                    error=repr(exc),
                    **_probe_snapshot(dict(ctx.state_view)),
                )
                raise
            if isinstance(result, RunFailure):
                status = "failure"
            elif isinstance(result, RunSuspended):
                status = "suspended"
            else:
                status = "success"
            emit_probe_event(
                probe,
                "workflow.step_finished",
                step=step_name,
                status=status,
                route_next=get_route_next_names(result),
                **_probe_snapshot(dict(ctx.state_view)),
            )
            return result

        return cast(StepHandler, _wrapped)

    return decorator
