from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from kogwistar.runtime import MappingStepResolver, StepContext
from kogwistar.runtime.models import (
    StepRunResult,
)

from .adapters import (
    build_authoritative_source_map,
    build_parser_input_dict,
    build_parser_source_map,
    select_primary_collection,
)
from .handler_exports import build_export_bundle
from .handler_layerwise import register_layerwise_parser_steps
from .handler_postparse import register_postparse_steps
from .handler_state import (
    correct_parser_pointer as _correct_parser_pointer,  # noqa: F401 - compatibility export
)
from .handler_state import (
    coverage_map as _coverage_map,  # noqa: F401 - compatibility export
)
from .handler_state import (
    parser_payload as _parser_payload,
)
from .handler_state import (
    parser_source_map as _parser_source_map,
)
from .handler_state import (
    state_float as _state_float,  # noqa: F401 - compatibility export
)
from .handler_state import state_int as _state_int  # noqa: F401 - compatibility export
from .handler_state import (
    state_mapping as _state_mapping,  # noqa: F401 - compatibility export
)
from .handler_state import (
    state_object_list as _state_object_list,  # noqa: F401 - compatibility export
)
from .handler_state import (
    state_object_map as _state_object_map,  # noqa: F401 - compatibility export
)
from .handler_state import (
    state_string_list as _state_string_list,  # noqa: F401 - compatibility export
)
from .handler_state import (
    strategy_attempt as _strategy_attempt,  # noqa: F401 - compatibility export
)
from .handler_support import (
    log_runtime_progress,
)
from .handler_support import (
    register_step as _register_step,
)
from .handler_support import (
    success as _success,
)
from .model_validation import validate_backend_payload
from .models import (
    WorkflowIngestInput,
)
from .parser_core import (
    default_parse_semantic_fn,
    initialize_parse_session,
)
from .runtime_contracts import WorkflowRuntimeDeps

_build_export_bundle = build_export_bundle
_log_runtime_progress = log_runtime_progress


def _validate_public_runtime_policy_conflicts(
    normalized_input: WorkflowIngestInput,
    runtime_deps: WorkflowRuntimeDeps,
) -> None:
    """Reject contradictory request and runtime strategy policies."""
    requested = normalized_input.parse_strategy
    configured = runtime_deps.get("split_strategy")
    if requested not in {"layer_excerpt", "layer_boundary"} or configured is None:
        return
    expected = "excerpt_first" if requested == "layer_excerpt" else "boundary_first"
    if configured != expected:
        raise ValueError(
            "conflicting parse strategy policies: request specifies "
            f"{requested!r}, runtime specifies {configured!r}"
        )


def register_base_ingest_steps(
    resolver: MappingStepResolver,
    *,
    runtime_deps: WorkflowRuntimeDeps,
) -> None:
    @_register_step(resolver, step_name="start", runtime_deps=runtime_deps)
    def _start(ctx: StepContext) -> StepRunResult:
        return _success("normalize_input")

    @_register_step(resolver, step_name="normalize_input", runtime_deps=runtime_deps)
    def _normalize_input(ctx: StepContext) -> StepRunResult:
        payload = validate_backend_payload(
            WorkflowIngestInput, ctx.state_view["input"]
        ).model_dump(
            field_mode="backend",
            dump_format="json",
        )
        with ctx.state_write as st:
            st["normalized_input"] = payload
        return _success("build_source_map")

    @_register_step(resolver, step_name="build_source_map", runtime_deps=runtime_deps)
    def _build_source_map(ctx: StepContext) -> StepRunResult:
        normalized = validate_backend_payload(
            WorkflowIngestInput, ctx.state_view["normalized_input"]
        )
        authoritative_source_map = build_authoritative_source_map(normalized)
        collection = select_primary_collection(normalized)
        parser_input_dict = build_parser_input_dict(collection)
        parser_source_map = build_parser_source_map(authoritative_source_map)
        with ctx.state_write as st:
            st["authoritative_source_map"] = {
                k: v.model_dump(field_mode="backend", dump_format="json")
                for k, v in authoritative_source_map.items()
            }
            st["parser_input_dict"] = parser_input_dict
            st["parser_source_map"] = parser_source_map
        return _success("init_parse_session")

    @_register_step(resolver, step_name="init_parse_session", runtime_deps=runtime_deps)
    def _init_parse_session(ctx: StepContext) -> StepRunResult:
        normalized = validate_backend_payload(
            WorkflowIngestInput, ctx.state_view["normalized_input"]
        )
        _validate_public_runtime_policy_conflicts(normalized, runtime_deps)
        collection = select_primary_collection(normalized)
        parse_semantic_fn = runtime_deps.get(
            "parse_semantic_fn", default_parse_semantic_fn
        )
        propose_layer_fn = runtime_deps.get("propose_layer_fn")
        session, frontier, root = initialize_parse_session(
            collection=collection,
            parser_input_dict=_parser_payload(ctx.state_view["parser_input_dict"]),
            parser_source_map=_parser_source_map(ctx.state_view["parser_source_map"]),
            max_depth=int(runtime_deps.get("max_depth", 10)),
            allow_review=bool(runtime_deps.get("allow_review", True)),
            split_strategy=runtime_deps.get("split_strategy", "excerpt_first"),
            fallback_split_strategy=runtime_deps.get(
                "fallback_split_strategy", "boundary_first"
            ),
            parse_semantic_fn=parse_semantic_fn if propose_layer_fn is None else None,
        )
        with ctx.state_write as st:
            st["parse_session"] = session.model_dump(
                field_mode="backend", dump_format="json"
            )
            st["layer_frontier_queue"] = [
                item.model_dump(field_mode="backend", dump_format="json")
                for item in frontier
            ]
            st["semantic_tree"] = root.model_dump()
        return _success("check_frontier_remaining")


def build_ingest_step_resolver(
    *,
    deps: Mapping[str, object] | None = None,
) -> MappingStepResolver:
    runtime_deps = cast(WorkflowRuntimeDeps, dict(deps or {}))
    resolver = MappingStepResolver()
    resolver.set_state_schema(
        {
            "normalized_input": "u",
            "authoritative_source_map": "u",
            "parser_input_dict": "u",
            "parser_source_map": "u",
            "parse_session": "u",
            "layer_frontier_queue": "u",
            "current_layer_context": "u",
            "current_layer_result": "u",
            "current_layer_review": "u",
            "semantic_tree": "u",
            "corrected_pointer_count": "u",
            "validation_report": "u",
            "export_bundle": "u",
            "canonical_write_result": "u",
            "workflow_errors": "a",
            "strategy_selection_error": "u",
            "strategy_attempt_counts": "u",
            "strategy_execution_history": "a",
        }
    )
    register_base_ingest_steps(resolver, runtime_deps=runtime_deps)
    register_layerwise_parser_steps(resolver, runtime_deps=runtime_deps)
    register_postparse_steps(resolver, runtime_deps=runtime_deps)
    return resolver
