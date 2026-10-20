from __future__ import annotations

from kogwistar.runtime import MappingStepResolver, StepContext
from kogwistar.runtime.models import RunFailure, StepRunResult

from .handler_exports import build_export_bundle
from .handler_state import (
    coverage_map,
    state_float,
    state_int,
    state_object_map,
    state_string_list,
)
from .handler_support import register_step, success
from .model_validation import validate_backend_payload
from .models import (
    CanonicalGraphWriteResult,
    ValidationReport,
    WorkflowExportBundle,
)
from .probe import emit_probe_event
from .runtime_contracts import WorkflowRuntimeDeps
from .semantics import (
    SemanticNode,
    classify_terminal_coverage_status,
    compute_pointer_coverage,
    compute_terminal_content_coverage,
)


def register_postparse_steps(
    resolver: MappingStepResolver,
    *,
    runtime_deps: WorkflowRuntimeDeps,
) -> None:
    @register_step(resolver, step_name="validate_tree", runtime_deps=runtime_deps)
    def _validate_tree(ctx: StepContext) -> StepRunResult:
        tree = SemanticNode.model_validate(ctx.state_view["semantic_tree"])
        authoritative_source_map = state_object_map(
            ctx.state_view["authoritative_source_map"]
        )
        text_only_map = {
            unit_id: {"text": rec["parser_text"], "id": unit_id}
            for unit_id, rec in authoritative_source_map.items()
            if rec.get("participates_in_semantic_text", True)
        }
        coverage = compute_pointer_coverage(tree, text_only_map)
        terminal_coverage = compute_terminal_content_coverage(tree, text_only_map)
        report = ValidationReport(
            overall_text_coverage=state_float(coverage.get("overall")),
            per_cluster_coverage=coverage_map(coverage.get("per_cluster")),
            terminal_coverage=terminal_coverage,
            terminal_coverage_status=classify_terminal_coverage_status(
                tree, terminal_coverage
            ),
            coverage_basis=str(terminal_coverage.get("coverage_basis", "legacy_union")),
            corrected_pointer_count=state_int(
                ctx.state_view.get("corrected_pointer_count")
            ),
            validation_notes=[],
        )
        bundle = build_export_bundle(ctx=ctx, runtime_deps=runtime_deps)
        threshold = float(runtime_deps.get("coverage_threshold", 1.0))
        if not terminal_coverage.get("valid", False):
            overall = state_float(terminal_coverage.get("overall"))
            ownership_message = (
                "terminal content ownership is incomplete or invalid: "
                f"overall={overall:.3f}"
            )
            # Keep the threshold signal when incomplete ownership also lowers
            # coverage, while still naming the stricter ownership invariant.
            error_message = (
                f"text coverage below threshold: {report.overall_text_coverage:.3f} < {threshold:.3f}; "
                f"{ownership_message}"
                if overall < threshold
                else ownership_message
            )
            with ctx.state_write as st:
                st["validation_report"] = report.model_dump(
                    field_mode="backend", dump_format="json"
                )
                st["export_bundle"] = bundle.model_dump(
                    field_mode="backend", dump_format="json"
                )
            return RunFailure(
                conversation_node_id=None,
                state_update=[],
                update={
                    "workflow_errors": [error_message],
                    "validation_report": report.model_dump(
                        field_mode="backend", dump_format="json"
                    ),
                },
                errors=[error_message],
            )
        if state_float(terminal_coverage.get("overall")) < threshold:
            error_message = f"text coverage below threshold: {report.overall_text_coverage:.3f} < {threshold:.3f}"
            with ctx.state_write as st:
                st["validation_report"] = report.model_dump(
                    field_mode="backend", dump_format="json"
                )
                st["export_bundle"] = bundle.model_dump(
                    field_mode="backend", dump_format="json"
                )
            return RunFailure(
                conversation_node_id=None,
                state_update=[],
                update={
                    "workflow_errors": [error_message],
                    "validation_report": report.model_dump(
                        field_mode="backend", dump_format="json"
                    ),
                },
                errors=[error_message],
            )
        with ctx.state_write as st:
            st["validation_report"] = report.model_dump(
                field_mode="backend", dump_format="json"
            )
        return success("export_graph")

    @register_step(resolver, step_name="export_graph", runtime_deps=runtime_deps)
    def _export_graph(ctx: StepContext) -> StepRunResult:
        bundle = build_export_bundle(ctx=ctx, runtime_deps=runtime_deps)
        with ctx.state_write as st:
            st["export_bundle"] = bundle.model_dump(
                field_mode="backend", dump_format="json"
            )
        return success("persist_canonical_graph")

    @register_step(
        resolver, step_name="persist_canonical_graph", runtime_deps=runtime_deps
    )
    def _persist_canonical_graph(ctx: StepContext) -> StepRunResult:
        bundle = validate_backend_payload(
            WorkflowExportBundle, ctx.state_view["export_bundle"]
        )
        persistence_client = runtime_deps.get("graph_persistence_client")
        if persistence_client is None:
            return RunFailure(
                conversation_node_id=None,
                state_update=[],
                errors=["no graph persistence client configured"],
            )
        try:
            write_result = persistence_client.persist_graph_payload(bundle)
        except Exception as exc:  # noqa: BLE001 - persistence failures become workflow failures
            # A failed remote write must not silently become a local-debug write.
            with ctx.state_write as st:
                st["export_bundle"] = bundle.model_dump(
                    field_mode="backend",
                    dump_format="json",
                )
            return RunFailure(
                conversation_node_id=None,
                state_update=[],
                errors=[f"canonical graph persistence failed: {exc}"],
            )
        if isinstance(write_result, dict):
            write_result = CanonicalGraphWriteResult.model_validate(write_result)
        emit_probe_event(
            runtime_deps.get("probe"),
            "workflow.persistence_result",
            persistence_mode=write_result.persistence_mode,
            kg_authority=write_result.kg_authority,
            canonical_write_confirmed=write_result.canonical_write_confirmed,
            nodes_written=write_result.nodes_written,
            edges_written=write_result.edges_written,
            transport=write_result.transport,
        )
        updated_bundle = bundle.model_copy(
            update={
                "persistence_mode": write_result.persistence_mode,
                "kg_authority": write_result.kg_authority,
                "canonical_write_confirmed": write_result.canonical_write_confirmed,
                "server_parser_used": write_result.server_parser_used,
                "canonical_write_result": write_result,
                "persisted_to_knowledge_engine": write_result.persistence_mode
                == "local_debug"
                and (write_result.nodes_written > 0 or write_result.edges_written > 0),
            }
        )
        with ctx.state_write as st:
            st["canonical_write_result"] = write_result.model_dump(
                field_mode="backend",
                dump_format="json",
            )
            st["export_bundle"] = updated_bundle.model_dump(
                field_mode="backend",
                dump_format="json",
            )
        return success("end")

    @register_step(resolver, step_name="end", runtime_deps=runtime_deps)
    def _end(ctx: StepContext) -> StepRunResult:
        return success(None)

    @register_step(resolver, step_name="parse_failure", runtime_deps=runtime_deps)
    def _parse_failure(ctx: StepContext) -> StepRunResult:
        errors = state_string_list(ctx.state_view.get("workflow_errors"))
        if not errors:
            errors = ["layer parsing failed after all strategies were exhausted"]
        return RunFailure(
            conversation_node_id=None,
            state_update=[],
            update={"workflow_errors": errors},
            errors=errors,
        )
