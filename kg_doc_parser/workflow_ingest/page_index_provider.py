"""Provider-backed PageIndex outline assignment and retry orchestration."""

from __future__ import annotations

import json
from dataclasses import asdict

from ..llm_structured_output import build_structured_output_runnable
from .page_index_candidates import (
    _assemble_page_index_blocks,
    _deterministic_block_assignments,
    _extract_candidate_blocks,
    _json_string_list,
    _salvage_page_index_assignments,
    _validate_block_assignments,
    _validate_page_index_block_structure,
)
from .page_index_contracts import (
    BlockAssignment,
    BlockAssignmentBatch,
    CandidateBlock,
    PageIndexBlockSpec,
    PageIndexDiagnostics,
    PageIndexSourceFormat,
)
from .page_index_provider_support import (
    callback_call_counts as _callback_call_counts,
)
from .page_index_provider_support import (
    notify_untracked_provider_call as _notify_untracked_provider_call,
)
from .page_index_tree import PageIndexTraceLogger
from .providers import (
    ProviderDiagnostics,
    ProviderDiagnosticsSink,
    WorkflowProviderSettings,
    build_chat_model_for_role,
    invoke_with_timeout,
)
from .serialization import JsonObject


def _object_list(value: object) -> list[object]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _page_index_assignment_debug_payload(assignments: list[BlockAssignment]) -> list[JsonObject]:
    return [
        {
            "block_id": assignment.block_id,
            "parent_id": assignment.parent_id,
            "node_type": assignment.node_type,
            "title": assignment.title,
        }
        for assignment in assignments
    ]

def _page_index_block_spec_debug_payload(block_specs: list[PageIndexBlockSpec]) -> list[JsonObject]:
    def _walk(spec: PageIndexBlockSpec, *, path: str) -> JsonObject:
        return {
            "path": path,
            "title": spec.title,
            "node_type": spec.node_type,
            "excerpt": spec.excerpt,
            "display_excerpt": spec.display_excerpt,
            "child_count": len(spec.child_nodes),
            "children": [
                _walk(child, path=f"{path}.{index + 1}") for index, child in enumerate(spec.child_nodes)
            ],
        }

    return [_walk(spec, path=str(index + 1)) for index, spec in enumerate(block_specs)]

def _page_index_retry_failure_summary(
    *,
    validation_errors: list[str],
    assignments: list[BlockAssignment] | None = None,
    parse_error: str | None = None,
) -> str:
    """Build a short repair note for a second-pass LLM prompt."""
    lines = ["First attempt failed. Repair the block hierarchy without inventing content."]
    if parse_error:
        lines.append(f"Parse error: {parse_error}")
    if validation_errors:
        lines.append("Validation errors: " + "; ".join(validation_errors[:4]))
    if assignments:
        lines.append(
            "Rejected assignment snapshot: "
            + json.dumps(
                _page_index_assignment_debug_payload(assignments[:8]),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    lines.append(
        "Fix guidance: preserve reading order, keep the same block_id values, "
        "and nest heading blocks under the correct earlier heading parent instead of flattening them."
    )
    return "\n".join(lines)

def _page_index_structure_failure_summary(
    *,
    validation_errors: list[str],
    assignments: list[BlockAssignment],
    block_specs: list[PageIndexBlockSpec],
) -> str:
    """Build a compact repair note for an assembled tree validation failure."""
    lines = ["Structure build failed. Repair the hierarchy while keeping every block grounded."]
    if validation_errors:
        lines.append("Structure validation errors: " + "; ".join(validation_errors[:4]))
    lines.append(
        "Current assignment snapshot: "
        + json.dumps(
            _page_index_assignment_debug_payload(assignments[:8]),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    lines.append(
        "Rejected tree snapshot: "
        + json.dumps(
            _page_index_block_spec_debug_payload(block_specs)[:4],
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    lines.append(
        "Fix guidance: reuse only existing block_id values, preserve reading order, "
        "nest child headings under their correct earlier heading parent, and avoid broad whole-page child blocks."
    )
    return "\n".join(lines)

def _page_index_assignment_prompt(
    *,
    page_number: int,
    source_format: PageIndexSourceFormat,
    candidates: list[CandidateBlock],
    retry_summary: str | None = None,
    retry_kind: str = "assignment",
) -> str:
    """Construct the structured-output prompt for a page-index assignment attempt."""
    prompt = (
        "You assign flat page-index blocks.\n"
        "Return JSON matching the schema exactly.\n"
        "Rules:\n"
        "- assign every block exactly once\n"
        "- use only the provided block_id values\n"
        "- parent_id must be null or reference an earlier block_id\n"
        "- preserve reading order\n"
        "- do not invent excerpts\n"
        "- do not create whole-page child blocks\n"
        "- use HEADING, PARAGRAPH, TERM, TABLE, TABLE_ROW_GROUP, or TABLE_ROW\n"
        "- provide a concise factual summary for every block; do not add facts absent from the block text\n"
        "The recursive tree is assembled deterministically after validation.\n"
        "Optional excerpt refinement is disabled by default.\n"
        f"Source format: {source_format}\n"
        f"Page number: {page_number}\n"
        f"Candidates: {json.dumps([asdict(candidate) for candidate in candidates], ensure_ascii=False, sort_keys=True)}"
    )
    if retry_summary:
        if retry_kind == "structure":
            prompt += (
                "\n\nPrevious structure build failed validation. Repair the block assignments using the compact "
                "failure summary below:\n"
                f"{retry_summary}\n"
            )
        else:
            prompt += (
                "\n\nPrevious attempt failed validation. Repair the structure using the compact failure summary below:\n"
                f"{retry_summary}\n"
            )
    return prompt

def _llm_page_outline(
    *,
    page_text: str,
    page_number: int,
    source_format: PageIndexSourceFormat,
    provider_settings: WorkflowProviderSettings,
    callbacks: list[object] | None = None,
    trace_log: PageIndexTraceLogger | None = None,
    provider_diagnostics_sink: ProviderDiagnosticsSink | None = None,
) -> tuple[list[PageIndexBlockSpec], PageIndexDiagnostics]:
    candidates = _extract_candidate_blocks(page_text, page_number=page_number, source_format=source_format)
    assignment_mode_prefix = "ollama" if provider_settings.parser.provider == "ollama" else "llm"
    if not candidates:
        return [], {
            "assignment_mode": "deterministic_fallback",
            "candidate_count": 0,
            "assignment_count": 0,
            "assignment_attempt_count": 0,
            "assignment_retry_used": False,
            "assignment_retry_succeeded": False,
            "structure_retry_used": False,
            "structure_retry_succeeded": False,
            "retry_used": False,
            "retry_succeeded": False,
            "assignment_validation_errors": ["no candidate blocks extracted"],
            "structure_validation_errors": [],
            "first_validation_errors": ["no candidate blocks extracted"],
            "retry_validation_errors": [],
            "validation_errors": ["no candidate blocks extracted"],
            "validation_warnings": [],
            "fallback_reason": "no_candidates",
        }
    if trace_log is not None:
        trace_log(f"page_index_llm_chat_build_start page_number={page_number} candidate_count={len(candidates)}")
    chat_kwargs = {"callbacks": callbacks} if callbacks is not None else {}
    chat = build_chat_model_for_role("parser", provider_settings, **chat_kwargs)
    if trace_log is not None:
        trace_log(f"page_index_llm_chat_build_done page_number={page_number}")
        trace_log(f"page_index_llm_structured_wrap_start page_number={page_number}")
    structured = build_structured_output_runnable(chat, BlockAssignmentBatch, include_raw=True)
    if trace_log is not None:
        trace_log(f"page_index_llm_structured_wrap_done page_number={page_number}")
    from langchain_core.messages import HumanMessage, SystemMessage

    def _invoke_attempt(
        *,
        retry_summary: str | None = None,
        retry_kind: str = "assignment",
        attempt_label: str,
        attempt_index: int,
    ) -> tuple[BlockAssignmentBatch | None, str | None]:
        prompt = _page_index_assignment_prompt(
            page_number=page_number,
            source_format=source_format,
            candidates=candidates,
            retry_summary=retry_summary,
            retry_kind=retry_kind,
        )
        if trace_log is not None:
            trace_log(f"page_index_llm_prompt_ready page_number={page_number} attempt={attempt_label}")
            trace_log(
                f"page_index_llm_invoke_start page_number={page_number} attempt={attempt_label} source_format={source_format}"
        )
        call_diagnostics: ProviderDiagnostics = {}
        provider_returned = False
        try:
            callback_counts = _callback_call_counts(callbacks)
            try:
                payload = invoke_with_timeout(
                    lambda: structured.invoke(
                        [
                            SystemMessage(content="You are a grounded page-index block assigner."),
                            HumanMessage(content=prompt),
                        ]
                    ),
                timeout_seconds=(
                    provider_settings.proposal_timeout_seconds
                    or provider_settings.parser.timeout_seconds
                ),
                diagnostics=call_diagnostics,
                operation="page_index_assignment",
                max_in_flight=provider_settings.parser.max_in_flight_calls,
                attempt_index=attempt_index,
                call_role="proposal",
                strategy="page_index",
                )
                provider_returned = True
            finally:
                _notify_untracked_provider_call(
                    callbacks, callback_counts, f"page-{page_number}-{attempt_label}"
                )
            if trace_log is not None:
                trace_log(f"page_index_llm_invoke_end page_number={page_number} attempt={attempt_label}")
        except Exception as exc:  # noqa: BLE001 - provider failures trigger bounded deterministic fallback.
            call_diagnostics.update(
                {
                    "success": False,
                    "failure_type": (
                        "structured_output_parse_failure"
                        if provider_returned
                        else call_diagnostics.get("failure_type", "transport/provider_exception")
                    ),
                    "error_type": type(exc).__name__,
                }
            )
            if provider_diagnostics_sink is not None:
                provider_diagnostics_sink(dict(call_diagnostics))
            return None, f"{type(exc).__name__}: {exc}"

        parsed = payload.get("parsed") if isinstance(payload, dict) else payload
        try:
            if parsed is None:
                error = payload.get("parsing_error") if isinstance(payload, dict) else None
                raise ValueError(f"parse_error: {error!r}")
            batch = parsed if isinstance(parsed, BlockAssignmentBatch) else BlockAssignmentBatch.model_validate(parsed)
        except Exception as exc:  # noqa: BLE001 - malformed structured output uses deterministic fallback.
            call_diagnostics.update(
                {
                    "success": False,
                    "failure_type": "structured_output_parse_failure",
                    "error_type": type(exc).__name__,
                }
            )
            if provider_diagnostics_sink is not None:
                provider_diagnostics_sink(dict(call_diagnostics))
            return None, f"{type(exc).__name__}: {exc}"
        call_diagnostics["success"] = True
        if provider_diagnostics_sink is not None:
            provider_diagnostics_sink(dict(call_diagnostics))
        return batch, None

    def _trace_assignment_raw(attempt_label: str, batch: BlockAssignmentBatch) -> None:
        if trace_log is None:
            return
        trace_log(
            "page_index_llm_assignment_raw "
            f"page_number={page_number} attempt={attempt_label} "
            f"payload={json.dumps(_page_index_assignment_debug_payload(batch.assignments), ensure_ascii=False, sort_keys=True)}"
        )

    def _fallback(
        *,
        assignment_attempt_count: int,
        assignment_retry_used: bool,
        first_validation_errors: list[str],
        assignment_retry_validation_errors: list[str],
        structure_validation_errors: list[str],
        validation_warnings: list[str],
        fallback_reason: str,
        retry_prompt_summary: str | None = None,
        structure_retry_prompt_summary: str | None = None,
    ) -> tuple[list[PageIndexBlockSpec], PageIndexDiagnostics]:
        fallback_assignments = _deterministic_block_assignments(candidates)
        fallback_block_specs = _assemble_page_index_blocks(
            candidates=candidates,
            assignments=fallback_assignments,
        )
        if trace_log is not None:
            trace_log(
                "page_index_llm_assignment_fallback "
                f"page_number={page_number} payload={json.dumps(_page_index_assignment_debug_payload(fallback_assignments), ensure_ascii=False, sort_keys=True)} "
                f"block_specs={json.dumps(_page_index_block_spec_debug_payload(fallback_block_specs), ensure_ascii=False, sort_keys=True)}"
            )
        validation_errors = structure_validation_errors or assignment_retry_validation_errors or first_validation_errors
        return fallback_block_specs, {
            "assignment_mode": "deterministic_fallback",
            "final_outcome": "deterministic_fallback",
            "candidate_count": len(candidates),
            "assignment_count": len(fallback_assignments),
            "assignment_attempt_count": assignment_attempt_count,
            "assignment_retry_used": assignment_retry_used,
            "assignment_retry_succeeded": False,
            "structure_retry_used": bool(structure_validation_errors),
            "structure_retry_succeeded": False,
            "retry_used": assignment_retry_used or bool(structure_validation_errors),
            "retry_succeeded": False,
            "assignment_validation_errors": _json_string_list(assignment_retry_validation_errors or first_validation_errors),
            "structure_validation_errors": _json_string_list(structure_validation_errors),
            "first_validation_errors": _json_string_list(first_validation_errors),
            "retry_validation_errors": _json_string_list(assignment_retry_validation_errors),
            "validation_errors": _json_string_list(validation_errors),
            "validation_warnings": _json_string_list(validation_warnings),
            "fallback_reason": fallback_reason,
            "retry_prompt_summary": retry_prompt_summary,
            "structure_retry_prompt_summary": structure_retry_prompt_summary,
        }

    def _try_branch_local_salvage(
        *,
        batch: BlockAssignmentBatch | None,
        attempt_label: str,
        assignment_attempt_count: int,
        validation_warnings: list[str],
        retry_prompt_summary: str | None = None,
        structure_retry_prompt_summary: str | None = None,
    ) -> tuple[list[PageIndexBlockSpec], PageIndexDiagnostics] | None:
        if batch is None:
            return None
        salvaged_assignments, salvage_diagnostics = _salvage_page_index_assignments(
            candidates=candidates,
            assignments=batch.assignments,
            page_text=page_text,
        )
        if salvaged_assignments is None:
            return None
        block_specs = _assemble_page_index_blocks(candidates=candidates, assignments=salvaged_assignments)
        if trace_log is not None:
            trace_log(
                "page_index_llm_branch_local_salvage "
                f"page_number={page_number} attempt={attempt_label} "
                f"repaired={salvage_diagnostics['repaired_block_count']} "
                f"invalid_block_ids={json.dumps(salvage_diagnostics['invalid_block_ids'], ensure_ascii=False)}"
            )
        return block_specs, {
            "assignment_mode": f"{assignment_mode_prefix}_branch_local_salvage",
            "final_outcome": "branch_local_salvage",
            "candidate_count": len(candidates),
            "assignment_count": len(salvaged_assignments),
            "assignment_attempt_count": assignment_attempt_count,
            "assignment_retry_used": assignment_attempt_count > 1,
            "assignment_retry_succeeded": False,
            "structure_retry_used": attempt_label == "structure_retry",
            "structure_retry_succeeded": False,
            "retry_used": assignment_attempt_count > 1,
            "retry_succeeded": False,
            "assignment_validation_errors": [],
            "structure_validation_errors": [],
            "first_validation_errors": [],
            "retry_validation_errors": [],
            "validation_errors": [],
            "validation_warnings": validation_warnings
            + _object_list(salvage_diagnostics["validation_warnings"]),
            "fallback_reason": None,
            "branch_local_salvage": salvage_diagnostics,
            "retry_prompt_summary": retry_prompt_summary,
            "structure_retry_prompt_summary": structure_retry_prompt_summary,
        }

    def _accept_or_retry_structure(
        *,
        batch: BlockAssignmentBatch,
        attempt_label: str,
        assignment_attempt_count: int,
        assignment_retry_used: bool,
        assignment_retry_succeeded: bool,
        first_validation_errors: list[str],
        assignment_retry_validation_errors: list[str],
        validation_warnings: list[str],
        retry_prompt_summary: str | None = None,
    ) -> tuple[list[PageIndexBlockSpec], PageIndexDiagnostics]:
        block_specs = _assemble_page_index_blocks(candidates=candidates, assignments=batch.assignments)
        structure_validation = _validate_page_index_block_structure(
            candidates=candidates,
            assignments=batch.assignments,
            block_specs=block_specs,
            page_text=page_text,
        )
        if trace_log is not None:
            trace_log(
                "page_index_structure_validation "
                f"page_number={page_number} attempt={attempt_label} errors={len(structure_validation.errors)}"
            )
        if structure_validation.valid:
            if trace_log is not None:
                trace_log(
                    "page_index_llm_assignment_accepted "
                    f"page_number={page_number} attempt={attempt_label} "
                    f"block_specs={json.dumps(_page_index_block_spec_debug_payload(block_specs), ensure_ascii=False, sort_keys=True)}"
                )
            mode = f"{assignment_mode_prefix}_flat_assignment"
            final_outcome = "first_pass_success"
            if assignment_retry_succeeded:
                mode = f"{assignment_mode_prefix}_flat_assignment_retry"
                final_outcome = "assignment_retry_success"
            return block_specs, {
                "assignment_mode": mode,
                "final_outcome": final_outcome,
                "candidate_count": len(candidates),
                "assignment_count": len(batch.assignments),
                "assignment_attempt_count": assignment_attempt_count,
                "assignment_retry_used": assignment_retry_used,
                "assignment_retry_succeeded": assignment_retry_succeeded,
                "structure_retry_used": False,
                "structure_retry_succeeded": False,
                "retry_used": assignment_retry_used,
                "retry_succeeded": assignment_retry_succeeded,
                "assignment_validation_errors": _json_string_list(assignment_retry_validation_errors or first_validation_errors),
                "structure_validation_errors": [],
                "first_validation_errors": _json_string_list(first_validation_errors),
                "retry_validation_errors": _json_string_list(assignment_retry_validation_errors),
                "validation_errors": [],
                "validation_warnings": _json_string_list(validation_warnings + list(structure_validation.warnings)),
                "fallback_reason": None,
                "retry_prompt_summary": retry_prompt_summary,
                "structure_retry_prompt_summary": None,
            }

        structure_errors = list(structure_validation.errors)
        structure_summary = _page_index_structure_failure_summary(
            validation_errors=structure_errors,
            assignments=batch.assignments,
            block_specs=block_specs,
        )
        if trace_log is not None:
            trace_log(
                "page_index_llm_structure_retry_start "
                f"page_number={page_number} failure_summary={structure_summary.splitlines()[0]}"
            )
            trace_log(f"page_index_llm_structure_retry_prompt_ready page_number={page_number}")
        next_attempt_count = assignment_attempt_count + 1
        structure_batch, structure_error = _invoke_attempt(
            retry_summary=structure_summary,
            retry_kind="structure",
            attempt_label="structure_retry",
            attempt_index=next_attempt_count,
        )
        if structure_batch is None:
            return _fallback(
                assignment_attempt_count=next_attempt_count,
                assignment_retry_used=assignment_retry_used,
                first_validation_errors=first_validation_errors,
                assignment_retry_validation_errors=assignment_retry_validation_errors,
                structure_validation_errors=structure_errors + [str(structure_error or "unknown structure retry parse error")],
                validation_warnings=validation_warnings + list(structure_validation.warnings),
                fallback_reason="structure_validation_failed",
                retry_prompt_summary=retry_prompt_summary,
                structure_retry_prompt_summary=structure_summary,
            )
        _trace_assignment_raw("structure_retry", structure_batch)
        structure_assignment_validation = _validate_block_assignments(
            candidates,
            structure_batch.assignments,
            page_text=page_text,
        )
        if not structure_assignment_validation.valid:
            if trace_log is not None:
                trace_log(
                    "page_index_llm_validation_failed "
                    f"page_number={page_number} attempt=structure_retry errors={len(structure_assignment_validation.errors)}"
                )
            salvaged = _try_branch_local_salvage(
                batch=structure_batch,
                attempt_label="structure_retry",
                assignment_attempt_count=next_attempt_count,
                validation_warnings=validation_warnings + list(structure_validation.warnings),
                retry_prompt_summary=retry_prompt_summary,
                structure_retry_prompt_summary=structure_summary,
            )
            if salvaged is not None:
                return salvaged
            return _fallback(
                assignment_attempt_count=next_attempt_count,
                assignment_retry_used=assignment_retry_used,
                first_validation_errors=first_validation_errors,
                assignment_retry_validation_errors=assignment_retry_validation_errors,
                structure_validation_errors=structure_errors + list(structure_assignment_validation.errors),
                validation_warnings=validation_warnings + list(structure_validation.warnings),
                fallback_reason="structure_validation_failed",
                retry_prompt_summary=retry_prompt_summary,
                structure_retry_prompt_summary=structure_summary,
            )
        repaired_block_specs = _assemble_page_index_blocks(candidates=candidates, assignments=structure_batch.assignments)
        repaired_structure_validation = _validate_page_index_block_structure(
            candidates=candidates,
            assignments=structure_batch.assignments,
            block_specs=repaired_block_specs,
            page_text=page_text,
        )
        if trace_log is not None:
            trace_log(
                "page_index_structure_validation "
                f"page_number={page_number} attempt=structure_retry errors={len(repaired_structure_validation.errors)}"
            )
        if not repaired_structure_validation.valid:
            salvaged = _try_branch_local_salvage(
                batch=structure_batch,
                attempt_label="structure_retry",
                assignment_attempt_count=next_attempt_count,
                validation_warnings=validation_warnings + list(repaired_structure_validation.warnings),
                retry_prompt_summary=retry_prompt_summary,
                structure_retry_prompt_summary=structure_summary,
            )
            if salvaged is not None:
                return salvaged
            return _fallback(
                assignment_attempt_count=next_attempt_count,
                assignment_retry_used=assignment_retry_used,
                first_validation_errors=first_validation_errors,
                assignment_retry_validation_errors=assignment_retry_validation_errors,
                structure_validation_errors=structure_errors + list(repaired_structure_validation.errors),
                validation_warnings=validation_warnings + list(repaired_structure_validation.warnings),
                fallback_reason="structure_validation_failed",
                retry_prompt_summary=retry_prompt_summary,
                structure_retry_prompt_summary=structure_summary,
            )
        if trace_log is not None:
            trace_log(
                "page_index_llm_assignment_accepted "
                f"page_number={page_number} attempt=structure_retry "
                f"block_specs={json.dumps(_page_index_block_spec_debug_payload(repaired_block_specs), ensure_ascii=False, sort_keys=True)}"
            )
        return repaired_block_specs, {
            "assignment_mode": f"{assignment_mode_prefix}_flat_assignment_structure_retry",
            "final_outcome": "structure_retry_success",
            "candidate_count": len(candidates),
            "assignment_count": len(structure_batch.assignments),
            "assignment_attempt_count": next_attempt_count,
            "assignment_retry_used": assignment_retry_used,
            "assignment_retry_succeeded": assignment_retry_succeeded,
            "structure_retry_used": True,
            "structure_retry_succeeded": True,
            "retry_used": True,
            "retry_succeeded": True,
            "assignment_validation_errors": _json_string_list(assignment_retry_validation_errors or first_validation_errors),
            "structure_validation_errors": _json_string_list(structure_errors),
            "first_validation_errors": _json_string_list(first_validation_errors),
            "retry_validation_errors": _json_string_list(assignment_retry_validation_errors),
            "validation_errors": [],
            "validation_warnings": _json_string_list(validation_warnings + list(repaired_structure_validation.warnings)),
            "fallback_reason": None,
            "retry_prompt_summary": retry_prompt_summary,
            "structure_retry_prompt_summary": structure_summary,
        }

    first_batch, first_error = _invoke_attempt(attempt_label="first", attempt_index=1)
    first_validation_errors: list[str] = []
    first_validation_warnings: list[str] = []
    if first_batch is not None:
        _trace_assignment_raw("first", first_batch)
        first_validation = _validate_block_assignments(candidates, first_batch.assignments, page_text=page_text)
        first_validation_errors = list(first_validation.errors)
        first_validation_warnings = list(first_validation.warnings)
        if first_validation.valid:
            return _accept_or_retry_structure(
                batch=first_batch,
                attempt_label="first",
                assignment_attempt_count=1,
                assignment_retry_used=False,
                assignment_retry_succeeded=False,
                first_validation_errors=[],
                assignment_retry_validation_errors=[],
                validation_warnings=first_validation_warnings,
            )
        if trace_log is not None:
            trace_log(
                f"page_index_llm_validation_failed page_number={page_number} attempt=first errors={len(first_validation.errors)}"
            )
    else:
        first_validation_errors = [str(first_error or "unknown parse error")]
        if trace_log is not None:
            trace_log(
                f"page_index_llm_validation_failed page_number={page_number} attempt=first errors=1"
            )

    retry_summary = _page_index_retry_failure_summary(
        validation_errors=first_validation_errors,
        assignments=first_batch.assignments if first_batch is not None else None,
        parse_error=first_error,
    )
    if trace_log is not None:
        trace_log(
            "page_index_llm_retry_start "
            f"page_number={page_number} failure_summary={retry_summary.splitlines()[0]}"
        )
        trace_log(f"page_index_llm_retry_prompt_ready page_number={page_number}")

    retry_batch, retry_error = _invoke_attempt(
        retry_summary=retry_summary,
        attempt_label="retry",
        attempt_index=2,
    )
    retry_validation_errors: list[str] = []
    retry_validation_warnings: list[str] = []
    if retry_batch is not None:
        _trace_assignment_raw("retry", retry_batch)
        retry_validation = _validate_block_assignments(candidates, retry_batch.assignments, page_text=page_text)
        retry_validation_errors = list(retry_validation.errors)
        retry_validation_warnings = list(retry_validation.warnings)
        if retry_validation.valid:
            return _accept_or_retry_structure(
                batch=retry_batch,
                attempt_label="retry",
                assignment_attempt_count=2,
                assignment_retry_used=True,
                assignment_retry_succeeded=True,
                first_validation_errors=first_validation_errors,
                assignment_retry_validation_errors=[],
                validation_warnings=retry_validation_warnings,
                retry_prompt_summary=retry_summary,
            )
        if trace_log is not None:
            trace_log(
                f"page_index_llm_validation_failed page_number={page_number} attempt=retry errors={len(retry_validation.errors)}"
            )
    else:
        retry_validation_errors = [str(retry_error or "unknown retry parse error")]

    for attempt_label, batch in (("retry", retry_batch), ("first", first_batch)):
        salvaged = _try_branch_local_salvage(
            batch=batch,
            attempt_label=attempt_label,
            assignment_attempt_count=2 if attempt_label == "retry" else 1,
            validation_warnings=retry_validation_warnings,
            retry_prompt_summary=retry_summary,
        )
        if salvaged is not None:
            return salvaged

    return _fallback(
        assignment_attempt_count=2,
        assignment_retry_used=True,
        first_validation_errors=first_validation_errors,
        assignment_retry_validation_errors=retry_validation_errors,
        structure_validation_errors=[],
        validation_warnings=retry_validation_warnings,
        fallback_reason="assignment_validation_failed"
        if first_batch is not None
        else "llm_unavailable_or_invalid",
        retry_prompt_summary=retry_summary,
    )
