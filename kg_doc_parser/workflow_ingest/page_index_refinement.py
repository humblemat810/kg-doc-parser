"""Provider-backed excerpt and hierarchical-summary refinements."""

from __future__ import annotations

import json
from collections.abc import Iterator
from copy import deepcopy

from ..llm_structured_output import build_structured_output_runnable
from .page_index_candidates import (
    _normalize_page_index_excerpt,
    _page_index_block_exceeds_excerpt_budget,
    _page_index_block_is_too_generic,
)
from .page_index_contracts import (
    ExcerptRefinementBatch,
    HierarchicalSummaryBatch,
    PageIndexBlockSpec,
    PageIndexDiagnostics,
)
from .page_index_provider_support import (
    callback_call_counts as _callback_call_counts,
)
from .page_index_provider_support import (
    increment_diagnostic as _increment_diagnostic,
)
from .page_index_provider_support import (
    notify_untracked_provider_call as _notify_untracked_provider_call,
)
from .page_index_tree import PageIndexTraceLogger
from .page_index_tree import resolve_pointer as _resolve_pointer
from .providers import (
    ProviderDiagnostics,
    ProviderDiagnosticsSink,
    WorkflowProviderSettings,
    build_chat_model_for_role,
    invoke_with_timeout,
)


def _page_index_path_id(path: tuple[int, ...]) -> str:
    return ".".join(str(index) for index in path)


def _iter_page_index_block_specs_with_paths(
    block_specs: list[PageIndexBlockSpec],
    *,
    path: tuple[int, ...] = (),
) -> Iterator[tuple[tuple[int, ...], PageIndexBlockSpec]]:
    for index, spec in enumerate(block_specs):
        current_path = path + (index,)
        yield current_path, spec
        yield from _iter_page_index_block_specs_with_paths(spec.child_nodes, path=current_path)


def _get_page_index_block_spec_at_path(
    block_specs: list[PageIndexBlockSpec],
    path: tuple[int, ...],
) -> PageIndexBlockSpec:
    current: PageIndexBlockSpec | None = None
    current_nodes = block_specs
    for index in path:
        current = current_nodes[index]
        current_nodes = current.child_nodes
    if current is None:
        raise IndexError("empty page index path")
    return current


def refine_page_index_block_excerpts(
    *,
    block_specs: list[PageIndexBlockSpec],
    page_text: str,
    page_number: int,
    unit_id: str,
    provider_settings: WorkflowProviderSettings,
    callbacks: list[object] | None = None,
    trace_log: PageIndexTraceLogger | None = None,
    provider_diagnostics_sink: ProviderDiagnosticsSink | None = None,
) -> tuple[list[PageIndexBlockSpec], PageIndexDiagnostics]:
    entries = [
        {
            "path_id": _page_index_path_id(path),
            "node_type": spec.node_type,
            "title": spec.title,
            "excerpt": spec.excerpt,
        }
        for path, spec in _iter_page_index_block_specs_with_paths(block_specs)
    ]
    diagnostics = {
        "refine_excerpts_enabled": True,
        "refine_excerpts_attempted": len(entries),
        "refine_excerpts_accepted": 0,
        "refine_excerpts_rejected": 0,
        "refine_excerpts_fallback": False,
    }
    if not entries:
        return block_specs, diagnostics

    if trace_log is not None:
        trace_log(
            f"page_index_refine_prepare page_number={page_number} unit_id={unit_id} block_count={len(entries)}"
        )
    chat_kwargs = {"callbacks": callbacks} if callbacks is not None else {}
    chat = build_chat_model_for_role("parser", provider_settings, **chat_kwargs)
    structured = build_structured_output_runnable(chat, ExcerptRefinementBatch, include_raw=True)
    import json as _json

    from langchain_core.messages import HumanMessage, SystemMessage

    prompt = (
        "You refine excerpts for a page-index tree.\n"
        "Return only shorter grounded excerpts for blocks you can improve.\n"
        "Rules:\n"
        "- keep path_id values exactly as provided\n"
        "- return an excerpt only if it is a strict exact substring of the page text\n"
        "- do not return empty excerpts\n"
        "- do not return the whole page text\n"
        "- do not change titles, node types, or parent relationships\n"
        "- prefer compact verbatim spans\n"
        f"Page number: {page_number}\n"
        f"Page text: {page_text}\n"
        f"Blocks: {_json.dumps(entries, ensure_ascii=False, sort_keys=True)}"
    )
    call_diagnostics: ProviderDiagnostics = {}
    provider_returned = False
    try:
        callback_counts = _callback_call_counts(callbacks)
        try:
            payload = invoke_with_timeout(
                lambda: structured.invoke(
                    [
                        SystemMessage(content="You are a grounded excerpt refiner."),
                        HumanMessage(content=prompt),
                    ]
                ),
                timeout_seconds=(
                    provider_settings.proposal_timeout_seconds
                    or provider_settings.parser.timeout_seconds
                ),
                diagnostics=call_diagnostics,
                operation="page_index_excerpt_refinement",
                max_in_flight=provider_settings.parser.max_in_flight_calls,
                attempt_index=1,
                call_role="refinement",
                strategy="page_index",
            )
            provider_returned = True
        finally:
            _notify_untracked_provider_call(callbacks, callback_counts, f"refine-page-{page_number}")
        parsed = payload.get("parsed") if isinstance(payload, dict) else payload
        if parsed is None:
            error = payload.get("parsing_error") if isinstance(payload, dict) else None
            raise ValueError(f"excerpt refinement failed: {error!r}")
        batch = (
            parsed
            if isinstance(parsed, ExcerptRefinementBatch)
            else ExcerptRefinementBatch.model_validate(parsed)
        )
    except Exception as exc:  # noqa: BLE001 - provider/parser failures use deterministic excerpt fallback.
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
        diagnostics["refine_excerpts_fallback"] = True
        diagnostics["refine_excerpts_rejected"] = diagnostics["refine_excerpts_attempted"]
        diagnostics["refine_excerpts_error"] = f"{type(exc).__name__}: {exc}"
        if trace_log is not None:
            trace_log(
                f"page_index_refine_fallback page_number={page_number} unit_id={unit_id} error={type(exc).__name__}"
            )
        return block_specs, diagnostics

    call_diagnostics["success"] = True
    if provider_diagnostics_sink is not None:
        provider_diagnostics_sink(dict(call_diagnostics))

    refined = deepcopy(block_specs)
    suggestions = {item.path_id: item.excerpt for item in batch.suggestions}
    for path, original_spec in _iter_page_index_block_specs_with_paths(refined):
        path_id = _page_index_path_id(path)
        suggestion = suggestions.get(path_id)
        if suggestion is None:
            continue
        current_excerpt = original_spec.excerpt
        proposed = suggestion.strip()
        if not proposed:
            diagnostics["refine_excerpts_rejected"] += 1
            continue
        if _normalize_page_index_excerpt(proposed) == _normalize_page_index_excerpt(current_excerpt):
            diagnostics["refine_excerpts_rejected"] += 1
            continue
        if _normalize_page_index_excerpt(proposed) == _normalize_page_index_excerpt(page_text):
            diagnostics["refine_excerpts_rejected"] += 1
            continue
        normalized_current = _normalize_page_index_excerpt(current_excerpt)
        normalized_proposed = _normalize_page_index_excerpt(proposed)
        if not original_spec.child_nodes and normalized_proposed in normalized_current:
            # A leaf owns its complete terminal source interval.  Accepting a
            # strict substring here would create an uncovered source gap even
            # though the proposed text is a valid substring of the old excerpt.
            diagnostics["refine_excerpts_rejected"] += 1
            continue
        try:
            _resolve_pointer(unit_id=unit_id, page_text=page_text, excerpt=proposed)
        except Exception:  # noqa: BLE001 - an invalid grounded pointer rejects only this candidate.
            diagnostics["refine_excerpts_rejected"] += 1
            continue

        ancestor_excerpts: tuple[str, ...] = ()
        current_nodes = refined
        for index in path[:-1]:
            ancestor = current_nodes[index]
            normalized = _normalize_page_index_excerpt(ancestor.excerpt)
            if normalized:
                ancestor_excerpts += (normalized,)
            current_nodes = ancestor.child_nodes
        parent_nodes = refined if len(path) == 1 else _get_page_index_block_spec_at_path(refined, path[:-1]).child_nodes
        sibling_norms = {
            _normalize_page_index_excerpt(sibling.excerpt)
            for sibling_index, sibling in enumerate(parent_nodes)
            if sibling_index != path[-1] and _normalize_page_index_excerpt(sibling.excerpt)
        }
        if normalized_proposed in ancestor_excerpts:
            diagnostics["refine_excerpts_rejected"] += 1
            continue
        if normalized_proposed in sibling_norms:
            normalized_source = _normalize_page_index_excerpt(page_text)
            if normalized_source.count(normalized_proposed) < 2:
                diagnostics["refine_excerpts_rejected"] += 1
                continue

        candidate_spec = deepcopy(original_spec)
        candidate_spec.excerpt = proposed
        if _page_index_block_exceeds_excerpt_budget(candidate_spec, page_text):
            diagnostics["refine_excerpts_rejected"] += 1
            continue
        if _page_index_block_is_too_generic(candidate_spec, ancestor_excerpts=ancestor_excerpts):
            diagnostics["refine_excerpts_rejected"] += 1
            continue

        original_spec.excerpt = proposed
        diagnostics["refine_excerpts_accepted"] += 1

    if trace_log is not None:
        trace_log(
            "page_index_refine_complete "
            f"page_number={page_number} accepted={diagnostics['refine_excerpts_accepted']} "
            f"rejected={diagnostics['refine_excerpts_rejected']}"
        )
    return refined, diagnostics
def refine_page_index_summaries_hierarchically(
    *,
    block_specs: list[PageIndexBlockSpec],
    page_number: int,
    provider_settings: WorkflowProviderSettings,
    callbacks: list[object] | None = None,
    trace_log: PageIndexTraceLogger | None = None,
    provider_diagnostics_sink: ProviderDiagnosticsSink | None = None,
) -> tuple[list[PageIndexBlockSpec], PageIndexDiagnostics]:
    """Optionally regenerate summaries one tree depth at a time.

    The initial assignment remains authoritative for structure and grounding.
    This pass only improves summaries, and each depth receives bounded child
    excerpts plus the already accepted summary of its direct parent.
    """

    paths = list(_iter_page_index_block_specs_with_paths(block_specs))
    diagnostics: PageIndexDiagnostics = {
        "hierarchical_summary_enabled": True,
        "hierarchical_summary_attempted": len(paths),
        "hierarchical_summary_accepted": 0,
        "hierarchical_summary_rejected": 0,
        "hierarchical_summary_fallback": False,
        "hierarchical_summary_call_count": 0,
    }
    if not paths:
        return block_specs, diagnostics

    chat_kwargs = {"callbacks": callbacks} if callbacks is not None else {}
    try:
        chat = build_chat_model_for_role("parser", provider_settings, **chat_kwargs)
        structured = build_structured_output_runnable(chat, HierarchicalSummaryBatch, include_raw=True)
    except Exception as exc:  # noqa: BLE001 - optional summary pass preserves the assignment output.
        diagnostics["hierarchical_summary_fallback"] = True
        diagnostics["hierarchical_summary_rejected"] = len(paths)
        diagnostics["hierarchical_summary_error"] = f"{type(exc).__name__}: {exc}"
        if provider_diagnostics_sink is not None:
            provider_diagnostics_sink(
                {
                    "operation": "page_index_hierarchical_summary",
                    "call_role": "summary",
                    "strategy": "page_index",
                    "attempt_index": 0,
                    "success": False,
                    "failure_type": "transport/provider_exception",
                    "error_type": type(exc).__name__,
                }
            )
        return block_specs, diagnostics
    from langchain_core.messages import HumanMessage, SystemMessage

    refined = deepcopy(block_specs)
    summary_by_path = {
        path: spec.summary.strip()
        for path, spec in _iter_page_index_block_specs_with_paths(refined)
    }
    max_entries_per_call = 32
    max_excerpt_chars = 1600
    max_parent_summary_chars = 800
    max_title_chars = 240
    paths_by_depth: dict[int, list[tuple[tuple[int, ...], PageIndexBlockSpec]]] = {}
    for path, spec in paths:
        paths_by_depth.setdefault(len(path), []).append((path, spec))

    call_index = 0
    for depth in sorted(paths_by_depth):
        depth_entries = paths_by_depth[depth]
        for offset in range(0, len(depth_entries), max_entries_per_call):
            chunk = depth_entries[offset : offset + max_entries_per_call]
            entries = [
                {
                    "path_id": _page_index_path_id(path),
                    "title": spec.title[:max_title_chars],
                    "node_type": spec.node_type,
                    "excerpt": spec.excerpt[:max_excerpt_chars],
                    "parent_path_id": _page_index_path_id(path[:-1]) if path[:-1] else None,
                    "parent_summary": summary_by_path.get(path[:-1], "")[:max_parent_summary_chars],
                }
                for path, spec in chunk
            ]
            call_index += 1
            diagnostics["hierarchical_summary_call_count"] = call_index
            prompt = (
                "Generate concise factual summaries for these page-index blocks.\n"
                "Return only the structured schema.\n"
                "Rules:\n"
                "- return only supplied path_id values\n"
                "- use only facts supported by each block excerpt\n"
                "- use the direct parent summary to avoid repetition and keep terminology coherent\n"
                "- do not copy the parent summary verbatim when the child adds no new fact; return an empty summary\n"
                "- do not alter structure, titles, or source grounding\n"
                f"Page number: {page_number}\n"
                f"Tree depth: {depth}\n"
                f"Blocks: {json.dumps(entries, ensure_ascii=False, sort_keys=True)}"
            )
            call_diagnostics: ProviderDiagnostics = {}
            provider_returned = False
            try:
                callback_counts = _callback_call_counts(callbacks)
                try:
                    payload = invoke_with_timeout(
                        lambda prompt_text=prompt: structured.invoke(
                            [
                                SystemMessage(content="You are a conservative hierarchical summary editor."),
                                HumanMessage(content=prompt_text),
                            ]
                        ),
                        timeout_seconds=(
                            provider_settings.proposal_timeout_seconds
                            or provider_settings.parser.timeout_seconds
                        ),
                        diagnostics=call_diagnostics,
                        operation="page_index_hierarchical_summary",
                        max_in_flight=provider_settings.parser.max_in_flight_calls,
                        attempt_index=call_index,
                        call_role="summary",
                        strategy="page_index",
                    )
                    provider_returned = True
                finally:
                    _notify_untracked_provider_call(callbacks, callback_counts, f"summary-page-{page_number}-{call_index}")
                parsed = payload.get("parsed") if isinstance(payload, dict) else payload
                if parsed is None:
                    error = payload.get("parsing_error") if isinstance(payload, dict) else None
                    raise ValueError(f"summary parse error: {error!r}")
                batch = (
                    parsed
                    if isinstance(parsed, HierarchicalSummaryBatch)
                    else HierarchicalSummaryBatch.model_validate(parsed)
                )
            except Exception as exc:  # noqa: BLE001 - summaries fall back to assignment output.
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
                diagnostics["hierarchical_summary_fallback"] = True
                _increment_diagnostic(diagnostics, "hierarchical_summary_rejected", len(chunk))
                if trace_log is not None:
                    trace_log(
                        f"page_index_hierarchical_summary_fallback page_number={page_number} "
                        f"depth={depth} error={type(exc).__name__}"
                    )
                continue

            call_diagnostics["success"] = True
            if provider_diagnostics_sink is not None:
                provider_diagnostics_sink(dict(call_diagnostics))
            expected_paths = {_page_index_path_id(path): path for path, _spec in chunk}
            for assignment in batch.assignments:
                path = expected_paths.get(assignment.path_id)
                if path is None or not assignment.summary.strip():
                    _increment_diagnostic(diagnostics, "hierarchical_summary_rejected")
                    continue
                summary = assignment.summary.strip()
                current_nodes = refined
                for index in path[:-1]:
                    current_nodes = current_nodes[index].child_nodes
                current_nodes[path[-1]].summary = summary
                summary_by_path[path] = summary
                _increment_diagnostic(diagnostics, "hierarchical_summary_accepted")

    return refined, diagnostics



