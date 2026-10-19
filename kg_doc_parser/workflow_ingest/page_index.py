"""Reusable page-index document parsing for text and Markdown inputs.

The pipeline keeps a fast heuristic mode for deterministic structure extraction
and provider-backed modes that reuse the existing parser provider boundary.
All modes normalize raw content into page-aware source units and return a
semantic tree with hydrated spans. Provider-backed modes use candidate
extraction plus flat block assignment, with optional excerpt refinement
disabled by default.

Example CLI
-----------
Heuristic mode:

    .venv\\Scripts\\python.exe -m pytest \
tests/test_workflow_ingest_page_index_pipeline.py::test_page_index_heuristic_parses_text_and_markdown[text] -q

    .venv\\Scripts\\python.exe -m pytest \
tests/test_workflow_ingest_page_index_pipeline.py::test_page_index_heuristic_parses_text_and_markdown[markdown] -q

Ollama mode with a local Qwen parser model:

    set KG_DOC_PARSER_PROVIDER=ollama
    set KG_DOC_PARSER_MODEL=qwen3:4b-instruct-2507-q8_0
    set KG_DOC_PARSER_BASE_URL=http://127.0.0.1:11434
    .venv\\Scripts\\python.exe -m pytest \
tests/test_workflow_ingest_page_index_pipeline.py::test_page_index_ollama_smoke_parses_text_and_markdown[text] -q

    set KG_DOC_PARSER_PROVIDER=ollama
    set KG_DOC_PARSER_MODEL=qwen3:4b-instruct-2507-q8_0
    set KG_DOC_PARSER_BASE_URL=http://127.0.0.1:11434
    .venv\\Scripts\\python.exe -m pytest \
tests/test_workflow_ingest_page_index_pipeline.py::test_page_index_ollama_smoke_parses_text_and_markdown[markdown] -q
"""

from __future__ import annotations

from typing import cast

from kogwistar.id_provider import stable_id

from .adapters import (
    build_authoritative_source_map,
    build_parser_input_dict,
    build_parser_source_map,
)
from .models import (
    LayerChildCandidate,
)
from .page_index_candidates import (
    _assemble_page_index_blocks,
    _deterministic_block_assignments,
    _extract_candidate_blocks,
    _validate_block_assignments,
    build_page_index_workflow_input,
)
from .page_index_contracts import (
    BlockAssignment,
    BlockAssignmentBatch,  # noqa: F401 - preserve public package exports
    CandidateBlock,  # noqa: F401 - preserve public package exports
    ExcerptRefinementSuggestion,  # noqa: F401 - preserve module import compatibility
    HierarchicalSummaryAssignment,  # noqa: F401 - preserve module import compatibility
    HierarchicalSummaryBatch,  # noqa: F401 - preserve module import compatibility
    PageIndexBlockSpec,
    PageIndexDiagnostics,
    PageIndexMode,
    PageIndexParseResult,
    PageIndexSourceFormat,
    PageIndexSourceRole,  # noqa: F401 - preserve module import compatibility
    PageIndexValidationResult,
)
from .page_index_provider import _llm_page_outline
from .page_index_provider_support import (  # noqa: F401 - compatibility re-export
    increment_diagnostic as _increment_diagnostic,
)
from .page_index_refinement import (
    refine_page_index_block_excerpts as _refine_page_index_block_excerpts,
)
from .page_index_refinement import (
    refine_page_index_summaries_hierarchically as _refine_page_index_summaries_hierarchically,
)
from .page_index_tree import (
    PageIndexTraceLogger,
    _find_page_unit,  # noqa: F401 - compatibility re-export
    _make_semantic_node,
    _page_index_find_best_fuzzy_span,  # noqa: F401 - compatibility re-export
    _page_index_normalize_text,  # noqa: F401 - compatibility re-export
    _stable_part,
    materialize_block_tree,
    normalize_page_index_unary_nodes,
    rebase_page_index_subtree,
)
from .parser_core import ParserSourceMap, finalize_semantic_tree
from .providers import (
    ProviderDiagnosticsSink,
    WorkflowProviderSettings,
)
from .semantics import (
    HydratedTextPointer,
    SemanticNode,
    compute_pointer_coverage,
    compute_terminal_content_coverage,
    validate_semantic_tree_integrity,
)
from .serialization import JsonValue

_PAGE_INDEX_PROVIDER_MODES = {"fake", "ollama", "openai", "azure", "gemini", "vertex", "codex"}


def _object_list(value: object) -> list[object]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _object_int(value: object, default: int) -> int:
    if type(value) is int:
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default












def _build_page_outline_from_candidates(
    *,
    page_text: str,
    page_number: int,
    source_format: PageIndexSourceFormat,
    assignment_mode: str,
    assignments: list[BlockAssignment],
    validation: PageIndexValidationResult,
) -> tuple[list[PageIndexBlockSpec], PageIndexDiagnostics]:
    candidates = _extract_candidate_blocks(page_text, page_number=page_number, source_format=source_format)
    if validation.valid:
        block_specs = _assemble_page_index_blocks(candidates=candidates, assignments=assignments)
    else:
        block_specs = _assemble_page_index_blocks(
            candidates=candidates,
            assignments=_deterministic_block_assignments(candidates),
        )
        assignment_mode = "deterministic_fallback"
    diagnostics = {
        "assignment_mode": assignment_mode,
        "candidate_count": len(candidates),
        "assignment_count": len(assignments),
        "validation_errors": list(validation.errors),
        "validation_warnings": list(validation.warnings),
        "fallback_reason": validation.fallback_reason if not validation.valid else None,
    }
    return block_specs, diagnostics


def _heuristic_page_outline(
    page_text: str,
    *,
    page_number: int,
    source_format: PageIndexSourceFormat,
) -> tuple[list[PageIndexBlockSpec], PageIndexDiagnostics]:
    candidates = _extract_candidate_blocks(page_text, page_number=page_number, source_format=source_format)
    assignments = _deterministic_block_assignments(candidates)
    validation = _validate_block_assignments(candidates, assignments, page_text=page_text)
    block_specs = _assemble_page_index_blocks(candidates=candidates, assignments=assignments)
    diagnostics = {
        "assignment_mode": "heuristic_deterministic",
        "candidate_count": len(candidates),
        "assignment_count": len(assignments),
        "validation_errors": list(validation.errors),
        "validation_warnings": list(validation.warnings),
        "fallback_reason": validation.fallback_reason,
    }
    return block_specs, diagnostics




def _materialize_block_tree(
    *,
    block_specs: list[PageIndexBlockSpec],
    page_text: str,
    unit_id: str,
    parent_id: str,
    level_from_root: int,
    start_at: int = 0,
    repair_stats: dict[str, int] | None = None,
    trace_log: PageIndexTraceLogger | None = None,
    summary_enabled: bool = True,
) -> tuple[list[SemanticNode], int]:
    return materialize_block_tree(
        block_specs=block_specs,
        page_text=page_text,
        unit_id=unit_id,
        parent_id=parent_id,
        level_from_root=level_from_root,
        start_at=start_at,
        repair_stats=repair_stats,
        trace_log=trace_log,
        summary_enabled=summary_enabled,
    )


def _rebase_page_index_subtree(
    children: list[SemanticNode],
    *,
    parent_id: str | None,
    level_from_root: int,
) -> list[SemanticNode]:
    return rebase_page_index_subtree(
        children, parent_id=parent_id, level_from_root=level_from_root
    )


def _normalize_page_index_unary_nodes(node: SemanticNode) -> SemanticNode:
    """Normalize PageIndex unary shapes without losing owned source or descendants.

    Structural wrappers can be collapsed when their content is carried by
    descendants. A semantic parent with independent source content retains
    its real child and gets a source-owning leaf. An unresolved branch fails
    rather than being silently replaced by an atomic leaf.
    """

    return normalize_page_index_unary_nodes(node)


def parse_page_index_layer(
    *,
    parent_id: str,
    parent_title: str,
    parent_pointers: list[HydratedTextPointer],
    parser_source_map: ParserSourceMap,
    source_format: PageIndexSourceFormat = "text",
    summary_enabled: bool = True,
) -> list[LayerChildCandidate]:
    """Refine each parent interval by exactly one deterministic page-index layer.

    This deliberately does not recurse through the materialized subtree.  The
    workflow owns recursion: returned candidates are committed as the current
    layer, and expandable children are sent through strategy selection again.
    """

    def _normalise(text: str) -> str:
        return " ".join(text.split()).strip().casefold()

    def _shift_pointer(pointer: HydratedTextPointer, offset: int) -> HydratedTextPointer:
        return pointer.model_copy(
            update={
                "start_char": pointer.start_char + offset,
                "end_char": pointer.end_char + offset if pointer.end_char >= 0 else pointer.end_char,
            }
        )

    candidates: list[LayerChildCandidate] = []
    for parent_pointer in parent_pointers:
        source = parser_source_map.get(parent_pointer.source_cluster_id, {})
        raw_source_text = source.get("text", "") if isinstance(source, dict) else ""
        source_text = raw_source_text if isinstance(raw_source_text, str) else ""
        if not source_text:
            continue
        source_end = len(source_text) if parent_pointer.end_char == -1 else parent_pointer.end_char + 1
        start = max(0, parent_pointer.start_char)
        end = min(len(source_text), max(start, source_end))
        layer_text = source_text[start:end]
        if not layer_text.strip():
            continue

        specs, _diagnostics = _heuristic_page_outline(
            layer_text,
            page_number=1,
            source_format=source_format,
        )
        if not specs:
            continue

        # A section aggregate normally starts with its own heading.  That
        # heading identifies the parent, so only its direct children refine
        # this layer; do not create a duplicate parent candidate.
        first = specs[0]
        if (
            first.source_role == "heading"
            and _normalise(first.title) == _normalise(parent_title)
        ):
            specs = list(first.child_nodes)
        if not specs:
            continue

        nodes, _ = _materialize_block_tree(
            block_specs=specs,
            page_text=layer_text,
            unit_id=parent_pointer.source_cluster_id,
            parent_id=parent_id,
            level_from_root=0,
            summary_enabled=summary_enabled,
        )
        for node in nodes:
            local_pointers = node.total_content_pointers or node.aggregate_content_pointers
            pointers = [_shift_pointer(pointer, start) for pointer in local_pointers]
            if not pointers:
                continue
            candidate_id = str(
                stable_id(
                    "workflow_ingest.page_index_layer_child",
                    _stable_part(parent_id),
                    _stable_part(node.node_type),
                    _stable_part([pointer.model_dump(mode="json") for pointer in pointers]),
                )
            )
            materialized_children: list[LayerChildCandidate] = []
            for child in node.child_nodes:
                child_local_pointers = child.total_content_pointers or child.aggregate_content_pointers
                child_pointers = [_shift_pointer(pointer, start) for pointer in child_local_pointers]
                if not child_pointers:
                    continue
                child_id = str(
                    stable_id(
                        "workflow_ingest.page_index_layer_materialized_child",
                        _stable_part(candidate_id),
                        _stable_part(child.node_type),
                        _stable_part([pointer.model_dump(mode="json") for pointer in child_pointers]),
                    )
                )
                materialized_children.append(
                    LayerChildCandidate(
                        node_id=child_id,
                        parent_node_id=candidate_id,
                        title=child.title,
                        node_type=child.node_type,
                        total_content_pointers=child_pointers,
                        expandable=bool(child.child_nodes),
                        metadata=cast(
                            dict[str, JsonValue],
                            {
                                **dict(child.metadata),
                                "parse_strategy": "page_index",
                                "page_index_materialized_child": True,
                                "summary": child.summary,
                            },
                        ),
                    )
                )
            candidates.append(
                LayerChildCandidate(
                    node_id=candidate_id,
                    parent_node_id=parent_id,
                    title=node.title,
                    node_type=node.node_type,
                    total_content_pointers=pointers,
                    expandable=bool(node.child_nodes),
                    metadata=cast(
                        dict[str, JsonValue],
                        {
                            **dict(node.metadata),
                            "parse_strategy": "page_index",
                            "page_index_layer_only": True,
                            "summary": node.summary,
                        },
                    ),
                    child_candidates=materialized_children,
                )
            )
    return candidates


def parse_page_index_document(
    *,
    document_id: str,
    title: str,
    raw_text: str,
    source_format: PageIndexSourceFormat = "text",
    mode: PageIndexMode = "heuristic",
    provider_settings: WorkflowProviderSettings | None = None,
    callbacks: list[object] | None = None,
    trace_log: PageIndexTraceLogger | None = None,
    provider_diagnostics_sink: ProviderDiagnosticsSink | None = None,
    refine_excerpts: bool = False,
    summary_enabled: bool = True,
    hierarchical_summary_enabled: bool | None = None,
    max_depth: int = 10,
) -> PageIndexParseResult:
    """Parse a plain text or Markdown document into a page-index semantic tree."""

    settings = provider_settings or WorkflowProviderSettings.from_env()
    assignment_mode_prefix = "ollama" if settings.parser.provider == "ollama" else "llm"
    effective_hierarchical_summary_enabled = (
        settings.page_index_hierarchical_summary_enabled
        if hierarchical_summary_enabled is None
        else hierarchical_summary_enabled
    )
    workflow_input = build_page_index_workflow_input(
        document_id=document_id,
        title=title,
        raw_text=raw_text,
        source_format=source_format,
    )
    authoritative_source_map = build_authoritative_source_map(workflow_input)
    parser_input_dict = build_parser_input_dict(workflow_input.collections[0])
    parser_source_map = build_parser_source_map(authoritative_source_map)

    page_units = sorted(
        (
            (unit_id, record)
            for unit_id, record in authoritative_source_map.items()
            if record.participates_in_semantic_text
        ),
        key=lambda item: (item[1].page_number, item[1].cluster_number or 0, item[0]),
    )

    root_pointers = [
        HydratedTextPointer(
            source_cluster_id=unit_id,
            start_char=0,
            end_char=max(0, len(record.text) - 1),
            verbatim_text=record.text,
        )
        for unit_id, record in page_units
    ]
    root_node = SemanticNode(
        title=title,
        node_type="DOCUMENT_ROOT",
        parent_id=None,
        level_from_root=0,
        total_content_pointers=root_pointers,
        child_nodes=[],
    )
    page_nodes: list[SemanticNode] = []
    page_diagnostics: list[PageIndexDiagnostics] = []
    for page_number, (unit_id, record) in enumerate(page_units, start=1):
        if trace_log is not None:
            trace_log(
                f"page_index_page_start page_number={page_number} unit_id={unit_id} mode={mode}"
            )
        page_text = record.text
        if mode == "heuristic":
            block_specs, page_diagnostics_item = _heuristic_page_outline(
                page_text,
                page_number=page_number,
                source_format=source_format,
            )
        elif mode in _PAGE_INDEX_PROVIDER_MODES:
            if settings.parser.provider != mode:
                raise ValueError(
                    f"{mode} mode requires parser provider={mode!r}; "
                    f"got {settings.parser.provider!r}"
                )
            if trace_log is not None:
                trace_log(f"page_index_llm_prepare page_number={page_number} provider={mode}")
            block_specs, page_diagnostics_item = _llm_page_outline(
                page_text=page_text,
                page_number=page_number,
                source_format=source_format,
                provider_settings=settings,
                callbacks=callbacks,
                trace_log=trace_log,
                provider_diagnostics_sink=provider_diagnostics_sink,
            )
        else:  # pragma: no cover - Literal guards this in type-checked code.
            raise ValueError(f"unsupported page index mode: {mode}")
        page_refinement_diagnostics = {
            "refine_excerpts_enabled": False,
            "refine_excerpts_attempted": 0,
            "refine_excerpts_accepted": 0,
            "refine_excerpts_rejected": 0,
            "refine_excerpts_fallback": False,
        }
        hierarchical_summary_diagnostics = {
            "hierarchical_summary_enabled": False,
            "hierarchical_summary_attempted": 0,
            "hierarchical_summary_accepted": 0,
            "hierarchical_summary_rejected": 0,
            "hierarchical_summary_fallback": False,
            "hierarchical_summary_call_count": 0,
        }
        if mode in _PAGE_INDEX_PROVIDER_MODES and refine_excerpts:
            block_specs, page_refinement_diagnostics = _refine_page_index_block_excerpts(
                block_specs=block_specs,
                page_text=page_text,
                page_number=page_number,
                unit_id=unit_id,
                provider_settings=settings,
                callbacks=callbacks,
                trace_log=trace_log,
                provider_diagnostics_sink=provider_diagnostics_sink,
            )
        if mode in _PAGE_INDEX_PROVIDER_MODES and summary_enabled and effective_hierarchical_summary_enabled:
            block_specs, hierarchical_summary_diagnostics = _refine_page_index_summaries_hierarchically(
                block_specs=block_specs,
                page_number=page_number,
                provider_settings=settings,
                callbacks=callbacks,
                trace_log=trace_log,
                provider_diagnostics_sink=provider_diagnostics_sink,
            )
        page_diagnostics_item = dict(page_diagnostics_item)
        page_diagnostics_item.update(page_refinement_diagnostics)
        page_diagnostics_item.update(hierarchical_summary_diagnostics)
        page_repair_stats: dict[str, int] = {
            "pointer_fuzzy_repairs": 0,
            "pointer_fuzzy_failures": 0,
        }
        page_diagnostics_item["page_number"] = page_number
        page_diagnostics_item["unit_id"] = unit_id
        page_diagnostics.append(page_diagnostics_item)
        if trace_log is not None and page_diagnostics_item.get("fallback_reason"):
            trace_log(
                "page_index_page_diagnostics "
                f"page_number={page_number} assignment_mode={page_diagnostics_item.get('assignment_mode')} "
                f"fallback_reason={page_diagnostics_item.get('fallback_reason')}"
            )

        page_node = _make_semantic_node(
            title=f"Page {page_number}",
            node_type="PAGE",
            parent_id=root_node.node_id,
            level_from_root=1,
            pointers=[
                HydratedTextPointer(
                    source_cluster_id=unit_id,
                    start_char=0,
                    end_char=max(0, len(page_text) - 1),
                    verbatim_text=page_text,
                )
            ],
            metadata={"page_index_role": "page", "semantic_kind": "page"},
        )
        child_nodes, _ = _materialize_block_tree(
            block_specs=block_specs,
            page_text=page_text,
            unit_id=unit_id,
            parent_id=page_node.node_id or document_id,
            level_from_root=2,
            repair_stats=page_repair_stats,
            trace_log=trace_log,
            summary_enabled=summary_enabled,
        )
        page_node.child_nodes.extend(child_nodes)
        page_nodes.append(page_node)
        page_diagnostics_item.update(page_repair_stats)
        if trace_log is not None:
            trace_log(
                f"page_index_page_end page_number={page_number} unit_id={unit_id} child_count={len(child_nodes)}"
            )

    semantic_tree = root_node.model_copy(update={"child_nodes": page_nodes})
    # Check ancestry and source ownership before wrapper normalization.  The
    # normalizer may intentionally reparent a valid structural wrapper; it
    # must never be allowed to hide a malformed provider parent reference.
    validate_semantic_tree_integrity(
        semantic_tree,
        parser_source_map=parser_source_map,
        max_depth=max_depth,
        validate_shape=False,
    )
    semantic_tree = _normalize_page_index_unary_nodes(semantic_tree)
    semantic_tree = finalize_semantic_tree(
        semantic_tree,
        parser_source_map=parser_source_map,
        max_depth=max_depth,
    )
    terminal_coverage = compute_terminal_content_coverage(
        semantic_tree, parser_source_map
    )
    if terminal_coverage["valid"] is not True:
        raise ValueError(
            "PageIndex terminal source ownership is incomplete or duplicated: "
            f"{terminal_coverage}"
        )
    coverage = compute_pointer_coverage(
        semantic_tree,
        cast(dict[str, dict[str, JsonValue]], parser_source_map),
    )
    validation_errors = [error for item in page_diagnostics for error in _object_list(item.get("validation_errors"))]
    validation_warnings = [warning for item in page_diagnostics for warning in _object_list(item.get("validation_warnings"))]
    first_validation_errors = [error for item in page_diagnostics for error in _object_list(item.get("first_validation_errors"))]
    retry_validation_errors = [error for item in page_diagnostics for error in _object_list(item.get("retry_validation_errors"))]
    assignment_validation_errors = [error for item in page_diagnostics for error in _object_list(item.get("assignment_validation_errors"))]
    structure_validation_errors = [error for item in page_diagnostics for error in _object_list(item.get("structure_validation_errors"))]
    fallback_reasons = [item.get("fallback_reason") for item in page_diagnostics if item.get("fallback_reason")]
    candidate_count = sum(_object_int(item.get("candidate_count"), 0) for item in page_diagnostics)
    assignment_count = sum(_object_int(item.get("assignment_count"), 0) for item in page_diagnostics)
    assignment_attempt_count = sum(_object_int(item.get("assignment_attempt_count"), 1) for item in page_diagnostics)
    assignment_retry_used = any(bool(item.get("assignment_retry_used")) for item in page_diagnostics)
    assignment_retry_succeeded = any(bool(item.get("assignment_retry_succeeded")) for item in page_diagnostics)
    structure_retry_used = any(bool(item.get("structure_retry_used")) for item in page_diagnostics)
    structure_retry_succeeded = any(bool(item.get("structure_retry_succeeded")) for item in page_diagnostics)
    retry_used = any(bool(item.get("retry_used")) for item in page_diagnostics)
    retry_succeeded = any(bool(item.get("retry_succeeded")) for item in page_diagnostics)
    assignment_modes = {str(item.get("assignment_mode") or "") for item in page_diagnostics if item.get("assignment_mode")}
    branch_local_salvage_used = any(bool(item.get("branch_local_salvage")) for item in page_diagnostics)
    branch_local_salvage = [
        item["branch_local_salvage"] for item in page_diagnostics if item.get("branch_local_salvage")
    ]
    refinement_enabled = any(bool(item.get("refine_excerpts_enabled")) for item in page_diagnostics)
    refinement_attempted = sum(_object_int(item.get("refine_excerpts_attempted"), 0) for item in page_diagnostics)
    refinement_accepted = sum(_object_int(item.get("refine_excerpts_accepted"), 0) for item in page_diagnostics)
    refinement_rejected = sum(_object_int(item.get("refine_excerpts_rejected"), 0) for item in page_diagnostics)
    refinement_fallback_pages = sum(1 for item in page_diagnostics if item.get("refine_excerpts_fallback"))
    pointer_fuzzy_repairs = sum(_object_int(item.get("pointer_fuzzy_repairs"), 0) for item in page_diagnostics)
    pointer_fuzzy_failures = sum(_object_int(item.get("pointer_fuzzy_failures"), 0) for item in page_diagnostics)
    if mode == "heuristic":
        overall_assignment_mode = "heuristic_deterministic"
    elif fallback_reasons:
        overall_assignment_mode = "deterministic_fallback"
    elif structure_retry_succeeded:
        overall_assignment_mode = f"{assignment_mode_prefix}_flat_assignment_structure_retry"
    elif assignment_retry_succeeded:
        overall_assignment_mode = f"{assignment_mode_prefix}_flat_assignment_retry"
    elif branch_local_salvage_used:
        overall_assignment_mode = f"{assignment_mode_prefix}_branch_local_salvage"
    elif retry_used:
        overall_assignment_mode = "deterministic_fallback"
    else:
        expected_assignment_modes = {
            f"{assignment_mode_prefix}_flat_assignment",
            f"{assignment_mode_prefix}_flat_assignment_retry",
            f"{assignment_mode_prefix}_flat_assignment_structure_retry",
        }
        overall_assignment_mode = (
            f"{assignment_mode_prefix}_flat_assignment"
            if assignment_modes.issubset(expected_assignment_modes) and assignment_modes
            else "deterministic_fallback"
        )
    if fallback_reasons:
        final_outcome = "deterministic_fallback"
    elif structure_retry_succeeded:
        final_outcome = "structure_retry_success"
    elif assignment_retry_succeeded:
        final_outcome = "assignment_retry_success"
    elif branch_local_salvage_used:
        final_outcome = "branch_local_salvage"
    else:
        final_outcome = "first_pass_success"
    diagnostics = {
        "assignment_mode": overall_assignment_mode,
        "final_outcome": final_outcome,
        "fallback_reason": fallback_reasons[0] if fallback_reasons else None,
        "candidate_count": candidate_count,
        "assignment_count": assignment_count,
        "assignment_attempt_count": assignment_attempt_count,
        "assignment_retry_used": assignment_retry_used,
        "assignment_retry_succeeded": assignment_retry_succeeded,
        "branch_local_salvage_used": branch_local_salvage_used,
        "branch_local_salvage": branch_local_salvage,
        "structure_retry_used": structure_retry_used,
        "structure_retry_succeeded": structure_retry_succeeded,
        "retry_used": retry_used,
        "retry_succeeded": retry_succeeded,
        "refine_excerpts_enabled": refinement_enabled,
        "refine_excerpts_attempted": refinement_attempted,
        "refine_excerpts_accepted": refinement_accepted,
        "refine_excerpts_rejected": refinement_rejected,
        "refine_excerpts_fallback": refinement_fallback_pages > 0,
        "refine_excerpts_fallback_pages": refinement_fallback_pages,
        "pointer_fuzzy_repairs": pointer_fuzzy_repairs,
        "pointer_fuzzy_failures": pointer_fuzzy_failures,
        "validation_errors": validation_errors,
        "assignment_validation_errors": assignment_validation_errors,
        "structure_validation_errors": structure_validation_errors,
        "first_validation_errors": first_validation_errors,
        "retry_validation_errors": retry_validation_errors,
        "validation_warnings": validation_warnings,
        "page_diagnostics": page_diagnostics,
    }
    if trace_log is not None:
        trace_log(
            "page_index_parse_complete "
            f"document_id={document_id} page_count={len(page_nodes)} coverage={coverage.get('coverage_ratio')}"
        )
    return PageIndexParseResult(
        mode=mode,
        source_format=source_format,
        workflow_input=workflow_input,
        authoritative_source_map=authoritative_source_map,
        parser_input_dict=parser_input_dict,
        parser_source_map=parser_source_map,
        semantic_tree=semantic_tree,
        coverage=coverage,
        diagnostics=diagnostics,
    )
