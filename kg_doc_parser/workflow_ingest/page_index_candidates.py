"""Deterministic PageIndex block extraction and assignment validation."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from .page_index_classification import (
    _classify_block,
    _is_markdown_table_block,
    _is_markdown_table_row,
    _is_numeric_heavy_or_table_like,
    _is_setext_underline,
)
from .page_index_contracts import (
    BlockAssignment,
    CandidateBlock,
    ExcerptRefinementSuggestion,
    HierarchicalSummaryAssignment,
    PageIndexBlockSpec,
    PageIndexDiagnostics,
    PageIndexSourceFormat,
    PageIndexSourceRole,
    PageIndexValidationResult,
)
from .page_index_contracts import BlockSpan as _BlockSpan
from .page_index_source_text import _split_pages, build_page_index_workflow_input
from .page_index_tree import (
    _find_page_unit,
    _page_index_find_best_fuzzy_span,
    _page_index_normalize_text,
)
from .serialization import JsonObject, JsonValue

__all__ = [
    "ExcerptRefinementSuggestion",
    "HierarchicalSummaryAssignment",
    "PageIndexSourceRole",
    "_classify_block",
    "_find_page_unit",
    "_is_markdown_table_block",
    "_is_markdown_table_row",
    "_is_numeric_heavy_or_table_like",
    "_is_setext_underline",
    "_page_index_find_best_fuzzy_span",
    "_page_index_normalize_text",
    "_split_pages",
    "build_page_index_workflow_input",
]


def _json_string_list(values: Sequence[str]) -> list[JsonValue]:
    result: list[JsonValue] = []
    result.extend(values)
    return result


def _json_string_mapping(values: Mapping[str, Sequence[str]]) -> JsonObject:
    return {key: _json_string_list(items) for key, items in values.items()}


def _normalize_page_index_excerpt(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _flatten_page_index_specs(
    specs: list[PageIndexBlockSpec],
) -> list[PageIndexBlockSpec]:
    flattened: list[PageIndexBlockSpec] = []

    def walk(items: list[PageIndexBlockSpec]) -> None:
        for item in items:
            flattened.append(item)
            walk(item.child_nodes)

    walk(specs)
    return flattened


def _page_index_block_exceeds_excerpt_budget(
    spec: PageIndexBlockSpec, page_text: str
) -> bool:
    """Keep the legacy predicate: grounded extents have no page-relative cap."""

    del spec, page_text
    return False


def _page_index_block_is_too_generic(
    spec: PageIndexBlockSpec,
    *,
    ancestor_excerpts: tuple[str, ...] = (),
) -> bool:
    excerpt_norm = _normalize_page_index_excerpt(spec.excerpt)
    if not excerpt_norm:
        return True
    if excerpt_norm in ancestor_excerpts:
        return True
    if len(spec.child_nodes) > 1:
        child_norms = [
            _normalize_page_index_excerpt(child.excerpt)
            for child in spec.child_nodes
            if _normalize_page_index_excerpt(child.excerpt)
        ]
        if len(child_norms) != len(spec.child_nodes):
            return True
        if len(set(child_norms)) == 1:
            return True
    return False


def _split_page_blocks(page_text: str, *, source_format: PageIndexSourceFormat = "text") -> list[_BlockSpan]:
    blocks: list[_BlockSpan] = []
    lines = page_text.splitlines(keepends=True)
    cursor = 0
    paragraph_start: int | None = None
    paragraph_start_line = 0
    paragraph_end = 0
    paragraph_end_line = 0
    paragraph_indent = 0
    saw_nonblank = False

    def _flush_paragraph() -> None:
        nonlocal paragraph_start, paragraph_end, paragraph_start_line, paragraph_end_line, paragraph_indent
        if paragraph_start is None:
            return
        raw = page_text[paragraph_start:paragraph_end]
        trimmed = raw.strip()
        if trimmed:
            relative_start = raw.find(trimmed)
            start_char = paragraph_start + relative_start
            end_char = start_char + len(trimmed) - 1
            blocks.append(
                _classify_block(
                    trimmed,
                    start_char,
                    end_char,
                    source_format=source_format,
                    is_first=False,
                    has_blank_before=True,
                    has_blank_after=True,
                    line_start=paragraph_start_line or 1,
                    line_end=paragraph_end_line or paragraph_start_line or 1,
                    indent=paragraph_indent,
                )
            )
        paragraph_start = None
        paragraph_start_line = 0
        paragraph_end = 0
        paragraph_end_line = 0
        paragraph_indent = 0

    index = 0
    while index < len(lines):
        line = lines[index]
        line_start = cursor
        line_end = cursor + len(line)
        line_body = line.rstrip("\r\n")
        stripped = line_body.strip()
        indent = len(line_body) - len(line_body.lstrip(" \t"))
        has_blank_before = index == 0 or not lines[index - 1].strip()
        has_blank_after = index + 1 >= len(lines) or not lines[index + 1].strip()
        if not stripped:
            _flush_paragraph()
            cursor = line_end
            index += 1
            continue
        next_line_body = lines[index + 1].rstrip("\r\n") if index + 1 < len(lines) else ""
        if next_line_body and _is_setext_underline(next_line_body):
            underline_level = 1 if next_line_body.strip().startswith("=") else 2
            _flush_paragraph()
            start_char = line_start + line_body.find(stripped)
            end_char = start_char + len(stripped) - 1
            line_block = _classify_block(
                stripped,
                start_char,
                end_char,
                source_format=source_format,
                is_first=not saw_nonblank,
                has_blank_before=has_blank_before,
                has_blank_after=has_blank_after,
                line_start=index + 1,
                line_end=index + 1,
                indent=indent,
                heading_level=underline_level,
            )
            saw_nonblank = True
            blocks.append(line_block)
            cursor = line_end + len(lines[index + 1])
            index += 2
            continue
        line_block = _classify_block(
            stripped,
            line_start + line_body.find(stripped),
            line_end - 1,
            source_format=source_format,
            is_first=not saw_nonblank,
            has_blank_before=has_blank_before,
            has_blank_after=has_blank_after,
            line_start=index + 1,
            line_end=index + 1,
            indent=indent,
        )
        saw_nonblank = True
        if line_block.node_type != "PARAGRAPH":
            _flush_paragraph()
            blocks.append(line_block)
        else:
            if paragraph_start is None:
                paragraph_start = line_start
                paragraph_start_line = index + 1
                paragraph_indent = indent
            paragraph_end = line_end
            paragraph_end_line = index + 1
        cursor = line_end
        index += 1
    _flush_paragraph()
    return blocks


def _extract_candidate_blocks(page_text: str, *, page_number: int, source_format: PageIndexSourceFormat) -> list[CandidateBlock]:
    candidates: list[CandidateBlock] = []
    for order, block in enumerate(_split_page_blocks(page_text, source_format=source_format), start=1):
        candidates.append(
            CandidateBlock(
                block_id=f"p{page_number:04d}-b{order:03d}",
                page_number=page_number,
                order=order,
                start_char=block.start_char,
                end_char=block.end_char,
                line_start=block.line_start,
                line_end=block.line_end,
                indent=block.indent,
                kind_hint=block.kind_hint,
                confidence=block.confidence,
                text=block.text,
                node_type_hint=block.node_type,
                title_hint=block.title,
                heading_level=block.heading_level,
            )
        )
    return candidates


def _candidate_heading_evidence(candidate: CandidateBlock) -> bool:
    return candidate.heading_level is not None or candidate.node_type_hint in {"SECTION", "SUBSECTION"} or candidate.kind_hint.startswith("heading")


def _candidate_term_evidence(candidate: CandidateBlock) -> bool:
    if candidate.node_type_hint == "TERM" or candidate.kind_hint.startswith("term"):
        return True
    normalized = " ".join(candidate.text.split())
    return (
        len(normalized.split()) <= 12
        and not normalized.endswith((".", "!", "?"))
        and bool(re.match(r"^[\w\(\)\-\*].*", normalized))
    )


def _deterministic_block_assignments(candidates: list[CandidateBlock]) -> list[BlockAssignment]:
    assignments: list[BlockAssignment] = []
    stack: list[tuple[int, str]] = []
    for candidate in candidates:
        parent_id: str | None = stack[-1][1] if stack else None
        if _candidate_heading_evidence(candidate):
            level = candidate.heading_level or 1
            while stack and stack[-1][0] >= level:
                stack.pop()
            parent_id = stack[-1][1] if stack else None
            assignments.append(
                BlockAssignment(
                    block_id=candidate.block_id,
                    parent_id=parent_id,
                    node_type="HEADING",
                    title=candidate.title_hint,
                )
            )
            stack.append((level, candidate.block_id))
            continue
        assignments.append(
            BlockAssignment(
                block_id=candidate.block_id,
                parent_id=parent_id,
                node_type=candidate.node_type_hint,
                title=candidate.title_hint or candidate.text[:120],
            )
        )
    return assignments


def _validate_block_assignments(
    candidates: list[CandidateBlock],
    assignments: list[BlockAssignment],
    *,
    page_text: str,
) -> PageIndexValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    candidate_ids = [candidate.block_id for candidate in candidates]
    assignment_ids = [assignment.block_id for assignment in assignments]
    candidate_by_id = {candidate.block_id: candidate for candidate in candidates}
    index_by_id = {candidate.block_id: index for index, candidate in enumerate(candidates)}

    if len(assignment_ids) != len(set(assignment_ids)):
        errors.append("duplicate block assignments detected")
    if set(assignment_ids) != set(candidate_ids):
        missing = [block_id for block_id in candidate_ids if block_id not in assignment_ids]
        unknown = [block_id for block_id in assignment_ids if block_id not in candidate_by_id]
        if missing:
            errors.append(f"missing block assignments: {missing}")
        if unknown:
            errors.append(f"unknown block ids: {unknown}")
    if assignment_ids != candidate_ids:
        errors.append("reading order was not preserved")

    parent_chain_map = {assignment.block_id: assignment.parent_id for assignment in assignments}
    heading_candidate_ids = [candidate.block_id for candidate in candidates if _candidate_heading_evidence(candidate)]
    heading_assignment_parent_ids = {
        assignment.block_id: assignment.parent_id
        for assignment in assignments
        if assignment.block_id in candidate_by_id
        and assignment.node_type in {"HEADING", "SECTION", "SUBSECTION"}
        and _candidate_heading_evidence(candidate_by_id[assignment.block_id])
    }
    baseline_assignments = _deterministic_block_assignments(candidates)
    baseline_heading_parent_ids = {
        assignment.block_id: assignment.parent_id
        for assignment in baseline_assignments
        if assignment.block_id in candidate_by_id and _candidate_heading_evidence(candidate_by_id[assignment.block_id])
    }
    for assignment in assignments:
        candidate = candidate_by_id.get(assignment.block_id)
        if candidate is None:
            continue
        parent_id = assignment.parent_id
        if parent_id is not None:
            parent_index = index_by_id.get(parent_id)
            if parent_index is None:
                errors.append(f"assignment {assignment.block_id!r} references unknown parent {parent_id!r}")
            elif parent_index >= index_by_id[assignment.block_id]:
                errors.append(f"assignment {assignment.block_id!r} references a forward or self parent {parent_id!r}")
        seen: set[str] = set()
        probe = parent_id
        while probe is not None:
            if probe in seen:
                errors.append(f"assignment {assignment.block_id!r} introduces a cycle")
                break
            seen.add(probe)
            probe = parent_chain_map.get(probe)
        if assignment.node_type in {"HEADING", "SECTION", "SUBSECTION"} and not _candidate_heading_evidence(candidate):
            child_count = sum(1 for other in assignments if other.parent_id == assignment.block_id)
            if child_count == 0:
                errors.append(f"{assignment.block_id!r} lacks heading evidence for {assignment.node_type}")
        if assignment.node_type == "TERM" and not _candidate_term_evidence(candidate):
            errors.append(f"{assignment.block_id!r} lacks term evidence")
    baseline_nested_heading_count = sum(1 for parent_id in baseline_heading_parent_ids.values() if parent_id is not None)
    assigned_nested_heading_count = sum(1 for parent_id in heading_assignment_parent_ids.values() if parent_id is not None)
    if baseline_nested_heading_count > 0 and assigned_nested_heading_count == 0 and len(heading_candidate_ids) >= 2:
        errors.append("heading structure was flattened")

    normalized_page_text = " ".join(page_text.split()).strip().lower()
    sibling_groups: dict[str | None, list[CandidateBlock]] = {}
    for assignment in assignments:
        if assignment.block_id not in candidate_by_id:
            continue
        sibling_groups.setdefault(assignment.parent_id, []).append(candidate_by_id[assignment.block_id])
    for parent_id, siblings in sibling_groups.items():
        if len(siblings) <= 1:
            continue
        if any(not block.text.strip() for block in siblings):
            errors.append(f"parent {parent_id!r} has empty sibling text")
            continue
        for left_index, left in enumerate(siblings):
            for right in siblings[left_index + 1 :]:
                if left.end_char < right.start_char or right.end_char < left.start_char:
                    continue
                errors.append(
                    f"parent {parent_id!r} reuses overlapping source occurrence "
                    f"{left.block_id!r}/{right.block_id!r}"
                )
    if any(" ".join(candidate.text.split()).strip().lower() == normalized_page_text for candidate in candidates) and len(candidates) > 1:
        errors.append("candidate excerpt duplicates the whole page")

    fallback_reason = None if not errors else "validation_failed"
    if warnings and fallback_reason is None:
        fallback_reason = None
    return PageIndexValidationResult(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        fallback_reason=fallback_reason,
    )


def _salvage_page_index_assignments(
    *,
    candidates: list[CandidateBlock],
    assignments: list[BlockAssignment],
    page_text: str,
) -> tuple[list[BlockAssignment] | None, PageIndexDiagnostics]:
    """Keep valid provider branches while replacing locally invalid blocks.

    Salvage is deliberately conservative: the assignment set must still cover
    every candidate in reading order, and the merged result must pass the same
    whole-page validators used by the normal provider path.  This prevents a
    partial provider response from silently becoming an authoritative tree.
    """

    candidate_ids = [candidate.block_id for candidate in candidates]
    assignment_ids = [assignment.block_id for assignment in assignments]
    if assignment_ids != candidate_ids or len(assignment_ids) != len(set(assignment_ids)):
        return None, {"reason": "assignment_set_not_complete_or_ordered"}

    candidate_by_id = {candidate.block_id: candidate for candidate in candidates}
    baseline = _deterministic_block_assignments(candidates)
    baseline_by_id = {assignment.block_id: assignment for assignment in baseline}
    index_by_id = {candidate.block_id: index for index, candidate in enumerate(candidates)}
    parent_by_id = {assignment.block_id: assignment.parent_id for assignment in assignments}
    children_by_id: dict[str, int] = {}
    for assignment in assignments:
        if assignment.parent_id is not None:
            children_by_id[assignment.parent_id] = children_by_id.get(assignment.parent_id, 0) + 1

    invalid_ids: set[str] = set()
    reasons: dict[str, list[str]] = {}

    def mark_invalid(block_id: str, reason: str) -> None:
        invalid_ids.add(block_id)
        reasons.setdefault(block_id, []).append(reason)

    for assignment in assignments:
        candidate = candidate_by_id[assignment.block_id]
        parent_id = assignment.parent_id
        if parent_id is not None:
            parent_index = index_by_id.get(parent_id)
            if parent_index is None:
                mark_invalid(assignment.block_id, "unknown_parent")
            elif parent_index >= index_by_id[assignment.block_id]:
                mark_invalid(assignment.block_id, "forward_or_self_parent")
        if (
            assignment.node_type in {"HEADING", "SECTION", "SUBSECTION"}
            and not _candidate_heading_evidence(candidate)
            and children_by_id.get(assignment.block_id, 0) == 0
        ):
            mark_invalid(assignment.block_id, "heading_without_heading_evidence")
        if assignment.node_type == "TERM" and not _candidate_term_evidence(candidate):
            mark_invalid(assignment.block_id, "term_without_term_evidence")

        seen: set[str] = set()
        probe = parent_id
        while probe is not None:
            if probe in seen or probe == assignment.block_id:
                mark_invalid(assignment.block_id, "parent_cycle")
                for cycle_id in seen:
                    mark_invalid(cycle_id, "parent_cycle")
                break
            seen.add(probe)
            probe = parent_by_id.get(probe)

    merged: list[BlockAssignment] = []
    for assignment in assignments:
        if assignment.block_id in invalid_ids:
            merged.append(baseline_by_id[assignment.block_id])
        else:
            merged.append(assignment)

    # A flattened provider tree is a page-level structural error. Preserve the
    # provider's labels and summaries, but restore deterministic heading links.
    baseline_heading_parents = {
        assignment.block_id: assignment.parent_id
        for assignment in baseline
        if assignment.node_type in {"HEADING", "SECTION", "SUBSECTION"}
    }
    merged_heading_count = sum(
        1
        for assignment in merged
        if assignment.node_type in {"HEADING", "SECTION", "SUBSECTION"} and assignment.parent_id is not None
    )
    if merged_heading_count == 0 and any(parent is not None for parent in baseline_heading_parents.values()):
        merged = [
            assignment.model_copy(update={"parent_id": baseline_heading_parents[assignment.block_id]})
            if assignment.node_type in {"HEADING", "SECTION", "SUBSECTION"}
            and baseline_heading_parents.get(assignment.block_id) is not None
            else assignment
            for assignment in merged
        ]

    assignment_validation = _validate_block_assignments(candidates, merged, page_text=page_text)
    if not assignment_validation.valid:
        return None, {
            "reason": "merged_assignment_validation_failed",
            "errors": _json_string_list(assignment_validation.errors),
            "invalid_block_ids": _json_string_list(sorted(invalid_ids)),
        }
    block_specs = _assemble_page_index_blocks(candidates=candidates, assignments=merged)
    structure_validation = _validate_page_index_block_structure(
        candidates=candidates,
        assignments=merged,
        block_specs=block_specs,
        page_text=page_text,
    )
    if not structure_validation.valid:
        return None, {
            "reason": "merged_structure_validation_failed",
            "errors": _json_string_list(structure_validation.errors),
            "invalid_block_ids": _json_string_list(sorted(invalid_ids)),
        }
    return merged, {
        "invalid_block_ids": _json_string_list(sorted(invalid_ids)),
        "repaired_block_count": len(invalid_ids),
        "repair_reasons": _json_string_mapping(reasons),
        "validation_warnings": _json_string_list(structure_validation.warnings),
    }


def _assemble_page_index_blocks(
    *,
    candidates: list[CandidateBlock],
    assignments: list[BlockAssignment],
) -> list[PageIndexBlockSpec]:
    candidate_by_id = {candidate.block_id: candidate for candidate in candidates}
    repaired_assignments = _repair_block_assignments_for_assembly(candidates=candidates, assignments=assignments)
    spec_by_id: dict[str, PageIndexBlockSpec] = {}
    roots: list[PageIndexBlockSpec] = []
    for assignment in repaired_assignments:
        candidate = candidate_by_id[assignment.block_id]
        spec = PageIndexBlockSpec(
            title=assignment.title or candidate.title_hint,
            node_type="HEADING" if _candidate_heading_evidence(candidate) else assignment.node_type,
            excerpt=candidate.text,
            summary=assignment.summary.strip(),
            source_role="heading" if _candidate_heading_evidence(candidate) else "content",
        )
        spec_by_id[assignment.block_id] = spec
    for assignment in repaired_assignments:
        spec = spec_by_id[assignment.block_id]
        parent_id = assignment.parent_id
        if parent_id and parent_id in spec_by_id:
            spec_by_id[parent_id].child_nodes.append(spec)
        else:
            roots.append(spec)
    return roots


def _repair_block_assignments_for_assembly(
    *,
    candidates: list[CandidateBlock],
    assignments: list[BlockAssignment],
) -> list[BlockAssignment]:
    candidate_ids = {candidate.block_id for candidate in candidates}
    index_by_id = {candidate.block_id: index for index, candidate in enumerate(candidates)}
    repaired: list[BlockAssignment] = []
    for assignment in assignments:
        parent_id = assignment.parent_id
        if parent_id is None or parent_id not in candidate_ids:
            repaired_parent_id = None
        else:
            parent_index = index_by_id.get(parent_id)
            child_index = index_by_id.get(assignment.block_id)
            if parent_index is None or child_index is None or parent_index >= child_index:
                repaired_parent_id = None
            else:
                repaired_parent_id = parent_id
        repaired.append(assignment.model_copy(update={"parent_id": repaired_parent_id}))
    return repaired


def _validate_page_index_block_structure(
    *,
    candidates: list[CandidateBlock],
    assignments: list[BlockAssignment],
    block_specs: list[PageIndexBlockSpec],
    page_text: str,
) -> PageIndexValidationResult:
    """Validate the assembled tree shape before accepting an LLM assignment."""
    errors: list[str] = []
    warnings: list[str] = []
    if candidates and not block_specs:
        errors.append("block tree is empty")

    normalized_page_text = _normalize_page_index_excerpt(page_text)
    flattened_specs = _flatten_page_index_specs(block_specs)
    spec_candidate_by_identity = (
        {id(spec): candidate for spec, candidate in zip(flattened_specs, candidates)}
        if len(flattened_specs) == len(candidates)
        else {}
    )
    baseline_assignments = _deterministic_block_assignments(candidates)
    candidate_by_id = {candidate.block_id: candidate for candidate in candidates}
    baseline_nested_heading_count = sum(
        1
        for assignment in baseline_assignments
        if assignment.parent_id is not None
        and assignment.block_id in candidate_by_id
        and _candidate_heading_evidence(candidate_by_id[assignment.block_id])
    )
    repaired_assignments = _repair_block_assignments_for_assembly(
        candidates=candidates,
        assignments=assignments,
    )
    repaired_nested_heading_count = sum(
        1
        for assignment in repaired_assignments
        if assignment.parent_id is not None
        and assignment.block_id in candidate_by_id
        and _candidate_heading_evidence(candidate_by_id[assignment.block_id])
    )
    if baseline_nested_heading_count > 0 and repaired_nested_heading_count == 0:
        errors.append("assembled heading structure was flattened")

    def _validate_siblings(siblings: list[PageIndexBlockSpec], *, path: str) -> None:
        normalized_sibling_excerpts: list[str] = []
        for index, spec in enumerate(siblings, start=1):
            current_path = f"{path}.{index}" if path else str(index)
            normalized_excerpt = _normalize_page_index_excerpt(spec.excerpt)
            if not normalized_excerpt:
                errors.append(f"block {current_path} has empty excerpt")
            if spec.display_excerpt is not None:
                display_excerpt = spec.display_excerpt.strip()
                if not display_excerpt:
                    errors.append(f"block {current_path} has an empty display excerpt")
                elif display_excerpt not in spec.excerpt:
                    errors.append(
                        f"block {current_path} display excerpt is not an exact substring of its grounded excerpt"
                    )
            if normalized_excerpt == normalized_page_text and len(candidates) > 1:
                errors.append(f"block {current_path} duplicates the whole page")
            if _page_index_block_exceeds_excerpt_budget(spec, page_text):
                errors.append(f"block {current_path} excerpt is too broad")
            if _page_index_block_is_too_generic(spec):
                errors.append(f"block {current_path} excerpt is too generic")
            normalized_sibling_excerpts.append(normalized_excerpt)
            _validate_siblings(spec.child_nodes, path=current_path)
        for left_index, left in enumerate(siblings):
            for right in siblings[left_index + 1 :]:
                left_candidate = spec_candidate_by_identity.get(id(left))
                right_candidate = spec_candidate_by_identity.get(id(right))
                if left_candidate is None or right_candidate is None:
                    if (
                        _normalize_page_index_excerpt(left.excerpt)
                        == _normalize_page_index_excerpt(right.excerpt)
                    ):
                        errors.append(
                            f"block siblings at {path or 'root'} repeat the same excerpt"
                        )
                    continue
                if left_candidate.end_char < right_candidate.start_char or right_candidate.end_char < left_candidate.start_char:
                    continue
                errors.append(
                    f"block siblings at {path or 'root'} reuse overlapping source occurrences"
                )

    _validate_siblings(block_specs, path="")
    return PageIndexValidationResult(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        fallback_reason=None if not errors else "structure_validation_failed",
    )
