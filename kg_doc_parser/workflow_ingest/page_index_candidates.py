"""Deterministic PageIndex block extraction and assignment validation."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from .models import (
    NormalizedPage,
    NormalizedSourceCollection,
    SourceUnit,
    WorkflowIngestInput,
)
from .page_index_contracts import (
    BlockAssignment,
    CandidateBlock,
    ExcerptRefinementSuggestion,  # noqa: F401 - preserve module import compatibility
    HierarchicalSummaryAssignment,  # noqa: F401 - preserve module import compatibility
    PageIndexBlockSpec,
    PageIndexDiagnostics,
    PageIndexNodeType,
    PageIndexSourceFormat,
    PageIndexSourceRole,  # noqa: F401 - preserve module import compatibility
    PageIndexValidationResult,
)
from .page_index_contracts import (
    BlockSpan as _BlockSpan,
)
from .page_index_tree import (
    _find_page_unit,  # noqa: F401 - compatibility re-export
    _page_index_find_best_fuzzy_span,  # noqa: F401 - compatibility re-export
    _page_index_normalize_text,  # noqa: F401 - compatibility re-export
)
from .serialization import JsonObject, JsonValue


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


def _split_pages(raw_text: str) -> list[str]:
    """Split a logical document into page-sized chunks."""

    pages = re.split(r"\f|^\s*--- PAGE BREAK ---\s*$", raw_text, flags=re.MULTILINE)
    return [page.strip("\n") for page in pages if page.strip()]


def _is_setext_underline(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and bool(re.fullmatch(r"=+|-+", stripped))


def _is_numeric_heavy_or_table_like(text: str) -> bool:
    compact = " ".join(text.split())
    if not compact:
        return False
    digit_count = sum(ch.isdigit() for ch in compact)
    alpha_count = sum(ch.isalpha() for ch in compact)
    pipe_count = compact.count("|")
    tab_count = compact.count("\t")
    if pipe_count >= 2 or tab_count >= 2:
        return True
    if alpha_count == 0 and digit_count >= 3:
        return True
    if digit_count >= 6 and digit_count > alpha_count:
        return True
    return bool(re.search(r"\b\d+\b(?:\s+\b\d+\b){2,}", compact))


def _is_markdown_table_row(text: str) -> bool:
    """Return whether one line has the shape of a Markdown table row."""

    stripped = text.strip()
    if stripped.startswith(("```", "~~~")):
        return False
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    return len(cells) >= 2 and "|" in stripped


def _is_markdown_table_block(text: str, *, source_format: PageIndexSourceFormat) -> bool:
    """Recognize a bounded Markdown table as one explicit atomic block."""

    if source_format != "markdown":
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2 or not all(_is_markdown_table_row(line) for line in lines):
        return False
    separator = lines[1].strip("|").split("|")
    return len(separator) >= 2 and all(bool(re.fullmatch(r"\s*:?-{3,}:?\s*", cell)) for cell in separator)


def build_page_index_workflow_input(
    *,
    document_id: str,
    title: str,
    raw_text: str,
    source_format: PageIndexSourceFormat,
) -> WorkflowIngestInput:
    pages = _split_pages(raw_text)
    normalized_pages: list[NormalizedPage] = []
    for page_number, page_text in enumerate(pages, start=1):
        normalized_pages.append(
            NormalizedPage(
                page_number=page_number,
                units=[
                    SourceUnit(
                        modality="text",
                        page_number=page_number,
                        cluster_number=0,
                        text=page_text,
                        embedding_space="default_text",
                        metadata={"source_format": source_format},
                    )
                ],
                metadata={"source_format": source_format},
            )
        )
    return WorkflowIngestInput(
        request_id=document_id,
        collections=[
            NormalizedSourceCollection(
                collection_id=document_id,
                title=title,
                modality="text",
                pages=normalized_pages,
                embedding_spaces=["default_text"],
                metadata={"source_format": source_format},
            )
        ],
    )


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


def _classify_block(
    text: str,
    start_char: int,
    end_char: int,
    *,
    source_format: PageIndexSourceFormat = "text",
    is_first: bool = False,
    has_blank_before: bool = False,
    has_blank_after: bool = False,
    line_start: int = 1,
    line_end: int = 1,
    indent: int = 0,
    heading_level: int | None = None,
) -> _BlockSpan:
    stripped = text.strip()
    first_line = stripped.splitlines()[0].strip()
    word_count = len(first_line.split())
    is_sentence_like = first_line.endswith((".", "!", "?"))
    surrounded_by_blank_lines = has_blank_before and has_blank_after
    numeric_heavy = _is_numeric_heavy_or_table_like(first_line)
    if _is_markdown_table_block(stripped, source_format=source_format):
        return _BlockSpan(
            start_char=start_char,
            end_char=end_char,
            text=stripped,
            node_type="TABLE",
            title="Table",
            line_start=line_start,
            line_end=line_end,
            indent=indent,
            kind_hint="markdown_table",
            confidence=0.98,
            heading_level=None,
        )
    md_heading = re.match(r"^(#{1,6})\s+(.*)$", first_line)
    if source_format == "markdown" and md_heading:
        level = len(md_heading.group(1))
        title = md_heading.group(2).strip() or first_line.lstrip("#").strip()
        node_type: PageIndexNodeType = "HEADING"
        return _BlockSpan(
            start_char=start_char,
            end_char=end_char,
            text=stripped,
            node_type=node_type,
            title=title,
            line_start=line_start,
            line_end=line_end,
            indent=indent,
            kind_hint="heading_markdown",
            confidence=0.98,
            heading_level=level,
        )
    if heading_level is not None:
        node_type = "HEADING"
        return _BlockSpan(
            start_char=start_char,
            end_char=end_char,
            text=stripped,
            node_type=node_type,
            title=first_line.rstrip(":").strip(),
            line_start=line_start,
            line_end=line_end,
            indent=indent,
            kind_hint="heading_setext",
            confidence=0.97,
            heading_level=heading_level,
        )

    plain_heading = re.match(r"^(Section|Clause|Article|Definitions?)\b[:\s].*", first_line, flags=re.IGNORECASE)
    numbered_heading = None if numeric_heavy else re.match(r"^\d+(?:\.\d+){0,4}(?:[.)])?\s+\S+", first_line)
    legal_clause = re.match(r"^\s*\((?:\d+|[a-z]|[ivx]+)\)\s+\S+", first_line, flags=re.IGNORECASE)
    bullet_item = re.match(r"^\s*(?:[-*+]|\d{1,3}[.)])\s+\S+", first_line)
    all_caps_heading = (
        len(first_line.split()) <= 8
        and any(ch.isalpha() for ch in first_line)
        and first_line.upper() == first_line
        and not is_sentence_like
        and not numeric_heavy
        and (is_first or surrounded_by_blank_lines)
    )
    title_like_first_line = is_first and len(first_line.split()) <= 10 and not is_sentence_like and not numeric_heavy
    short_title_case = (
        len(first_line.split()) <= 12
        and not is_sentence_like
        and first_line[:1].isupper()
        and any(ch.isalpha() for ch in first_line)
        and not numeric_heavy
        and (is_first or surrounded_by_blank_lines)
    )

    if plain_heading or numbered_heading or all_caps_heading or title_like_first_line or short_title_case:
        if numbered_heading:
            level = max(2, first_line.count(".") + 2)
        elif title_like_first_line or (all_caps_heading and is_first):
            level = 1
        else:
            level = 2
        title = first_line.rstrip(":").strip()
        node_type: PageIndexNodeType = "HEADING"
        confidence = 0.95 if (plain_heading or numbered_heading) else 0.76
        if surrounded_by_blank_lines:
            confidence = min(0.98, confidence + 0.06)
        if numeric_heavy:
            confidence = max(0.4, confidence - 0.2)
        return _BlockSpan(
            start_char=start_char,
            end_char=end_char,
            text=stripped,
            node_type=node_type,
            title=title,
            line_start=line_start,
            line_end=line_end,
            indent=indent,
            kind_hint="heading_plain",
            confidence=confidence,
            heading_level=level,
        )
    if legal_clause or bullet_item:
        title = re.sub(
            r"^\s*(?:[-*+]|\d{1,3}[.)]|\((?:\d+|[a-z]|[ivx]+)\))\s+",
            "",
            first_line,
            flags=re.IGNORECASE,
        ).strip()
        return _BlockSpan(
            start_char=start_char,
            end_char=end_char,
            text=stripped,
            node_type="TERM",
            title=title or first_line,
            line_start=line_start,
            line_end=line_end,
            indent=indent,
            kind_hint="term_list_item",
            confidence=0.74,
            heading_level=None,
        )
    title = first_line[:120].rstrip()
    return _BlockSpan(
        start_char=start_char,
        end_char=end_char,
        text=stripped,
        node_type="PARAGRAPH",
        title=title,
        line_start=line_start,
        line_end=line_end,
        indent=indent,
        kind_hint="paragraph_table_like" if numeric_heavy else "paragraph",
        confidence=0.62 if numeric_heavy else (0.8 if word_count > 1 else 0.65),
        heading_level=None,
    )


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
