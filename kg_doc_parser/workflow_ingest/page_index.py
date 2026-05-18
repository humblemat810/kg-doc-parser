from __future__ import annotations

"""Reusable page-index document parsing for text and Markdown inputs.

The pipeline keeps a fast heuristic mode for deterministic structure extraction
and an Ollama-backed mode that reuses the existing parser provider boundary.
Both modes normalize raw content into page-aware source units and return a
semantic tree with hydrated spans. Ollama mode uses candidate extraction plus
flat block assignment, with optional excerpt refinement disabled by default.

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

import re
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from .adapters import build_authoritative_source_map, build_parser_input_dict, build_parser_source_map
from .models import GroundedSourceRecord, NormalizedPage, NormalizedSourceCollection, SourceUnit, WorkflowIngestInput
from .providers import WorkflowProviderSettings, build_chat_model_for_role
from kogwistar.fuzzy_offsets import FuzzySpanHit as _FuzzyHit, find_best_fuzzy_span
from .semantics import HydratedTextPointer, SemanticNode, compute_pointer_coverage, correct_and_validate_pointer

PageIndexMode = Literal["heuristic", "ollama"]
PageIndexSourceFormat = Literal["text", "markdown"]
PageIndexNodeType = Literal["SECTION", "SUBSECTION", "PARAGRAPH", "TERM"]


class PageIndexBlockSpec(BaseModel):
    """Recursive structural block emitted by the page-index parser."""

    title: str = Field(
        description="Short label for this block, usually the heading text or a concise paragraph label."
    )
    node_type: PageIndexNodeType = Field(
        description="One of SECTION, SUBSECTION, PARAGRAPH, or TERM. Preserve hierarchy depth, do not flatten it."
    )
    excerpt: str = Field(
        description="A short verbatim excerpt from the source page that grounds this block. Do not use the whole page text; keep it tight and exact."
    )
    child_nodes: list["PageIndexBlockSpec"] = Field(
        default_factory=list,
        description="Direct children only. Use nested children for substructure; do not duplicate the same excerpt across siblings.",
    )


PageIndexBlockSpec.model_rebuild()


@dataclass(slots=True)
class CandidateBlock:
    block_id: str
    page_number: int
    order: int
    start_char: int
    end_char: int
    line_start: int
    line_end: int
    indent: int
    kind_hint: str
    confidence: float
    text: str
    node_type_hint: PageIndexNodeType
    title_hint: str
    heading_level: int | None = None


class BlockAssignment(BaseModel):
    block_id: str = Field(description="Stable candidate block identifier.")
    parent_id: str | None = Field(
        default=None,
        description="Parent block identifier or null when the block is a root.",
    )
    node_type: PageIndexNodeType = Field(description="Assigned node type for the block.")
    title: str = Field(description="Assigned display title for the block.")


class BlockAssignmentBatch(BaseModel):
    assignments: list[BlockAssignment] = Field(default_factory=list)


class ExcerptRefinementSuggestion(BaseModel):
    path_id: str = Field(description="Stable tree path identifier for the block to refine.")
    excerpt: str = Field(description="Refined excerpt proposed for the block.")


class ExcerptRefinementBatch(BaseModel):
    suggestions: list[ExcerptRefinementSuggestion] = Field(default_factory=list)


@dataclass(slots=True)
class PageIndexValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]
    fallback_reason: str | None = None


@dataclass(slots=True)
class PageIndexParseResult:
    mode: PageIndexMode
    source_format: PageIndexSourceFormat
    workflow_input: WorkflowIngestInput
    authoritative_source_map: dict[str, GroundedSourceRecord]
    parser_input_dict: dict[str, Any]
    parser_source_map: dict[str, dict[str, Any]]
    semantic_tree: SemanticNode
    coverage: dict[str, Any]
    diagnostics: dict[str, Any]


@dataclass(slots=True)
class _PageUnit:
    page_number: int
    unit_id: str
    text: str


@dataclass(slots=True)
class _BlockSpan:
    start_char: int
    end_char: int
    text: str
    node_type: PageIndexNodeType
    title: str
    line_start: int = 1
    line_end: int = 1
    indent: int = 0
    kind_hint: str = "paragraph"
    confidence: float = 0.8
    heading_level: int | None = None


def _page_index_assignment_debug_payload(assignments: list[BlockAssignment]) -> list[dict[str, Any]]:
    return [
        {
            "block_id": assignment.block_id,
            "parent_id": assignment.parent_id,
            "node_type": assignment.node_type,
            "title": assignment.title,
        }
        for assignment in assignments
    ]


def _page_index_block_spec_debug_payload(block_specs: list[PageIndexBlockSpec]) -> list[dict[str, Any]]:
    def _walk(spec: PageIndexBlockSpec, *, path: str) -> dict[str, Any]:
        return {
            "path": path,
            "title": spec.title,
            "node_type": spec.node_type,
            "excerpt": spec.excerpt,
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
        "- use SECTION, SUBSECTION, PARAGRAPH, and TERM only\n"
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
    if re.search(r"\b\d+\b(?:\s+\b\d+\b){2,}", compact):
        return True
    return False


def _page_index_normalize_text(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _page_index_find_best_fuzzy_span(
    *,
    page_text: str,
    excerpt: str,
    origin_start: int,
    scan_band: int | None = None,
) -> _FuzzyHit | None:
    page_text_norm = _page_index_normalize_text(page_text)
    return find_best_fuzzy_span(
        content=page_text,
        excerpt=excerpt,
        origin_start=origin_start,
        scan_band=scan_band,
        candidate_filter=lambda candidate: _page_index_normalize_text(candidate) != page_text_norm,
    )


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
    md_heading = re.match(r"^(#{1,6})\s+(.*)$", first_line)
    if source_format == "markdown" and md_heading:
        level = len(md_heading.group(1))
        title = md_heading.group(2).strip() or first_line.lstrip("#").strip()
        node_type: PageIndexNodeType = "SECTION" if level <= 2 else "SUBSECTION"
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
        node_type = "SECTION" if heading_level <= 1 else "SUBSECTION"
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
        node_type = "SECTION" if level <= 2 else "SUBSECTION"
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
        if candidate.node_type_hint in {"SECTION", "SUBSECTION"}:
            level = candidate.heading_level or (1 if candidate.node_type_hint == "SECTION" else 2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            parent_id = stack[-1][1] if stack else None
            assignments.append(
                BlockAssignment(
                    block_id=candidate.block_id,
                    parent_id=parent_id,
                    node_type=candidate.node_type_hint,
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
        and assignment.node_type in {"SECTION", "SUBSECTION"}
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
        if assignment.node_type in {"SECTION", "SUBSECTION"} and not _candidate_heading_evidence(candidate):
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
        normalized = [" ".join(block.text.split()).strip().lower() for block in siblings if block.text.strip()]
        if len(normalized) != len(siblings):
            errors.append(f"parent {parent_id!r} has empty sibling text")
            continue
        if len(set(normalized)) == 1:
            errors.append(f"parent {parent_id!r} repeats the same sibling excerpt")
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
            node_type=assignment.node_type,
            excerpt=candidate.text,
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
            elif normalized_excerpt == normalized_page_text and len(candidates) > 1:
                errors.append(f"block {current_path} duplicates the whole page")
            elif _page_index_block_exceeds_excerpt_budget(spec, page_text):
                errors.append(f"block {current_path} excerpt is too broad")
            elif _page_index_block_is_too_generic(spec, page_text):
                errors.append(f"block {current_path} excerpt is too generic")
            normalized_sibling_excerpts.append(normalized_excerpt)
            _validate_siblings(spec.child_nodes, path=current_path)
        non_empty = [excerpt for excerpt in normalized_sibling_excerpts if excerpt]
        if len(non_empty) > 1 and len(set(non_empty)) == 1:
            errors.append(f"block siblings at {path or 'root'} repeat the same excerpt")

    _validate_siblings(block_specs, path="")
    return PageIndexValidationResult(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        fallback_reason=None if not errors else "structure_validation_failed",
    )


def _normalize_page_index_excerpt(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _page_index_path_id(path: tuple[int, ...]) -> str:
    return ".".join(str(index) for index in path)


def _iter_page_index_block_specs_with_paths(
    block_specs: list[PageIndexBlockSpec],
    *,
    path: tuple[int, ...] = (),
):
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


def _refine_page_index_block_excerpts(
    *,
    block_specs: list[PageIndexBlockSpec],
    page_text: str,
    page_number: int,
    unit_id: str,
    provider_settings: WorkflowProviderSettings,
    trace_log: Callable[[str], None] | None = None,
) -> tuple[list[PageIndexBlockSpec], dict[str, Any]]:
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
    chat = build_chat_model_for_role("parser", provider_settings)
    structured = chat.with_structured_output(ExcerptRefinementBatch, include_raw=True)
    from langchain_core.messages import HumanMessage, SystemMessage
    import json as _json

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
    try:
        payload = structured.invoke(
            [
                SystemMessage(content="You are a grounded excerpt refiner."),
                HumanMessage(content=prompt),
            ]
        )
        parsed = payload.get("parsed") if isinstance(payload, dict) else payload
        if parsed is None:
            error = payload.get("parsing_error") if isinstance(payload, dict) else None
            raise ValueError(f"excerpt refinement failed: {error!r}")
        batch = (
            parsed
            if isinstance(parsed, ExcerptRefinementBatch)
            else ExcerptRefinementBatch.model_validate(parsed)
        )
    except Exception as exc:
        diagnostics["refine_excerpts_fallback"] = True
        diagnostics["refine_excerpts_rejected"] = diagnostics["refine_excerpts_attempted"]
        diagnostics["refine_excerpts_error"] = f"{type(exc).__name__}: {exc}"
        if trace_log is not None:
            trace_log(
                f"page_index_refine_fallback page_number={page_number} unit_id={unit_id} error={type(exc).__name__}"
            )
        return block_specs, diagnostics

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
        try:
            _resolve_pointer(unit_id=unit_id, page_text=page_text, excerpt=proposed)
        except Exception:
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
        sibling_norms = [
            _normalize_page_index_excerpt(sibling.excerpt)
            for sibling_index, sibling in enumerate(parent_nodes)
            if sibling_index != path[-1] and _normalize_page_index_excerpt(sibling.excerpt)
        ]
        normalized_proposed = _normalize_page_index_excerpt(proposed)
        if normalized_proposed in ancestor_excerpts or normalized_proposed in sibling_norms:
            diagnostics["refine_excerpts_rejected"] += 1
            continue

        candidate_spec = deepcopy(original_spec)
        candidate_spec.excerpt = proposed
        if _page_index_block_exceeds_excerpt_budget(candidate_spec, page_text):
            diagnostics["refine_excerpts_rejected"] += 1
            continue
        if _page_index_block_is_too_generic(candidate_spec, page_text, ancestor_excerpts=ancestor_excerpts):
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


def _build_page_outline_from_candidates(
    *,
    page_text: str,
    page_number: int,
    source_format: PageIndexSourceFormat,
    assignment_mode: str,
    assignments: list[BlockAssignment],
    validation: PageIndexValidationResult,
) -> tuple[list[PageIndexBlockSpec], dict[str, Any]]:
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
) -> tuple[list[PageIndexBlockSpec], dict[str, Any]]:
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


def _llm_page_outline(
    *,
    page_text: str,
    page_number: int,
    source_format: PageIndexSourceFormat,
    provider_settings: WorkflowProviderSettings,
    trace_log: Callable[[str], None] | None = None,
) -> tuple[list[PageIndexBlockSpec], dict[str, Any]]:
    candidates = _extract_candidate_blocks(page_text, page_number=page_number, source_format=source_format)
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
    chat = build_chat_model_for_role("parser", provider_settings)
    if trace_log is not None:
        trace_log(f"page_index_llm_chat_build_done page_number={page_number}")
        trace_log(f"page_index_llm_structured_wrap_start page_number={page_number}")
    structured = chat.with_structured_output(BlockAssignmentBatch, include_raw=True)
    if trace_log is not None:
        trace_log(f"page_index_llm_structured_wrap_done page_number={page_number}")
    from langchain_core.messages import HumanMessage, SystemMessage

    def _invoke_attempt(
        *,
        retry_summary: str | None = None,
        retry_kind: str = "assignment",
        attempt_label: str,
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
        try:
            payload = structured.invoke(
                [
                    SystemMessage(content="You are a grounded page-index block assigner."),
                    HumanMessage(content=prompt),
                ]
            )
            if trace_log is not None:
                trace_log(f"page_index_llm_invoke_end page_number={page_number} attempt={attempt_label}")
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

        parsed = payload.get("parsed") if isinstance(payload, dict) else payload
        if parsed is None:
            error = payload.get("parsing_error") if isinstance(payload, dict) else None
            return None, f"parse_error: {error!r}"
        batch = parsed if isinstance(parsed, BlockAssignmentBatch) else BlockAssignmentBatch.model_validate(parsed)
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
    ) -> tuple[list[PageIndexBlockSpec], dict[str, Any]]:
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
            "assignment_validation_errors": assignment_retry_validation_errors or first_validation_errors,
            "structure_validation_errors": structure_validation_errors,
            "first_validation_errors": first_validation_errors,
            "retry_validation_errors": assignment_retry_validation_errors,
            "validation_errors": validation_errors,
            "validation_warnings": validation_warnings,
            "fallback_reason": fallback_reason,
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
    ) -> tuple[list[PageIndexBlockSpec], dict[str, Any]]:
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
            mode = "llm_flat_assignment"
            final_outcome = "first_pass_success"
            if assignment_retry_succeeded:
                mode = "llm_flat_assignment_retry"
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
                "assignment_validation_errors": assignment_retry_validation_errors or first_validation_errors,
                "structure_validation_errors": [],
                "first_validation_errors": first_validation_errors,
                "retry_validation_errors": assignment_retry_validation_errors,
                "validation_errors": [],
                "validation_warnings": validation_warnings + list(structure_validation.warnings),
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
        structure_batch, structure_error = _invoke_attempt(
            retry_summary=structure_summary,
            retry_kind="structure",
            attempt_label="structure_retry",
        )
        next_attempt_count = assignment_attempt_count + 1
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
            "assignment_mode": "llm_flat_assignment_structure_retry",
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
            "assignment_validation_errors": assignment_retry_validation_errors or first_validation_errors,
            "structure_validation_errors": structure_errors,
            "first_validation_errors": first_validation_errors,
            "retry_validation_errors": assignment_retry_validation_errors,
            "validation_errors": [],
            "validation_warnings": validation_warnings + list(repaired_structure_validation.warnings),
            "fallback_reason": None,
            "retry_prompt_summary": retry_prompt_summary,
            "structure_retry_prompt_summary": structure_summary,
        }

    first_batch, first_error = _invoke_attempt(attempt_label="first")
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

    retry_batch, retry_error = _invoke_attempt(retry_summary=retry_summary, attempt_label="retry")
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

    return _fallback(
        assignment_attempt_count=2,
        assignment_retry_used=True,
        first_validation_errors=first_validation_errors,
        assignment_retry_validation_errors=retry_validation_errors,
        structure_validation_errors=[],
        validation_warnings=retry_validation_warnings,
        fallback_reason="assignment_validation_failed" if first_validation_errors else "llm_unavailable_or_invalid",
        retry_prompt_summary=retry_summary,
    )


def _find_page_unit(authoritative_source_map: dict[str, GroundedSourceRecord], page_number: int) -> tuple[str, str]:
    for unit_id, record in authoritative_source_map.items():
        if record.page_number == page_number and record.participates_in_semantic_text:
            return unit_id, record.text
    raise ValueError(f"missing text page for page number {page_number}")


def _resolve_pointer(
    *,
    unit_id: str,
    page_text: str,
    excerpt: str,
    start_at: int = 0,
    repair_stats: dict[str, int] | None = None,
    trace_log: Callable[[str], None] | None = None,
) -> HydratedTextPointer:
    needle = excerpt.strip() or excerpt or page_text.strip()
    candidate = HydratedTextPointer(
        source_cluster_id=unit_id,
        start_char=max(0, start_at),
        end_char=max(0, start_at + max(len(needle), 1) - 1),
        verbatim_text=needle,
    )
    resolved = correct_and_validate_pointer(candidate, {unit_id: {"text": page_text}})
    if resolved is None:
        fuzzy_hit = _page_index_find_best_fuzzy_span(
            page_text=page_text,
            excerpt=needle,
            origin_start=start_at,
        )
        if fuzzy_hit is None:
            if repair_stats is not None:
                repair_stats["pointer_fuzzy_failures"] = int(repair_stats.get("pointer_fuzzy_failures", 0)) + 1
            raise ValueError(f"unable to resolve excerpt against page text for {unit_id!r}: {needle!r}")
        fuzzy_candidate = HydratedTextPointer(
            source_cluster_id=unit_id,
            start_char=fuzzy_hit.start,
            end_char=fuzzy_hit.end - 1,
            verbatim_text=page_text[fuzzy_hit.start : fuzzy_hit.end],
        )
        resolved = correct_and_validate_pointer(fuzzy_candidate, {unit_id: {"text": page_text}})
        if resolved is None:
            if repair_stats is not None:
                repair_stats["pointer_fuzzy_failures"] = int(repair_stats.get("pointer_fuzzy_failures", 0)) + 1
            raise ValueError(f"unable to resolve excerpt against page text for {unit_id!r}: {needle!r}")
        if repair_stats is not None:
            repair_stats["pointer_fuzzy_repairs"] = int(repair_stats.get("pointer_fuzzy_repairs", 0)) + 1
        if trace_log is not None:
            trace_log(
                "page_index_pointer_fuzzy_repair "
                f"unit_id={unit_id} start={fuzzy_hit.start} end={fuzzy_hit.end} score={fuzzy_hit.score:.2f}"
            )
    return resolved


def _make_semantic_node(
    *,
    title: str,
    node_type: str,
    parent_id: str | None,
    level_from_root: int,
    pointers: list[HydratedTextPointer],
) -> SemanticNode:
    return SemanticNode(
        title=title,
        node_type=node_type,
        parent_id=parent_id,
        level_from_root=level_from_root,
        total_content_pointers=pointers,
        child_nodes=[],
    )


def _materialize_block_tree(
    *,
    block_specs: list[PageIndexBlockSpec],
    page_text: str,
    unit_id: str,
    parent_id: str,
    level_from_root: int,
    start_at: int = 0,
    repair_stats: dict[str, int] | None = None,
    trace_log: Callable[[str], None] | None = None,
) -> tuple[list[SemanticNode], int]:
    nodes: list[SemanticNode] = []
    cursor = start_at
    for spec in block_specs:
        pointer = _resolve_pointer(
            unit_id=unit_id,
            page_text=page_text,
            excerpt=spec.excerpt,
            start_at=cursor,
            repair_stats=repair_stats,
            trace_log=trace_log,
        )
        cursor = pointer.end_char + 1
        node = _make_semantic_node(
            title=spec.title,
            node_type=spec.node_type,
            parent_id=parent_id,
            level_from_root=level_from_root,
            pointers=[pointer],
        )
        child_nodes, cursor = _materialize_block_tree(
            block_specs=spec.child_nodes,
            page_text=page_text,
            unit_id=unit_id,
            parent_id=node.node_id or parent_id,
            level_from_root=level_from_root + 1,
            start_at=cursor,
            repair_stats=repair_stats,
            trace_log=trace_log,
        )
        node.child_nodes.extend(child_nodes)
        nodes.append(node)
    return nodes, cursor


def _page_index_block_exceeds_excerpt_budget(spec: PageIndexBlockSpec, page_text: str) -> bool:
    page_text_norm = page_text.strip()
    excerpt_norm = spec.excerpt.strip()
    if not excerpt_norm:
        return True
    if excerpt_norm == page_text_norm:
        return True
    if len(excerpt_norm) > max(240, int(len(page_text_norm) * 0.75)):
        return True
    return any(_page_index_block_exceeds_excerpt_budget(child, page_text) for child in spec.child_nodes)


def _page_index_block_is_too_generic(
    spec: PageIndexBlockSpec,
    page_text: str,
    *,
    ancestor_excerpts: tuple[str, ...] = (),
) -> bool:
    def _normalize(text: str) -> str:
        return " ".join(text.split()).strip().lower()

    page_text_norm = _normalize(page_text)
    excerpt_norm = _normalize(spec.excerpt)
    if not excerpt_norm:
        return True
    if excerpt_norm == page_text_norm:
        return True
    if excerpt_norm in ancestor_excerpts:
        return True
    if len(spec.child_nodes) > 1:
        child_norms = [_normalize(child.excerpt) for child in spec.child_nodes if _normalize(child.excerpt)]
        if len(child_norms) != len(spec.child_nodes):
            return True
        if len(set(child_norms)) == 1:
            return True
    if len(excerpt_norm) > max(240, int(len(page_text_norm) * 0.75)):
        return True
    next_ancestors = ancestor_excerpts + (excerpt_norm,)
    return any(
        _page_index_block_is_too_generic(child, page_text, ancestor_excerpts=next_ancestors)
        for child in spec.child_nodes
    )


def parse_page_index_document(
    *,
    document_id: str,
    title: str,
    raw_text: str,
    source_format: PageIndexSourceFormat = "text",
    mode: PageIndexMode = "heuristic",
    provider_settings: WorkflowProviderSettings | None = None,
    trace_log: Callable[[str], None] | None = None,
    refine_excerpts: bool = False,
) -> PageIndexParseResult:
    """Parse a plain text or Markdown document into a page-index semantic tree."""

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

    root_pointers: list[HydratedTextPointer] = []
    page_nodes: list[SemanticNode] = []
    page_diagnostics: list[dict[str, Any]] = []
    for page_number, (unit_id, record) in enumerate(page_units, start=1):
        if trace_log is not None:
            trace_log(
                f"page_index_page_start page_number={page_number} unit_id={unit_id} mode={mode}"
            )
        root_pointers.append(
            HydratedTextPointer(
                source_cluster_id=unit_id,
                start_char=0,
                end_char=max(0, len(record.text) - 1),
                verbatim_text=record.text,
            )
        )
        page_text = record.text
        if mode == "heuristic":
            block_specs, page_diagnostics_item = _heuristic_page_outline(
                page_text,
                page_number=page_number,
                source_format=source_format,
            )
        elif mode == "ollama":
            settings = provider_settings or WorkflowProviderSettings.from_env()
            if settings.parser.provider != "ollama":
                raise ValueError("ollama mode requires KG_DOC_PARSER_PROVIDER=ollama")
            if trace_log is not None:
                trace_log(f"page_index_llm_prepare page_number={page_number} provider=ollama")
            block_specs, page_diagnostics_item = _llm_page_outline(
                page_text=page_text,
                page_number=page_number,
                source_format=source_format,
                provider_settings=settings,
                trace_log=trace_log,
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
        if mode == "ollama" and refine_excerpts:
            block_specs, page_refinement_diagnostics = _refine_page_index_block_excerpts(
                block_specs=block_specs,
                page_text=page_text,
                page_number=page_number,
                unit_id=unit_id,
                provider_settings=settings,
                trace_log=trace_log,
            )
        page_diagnostics_item = dict(page_diagnostics_item)
        page_diagnostics_item.update(page_refinement_diagnostics)
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
            parent_id=None,
            level_from_root=1,
            pointers=[
                HydratedTextPointer(
                    source_cluster_id=unit_id,
                    start_char=0,
                    end_char=max(0, len(page_text) - 1),
                    verbatim_text=page_text,
                )
            ],
        )
        child_nodes, _ = _materialize_block_tree(
            block_specs=block_specs,
            page_text=page_text,
            unit_id=unit_id,
            parent_id=page_node.node_id or document_id,
            level_from_root=2,
            repair_stats=page_repair_stats,
            trace_log=trace_log,
        )
        page_node.child_nodes.extend(child_nodes)
        page_nodes.append(page_node)
        page_diagnostics_item.update(page_repair_stats)
        if trace_log is not None:
            trace_log(
                f"page_index_page_end page_number={page_number} unit_id={unit_id} child_count={len(child_nodes)}"
            )

    semantic_tree = SemanticNode(
        title=title,
        node_type="DOCUMENT_ROOT",
        parent_id=None,
        level_from_root=0,
        total_content_pointers=root_pointers,
        child_nodes=page_nodes,
    )
    coverage = compute_pointer_coverage(semantic_tree, parser_source_map)
    validation_errors = [error for item in page_diagnostics for error in item.get("validation_errors", [])]
    validation_warnings = [warning for item in page_diagnostics for warning in item.get("validation_warnings", [])]
    first_validation_errors = [error for item in page_diagnostics for error in item.get("first_validation_errors", [])]
    retry_validation_errors = [error for item in page_diagnostics for error in item.get("retry_validation_errors", [])]
    assignment_validation_errors = [error for item in page_diagnostics for error in item.get("assignment_validation_errors", [])]
    structure_validation_errors = [error for item in page_diagnostics for error in item.get("structure_validation_errors", [])]
    fallback_reasons = [item.get("fallback_reason") for item in page_diagnostics if item.get("fallback_reason")]
    candidate_count = sum(int(item.get("candidate_count", 0) or 0) for item in page_diagnostics)
    assignment_count = sum(int(item.get("assignment_count", 0) or 0) for item in page_diagnostics)
    assignment_attempt_count = sum(int(item.get("assignment_attempt_count", 1) or 1) for item in page_diagnostics)
    assignment_retry_used = any(bool(item.get("assignment_retry_used")) for item in page_diagnostics)
    assignment_retry_succeeded = any(bool(item.get("assignment_retry_succeeded")) for item in page_diagnostics)
    structure_retry_used = any(bool(item.get("structure_retry_used")) for item in page_diagnostics)
    structure_retry_succeeded = any(bool(item.get("structure_retry_succeeded")) for item in page_diagnostics)
    retry_used = any(bool(item.get("retry_used")) for item in page_diagnostics)
    retry_succeeded = any(bool(item.get("retry_succeeded")) for item in page_diagnostics)
    assignment_modes = {str(item.get("assignment_mode") or "") for item in page_diagnostics if item.get("assignment_mode")}
    refinement_enabled = any(bool(item.get("refine_excerpts_enabled")) for item in page_diagnostics)
    refinement_attempted = sum(int(item.get("refine_excerpts_attempted", 0) or 0) for item in page_diagnostics)
    refinement_accepted = sum(int(item.get("refine_excerpts_accepted", 0) or 0) for item in page_diagnostics)
    refinement_rejected = sum(int(item.get("refine_excerpts_rejected", 0) or 0) for item in page_diagnostics)
    refinement_fallback_pages = sum(1 for item in page_diagnostics if item.get("refine_excerpts_fallback"))
    pointer_fuzzy_repairs = sum(int(item.get("pointer_fuzzy_repairs", 0) or 0) for item in page_diagnostics)
    pointer_fuzzy_failures = sum(int(item.get("pointer_fuzzy_failures", 0) or 0) for item in page_diagnostics)
    if mode == "heuristic":
        overall_assignment_mode = "heuristic_deterministic"
    elif fallback_reasons:
        overall_assignment_mode = "deterministic_fallback"
    elif structure_retry_succeeded:
        overall_assignment_mode = "llm_flat_assignment_structure_retry"
    elif assignment_retry_succeeded:
        overall_assignment_mode = "llm_flat_assignment_retry"
    elif retry_used:
        overall_assignment_mode = "deterministic_fallback"
    else:
        overall_assignment_mode = "llm_flat_assignment" if assignment_modes == {"llm_flat_assignment"} else "deterministic_fallback"
    if fallback_reasons:
        final_outcome = "deterministic_fallback"
    elif structure_retry_succeeded:
        final_outcome = "structure_retry_success"
    elif assignment_retry_succeeded:
        final_outcome = "assignment_retry_success"
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
