from __future__ import annotations

"""Reusable page-index document parsing for text and Markdown inputs.

The pipeline keeps a fast heuristic mode for deterministic structure extraction
and an Ollama-backed mode that reuses the existing parser provider boundary.
Both modes normalize raw content into page-aware source units and return a
semantic tree with hydrated spans.

Example CLI
-----------
Heuristic mode:

    .venv\\Scripts\\python.exe -m pytest \
tests/test_workflow_ingest_page_index_pipeline.py::test_page_index_heuristic_parses_text_and_markdown[text] -q

    .venv\\Scripts\\python.exe -m pytest \
tests/test_workflow_ingest_page_index_pipeline.py::test_page_index_heuristic_parses_text_and_markdown[markdown] -q

Ollama mode with a local Gemma parser model:

    set KG_DOC_PARSER_PROVIDER=ollama
    set KG_DOC_PARSER_MODEL=gemma4
    set KG_DOC_PARSER_BASE_URL=http://127.0.0.1:11434
    .venv\\Scripts\\python.exe -m pytest \
tests/test_workflow_ingest_page_index_pipeline.py::test_page_index_ollama_smoke_parses_text_and_markdown[text] -q

    set KG_DOC_PARSER_PROVIDER=ollama
    set KG_DOC_PARSER_MODEL=gemma4
    set KG_DOC_PARSER_BASE_URL=http://127.0.0.1:11434
    .venv\\Scripts\\python.exe -m pytest \
tests/test_workflow_ingest_page_index_pipeline.py::test_page_index_ollama_smoke_parses_text_and_markdown[markdown] -q
"""

import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from .adapters import build_authoritative_source_map, build_parser_input_dict, build_parser_source_map
from .models import GroundedSourceRecord, NormalizedPage, NormalizedSourceCollection, SourceUnit, WorkflowIngestInput
from .providers import WorkflowProviderSettings, build_chat_model_for_role
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


def _split_pages(raw_text: str) -> list[str]:
    """Split a logical document into page-sized chunks."""

    pages = re.split(r"\f|^\s*--- PAGE BREAK ---\s*$", raw_text, flags=re.MULTILINE)
    return [page.strip("\n") for page in pages if page.strip()]


def _is_setext_underline(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and bool(re.fullmatch(r"=+|-+", stripped))


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
    line_start: int = 1,
    line_end: int = 1,
    indent: int = 0,
    heading_level: int | None = None,
) -> _BlockSpan:
    stripped = text.strip()
    first_line = stripped.splitlines()[0].strip()
    word_count = len(first_line.split())
    is_sentence_like = first_line.endswith((".", "!", "?"))
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
    numbered_heading = re.match(r"^\d+(?:\.\d+){0,4}(?:[.)])?\s+\S+", first_line)
    legal_clause = re.match(r"^\s*\((?:\d+|[a-z]|[ivx]+)\)\s+\S+", first_line, flags=re.IGNORECASE)
    bullet_item = re.match(r"^\s*(?:[-*+•]|\d{1,3}[.)])\s+\S+", first_line)
    all_caps_heading = (
        len(first_line.split()) <= 8
        and any(ch.isalpha() for ch in first_line)
        and first_line.upper() == first_line
        and not is_sentence_like
    )
    title_like_first_line = is_first and len(first_line.split()) <= 10 and not is_sentence_like
    short_title_case = (
        len(first_line.split()) <= 12
        and not is_sentence_like
        and first_line[:1].isupper()
        and any(ch.isalpha() for ch in first_line)
    )

    if plain_heading or numbered_heading or all_caps_heading or title_like_first_line or short_title_case:
        if title_like_first_line or (all_caps_heading and is_first):
            level = 1
        elif numbered_heading:
            level = max(2, first_line.count(".") + 2)
        else:
            level = 2
        title = first_line.rstrip(":").strip()
        node_type = "SECTION" if level <= 2 else "SUBSECTION"
        confidence = 0.95 if (plain_heading or numbered_heading) else 0.76
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
            r"^\s*(?:[-*+•]|\d{1,3}[.)]|\((?:\d+|[a-z]|[ivx]+)\))\s+",
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
        kind_hint="paragraph",
        confidence=0.8 if word_count > 1 else 0.65,
        heading_level=None,
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
    line_start: int = 1,
    line_end: int = 1,
    indent: int = 0,
    heading_level: int | None = None,
) -> _BlockSpan:
    stripped = text.strip()
    first_line = stripped.splitlines()[0].strip()
    word_count = len(first_line.split())
    is_sentence_like = first_line.endswith((".", "!", "?"))
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
    numbered_heading = re.match(r"^\d+(?:\.\d+){0,4}(?:[.)])?\s+\S+", first_line)
    legal_clause = re.match(r"^\s*\((?:\d+|[a-z]|[ivx]+)\)\s+\S+", first_line, flags=re.IGNORECASE)
    bullet_item = re.match(r"^\s*(?:[-*+•]|\d{1,3}[.)])\s+\S+", first_line)
    all_caps_heading = (
        len(first_line.split()) <= 8
        and any(ch.isalpha() for ch in first_line)
        and first_line.upper() == first_line
        and not is_sentence_like
    )
    title_like_first_line = is_first and len(first_line.split()) <= 10 and not is_sentence_like
    short_title_case = (
        len(first_line.split()) <= 12
        and not is_sentence_like
        and first_line[:1].isupper()
        and any(ch.isalpha() for ch in first_line)
    )

    if plain_heading or numbered_heading or all_caps_heading or title_like_first_line or short_title_case:
        if title_like_first_line or (all_caps_heading and is_first):
            level = 1
        elif numbered_heading:
            level = max(2, first_line.count(".") + 2)
        else:
            level = 2
        title = first_line.rstrip(":").strip()
        node_type = "SECTION" if level <= 2 else "SUBSECTION"
        confidence = 0.95 if (plain_heading or numbered_heading) else 0.76
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
            r"^\s*(?:[-*+•]|\d{1,3}[.)]|\((?:\d+|[a-z]|[ivx]+)\))\s+",
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
        kind_hint="paragraph",
        confidence=0.8 if word_count > 1 else 0.65,
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
                    title=candidate.title_hint or candidate.title_hint,
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
    spec_by_id: dict[str, PageIndexBlockSpec] = {}
    roots: list[PageIndexBlockSpec] = []
    for assignment in assignments:
        candidate = candidate_by_id[assignment.block_id]
        spec = PageIndexBlockSpec(
            title=assignment.title or candidate.title_hint,
            node_type=assignment.node_type,
            excerpt=candidate.text,
        )
        spec_by_id[assignment.block_id] = spec
    for assignment in assignments:
        spec = spec_by_id[assignment.block_id]
        parent_id = assignment.parent_id
        if parent_id and parent_id in spec_by_id:
            spec_by_id[parent_id].child_nodes.append(spec)
        else:
            roots.append(spec)
    return roots


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
    import json as _json

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
        "The recursive tree is built deterministically after validation.\n"
        f"Source format: {source_format}\n"
        f"Page number: {page_number}\n"
        f"Candidates: {_json.dumps([asdict(candidate) for candidate in candidates], ensure_ascii=False, sort_keys=True)}"
    )
    if trace_log is not None:
        trace_log(f"page_index_llm_prompt_ready page_number={page_number}")
        trace_log(f"page_index_llm_invoke_start page_number={page_number} source_format={source_format}")
    try:
        payload = structured.invoke(
            [
                SystemMessage(content="You are a grounded page-index block assigner."),
                HumanMessage(content=prompt),
            ]
        )
        if trace_log is not None:
            trace_log(f"page_index_llm_invoke_end page_number={page_number}")
        parsed = payload.get("parsed") if isinstance(payload, dict) else payload
        if parsed is None:
            error = payload.get("parsing_error") if isinstance(payload, dict) else None
            raise ValueError(f"ollama page index parse failed: {error!r}")
        batch = parsed if isinstance(parsed, BlockAssignmentBatch) else BlockAssignmentBatch.model_validate(parsed)
        validation = _validate_block_assignments(candidates, batch.assignments, page_text=page_text)
        if validation.valid:
            return _assemble_page_index_blocks(candidates=candidates, assignments=batch.assignments), {
                "assignment_mode": "ollama_flat_assignment",
                "candidate_count": len(candidates),
                "assignment_count": len(batch.assignments),
                "validation_errors": list(validation.errors),
                "validation_warnings": list(validation.warnings),
                "fallback_reason": None,
            }
        if trace_log is not None:
            trace_log(
                f"page_index_llm_validation_failed page_number={page_number} errors={len(validation.errors)}"
            )
        return _assemble_page_index_blocks(
            candidates=candidates,
            assignments=_deterministic_block_assignments(candidates),
        ), {
            "assignment_mode": "deterministic_fallback",
            "candidate_count": len(candidates),
            "assignment_count": len(batch.assignments),
            "validation_errors": list(validation.errors),
            "validation_warnings": list(validation.warnings),
            "fallback_reason": validation.fallback_reason,
        }
    except Exception as exc:
        if trace_log is not None:
            trace_log(f"page_index_llm_assignment_fallback page_number={page_number} error={type(exc).__name__}")
        return _assemble_page_index_blocks(
            candidates=candidates,
            assignments=_deterministic_block_assignments(candidates),
        ), {
            "assignment_mode": "deterministic_fallback",
            "candidate_count": len(candidates),
            "assignment_count": len(candidates),
            "validation_errors": [f"{type(exc).__name__}: {exc}"],
            "validation_warnings": [],
            "fallback_reason": "llm_unavailable_or_invalid",
        }


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
        raise ValueError(f"unable to resolve excerpt against page text for {unit_id!r}: {needle!r}")
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
) -> tuple[list[SemanticNode], int]:
    nodes: list[SemanticNode] = []
    cursor = start_at
    for spec in block_specs:
        pointer = _resolve_pointer(unit_id=unit_id, page_text=page_text, excerpt=spec.excerpt, start_at=cursor)
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
        page_diagnostics_item = dict(page_diagnostics_item)
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
        )
        page_node.child_nodes.extend(child_nodes)
        page_nodes.append(page_node)
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
    fallback_reasons = [item.get("fallback_reason") for item in page_diagnostics if item.get("fallback_reason")]
    candidate_count = sum(int(item.get("candidate_count", 0) or 0) for item in page_diagnostics)
    assignment_count = sum(int(item.get("assignment_count", 0) or 0) for item in page_diagnostics)
    assignment_modes = {str(item.get("assignment_mode") or "") for item in page_diagnostics if item.get("assignment_mode")}
    if mode == "heuristic":
        overall_assignment_mode = "heuristic_deterministic"
    else:
        overall_assignment_mode = "ollama_flat_assignment" if assignment_modes == {"ollama_flat_assignment"} else "deterministic_fallback"
    diagnostics = {
        "assignment_mode": overall_assignment_mode,
        "fallback_reason": fallback_reasons[0] if fallback_reasons else None,
        "candidate_count": candidate_count,
        "assignment_count": assignment_count,
        "validation_errors": validation_errors,
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
