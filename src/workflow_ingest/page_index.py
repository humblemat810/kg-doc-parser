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
from dataclasses import dataclass
from typing import Any, Literal

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

    title: str
    node_type: PageIndexNodeType
    excerpt: str
    child_nodes: list["PageIndexBlockSpec"] = Field(default_factory=list)


PageIndexBlockSpec.model_rebuild()


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
    heading_level: int | None = None


def _split_pages(raw_text: str) -> list[str]:
    """Split a logical document into page-sized chunks."""

    pages = re.split(r"\f|^\s*--- PAGE BREAK ---\s*$", raw_text, flags=re.MULTILINE)
    return [page.strip("\n") for page in pages if page.strip()]


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
    cursor = 0
    paragraph_start: int | None = None
    paragraph_end = 0
    saw_nonblank = False

    def _flush_paragraph() -> None:
        nonlocal paragraph_start, paragraph_end
        if paragraph_start is None:
            return
        raw = page_text[paragraph_start:paragraph_end]
        trimmed = raw.strip()
        if trimmed:
            relative_start = raw.find(trimmed)
            start_char = paragraph_start + relative_start
            end_char = start_char + len(trimmed) - 1
            blocks.append(_classify_block(trimmed, start_char, end_char, source_format=source_format, is_first=False))
        paragraph_start = None

    for line in page_text.splitlines(keepends=True):
        line_start = cursor
        line_end = cursor + len(line)
        stripped = line.strip()
        if not stripped:
            _flush_paragraph()
            cursor = line_end
            continue
        line_block = _classify_block(
            stripped,
            line_start,
            line_end - 1,
            source_format=source_format,
            is_first=not saw_nonblank,
        )
        saw_nonblank = True
        if line_block.node_type != "PARAGRAPH":
            _flush_paragraph()
            blocks.append(line_block)
        else:
            if paragraph_start is None:
                paragraph_start = line_start
            paragraph_end = line_end
        cursor = line_end
    _flush_paragraph()
    return blocks


def _classify_block(text: str, start_char: int, end_char: int, *, source_format: PageIndexSourceFormat = "text", is_first: bool = False) -> _BlockSpan:
    stripped = text.strip()
    first_line = stripped.splitlines()[0].strip()
    md_heading = re.match(r"^(#{1,6})\s+(.*)$", first_line)
    if source_format == "markdown" and md_heading:
        level = len(md_heading.group(1))
        title = md_heading.group(2).strip() or first_line.lstrip("#").strip()
        node_type: PageIndexNodeType = "SECTION" if level <= 2 else "SUBSECTION"
        return _BlockSpan(start_char=start_char, end_char=end_char, text=stripped, node_type=node_type, title=title, heading_level=level)

    plain_heading = re.match(r"^(Section|Clause|Article|Definitions?)\b[:\s].*", first_line, flags=re.IGNORECASE)
    numbered_section = re.match(r"^\d+(?:\.\d+)+\s+\S+", first_line)
    term_like = re.match(r"^\s*(?:\d+[.)]|[-*+])\s+\S+", first_line)
    all_caps_heading = (
        len(first_line.split()) <= 8
        and any(ch.isalpha() for ch in first_line)
        and first_line.upper() == first_line
    )
    title_like_first_line = is_first and len(first_line.split()) <= 6 and not first_line.endswith((".", "!", "?"))

    if plain_heading or numbered_section or all_caps_heading or title_like_first_line:
        if title_like_first_line or (all_caps_heading and is_first):
            level = 1
        elif numbered_section:
            level = max(2, first_line.count(".") + 2)
        else:
            level = 2
        title = first_line.rstrip(":").strip()
        node_type = "SECTION" if level <= 2 else "SUBSECTION"
        return _BlockSpan(start_char=start_char, end_char=end_char, text=stripped, node_type=node_type, title=title, heading_level=level)
    if term_like:
        title = re.sub(r"^\s*(?:\d+[.)]|[-*+])\s+", "", first_line).strip()
        return _BlockSpan(start_char=start_char, end_char=end_char, text=stripped, node_type="TERM", title=title or first_line, heading_level=None)
    title = first_line[:80].rstrip()
    return _BlockSpan(start_char=start_char, end_char=end_char, text=stripped, node_type="PARAGRAPH", title=title, heading_level=None)


def _heuristic_page_outline(page_text: str, *, page_number: int, source_format: PageIndexSourceFormat) -> list[PageIndexBlockSpec]:
    blocks = _split_page_blocks(page_text, source_format=source_format)
    stack: list[tuple[int, PageIndexBlockSpec]] = []
    roots: list[PageIndexBlockSpec] = []
    for index, block in enumerate(blocks):
        classified = _classify_block(block.text, block.start_char, block.end_char, source_format=source_format, is_first=index == 0)
        spec = PageIndexBlockSpec(
            title=classified.title,
            node_type=classified.node_type,
            excerpt=classified.text,
        )
        if classified.node_type in {"SECTION", "SUBSECTION"}:
            while stack and stack[-1][0] >= int(classified.heading_level or 1):
                stack.pop()
            if stack:
                stack[-1][1].child_nodes.append(spec)
            else:
                roots.append(spec)
            stack.append((int(classified.heading_level or 1), spec))
            continue
        parent = stack[-1][1] if stack else None
        if parent is None:
            roots.append(spec)
        else:
            parent.child_nodes.append(spec)
    return roots


def _llm_page_outline(
    *,
    page_text: str,
    page_number: int,
    source_format: PageIndexSourceFormat,
    provider_settings: WorkflowProviderSettings,
) -> list[PageIndexBlockSpec]:
    chat = build_chat_model_for_role("parser", provider_settings)
    structured = chat.with_structured_output(PageIndexBlockSpec, include_raw=True)
    from langchain_core.messages import HumanMessage, SystemMessage

    prompt = (
        "You are a document parser for a page-index pipeline.\n"
        "Return a hierarchy of section, subsection, paragraph, and term blocks.\n"
        "Prefer a deeper tree when the document contains nested numbering, subclauses, or subheadings.\n"
        "Do not flatten nested structure into one section with many children if the text supports a parent/child relationship.\n"
        "Treat headings and numbered clauses as hierarchy cues: page title > section > subsection > paragraph > term.\n"
        "Keep paragraphs grouped under the nearest heading, and keep terms nested under the clause or subsection they belong to.\n"
        "Use verbatim excerpts from the supplied page text.\n"
        f"Source format: {source_format}\n"
        f"Page number: {page_number}\n"
        "Do not invent content. Keep excerpts short but exact."
    )
    payload = structured.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=page_text),
        ]
    )
    parsed = payload.get("parsed") if isinstance(payload, dict) else payload
    if parsed is None:
        error = payload.get("parsing_error") if isinstance(payload, dict) else None
        raise ValueError(f"ollama page index parse failed: {error!r}")
    if isinstance(parsed, PageIndexBlockSpec):
        return [parsed]
    if isinstance(parsed, list):
        return [PageIndexBlockSpec.model_validate(item) for item in parsed]
    if isinstance(parsed, dict) and "child_nodes" in parsed:
        return [PageIndexBlockSpec.model_validate(parsed)]
    raise TypeError(f"unexpected ollama page index payload: {type(parsed)!r}")


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


def parse_page_index_document(
    *,
    document_id: str,
    title: str,
    raw_text: str,
    source_format: PageIndexSourceFormat = "text",
    mode: PageIndexMode = "heuristic",
    provider_settings: WorkflowProviderSettings | None = None,
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
    for page_number, (unit_id, record) in enumerate(page_units, start=1):
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
            block_specs = _heuristic_page_outline(page_text, page_number=page_number, source_format=source_format)
        elif mode == "ollama":
            settings = provider_settings or WorkflowProviderSettings.from_env()
            if settings.parser.provider != "ollama":
                raise ValueError("ollama mode requires KG_DOC_PARSER_PROVIDER=ollama")
            block_specs = _llm_page_outline(
                page_text=page_text,
                page_number=page_number,
                source_format=source_format,
                provider_settings=settings,
            )
        else:  # pragma: no cover - Literal guards this in type-checked code.
            raise ValueError(f"unsupported page index mode: {mode}")

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

    semantic_tree = SemanticNode(
        title=title,
        node_type="DOCUMENT_ROOT",
        parent_id=None,
        level_from_root=0,
        total_content_pointers=root_pointers,
        child_nodes=page_nodes,
    )
    coverage = compute_pointer_coverage(semantic_tree, parser_source_map)
    return PageIndexParseResult(
        mode=mode,
        source_format=source_format,
        workflow_input=workflow_input,
        authoritative_source_map=authoritative_source_map,
        parser_input_dict=parser_input_dict,
        parser_source_map=parser_source_map,
        semantic_tree=semantic_tree,
        coverage=coverage,
    )
