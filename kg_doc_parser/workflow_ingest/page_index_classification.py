"""Classify grounded text blocks into PageIndex structural types."""

from __future__ import annotations

import re

from .page_index_contracts import (
    BlockSpan as _BlockSpan,
    PageIndexNodeType,
    PageIndexSourceFormat,
)


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


def _is_markdown_table_block(
    text: str, *, source_format: PageIndexSourceFormat
) -> bool:
    """Recognize a bounded Markdown table as one explicit atomic block."""

    if source_format != "markdown":
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2 or not all(_is_markdown_table_row(line) for line in lines):
        return False
    separator = lines[1].strip("|").split("|")
    return len(separator) >= 2 and all(
        bool(re.fullmatch(r"\s*:?-{3,}:?\s*", cell)) for cell in separator
    )


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

    plain_heading = re.match(
        r"^(Section|Clause|Article|Definitions?)\b[:\s].*",
        first_line,
        flags=re.IGNORECASE,
    )
    numbered_heading = (
        None
        if numeric_heavy
        else re.match(r"^\d+(?:\.\d+){0,4}(?:[.)])?\s+\S+", first_line)
    )
    legal_clause = re.match(
        r"^\s*\((?:\d+|[a-z]|[ivx]+)\)\s+\S+", first_line, flags=re.IGNORECASE
    )
    bullet_item = re.match(r"^\s*(?:[-*+]|\d{1,3}[.)])\s+\S+", first_line)
    all_caps_heading = (
        len(first_line.split()) <= 8
        and any(ch.isalpha() for ch in first_line)
        and first_line.upper() == first_line
        and not is_sentence_like
        and not numeric_heavy
        and (is_first or surrounded_by_blank_lines)
    )
    title_like_first_line = (
        is_first
        and len(first_line.split()) <= 10
        and not is_sentence_like
        and not numeric_heavy
    )
    short_title_case = (
        len(first_line.split()) <= 12
        and not is_sentence_like
        and first_line[:1].isupper()
        and any(ch.isalpha() for ch in first_line)
        and not numeric_heavy
        and (is_first or surrounded_by_blank_lines)
    )

    if (
        plain_heading
        or numbered_heading
        or all_caps_heading
        or title_like_first_line
        or short_title_case
    ):
        if numbered_heading:
            level = max(2, first_line.count(".") + 2)
        elif title_like_first_line or (all_caps_heading and is_first):
            level = 1
        else:
            level = 2
        title = first_line.rstrip(":").strip()
        node_type = "HEADING"
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
