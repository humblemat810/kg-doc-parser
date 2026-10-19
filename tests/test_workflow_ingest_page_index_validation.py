"""Page-index source assignment and structural validation tests."""

from __future__ import annotations

import pytest

import kg_doc_parser.workflow_ingest.page_index_candidates as page_index_candidates_module
from kg_doc_parser.workflow_ingest import (
    BlockAssignment,
    CandidateBlock,
)
from kg_doc_parser.workflow_ingest.page_index_contracts import PageIndexBlockSpec

pytestmark = [pytest.mark.workflow]

def test_page_index_candidate_extraction_and_validation() -> None:
    raw_text = "# Root\n\nIntro paragraph.\n\n## Child\n\n- Term item\n"
    candidates = page_index_candidates_module._extract_candidate_blocks(raw_text, page_number=1, source_format="markdown")

    assert [candidate.node_type_hint for candidate in candidates] == ["HEADING", "PARAGRAPH", "HEADING", "TERM"]
    assert candidates[0].line_start == 1
    assert candidates[1].line_start == 3
    assert candidates[2].line_start == 5
    assert candidates[3].kind_hint == "term_list_item"

    assignments = page_index_candidates_module._deterministic_block_assignments(candidates)
    validation = page_index_candidates_module._validate_block_assignments(candidates, assignments, page_text=raw_text)

    assert validation.valid is True
    assert validation.errors == []
    assert validation.fallback_reason is None


def test_page_index_assemble_repairs_forward_parent_to_root() -> None:
    candidates = [
        CandidateBlock(
            block_id="p0001-b001",
            page_number=1,
            order=1,
            start_char=0,
            end_char=18,
            line_start=1,
            line_end=1,
            indent=0,
            kind_hint="heading_markdown",
            confidence=0.98,
            text="# Root",
            node_type_hint="SECTION",
            title_hint="Root",
            heading_level=1,
        ),
        CandidateBlock(
            block_id="p0001-b002",
            page_number=1,
            order=2,
            start_char=8,
            end_char=31,
            line_start=3,
            line_end=3,
            indent=0,
            kind_hint="paragraph",
            confidence=0.8,
            text="Intro paragraph.",
            node_type_hint="PARAGRAPH",
            title_hint="Intro paragraph",
        ),
        CandidateBlock(
            block_id="p0001-b003",
            page_number=1,
            order=3,
            start_char=33,
            end_char=50,
            line_start=5,
            line_end=5,
            indent=0,
            kind_hint="heading_markdown",
            confidence=0.98,
            text="## Child",
            node_type_hint="SECTION",
            title_hint="Child",
            heading_level=2,
        ),
        CandidateBlock(
            block_id="p0001-b004",
            page_number=1,
            order=4,
            start_char=52,
            end_char=64,
            line_start=7,
            line_end=7,
            indent=0,
            kind_hint="term_list_item",
            confidence=0.74,
            text="- Term item",
            node_type_hint="TERM",
            title_hint="Term item",
        ),
    ]
    assignments = [
        BlockAssignment(block_id="p0001-b001", parent_id=None, node_type="SECTION", title="Root"),
        BlockAssignment(block_id="p0001-b002", parent_id="p0001-b003", node_type="PARAGRAPH", title="Intro paragraph"),
        BlockAssignment(block_id="p0001-b003", parent_id="p0001-b001", node_type="SECTION", title="Child"),
        BlockAssignment(block_id="p0001-b004", parent_id="p0001-b003", node_type="TERM", title="Term item"),
    ]

    assembled = page_index_candidates_module._assemble_page_index_blocks(candidates=candidates, assignments=assignments)

    assert [node.title for node in assembled] == ["Root", "Intro paragraph"]
    assert assembled[0].child_nodes[0].title == "Child"
    assert assembled[0].child_nodes[0].child_nodes[0].title == "Term item"


def test_page_index_classifies_all_caps_heading_but_demotes_inline_emphasis() -> None:
    heading = page_index_candidates_module._classify_block(
        "OPERATIONS OVERVIEW",
        0,
        len("OPERATIONS OVERVIEW") - 1,
        source_format="text",
        has_blank_before=True,
        has_blank_after=True,
    )
    emphasis = page_index_candidates_module._classify_block(
        "IMPORTANT NOTE FOR STAFF",
        0,
        len("IMPORTANT NOTE FOR STAFF") - 1,
        source_format="text",
        has_blank_before=False,
        has_blank_after=False,
    )

    assert heading.node_type == "HEADING"
    assert heading.confidence > emphasis.confidence
    assert emphasis.node_type == "PARAGRAPH"


def test_page_index_extracts_flat_pages_without_headings_as_paragraphs() -> None:
    raw_text = (
        "This is the first paragraph with ordinary prose.\n"
        "It continues on the next line with no heading cue.\n\n"
        "This is the second paragraph, also plain prose.\n"
        "It remains descriptive and sentence-like."
    )

    candidates = page_index_candidates_module._extract_candidate_blocks(raw_text, page_number=1, source_format="text")

    assert len(candidates) == 2
    assert all(candidate.node_type_hint == "PARAGRAPH" for candidate in candidates)


def test_page_index_extracts_dense_clause_candidates() -> None:
    raw_text = "1.2.3 Findings\n\n(a) Alpha clause\n\n(i) Roman item\n"

    candidates = page_index_candidates_module._extract_candidate_blocks(raw_text, page_number=1, source_format="text")

    assert [candidate.node_type_hint for candidate in candidates] == ["HEADING", "TERM", "TERM"]
    assert candidates[0].heading_level == 4
    assert candidates[1].kind_hint == "term_list_item"
    assert candidates[2].kind_hint == "term_list_item"


def test_page_index_demotes_numeric_heavy_table_like_line() -> None:
    candidate = page_index_candidates_module._classify_block(
        "2024 2025 2026 2027 2028 2029",
        0,
        len("2024 2025 2026 2027 2028 2029") - 1,
        source_format="text",
        has_blank_before=True,
        has_blank_after=True,
    )

    assert candidate.node_type == "PARAGRAPH"
    assert candidate.kind_hint == "paragraph_table_like"
    assert candidate.confidence < 0.8


def test_page_index_validator_rejects_missing_and_forward_parent() -> None:
    raw_text = "# Root\n\nIntro paragraph.\n\n## Child\n\n- Term item\n"
    candidates = page_index_candidates_module._extract_candidate_blocks(raw_text, page_number=1, source_format="markdown")
    assignments = page_index_candidates_module._deterministic_block_assignments(candidates)
    bad_assignments = [
        BlockAssignment(block_id=assignments[0].block_id, parent_id=assignments[2].block_id, node_type=assignments[0].node_type, title=assignments[0].title),
        BlockAssignment(block_id=assignments[1].block_id, parent_id=assignments[0].block_id, node_type=assignments[1].node_type, title=assignments[1].title),
    ]

    validation = page_index_candidates_module._validate_block_assignments(candidates, bad_assignments, page_text=raw_text)

    assert validation.valid is False
    assert any("missing block assignments" in error for error in validation.errors)
    assert any("forward or self parent" in error for error in validation.errors)


def test_page_index_validator_rejects_duplicates_unknown_ids_and_headingless_sections() -> None:
    raw_text = "# Root\n\nIntro paragraph.\n"
    candidates = page_index_candidates_module._extract_candidate_blocks(raw_text, page_number=1, source_format="markdown")
    assignments = page_index_candidates_module._deterministic_block_assignments(candidates)
    bad_assignments = [
        assignments[0],
        assignments[1],
        assignments[0].model_copy(),
        BlockAssignment(block_id="p0001-b999", parent_id=None, node_type="TERM", title="Ghost"),
    ]

    validation = page_index_candidates_module._validate_block_assignments(candidates, bad_assignments, page_text=raw_text)

    assert validation.valid is False
    assert any("duplicate block assignments" in error for error in validation.errors)
    assert any("unknown block ids" in error for error in validation.errors)

    section_candidate = CandidateBlock(
        block_id="p0002-b001",
        page_number=1,
        order=1,
        start_char=0,
        end_char=24,
        line_start=1,
        line_end=1,
        indent=0,
        kind_hint="paragraph",
        confidence=0.7,
        text="Ordinary sentence text.",
        node_type_hint="PARAGRAPH",
        title_hint="Ordinary sentence text",
    )
    section_validation = page_index_candidates_module._validate_block_assignments(
        [section_candidate],
        [BlockAssignment(block_id="p0002-b001", parent_id=None, node_type="SECTION", title="Ordinary sentence text")],
        page_text="Ordinary sentence text.",
    )

    assert section_validation.valid is False
    assert any("lacks heading evidence" in error for error in section_validation.errors)


def test_page_index_validator_rejects_cycles_and_bad_term_evidence() -> None:
    candidates = [
        CandidateBlock(
            block_id="p0001-b001",
            page_number=1,
            order=1,
            start_char=0,
            end_char=5,
            line_start=1,
            line_end=1,
            indent=0,
            kind_hint="paragraph",
            confidence=0.8,
            text="Alpha beta gamma.",
            node_type_hint="PARAGRAPH",
            title_hint="Alpha beta gamma",
        ),
        CandidateBlock(
            block_id="p0001-b002",
            page_number=1,
            order=2,
            start_char=6,
            end_char=90,
            line_start=2,
            line_end=2,
            indent=0,
            kind_hint="paragraph",
            confidence=0.8,
            text="Delta epsilon is a deliberately long sentence that should not qualify as a term block.",
            node_type_hint="PARAGRAPH",
            title_hint="Delta epsilon is a deliberately long sentence",
        ),
    ]
    assignments = [
        BlockAssignment(block_id="p0001-b001", parent_id="p0001-b002", node_type="SECTION", title="Alpha"),
        BlockAssignment(block_id="p0001-b002", parent_id="p0001-b001", node_type="TERM", title="Delta"),
    ]

    validation = page_index_candidates_module._validate_block_assignments(candidates, assignments, page_text="Alpha beta gamma.\nDelta epsilon.")

    assert validation.valid is False
    assert any("cycle" in error for error in validation.errors)
    assert any("term evidence" in error for error in validation.errors)


def test_page_index_validator_allows_repeated_sibling_text_at_distinct_occurrences() -> None:
    candidates = [
        CandidateBlock(
            block_id="p0001-b001",
            page_number=1,
            order=1,
            start_char=0,
            end_char=15,
            line_start=1,
            line_end=1,
            indent=0,
            kind_hint="heading_markdown",
            confidence=0.98,
            text="Shared content",
            node_type_hint="SECTION",
            title_hint="Shared content",
            heading_level=1,
        ),
        CandidateBlock(
            block_id="p0001-b002",
            page_number=1,
            order=2,
            start_char=16,
            end_char=31,
            line_start=2,
            line_end=2,
            indent=0,
            kind_hint="paragraph",
            confidence=0.8,
            text="Shared content",
            node_type_hint="PARAGRAPH",
            title_hint="Shared content",
        ),
        CandidateBlock(
            block_id="p0001-b003",
            page_number=1,
            order=3,
            start_char=32,
            end_char=47,
            line_start=3,
            line_end=3,
            indent=0,
            kind_hint="paragraph",
            confidence=0.8,
            text="Shared content",
            node_type_hint="PARAGRAPH",
            title_hint="Shared content",
        ),
    ]
    assignments = [
        BlockAssignment(block_id="p0001-b001", parent_id=None, node_type="SECTION", title="Shared content"),
        BlockAssignment(block_id="p0001-b002", parent_id="p0001-b001", node_type="PARAGRAPH", title="Shared content"),
        BlockAssignment(block_id="p0001-b003", parent_id="p0001-b001", node_type="PARAGRAPH", title="Shared content"),
    ]

    validation = page_index_candidates_module._validate_block_assignments(candidates, assignments, page_text="Shared content")

    assert validation.valid is False
    assert not any("repeats the same sibling excerpt" in error for error in validation.errors)
    assert any("duplicates the whole page" in error for error in validation.errors)


def test_page_index_validator_rejects_reused_sibling_source_occurrence() -> None:
    candidates = [
        CandidateBlock(
            block_id="p0001-b001",
            page_number=1,
            order=1,
            start_char=0,
            end_char=15,
            line_start=1,
            line_end=1,
            indent=0,
            kind_hint="heading_markdown",
            confidence=0.98,
            text="Shared content",
            node_type_hint="SECTION",
            title_hint="Shared content",
            heading_level=1,
        ),
        CandidateBlock(
            block_id="p0001-b002",
            page_number=1,
            order=2,
            start_char=16,
            end_char=31,
            line_start=2,
            line_end=2,
            indent=0,
            kind_hint="paragraph",
            confidence=0.8,
            text="Shared content",
            node_type_hint="PARAGRAPH",
            title_hint="Shared content",
        ),
        CandidateBlock(
            block_id="p0001-b003",
            page_number=1,
            order=3,
            start_char=16,
            end_char=31,
            line_start=3,
            line_end=3,
            indent=0,
            kind_hint="paragraph",
            confidence=0.8,
            text="Shared content",
            node_type_hint="PARAGRAPH",
            title_hint="Shared content",
        ),
    ]
    assignments = [
        BlockAssignment(block_id="p0001-b001", parent_id=None, node_type="SECTION", title="Shared content"),
        BlockAssignment(block_id="p0001-b002", parent_id="p0001-b001", node_type="PARAGRAPH", title="Shared content"),
        BlockAssignment(block_id="p0001-b003", parent_id="p0001-b001", node_type="PARAGRAPH", title="Shared content"),
    ]

    validation = page_index_candidates_module._validate_block_assignments(
        candidates,
        assignments,
        page_text="Shared content",
    )

    assert validation.valid is False
    assert any("reuses overlapping source occurrence" in error for error in validation.errors)


def test_page_index_nested_validation_reports_only_the_failing_descendant() -> None:
    page_text = "# Root\n\nChild content.\n"
    candidates = page_index_candidates_module._extract_candidate_blocks(
        page_text,
        page_number=1,
        source_format="markdown",
    )
    assignments = page_index_candidates_module._deterministic_block_assignments(candidates)
    specs = [
        PageIndexBlockSpec(
            title="Root",
            node_type="SECTION",
            excerpt="# Root",
            source_role="heading",
            child_nodes=[
                PageIndexBlockSpec(
                    title="Child content.",
                    node_type="PARAGRAPH",
                    excerpt="",
                    source_role="content",
                )
            ],
        )
    ]

    validation = page_index_candidates_module._validate_page_index_block_structure(
        candidates=candidates,
        assignments=assignments,
        block_specs=specs,
        page_text=page_text,
    )

    assert validation.valid is False
    assert "block 1.1 has empty excerpt" in validation.errors
    assert "block 1 has empty excerpt" not in validation.errors


