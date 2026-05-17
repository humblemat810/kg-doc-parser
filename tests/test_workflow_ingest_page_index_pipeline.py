from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest
from joblib import Memory

from _kogwistar_test_helpers import build_workflow_engine_triplet, drain_phase1_indexes_until_idle
import kg_doc_parser.workflow_ingest.page_index as page_index_module
from kg_doc_parser.workflow_ingest import (
    BlockAssignment,
    BlockAssignmentBatch,
    CandidateBlock,
    PageIndexParseResult,
    ProviderEndpointConfig,
    WorkflowProviderSettings,
    parse_page_index_document,
)
from kg_doc_parser.workflow_ingest.service import run_ingest_workflow
from kg_doc_parser.workflow_ingest.semantics import semantic_tree_to_kge_payload


pytestmark = [pytest.mark.workflow]


def _fixture_text(name: str) -> str:
    return (Path(__file__).parent / "fixtures" / "page_index" / name).read_text(encoding="utf-8")


def _scratch(name: str) -> Path:
    root = Path("tests") / ".tmp_workflow_ingest_page_index"
    path = root / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _manual_ollama_case_cache_dir(*, fixture_name: str, parser_model: str) -> Path:
    safe_model = parser_model.replace(":", "_").replace("/", "_")
    safe_fixture = Path(fixture_name).stem
    path = Path("tests") / ".tmp_workflow_ingest_page_index" / "manual_ollama_cache" / safe_model / safe_fixture
    path.mkdir(parents=True, exist_ok=True)
    return path


def _node_signature(node) -> tuple[str, str, tuple]:
    return (
        node.node_type,
        node.title,
        tuple(_node_signature(child) for child in node.child_nodes),
    )


def _normalized_node_signature(node) -> tuple[str, str, tuple]:
    node_type = "HEADING" if node.node_type in {"SECTION", "SUBSECTION"} else node.node_type
    return (
        node_type,
        node.title,
        tuple(_normalized_node_signature(child) for child in node.child_nodes),
    )


def _collect_node_types(node) -> set[str]:
    kinds = {node.node_type}
    for child in node.child_nodes:
        kinds.update(_collect_node_types(child))
    return kinds


def _max_depth(node) -> int:
    if not node.child_nodes:
        return 1
    return 1 + max(_max_depth(child) for child in node.child_nodes)


def _install_fake_page_index_chat(
    monkeypatch: pytest.MonkeyPatch,
    *,
    assignment_payload,
    refinement_payload=None,
    refinement_parsing_error: str | None = None,
) -> None:
    class _FakeStructured:
        def __init__(self, schema):
            self.schema = schema

        def invoke(self, messages):
            if self.schema is page_index_module.BlockAssignmentBatch:
                return {"parsed": assignment_payload}
            if self.schema is page_index_module.ExcerptRefinementBatch:
                if refinement_parsing_error is not None:
                    return {"parsed": None, "parsing_error": refinement_parsing_error}
                return {"parsed": refinement_payload}
            raise AssertionError(f"unexpected structured schema: {self.schema!r}")

    class _FakeChat:
        def with_structured_output(self, schema, include_raw=True):
            return _FakeStructured(schema)

    monkeypatch.setattr(page_index_module, "build_chat_model_for_role", lambda *args, **kwargs: _FakeChat())


def test_page_index_module_exports_hybrid_primitives() -> None:
    assert hasattr(page_index_module, "CandidateBlock")
    assert hasattr(page_index_module, "BlockAssignment")
    assert hasattr(page_index_module, "BlockAssignmentBatch")
    assert hasattr(page_index_module, "PageIndexValidationResult")
    assert CandidateBlock is page_index_module.CandidateBlock
    assert BlockAssignment is page_index_module.BlockAssignment
    assert BlockAssignmentBatch is page_index_module.BlockAssignmentBatch
    assert PageIndexParseResult is page_index_module.PageIndexParseResult


@pytest.mark.ci
@pytest.mark.parametrize(
    "fixture_name, source_format",
    [
        pytest.param("sample_page_index.txt", "text", id="text"),
        pytest.param("sample_page_index.md", "markdown", id="markdown"),
    ],
)
def test_page_index_heuristic_parses_text_and_markdown(fixture_name: str, source_format: str) -> None:
    raw_text = _fixture_text(fixture_name)
    result: PageIndexParseResult = parse_page_index_document(
        document_id=f"page-index-{source_format}",
        title="Page Index Document",
        raw_text=raw_text,
        source_format=source_format,
        mode="heuristic",
    )

    assert result.mode == "heuristic"
    assert result.source_format == source_format
    assert len(result.workflow_input.collections[0].pages) == 2
    assert len(result.semantic_tree.child_nodes) == 2
    assert [page.title for page in result.semantic_tree.child_nodes] == ["Page 1", "Page 2"]
    assert result.authoritative_source_map.keys() == result.parser_source_map.keys()
    assert result.coverage["overall"] > 0.95

    node_types = _collect_node_types(result.semantic_tree)
    assert "PAGE" in node_types
    assert "SECTION" in node_types
    assert "SUBSECTION" in node_types
    assert "PARAGRAPH" in node_types
    assert "TERM" in node_types
    assert _max_depth(result.semantic_tree) >= 6

    payload = semantic_tree_to_kge_payload(result.semantic_tree, doc_id=result.workflow_input.request_id)
    assert len(payload["nodes"]) >= 8
    assert len(payload["edges"]) >= 4


def test_page_index_candidate_extraction_and_validation() -> None:
    raw_text = "# Root\n\nIntro paragraph.\n\n## Child\n\n- Term item\n"
    candidates = page_index_module._extract_candidate_blocks(raw_text, page_number=1, source_format="markdown")

    assert [candidate.node_type_hint for candidate in candidates] == ["SECTION", "PARAGRAPH", "SECTION", "TERM"]
    assert candidates[0].line_start == 1
    assert candidates[1].line_start == 3
    assert candidates[2].line_start == 5
    assert candidates[3].kind_hint == "term_list_item"

    assignments = page_index_module._deterministic_block_assignments(candidates)
    validation = page_index_module._validate_block_assignments(candidates, assignments, page_text=raw_text)

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

    assembled = page_index_module._assemble_page_index_blocks(candidates=candidates, assignments=assignments)

    assert [node.title for node in assembled] == ["Root", "Intro paragraph"]
    assert assembled[0].child_nodes[0].title == "Child"
    assert assembled[0].child_nodes[0].child_nodes[0].title == "Term item"


def test_page_index_classifies_all_caps_heading_but_demotes_inline_emphasis() -> None:
    heading = page_index_module._classify_block(
        "OPERATIONS OVERVIEW",
        0,
        len("OPERATIONS OVERVIEW") - 1,
        source_format="text",
        has_blank_before=True,
        has_blank_after=True,
    )
    emphasis = page_index_module._classify_block(
        "IMPORTANT NOTE FOR STAFF",
        0,
        len("IMPORTANT NOTE FOR STAFF") - 1,
        source_format="text",
        has_blank_before=False,
        has_blank_after=False,
    )

    assert heading.node_type == "SECTION"
    assert heading.confidence > emphasis.confidence
    assert emphasis.node_type == "PARAGRAPH"


def test_page_index_extracts_flat_pages_without_headings_as_paragraphs() -> None:
    raw_text = (
        "This is the first paragraph with ordinary prose.\n"
        "It continues on the next line with no heading cue.\n\n"
        "This is the second paragraph, also plain prose.\n"
        "It remains descriptive and sentence-like."
    )

    candidates = page_index_module._extract_candidate_blocks(raw_text, page_number=1, source_format="text")

    assert len(candidates) == 2
    assert all(candidate.node_type_hint == "PARAGRAPH" for candidate in candidates)


def test_page_index_extracts_dense_clause_candidates() -> None:
    raw_text = "1.2.3 Findings\n\n(a) Alpha clause\n\n(i) Roman item\n"

    candidates = page_index_module._extract_candidate_blocks(raw_text, page_number=1, source_format="text")

    assert [candidate.node_type_hint for candidate in candidates] == ["SUBSECTION", "TERM", "TERM"]
    assert candidates[0].heading_level == 4
    assert candidates[1].kind_hint == "term_list_item"
    assert candidates[2].kind_hint == "term_list_item"


def test_page_index_demotes_numeric_heavy_table_like_line() -> None:
    candidate = page_index_module._classify_block(
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
    candidates = page_index_module._extract_candidate_blocks(raw_text, page_number=1, source_format="markdown")
    assignments = page_index_module._deterministic_block_assignments(candidates)
    bad_assignments = [
        BlockAssignment(block_id=assignments[0].block_id, parent_id=assignments[2].block_id, node_type=assignments[0].node_type, title=assignments[0].title),
        BlockAssignment(block_id=assignments[1].block_id, parent_id=assignments[0].block_id, node_type=assignments[1].node_type, title=assignments[1].title),
    ]

    validation = page_index_module._validate_block_assignments(candidates, bad_assignments, page_text=raw_text)

    assert validation.valid is False
    assert any("missing block assignments" in error for error in validation.errors)
    assert any("forward or self parent" in error for error in validation.errors)


def test_page_index_validator_rejects_duplicates_unknown_ids_and_headingless_sections() -> None:
    raw_text = "# Root\n\nIntro paragraph.\n"
    candidates = page_index_module._extract_candidate_blocks(raw_text, page_number=1, source_format="markdown")
    assignments = page_index_module._deterministic_block_assignments(candidates)
    bad_assignments = [
        assignments[0],
        assignments[1],
        assignments[0].model_copy(),
        BlockAssignment(block_id="p0001-b999", parent_id=None, node_type="TERM", title="Ghost"),
    ]

    validation = page_index_module._validate_block_assignments(candidates, bad_assignments, page_text=raw_text)

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
    section_validation = page_index_module._validate_block_assignments(
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

    validation = page_index_module._validate_block_assignments(candidates, assignments, page_text="Alpha beta gamma.\nDelta epsilon.")

    assert validation.valid is False
    assert any("cycle" in error for error in validation.errors)
    assert any("term evidence" in error for error in validation.errors)


def test_page_index_validator_rejects_repeated_sibling_and_whole_page_duplication() -> None:
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

    validation = page_index_module._validate_block_assignments(candidates, assignments, page_text="Shared content")

    assert validation.valid is False
    assert any("repeats the same sibling excerpt" in error for error in validation.errors)
    assert any("duplicates the whole page" in error for error in validation.errors)


def test_page_index_ollama_flat_assignment_parses_and_assembles(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_text = "# Root\n\nIntro paragraph.\n\n## Child\n\n- Term item\n"
    assignments = BlockAssignmentBatch(
        assignments=[
            BlockAssignment(block_id="p0001-b001", parent_id=None, node_type="SECTION", title="Root"),
            BlockAssignment(block_id="p0001-b002", parent_id="p0001-b001", node_type="PARAGRAPH", title="Intro paragraph"),
            BlockAssignment(block_id="p0001-b003", parent_id="p0001-b001", node_type="SUBSECTION", title="Child"),
            BlockAssignment(block_id="p0001-b004", parent_id="p0001-b003", node_type="TERM", title="Term item"),
        ]
    )

    class _FakeStructured:
        def __init__(self, payload: BlockAssignmentBatch):
            self._payload = payload

        def invoke(self, messages):
            return {"parsed": self._payload}

    class _FakeChat:
        def __init__(self, payload: BlockAssignmentBatch):
            self._payload = payload

        def with_structured_output(self, schema, include_raw=True):
            return _FakeStructured(self._payload)

    monkeypatch.setattr(page_index_module, "build_chat_model_for_role", lambda *args, **kwargs: _FakeChat(assignments))

    provider_settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(provider="ollama", model="fake", base_url="http://127.0.0.1:11434")
    )
    result = parse_page_index_document(
        document_id="page-index-ollama-flat",
        title="Page Index Document",
        raw_text=raw_text,
        source_format="markdown",
        mode="ollama",
        provider_settings=provider_settings,
    )

    assert result.diagnostics["assignment_mode"] == "ollama_flat_assignment"
    assert result.diagnostics["fallback_reason"] is None
    root = result.semantic_tree.child_nodes[0].child_nodes[0]
    assert root.title == "Root"
    assert root.child_nodes[1].title == "Child"
    assert root.child_nodes[1].child_nodes[0].title == "Term item"


def test_page_index_ollama_malformed_payload_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_text = "# Root\n\nIntro paragraph.\n\n## Child\n\n- Term item\n"

    class _FakeStructured:
        def invoke(self, messages):
            return {"parsed": None, "parsing_error": "malformed json"}

    class _FakeChat:
        def with_structured_output(self, schema, include_raw=True):
            return _FakeStructured()

    monkeypatch.setattr(page_index_module, "build_chat_model_for_role", lambda *args, **kwargs: _FakeChat())

    provider_settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(provider="ollama", model="fake", base_url="http://127.0.0.1:11434")
    )
    result = parse_page_index_document(
        document_id="page-index-ollama-malformed",
        title="Page Index Document",
        raw_text=raw_text,
        source_format="markdown",
        mode="ollama",
        provider_settings=provider_settings,
    )

    assert result.diagnostics["assignment_mode"] == "deterministic_fallback"
    assert result.diagnostics["fallback_reason"] == "llm_unavailable_or_invalid"
    assert result.semantic_tree.child_nodes[0].child_nodes[0].title == "Root"


def test_page_index_ollama_invalid_assignments_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_text = "# Root\n\nIntro paragraph.\n\n## Child\n\n- Term item\n"
    bad_assignments = BlockAssignmentBatch(
        assignments=[
            BlockAssignment(block_id="p0001-b001", parent_id=None, node_type="SECTION", title="Root"),
            BlockAssignment(block_id="p0001-b002", parent_id="p0001-b001", node_type="PARAGRAPH", title="Intro paragraph"),
        ]
    )

    class _FakeStructured:
        def __init__(self, payload: BlockAssignmentBatch):
            self._payload = payload

        def invoke(self, messages):
            return {"parsed": self._payload}

    class _FakeChat:
        def __init__(self, payload: BlockAssignmentBatch):
            self._payload = payload

        def with_structured_output(self, schema, include_raw=True):
            return _FakeStructured(self._payload)

    monkeypatch.setattr(page_index_module, "build_chat_model_for_role", lambda *args, **kwargs: _FakeChat(bad_assignments))

    provider_settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(provider="ollama", model="fake", base_url="http://127.0.0.1:11434")
    )
    result = parse_page_index_document(
        document_id="page-index-ollama-fallback",
        title="Page Index Document",
        raw_text=raw_text,
        source_format="markdown",
        mode="ollama",
        provider_settings=provider_settings,
    )

    assert result.diagnostics["assignment_mode"] == "deterministic_fallback"
    assert result.diagnostics["validation_errors"]
    assert result.semantic_tree.child_nodes[0].child_nodes[0].title == "Root"


def test_page_index_refinement_disabled_by_default_preserves_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_text = "# Root\n\nIntro paragraph.\n\n## Child\n\n- Term item\n"
    assignments = BlockAssignmentBatch(
        assignments=[
            BlockAssignment(block_id="p0001-b001", parent_id=None, node_type="SECTION", title="Root"),
            BlockAssignment(block_id="p0001-b002", parent_id="p0001-b001", node_type="PARAGRAPH", title="Intro paragraph"),
            BlockAssignment(block_id="p0001-b003", parent_id="p0001-b001", node_type="SUBSECTION", title="Child"),
            BlockAssignment(block_id="p0001-b004", parent_id="p0001-b003", node_type="TERM", title="Term item"),
        ]
    )
    _install_fake_page_index_chat(monkeypatch, assignment_payload=assignments)

    provider_settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(provider="ollama", model="fake", base_url="http://127.0.0.1:11434")
    )
    default_result = parse_page_index_document(
        document_id="page-index-ollama-default-refine",
        title="Page Index Document",
        raw_text=raw_text,
        source_format="markdown",
        mode="ollama",
        provider_settings=provider_settings,
    )
    explicit_result = parse_page_index_document(
        document_id="page-index-ollama-explicit-refine-off",
        title="Page Index Document",
        raw_text=raw_text,
        source_format="markdown",
        mode="ollama",
        provider_settings=provider_settings,
        refine_excerpts=False,
    )

    assert _normalized_node_signature(default_result.semantic_tree) == _normalized_node_signature(explicit_result.semantic_tree)
    assert default_result.diagnostics["refine_excerpts_enabled"] is False
    assert default_result.diagnostics["refine_excerpts_attempted"] == 0


def test_page_index_refinement_accepts_shorter_exact_substring(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_text = "# Root\n\nIntro paragraph.\n\n## Child\n\n- Term item\n"
    refinement = page_index_module.ExcerptRefinementBatch(
        suggestions=[
            page_index_module.ExcerptRefinementSuggestion(path_id="0.0", excerpt="Intro paragraph"),
        ]
    )
    _install_fake_page_index_chat(
        monkeypatch,
        assignment_payload=BlockAssignmentBatch(assignments=[]),
        refinement_payload=refinement,
    )

    block_specs = [
        page_index_module.PageIndexBlockSpec(
            title="Root",
            node_type="SECTION",
            excerpt="# Root",
            child_nodes=[
                page_index_module.PageIndexBlockSpec(
                    title="Intro paragraph",
                    node_type="PARAGRAPH",
                    excerpt="Intro paragraph.",
                    child_nodes=[],
                )
            ],
        )
    ]

    provider_settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(provider="ollama", model="fake", base_url="http://127.0.0.1:11434")
    )
    refined, diagnostics = page_index_module._refine_page_index_block_excerpts(
        block_specs=block_specs,
        page_text=raw_text,
        page_number=1,
        unit_id="page-1",
        provider_settings=provider_settings,
    )

    assert refined[0].child_nodes[0].excerpt == "Intro paragraph"
    assert diagnostics["refine_excerpts_enabled"] is True
    assert diagnostics["refine_excerpts_accepted"] == 1
    assert diagnostics["refine_excerpts_rejected"] == 0
    assert diagnostics["refine_excerpts_fallback"] is False


def test_page_index_refinement_rejects_non_substring_and_keeps_original(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_text = "# Root\n\nIntro paragraph.\n\n## Child\n\n- Term item\n"
    assignments = BlockAssignmentBatch(
        assignments=[
            BlockAssignment(block_id="p0001-b001", parent_id=None, node_type="SECTION", title="Root"),
            BlockAssignment(block_id="p0001-b002", parent_id="p0001-b001", node_type="PARAGRAPH", title="Intro paragraph"),
            BlockAssignment(block_id="p0001-b003", parent_id="p0001-b001", node_type="SUBSECTION", title="Child"),
            BlockAssignment(block_id="p0001-b004", parent_id="p0001-b003", node_type="TERM", title="Term item"),
        ]
    )
    refinement = page_index_module.ExcerptRefinementBatch(
        suggestions=[
            page_index_module.ExcerptRefinementSuggestion(path_id="0.0", excerpt="not present"),
        ]
    )
    _install_fake_page_index_chat(monkeypatch, assignment_payload=assignments, refinement_payload=refinement)

    provider_settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(provider="ollama", model="fake", base_url="http://127.0.0.1:11434")
    )
    result = parse_page_index_document(
        document_id="page-index-ollama-refine-reject",
        title="Page Index Document",
        raw_text=raw_text,
        source_format="markdown",
        mode="ollama",
        provider_settings=provider_settings,
        refine_excerpts=True,
    )
    baseline_result = parse_page_index_document(
        document_id="page-index-ollama-refine-reject-baseline",
        title="Page Index Document",
        raw_text=raw_text,
        source_format="markdown",
        mode="ollama",
        provider_settings=provider_settings,
    )

    root = result.semantic_tree.child_nodes[0].child_nodes[0]
    baseline_root = baseline_result.semantic_tree.child_nodes[0].child_nodes[0]
    assert root.child_nodes[0].total_content_pointers[0].verbatim_text == baseline_root.child_nodes[0].total_content_pointers[0].verbatim_text
    assert result.diagnostics["refine_excerpts_enabled"] is True
    assert result.diagnostics["refine_excerpts_accepted"] == 0
    assert result.diagnostics["refine_excerpts_rejected"] == 1
    assert result.diagnostics["refine_excerpts_fallback"] is False


def test_page_index_refinement_rejects_whole_page_and_duplicate_excerpts(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_text = "# Root\n\nAlpha paragraph.\n\nBeta paragraph.\n"
    assignments = BlockAssignmentBatch(
        assignments=[
            BlockAssignment(block_id="p0001-b001", parent_id=None, node_type="SECTION", title="Root"),
            BlockAssignment(block_id="p0001-b002", parent_id="p0001-b001", node_type="PARAGRAPH", title="Alpha paragraph"),
            BlockAssignment(block_id="p0001-b003", parent_id="p0001-b001", node_type="PARAGRAPH", title="Beta paragraph"),
        ]
    )
    refinement = page_index_module.ExcerptRefinementBatch(
        suggestions=[
            page_index_module.ExcerptRefinementSuggestion(path_id="0", excerpt=raw_text),
            page_index_module.ExcerptRefinementSuggestion(path_id="0.0", excerpt="Alpha paragraph."),
            page_index_module.ExcerptRefinementSuggestion(path_id="0.1", excerpt="Alpha paragraph."),
        ]
    )
    _install_fake_page_index_chat(monkeypatch, assignment_payload=assignments, refinement_payload=refinement)

    provider_settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(provider="ollama", model="fake", base_url="http://127.0.0.1:11434")
    )
    result = parse_page_index_document(
        document_id="page-index-ollama-refine-dup",
        title="Page Index Document",
        raw_text=raw_text,
        source_format="markdown",
        mode="ollama",
        provider_settings=provider_settings,
        refine_excerpts=True,
    )
    baseline_result = parse_page_index_document(
        document_id="page-index-ollama-refine-dup-baseline",
        title="Page Index Document",
        raw_text=raw_text,
        source_format="markdown",
        mode="ollama",
        provider_settings=provider_settings,
    )

    root = result.semantic_tree.child_nodes[0].child_nodes[0]
    baseline_root = baseline_result.semantic_tree.child_nodes[0].child_nodes[0]
    assert root.total_content_pointers[0].verbatim_text == baseline_root.total_content_pointers[0].verbatim_text
    assert root.child_nodes[0].total_content_pointers[0].verbatim_text == baseline_root.child_nodes[0].total_content_pointers[0].verbatim_text
    assert root.child_nodes[1].total_content_pointers[0].verbatim_text == baseline_root.child_nodes[1].total_content_pointers[0].verbatim_text
    assert result.diagnostics["refine_excerpts_enabled"] is True
    assert result.diagnostics["refine_excerpts_accepted"] == 0
    assert result.diagnostics["refine_excerpts_rejected"] == 3
    assert result.diagnostics["refine_excerpts_fallback"] is False


def test_page_index_refinement_transport_failure_keeps_original_excerpts(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_text = "# Root\n\nIntro paragraph.\n\n## Child\n\n- Term item\n"
    assignments = BlockAssignmentBatch(
        assignments=[
            BlockAssignment(block_id="p0001-b001", parent_id=None, node_type="SECTION", title="Root"),
            BlockAssignment(block_id="p0001-b002", parent_id="p0001-b001", node_type="PARAGRAPH", title="Intro paragraph"),
            BlockAssignment(block_id="p0001-b003", parent_id="p0001-b001", node_type="SUBSECTION", title="Child"),
            BlockAssignment(block_id="p0001-b004", parent_id="p0001-b003", node_type="TERM", title="Term item"),
        ]
    )
    _install_fake_page_index_chat(
        monkeypatch,
        assignment_payload=assignments,
        refinement_parsing_error="malformed refinement payload",
    )

    provider_settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(provider="ollama", model="fake", base_url="http://127.0.0.1:11434")
    )
    result = parse_page_index_document(
        document_id="page-index-ollama-refine-fallback",
        title="Page Index Document",
        raw_text=raw_text,
        source_format="markdown",
        mode="ollama",
        provider_settings=provider_settings,
        refine_excerpts=True,
    )
    baseline_result = parse_page_index_document(
        document_id="page-index-ollama-refine-fallback-baseline",
        title="Page Index Document",
        raw_text=raw_text,
        source_format="markdown",
        mode="ollama",
        provider_settings=provider_settings,
    )

    root = result.semantic_tree.child_nodes[0].child_nodes[0]
    baseline_root = baseline_result.semantic_tree.child_nodes[0].child_nodes[0]
    assert root.child_nodes[0].total_content_pointers[0].verbatim_text == baseline_root.child_nodes[0].total_content_pointers[0].verbatim_text
    assert result.diagnostics["refine_excerpts_enabled"] is True
    assert result.diagnostics["refine_excerpts_fallback"] is True
    assert result.diagnostics["refine_excerpts_accepted"] == 0
    assert result.diagnostics["refine_excerpts_rejected"] == result.diagnostics["refine_excerpts_attempted"]


def test_page_index_refinement_reports_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_text = "# Root\n\nIntro paragraph.\n"
    assignments = BlockAssignmentBatch(
        assignments=[
            BlockAssignment(block_id="p0001-b001", parent_id=None, node_type="SECTION", title="Root"),
            BlockAssignment(block_id="p0001-b002", parent_id="p0001-b001", node_type="PARAGRAPH", title="Intro paragraph"),
        ]
    )
    refinement = page_index_module.ExcerptRefinementBatch(
        suggestions=[
            page_index_module.ExcerptRefinementSuggestion(path_id="0.0", excerpt="Intro paragraph"),
        ]
    )
    _install_fake_page_index_chat(monkeypatch, assignment_payload=assignments, refinement_payload=refinement)

    provider_settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(provider="ollama", model="fake", base_url="http://127.0.0.1:11434")
    )
    result = parse_page_index_document(
        document_id="page-index-ollama-refine-counts",
        title="Page Index Document",
        raw_text=raw_text,
        source_format="markdown",
        mode="ollama",
        provider_settings=provider_settings,
        refine_excerpts=True,
    )

    assert result.diagnostics["refine_excerpts_enabled"] is True
    assert result.diagnostics["refine_excerpts_attempted"] == 2
    assert result.diagnostics["refine_excerpts_accepted"] == 1
    assert result.diagnostics["refine_excerpts_rejected"] == 0
    assert result.diagnostics["refine_excerpts_fallback"] is False


def test_page_index_ollama_flat_but_shallow_assignment_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_text = "# Root\n\nIntro paragraph.\n\n## Child\n\n- Term item\n"
    shallow_assignments = BlockAssignmentBatch(
        assignments=[
            BlockAssignment(block_id="p0001-b001", parent_id=None, node_type="SECTION", title="Root"),
            BlockAssignment(block_id="p0001-b002", parent_id="p0001-b001", node_type="PARAGRAPH", title="Intro paragraph"),
            BlockAssignment(block_id="p0001-b003", parent_id="p0001-b001", node_type="PARAGRAPH", title="Child"),
            BlockAssignment(block_id="p0001-b004", parent_id="p0001-b001", node_type="PARAGRAPH", title="Term item"),
        ]
    )

    class _FakeStructured:
        def __init__(self, payload: BlockAssignmentBatch):
            self._payload = payload

        def invoke(self, messages):
            return {"parsed": self._payload}

    class _FakeChat:
        def __init__(self, payload: BlockAssignmentBatch):
            self._payload = payload

        def with_structured_output(self, schema, include_raw=True):
            return _FakeStructured(self._payload)

    monkeypatch.setattr(page_index_module, "build_chat_model_for_role", lambda *args, **kwargs: _FakeChat(shallow_assignments))

    provider_settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(provider="ollama", model="fake", base_url="http://127.0.0.1:11434")
    )
    result = parse_page_index_document(
        document_id="page-index-ollama-shallow",
        title="Page Index Document",
        raw_text=raw_text,
        source_format="markdown",
        mode="ollama",
        provider_settings=provider_settings,
    )

    assert result.diagnostics["assignment_mode"] == "deterministic_fallback"
    assert any("heading structure was flattened" in error for error in result.diagnostics["validation_errors"])
    assert result.semantic_tree.child_nodes[0].child_nodes[0].title == "Root"

# Manual examples with explicit parametrized node ids:
#
# Heuristic, text:
#   .venv\Scripts\python.exe -m pytest tests/test_workflow_ingest_page_index_pipeline.py::test_page_index_heuristic_parses_text_and_markdown[text] -q
#
# Heuristic, markdown:
#   .venv\Scripts\python.exe -m pytest tests/test_workflow_ingest_page_index_pipeline.py::test_page_index_heuristic_parses_text_and_markdown[markdown] -q
#
# Ollama, text:
#   set KG_DOC_PARSER_PROVIDER=ollama
#   set KG_DOC_PARSER_MODEL=qwen3:4b-instruct-2507-q8_0
#   set KG_DOC_PARSER_BASE_URL=http://127.0.0.1:11434
#   .venv\Scripts\python.exe -m pytest tests/test_workflow_ingest_page_index_pipeline.py::test_page_index_ollama_smoke_parses_text_and_markdown[text] -q
#
# Ollama, markdown:
#   set KG_DOC_PARSER_PROVIDER=ollama
#   set KG_DOC_PARSER_MODEL=qwen3:4b-instruct-2507-q8_0
#   set KG_DOC_PARSER_BASE_URL=http://127.0.0.1:11434
#   .venv\Scripts\python.exe -m pytest tests/test_workflow_ingest_page_index_pipeline.py::test_page_index_ollama_smoke_parses_text_and_markdown[markdown] -q

@pytest.mark.ci
def test_page_index_heuristic_plain_text_and_markdown_share_structure() -> None:
    text_result = parse_page_index_document(
        document_id="page-index-text",
        title="Page Index Document",
        raw_text=_fixture_text("sample_page_index.txt"),
        source_format="text",
        mode="heuristic",
    )
    markdown_result = parse_page_index_document(
        document_id="page-index-markdown",
        title="Page Index Document",
        raw_text=_fixture_text("sample_page_index.md"),
        source_format="markdown",
        mode="heuristic",
    )

    assert _normalized_node_signature(text_result.semantic_tree) == _normalized_node_signature(markdown_result.semantic_tree)


@pytest.mark.ci_full
@pytest.mark.parametrize(
    "fixture_name, source_format",
    [
        pytest.param("sample_page_index.txt", "text", id="text"),
        pytest.param("sample_page_index.md", "markdown", id="markdown"),
    ],
)
@pytest.mark.parametrize(
    "parser_model",
    [
        pytest.param("gemma4:e2b", id="gemma4-e2b"),
        pytest.param("gemma4:latest", id="gemma4-latest"),
    ],
)
def test_page_index_ollama_smoke_parses_text_and_markdown(
    fixture_name: str,
    source_format: str,
    parser_model: str,
) -> None:
    pytest.importorskip("langchain_ollama")
    provider_settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(
            provider="ollama",
            model=parser_model,
            base_url=os.getenv("KG_DOC_PARSER_BASE_URL", "http://127.0.0.1:11434"),
        )
    )
    raw_text = _fixture_text(fixture_name)

    try:
        result = parse_page_index_document(
            document_id=f"page-index-ollama-{source_format}",
            title="Page Index Document",
            raw_text=raw_text,
            source_format=source_format,
            mode="ollama",
            provider_settings=provider_settings,
        )
    except Exception as exc:
        message = str(exc).lower()
        if any(token in message for token in ("connect", "connection", "refused", "model", "ollama")):
            pytest.skip(f"ollama parser unavailable: {exc}")
        raise

    assert result.mode == "ollama"
    assert result.source_format == source_format
    assert len(result.semantic_tree.child_nodes) == 2
    assert result.authoritative_source_map.keys() == result.parser_source_map.keys()
    assert result.coverage["overall"] > 0.80


@pytest.mark.ci_full
@pytest.mark.parametrize(
    "fixture_name, source_format",
    [
        pytest.param("sample_page_index.txt", "text", id="text"),
        pytest.param("sample_page_index.md", "markdown", id="markdown"),
    ],
)
@pytest.mark.parametrize(
    "parser_model",
    [
        pytest.param("gemma4:e2b", id="gemma4-e2b"),
        pytest.param("gemma4:latest", id="gemma4-latest"),
    ],
)
def test_page_index_workflow_ingest_with_ollama_manual_case(
    fixture_name: str,
    source_format: str,
    parser_model: str,
) -> None:
    """Manual smoke case with a stable cache dir.

    The cached Ollama parse is intentionally replayable for interactive runs.
    If the local model changes, the prompt changes, or the result looks stale,
    delete the per-case cache directory and rerun to force a fresh parse.
    """

    pytest.importorskip("langchain_ollama")
    scratch = _scratch("workflow_ingest_ollama")
    workflow_engine, conversation_engine, knowledge_engine = build_workflow_engine_triplet(scratch / "engines", "in_memory")
    raw_text = _fixture_text(fixture_name)
    provider_settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(
            provider="ollama",
            model=parser_model,
            base_url=os.getenv("KG_DOC_PARSER_BASE_URL", "http://127.0.0.1:11434"),
        )
    )

    cache_dir = _manual_ollama_case_cache_dir(fixture_name=fixture_name, parser_model=parser_model)
    memory = Memory(location=cache_dir, verbose=0)

    @memory.cache
    def _parse_cached(*, document_id: str, title: str, raw_text: str, source_format: str, provider_settings: WorkflowProviderSettings):
        return parse_page_index_document(
            document_id=document_id,
            title=title,
            raw_text=raw_text,
            source_format=source_format,
            mode="ollama",
            provider_settings=provider_settings,
        )

    parsed = _parse_cached(
        document_id=f"page-index-workflow-{source_format}",
        title="Page Index Document",
        raw_text=raw_text,
        source_format=source_format,
        provider_settings=provider_settings,
    )

    def _parse_semantic_fn(*, collection, parser_input_dict, parser_source_map):
        return parsed.semantic_tree

    try:
        run, bundle = run_ingest_workflow(
            inp=parsed.workflow_input,
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            knowledge_engine=knowledge_engine,
            deps={"parse_semantic_fn": _parse_semantic_fn},
        )
        drain_phase1_indexes_until_idle(workflow_engine, conversation_engine, knowledge_engine)
    except Exception as exc:
        message = str(exc).lower()
        if any(token in message for token in ("connect", "connection", "refused", "model", "ollama")):
            pytest.skip(f"ollama workflow ingest unavailable: {exc}")
        raise

    assert run.status == "succeeded"
    assert bundle is not None
    assert bundle.graph_payload["doc_id"] == parsed.workflow_input.request_id
    assert len(bundle.graph_payload["nodes"]) >= 1
