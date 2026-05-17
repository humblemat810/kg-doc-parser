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
#   set KG_DOC_PARSER_MODEL=gemma4:e2b
#   set KG_DOC_PARSER_BASE_URL=http://127.0.0.1:11434
#   .venv\Scripts\python.exe -m pytest tests/test_workflow_ingest_page_index_pipeline.py::test_page_index_ollama_smoke_parses_text_and_markdown[text] -q
#
# Ollama, markdown:
#   set KG_DOC_PARSER_PROVIDER=ollama
#   set KG_DOC_PARSER_MODEL=gemma4:e2b
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
