from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest
from _kogwistar_test_helpers import (
    build_workflow_engine_triplet,
    drain_phase1_indexes_until_idle,
)
from _ollama_test_helpers import ollama_available
from kogwistar.utils.cache_backend import Memory
from page_index_provider_test_support import (
    install_fake_page_index_chat as _install_fake_page_index_chat,
)
from page_index_test_support import (
    fixture_text as _fixture_text,
)

import kg_doc_parser.workflow_ingest.page_index as page_index_module
import kg_doc_parser.workflow_ingest.page_index_candidates as page_index_candidates_module
import kg_doc_parser.workflow_ingest.page_index_provider as page_index_provider_module
import kg_doc_parser.workflow_ingest.page_index_refinement as page_index_refinement_module
from kg_doc_parser.workflow_ingest import (
    BlockAssignment,
    BlockAssignmentBatch,
    CandidateBlock,
    PageIndexBlockSpec,
    PageIndexParseResult,
    ProviderEndpointConfig,
    WorkflowProviderSettings,
    parse_page_index_document,
    parse_page_index_layer,
)
from kg_doc_parser.workflow_ingest.semantics import (
    HydratedTextPointer,
    SemanticNode,
    compute_terminal_content_coverage,
    semantic_tree_to_kge_payload,
    validate_semantic_tree_integrity,
)
from kg_doc_parser.workflow_ingest.service import run_ingest_workflow

pytestmark = [pytest.mark.workflow]


def _find_nodes(root: SemanticNode, title: str) -> list[SemanticNode]:
    found: list[SemanticNode] = []
    pending = [root]
    while pending:
        node = pending.pop()
        if node.title == title:
            found.append(node)
        pending.extend(reversed(node.child_nodes))
    return found


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


def test_page_index_wrapper_rebase_updates_all_descendant_levels() -> None:
    left = SemanticNode(
        node_id="left",
        parent_id="page",
        title="Left",
        level_from_root=2,
        child_nodes=[
            SemanticNode(node_id="left-a", parent_id="left", title="A", level_from_root=3),
            SemanticNode(node_id="left-b", parent_id="left", title="B", level_from_root=3),
        ],
    )
    right = SemanticNode(
        node_id="right",
        parent_id="page",
        title="Right",
        level_from_root=2,
        child_nodes=[
            SemanticNode(node_id="right-a", parent_id="right", title="A", level_from_root=3),
            SemanticNode(node_id="right-b", parent_id="right", title="B", level_from_root=3),
        ],
    )
    page = SemanticNode(
        node_id="page",
        parent_id="root",
        title="Page",
        level_from_root=1,
        metadata={"page_index_role": "page"},
        child_nodes=[left, right],
    )
    root = SemanticNode(
        node_id="root",
        title="Root",
        node_type="DOCUMENT_ROOT",
        child_nodes=[page],
    )

    normalized = page_index_module._normalize_page_index_unary_nodes(root)

    assert [node.node_id for node in normalized.child_nodes] == ["left", "right"]
    assert all(node.level_from_root == 1 for node in normalized.child_nodes)
    assert all(
        grandchild.level_from_root == 2
        for node in normalized.child_nodes
        for grandchild in node.child_nodes
    )


def test_page_index_atomic_fallback_preserves_child_evidence_metadata() -> None:
    child = SemanticNode(
        node_id="child",
        parent_id="parent",
        title="Meaningful child",
        node_type="SECTION",
        total_content_pointers=[
            HydratedTextPointer(
                source_cluster_id="unit",
                start_char=0,
                end_char=4,
                verbatim_text="Alpha",
            )
        ],
    )
    parent = SemanticNode(
        node_id="parent",
        title="Wrapper",
        node_type="TEXT_FLOW",
        child_nodes=[child],
    )

    normalized = page_index_module._normalize_page_index_unary_nodes(parent)

    assert normalized.child_nodes == []
    assert normalized.total_content_pointers[0].verbatim_text == "Alpha"
    assert normalized.metadata["atomic_retained_child"] == {
        "node_id": "child",
        "title": "Meaningful child",
        "node_type": "SECTION",
    }


@pytest.mark.ci
def test_page_index_unary_normalization_preserves_nested_descendants_and_ownership() -> None:
    source = "Overview\nFirst row\nSecond row\nEndnote\n"
    source_map = {"page_1": {"text": source}}

    def pointer(excerpt: str) -> HydratedTextPointer:
        start = source.index(excerpt)
        return HydratedTextPointer(
            source_cluster_id="page_1",
            start_char=start,
            end_char=start + len(excerpt) - 1,
            verbatim_text=excerpt,
        )

    rows = [
        SemanticNode(
            node_id=node_id,
            parent_id="inner",
            node_type="TABLE_ROW",
            title=excerpt,
            total_content_pointers=[pointer(excerpt)],
            level_from_root=3,
        )
        for node_id, excerpt in (("r1", "First row"), ("r2", "Second row"))
    ]
    root = SemanticNode(
        node_id="doc",
        node_type="DOCUMENT_ROOT",
        title="Document",
        child_nodes=[
            SemanticNode(
                node_id="p",
                parent_id="doc",
                node_type="PARAGRAPH",
                title="Paragraph",
                total_content_pointers=[pointer("Overview")],
                child_nodes=[
                    SemanticNode(
                        node_id="inner",
                        parent_id="p",
                        node_type="TABLE",
                        title="Table",
                        child_nodes=rows,
                        level_from_root=2,
                    )
                ],
                level_from_root=1,
            ),
            SemanticNode(
                node_id="end",
                parent_id="doc",
                title="End",
                total_content_pointers=[pointer("Endnote")],
                level_from_root=1,
            ),
        ],
    )

    normalized = page_index_module._normalize_page_index_unary_nodes(root)
    validate_semantic_tree_integrity(normalized, parser_source_map=source_map)
    pending = [normalized]
    nodes = []
    while pending:
        node = pending.pop()
        nodes.append(node)
        pending.extend(node.child_nodes)
    descendants = {node.node_id for node in nodes}
    assert {"r1", "r2", "end"}.issubset(descendants)
    assert all(len(node.child_nodes) != 1 for node in nodes)
    assert compute_terminal_content_coverage(normalized, source_map)["valid"] is True


@pytest.mark.ci
@pytest.mark.parametrize("corrupt_pointer", [False, True])
def test_public_page_index_preserves_nested_rows_or_rejects_bad_grounding(
    monkeypatch: pytest.MonkeyPatch, corrupt_pointer: bool
) -> None:
    source = "Overview\nFirst row\nSecond row\nEndnote\n"

    def materialize_blocks(
        *, unit_id: str, parent_id: str, level_from_root: int, **_kwargs: object
    ) -> tuple[list[SemanticNode], int]:
        def pointer(excerpt: str) -> HydratedTextPointer:
            start = source.index(excerpt)
            return HydratedTextPointer(
                source_cluster_id=unit_id,
                start_char=start,
                end_char=start + len(excerpt) - 1,
                verbatim_text=("forged" if corrupt_pointer and excerpt == "First row" else excerpt),
            )

        rows = [
            SemanticNode(
                node_id=node_id,
                parent_id="inner",
                node_type="TABLE_ROW",
                title=excerpt,
                total_content_pointers=[pointer(excerpt)],
                level_from_root=level_from_root + 2,
            )
            for node_id, excerpt in (("r1", "First row"), ("r2", "Second row"))
        ]
        return [
            SemanticNode(
                node_id="p",
                parent_id=parent_id,
                node_type="PARAGRAPH",
                title="Overview",
                total_content_pointers=[pointer("Overview")],
                child_nodes=[
                    SemanticNode(
                        node_id="inner",
                        parent_id="p",
                        node_type="TABLE",
                        title="Table",
                        child_nodes=rows,
                        level_from_root=level_from_root + 1,
                    )
                ],
                level_from_root=level_from_root,
            ),
            SemanticNode(
                node_id="end",
                parent_id=parent_id,
                title="Endnote",
                total_content_pointers=[pointer("Endnote")],
                level_from_root=level_from_root,
            ),
        ], len(source)

    monkeypatch.setattr(page_index_module, "_materialize_block_tree", materialize_blocks)
    if corrupt_pointer:
        with pytest.raises(ValueError, match="pointer excerpt does not match authoritative source slice"):
            parse_page_index_document(
                document_id="nested-rows", title="Document", raw_text=source, mode="heuristic"
            )
        return

    result = parse_page_index_document(
        document_id="nested-rows", title="Document", raw_text=source, mode="heuristic"
    )
    pending = [result.semantic_tree]
    nodes = []
    while pending:
        node = pending.pop()
        nodes.append(node)
        pending.extend(node.child_nodes)
    assert {"r1", "r2", "end"}.issubset({node.node_id for node in nodes})
    assert all(len(node.child_nodes) != 1 for node in nodes)
    assert compute_terminal_content_coverage(
        result.semantic_tree, result.parser_source_map
    )["valid"] is True


@pytest.mark.ci
@pytest.mark.parametrize(
    ("parent_type", "own_text", "child_type", "child_text"),
    [
        ("HEADING", "Financial Results", "TABLE", "| Year | Revenue |\n| 2025 | 120 |"),
        ("PARAGRAPH", "Introduction:", "TABLE", "| A | B |\n| x | y |"),
        ("SECTION", "Experiment notes", "FIGURE", "[F1]"),
        (
            "HEADING",
            "Ledger",
            "TABLE",
            "| idx | val |\n" + "\n".join(f"| {index:02d} | value_{index:02d} |" for index in range(1, 52)),
        ),
    ],
)
def test_public_page_index_morphs_content_parent_without_splitting_atomic_child(
    monkeypatch: pytest.MonkeyPatch,
    parent_type: str,
    own_text: str,
    child_type: str,
    child_text: str,
) -> None:
    source = f"{own_text}\n{child_text}\n"

    def materialize_blocks(
        *, unit_id: str, parent_id: str, level_from_root: int, **_kwargs: object
    ) -> tuple[list[SemanticNode], int]:
        def pointer(excerpt: str) -> HydratedTextPointer:
            start = source.index(excerpt)
            return HydratedTextPointer(
                source_cluster_id=unit_id,
                start_char=start,
                end_char=start + len(excerpt) - 1,
                verbatim_text=excerpt,
            )

        parent = SemanticNode(
            node_id="content-parent",
            parent_id=parent_id,
            node_type=parent_type,
            title=own_text,
            total_content_pointers=[pointer(own_text)],
            child_nodes=[
                SemanticNode(
                    node_id="atomic-child",
                    parent_id="content-parent",
                    node_type=child_type,
                    title=child_type,
                    total_content_pointers=[pointer(child_text)],
                    level_from_root=level_from_root + 1,
                )
            ],
            level_from_root=level_from_root,
        )
        return [parent], len(source)

    monkeypatch.setattr(page_index_module, "_materialize_block_tree", materialize_blocks)
    result = parse_page_index_document(
        document_id="content-and-atomic-child",
        title="Document",
        raw_text=source,
        mode="heuristic",
    )
    pending = [result.semantic_tree]
    nodes = []
    while pending:
        node = pending.pop()
        nodes.append(node)
        pending.extend(node.child_nodes)
    atomic = next(node for node in nodes if node.node_id == "atomic-child")
    assert atomic.node_type == child_type
    assert atomic.child_nodes == []
    assert atomic.total_content_pointers[0].verbatim_text == child_text
    assert all(len(node.child_nodes) != 1 for node in nodes)
    assert compute_terminal_content_coverage(
        result.semantic_tree, result.parser_source_map
    )["valid"] is True


@pytest.mark.ci
def test_page_index_collapses_contentless_wrappers_without_losing_source() -> None:
    source = "First entry\nSecond entry\n"
    source_map = {"page_1": {"text": source}}

    def leaf(node_id: str, excerpt: str) -> SemanticNode:
        start = source.index(excerpt)
        return SemanticNode(
            node_id=node_id,
            parent_id="section",
            title=excerpt,
            total_content_pointers=[
                HydratedTextPointer(
                    source_cluster_id="page_1",
                    start_char=start,
                    end_char=start + len(excerpt) - 1,
                    verbatim_text=excerpt,
                )
            ],
            level_from_root=3,
        )

    root = SemanticNode(
        node_id="doc",
        node_type="DOCUMENT_ROOT",
        title="Doc",
        child_nodes=[
            SemanticNode(
                node_id="wrap",
                parent_id="doc",
                node_type="PAGE",
                title="Page",
                child_nodes=[
                    SemanticNode(
                        node_id="section",
                        parent_id="wrap",
                        node_type="SECTION",
                        title="Section",
                        child_nodes=[leaf("a", "First entry"), leaf("b", "Second entry")],
                        level_from_root=2,
                    )
                ],
                level_from_root=1,
            )
        ],
    )
    normalized = page_index_module._normalize_page_index_unary_nodes(root)
    validate_semantic_tree_integrity(normalized, parser_source_map=source_map)
    assert {node.node_id for node in normalized.child_nodes} == {"a", "b"}
    assert compute_terminal_content_coverage(normalized, source_map)["valid"] is True


@pytest.mark.ci
def test_atomic_fallback_does_not_claim_missing_content_from_parent_aggregate() -> None:
    source = "Heading\nBody"
    source_map = {"page_1": {"text": source}}
    parent = SemanticNode(
        node_id="page",
        node_type="PAGE",
        title="Page",
        aggregate_content_pointers=[
            HydratedTextPointer(
                source_cluster_id="page_1",
                start_char=0,
                end_char=len(source) - 1,
                verbatim_text=source,
            )
        ],
        child_nodes=[
            SemanticNode(
                node_id="heading",
                parent_id="page",
                node_type="HEADING_TEXT",
                title="Heading",
                total_content_pointers=[
                    HydratedTextPointer(
                        source_cluster_id="page_1",
                        start_char=0,
                        end_char=len("Heading") - 1,
                        verbatim_text="Heading",
                    )
                ],
                level_from_root=1,
            )
        ],
    )

    normalized = page_index_module._normalize_page_index_unary_nodes(parent)
    assert normalized.total_content_pointers[0].verbatim_text == "Heading"
    assert compute_terminal_content_coverage(normalized, source_map)["valid"] is False


@pytest.mark.ci
@pytest.mark.parametrize("corruption", ["missing", "duplicate"])
def test_public_page_index_rejects_invalid_terminal_ownership(
    monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    normalize = page_index_module._normalize_page_index_unary_nodes

    def corrupt_after_normalization(root: SemanticNode) -> SemanticNode:
        normalized = normalize(root)
        pending = [normalized]
        leaves = []
        while pending:
            node = pending.pop()
            if not node.child_nodes and node.total_content_pointers:
                leaves.append(node)
            pending.extend(node.child_nodes)
        assert len(leaves) >= 2
        if corruption == "missing":
            leaves[0].total_content_pointers = []
        else:
            leaves[1].total_content_pointers.append(leaves[0].total_content_pointers[0])
        return normalized

    monkeypatch.setattr(
        page_index_module, "_normalize_page_index_unary_nodes", corrupt_after_normalization
    )
    with pytest.raises(ValueError, match="terminal source ownership"):
        parse_page_index_document(
            document_id="ownership-check",
            title="Title",
            raw_text="# Title\n\nAlpha.\n\nBeta.",
            source_format="markdown",
            mode="heuristic",
        )


def _ollama_parser_model_params() -> list[object]:
    configured = os.getenv("KG_DOC_PARSER_OLLAMA_MODELS")
    models = [
        item.strip()
        for item in (configured.split(",") if configured else ("gemma4:e2b", "gemma4:latest"))
        if item.strip()
    ]
    return [pytest.param(model, id=model.replace(":", "-")) for model in models]


def _ollama_provider_limits() -> dict[str, int | float | None]:
    output_tokens = os.getenv("KG_DOC_PARSER_OLLAMA_MAX_OUTPUT_TOKENS")
    return {
        "max_retries": int(os.getenv("KG_DOC_PARSER_OLLAMA_MAX_RETRIES", "2")),
        "max_output_tokens": int(output_tokens) if output_tokens else None,
        "timeout_seconds": float(os.getenv("KG_DOC_PARSER_OLLAMA_TIMEOUT_SECONDS", "120")),
    }


def _node_signature(node) -> tuple[str, str, tuple]:
    return (
        node.node_type,
        node.title,
        tuple(_node_signature(child) for child in node.child_nodes),
    )


def _collect_node_types(node) -> set[str]:
    kinds = {node.node_type}
    for child in node.child_nodes:
        kinds.update(_collect_node_types(child))
    return kinds


def _install_long_table_assignment_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeStructured:
        def invoke(self, messages):
            return {
                "parsed": BlockAssignmentBatch(
                    assignments=[
                        BlockAssignment(
                            block_id="p0001-b001",
                            parent_id=None,
                            node_type="SECTION",
                            title="Sample Document Title",
                        ),
                        BlockAssignment(
                            block_id="p0001-b002",
                            parent_id="p0001-b001",
                            node_type="PARAGRAPH",
                            title="ID Description Value",
                        ),
                    ]
                )
            }

    class _FakeChat:
        def with_structured_output(self, schema, include_raw=True, **kwargs):
            assert schema is BlockAssignmentBatch
            return _FakeStructured()

    monkeypatch.setattr(page_index_provider_module, "build_chat_model_for_role", lambda *args, **kwargs: _FakeChat())
    monkeypatch.setattr(
        page_index_refinement_module,
        "build_chat_model_for_role",
        lambda *args, **kwargs: _FakeChat(),
    )


def _max_depth(node) -> int:
    if not node.child_nodes:
        return 1
    return 1 + max(_max_depth(child) for child in node.child_nodes)


def test_page_index_llm_structured_output_prefers_function_calling(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, object]] = []

    class _FakeStructured:
        def invoke(self, messages):
            return {
                "parsed": page_index_module.BlockAssignmentBatch(
                    assignments=[
                        page_index_module.BlockAssignment(
                            block_id="p0001-b001",
                            parent_id=None,
                            node_type="SECTION",
                            title="Root",
                        )
                    ]
                )
            }

    class _FakeChat:
        def with_structured_output(self, schema, include_raw=True, **kwargs):
            captured.append({"schema": schema.__name__, "include_raw": include_raw, **kwargs})
            return _FakeStructured()

    monkeypatch.setattr(page_index_provider_module, "build_chat_model_for_role", lambda *args, **kwargs: _FakeChat())

    provider_settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(provider="azure", model="gpt-5-nano", base_url="https://example.openai.azure.com/")
    )
    page_index_provider_module._llm_page_outline(
        page_text="# Root\n",
        page_number=1,
        source_format="markdown",
        provider_settings=provider_settings,
    )

    assert captured
    assert captured[0]["method"] == "json_schema"


def test_page_index_provider_diagnostics_include_attempt_context(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_page_index_chat(
        monkeypatch,
        assignment_payload=BlockAssignmentBatch(
            assignments=[
                BlockAssignment(
                    block_id="p0001-b001",
                    parent_id=None,
                    node_type="SECTION",
                    title="Root",
                )
            ]
        ),
    )
    diagnostics: list[dict[str, object]] = []
    settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(provider="ollama", model="fake", base_url="http://127.0.0.1:11434")
    )

    page_index_provider_module._llm_page_outline(
        page_text="# Root\n",
        page_number=1,
        source_format="markdown",
        provider_settings=settings,
        provider_diagnostics_sink=diagnostics.append,
    )

    assert len(diagnostics) == 1
    record = diagnostics[0]
    assert record["operation"] == "page_index_assignment"
    assert record["call_role"] == "proposal"
    assert record["strategy"] == "page_index"
    assert record["attempt_index"] == 1
    assert record["success"] is True
    assert isinstance(record["elapsed_ms"], int)


@pytest.mark.parametrize(
    "provider",
    ["fake", "ollama", "openai", "azure", "gemini", "vertex", "codex"],
)
def test_all_declared_page_index_modes_share_semantic_path_with_fake_provider(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    _install_fake_page_index_chat(
        monkeypatch,
        assignment_payload=BlockAssignmentBatch(
            assignments=[
                BlockAssignment(
                    block_id="p0001-b001",
                    parent_id=None,
                    node_type="SECTION",
                    title="Root",
                )
            ]
        ),
    )
    diagnostics: list[dict[str, object]] = []
    settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(provider=provider, model="test-model")
    )

    result = parse_page_index_document(
        document_id=f"mode-{provider}",
        title="Mode Matrix",
        raw_text="# Root\n",
        source_format="markdown",
        mode=provider,  # type: ignore[arg-type]
        provider_settings=settings,
        provider_diagnostics_sink=diagnostics.append,
    )

    assert result.semantic_tree.child_nodes == []
    assert result.semantic_tree.metadata["atomic_retained"] is True
    assert diagnostics
    assert {record["operation"] for record in diagnostics} == {"page_index_assignment"}
    assert all(record["strategy"] == "page_index" for record in diagnostics)
    assert all(record["success"] is True for record in diagnostics)


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
    assert all(page.parent_id == result.semantic_tree.node_id for page in result.semantic_tree.child_nodes)
    assert len({page.node_id for page in result.semantic_tree.child_nodes}) == 2
    restored_tree = SemanticNode.model_validate(result.semantic_tree.model_dump())
    assert all(page.parent_id == restored_tree.node_id for page in restored_tree.child_nodes)
    assert result.authoritative_source_map.keys() == result.parser_source_map.keys()
    assert result.coverage["overall"] > 0.95

    node_types = _collect_node_types(result.semantic_tree)
    assert "PAGE" in node_types
    assert "HEADING" in node_types
    assert "HEADING_TEXT" in node_types
    assert "PARAGRAPH" in node_types
    assert "TERM" in node_types
    assert _max_depth(result.semantic_tree) >= 6

    payload = semantic_tree_to_kge_payload(
        result.semantic_tree,
        doc_id=result.workflow_input.request_id,
        parser_source_map=result.parser_source_map,
    )
    assert len(payload["nodes"]) >= 8
    assert len(payload["edges"]) >= 4


@pytest.mark.ci
def test_page_index_provider_accepts_atomic_long_table_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_text = _fixture_text("title_long_table.md")
    _install_long_table_assignment_provider(monkeypatch)
    provider_settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(provider="ollama", model="fake", base_url="http://127.0.0.1:11434")
    )

    result = parse_page_index_document(
        document_id="page-index-title-long-table",
        title="Sample Document Title",
        raw_text=raw_text,
        source_format="markdown",
        mode="ollama",
        provider_settings=provider_settings,
        summary_enabled=False,
    )
    assert result.diagnostics["assignment_mode"] != "deterministic_fallback"
    assert not any("too broad" in error for error in result.diagnostics.get("validation_errors", []))
    assert result.coverage["overall"] == pytest.approx(1.0)
    assert result.semantic_tree.child_nodes


@pytest.mark.ci
def test_page_index_marks_markdown_tables_as_explicit_table_nodes() -> None:
    raw_text = "# Revenue\n\n| Year | Value |\n| --- | --- |\n| 2024 | 10 |\n| 2025 | 12 |\n"
    result = parse_page_index_document(
        document_id="page-index-table-kind",
        title="Revenue",
        raw_text=raw_text,
        source_format="markdown",
        mode="heuristic",
        summary_enabled=False,
    )
    node_types = _collect_node_types(result.semantic_tree)
    assert "TABLE" in node_types
    def _find_tables(node):
        matches = [node] if node.node_type == "TABLE" else []
        for child in node.child_nodes:
            matches.extend(_find_tables(child))
        return matches

    table_nodes = _find_tables(result.semantic_tree)
    assert table_nodes
    assert table_nodes[0].total_content_pointers


@pytest.mark.ci
def test_page_index_display_excerpt_is_optional_and_cannot_replace_grounding() -> None:
    candidate = CandidateBlock(
        block_id="p0001-b001",
        page_number=1,
        order=1,
        start_char=0,
        end_char=10,
        line_start=1,
        line_end=1,
        indent=0,
        kind_hint="paragraph",
        confidence=0.8,
        text="Alpha Beta",
        node_type_hint="PARAGRAPH",
        title_hint="Alpha",
    )
    assignment = BlockAssignment(
        block_id=candidate.block_id,
        parent_id=None,
        node_type="PARAGRAPH",
        title="Alpha",
    )
    valid = page_index_candidates_module._validate_page_index_block_structure(
        candidates=[candidate],
        assignments=[assignment],
        block_specs=[
            PageIndexBlockSpec(
                title="Alpha",
                node_type="PARAGRAPH",
                excerpt="Alpha Beta",
                display_excerpt="Beta",
            )
        ],
        page_text="Alpha Beta",
    )
    assert valid.valid
    invalid = page_index_candidates_module._validate_page_index_block_structure(
        candidates=[candidate],
        assignments=[assignment],
        block_specs=[
            PageIndexBlockSpec(
                title="Alpha",
                node_type="PARAGRAPH",
                excerpt="Alpha Beta",
                display_excerpt="Gamma",
            )
        ],
        page_text="Alpha Beta",
    )
    assert not invalid.valid
    assert any("display excerpt" in error for error in invalid.errors)


@pytest.mark.manual
def test_manual_plain_text_short_title_with_full_page_table_stays_grounded() -> None:
    raw_text = _fixture_text("manual_short_title_full_table.txt")
    candidates = page_index_module._extract_candidate_blocks(
        raw_text,
        page_number=1,
        source_format="text",
    )

    assert len(candidates) == 2
    assert candidates[0].node_type_hint == "HEADING"
    assert candidates[0].title_hint == "AI Chip Revenue"
    assert candidates[1].node_type_hint == "PARAGRAPH"
    assert candidates[1].kind_hint == "paragraph_table_like"
    assert candidates[1].text.startswith("Metric 2023 2024 2025 2026")
    assert candidates[1].text.count("\n") >= 25

    result = parse_page_index_document(
        document_id="manual-short-title-table",
        title="AI Chip Revenue",
        raw_text=raw_text,
        source_format="text",
        mode="heuristic",
        summary_enabled=True,
    )
    page = result.semantic_tree
    assert [node.node_type for node in page.child_nodes] == ["HEADING_TEXT", "PARAGRAPH"]
    heading = page.child_nodes[0]
    assert heading.title == "AI Chip Revenue"
    table_node = page.child_nodes[1]
    assert table_node.total_content_pointers
    assert table_node.total_content_pointers[0].verbatim_text.startswith(
        "Metric 2023 2024 2025 2026"
    )
    assert result.coverage["overall"] > 0.95


@pytest.mark.manual
def test_manual_irregular_table_pages_keep_page_boundaries_and_one_layer_contract() -> None:
    raw_text = _fixture_text("manual_irregular_table_pages.txt")
    result = parse_page_index_document(
        document_id="manual-irregular-table-pages",
        title="Operations Snapshot",
        raw_text=raw_text,
        source_format="text",
        mode="heuristic",
        summary_enabled=False,
    )

    assert len(result.semantic_tree.child_nodes) == 2
    assert [page.title for page in result.semantic_tree.child_nodes] == ["Page 1", "Page 2"]
    assert result.authoritative_source_map.keys() == result.parser_source_map.keys()
    assert all(page.child_nodes for page in result.semantic_tree.child_nodes)
    assert all(
        pointer.source_cluster_id in result.authoritative_source_map
        for page in result.semantic_tree.child_nodes
        for node in page.child_nodes
        for pointer in node.total_content_pointers
    )

    first_page = result.semantic_tree.child_nodes[0]
    table_parent = first_page.child_nodes[1]
    candidates = parse_page_index_layer(
        parent_id=table_parent.node_id,
        parent_title=table_parent.title,
        parent_pointers=table_parent.total_content_pointers,
        parser_source_map=result.parser_source_map,
        source_format="text",
        summary_enabled=False,
    )
    assert candidates
    assert all(candidate.metadata["page_index_layer_only"] is True for candidate in candidates)
    assert all(candidate.parent_node_id == table_parent.node_id for candidate in candidates)
    assert all(
        pointer.source_cluster_id == table_parent.total_content_pointers[0].source_cluster_id
        for candidate in candidates
        for pointer in candidate.total_content_pointers
    )


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
def test_page_index_summary_can_be_disabled_without_changing_grounding() -> None:
    result = parse_page_index_document(
        document_id="page-index-summary-disabled",
        title="Page Index Document",
        raw_text="# Results\n\nIntro.\n",
        source_format="markdown",
        mode="heuristic",
        summary_enabled=False,
    )
    results = result.semantic_tree
    assert results.summary == ""
    assert results.child_nodes[0].total_content_pointers[0].verbatim_text == "# Results"

    payload = semantic_tree_to_kge_payload(
        result.semantic_tree,
        doc_id=result.workflow_input.request_id,
        parser_source_map=result.parser_source_map,
    )
    persisted = next(node for node in payload["nodes"] if node["id"] == results.node_id)
    assert persisted["summary"] == ""
    assert persisted["metadata"]["summary_unavailable"] is True


@pytest.mark.manual
@pytest.mark.llm_real
@pytest.mark.requires_ollama
@pytest.mark.parametrize(
    "fixture_name, source_format",
    [
        pytest.param("sample_page_index.txt", "text", id="text"),
        pytest.param("sample_page_index.md", "markdown", id="markdown"),
    ],
)
@pytest.mark.parametrize(
    "parser_model",
    _ollama_parser_model_params(),
)
def test_page_index_ollama_smoke_parses_text_and_markdown(
    fixture_name: str,
    source_format: str,
    parser_model: str,
) -> None:
    pytest.importorskip("langchain_ollama")
    base_url = os.getenv("KG_DOC_PARSER_BASE_URL", "http://127.0.0.1:11434")
    available, reason = ollama_available(base_url, parser_model)
    if not available:
        pytest.skip(f"ollama parser unavailable: {reason}")
    limits = _ollama_provider_limits()
    provider_settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(
            provider="ollama",
            model=parser_model,
            base_url=base_url,
            **limits,
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
        if any(token in message for token in ("connect", "connection", "refused", "timed out")):
            pytest.skip(f"ollama parser unavailable: {exc}")
        raise

    assert result.mode == "ollama"
    assert result.source_format == source_format
    assert len(result.semantic_tree.child_nodes) == 2
    assert result.authoritative_source_map.keys() == result.parser_source_map.keys()
    assert result.coverage["overall"] > 0.80


@pytest.mark.manual
@pytest.mark.llm_real
@pytest.mark.requires_ollama
@pytest.mark.parametrize(
    "fixture_name, source_format",
    [
        pytest.param("sample_page_index.txt", "text", id="text"),
        pytest.param("sample_page_index.md", "markdown", id="markdown"),
    ],
)
@pytest.mark.parametrize(
    "parser_model",
    _ollama_parser_model_params(),
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
    base_url = os.getenv("KG_DOC_PARSER_BASE_URL", "http://127.0.0.1:11434")
    available, reason = ollama_available(base_url, parser_model)
    if not available:
        pytest.skip(f"ollama workflow ingest unavailable: {reason}")
    limits = _ollama_provider_limits()
    scratch = _scratch("workflow_ingest_ollama")
    workflow_engine, conversation_engine, knowledge_engine = build_workflow_engine_triplet(scratch / "engines", "in_memory")
    raw_text = _fixture_text(fixture_name)
    provider_settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(
            provider="ollama",
            model=parser_model,
            base_url=base_url,
            **limits,
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
        if any(token in message for token in ("connect", "connection", "refused", "timed out")):
            pytest.skip(f"ollama workflow ingest unavailable: {exc}")
        raise

    assert run.status == "succeeded"
    assert bundle is not None
    assert bundle.graph_payload["doc_id"] == parsed.workflow_input.request_id
    assert len(bundle.graph_payload["nodes"]) >= 1
