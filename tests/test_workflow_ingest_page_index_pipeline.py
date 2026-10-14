from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import kg_doc_parser.workflow_ingest.page_index as page_index_module
import pytest
from _kogwistar_test_helpers import (
    build_workflow_engine_triplet,
    drain_phase1_indexes_until_idle,
)
from _ollama_test_helpers import ollama_available
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
from kg_doc_parser.workflow_ingest.models import CurrentLayerResult, LayerChildCandidate
from kg_doc_parser.workflow_ingest.parser_core import commit_layer_children
from kg_doc_parser.workflow_ingest.semantics import (
    HydratedTextPointer,
    SemanticNode,
    compute_terminal_content_coverage,
    semantic_tree_to_kge_payload,
    validate_semantic_tree_integrity,
)
from kg_doc_parser.workflow_ingest.service import run_ingest_workflow
from kogwistar.utils.cache_backend import Memory

pytestmark = [pytest.mark.workflow]


def _fixture_text(name: str) -> str:
    return (Path(__file__).parent / "fixtures" / "page_index" / name).read_text(encoding="utf-8")


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

    monkeypatch.setattr(page_index_module, "build_chat_model_for_role", lambda *args, **kwargs: _FakeChat())


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
    capture_messages: list[object] | None = None,
) -> None:
    class _FakeStructured:
        def __init__(self, schema):
            self.schema = schema

        def invoke(self, messages):
            if capture_messages is not None:
                capture_messages.extend(messages)
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

    monkeypatch.setattr(page_index_module, "build_chat_model_for_role", lambda *args, **kwargs: _FakeChat())

    provider_settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(provider="azure", model="gpt-5-nano", base_url="https://example.openai.azure.com/")
    )
    page_index_module._llm_page_outline(
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

    page_index_module._llm_page_outline(
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


def test_page_index_hierarchical_summary_passes_parent_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_text = "# Root\n\nRoot detail.\n\n## Child\n\nChild detail.\n"
    candidates = page_index_module._extract_candidate_blocks(
        raw_text,
        page_number=1,
        source_format="markdown",
    )
    assignments = page_index_module._deterministic_block_assignments(candidates)
    assignment_payload = BlockAssignmentBatch(
        assignments=[assignment.model_copy(update={"summary": f"initial-{index}"}) for index, assignment in enumerate(assignments)]
    )
    captured: list[object] = []

    class _SummaryStructured:
        def __init__(self, schema):
            self.schema = schema

        def invoke(self, messages):
            captured.extend(messages)
            if self.schema is page_index_module.BlockAssignmentBatch:
                return {"parsed": assignment_payload}
            if self.schema is page_index_module.HierarchicalSummaryBatch:
                prompt = str(messages[-1].content)
                entries = json.loads(prompt.split("Blocks: ", 1)[1])
                return {
                    "parsed": page_index_module.HierarchicalSummaryBatch(
                        assignments=[
                            page_index_module.HierarchicalSummaryAssignment(
                                path_id=str(entry["path_id"]),
                                summary=f"summary-{entry['path_id']}",
                            )
                            for entry in entries
                        ]
                    )
                }
            raise AssertionError(f"unexpected structured schema: {self.schema!r}")

    class _SummaryChat:
        def with_structured_output(self, schema, include_raw=True, **kwargs):
            return _SummaryStructured(schema)

    monkeypatch.setattr(page_index_module, "build_chat_model_for_role", lambda *args, **kwargs: _SummaryChat())
    diagnostics: list[dict[str, object]] = []
    settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(provider="ollama", model="test-model")
    )

    result = parse_page_index_document(
        document_id="hierarchical-summary",
        title="Hierarchy",
        raw_text=raw_text,
        source_format="markdown",
        mode="ollama",
        provider_settings=settings,
        provider_diagnostics_sink=diagnostics.append,
        hierarchical_summary_enabled=True,
    )

    assert result.semantic_tree.child_nodes
    page_diagnostics = result.diagnostics["page_diagnostics"][0]
    assert page_diagnostics["hierarchical_summary_enabled"] is True
    assert page_diagnostics["hierarchical_summary_accepted"] > 0
    assert any(record["operation"] == "page_index_hierarchical_summary" for record in diagnostics)
    assert any(
        '"parent_summary": "summary-' in str(message.content)
        for message in captured
        if hasattr(message, "content")
    )


def test_page_index_hierarchical_summary_provider_failure_keeps_initial_summaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_text = "# Root\n\nRoot detail.\n\n## Child\n\nChild detail.\n"
    candidates = page_index_module._extract_candidate_blocks(
        raw_text,
        page_number=1,
        source_format="markdown",
    )
    assignments = page_index_module._deterministic_block_assignments(candidates)
    initial_summaries = [assignment.summary for assignment in assignments]

    _install_fake_page_index_chat(
        monkeypatch,
        assignment_payload=BlockAssignmentBatch(assignments=assignments),
    )
    outline_builder = page_index_module.build_chat_model_for_role
    call_count = 0

    def _raise_on_summary_model(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return outline_builder(*args, **kwargs)
        raise RuntimeError("summary model unavailable")

    monkeypatch.setattr(page_index_module, "build_chat_model_for_role", _raise_on_summary_model)
    diagnostics: list[dict[str, object]] = []
    settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(provider="ollama", model="test-model")
    )

    result = parse_page_index_document(
        document_id="hierarchical-summary-fallback",
        title="Hierarchy",
        raw_text=raw_text,
        source_format="markdown",
        mode="ollama",
        provider_settings=settings,
        provider_diagnostics_sink=diagnostics.append,
        hierarchical_summary_enabled=True,
    )

    actual_summaries: list[str] = []

    def _collect_summaries(node) -> None:
        if node.node_type != "DOCUMENT_ROOT":
            actual_summaries.append(node.summary)
        for child in node.child_nodes:
            _collect_summaries(child)

    _collect_summaries(result.semantic_tree)
    assert actual_summaries
    assert all(not summary.startswith("summary-") for summary in actual_summaries)
    assert all(not summary.startswith("summary-") for summary in initial_summaries)
    page_diagnostics = result.diagnostics["page_diagnostics"][0]
    assert page_diagnostics["hierarchical_summary_fallback"] is True
    assert page_diagnostics["hierarchical_summary_accepted"] == 0
    summary_failures = [
        record for record in diagnostics if record["operation"] == "page_index_hierarchical_summary"
    ]
    assert summary_failures
    assert summary_failures[0]["failure_type"] == "transport/provider_exception"


def test_page_index_hierarchical_summary_requires_ordinary_summaries(
    monkeypatch: pytest.MonkeyPatch,
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
    settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(provider="ollama", model="test-model")
    )

    result = parse_page_index_document(
        document_id="hierarchical-summary-disabled",
        title="Hierarchy",
        raw_text="# Root\n\nRoot detail.\n",
        source_format="markdown",
        mode="ollama",
        provider_settings=settings,
        summary_enabled=False,
        hierarchical_summary_enabled=True,
    )

    page_diagnostics = result.diagnostics["page_diagnostics"][0]
    assert page_diagnostics["hierarchical_summary_enabled"] is False
    assert page_diagnostics["hierarchical_summary_call_count"] == 0


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
def test_page_index_layer_refines_only_direct_children_of_one_parent() -> None:
    source_text = "# Parent\n\nParent introduction.\n\n## Child\n\nChild detail.\n"
    candidates = parse_page_index_layer(
        parent_id="parent",
        parent_title="Parent",
        parent_pointers=[
            HydratedTextPointer(
                source_cluster_id="unit-1",
                start_char=0,
                end_char=len(source_text) - 1,
                verbatim_text=source_text,
            )
        ],
        parser_source_map={"unit-1": {"text": source_text}},
        source_format="markdown",
    )

    assert candidates
    assert {candidate.parent_node_id for candidate in candidates} == {"parent"}
    assert all(candidate.metadata["page_index_layer_only"] is True for candidate in candidates)
    child_heading = next(candidate for candidate in candidates if candidate.title == "Child")
    assert child_heading.expandable is True
    assert [child.title for child in child_heading.child_candidates] == [
        "Child",
        "Child detail.",
    ]
    assert child_heading.child_candidates[0].node_type == "HEADING_TEXT"
    assert child_heading.child_candidates[0].parent_node_id == child_heading.node_id
    # Only the direct title/content leaves are materialized. Deeper sections
    # remain available for the child's later frontier refinement.
    assert all(candidate.title != "Child detail." for candidate in candidates)


@pytest.mark.ci
def test_page_index_materialized_title_leaf_survives_layer_commit() -> None:
    pointer = HydratedTextPointer(
        source_cluster_id="unit-1",
        start_char=0,
        end_char=4,
        verbatim_text="Title",
    )
    heading = LayerChildCandidate(
        node_id="heading",
        parent_node_id="root",
        title="Title",
        node_type="HEADING",
        total_content_pointers=[pointer],
        child_candidates=[
            LayerChildCandidate(
                node_id="heading-text",
                parent_node_id="heading",
                title="Title",
                node_type="HEADING_TEXT",
                total_content_pointers=[pointer],
                expandable=False,
            ),
            LayerChildCandidate(
                node_id="heading-body",
                parent_node_id="heading",
                title="Body",
                node_type="TEXT_FLOW",
                total_content_pointers=[pointer],
                expandable=False,
            ),
        ],
    )
    sibling = LayerChildCandidate(
        node_id="heading-sibling",
        parent_node_id="root",
        title="Other",
        node_type="TEXT_FLOW",
        total_content_pointers=[
            HydratedTextPointer(
                source_cluster_id="unit-1",
                start_char=5,
                end_char=9,
                verbatim_text="Other",
            )
        ],
    )
    committed = commit_layer_children(
        semantic_tree=SemanticNode(node_id="root", parent_id=None, title="Root", node_type="ROOT"),
        current_layer_result=CurrentLayerResult(children=[heading, sibling], satisfied=True),
        current_depth=0,
    )
    assert committed.child_nodes[0].title == "Title"
    assert committed.child_nodes[0].child_nodes[0].node_type == "HEADING_TEXT"


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
    valid = page_index_module._validate_page_index_block_structure(
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
    invalid = page_index_module._validate_page_index_block_structure(
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


def test_page_index_candidate_extraction_and_validation() -> None:
    raw_text = "# Root\n\nIntro paragraph.\n\n## Child\n\n- Term item\n"
    candidates = page_index_module._extract_candidate_blocks(raw_text, page_number=1, source_format="markdown")

    assert [candidate.node_type_hint for candidate in candidates] == ["HEADING", "PARAGRAPH", "HEADING", "TERM"]
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

    candidates = page_index_module._extract_candidate_blocks(raw_text, page_number=1, source_format="text")

    assert len(candidates) == 2
    assert all(candidate.node_type_hint == "PARAGRAPH" for candidate in candidates)


def test_page_index_extracts_dense_clause_candidates() -> None:
    raw_text = "1.2.3 Findings\n\n(a) Alpha clause\n\n(i) Roman item\n"

    candidates = page_index_module._extract_candidate_blocks(raw_text, page_number=1, source_format="text")

    assert [candidate.node_type_hint for candidate in candidates] == ["HEADING", "TERM", "TERM"]
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

    validation = page_index_module._validate_block_assignments(candidates, assignments, page_text="Shared content")

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

    validation = page_index_module._validate_block_assignments(
        candidates,
        assignments,
        page_text="Shared content",
    )

    assert validation.valid is False
    assert any("reuses overlapping source occurrence" in error for error in validation.errors)


def test_page_index_nested_validation_reports_only_the_failing_descendant() -> None:
    page_text = "# Root\n\nChild content.\n"
    candidates = page_index_module._extract_candidate_blocks(
        page_text,
        page_number=1,
        source_format="markdown",
    )
    assignments = page_index_module._deterministic_block_assignments(candidates)
    specs = [
        page_index_module.PageIndexBlockSpec(
            title="Root",
            node_type="SECTION",
            excerpt="# Root",
            source_role="heading",
            child_nodes=[
                page_index_module.PageIndexBlockSpec(
                    title="Child content.",
                    node_type="PARAGRAPH",
                    excerpt="",
                    source_role="content",
                )
            ],
        )
    ]

    validation = page_index_module._validate_page_index_block_structure(
        candidates=candidates,
        assignments=assignments,
        block_specs=specs,
        page_text=page_text,
    )

    assert validation.valid is False
    assert "block 1.1 has empty excerpt" in validation.errors
    assert "block 1 has empty excerpt" not in validation.errors


def test_page_index_resolve_pointer_keeps_exact_match_unchanged() -> None:
    page_text = "Alpha beta gamma."

    resolved = page_index_module._resolve_pointer(
        unit_id="p0001",
        page_text=page_text,
        excerpt="Alpha beta",
        start_at=0,
    )

    assert resolved.start_char == 0
    assert resolved.end_char == len("Alpha beta") - 1
    assert resolved.verbatim_text == "Alpha beta"


def test_page_index_resolve_pointer_repairs_small_typo_near_origin() -> None:
    page_text = "Alpha beta gamma delta epsilon."

    resolved = page_index_module._resolve_pointer(
        unit_id="p0001",
        page_text=page_text,
        excerpt="Alpha beta gamma deltx epsilon",
        start_at=0,
    )

    assert resolved.start_char == 0
    assert resolved.verbatim_text == "Alpha beta gamma delta epsilon"
    assert resolved.verbatim_text != "Alpha beta gamma deltx epsilon"


def test_page_index_resolve_pointer_prefers_nearest_fuzzy_match() -> None:
    page_text = "Alpha beta gamma delta epsilon one. Noise. Alpha beta gamma delta epsilon two."

    resolved = page_index_module._resolve_pointer(
        unit_id="p0001",
        page_text=page_text,
        excerpt="Alpha beta gamma deltx epsilon",
        start_at=50,
    )

    assert resolved.verbatim_text == "Alpha beta gamma delta epsilon"
    assert resolved.start_char == page_text.index("Alpha beta gamma delta epsilon", 20)


def test_page_index_resolve_pointer_rejects_low_similarity_typo() -> None:
    with pytest.raises(ValueError, match="unable to resolve excerpt"):
        page_index_module._resolve_pointer(
            unit_id="p0001",
            page_text="Alpha beta gamma.",
            excerpt="zzzzzz",
            start_at=0,
        )


def test_page_index_resolve_pointer_rejects_whole_page_fuzzy_repair() -> None:
    page_text = "Alpha beta"

    with pytest.raises(ValueError, match="unable to resolve excerpt"):
        page_index_module._resolve_pointer(
            unit_id="p0001",
            page_text=page_text,
            excerpt="Alpah beta",
            start_at=0,
        )


def test_page_index_materialize_records_fuzzy_repair_diagnostics() -> None:
    page_text = "Alpha beta gamma delta epsilon."
    block_specs = [
        page_index_module.PageIndexBlockSpec(
            title="Alpha",
            node_type="SECTION",
            excerpt="Alpha beta gamma deltx epsilon",
        )
    ]
    repair_stats = {"pointer_fuzzy_repairs": 0, "pointer_fuzzy_failures": 0}

    nodes, cursor = page_index_module._materialize_block_tree(
        block_specs=block_specs,
        page_text=page_text,
        unit_id="p0001",
        parent_id="root",
        level_from_root=1,
        repair_stats=repair_stats,
    )

    assert cursor > 0
    assert len(nodes) == 1
    assert nodes[0].total_content_pointers[0].verbatim_text == "Alpha beta gamma delta epsilon"
    assert repair_stats["pointer_fuzzy_repairs"] == 1
    assert repair_stats["pointer_fuzzy_failures"] == 0


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
    root = next(node for node in result.semantic_tree.child_nodes if node.title == "Root")
    assert root.node_type == "HEADING_TEXT"
    heading_leaf = root
    assert heading_leaf.node_type == "HEADING_TEXT"
    assert heading_leaf.total_content_pointers[0].verbatim_text == "# Root"
    child = next(node for node in result.semantic_tree.child_nodes if node.title == "Child")
    assert child.child_nodes[-1].title == "Term item"


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
    assert result.semantic_tree.child_nodes[0].title == "Root"


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
    assert result.semantic_tree.child_nodes[0].title == "Root"


def test_page_index_ollama_salvages_valid_branches_after_local_assignment_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_text = "# Root\n\nIntro paragraph.\n\n## Child\n\n- Term item\n"
    assignments = BlockAssignmentBatch(
        assignments=[
            BlockAssignment(block_id="p0001-b001", parent_id=None, node_type="SECTION", title="Provider Root"),
            BlockAssignment(
                block_id="p0001-b002",
                parent_id="p0001-b001",
                node_type="PARAGRAPH",
                title="Provider intro",
            ),
            BlockAssignment(block_id="p0001-b003", parent_id="p0001-b001", node_type="SUBSECTION", title="Provider child"),
            BlockAssignment(
                block_id="p0001-b004",
                parent_id="missing-parent",
                node_type="TERM",
                title="Provider term",
            ),
        ]
    )

    class _FakeStructured:
        def invoke(self, messages):
            return {"parsed": assignments}

    class _FakeChat:
        def with_structured_output(self, schema, include_raw=True):
            return _FakeStructured()

    monkeypatch.setattr(page_index_module, "build_chat_model_for_role", lambda *args, **kwargs: _FakeChat())

    provider_settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(provider="ollama", model="fake", base_url="http://127.0.0.1:11434")
    )
    result = parse_page_index_document(
        document_id="page-index-ollama-branch-salvage",
        title="Page Index Document",
        raw_text=raw_text,
        source_format="markdown",
        mode="ollama",
        provider_settings=provider_settings,
    )

    assert result.diagnostics["assignment_mode"] == "ollama_branch_local_salvage", result.diagnostics
    assert result.diagnostics["branch_local_salvage"][0]["invalid_block_ids"] == ["p0001-b004"]
    root = next(node for node in result.semantic_tree.child_nodes if node.title == "Provider Root")
    assert root.node_type == "HEADING_TEXT"
    child = next(node for node in result.semantic_tree.child_nodes if node.title == "Provider child")
    term = next(node for node in child.child_nodes if node.title == "Term item")
    assert term.total_content_pointers[0].verbatim_text == "- Term item"


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

    root = result.semantic_tree
    baseline_root = baseline_result.semantic_tree
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

    root = result.semantic_tree
    baseline_root = baseline_result.semantic_tree
    assert root.total_content_pointers
    assert baseline_root.total_content_pointers
    root_heading = root.child_nodes[0]
    baseline_heading = baseline_root.child_nodes[0]
    assert root_heading.node_type == "HEADING_TEXT"
    assert root_heading.total_content_pointers[0].verbatim_text == baseline_heading.total_content_pointers[0].verbatim_text
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

    root = result.semantic_tree
    baseline_root = baseline_result.semantic_tree
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

    class _UsageCallback:
        recorded_call_count = 0

        def __init__(self) -> None:
            self.call_keys: list[str] = []

        def record_untracked_call(self, call_key: str) -> None:
            self.call_keys.append(call_key)
            self.recorded_call_count += 1

    usage_callback = _UsageCallback()
    provider_settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(provider="openai", model="fake")
    )
    result = parse_page_index_document(
        document_id="page-index-openai-refine-counts",
        title="Page Index Document",
        raw_text=raw_text,
        source_format="markdown",
        mode="openai",
        provider_settings=provider_settings,
        callbacks=[usage_callback],
        refine_excerpts=True,
    )

    assert result.diagnostics["refine_excerpts_enabled"] is True
    assert result.diagnostics["refine_excerpts_attempted"] == 2
    assert result.diagnostics["refine_excerpts_accepted"] == 1
    assert result.diagnostics["refine_excerpts_rejected"] == 0
    assert result.diagnostics["refine_excerpts_fallback"] is False
    assert usage_callback.call_keys == ["page-1-first", "refine-page-1"]


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
    assert result.semantic_tree.child_nodes[0].title == "Root"

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


@pytest.mark.ci
def test_page_index_heading_container_projects_source_text_to_leaf() -> None:
    result = parse_page_index_document(
        document_id="page-index-heading-leaf",
        title="Page Index Document",
        raw_text="# Results\n\nIntro.\n\n## Measurements\n\nBody.\n",
        source_format="markdown",
        mode="heuristic",
    )

    page = result.semantic_tree
    heading = page.child_nodes[0]
    assert heading.node_type == "HEADING_TEXT"
    assert heading.title == "Results"
    assert heading.metadata["page_index_role"] == "heading_text"
    assert heading.total_content_pointers[0].verbatim_text == "# Results"
    assert heading.summary == ""
    assert heading.metadata["summary_unavailable"] is True

    measurements = next(node for node in page.child_nodes if node.title == "Measurements")
    assert measurements.node_type == "HEADING"
    assert measurements.total_content_pointers == []
    assert measurements.aggregate_content_pointers[0].verbatim_text == "## Measurements\n\nBody."
    measurement_heading = measurements.child_nodes[0]
    assert measurement_heading.node_type == "HEADING_TEXT"
    assert measurement_heading.total_content_pointers[0].verbatim_text == "## Measurements"


@pytest.mark.ci
def test_page_index_repeated_heading_titles_have_distinct_stable_ids() -> None:
    result = parse_page_index_document(
        document_id="page-index-repeated-heading",
        title="Page Index Document",
        raw_text="# Repeat\n\nFirst body.\n\n# Repeat\n\nSecond body.\n",
        source_format="markdown",
        mode="heuristic",
    )

    page = result.semantic_tree
    repeated = [
        node
        for node in page.child_nodes
        if node.metadata.get("page_index_role") == "heading_container" and node.title == "Repeat"
    ]

    assert len(repeated) == 2
    assert repeated[0].node_id != repeated[1].node_id
    assert repeated[0].total_content_pointers == []
    assert repeated[1].total_content_pointers == []


@pytest.mark.ci
def test_page_index_heading_ids_ignore_title_edits_at_same_source_occurrence() -> None:
    def materialize(title: str):
        spec = PageIndexBlockSpec(
            title=title,
            node_type="HEADING",
            excerpt="# Stable occurrence",
            source_role="heading",
        )
        nodes, _ = page_index_module._materialize_block_tree(
            block_specs=[spec],
            page_text="# Stable occurrence",
            unit_id="unit-1",
            parent_id="page-1",
            level_from_root=2,
        )
        return nodes[0]

    original = materialize("Original title")
    edited = materialize("Edited title")

    assert original.node_id == edited.node_id
    assert original.metadata["page_index_role"] == "heading_leaf"
    assert original.total_content_pointers == edited.total_content_pointers
    assert original.total_content_pointers == edited.total_content_pointers

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
