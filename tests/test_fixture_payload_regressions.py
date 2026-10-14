from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import pytest

import kg_doc_parser.workflow_ingest.page_index as page_index_module
from kg_doc_parser.workflow_ingest.cache import WorkflowLLMCallCache
from kg_doc_parser.workflow_ingest.models import (
    CurrentLayerContext,
    CurrentLayerResult,
    LayerChildCandidate,
    ParseSessionState,
)
from kg_doc_parser.workflow_ingest.page_index import (
    parse_page_index_document,
    parse_page_index_layer,
)
from kg_doc_parser.workflow_ingest.parser_core import (
    commit_layer_children,
    initialize_parse_session,
    propose_layer_breakdown,
    source_map_fingerprint,
    switch_split_strategy,
)
from kg_doc_parser.workflow_ingest.semantics import (
    HydratedTextPointer,
    SemanticNode,
    compute_terminal_content_coverage,
    validate_semantic_tree_integrity,
)
from kg_doc_parser.workflow_ingest.strategy import (
    HARD_CODED_STRATEGY_PRIORITY,
    ParseStrategyAssessment,
    ParseStrategyTriage,
    select_parse_strategy,
)

pytestmark = [pytest.mark.workflow, pytest.mark.ci]

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "kg_doc_parser_test_payloads.json"
EXPECTED_TREE_CASE_IDS = {
    f"P{number:03d}_{name}"
    for number, name in (
        (1, "title_atomic_table"),
        (2, "paragraph_atomic_table"),
        (3, "section_single_figure"),
        (4, "contentless_wrapper"),
        (5, "content_subtree_preservation"),
        (6, "root_wrapper_single_leaf"),
        (7, "llm_title_not_grounding"),
        (8, "70_level_unary_chain"),
        (9, "identical_content_distinct_offsets"),
        (10, "duplicate_ownership_reject"),
        (11, "aggregate_cannot_hide_loss"),
        (12, "pointer_mismatch_reject"),
        (13, "unicode_crlf_offsets"),
        (14, "case_distinct_source"),
        (15, "51_row_atomic_table"),
        (16, "valid_branching_control"),
        (17, "overlap_ambiguous_exclusive_parent"),
        (18, "invalid_parent_reference"),
    )
}
EXPECTED_WORKFLOW_CASE_IDS = {
    "W001_layer_boundary_cud_repairs",
    "W002_layer_boundary_budget_exhaustion",
    "W001_layer_excerpt_cud_repairs",
    "W002_layer_excerpt_budget_exhaustion",
    "W003_explicit_fallback_after_exhaustion",
    "W004_one_layer_page_index_no_recursive_bottom",
    "W005_triage_active",
    "W006_triage_provider_failure",
    "W007_page_index_call_cache_reuse",
    "W008_page_index_cache_invalidation",
    "W009_standalone_checkpoint_resume",
    "W010_resume_stale_document_reject",
}


class _FixtureCollection:
    def __init__(self, collection_id: str, title: str) -> None:
        self._collection_id = collection_id
        self._title = title

    @property
    def collection_id(self) -> str:
        return self._collection_id

    @property
    def title(self) -> str:
        return self._title


def _fixture_bundle() -> Mapping[str, object]:
    return cast(
        Mapping[str, object],
        json.loads(FIXTURE_PATH.read_text(encoding="utf-8")),
    )


def _object(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"fixture field {name!r} must be an object")
    return value


def _objects(value: object, *, name: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list):
        raise TypeError(f"fixture field {name!r} must be a list")
    return tuple(_object(item, name=name) for item in value)


def _case_by_id(cases: tuple[Mapping[str, object], ...], case_id: str) -> Mapping[str, object]:
    matches = [case for case in cases if case.get("id") == case_id]
    assert len(matches) == 1, f"fixture case {case_id!r} must occur exactly once"
    return matches[0]


def _rebase_fixture_tree(
    root: SemanticNode,
    *,
    unit_id: str,
    parent_id: str,
    level_from_root: int,
) -> list[SemanticNode]:
    """Adapt fixture IDs to the source unit and page created by the public API."""

    pending: list[tuple[SemanticNode, str | None, int, bool]] = [
        (child, parent_id, level_from_root, False)
        for child in reversed(root.child_nodes)
    ]
    known_ids = {node.node_id for node in _walk(root)}
    rebased: dict[int, SemanticNode] = {}
    while pending:
        current, expected_parent_id, level, exiting = pending.pop()
        marker = id(current)
        if exiting:
            pointers = [
                pointer.model_copy(update={"source_cluster_id": unit_id})
                for pointer in current.total_content_pointers
            ]
            aggregate_pointers = [
                pointer.model_copy(update={"source_cluster_id": unit_id})
                for pointer in current.aggregate_content_pointers
            ]
            parent = (
                expected_parent_id
                if current.parent_id in known_ids
                else current.parent_id
            )
            rebased[marker] = current.model_copy(
                update={
                    "parent_id": parent,
                    "level_from_root": level,
                    "total_content_pointers": pointers,
                    "aggregate_content_pointers": aggregate_pointers,
                    "child_nodes": [rebased[id(child)] for child in current.child_nodes],
                }
            )
            continue
        pending.append((current, expected_parent_id, level, True))
        for child in reversed(current.child_nodes):
            pending.append((child, current.node_id, level + 1, False))
    return [rebased[id(child)] for child in root.child_nodes]


def _walk(root: SemanticNode) -> list[SemanticNode]:
    nodes: list[SemanticNode] = []
    pending = [root]
    while pending:
        node = pending.pop()
        nodes.append(node)
        pending.extend(reversed(node.child_nodes))
    return nodes


@pytest.mark.parametrize("case_id", sorted(EXPECTED_TREE_CASE_IDS))
def test_fixture_tree_case_runs_through_public_page_index_path(
    monkeypatch: pytest.MonkeyPatch,
    case_id: str,
) -> None:
    bundle = _fixture_bundle()
    tree_cases = _objects(bundle.get("tree_cases"), name="tree_cases")
    case = _case_by_id(tree_cases, case_id)
    expected = _object(case.get("expected"), name=f"{case_id}.expected")
    outcome = str(expected.get("outcome"))
    source_map = _object(case.get("source_map"), name=f"{case_id}.source_map")
    source_text = str(_object(next(iter(source_map.values())), name=f"{case_id}.source").get("text"))
    input_tree = SemanticNode.model_validate(_object(case.get("input_tree"), name=f"{case_id}.input_tree"))

    def materialize_blocks(
        *, unit_id: str, parent_id: str, level_from_root: int, **_: object
    ) -> tuple[list[SemanticNode], int]:
        children = _rebase_fixture_tree(
            input_tree,
            unit_id=unit_id,
            parent_id=parent_id,
            level_from_root=level_from_root,
        )
        return children, len(source_text)

    monkeypatch.setattr(page_index_module, "_materialize_block_tree", materialize_blocks)
    if outcome == "explicit_failure":
        with pytest.raises(ValueError):
            parse_page_index_document(
                document_id=case_id,
                title=str(input_tree.title),
                raw_text=source_text,
                source_format="markdown",
                mode="heuristic",
                summary_enabled=False,
            )
        return

    try:
        result = parse_page_index_document(
            document_id=case_id,
            title=str(input_tree.title),
            raw_text=source_text,
            source_format="markdown",
            mode="heuristic",
            summary_enabled=False,
        )
    except ValueError:
        if outcome == "normalized_or_explicit_failure":
            return
        raise

    validate_semantic_tree_integrity(
        result.semantic_tree,
        parser_source_map=result.parser_source_map,
    )
    terminal_coverage = compute_terminal_content_coverage(
        result.semantic_tree,
        result.parser_source_map,
    )
    assert terminal_coverage["valid"] is True
    assert all(len(node.child_nodes) != 1 for node in _walk(result.semantic_tree))
    for node in _walk(result.semantic_tree):
        for pointer in node.total_content_pointers:
            source = result.authoritative_source_map[pointer.source_cluster_id].text
            end = len(source) if pointer.end_char < 0 else pointer.end_char + 1
            assert source[pointer.start_char:end] == pointer.verbatim_text


def test_fixture_payload_pack_has_all_mandatory_tree_and_workflow_cases() -> None:
    bundle = _fixture_bundle()
    assert bundle.get("schema") == "kg_doc_parser_regression_fixtures.v1"
    tree_cases = _objects(bundle.get("tree_cases"), name="tree_cases")
    workflow_cases = _objects(bundle.get("workflow_cases"), name="workflow_cases")
    assert {str(case["id"]) for case in tree_cases} == EXPECTED_TREE_CASE_IDS
    assert {str(case["id"]) for case in workflow_cases} == EXPECTED_WORKFLOW_CASE_IDS
    for case in (*tree_cases, *workflow_cases):
        assert isinstance(case.get("description"), str)
        assert isinstance(case.get("expected"), Mapping)


def _workflow_case(case_id: str) -> Mapping[str, object]:
    cases = _objects(_fixture_bundle().get("workflow_cases"), name="workflow_cases")
    return _case_by_id(cases, case_id)


def _workflow_source(case_id: str) -> tuple[str, Mapping[str, object]]:
    case = _workflow_case(case_id)
    source_map = _object(case.get("source_map"), name=f"{case_id}.source_map")
    source = _object(next(iter(source_map.values())), name=f"{case_id}.source")
    return str(source.get("text")), source_map


def _source_pointer(text: str, *, start: int = 0, end: int | None = None) -> HydratedTextPointer:
    final = len(text) - 1 if end is None else end
    return HydratedTextPointer(
        source_cluster_id="page_1",
        start_char=start,
        end_char=final,
        verbatim_text=text[start : final + 1],
    )


@pytest.mark.parametrize(
    "case_id",
    ["W001_layer_boundary_cud_repairs", "W001_layer_excerpt_cud_repairs"],
)
def test_workflow_cud_fixture_rejects_unary_then_commits_repair(case_id: str) -> None:
    source, _ = _workflow_source(case_id)
    parent = SemanticNode(
        node_id="parent",
        node_type="SECTION",
        title="Everything",
        total_content_pointers=[_source_pointer(source)],
    )
    initial = CurrentLayerResult(
        children=[
            LayerChildCandidate(
                node_id="everything",
                parent_node_id="parent",
                title="Everything",
                node_type="PARAGRAPH",
                total_content_pointers=[_source_pointer(source)],
            )
        ]
    )
    with pytest.raises(ValueError, match="unary refinement"):
        commit_layer_children(
            semantic_tree=parent,
            current_layer_result=initial,
            current_depth=0,
            parent_node_ids=["parent"],
            parser_source_map={"page_1": {"text": source}},
        )

    corrected = CurrentLayerResult(
        children=[
            LayerChildCandidate(
                node_id=node_id,
                parent_node_id="parent",
                title=title,
                node_type="HEADING_TEXT" if node_id == "title" else "PARAGRAPH",
                total_content_pointers=[_source_pointer(source, start=start, end=end)],
            )
            for node_id, title, start, end in (
                ("title", "Title", 0, 4),
                ("first", "First", 6, 10),
                ("second", "Second", 12, 17),
            )
        ]
    )
    committed = commit_layer_children(
        semantic_tree=parent,
        current_layer_result=corrected,
        current_depth=0,
        parent_node_ids=["parent"],
        parser_source_map={"page_1": {"text": source}},
    )
    assert [child.node_id for child in committed.child_nodes] == ["title", "first", "second"]
    validate_semantic_tree_integrity(
        committed,
        parser_source_map={"page_1": {"text": source}},
    )


def test_workflow_page_index_fixture_is_one_layer_and_retriages_children() -> None:
    source, _ = _workflow_source("W004_one_layer_page_index_no_recursive_bottom")
    candidates = parse_page_index_layer(
        parent_id="parent",
        parent_title="Everything",
        parent_pointers=[_source_pointer(source)],
        parser_source_map={"page_1": {"text": source}},
        source_format="markdown",
    )
    assert [candidate.title for candidate in candidates] == ["Title"]
    assert len(candidates[0].child_candidates) == 2
    assert all(not child.child_candidates for child in candidates[0].child_candidates)


@pytest.mark.parametrize(
    "case_id",
    ["W002_layer_boundary_budget_exhaustion", "W002_layer_excerpt_budget_exhaustion"],
)
def test_workflow_retry_budget_exhaustion_cannot_be_reused_as_success(case_id: str) -> None:
    case = _workflow_case(case_id)
    strategy = str(case["strategy"])
    assert _object(case["expected"], name=f"{case_id}.expected")["no_successful_commit"] is True
    split_strategy = "boundary_first" if strategy == "layer_boundary" else "excerpt_first"
    fallback_strategy = "excerpt_first" if split_strategy == "boundary_first" else "boundary_first"
    session = ParseSessionState(
        collection_id=case_id,
        root_node_id="root",
        split_strategy=split_strategy,
        fallback_split_strategy=fallback_strategy,
    )
    context = CurrentLayerContext(
        depth=0,
        parent_node_ids=["root"],
        split_strategy=session.split_strategy,
        retry_count=2,
        max_retries=2,
    )
    switched_session, switched_context = switch_split_strategy(
        parse_session=session,
        current_layer_context=context,
    )
    assert switched_session.strategy_switch_count == 1
    assert switched_context.retry_count == 0
    exhausted_context = switched_context.model_copy(
        update={"split_strategy": switched_session.fallback_split_strategy}
    )
    with pytest.raises(ValueError, match="fallback split strategy already exhausted"):
        switch_split_strategy(
            parse_session=switched_session,
            current_layer_context=exhausted_context,
        )


def test_workflow_explicit_fallback_is_distinct_from_failed_strategy() -> None:
    case = _workflow_case("W003_explicit_fallback_after_exhaustion")
    expected = _object(case["expected"], name="W003.expected")
    assert expected["fallback_must_not_be_logged_as_boundary_success"] is True
    strategy = select_parse_strategy(
        requested="page_index",
        context={"fixture": "W003", "reason": "retry_exhausted"},
        triage_enabled=False,
    )
    assert strategy.selected_strategy == "page_index"
    assert strategy.source == "config"
    assert strategy.rationale != "boundary strategy succeeded"


def test_workflow_checkpoint_fixture_rejects_a_stale_source_revision() -> None:
    case = _workflow_case("W010_resume_stale_document_reject")
    expected = _object(case["expected"], name="W010.expected")
    assert expected["reject_stale_checkpoint"] is True
    original_map: dict[str, dict[str, object]] = {"page_1": {"text": "Title\nFirst\nSecond\n"}}
    changed_map: dict[str, dict[str, object]] = {"page_1": {"text": "Title\nChanged\nSecond\n"}}
    assert source_map_fingerprint(original_map) != source_map_fingerprint(changed_map)
    session, _frontier, root = initialize_parse_session(
        collection=_FixtureCollection("W010", "Fixture"),
        parser_input_dict={},
        parser_source_map=original_map,
    )
    with pytest.raises(ValueError, match="does not match the parse session source revision"):
        propose_layer_breakdown(
            collection=_FixtureCollection("W010", "Fixture"),
            parser_input_dict={},
            parser_source_map=changed_map,
            parse_session=session,
            current_layer_context=CurrentLayerContext(depth=0),
            semantic_tree=root,
            propose_layer_fn=lambda **_kwargs: CurrentLayerResult(children=[]),
        )


def test_workflow_triage_fixtures_preserve_provider_and_fallback_decisions() -> None:
    assessments = [
        ParseStrategyAssessment(
            strategy=strategy,
            advantages=["bounded"],
            limitations=["provider dependent"],
        )
        for strategy in HARD_CODED_STRATEGY_PRIORITY
    ]
    active = _workflow_case("W005_triage_active")
    active_script = _objects(active.get("script"), name="W005.script")
    assert active_script[0]["triage_enabled"] is True
    active_decision = select_parse_strategy(
        requested="auto",
        context={"fixture": "W005"},
        triage_enabled=True,
        triage_fn=lambda _context: ParseStrategyTriage(
            selected_strategy="layer_boundary",
            confidence=0.95,
            rationale="fixture provider decision",
            assessments=assessments,
        ),
    )
    assert active_decision.source == "llm_triage"
    assert active_decision.selected_strategy == "layer_boundary"

    failed = _workflow_case("W006_triage_provider_failure")
    failed_script = _objects(failed.get("script"), name="W006.script")
    assert failed_script[0]["provider_error"] == "timeout"
    fallback_decision = select_parse_strategy(
        requested="auto",
        context={"fixture": "W006"},
        triage_enabled=True,
        triage_fn=lambda _context: (_ for _ in ()).throw(TimeoutError("fixture timeout")),
    )
    assert fallback_decision.source == "llm_triage_fallback"


def test_workflow_cache_fixtures_reuse_and_invalidate_effective_requests() -> None:
    cache_root = Path(".pytest-local-cache")
    cache_root.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=cache_root) as directory:
        cache = WorkflowLLMCallCache(directory, namespace={"fixture": "W007"})
        calls = 0

        def provider_call() -> dict[str, str]:
            nonlocal calls
            calls += 1
            return {"result": "validated"}

        request = {"source_sha": "a", "provider": "model-a"}
        assert cache.cached_call(operation="page_index", fingerprint=request, fn=provider_call) == {
            "result": "validated"
        }
        assert cache.cached_call(operation="page_index", fingerprint=request, fn=provider_call) == {
            "result": "validated"
        }
        assert calls == 1

        invalidated = _workflow_case("W008_page_index_cache_invalidation")
        assert _objects(invalidated.get("script"), name="W008.script")[0]["source_sha"] == "a"
        assert cache.cached_call(
            operation="page_index",
            fingerprint={"source_sha": "b", "provider": "model-a"},
            fn=provider_call,
        ) == {"result": "validated"}
        assert cache.cached_call(
            operation="page_index",
            fingerprint={"source_sha": "a", "provider": "model-b"},
            fn=provider_call,
        ) == {"result": "validated"}
        assert cache.cached_call(
            operation="page_index",
            fingerprint={"source_sha": "a", "provider": "model-a", "schema_version": "next"},
            fn=provider_call,
        ) == {"result": "validated"}
        isolated_cache = WorkflowLLMCallCache(directory, namespace={"fixture": "other-tenant"})
        assert isolated_cache.cached_call(
            operation="page_index",
            fingerprint=request,
            fn=provider_call,
        ) == {"result": "validated"}
        assert calls == 5
