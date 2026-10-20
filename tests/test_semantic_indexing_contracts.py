"""Legacy-compatible semantic indexing contracts and batch invariants."""

from __future__ import annotations

from uuid import UUID

import pytest

from kg_doc_parser import semantic_document_splitting_layerwise_edits as legacy
from kg_doc_parser.semantic_indexing_contracts import (
    BatchIndexResponse,
    IdMapping,
    IndexingResponse,
    IndexMode,
    available_node_ids,
    batch_nodes,
)
from kg_doc_parser.semantic_pointer_matching import (
    all_exact_occurrences,
    safe_slice,
    soft_exact_positions,
)
from kg_doc_parser.workflow_ingest.semantics import HydratedTextPointer


@pytest.mark.ci
def test_legacy_module_reexports_indexing_contracts() -> None:
    assert legacy.IndexingResponse is IndexingResponse
    assert legacy.BatchIndexResponse is BatchIndexResponse
    assert legacy.IdMapping is IdMapping
    assert legacy.IndexMode is IndexMode
    assert legacy.batch_nodes is batch_nodes
    assert legacy._safe_slice is safe_slice
    assert legacy._all_exact_occurrences is all_exact_occurrences
    assert legacy._soft_exact_positions is soft_exact_positions


@pytest.mark.ci
def test_indexing_response_consumes_exactly_one_expected_node_id() -> None:
    token = available_node_ids.set({"node-1"})
    try:
        result = IndexingResponse(
            node_id="node-1",
            canonical_title="Title",
            keywords=["key"],
            aliases=[],
            provision="Clause 1",
        )
        assert result.node_id == "node-1"
        assert available_node_ids.get() == set()
    finally:
        available_node_ids.reset(token)

    token = available_node_ids.set(set())
    try:
        with pytest.raises(Exception, match="duplicated or not exist"):
            IndexingResponse(
                node_id="unexpected",
                canonical_title="Title",
                keywords=[],
                aliases=[],
                provision="",
            )
    finally:
        available_node_ids.reset(token)


@pytest.mark.ci
def test_indexing_response_requires_an_active_batch_context() -> None:
    assert available_node_ids.get() is None
    with pytest.raises(Exception, match="active node-id context"):
        IndexingResponse(
            node_id="node-outside-batch",
            canonical_title="Title",
            keywords=[],
            aliases=[],
            provision="",
        )
    assert available_node_ids.get() is None


@pytest.mark.ci
def test_batch_nodes_preserves_order_and_node_limit() -> None:
    items = [{"node_id": f"node-{index}"} for index in range(5)]

    batches = list(batch_nodes(items, max_nodes=2))

    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert [item for batch in batches for item in batch] == items


@pytest.mark.ci
def test_batch_nodes_allows_one_oversized_item_without_losing_order() -> None:
    items = [{"node_id": "large"}, {"node_id": "next"}]

    batches = list(
        batch_nodes(
            items,
            max_input_tokens=1,
            base_prompt_tokens=0,
            per_node_overhead_tokens=0,
            token_estimator=lambda _text: 5,
        )
    )

    assert batches == [[items[0]], [items[1]]]


@pytest.mark.ci
def test_semantic_pointer_matching_preserves_inclusive_offsets() -> None:
    assert safe_slice("alpha beta", 6, 9) == "beta"
    assert safe_slice("alpha beta", 0, -1) == "alpha beta"
    assert all_exact_occurrences("alpha alpha", "alpha") == [(0, 4), (6, 10)]


@pytest.mark.ci
def test_semantic_pointer_matching_reports_whitespace_normalization() -> None:
    matches, diagnostic = soft_exact_positions("alpha\n\tbeta", "alpha beta")

    assert matches == [(0, 10)]
    assert diagnostic == {"name": "exact", "collapsed": True}


@pytest.mark.ci
def test_semantic_pointer_matching_keeps_fuzzy_repair_opt_in() -> None:
    assert soft_exact_positions("alpha beta", "alpha zeta") == ([], None)


@pytest.mark.ci
def test_semantic_index_payload_keeps_short_ids_parentage_and_exact_leaf_text() -> None:
    root_id = UUID("00000000-0000-0000-0000-000000000001")
    child_id = UUID("00000000-0000-0000-0000-000000000002")
    root = legacy.SemanticNode(
        node_id=root_id,
        node_type="DOCUMENT_ROOT",
        title="Document",
        total_content_pointers=[],
        child_nodes=[],
        level_from_root=0,
    )
    child = legacy.SemanticNode(
        node_id=child_id,
        parent_id=root_id,
        node_type="TEXT_FLOW",
        title="Section",
        total_content_pointers=[
            HydratedTextPointer(
                source_cluster_id="source-1",
                start_char=0,
                end_char=10,
                verbatim_text="exact text",
            )
        ],
        child_nodes=[],
        level_from_root=1,
    )

    prepared = legacy._prepare_nodes([root, child])
    root_short_id = prepared.id_map.forward_map[str(root_id)]
    child_short_id = prepared.id_map.forward_map[str(child_id)]

    assert prepared.node_by_id[child_short_id]["parent_id"] == root_short_id
    assert legacy._leaf_payload(prepared.node_by_id[child_short_id]) == {
        "node_id": child_short_id,
        "title": "Section",
        "kind": "LEAF",
        "contents": ["exact text"],
    }


@pytest.mark.ci
def test_legacy_source_text_reads_only_real_text_fields() -> None:
    assert (
        legacy._source_map_entry_text({"text": "primary", "content": "fallback"})
        == "primary"
    )
    assert legacy._source_map_entry_text({"text": " ", "content": "fallback"}) == "fallback"
    assert legacy._source_map_entry_text(7) == ""


@pytest.mark.ci
def test_legacy_source_text_rejects_non_string_nested_evidence() -> None:
    source = {
        "OCR_text_clusters": [
            {"text": 17, "content": "actual cluster"},
            {"text": "second cluster"},
            "not a cluster",
        ],
        "non_text_objects": [{"description": False}, {"text": "Figure caption"}],
    }

    assert legacy._source_map_entry_text(source) == "actual cluster\nsecond cluster"
    assert legacy._source_map_entry_text(
        {"non_text_objects": [{"description": False}, {"text": "Figure caption"}]}
    ) == "Figure caption"
