from __future__ import annotations

from kg_doc_parser.workflow_ingest.layered_contracts import (
    LayeredParseExpandRequest,
    LayeredParseLimits,
    LayeredParseSeedRequest,
    expand_layered_frontier,
    initialize_layered_parse,
)
from kg_doc_parser.workflow_ingest.models import CurrentLayerResult


def _seed_request() -> LayeredParseSeedRequest:
    return LayeredParseSeedRequest(
        collection={"collection_id": "doc-1", "title": "Demo"},
        parser_input={"title": "Demo"},
        source_map={
            "cluster-1": {
                "text": "Alpha paragraph. Beta paragraph.",
                "participates_in_semantic_text": True,
            }
        },
        limits=LayeredParseLimits(max_depth=3, max_frontier_items=1),
    )


def test_seed_and_expand_are_bounded_and_json_serializable() -> None:
    seeded = initialize_layered_parse(_seed_request())
    assert seeded.diagnostics["phase"] == "parse_seeded"
    assert len(seeded.frontier) == 1
    payload = seeded.model_dump(mode="json")
    assert payload["session"]["max_depth"] == 3

    expanded = expand_layered_frontier(
        LayeredParseExpandRequest(
            session=seeded.session,
            frontier=seeded.frontier,
            semantic_tree=seeded.root.model_dump(mode="json"),
            collection=_seed_request().collection,
            parser_input=_seed_request().parser_input,
            source_map=_seed_request().source_map,
            limits=LayeredParseLimits(max_depth=3, max_frontier_items=1),
        ),
        propose_layer_fn=lambda **_kwargs: CurrentLayerResult(children=[], satisfied=True),
    )
    assert expanded.usage.parser_calls == 1
    assert expanded.consumed_frontier
    assert expanded.model_dump(mode="json")["diagnostics"]["phase"] in {
        "parse_expanding",
        "parsed_graph_persisted",
    }


def test_empty_frontier_is_stable_without_parser_call() -> None:
    seeded = initialize_layered_parse(_seed_request())
    result = expand_layered_frontier(
        LayeredParseExpandRequest(
            session=seeded.session,
            frontier=[],
            semantic_tree=seeded.root.model_dump(mode="json"),
            collection=_seed_request().collection,
            parser_input=_seed_request().parser_input,
            source_map=_seed_request().source_map,
        )
    )
    assert result.stable is True
    assert result.usage.parser_calls == 0


def test_expansion_preserves_same_depth_items_outside_batch() -> None:
    seeded = initialize_layered_parse(_seed_request())
    first = seeded.frontier[0]
    second = first.model_copy(update={"parent_node_id": "doc-1|root-2", "order": 1})
    result = expand_layered_frontier(
        LayeredParseExpandRequest(
            session=seeded.session,
            frontier=[first, second],
            semantic_tree=seeded.root.model_dump(mode="json"),
            collection=_seed_request().collection,
            parser_input=_seed_request().parser_input,
            source_map=_seed_request().source_map,
            limits=LayeredParseLimits(max_depth=3, max_frontier_items=1),
        ),
        propose_layer_fn=lambda **_kwargs: CurrentLayerResult(children=[], satisfied=True),
    )
    assert len(result.consumed_frontier) == 1
    assert result.remaining_frontier == [second]
    assert result.stable is False
