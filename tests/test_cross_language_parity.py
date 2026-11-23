from __future__ import annotations

import json
from pathlib import Path

from kg_doc_parser.semantic_indexing_contracts import (
    batch_nodes,
    default_token_estimate,
    json_compact,
    render_nodes_payload,
)
from kg_doc_parser.semantic_source_text import source_map_entry_text
from kg_doc_parser.workflow_ingest.page_index_source_text import _split_pages
from kg_doc_parser.workflow_ingest.page_index_classification import _classify_block
from kg_doc_parser.workflow_ingest.page_index import _heuristic_page_outline


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "rust"
    / "crates"
    / "kg-doc-parser"
    / "tests"
    / "fixtures"
    / "semantic_contracts.json"
)


def test_source_map_entry_text_matches_rust() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for case in payload["source_map_entry_text"]:
        assert source_map_entry_text(case["entry"]) == case["expected"]


def test_page_index_split_pages_matches_rust() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for case in payload["page_index_split_pages"]:
        assert _split_pages(case["raw_text"]) == case["expected"]


def test_page_index_classify_blocks_matches_rust() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for case in payload["page_index_classify_blocks"]:
        options = case["options"]
        actual = _classify_block(
            case["text"],
            case["start_char"],
            case["end_char"],
            source_format=options["source_format"],
            is_first=options.get("is_first", False),
        )
        expected = case["expected"]
        assert actual.node_type == expected["node_type"]
        assert actual.title == expected["title"]
        assert actual.kind_hint == expected["kind_hint"]
        assert actual.heading_level == expected.get("heading_level")


def test_page_index_heuristic_outlines_match_rust() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for case in payload["page_index_outlines"]:
        specs, diagnostics = _heuristic_page_outline(
            case["page_text"],
            page_number=case["page_number"],
            source_format=case["source_format"],
        )
        assert [spec.model_dump(mode="json") for spec in specs] == case["expected_block_specs"]
        assert diagnostics == case["expected_diagnostics"]


def test_semantic_indexing_contracts_match_rust() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for case in payload["default_token_estimate"]:
        assert default_token_estimate(case["text"]) == case["expected"]
    for case in payload["render_nodes_payload"]:
        assert render_nodes_payload(case["nodes"]) == case["expected"]
    for case in payload["batch_nodes"]:
        assert (
            list(
                batch_nodes(
                    case["items"],
                    max_nodes=case["max_nodes"],
                    max_input_tokens=None,
                    base_prompt_tokens=case["base_prompt_tokens"],
                    per_node_overhead_tokens=case["per_node_overhead_tokens"],
                    render_item_for_estimate=json_compact,
                )
            )
            == case["expected"]
        )
