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
