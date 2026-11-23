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
from kg_doc_parser.semantic_pointer_matching import (
    all_exact_occurrences,
    best_occurrence_by_proximity,
    safe_slice,
    soft_exact_positions,
    whitespace_collapse,
)
from kg_doc_parser.workflow_ingest.page_index_source_text import _split_pages
from kg_doc_parser.workflow_ingest.page_index_classification import _classify_block
from kg_doc_parser.workflow_ingest.page_index import _heuristic_page_outline
from kg_doc_parser.workflow_ingest.page_index_tree import page_index_normalize_text
from kg_doc_parser.workflow_ingest.page_index_candidates import _extract_candidate_blocks
from kg_doc_parser.workflow_ingest.parser_core_session import source_map_fingerprint
from kg_doc_parser.workflow_ingest.ocr_artifacts import artifact_schema_error
from kg_doc_parser.workflow_ingest.ocr_providers import list_ocr_provider_capabilities
from kg_doc_parser.workflow_ingest.document_models import (
    BoundingBox,
    SourceLocation,
)


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


def test_pointer_matching_contracts_match_rust() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))["pointer_matching"]
    for case in payload["safe_slice"]:
        assert safe_slice(case["text"], case["start"], case["end_inclusive"]) == case["expected"]
    for case in payload["all_exact_occurrences"]:
        assert [
            list(span)
            for span in all_exact_occurrences(case["haystack"], case["needle"])
        ] == case["expected"]
    for case in payload["best_occurrence_by_proximity"]:
        assert (
            list(
                best_occurrence_by_proximity(
                    [tuple(span) for span in case["occurrences"]],
                    case["proposed_start"],
                )
                or ()
            )
            == case["expected"]
        )
    for case in payload["whitespace_collapse"]:
        assert whitespace_collapse(case["text"]) == case["expected"]
    for case in payload["soft_exact_positions"]:
        spans, diagnostic = soft_exact_positions(
            case["source_text"],
            case["verbatim"],
            fuzzy_threshold=None,
        )
        assert [list(span) for span in spans] == case["expected"]
        assert diagnostic == case["diagnostic"]


def test_ocr_artifact_schema_errors_match_rust() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for case in payload["ocr_artifact_schema"]:
        assert artifact_schema_error(case) == case["expected"]


def test_page_index_normalize_text_matches_rust() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for case in payload["page_index_normalize_text"]:
        assert page_index_normalize_text(case["text"]) == case["expected"]


def test_source_map_fingerprint_matches_rust() -> None:
    source_map = {
        "z": {"text": "second"},
        "a": {"text": "first"},
    }
    fingerprint = source_map_fingerprint(source_map)
    assert len(fingerprint) == 64
    assert fingerprint == source_map_fingerprint(
        {
            "a": {"text": "first"},
            "z": {"text": "second"},
        }
    )


def test_ocr_provider_inventory_shape_matches_rust() -> None:
    capabilities = list_ocr_provider_capabilities()
    assert len(capabilities) == 19
    assert capabilities[0].provider == "docling"
    assert capabilities[6].provider == "rapidocr"
    assert capabilities[8].provider == "bonsai"


def test_document_model_validation_matches_rust() -> None:
    bounding_box = BoundingBox(
        x0=0.0,
        y0=0.0,
        x1=1.0,
        y1=1.0,
        coordinate_system="relative",
    )
    assert bounding_box.x0 < bounding_box.x1
    assert bounding_box.y0 < bounding_box.y1
    assert all(0.0 <= value <= 1.0 for value in (bounding_box.x0, bounding_box.x1))
    source_location = SourceLocation(
        page_index=1,
        printed_page_label="1",
        bbox=bounding_box,
        accuracy="region",
    )
    assert source_location.page_index == 1
    assert source_location.bbox is not None
    assert source_location.accuracy == "region"


def test_page_index_candidate_extraction_matches_rust() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for case in payload["page_index_candidates"]:
        candidates = _extract_candidate_blocks(
            case["page_text"],
            page_number=case["page_number"],
            source_format=case["source_format"],
        )
        assert [candidate.text for candidate in candidates] == case["expected_texts"]
        assert [candidate.kind_hint for candidate in candidates] == case["expected_kinds"]


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
