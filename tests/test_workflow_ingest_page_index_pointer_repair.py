"""Tests for exact and fuzzy source-pointer repair in PageIndex trees."""

from __future__ import annotations

import pytest

import kg_doc_parser.workflow_ingest.page_index as page_index_module
import kg_doc_parser.workflow_ingest.page_index_tree as page_index_tree_module

pytestmark = [pytest.mark.workflow]


def test_page_index_resolve_pointer_keeps_exact_match_unchanged() -> None:
    page_text = "Alpha beta gamma."

    resolved = page_index_tree_module.resolve_pointer(
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

    resolved = page_index_tree_module.resolve_pointer(
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

    resolved = page_index_tree_module.resolve_pointer(
        unit_id="p0001",
        page_text=page_text,
        excerpt="Alpha beta gamma deltx epsilon",
        start_at=50,
    )

    assert resolved.verbatim_text == "Alpha beta gamma delta epsilon"
    assert resolved.start_char == page_text.index("Alpha beta gamma delta epsilon", 20)


def test_page_index_resolve_pointer_rejects_low_similarity_typo() -> None:
    with pytest.raises(ValueError, match="unable to resolve excerpt"):
        page_index_tree_module.resolve_pointer(
            unit_id="p0001",
            page_text="Alpha beta gamma.",
            excerpt="zzzzzz",
            start_at=0,
        )


def test_page_index_resolve_pointer_rejects_whole_page_fuzzy_repair() -> None:
    with pytest.raises(ValueError, match="unable to resolve excerpt"):
        page_index_tree_module.resolve_pointer(
            unit_id="p0001",
            page_text="Alpha beta",
            excerpt="Alpah beta",
            start_at=0,
        )


def test_page_index_materialize_records_fuzzy_repair_diagnostics() -> None:
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
        page_text="Alpha beta gamma delta epsilon.",
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
