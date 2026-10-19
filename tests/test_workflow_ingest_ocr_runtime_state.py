from __future__ import annotations

import pytest

from kg_doc_parser.workflow_ingest import ocr_runtime as ocr_runtime_module


def test_ocr_runtime_state_boundary_rejects_wrong_known_field_types() -> None:
    with pytest.raises(TypeError, match="page_cursor.*integer"):
        ocr_runtime_module._typed_state({"page_cursor": "1"})

    with pytest.raises(TypeError, match="completed_pages.*list of integers"):
        ocr_runtime_module._typed_state({"completed_pages": [1, "2"]})

    with pytest.raises(TypeError, match="page_sources contains a malformed entry"):
        ocr_runtime_module._typed_state(
            {"page_sources": [{"page_number": 1, "image_path": 42}]}
        )

    with pytest.raises(TypeError, match="page_sources contains a malformed entry"):
        ocr_runtime_module._page_sources({"page_sources": [{"page_number": 1}]})

    with pytest.raises(TypeError, match="grouping_passes.*pass 0 must be a list"):
        ocr_runtime_module._typed_state({"grouping_passes": [{"proposal": 1}]})

    with pytest.raises(TypeError, match=r"grouping_passes\[0\].*JSON objects"):
        ocr_runtime_module._typed_state({"grouping_passes": [["not-a-proposal"]]})

    with pytest.raises(TypeError, match=r"tree.*opaque.*JSON serializable"):
        ocr_runtime_module._typed_state({"tree": {"opaque": object()}})

    with pytest.raises(TypeError, match=r"future_state.*JSON serializable"):
        ocr_runtime_module._typed_state({"future_state": {"opaque": object()}})

    with pytest.raises(TypeError, match="last_reused_page.*integer or null"):
        ocr_runtime_module._typed_state({"last_reused_page": "1"})

    with pytest.raises(TypeError, match="raw_pages.*must be a list or null"):
        ocr_runtime_module._typed_state({"raw_pages": {"page": 1}})

    with pytest.raises(TypeError, match=r"raw_pages\[0\].*JSON object"):
        ocr_runtime_module._typed_state({"raw_pages": ["not-an-object"]})


def test_ocr_runtime_state_boundary_rebuilds_validated_known_containers() -> None:
    raw_pages = [[{"proposal_id": "p1"}]]
    raw_sources = [{"page_number": 1, "image_path": "page-1.png"}]
    raw_ocr_pages = [{"pdf_page_num": 1, "text_clusters": []}]
    state = ocr_runtime_module._typed_state(
        {
            "completed_pages": [1],
            "reused_pages": [1],
            "grouping_passes": raw_pages,
            "page_sources": raw_sources,
            "raw_pages": raw_ocr_pages,
        }
    )

    completed_pages = state.get("completed_pages")
    reused_pages = state.get("reused_pages")
    grouping_passes = state.get("grouping_passes")
    page_sources = state.get("page_sources")
    stored_ocr_pages = state.get("raw_pages")
    assert completed_pages == [1]
    assert reused_pages == [1]
    assert grouping_passes == [[{"proposal_id": "p1"}]]
    assert page_sources == [{"page_number": 1, "image_path": "page-1.png"}]
    assert grouping_passes is not raw_pages
    assert page_sources is not raw_sources
    assert stored_ocr_pages == raw_ocr_pages
    assert stored_ocr_pages is not raw_ocr_pages


def test_ocr_runtime_child_state_uses_the_same_typed_boundary() -> None:
    with pytest.raises(TypeError, match="grouping child returned invalid typed state"):
        ocr_runtime_module._typed_child_state(
            {"page_cursor": "not-an-integer"},
            label="grouping child",
        )
