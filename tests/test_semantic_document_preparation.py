from __future__ import annotations

import pytest

from kg_doc_parser.semantic_document_splitting_layerwise_edits import (
    prepare_document_for_llm,
)


def test_prepare_document_for_llm_validates_and_indexes_source_clusters() -> None:
    prepared, source_map = prepare_document_for_llm(
        {
            "source.pdf": [
                {
                    "pdf_page_num": 3,
                    "OCR_text_clusters": [{"text": "Page text", "bbox": [1, 2, 3, 4]}],
                    "non_text_objects": [],
                }
            ]
        }
    )

    assert prepared["document_filename"] == "source.pdf"
    assert prepared["pages"][0]["OCR_text_clusters"][0]["id"] == "p3_c0"
    assert source_map["p3_c0"]["text"] == "Page text"


@pytest.mark.parametrize(
    "raw_document",
    [
        {},
        {"source.pdf": {"pdf_page_num": 1}},
        {"source.pdf": [{"pdf_page_num": 1, "OCR_text_clusters": "not-a-list"}]},
        {
            "source.pdf": [
                {
                    "pdf_page_num": 1,
                    "OCR_text_clusters": ["not-a-cluster-object"],
                    "non_text_objects": [],
                }
            ]
        },
    ],
)
def test_prepare_document_for_llm_rejects_malformed_json_shapes(
    raw_document: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        prepare_document_for_llm(raw_document)
