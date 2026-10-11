from __future__ import annotations

import pytest

from kg_doc_parser.models import SplitPage, SplitPageMeta, TextCluster


@pytest.mark.ci
def test_split_page_table_export_keeps_unordered_text_clusters() -> None:
    page = SplitPage.model_construct(
        pdf_page_num=1,
        metadata=SplitPageMeta(
            ocr_model_name="fake",
            ocr_datetime=0.0,
            ocr_json_version="1",
        ),
        OCR_text_clusters=[
            TextCluster(
                text="first",
                bb_x_min=0.0,
                bb_x_max=1.0,
                bb_y_min=0.0,
                bb_y_max=1.0,
                cluster_number=0,
            ),
            TextCluster(
                text="second",
                bb_x_min=0.0,
                bb_x_max=1.0,
                bb_y_min=1.0,
                bb_y_max=2.0,
                cluster_number=1,
            ),
        ],
        non_text_objects=[],
        is_empty_page=False,
        printed_page_number=None,
        meaningful_ordering=[0],
        page_x_min=0.0,
        page_x_max=2.0,
        page_y_min=0.0,
        page_y_max=2.0,
        estimated_rotation_degrees=0.0,
        incomplete_words_on_edge=False,
        incomplete_text=False,
        data_loss_likelihood=0.0,
        scan_quality="high",
        contains_table=True,
    )

    exported = page.to_doc()

    assert [cluster["text"] for cluster in exported["OCR_text_clusters"]] == ["first", "second"]
