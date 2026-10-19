from kg_doc_parser.models import NonText_box_2d
from kg_doc_parser.ocr import (
    OCRMetaResponse,
    RawOCRResponse,
    RawOCRResponse_to_OCRClusterResponse,
    TextBox,
    TextBoxResponse,
    TextBoxResponsePlusMetaResponse_to_OCRClusterResponse,
    NonTextObject,
    box_2d,
)


def test_legacy_ocr_conversion_preserves_source_item_accounting() -> None:
    response = RawOCRResponse(
        boxes_2d=[box_2d(box_2d=[0, 0, 20, 40], label="heading", id=0)],
        non_text_objects=[
            NonText_box_2d(
                description="figure",
                box_2d=[20, 0, 40, 40],
                id=1,
            )
        ],
        is_empty_page=False,
        printed_page_number=None,
        meaningful_ordering=[0, 1],
        page_x_min=0.0,
        page_x_max=40.0,
        page_y_min=0.0,
        page_y_max=40.0,
        estimated_rotation_degrees=0.0,
        incomplete_words_on_edge=False,
        incomplete_text=False,
        data_loss_likelihood=0.0,
        scan_quality="high",
        contains_table=False,
    )

    converted = RawOCRResponse_to_OCRClusterResponse(response)

    assert converted.source_item_count == 2
    assert converted.unresolved_item_count == 0
    assert converted.non_text_objects[0].description == "figure"


def test_legacy_text_box_retry_conversion_is_source_accounted() -> None:
    response = RawOCRResponse_to_OCRClusterResponse(
        TextBoxResponse(
            text_blocks=[
                TextBox(text="heading", bounding_box=[0, 0, 20, 40], id=0)
            ],
            non_text_blocks=[
                NonTextObject(text="figure", bounding_box=[20, 0, 40, 40], id=1)
            ],
            meaningful_ordering=[0, 1],
            printed_page_number="1",
        )
    )

    assert response.source_item_count == 2
    assert response.unresolved_item_count == 0
    assert response.non_text_objects[0].description == "figure"


def test_legacy_text_box_meta_conversion_preserves_source_item_accounting() -> None:
    response = TextBoxResponsePlusMetaResponse_to_OCRClusterResponse(
        TextBoxResponse(
            text_blocks=[
                TextBox(text="heading", bounding_box=[0, 0, 20, 40], id=0)
            ],
            non_text_blocks=[
                NonTextObject(text="figure", bounding_box=[20, 0, 40, 40], id=1)
            ],
            meaningful_ordering=[0, 1],
            printed_page_number="1",
        ),
        OCRMetaResponse(
            printed_page_number="1",
            page_x_min=0.0,
            page_x_max=40.0,
            page_y_min=0.0,
            page_y_max=40.0,
            estimated_rotation_degrees=0.0,
            incomplete_words_on_edge=False,
            incomplete_text=False,
            data_loss_likelihood=0.0,
            scan_quality="high",
            contains_table=False,
        ),
    )

    assert response.source_item_count == 2
    assert response.unresolved_item_count == 0
    assert response.non_text_objects[0].description == "figure"
