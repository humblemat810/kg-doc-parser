from __future__ import annotations

from pathlib import Path

from PIL import Image

from kg_doc_parser.models import OCRClusterResponse, TextCluster
from kg_doc_parser.workflow_ingest import (
    GroupProposal,
    OCRImagePayload,
    WorkflowProviderSettings,
    apply_grouping_passes,
    apply_grouping_proposals,
    build_document_tree,
    merge_groups,
    normalize_ocr_pages,
    prepare_ocr_workflow_input,
)
from kg_doc_parser.workflow_ingest.models import SourceUnit, WorkflowIngestInput


def test_build_document_tree_preserves_two_pages_and_exact_text() -> None:
    inp = WorkflowIngestInput.from_text(document_id="doc", text="first\nsecond")
    tree = build_document_tree(inp)
    assert tree is not None
    assert tree.render_preview() == "first\nsecond"
    assert {item.location.page_index for item in tree.source_items} == {0}


def test_repeated_identical_ocr_text_keeps_distinct_source_identity() -> None:
    inp = WorkflowIngestInput(
        request_id="repeat-doc",
        collections=[
            {
                "collection_id": "repeat-doc",
                "title": "Repeated",
                "modality": "ocr",
                "pages": [
                    {
                        "page_number": 1,
                        "units": [
                            SourceUnit(modality="text", page_number=1, text="same"),
                            SourceUnit(modality="text", page_number=1, text="same"),
                        ],
                    }
                ],
            }
        ],
    )
    tree = build_document_tree(inp)
    assert tree is not None
    assert [item.text for item in tree.source_items] == ["same", "same"]
    assert len({item.item_id for item in tree.source_items}) == 2
    assert tree.render_preview() == "samesame"


def test_grouping_can_join_a_cross_page_continuation_without_rewriting_source() -> None:
    inp = WorkflowIngestInput(
        request_id="cross-page-doc",
        collections=[
            {
                "collection_id": "cross-page-doc",
                "title": "Cross page",
                "modality": "ocr",
                "pages": [
                    {"page_number": 1, "units": [SourceUnit(modality="text", page_number=1, text="head")]},
                    {
                        "page_number": 2,
                        "units": [
                            SourceUnit(modality="text", page_number=2, text="tail"),
                            SourceUnit(modality="text", page_number=2, text="after"),
                        ],
                    },
                ],
            }
        ],
    )
    tree = build_document_tree(inp)
    assert tree is not None
    first, second = tree.source_items[:2]
    grouped = apply_grouping_proposals(
        tree,
        (
            GroupProposal(
                kind="layout:cross-page-paragraph",
                members=(f"leaf:{first.item_id}", f"leaf:{second.item_id}"),
            ),
        ),
        grouping_policy_version="v1",
    )
    group = next(node for node in grouped.nodes if node.kind == "layout:cross-page-paragraph")
    assert {item.location.page_index for item in grouped.source_items if item.item_id in {first.item_id, second.item_id}} == {0, 1}
    assert {child.node_id for child in group.children} == {f"leaf:{first.item_id}", f"leaf:{second.item_id}"}


def test_normalize_ocr_pages_marks_blank_pages_without_fabricating_source_items() -> None:
    inp = normalize_ocr_pages(
        document_id="blank-doc",
        title="Blank",
        pages=[{"pdf_page_num": 1, "OCR_text_clusters": [], "non_text_objects": []}],
    )
    page = inp.collections[0].pages[0]
    assert page.units == []
    assert page.metadata["is_blank_page"] is True
    assert build_document_tree(inp) is None


def test_normalize_ocr_pages_preserves_declared_order_and_provider_metadata() -> None:
    inp = normalize_ocr_pages(
        document_id="ordered-doc",
        title="Ordered",
        pages=[
            {
                "pdf_page_num": 1,
                "meaningful_ordering": [2, 1],
                "metadata": {
                    "ocr_provider": "rapidocr",
                    "ocr_model_name": "rapidocr-v1",
                    "ocr_configuration_fingerprint": "config-v1",
                },
                "OCR_text_clusters": [
                    {"cluster_number": 1, "text": "second", "bb_x_max": 10, "bb_y_max": 10},
                    {"cluster_number": 2, "text": "first", "bb_x_max": 10, "bb_y_max": 10},
                ],
                "non_text_objects": [],
            }
        ],
    )
    tree = build_document_tree(inp)
    assert tree is not None
    assert [item.text for item in tree.source_items] == ["first", "second"]
    assert all(item.ocr_metadata is not None for item in tree.source_items)
    assert tree.source_items[0].ocr_metadata.model == "rapidocr-v1"


def test_unknown_ocr_coordinate_system_downgrades_to_page_accuracy() -> None:
    inp = normalize_ocr_pages(
        document_id="unknown-coordinates",
        title="Unknown coordinates",
        pages=[
            {
                "pdf_page_num": 1,
                "OCR_text_clusters": [
                    {
                        "cluster_number": 0,
                        "text": "opaque frame",
                        "bb_x_max": 10,
                        "bb_y_max": 10,
                        "coordinate_system": "provider-native-v2",
                    }
                ],
                "non_text_objects": [],
            }
        ],
    )
    tree = build_document_tree(inp)
    assert tree is not None
    assert tree.source_items[0].location.accuracy == "page"
    assert tree.source_items[0].location.bbox is None


def test_build_document_tree_keeps_nontext_observation_without_fake_media() -> None:
    inp = WorkflowIngestInput(
        request_id="doc",
        collections=[
            {
                "collection_id": "doc",
                "title": "doc",
                "modality": "ocr",
                "pages": [
                    {
                        "page_number": 1,
                        "units": [SourceUnit(modality="image_region", page_number=1, description="diagram")],
                    }
                ],
            }
        ],
    )
    tree = build_document_tree(inp)
    assert tree is not None
    assert tree.source_items[0].kind == "non_text:description"
    assert tree.source_items[0].media is None


def test_build_document_tree_uses_only_declared_content_addressed_media() -> None:
    digest = "a" * 64
    inp = WorkflowIngestInput(
        request_id="doc",
        collections=[
            {
                "collection_id": "doc",
                "title": "doc",
                "modality": "ocr",
                "pages": [
                    {
                        "page_number": 1,
                        "units": [
                            SourceUnit(
                                modality="pure_image",
                                page_number=1,
                                source_uri="assets/figure.png",
                                metadata={
                                    "mime_type": "image/png",
                                    "media_sha256": digest,
                                    "media_byte_length": 12,
                                },
                            )
                        ],
                    }
                ],
            }
        ],
    )
    tree = build_document_tree(inp)
    assert tree is not None
    assert tree.source_items[0].media is not None
    assert tree.source_items[0].text is None


def test_apply_grouping_proposals_keeps_source_coverage_and_rejects_bad_members() -> None:
    inp = WorkflowIngestInput(
        request_id="doc",
        collections=[
            {
                "collection_id": "doc",
                "title": "doc",
                "modality": "ocr",
                "pages": [
                    {"page_number": 1, "units": [
                        SourceUnit(modality="text", page_number=1, cluster_number=0, text="a"),
                        SourceUnit(modality="text", page_number=1, cluster_number=1, text="b"),
                    ]},
                ],
            }
        ],
    )
    tree = build_document_tree(inp)
    assert tree is not None
    grouped = apply_grouping_proposals(
        tree,
        (GroupProposal(kind="layout:pair", members=("leaf:doc|p1_t0", "leaf:doc|p1_t1")),),
        grouping_policy_version="v1",
    )
    assert grouped.render_preview() == "ab"
    assert grouped.tree_revision == 2


def test_grouping_passes_use_the_new_frontier_for_the_next_pass() -> None:
    inp = WorkflowIngestInput.from_text(document_id="doc", text="a")
    # Build a synthetic three-item baseline to exercise two adaptive passes.
    inp.collections[0].pages[0].units = [
        SourceUnit(modality="text", page_number=1, cluster_number=0, text="a"),
        SourceUnit(modality="text", page_number=1, cluster_number=1, text="b"),
        SourceUnit(modality="text", page_number=1, cluster_number=2, text="c"),
    ]
    tree = build_document_tree(inp)
    assert tree is not None
    first_frontier, _ = merge_groups(
        document_id="doc",
        frontier=("leaf:doc|p1_t0", "leaf:doc|p1_t1", "leaf:doc|p1_t2"),
        proposals=(GroupProposal(kind="layout:pair", members=("leaf:doc|p1_t0", "leaf:doc|p1_t1")),),
        grouping_policy_version="v1:1",
    )
    grouped = apply_grouping_passes(
        tree,
        (
            (GroupProposal(kind="layout:pair", members=("leaf:doc|p1_t0", "leaf:doc|p1_t1")),),
            (GroupProposal(kind="layout:section", members=(first_frontier[0], "leaf:doc|p1_t2")),),
        ),
        grouping_policy_version="v1",
    )
    assert grouped.tree_revision == 3


def test_prepare_ocr_workflow_emits_the_tree_at_the_existing_pipeline_boundary(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (40, 40), "white").save(image_path)

    def fake_runner(image: Path, page_number: int, settings: object) -> OCRClusterResponse:
        del image, page_number, settings
        return OCRClusterResponse(
            OCR_text_clusters=[TextCluster(text="OCR text", bb_x_min=1, bb_x_max=20, bb_y_min=1, bb_y_max=10, cluster_number=0)],
            non_text_objects=[],
            is_empty_page=False,
            printed_page_number=None,
            meaningful_ordering=[0],
            page_x_min=0,
            page_x_max=40,
            page_y_min=0,
            page_y_max=40,
            estimated_rotation_degrees=0,
            incomplete_words_on_edge=False,
            incomplete_text=False,
            data_loss_likelihood=0,
            scan_quality="high",
            contains_table=False,
        )

    artifacts = prepare_ocr_workflow_input(
        document_id="doc",
        title="Doc",
        output_dir=tmp_path / "out",
        image_payloads=(OCRImagePayload(page_number=1, image_path=str(image_path)),),
        provider_settings=WorkflowProviderSettings(),
        ocr_runner=fake_runner,
    )
    assert artifacts.document_tree is not None
    assert artifacts.document_tree_path.exists()
    assert artifacts.document_tree.source_items[0].item_id == "doc|p1_t0"
    assert artifacts.workflow_input.collections[0].metadata["document_tree"]
