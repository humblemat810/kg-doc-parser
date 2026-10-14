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
    prepare_ocr_workflow_input,
)
from kg_doc_parser.workflow_ingest.models import SourceUnit, WorkflowIngestInput


def test_build_document_tree_preserves_two_pages_and_exact_text() -> None:
    inp = WorkflowIngestInput.from_text(document_id="doc", text="first\nsecond")
    tree = build_document_tree(inp)
    assert tree is not None
    assert tree.render_preview() == "first\nsecond"
    assert {item.location.page_index for item in tree.source_items} == {0}


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
