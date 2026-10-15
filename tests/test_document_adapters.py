from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from kg_doc_parser.models import OCRClusterResponse, TextCluster
from kg_doc_parser.workflow_ingest import (
    GroupProposal,
    OCRImagePayload,
    UnrepresentedSourceDiagnostic,
    UnrepresentedSourceUnitError,
    WorkflowProviderSettings,
    apply_grouping_passes,
    apply_grouping_proposals,
    build_document_tree,
    merge_groups,
    normalize_ocr_pages,
    prepare_ocr_workflow_input,
    suggest_cross_page_continuations,
)
from kg_doc_parser.workflow_ingest.document_models import Summary
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


def test_cross_page_continuation_suggestion_requires_ocr_boundary_signal() -> None:
    inp = WorkflowIngestInput(
        request_id="continuation-signal",
        collections=[
            {
                "collection_id": "continuation-signal",
                "title": "Continuation",
                "modality": "ocr",
                "pages": [
                    {
                        "page_number": 1,
                        "metadata": {"incomplete_text": True},
                        "units": [SourceUnit(modality="text", page_number=1, text="head")],
                    },
                    {"page_number": 2, "units": [SourceUnit(modality="text", page_number=2, text="tail")]},
                ],
            }
        ],
    )
    tree = build_document_tree(inp)
    assert tree is not None
    proposals = suggest_cross_page_continuations(tree)
    assert len(proposals) == 1
    assert proposals[0].kind == "layout:cross-page-continuation"


def test_cross_page_continuation_never_merges_text_into_nontext() -> None:
    inp = WorkflowIngestInput(
        request_id="mixed-pages",
        collections=[
            {
                "collection_id": "mixed-pages",
                "title": "Mixed pages",
                "modality": "ocr",
                "pages": [
                    {
                        "page_number": 1,
                        "metadata": {"incomplete_text": True},
                        "units": [SourceUnit(modality="text", page_number=1, text="continued")],
                    },
                    {
                        "page_number": 2,
                        "units": [
                            SourceUnit(
                                modality="image_region",
                                page_number=2,
                                description="figure caption",
                            )
                        ],
                    },
                ],
            }
        ],
    )

    tree = build_document_tree(inp)
    assert tree is not None
    assert suggest_cross_page_continuations(tree) == ()


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


def test_normalize_ocr_pages_preserves_unified_text_and_nontext_order() -> None:
    inp = normalize_ocr_pages(
        document_id="mixed-order-doc",
        title="Mixed order",
        pages=[
            {
                "pdf_page_num": 1,
                "meaningful_ordering": [0, 1, 2],
                "OCR_text_clusters": [
                    {"cluster_number": 0, "text": "before", "bb_x_max": 10, "bb_y_max": 10},
                    {"cluster_number": 2, "text": "after", "bb_x_max": 10, "bb_y_max": 10},
                ],
                "non_text_objects": [
                    {
                        "cluster_number": 1,
                        "description": "figure",
                        "bb_x_max": 10,
                        "bb_y_max": 10,
                    }
                ],
            }
        ],
    )

    units = inp.collections[0].pages[0].units
    assert [unit.cluster_number for unit in units] == [0, 1, 2]
    assert [unit.modality for unit in units] == ["ocr_text", "image_region", "ocr_text"]


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


def test_synthetic_page_box_does_not_claim_region_accuracy() -> None:
    inp = normalize_ocr_pages(
        document_id="synthetic-page-box",
        title="Synthetic page box",
        pages=[
            {
                "pdf_page_num": 1,
                "spatial_accuracy": "page",
                "OCR_text_clusters": [
                    {
                        "cluster_number": 0,
                        "text": "page fallback",
                        "bb_x_max": 100,
                        "bb_y_max": 200,
                    }
                ],
                "non_text_objects": [],
            }
        ],
    )

    tree = build_document_tree(inp)
    assert tree is not None
    assert tree.source_items[0].location.accuracy == "page"


def test_explicit_region_precision_is_preserved() -> None:
    inp = normalize_ocr_pages(
        document_id="declared-region",
        title="Declared region",
        pages=[
            {
                "pdf_page_num": 1,
                "spatial_accuracy": "region",
                "OCR_text_clusters": [
                    {
                        "cluster_number": 0,
                        "text": "located text",
                        "bb_x_max": 10,
                        "bb_y_max": 10,
                    }
                ],
                "non_text_objects": [],
            }
        ],
    )

    tree = build_document_tree(inp)
    assert tree is not None
    assert tree.source_items[0].location.accuracy == "region"


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


def test_build_document_tree_rejects_unrepresented_nontext_unit() -> None:
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
                                unit_id="figure-1",
                                modality="pure_image",
                                page_number=1,
                                source_uri="assets/figure.png",
                            )
                        ],
                    }
                ],
            }
        ],
    )

    with pytest.raises(UnrepresentedSourceUnitError) as raised:
        build_document_tree(inp)

    assert raised.value.diagnostic == UnrepresentedSourceDiagnostic(
        unit_id="figure-1",
        page_number=1,
        modality="pure_image",
        reason=(
            "non-text evidence requires a description or a verified "
            "content-addressed media reference"
        ),
    )


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


def test_final_group_keeps_identity_and_kind_under_document_root() -> None:
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
                            SourceUnit(modality="text", page_number=1, cluster_number=0, text="a"),
                            SourceUnit(modality="text", page_number=1, cluster_number=1, text="b"),
                        ],
                    }
                ],
            }
        ],
    )
    tree = build_document_tree(inp)
    assert tree is not None
    grouped = apply_grouping_proposals(
        tree,
        (GroupProposal(kind="layout:section", members=("leaf:doc|p1_t0", "leaf:doc|p1_t1")),),
        grouping_policy_version="v1",
    )

    root = next(node for node in grouped.nodes if node.node_id == grouped.root_id)
    assert len(root.children) == 1
    group_id = root.children[0].node_id
    group = next(node for node in grouped.nodes if node.node_id == group_id)
    assert group.kind == "layout:section"
    assert tuple(child.node_id for child in group.children) == (
        "leaf:doc|p1_t0",
        "leaf:doc|p1_t1",
    )


def test_replaying_an_accepted_grouping_proposal_is_idempotent() -> None:
    inp = WorkflowIngestInput.from_text(document_id="doc", text="alpha")
    inp.collections[0].pages[0].units = [
        SourceUnit(modality="text", page_number=1, cluster_number=0, text="alpha"),
        SourceUnit(modality="text", page_number=1, cluster_number=1, text="beta"),
    ]
    tree = build_document_tree(inp)
    assert tree is not None
    proposal = GroupProposal(
        kind="layout:section",
        members=("leaf:doc|p1_t0", "leaf:doc|p1_t1"),
    )
    grouped = apply_grouping_proposals(tree, (proposal,), grouping_policy_version="v1")
    replayed = apply_grouping_proposals(grouped, (proposal,), grouping_policy_version="v1")
    assert replayed.model_dump(mode="json") == grouped.model_dump(mode="json")


def test_grouping_change_invalidates_root_summary_without_changing_source_items() -> None:
    inp = WorkflowIngestInput.from_text(document_id="doc", text="alpha")
    inp.collections[0].pages[0].units = [
        SourceUnit(modality="text", page_number=1, cluster_number=0, text="alpha"),
        SourceUnit(modality="text", page_number=1, cluster_number=1, text="beta"),
    ]
    tree = build_document_tree(inp)
    assert tree is not None
    summary = Summary(
        status="generated",
        text="old grouping summary",
        evidence_item_ids=("doc|p1_t0", "doc|p1_t1"),
        provider_model="fake",
        prompt_revision="v1",
        input_fingerprint="old",
    )
    tree = tree.model_copy(
        update={
            "nodes": tuple(
                node.model_copy(update={"summary": summary})
                if node.node_id == tree.root_id
                else node
                for node in tree.nodes
            )
        }
    )
    grouped = apply_grouping_proposals(
        tree,
        (
            GroupProposal(
                kind="layout:section",
                members=("leaf:doc|p1_t0", "leaf:doc|p1_t1"),
            ),
        ),
        grouping_policy_version="v1",
    )
    root = next(node for node in grouped.nodes if node.node_id == grouped.root_id)
    assert root.summary is None
    assert grouped.source_items == tree.source_items


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
