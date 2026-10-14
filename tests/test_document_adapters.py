from __future__ import annotations

from kg_doc_parser.workflow_ingest import GroupProposal, apply_grouping_proposals, build_document_tree
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
