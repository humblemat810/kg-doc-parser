from __future__ import annotations

import pytest

from kg_doc_parser.workflow_ingest.document_adapters import (
    apply_summary_updates,
    select_summary_targets,
)
from kg_doc_parser.workflow_ingest.document_grouping import GroupProposal, merge_groups
from kg_doc_parser.workflow_ingest.document_models import (
    DocumentNode,
    DocumentTree,
    MediaRef,
    SourceItem,
    SourceLocation,
    SourceSlice,
    Summary,
    SummaryPolicy,
)


def _item(item_id: str, text: str) -> SourceItem:
    return SourceItem(
        item_id=item_id,
        kind="text",
        text=text,
        location=SourceLocation(page_index=0, accuracy="page"),
    )


def _leaf(node_id: str, item_id: str, text: str, *, kind: str = "paragraph") -> DocumentNode:
    return DocumentNode(
        node_id=node_id,
        kind=kind,
        parts=(SourceSlice(item_id=item_id, start=0, end=len(text)),),
    )


def test_document_tree_requires_exact_source_coverage() -> None:
    item = _item("a", "same")
    tree = DocumentTree(
        document_id="doc",
        source_revision="r1",
        tree_revision=1,
        root_id="root",
        source_items=(item,),
        nodes=(_leaf("root", "a", "same", kind="document"),),
    )
    assert tree.source_fingerprint()

    with pytest.raises(ValueError, match="incomplete source coverage"):
        DocumentTree(
            document_id="doc",
            source_revision="r1",
            tree_revision=1,
            root_id="root",
            source_items=(item,),
            nodes=(DocumentNode(node_id="root", kind="document", parts=(SourceSlice(item_id="a", start=0, end=3),)),),
        )


def test_document_tree_rejects_unary_internal_nodes() -> None:
    with pytest.raises(ValueError, match="unary"):
        DocumentNode(node_id="root", kind="document", children=({"node_id": "child"},))


def test_merge_groups_is_deterministic_and_preserves_unmerged_members() -> None:
    proposal = GroupProposal(kind="layout:paragraph", members=("a", "b"))
    first = merge_groups(document_id="doc", frontier=("a", "b", "c"), proposals=(proposal,), grouping_policy_version="v1")
    second = merge_groups(document_id="doc", frontier=("a", "b", "c"), proposals=(proposal,), grouping_policy_version="v1")
    assert first == second
    assert first[0] == (first[0][0], "c")
    assert len(first[1][0].children) == 2


def test_merge_groups_rejects_nonadjacent_or_overlapping_members() -> None:
    with pytest.raises(ValueError):
        merge_groups(
            document_id="doc",
            frontier=("a", "b", "c"),
            proposals=(GroupProposal(kind="section", members=("a", "c")),),
            grouping_policy_version="v1",
        )


def test_identical_text_items_remain_distinct_by_identity() -> None:
    left = _item("left", "same")
    right = _item("right", "same")
    assert left.text == right.text and left.item_id != right.item_id


def test_generated_summary_requires_current_descendant_fingerprint() -> None:
    base = DocumentTree(
        document_id="doc",
        source_revision="r1",
        tree_revision=1,
        root_id="root",
        source_items=(_item("a", "one"), _item("b", "two")),
        nodes=(
            _leaf("a-leaf", "a", "one"),
            _leaf("b-leaf", "b", "two"),
            DocumentNode(
                node_id="root",
                kind="document",
                children=({"node_id": "a-leaf"}, {"node_id": "b-leaf"}),
            ),
        ),
    )
    fingerprint = base.summary_input_fingerprint("root")
    payload = base.model_dump(mode="json")
    payload["nodes"][-1]["summary"] = {
        "status": "generated",
        "text": "one and two",
        "evidence_item_ids": ["a", "b"],
        "provider_model": "fake",
        "prompt_revision": "prompt-v1",
        "input_fingerprint": fingerprint,
    }
    validated = DocumentTree.model_validate(payload)
    assert validated.nodes[-1].summary is not None

    payload["nodes"][-1]["summary"]["input_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="summary fingerprint is stale"):
        DocumentTree.model_validate(payload)


def test_summary_updates_validate_grounding_before_tree_persistence() -> None:
    base = DocumentTree(
        document_id="summary-update",
        source_revision="r1",
        tree_revision=1,
        root_id="root",
        source_items=(_item("a", "one"), _item("b", "two")),
        nodes=(
            _leaf("a-leaf", "a", "one"),
            _leaf("b-leaf", "b", "two"),
            DocumentNode(
                node_id="root",
                kind="document",
                children=({"node_id": "a-leaf"}, {"node_id": "b-leaf"}),
            ),
        ),
    )
    updated = apply_summary_updates(
        base,
        {
            "root": Summary(
                status="generated",
                text="one and two",
                evidence_item_ids=("a", "b"),
                provider_model="fake",
                prompt_revision="prompt-v1",
                input_fingerprint=base.summary_input_fingerprint("root"),
            )
        },
    )
    assert updated.tree_revision == 2
    assert updated.nodes[-1].summary is not None
    with pytest.raises(ValueError, match="summary fingerprint is stale"):
        apply_summary_updates(
            base,
            {
                "root": Summary(
                    status="generated",
                    text="stale",
                    evidence_item_ids=("a",),
                    provider_model="fake",
                    prompt_revision="prompt-v1",
                    input_fingerprint="0" * 64,
                )
            },
        )
    with pytest.raises(ValueError, match="unknown node"):
        apply_summary_updates(base, {"missing": Summary(status="pending")})


def test_summary_target_planner_is_bounded_and_skips_low_information_nodes() -> None:
    tree = DocumentTree(
        document_id="summary-policy",
        source_revision="r1",
        tree_revision=1,
        root_id="root",
        source_items=(_item("a", "alpha"), _item("b", "beta"), _item("c", "x")),
        nodes=(
            _leaf("a-leaf", "a", "alpha"),
            _leaf("b-leaf", "b", "beta"),
            _leaf("c-leaf", "c", "x"),
            DocumentNode(
                node_id="section",
                kind="section",
                children=({"node_id": "a-leaf"}, {"node_id": "b-leaf"}),
            ),
            DocumentNode(
                node_id="root",
                kind="document",
                children=({"node_id": "section"}, {"node_id": "c-leaf"}),
            ),
        ),
    )
    assert select_summary_targets(
        tree,
        SummaryPolicy(minimum_source_chars=100),
    ) == ()
    assert select_summary_targets(
        tree,
        SummaryPolicy(maximum_input_tokens=10, maximum_targets=1),
    ) == ("root",)
    assert select_summary_targets(tree, SummaryPolicy(enabled=False)) == ()


def test_title_and_atomic_sixty_row_table_remain_two_grounded_children() -> None:
    title = _item("title", "Report")
    table_text = "\n".join(f"row-{index}|value-{index}" for index in range(60))
    table = _item("table", table_text)
    tree = DocumentTree(
        document_id="table-doc",
        source_revision="r1",
        tree_revision=1,
        root_id="root",
        source_items=(title, table),
        nodes=(
            _leaf("title-leaf", "title", title.text or "", kind="heading"),
            _leaf("table-leaf", "table", table.text or "", kind="table"),
            DocumentNode(
                node_id="root",
                kind="document",
                children=({"node_id": "title-leaf"}, {"node_id": "table-leaf"}),
            ),
        ),
    )
    assert len(tree.nodes[-1].children) == 2
    assert tree.render_preview() == f"Report{table_text}"


def test_multimodal_media_and_time_ranges_round_trip_without_fabricated_geometry() -> None:
    media_specs = (
        ("image", SourceLocation(page_index=0, accuracy="page")),
        ("table", SourceLocation(page_index=1, accuracy="page")),
        ("diagram", SourceLocation(page_index=2, accuracy="page")),
        ("audio", SourceLocation(start_seconds=1.0, end_seconds=2.5, accuracy="time")),
        ("video", SourceLocation(start_seconds=4.0, end_seconds=8.0, accuracy="time")),
    )
    items = tuple(
        SourceItem(
            item_id=f"{kind}-1",
            kind=kind,
            location=location,
            media=MediaRef(
                uri=f"https://example.invalid/{kind}.bin",
                mime_type="application/octet-stream",
                sha256_hex=f"{index + 1:064x}",
                byte_length=index + 1,
            ),
        )
        for index, (kind, location) in enumerate(media_specs)
    )
    leaves = tuple(
        DocumentNode(
            node_id=f"leaf:{item.item_id}",
            kind=item.kind,
            parts=(SourceSlice(item_id=item.item_id),),
        )
        for item in items
    )
    tree = DocumentTree(
        document_id="multimodal-doc",
        source_revision="r1",
        tree_revision=1,
        root_id="root",
        source_items=items,
        nodes=(*leaves, DocumentNode(
            node_id="root",
            kind="document",
            children=tuple({"node_id": leaf.node_id} for leaf in leaves),
        )),
    )

    round_tripped = DocumentTree.model_validate(tree.model_dump(mode="json"))
    assert [item.kind for item in round_tripped.source_items] == [spec[0] for spec in media_specs]
    assert [item.media.sha256_hex for item in round_tripped.source_items if item.media] == [
        f"{index + 1:064x}" for index in range(len(media_specs))
    ]
    assert round_tripped.source_items[-2].location.accuracy == "time"
    assert round_tripped.source_items[-1].location.end_seconds == 8.0


def test_source_fingerprint_includes_authoritative_reading_order() -> None:
    first = DocumentTree(
        document_id="order-doc",
        source_revision="r1",
        tree_revision=1,
        root_id="root",
        source_items=(_item("a", "a"), _item("b", "b")),
        nodes=(
            _leaf("a-leaf", "a", "a"),
            _leaf("b-leaf", "b", "b"),
            DocumentNode(node_id="root", kind="document", children=({"node_id": "a-leaf"}, {"node_id": "b-leaf"})),
        ),
    )
    reordered = first.model_dump(mode="json")
    reordered["source_items"] = list(reversed(reordered["source_items"]))
    second = DocumentTree.model_validate(reordered)
    assert first.source_fingerprint() != second.source_fingerprint()
