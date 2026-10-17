from __future__ import annotations

from collections.abc import Sequence
from typing import overload

import pytest

from kg_doc_parser.workflow_ingest.document_adapters import (
    apply_grouping_proposals,
    apply_summary_updates,
    extract_reference_mentions,
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


def _item(item_id: str, text: str, *, page_index: int = 0) -> SourceItem:
    return SourceItem(
        item_id=item_id,
        kind="text",
        text=text,
        location=SourceLocation(page_index=page_index, accuracy="page"),
    )


def _leaf(node_id: str, item_id: str, text: str, *, kind: str = "paragraph") -> DocumentNode:
    return DocumentNode(
        node_id=node_id,
        kind=kind,
        parts=(SourceSlice(item_id=item_id, start=0, end=len(text)),),
    )


class _CountingSequence(Sequence[str]):
    """Expose frontier access counts without making the grouping test timing-based."""

    def __init__(self, values: tuple[str, ...]) -> None:
        self._values = values
        self.accesses = 0

    def __len__(self) -> int:
        return len(self._values)

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[str, ...]: ...

    def __getitem__(self, index: int | slice) -> str | tuple[str, ...]:
        self.accesses += 1
        return self._values[index]


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
        DocumentNode(node_id="section", kind="section", children=({"node_id": "child"},))

    # The synthetic document root may retain one final meaningful group so
    # normalization does not discard that group's identity or provenance.
    DocumentNode(node_id="root", kind="document", children=({"node_id": "section"},))


def test_merge_groups_is_deterministic_and_preserves_unmerged_members() -> None:
    proposal = GroupProposal(kind="layout:paragraph", members=("a", "b"))
    first = merge_groups(document_id="doc", frontier=("a", "b", "c"), proposals=(proposal,), grouping_policy_version="v1")
    second = merge_groups(document_id="doc", frontier=("a", "b", "c"), proposals=(proposal,), grouping_policy_version="v1")
    assert first == second
    assert first[0] == (first[0][0], "c")
    assert len(first[1][0].children) == 2


def test_merge_groups_only_reifies_the_affected_frontier_window() -> None:
    proposal = GroupProposal(kind="section", members=("b", "c"))

    frontier, created = merge_groups(
        document_id="doc",
        frontier=("a", "b", "c", "d", "e"),
        proposals=(proposal,),
        grouping_policy_version="v1",
    )

    assert len(frontier) == 4
    assert frontier[0] == "a"
    assert frontier[2:] == ("d", "e")
    assert frontier[1].startswith("merge:")
    assert len(created) == 1
    assert tuple(child.node_id for child in created[0].children) == ("b", "c")


def test_merge_groups_keeps_large_unaffected_frontier_stable() -> None:
    frontier = tuple(f"leaf-{index}" for index in range(256))
    proposal = GroupProposal(kind="section", members=("leaf-254", "leaf-255"))

    updated_frontier, created = merge_groups(
        document_id="large-doc",
        frontier=frontier,
        proposals=(proposal,),
        grouping_policy_version="v1",
    )

    assert len(created) == 1
    assert len(updated_frontier) == 255
    assert updated_frontier[:-1] == frontier[:-2]
    assert updated_frontier[-1].startswith("merge:")


def test_merge_groups_scans_frontier_linearly_without_nested_rebuilds() -> None:
    values = tuple(f"leaf-{index}" for index in range(512))
    frontier = _CountingSequence(values)
    proposal = GroupProposal(kind="section", members=("leaf-254", "leaf-255"))

    updated_frontier, created = merge_groups(
        document_id="linear-frontier-doc",
        frontier=frontier,
        proposals=(proposal,),
        grouping_policy_version="v1",
    )

    assert len(created) == 1
    assert len(updated_frontier) == 511
    # Validation, position indexing, and result emission each scan once. The
    # small allowance covers Python's sequence-stop probes; a quadratic
    # rebuild would exceed this bound as the frontier grows.
    assert frontier.accesses <= 3 * len(values) + 3


def test_grouping_replay_can_include_accepted_and_new_proposals() -> None:
    tree = DocumentTree(
        document_id="mixed-replay",
        source_revision="r1",
        tree_revision=1,
        root_id="root",
        source_items=tuple(_item(f"item-{index}", str(index)) for index in range(4)),
        nodes=tuple(
            [
                _leaf(f"leaf-{index}", f"item-{index}", str(index))
                for index in range(4)
            ]
            + [
                DocumentNode(
                    node_id="root",
                    kind="document",
                    children=tuple(
                        {"node_id": f"leaf-{index}"} for index in range(4)
                    ),
                )
            ]
        ),
    )
    first = apply_grouping_proposals(
        tree,
        (GroupProposal(kind="section", members=("leaf-0", "leaf-1")),),
        grouping_policy_version="v1",
    )
    second = apply_grouping_proposals(
        first,
        (
            GroupProposal(kind="section", members=("leaf-0", "leaf-1")),
            GroupProposal(kind="section", members=("leaf-2", "leaf-3")),
        ),
        grouping_policy_version="v1",
    )

    root = next(node for node in second.nodes if node.node_id == "root")
    assert len(root.children) == 2
    assert all(child.node_id.startswith("merge:") for child in root.children)
    assert second.tree_revision == first.tree_revision + 1


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
        SummaryPolicy(
            maximum_input_tokens=2048,
            maximum_targets=1,
            eligible_kinds=("document",),
        ),
    ) == ("root",)
    assert select_summary_targets(tree, SummaryPolicy(enabled=False)) == ()


def test_summary_target_planner_counts_exact_slices_not_whole_source_items() -> None:
    source = _item("source", "alpha beta gamma")
    tree = DocumentTree(
        document_id="slice-budget",
        source_revision="r1",
        tree_revision=1,
        root_id="root",
        source_items=(source,),
        nodes=(
            DocumentNode(
                node_id="section",
                kind="section",
                children=(
                    {"node_id": "first"},
                    {"node_id": "second"},
                ),
            ),
            DocumentNode(
                node_id="first",
                kind="paragraph",
                parts=(SourceSlice(item_id="source", start=0, end=5),),
            ),
            DocumentNode(
                node_id="second",
                kind="paragraph",
                parts=(SourceSlice(item_id="source", start=5, end=10),),
            ),
            DocumentNode(
                node_id="tail",
                kind="paragraph",
                parts=(SourceSlice(item_id="source", start=10, end=16),),
            ),
            DocumentNode(
                node_id="root",
                kind="document",
                children=({"node_id": "section"}, {"node_id": "tail"}),
            ),
        ),
    )

    assert select_summary_targets(
        tree,
        SummaryPolicy(
            minimum_source_items=1,
            minimum_source_chars=12,
            minimum_information_gain_chars=0,
        ),
    ) == ("root",)
    assert select_summary_targets(
        tree,
        SummaryPolicy(
            minimum_source_items=1,
            minimum_source_chars=12,
            minimum_information_gain_chars=0,
            eligible_kinds=("section",),
        ),
    ) == ()


def test_summary_target_planner_does_not_send_oversized_evidence() -> None:
    large = "x" * 20_000
    tree = DocumentTree(
        document_id="oversized-summary",
        source_revision="r1",
        tree_revision=1,
        root_id="root",
        source_items=(_item("a", large), _item("b", large)),
        nodes=(
            _leaf("a-leaf", "a", large),
            _leaf("b-leaf", "b", large),
            DocumentNode(
                node_id="root",
                kind="document",
                children=({"node_id": "a-leaf"}, {"node_id": "b-leaf"}),
            ),
        ),
    )

    assert select_summary_targets(
        tree,
        SummaryPolicy(maximum_input_tokens=128, maximum_targets=1),
    ) == ()


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


@pytest.mark.ci
def test_source_fingerprint_includes_document_and_revision_namespaces() -> None:
    base = DocumentTree(
        document_id="namespace-doc",
        source_revision="source-rev-1",
        tree_revision=1,
        root_id="root",
        source_items=(_item("source", "stable text"),),
        nodes=(
            _leaf("source-leaf", "source", "stable text"),
            DocumentNode(node_id="root", kind="document", children=({"node_id": "source-leaf"},)),
        ),
    )

    same_items_other_document = base.model_copy(update={"document_id": "other-document"})
    same_items_other_revision = base.model_copy(update={"source_revision": "source-rev-2"})

    assert base.source_fingerprint() != same_items_other_document.source_fingerprint()
    assert base.source_fingerprint() != same_items_other_revision.source_fingerprint()
    assert base.summary_input_fingerprint("root") != same_items_other_document.summary_input_fingerprint("root")
    assert base.summary_input_fingerprint("root") != same_items_other_revision.summary_input_fingerprint("root")


def test_explicit_reference_mentions_are_source_grounded_and_outside_partition() -> None:
    text = "Intro; see page 5 and refer to section 4.2 for details."
    item = _item("source", text)
    tree = DocumentTree(
        document_id="references",
        source_revision="r1",
        tree_revision=1,
        root_id="root",
        source_items=(item,),
        nodes=(_leaf("root", "source", text, kind="document"),),
    )

    mentions = extract_reference_mentions(tree)
    assert [(mention.target_kind, mention.target_label) for mention in mentions] == [
        ("page", "5"),
        ("section", "4.2"),
    ]
    assert all(mention.status == "pending" for mention in mentions)
    assert all(mention.source.item_id == "source" for mention in mentions)
    assert all(text[mention.source.start : mention.source.end] for mention in mentions)
    assert tree.render_preview() == text


def test_reference_mentions_cross_ocr_units_preserve_exact_source_slices() -> None:
    first = _item("page-1-cluster-1", "Intro; see Section")
    second = _item("page-2-cluster-1", " 4.2 for details.", page_index=1)
    tree = DocumentTree(
        document_id="cross-unit-references",
        source_revision="r1",
        tree_revision=1,
        root_id="root",
        source_items=(first, second),
        nodes=(
            _leaf("first", first.item_id, first.text or ""),
            _leaf("second", second.item_id, second.text or ""),
            DocumentNode(
                node_id="root",
                kind="document",
                children=({"node_id": "first"}, {"node_id": "second"}),
            ),
        ),
    )

    mentions = extract_reference_mentions(tree)
    assert len(mentions) == 1
    mention = mentions[0]
    assert (mention.target_kind, mention.target_label) == ("section", "4.2")
    assert [(source.item_id, source.start, source.end) for source in mention.sources] == [
        (first.item_id, 7, len(first.text)),
        (second.item_id, 0, 4),
    ]


def test_reference_mentions_round_trip_and_validate_targets_without_becoming_children() -> None:
    text = "See figure 2."
    base = DocumentTree(
        document_id="reference-roundtrip",
        source_revision="r1",
        tree_revision=1,
        root_id="root",
        source_items=(_item("source", text), _item("target", "Figure")),
        nodes=(
            _leaf("source-leaf", "source", text),
            _leaf("target-leaf", "target", "Figure"),
            DocumentNode(node_id="root", kind="document", children=({"node_id": "source-leaf"}, {"node_id": "target-leaf"})),
        ),
    )
    mention = extract_reference_mentions(base)[0].model_copy(
        update={"target_node_id": "target-leaf", "status": "resolved"}
    )
    payload = base.model_dump(mode="json")
    payload["references"] = [mention.model_dump(mode="json")]
    round_tripped = DocumentTree.model_validate(payload)
    assert round_tripped.references[0].target_node_id == "target-leaf"
    assert all(reference.source.item_id != child.node_id for reference in round_tripped.references for child in round_tripped.nodes[-1].children)


def test_reference_mentions_reject_unknown_targets_and_non_text_sources() -> None:
    text = "See page 5."
    base = DocumentTree(
        document_id="reference-invalid",
        source_revision="r1",
        tree_revision=1,
        root_id="root",
        source_items=(_item("source", text),),
        nodes=(_leaf("root", "source", text, kind="document"),),
    )
    payload = base.model_dump(mode="json")
    payload["references"] = [
        {
            "mention_id": "mention:bad",
            "source": {"item_id": "source", "start": 0, "end": len(text)},
            "target_kind": "page",
            "target_label": "5",
            "target_node_id": "missing",
            "status": "resolved",
        }
    ]
    with pytest.raises(ValueError, match="target node"):
        DocumentTree.model_validate(payload)


def test_reference_mention_round_trips_multiple_cross_page_source_slices() -> None:
    first_text = "See page 2 for the continuation"
    second_text = "Continuation on the next page"
    base = DocumentTree(
        document_id="cross-page-reference",
        source_revision="r1",
        tree_revision=1,
        root_id="root",
        source_items=(
            _item("page-1", first_text, page_index=0),
            _item("page-2", second_text, page_index=1),
        ),
        nodes=(
            _leaf("page-1-leaf", "page-1", first_text),
            _leaf("page-2-leaf", "page-2", second_text),
            DocumentNode(
                node_id="root",
                kind="document",
                children=(
                    {"node_id": "page-1-leaf"},
                    {"node_id": "page-2-leaf"},
                ),
            ),
        ),
    )
    mention = extract_reference_mentions(base)[0].model_copy(
        update={
            "additional_sources": (
                SourceSlice(item_id="page-2", start=0, end=len(second_text)),
            )
        }
    )
    payload = base.model_dump(mode="json")
    payload["references"] = [mention.model_dump(mode="json")]

    restored = DocumentTree.model_validate(payload)
    assert len(restored.references) == 1
    assert [source.item_id for source in restored.references[0].sources] == ["page-1", "page-2"]
    assert [source.start for source in restored.references[0].sources] == [0, 0]

    duplicate = dict(payload["references"][0])
    duplicate["additional_sources"] = [duplicate["source"]]
    with pytest.raises(ValueError, match="duplicate source slice"):
        DocumentTree.model_validate({**payload, "references": [duplicate]})
