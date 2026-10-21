"""SQLite grouping-tree and checkpoint persistence contracts."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import pytest
from ocr_workflow_support import scratch_path as _scratch

from kg_doc_parser.workflow_ingest.document_models import (
    ChildRef,
    DocumentNode,
    DocumentTree,
    SourceItem,
    SourceLocation,
    SourceSlice,
)
from kg_doc_parser.workflow_ingest.ocr_state_store import (
    GroupingCheckpointConflict,
    OCRWorkflowStateStore,
)

pytestmark = [pytest.mark.workflow]

def test_ocr_state_store_round_trips_grouped_tree_and_invalidates_changed_source() -> (
    None
):
    scratch = _scratch("ocr_grouping_checkpoint")
    tree = DocumentTree(
        document_id="grouped-doc",
        source_revision="ocr-v1",
        tree_revision=1,
        root_id="root",
        source_items=(
            SourceItem(
                item_id="a",
                kind="text",
                location=SourceLocation(page_index=0, accuracy="page"),
                text="alpha",
            ),
            SourceItem(
                item_id="b",
                kind="text",
                location=SourceLocation(page_index=0, accuracy="page"),
                text="beta",
            ),
            SourceItem(
                item_id="c",
                kind="text",
                location=SourceLocation(page_index=0, accuracy="page"),
                text="gamma",
            ),
        ),
        nodes=(
            DocumentNode(
                node_id="root",
                kind="document",
                children=(ChildRef(node_id="group"), ChildRef(node_id="leaf")),
            ),
            DocumentNode(
                node_id="group",
                kind="section",
                parts=(
                    SourceSlice(item_id="a", start=0, end=5),
                    SourceSlice(item_id="b", start=0, end=4, separator_before="\n"),
                ),
            ),
            DocumentNode(
                node_id="leaf",
                kind="text",
                parts=(SourceSlice(item_id="c", start=0, end=5),),
            ),
        ),
    )
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="grouped-doc",
        title="Grouped",
        source_kind="image",
        total_pages=1,
        input_fingerprint="source-v1",
    )
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            """
            INSERT INTO page_state (
                document_id, page_number, stage, attempt_count, status,
                content_hash, artifact_path, config_fingerprint
            ) VALUES (?, ?, 'ocr', ?, 'completed', ?, ?, ?)
            """,
            ("grouped-doc", 1, 1, "page-hash", "page-1.json", "ocr-config-v1"),
        )
    revision = store.save_document_tree(
        document_id="grouped-doc", tree=tree, grouping_policy_version="group-v1"
    )
    assert len(revision) == 64
    assert store.document_tree_revision(document_id="grouped-doc") == revision
    checkpoint = store.load_grouping_checkpoint_metadata(
        document_id="grouped-doc",
        source_fingerprint=tree.source_fingerprint(),
        grouping_policy_version="group-v1",
    )
    assert checkpoint is not None
    assert checkpoint.accepted_page_frontier == (1,)
    assert checkpoint.tree_revision == revision
    assert checkpoint.merge_revision == 1
    assert checkpoint.last_runtime_step == "grouping_checkpoint"
    accepted_tree = tree.model_copy(update={"tree_revision": 2})
    accepted_revision = store.save_document_tree(
        document_id="grouped-doc",
        tree=accepted_tree,
        grouping_policy_version="group-v1",
        expected_tree_revision=revision,
    )
    updated_checkpoint = store.load_grouping_checkpoint_metadata(
        document_id="grouped-doc",
        source_fingerprint=accepted_tree.source_fingerprint(),
        grouping_policy_version="group-v1",
    )
    assert updated_checkpoint is not None
    assert updated_checkpoint.merge_revision == 2
    # A crash after the SQLite commit but before the runtime checkpoint must
    # make an exact replay a no-op, not turn an accepted tree into a conflict.
    assert (
        store.save_document_tree(
            document_id="grouped-doc",
            tree=accepted_tree,
            grouping_policy_version="group-v1",
            expected_tree_revision=revision,
        )
        == accepted_revision
    )
    replayed_checkpoint = store.load_grouping_checkpoint_metadata(
        document_id="grouped-doc",
        source_fingerprint=accepted_tree.source_fingerprint(),
        grouping_policy_version="group-v1",
    )
    assert replayed_checkpoint is not None
    assert replayed_checkpoint.merge_revision == 2
    with pytest.raises(RuntimeError, match="create conflict"):
        store.save_document_tree(
            document_id="grouped-doc",
            tree=tree,
            grouping_policy_version="group-v1",
            require_no_existing_tree=True,
        )
    with pytest.raises(RuntimeError, match="revision conflict"):
        store.save_document_tree(
            document_id="grouped-doc",
            tree=tree.model_copy(update={"tree_revision": 3}),
            grouping_policy_version="group-v1",
            expected_tree_revision="stale-revision",
        )
    with pytest.raises(RuntimeError, match="requires an expected tree revision"):
        store.save_document_tree(
            document_id="grouped-doc",
            tree=tree,
            grouping_policy_version="group-v1",
        )
    assert store.load_grouping_frontier(
        document_id="grouped-doc",
        source_fingerprint=tree.source_fingerprint(),
        grouping_policy_version="group-v1",
    ) == ("group", "leaf")
    assert (
        store.load_grouping_frontier(
            document_id="grouped-doc",
            source_fingerprint=tree.source_fingerprint(),
            grouping_policy_version="different-policy",
        )
        is None
    )
    restored = store.load_document_tree(
        document_id="grouped-doc", source_fingerprint=tree.source_fingerprint()
    )
    assert restored is not None
    assert restored.model_dump(mode="json") == accepted_tree.model_dump(mode="json")
    assert (
        store.load_document_tree(
            document_id="grouped-doc",
            source_fingerprint=tree.source_fingerprint(),
            grouping_policy_version="different-policy",
        )
        is None
    )

    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "DELETE FROM document_tree_state WHERE document_id = ?", ("grouped-doc",)
        )
    with pytest.raises(RuntimeError, match="create conflict.*frontier"):
        store.save_document_tree(
            document_id="grouped-doc",
            tree=tree,
            grouping_policy_version="group-v1",
            require_no_existing_tree=True,
        )

    with pytest.raises(RuntimeError, match="orphaned frontier"):
        store.save_document_tree(
            document_id="grouped-doc",
            tree=tree,
            grouping_policy_version="group-v1",
        )

    inconsistent_store = OCRWorkflowStateStore(scratch / "inconsistent.sqlite")
    inconsistent_store.ensure_document(
        document_id="grouped-doc",
        title="Grouped",
        source_kind="image",
        total_pages=1,
        input_fingerprint="source-v1",
    )
    inconsistent_revision = inconsistent_store.save_document_tree(
        document_id="grouped-doc",
        tree=tree,
        grouping_policy_version="group-v1",
    )
    with sqlite3.connect(inconsistent_store.db_path) as conn:
        conn.execute(
            "DELETE FROM grouping_frontier_state WHERE document_id = ?",
            ("grouped-doc",),
        )
    with pytest.raises(RuntimeError, match="frontier is missing"):
        inconsistent_store.save_document_tree(
            document_id="grouped-doc",
            tree=tree,
            grouping_policy_version="group-v1",
            expected_tree_revision=inconsistent_revision,
        )

    store.ensure_document(
        document_id="grouped-doc",
        title="Grouped",
        source_kind="image",
        total_pages=1,
        input_fingerprint="source-v2",
    )
    assert store.load_document_tree(document_id="grouped-doc") is None
    assert store.load_grouping_frontier(document_id="grouped-doc") is None
    assert store.load_grouping_checkpoint_metadata(document_id="grouped-doc") is None


@pytest.mark.ci
def test_grouping_checkpoint_rejects_stale_writer_after_source_revision_changes() -> (
    None
):
    scratch = _scratch("ocr_grouping_stale_source_writer")
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")

    def source_tree(source_revision: str, text: str) -> DocumentTree:
        return DocumentTree(
            document_id="stale-grouping-doc",
            source_revision=source_revision,
            tree_revision=1,
            root_id="root",
            source_items=(
                SourceItem(
                    item_id="item",
                    kind="text",
                    text=text,
                    location=SourceLocation(page_index=0, accuracy="page"),
                ),
            ),
            nodes=(
                DocumentNode(
                    node_id="root",
                    kind="document",
                    parts=(SourceSlice(item_id="item", start=0, end=len(text)),),
                ),
            ),
        )

    old_tree = source_tree("ocr-v1:source-v1", "old source")
    store.ensure_document(
        document_id=old_tree.document_id,
        title="Source revision race",
        source_kind="image",
        total_pages=1,
        input_fingerprint="source-v1",
    )
    store.save_document_tree(
        document_id=old_tree.document_id,
        tree=old_tree,
        require_no_existing_tree=True,
    )

    store.ensure_document(
        document_id=old_tree.document_id,
        title="Source revision race",
        source_kind="image",
        total_pages=1,
        input_fingerprint="source-v2",
    )
    with pytest.raises(GroupingCheckpointConflict, match="source revision conflict"):
        store.save_document_tree(
            document_id=old_tree.document_id,
            tree=old_tree,
            require_no_existing_tree=True,
        )

    new_tree = source_tree("ocr-v1:source-v2", "new source")
    store.save_document_tree(
        document_id=new_tree.document_id,
        tree=new_tree,
        require_no_existing_tree=True,
    )


@pytest.mark.ci
def test_grouping_checkpoint_concurrent_updates_accept_one_revision() -> None:
    scratch = _scratch("ocr_grouping_checkpoint_concurrent_updates")
    store_path = scratch / "ocr-state.sqlite"
    document_id = "concurrent-grouping-checkpoint-doc"
    store = OCRWorkflowStateStore(store_path)
    store.ensure_document(
        document_id=document_id,
        title="Concurrent grouping checkpoint",
        source_kind="image",
        total_pages=1,
        input_fingerprint="source-v1",
    )
    initial_tree = DocumentTree(
        document_id=document_id,
        source_revision="ocr-v1:source-v1",
        tree_revision=1,
        root_id="root",
        source_items=(
            SourceItem(
                item_id="item",
                kind="text",
                text="checkpoint",
                location=SourceLocation(page_index=0, accuracy="page"),
            ),
        ),
        nodes=(
            DocumentNode(
                node_id="root",
                kind="document",
                parts=(SourceSlice(item_id="item", start=0, end=10),),
            ),
        ),
    )
    initial_revision = store.save_document_tree(
        document_id=document_id,
        tree=initial_tree,
        require_no_existing_tree=True,
    )
    candidates = (
        initial_tree.model_copy(update={"tree_revision": 2}),
        initial_tree.model_copy(update={"tree_revision": 3}),
    )
    stores = (OCRWorkflowStateStore(store_path), OCRWorkflowStateStore(store_path))
    barrier = threading.Barrier(2)

    def _update(
        index: int,
    ) -> tuple[Literal["accepted", "conflict"], str | None]:
        barrier.wait(timeout=10)
        try:
            revision = stores[index].save_document_tree(
                document_id=document_id,
                tree=candidates[index],
                expected_tree_revision=initial_revision,
            )
        except GroupingCheckpointConflict:
            return "conflict", None
        return "accepted", revision

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(_update, (0, 1)))

    accepted = [revision for status, revision in outcomes if status == "accepted"]
    assert len(accepted) == 1
    assert sum(status == "conflict" for status, _ in outcomes) == 1
    restored = store.load_document_tree(document_id=document_id)
    assert restored in candidates
    assert store.document_tree_revision(document_id=document_id) == accepted[0]
    metadata = store.load_grouping_checkpoint_metadata(document_id=document_id)
    assert metadata is not None
    assert metadata.tree_revision == accepted[0]
    assert metadata.merge_revision == 2


@pytest.mark.ci
def test_grouping_checkpoint_load_rejects_mismatched_companion_records() -> None:
    scratch = _scratch("ocr_grouping_checkpoint_companion_mismatch")
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    tree = DocumentTree(
        document_id="checkpoint-companion-doc",
        source_revision="ocr-v1:source-v1",
        tree_revision=1,
        root_id="root",
        source_items=(
            SourceItem(
                item_id="item",
                kind="text",
                text="checkpoint",
                location=SourceLocation(page_index=0, accuracy="page"),
            ),
        ),
        nodes=(
            DocumentNode(
                node_id="root",
                kind="document",
                parts=(SourceSlice(item_id="item", start=0, end=10),),
            ),
        ),
    )
    store.ensure_document(
        document_id=tree.document_id,
        title="Checkpoint companion validation",
        source_kind="image",
        total_pages=1,
        input_fingerprint="source-v1",
    )
    revision = store.save_document_tree(
        document_id=tree.document_id,
        tree=tree,
        require_no_existing_tree=True,
    )
    assert store.load_document_tree(document_id=tree.document_id) == tree

    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE document_tree_state SET source_fingerprint = ? WHERE document_id = ?",
            (sqlite3.Binary(b"corrupt-source"), tree.document_id),
        )
    with pytest.raises(ValueError, match="source_fingerprint.*must be text"):
        store.load_document_tree(document_id=tree.document_id)
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE document_tree_state SET source_fingerprint = ? WHERE document_id = ?",
            (tree.source_fingerprint(), tree.document_id),
        )

    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE grouping_checkpoint_metadata SET tree_revision = ? WHERE document_id = ?",
            ("stale-revision", tree.document_id),
        )
    with pytest.raises(
        GroupingCheckpointConflict, match="idempotent grouping checkpoint replay"
    ):
        store.save_document_tree(
            document_id=tree.document_id,
            tree=tree,
        )
    with pytest.raises(ValueError, match="checkpoint records disagree"):
        store.load_document_tree(document_id=tree.document_id)

    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE grouping_checkpoint_metadata SET tree_revision = ? WHERE document_id = ?",
            (revision, tree.document_id),
        )
        conn.execute(
            "UPDATE grouping_frontier_state SET frontier_json = ? WHERE document_id = ?",
            (json.dumps(["foreign-frontier"]), tree.document_id),
        )
    with pytest.raises(
        GroupingCheckpointConflict, match="idempotent grouping checkpoint replay"
    ):
        store.save_document_tree(
            document_id=tree.document_id,
            tree=tree,
        )
    with pytest.raises(ValueError, match="frontier does not match"):
        store.load_document_tree(document_id=tree.document_id)

    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE grouping_frontier_state SET frontier_json = ? WHERE document_id = ?",
            (json.dumps(["root"]), tree.document_id),
        )
        conn.execute(
            "UPDATE grouping_checkpoint_metadata "
            "SET accepted_page_frontier_json = ? WHERE document_id = ?",
            (json.dumps([1]), tree.document_id),
        )
    with pytest.raises(
        GroupingCheckpointConflict, match="idempotent grouping checkpoint replay"
    ):
        store.save_document_tree(
            document_id=tree.document_id,
            tree=tree,
        )
    with pytest.raises(ValueError, match="accepted-page frontier"):
        store.load_document_tree(document_id=tree.document_id)

    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE grouping_checkpoint_metadata "
            "SET accepted_page_frontier_json = ? WHERE document_id = ?",
            (json.dumps([]), tree.document_id),
        )
        tree_row = conn.execute(
            "SELECT tree_json FROM document_tree_state WHERE document_id = ?",
            (tree.document_id,),
        ).fetchone()
        assert tree_row is not None
        payload = json.loads(str(tree_row[0]))
        payload["source_items"][0]["text"] = "tampered!!"
        tree_json = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        tampered_revision = hashlib.sha256(tree_json.encode("utf-8")).hexdigest()
        conn.execute(
            "UPDATE document_tree_state SET tree_json = ?, tree_revision = ? WHERE document_id = ?",
            (tree_json, tampered_revision, tree.document_id),
        )
        conn.execute(
            "UPDATE grouping_checkpoint_metadata SET tree_revision = ? WHERE document_id = ?",
            (tampered_revision, tree.document_id),
        )
    with pytest.raises(ValueError, match="tree source fingerprint is inconsistent"):
        store.load_document_tree(document_id=tree.document_id)

    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "DELETE FROM document_tree_state WHERE document_id = ?",
            (tree.document_id,),
        )
        conn.execute(
            "DELETE FROM grouping_frontier_state WHERE document_id = ?",
            (tree.document_id,),
        )
    with pytest.raises(
        GroupingCheckpointConflict, match="orphaned frontier or metadata"
    ):
        store.save_document_tree(
            document_id=tree.document_id,
            tree=tree,
            require_no_existing_tree=True,
        )
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "DELETE FROM grouping_checkpoint_metadata WHERE document_id = ?",
            (tree.document_id,),
        )
    store.save_document_tree(
        document_id=tree.document_id,
        tree=tree,
        require_no_existing_tree=True,
    )
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "DELETE FROM grouping_checkpoint_metadata WHERE document_id = ?",
            (tree.document_id,),
        )
    with pytest.raises(ValueError, match="missing a companion record"):
        store.load_document_tree(document_id=tree.document_id)
