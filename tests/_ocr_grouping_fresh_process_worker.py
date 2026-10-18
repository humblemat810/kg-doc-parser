from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

from _kogwistar_test_helpers import build_workflow_engine_triplet
from kogwistar.engine_core.rust_meta_sqlite import build_sqlite_meta_store

from kg_doc_parser.workflow_ingest.document_grouping import GroupProposal
from kg_doc_parser.workflow_ingest.document_models import (
    ChildRef,
    DocumentNode,
    DocumentTree,
    SourceItem,
    SourceLocation,
    SourceSlice,
    Summary,
    SummaryPolicy,
)
from kg_doc_parser.workflow_ingest.ocr_pipeline import OCRWorkflowStateStore
from kg_doc_parser.workflow_ingest.ocr_runtime import run_grouping_workflow


def _tree() -> DocumentTree:
    return DocumentTree(
        document_id="fresh-grouping-document",
        source_revision="grouping-v1:fresh-process",
        tree_revision=1,
        root_id="root",
        source_items=(
            SourceItem(
                item_id="item-a",
                kind="text",
                text="alpha",
                location=SourceLocation(page_index=0, accuracy="page"),
            ),
            SourceItem(
                item_id="item-b",
                kind="text",
                text="beta",
                location=SourceLocation(page_index=0, accuracy="page"),
            ),
        ),
        nodes=(
            DocumentNode(
                node_id="leaf-a",
                kind="text",
                parts=(SourceSlice(item_id="item-a", start=0, end=5),),
            ),
            DocumentNode(
                node_id="leaf-b",
                kind="text",
                parts=(SourceSlice(item_id="item-b", start=0, end=4),),
            ),
            DocumentNode(
                node_id="root",
                kind="document",
                children=(ChildRef(node_id="leaf-a"), ChildRef(node_id="leaf-b")),
            ),
        ),
    )


def main() -> int:
    os.environ.setdefault("KOGWISTAR_IMPL_META_STORE", "python")
    scratch = Path(sys.argv[1])
    mode = sys.argv[2]
    engines_dir = scratch / "engines"
    workflow_engine, conversation_engine, _ = build_workflow_engine_triplet(
        engines_dir, "in_memory"
    )
    for name, engine in (
        ("workflow", workflow_engine),
        ("conversation", conversation_engine),
    ):
        metadata = build_sqlite_meta_store(engines_dir / name, "meta.sqlite")
        metadata.ensure_initialized()
        engine.metadata = metadata
        engine._phase1_enable_index_jobs = False

    state_store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    policy = SummaryPolicy(
        eligible_kinds=("document",),
        minimum_source_items=1,
        minimum_source_chars=1,
        minimum_information_gain_chars=0,
        generation_fingerprint="fresh-grouping-summary-v1",
    )

    if mode == "resume":
        # The interrupted process cannot release its lease.  Exercise the
        # real recovery path after the short test lease expires.
        time.sleep(30.2)

    def summary_generator(
        *,
        tree: DocumentTree,
        node_id: str,
        source_items: tuple[SourceItem, ...],
        input_fingerprint: str,
    ) -> Summary:
        if mode == "hold":
            (scratch / "grouping-summary-started").write_text(
                "ready", encoding="utf-8"
            )
            time.sleep(120.0)
        return Summary(
            status="generated",
            text=f"summary:{node_id}",
            evidence_item_ids=tuple(item.item_id for item in source_items),
            provider_model="fresh-grouping-summary",
            prompt_revision="summary-v1",
            input_fingerprint=input_fingerprint,
        )

    try:
        result, run_id = run_grouping_workflow(
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            state_store=state_store,
            tree=_tree(),
            document_id="fresh-grouping-document",
            grouping_policy_version="fresh-grouping-v1",
            grouping_passes=(
                (GroupProposal(kind="section", members=("leaf-a", "leaf-b")),),
            ),
            summary_policy=policy,
            summary_generator=summary_generator,
            run_id="ocr-grouping|fresh-grouping-document",
            resume_from_checkpoint=mode == "resume",
            run_ownership_lease_ms=30_000,
        )
    except Exception:  # noqa: BLE001 - child process persists recovery diagnostics
        if mode == "hold":
            (scratch / "hold-error.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            return 17
        (scratch / "resume-error.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        return 19

    (scratch / "resume-result.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "tree": result.model_dump(mode="json"),
                "section_count": sum(node.kind == "section" for node in result.nodes),
                "summary_count": sum(node.summary is not None for node in result.nodes),
            }
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
