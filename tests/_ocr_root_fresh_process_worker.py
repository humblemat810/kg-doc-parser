from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

from _kogwistar_test_helpers import build_workflow_engine_triplet
from kogwistar.engine_core.rust_meta_sqlite import build_sqlite_meta_store

from kg_doc_parser.models import OCRClusterResponse, TextCluster
from kg_doc_parser.workflow_ingest import (
    EmbeddingProviderConfig,
    OCRImagePayload,
    ProviderEndpointConfig,
    WorkflowProviderSettings,
    run_ocr_ingest_workflow,
)
from kg_doc_parser.workflow_ingest.semantics import HydratedTextPointer, SemanticNode


def _response(page_number: int) -> OCRClusterResponse:
    return OCRClusterResponse(
        OCR_text_clusters=[
            TextCluster(
                text=f"root page {page_number}",
                bb_x_min=10.0,
                bb_x_max=500.0,
                bb_y_min=10.0,
                bb_y_max=50.0,
                cluster_number=0,
            )
        ],
        non_text_objects=[],
        is_empty_page=False,
        printed_page_number=str(page_number),
        meaningful_ordering=[0],
        page_x_min=0.0,
        page_x_max=1200.0,
        page_y_min=0.0,
        page_y_max=800.0,
        estimated_rotation_degrees=0.0,
        incomplete_words_on_edge=False,
        incomplete_text=False,
        data_loss_likelihood=0.0,
        scan_quality="high",
        contains_table=False,
    )


def _semantic_tree(
    *,
    collection: object,
    parser_input_dict: dict[str, object],
    parser_source_map: dict[str, dict[str, object]],
) -> SemanticNode:
    del parser_input_dict
    root = SemanticNode(
        title=str(getattr(collection, "title", "Fresh Root")),
        node_type="DOCUMENT_ROOT",
        total_content_pointers=[],
        child_nodes=[],
        level_from_root=0,
    )
    for cluster_id, record in parser_source_map.items():
        text = record.get("text")
        if not isinstance(text, str) or not text:
            continue
        root.child_nodes.append(
            SemanticNode(
                title=f"section:{cluster_id}",
                node_type="TEXT_FLOW",
                total_content_pointers=[
                    HydratedTextPointer(
                        source_cluster_id=cluster_id,
                        start_char=0,
                        end_char=len(text) - 1,
                        verbatim_text=text,
                    )
                ],
                child_nodes=[],
                level_from_root=1,
                parent_id=root.node_id,
            )
        )
    return root


def main() -> int:
    os.environ.setdefault("KOGWISTAR_IMPL_META_STORE", "python")
    scratch = Path(sys.argv[1])
    mode = sys.argv[2]
    image_payloads = [
        OCRImagePayload(
            page_number=page,
            image_path=str(scratch / "source" / f"page-{page}.png"),
        )
        for page in (1, 2)
    ]
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fresh-root-ocr"),
        parser=ProviderEndpointConfig(provider="fake", model="fresh-root-parser"),
        embedding=EmbeddingProviderConfig(
            provider="fake", model="fresh-root-embed", dimension=1
        ),
        triage_enabled=False,
    )
    calls: list[int] = []

    def runner(
        image_path: Path,
        page_number: int,
        provider_settings: WorkflowProviderSettings,
    ) -> OCRClusterResponse:
        del image_path, provider_settings
        calls.append(page_number)
        if mode == "interrupt" and page_number == 2:
            raise RuntimeError("fresh root interruption")
        if mode == "hold" and page_number == 2:
            (scratch / "page-2-started").write_text("ready", encoding="utf-8")
            time.sleep(120.0)
        if mode.startswith("race-") and page_number == 2:
            (scratch / f"{mode}-page-2-started").write_text(
                "ready", encoding="utf-8"
            )
            time.sleep(3.0)
        if mode in {"resume", "race-a", "race-b"} and page_number == 1:
            raise AssertionError("accepted page was replayed")
        return _response(page_number)

    workflow_engine, conversation_engine, knowledge_engine = build_workflow_engine_triplet(
        scratch / "engines", "in_memory"
    )
    for name, engine in (
        ("workflow", workflow_engine),
        ("conversation", conversation_engine),
        ("knowledge", knowledge_engine),
    ):
        metadata = build_sqlite_meta_store(scratch / "engines" / name, "meta.sqlite")
        metadata.ensure_initialized()
        engine.metadata = metadata
        engine._phase1_enable_index_jobs = False
    try:
        if mode in {"resume", "race-a", "race-b"}:
            # Simulate a fresh process after the crash-recovery lease expires.
            # The production default remains long-lived; this test-only lease
            # keeps the subprocess test bounded without bypassing ownership.
            time.sleep(30.2)
        result, bundle, _ = run_ocr_ingest_workflow(
            document_id="fresh-root-document",
            title="Fresh Root",
            output_dir=scratch / "artifacts",
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            knowledge_engine=knowledge_engine,
            image_payloads=image_payloads,
            provider_settings=settings,
            ocr_runner=runner,
            deps={"parse_semantic_fn": _semantic_tree},
            run_id="ocr|fresh-root-document",
            resume_from_checkpoint=mode in {"resume", "race-a", "race-b"},
            run_ownership_lease_ms=30_000,
        )
    except Exception as error:
        if mode != "interrupt":
            error_name = (
                f"{mode}-error.txt" if mode.startswith("race-") else "resume-error.txt"
            )
            (scratch / error_name).write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            raise
        (scratch / "interrupt-result.json").write_text(
            json.dumps({"error": str(error), "calls": calls}), encoding="utf-8"
        )
        return 17

    if mode not in {"resume", "resume-missing-artifact", "race-a", "race-b"} or result.status != "succeeded" or bundle is None:
        raise AssertionError("fresh root resume did not produce one accepted result")
    result_name = (
        f"{mode}-result.json"
        if mode.startswith("race-") or mode == "resume-missing-artifact"
        else "resume-result.json"
    )
    (scratch / result_name).write_text(
        json.dumps(
            {
                "status": result.status,
                "run_id": result.run_id,
                "calls": calls,
                "has_grouping_tree": bundle.grouping_tree is not None,
            }
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
