from __future__ import annotations

import json
import sys
from pathlib import Path

from _kogwistar_test_helpers import build_workflow_engine_triplet

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


def _semantic_tree(*, collection: object, parser_source_map: dict[str, dict[str, object]]) -> SemanticNode:
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
        if mode == "resume" and page_number == 1:
            raise AssertionError("accepted page was replayed")
        return _response(page_number)

    workflow_engine, conversation_engine, knowledge_engine = build_workflow_engine_triplet(
        scratch / "engines", "in_memory"
    )
    try:
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
            resume_from_checkpoint=mode == "resume",
        )
    except Exception as error:
        if mode != "interrupt":
            raise
        (scratch / "interrupt-result.json").write_text(
            json.dumps({"error": str(error), "calls": calls}), encoding="utf-8"
        )
        return 17

    if mode != "resume" or result.status != "succeeded" or bundle is None:
        raise AssertionError("fresh root resume did not produce one accepted result")
    (scratch / "resume-result.json").write_text(
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
