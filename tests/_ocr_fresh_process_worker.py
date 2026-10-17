from __future__ import annotations

import json
import sys
from pathlib import Path

from _kogwistar_test_helpers import build_workflow_engine_triplet

from kg_doc_parser.models import OCRClusterResponse, TextCluster
from kg_doc_parser.workflow_ingest import (
    EmbeddingProviderConfig,
    ProviderEndpointConfig,
    WorkflowProviderSettings,
)
from kg_doc_parser.workflow_ingest.ocr_pipeline import (
    OCRPageProcessingContext,
    OCRSourcePlan,
    OCRWorkflowStateStore,
    _ocr_configuration_fingerprint,
)
from kg_doc_parser.workflow_ingest.ocr_runtime import run_ocr_page_workflow


def _response(page_number: int) -> OCRClusterResponse:
    return OCRClusterResponse(
        OCR_text_clusters=[
            TextCluster(
                text=f"fresh page {page_number}",
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


def main() -> int:
    scratch = Path(sys.argv[1])
    mode = sys.argv[2]
    source_plan = OCRSourcePlan(
        pdf_source=None,
        rendered_dir=scratch / "rendered",
        legacy_dir=scratch / "legacy",
        page_sources=[
            (1, scratch / "source" / "page-1.png"),
            (2, scratch / "source" / "page-2.png"),
        ],
        source_kind="image",
        input_fingerprint="fresh-process-source",
    )
    source_plan.rendered_dir.mkdir(parents=True, exist_ok=True)
    source_plan.legacy_dir.mkdir(parents=True, exist_ok=True)
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fresh-process-ocr"),
        embedding=EmbeddingProviderConfig(
            provider="fake", model="fresh-process-embed", dimension=1
        ),
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
            raise RuntimeError("fresh-process interruption")
        if mode == "resume" and page_number == 1:
            raise AssertionError("accepted page was replayed")
        return _response(page_number)

    workflow_engine, conversation_engine, _ = build_workflow_engine_triplet(
        scratch / "engines", "in_memory"
    )
    state_store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    context = OCRPageProcessingContext(
        document_id="fresh-process-document",
        title="Fresh Process",
        source_kind="image",
        total_pages=2,
        input_fingerprint=source_plan.input_fingerprint,
        candidate_models=["fresh-process-ocr"],
        provider_settings=settings,
        ocr_page_runner=runner,
        state_store=state_store,
        probe=None,
        configuration_fingerprints=(
            _ocr_configuration_fingerprint(
                settings, candidate_models=["fresh-process-ocr"]
            ),
        ),
    )
    raw_pages, completed, reused, run_id = run_ocr_page_workflow(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        context=context,
        source_plan=source_plan,
        progress_path=scratch / "ocr-progress.json",
        run_id="ocr|fresh-process-document",
        # The test deliberately reopens the durable OCR state without asking
        # an in-memory graph backend to restore a checkpoint it cannot retain.
        resume_from_checkpoint=False,
    )
    (scratch / f"{mode}-result.json").write_text(
        json.dumps(
            {
                "pages": len(raw_pages),
                "completed": completed,
                "reused": reused,
                "run_id": run_id,
                "calls": calls,
            }
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
