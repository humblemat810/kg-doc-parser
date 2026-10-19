from __future__ import annotations

import json
import sys
from pathlib import Path

from kg_doc_parser.models import NonTextCluster, OCRClusterResponse, TextCluster
from kg_doc_parser.workflow_ingest import (
    EmbeddingProviderConfig,
    OCRImagePayload,
    OCRQualityPolicy,
    OCRWorkflowStateStore,
    ProviderEndpointConfig,
    WorkflowProviderSettings,
    prepare_ocr_workflow_input,
)


def _response(page_number: int, *, valid: bool) -> OCRClusterResponse:
    unresolved_item_count = 0 if valid else 1
    non_text_objects = (
        [
            NonTextCluster(
                description="figure: recovered diagram",
                bb_x_min=520.0,
                bb_x_max=900.0,
                bb_y_min=100.0,
                bb_y_max=420.0,
                cluster_number=1,
            )
        ]
        if valid
        else []
    )
    return OCRClusterResponse(
        OCR_text_clusters=[
            TextCluster(
                text="strict quality restart",
                bb_x_min=10.0,
                bb_x_max=500.0,
                bb_y_min=10.0,
                bb_y_max=50.0,
                cluster_number=0,
            )
        ],
        non_text_objects=non_text_objects,
        is_empty_page=False,
        printed_page_number=str(page_number),
        meaningful_ordering=[0, 1] if valid else [0],
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
        source_item_count=2,
        unresolved_item_count=unresolved_item_count,
    )


def main() -> int:
    scratch = Path(sys.argv[1])
    mode = sys.argv[2]
    if mode not in {"reject", "resume"}:
        raise ValueError(f"unsupported worker mode: {mode}")
    calls = 0

    def runner(
        image_path: Path,
        page_number: int,
        provider_settings: WorkflowProviderSettings,
    ) -> OCRClusterResponse:
        nonlocal calls
        del image_path, provider_settings
        calls += 1
        return _response(page_number, valid=mode == "resume")

    try:
        prepare_ocr_workflow_input(
            document_id="strict-quality-fresh-process-doc",
            title="Strict Quality Fresh Process",
            output_dir=scratch / "artifacts",
            image_payloads=[
                OCRImagePayload(
                    page_number=1,
                    image_path=str(scratch / "source" / "page-1.png"),
                )
            ],
            provider_settings=WorkflowProviderSettings(
                ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
                embedding=EmbeddingProviderConfig(
                    provider="fake", model="fake-embed", dimension=1
                ),
            ),
            ocr_runner=runner,
            quality_policy=OCRQualityPolicy(reject_unresolved_items=True),
        )
        status = "completed"
    except RuntimeError as exc:
        if mode != "reject" or "OCR preparation incomplete" not in str(exc):
            raise
        status = "rejected"

    state_store = OCRWorkflowStateStore(
        scratch / "artifacts" / "ocr-state.sqlite"
    )
    page_state = state_store.get_page_state(
        document_id="strict-quality-fresh-process-doc",
        page_number=1,
        stage="ocr",
    )
    if page_state is None:
        raise RuntimeError("fresh-process OCR state was not persisted")
    (scratch / f"{mode}-result.json").write_text(
        json.dumps(
            {
                "calls": calls,
                "status": status,
                "page_status": page_state.status,
                "artifact_path": page_state.artifact_path,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
