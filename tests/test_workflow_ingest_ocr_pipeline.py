from __future__ import annotations

"""Workflow-first OCR ingest tests.

This file is intentionally split into two layers:

- small CI-safe tests that validate resume, legacy artifact compatibility, and
  PDF raster handoff without needing a live model
- manual OCR matrix cases that are meant to be clicked from VS Code and then
  inspected on disk

For the manual cases, inspect these artifacts under the stable case directory:

- `source/`
- `artifacts/rendered_pages/`
- `artifacts/legacy_split_pages/<document>/page_N.json`
- `artifacts/ocr-state.sqlite`
- `artifacts/ocr-progress.json`
- `artifacts/ocr-summary.json`
- `workflow-events.jsonl`

If a local OCR model returns stale or low-quality output, delete that specific
manual case directory and rerun the clicked case so the workflow regenerates
the per-page OCR artifacts from scratch.

Manual OCR matrix flow
----------------------

Image case:

    source/page_1.png + source/page_2.png
        -> OCR model call per page
        -> artifacts/rendered_pages/<document>/
        -> artifacts/legacy_split_pages/<document>/page_N.json
        -> artifacts/ocr-state.sqlite
        -> artifacts/ocr-progress.json
        -> normalize_ocr_pages(...)
        -> run_ingest_workflow(...)
        -> artifacts/ocr-summary.json

PDF case:

    source/manual_source.pdf
        -> render PDF page-by-page into artifacts/rendered_pages/<pdf-stem>/
        -> OCR model call per rendered page
        -> artifacts/legacy_split_pages/<pdf-name>/page_N.json
        -> artifacts/ocr-state.sqlite
        -> artifacts/ocr-progress.json
        -> normalize_ocr_pages(...)
        -> run_ingest_workflow(...)
        -> artifacts/ocr-summary.json

Both manual matrix variants end in the same normalized workflow-ingest path.

"""

import json
import logging
import os
import sqlite3
from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image, ImageDraw

from _kogwistar_test_helpers import build_workflow_engine_triplet, drain_phase1_indexes_until_idle
from src.models import OCRClusterResponse, TextCluster
from src.ocr import regen_doc
from src.workflow_ingest import (
    EmbeddingProviderConfig,
    OCRImagePayload,
    ProviderEndpointConfig,
    WorkflowProviderSettings,
    prepare_ocr_workflow_input,
    run_ocr_ingest_workflow,
)
from src.workflow_ingest.semantics import HydratedTextPointer, SemanticNode


pytestmark = [pytest.mark.workflow]

_LOGGER = logging.getLogger(__name__)


def _scratch(name: str) -> Path:
    root = Path("tests") / ".tmp_workflow_ingest_ocr"
    path = root / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _manual_case_dir(*, input_kind: str, provider: str, model: str) -> Path:
    safe_model = model.replace(":", "_").replace("/", "_")
    path = Path("tests") / ".tmp_workflow_ingest_ocr" / "manual_cases" / provider / safe_model / input_kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def _draw_test_image(path: Path, *, lines: list[str]) -> None:
    image = Image.new("RGB", (1200, 800), "white")
    draw = ImageDraw.Draw(image)
    y = 60
    for line in lines:
        draw.text((60, y), line, fill="black")
        y += 80
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG")


def _write_two_page_pdf(image_paths: list[Path], pdf_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    first, rest = images[0], images[1:]
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    first.save(pdf_path, "PDF", save_all=True, append_images=rest)
    for image in images:
        image.close()


def _fake_ocr_response(page_number: int, text: str) -> OCRClusterResponse:
    return OCRClusterResponse(
        OCR_text_clusters=[
            TextCluster(
                text=text,
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


def _fake_semantic_tree(*, collection, parser_input_dict, parser_source_map):
    root = SemanticNode(
        title=collection.title,
        node_type="DOCUMENT_ROOT",
        total_content_pointers=[],
        child_nodes=[],
        level_from_root=0,
    )
    for unit_id, record in parser_source_map.items():
        if not record.get("participates_in_semantic_text", True):
            continue
        text = record["text"]
        root.child_nodes.append(
            SemanticNode(
                title=f"section:{unit_id}",
                node_type="TEXT_FLOW",
                total_content_pointers=[
                    HydratedTextPointer(
                        source_cluster_id=unit_id,
                        start_char=0,
                        end_char=max(0, len(text) - 1),
                        verbatim_text=text,
                    )
                ],
                child_nodes=[],
                level_from_root=1,
                parent_id=root.node_id,
            )
        )
    return root


def _skip_if_live_ocr_unavailable(exc: Exception) -> None:
    message = str(exc).lower()
    if any(
        token in message
        for token in (
            "connect",
            "connection",
            "refused",
            "model",
            "ollama",
            "api key",
            "google_api_key",
            "poppler",
            "pdfinfo",
        )
    ):
        pytest.skip(f"live ocr unavailable: {exc}")


@pytest.mark.ci
def test_prepare_ocr_workflow_input_resumes_completed_pages() -> None:
    scratch = _scratch("ocr_resume")
    images_dir = scratch / "images"
    image_paths = [images_dir / "page_1.png", images_dir / "page_2.png"]
    _draw_test_image(image_paths[0], lines=["Resume test page one"])
    _draw_test_image(image_paths[1], lines=["Resume test page two"])
    payloads = [
        OCRImagePayload(page_number=1, image_path=str(image_paths[0])),
        OCRImagePayload(page_number=2, image_path=str(image_paths[1])),
    ]
    calls = {"count": 0}

    def _runner(image_path: Path, page_number: int, provider_settings: WorkflowProviderSettings):
        calls["count"] += 1
        return _fake_ocr_response(page_number, f"Resume page {page_number}")

    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        embedding=EmbeddingProviderConfig(provider="fake", model="fake-embed", dimension=1),
    )
    first = prepare_ocr_workflow_input(
        document_id="ocr-resume-doc",
        title="OCR Resume Doc",
        output_dir=scratch / "artifacts",
        image_payloads=payloads,
        provider_settings=settings,
        ocr_runner=_runner,
    )
    second = prepare_ocr_workflow_input(
        document_id="ocr-resume-doc",
        title="OCR Resume Doc",
        output_dir=scratch / "artifacts",
        image_payloads=payloads,
        provider_settings=settings,
        ocr_runner=_runner,
    )

    assert calls["count"] == 2
    assert first.completed_pages == [1, 2]
    assert second.completed_pages == [1, 2]
    assert second.reused_pages == [1, 2]
    assert second.state_db_path.exists()
    progress = json.loads(second.progress_path.read_text(encoding="utf-8"))
    assert sorted(progress["pages"].keys()) == ["1", "2"]


@pytest.mark.ci
def test_prepare_ocr_workflow_input_rebuilds_state_db_from_existing_artifacts() -> None:
    scratch = _scratch("ocr_rebuild")
    image_path = scratch / "images" / "page_1.png"
    _draw_test_image(image_path, lines=["Rebuild page"])
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        embedding=EmbeddingProviderConfig(provider="fake", model="fake-embed", dimension=1),
    )
    calls = {"count": 0}

    def _runner(image_path: Path, page_number: int, provider_settings: WorkflowProviderSettings):
        calls["count"] += 1
        return _fake_ocr_response(page_number, "Rebuild page")

    first = prepare_ocr_workflow_input(
        document_id="ocr-rebuild-doc",
        title="OCR Rebuild Doc",
        output_dir=scratch / "artifacts",
        image_payloads=[OCRImagePayload(page_number=1, image_path=str(image_path))],
        provider_settings=settings,
        ocr_runner=_runner,
    )
    first.state_db_path.unlink()
    second = prepare_ocr_workflow_input(
        document_id="ocr-rebuild-doc",
        title="OCR Rebuild Doc",
        output_dir=scratch / "artifacts",
        image_payloads=[OCRImagePayload(page_number=1, image_path=str(image_path))],
        provider_settings=settings,
        ocr_runner=_runner,
    )

    assert calls["count"] == 1
    assert second.state_db_path.exists()
    assert second.reused_pages == [1]


@pytest.mark.ci
def test_prepare_ocr_workflow_input_writes_legacy_compatible_artifacts() -> None:
    scratch = _scratch("ocr_legacy_compat")
    image_path = scratch / "images" / "page_1.png"
    _draw_test_image(image_path, lines=["Compatibility page"])
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        embedding=EmbeddingProviderConfig(provider="fake", model="fake-embed", dimension=1),
    )
    artifacts = prepare_ocr_workflow_input(
        document_id="ocr-compat-doc",
        title="OCR Compatibility Doc",
        output_dir=scratch / "artifacts",
        image_payloads=[OCRImagePayload(page_number=1, image_path=str(image_path))],
        provider_settings=settings,
        ocr_runner=lambda image_path, page_number, provider_settings: _fake_ocr_response(page_number, "Compatibility page"),
    )

    regenerated = regen_doc(str(artifacts.legacy_dir), use_raw=True)

    assert artifacts.legacy_dir.exists()
    assert len(regenerated) == 1
    assert regenerated[0]["OCR_text_clusters"][0]["text"] == "Compatibility page"
    assert artifacts.workflow_input.collections[0].pages[0].units[0].text == "Compatibility page"


@pytest.mark.ci
def test_prepare_ocr_workflow_input_accepts_pdf_with_injected_rasterizer() -> None:
    scratch = _scratch("ocr_pdf")
    pdf_path = scratch / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
    rendered_image = scratch / "rendered" / "page_1.png"
    _draw_test_image(rendered_image, lines=["Injected raster page"])
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        embedding=EmbeddingProviderConfig(provider="fake", model="fake-embed", dimension=1),
    )

    def _rasterizer(pdf_path: Path, rendered_dir: Path) -> list[Path]:
        rendered_dir.mkdir(parents=True, exist_ok=True)
        target = rendered_dir / "page_1.png"
        shutil_path = rendered_image
        if shutil_path.resolve() != target.resolve():
            import shutil

            shutil.copy2(shutil_path, target)
        return [target]

    artifacts = prepare_ocr_workflow_input(
        document_id="ocr-pdf-doc",
        title="OCR PDF Doc",
        output_dir=scratch / "artifacts",
        pdf_path=pdf_path,
        provider_settings=settings,
        ocr_runner=lambda image_path, page_number, provider_settings: _fake_ocr_response(page_number, "Injected raster page"),
        pdf_rasterizer=_rasterizer,
    )

    assert len(artifacts.ocr_pages) == 1
    assert artifacts.workflow_input.collections[0].pages[0].units[0].text == "Injected raster page"


@pytest.mark.ci
def test_prepare_ocr_workflow_input_retries_failed_pages_and_tracks_attempts() -> None:
    scratch = _scratch("ocr_retry")
    images_dir = scratch / "images"
    image_paths = [images_dir / "page_1.png", images_dir / "page_2.png"]
    _draw_test_image(image_paths[0], lines=["Retry test page one"])
    _draw_test_image(image_paths[1], lines=["Retry test page two"])
    payloads = [
        OCRImagePayload(page_number=1, image_path=str(image_paths[0])),
        OCRImagePayload(page_number=2, image_path=str(image_paths[1])),
    ]
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        embedding=EmbeddingProviderConfig(provider="fake", model="fake-embed", dimension=1),
    )
    calls = {"page1": 0, "page2": 0}

    def _runner(image_path: Path, page_number: int, provider_settings: WorkflowProviderSettings):
        if page_number == 1:
            calls["page1"] += 1
            if calls["page1"] == 1:
                raise RuntimeError("temporary ocr failure")
        else:
            calls["page2"] += 1
        return _fake_ocr_response(page_number, f"Retry page {page_number}")

    with pytest.raises(RuntimeError, match="OCR preparation incomplete"):
        prepare_ocr_workflow_input(
            document_id="ocr-retry-doc",
            title="OCR Retry Doc",
            output_dir=scratch / "artifacts",
            image_payloads=payloads,
            provider_settings=settings,
            ocr_runner=_runner,
        )

    second = prepare_ocr_workflow_input(
        document_id="ocr-retry-doc",
        title="OCR Retry Doc",
        output_dir=scratch / "artifacts",
        image_payloads=payloads,
        provider_settings=settings,
        ocr_runner=_runner,
    )

    with sqlite3.connect(second.state_db_path) as conn:
        row = conn.execute(
            """
            SELECT attempt_count, status
            FROM page_state
            WHERE document_id = ? AND page_number = ? AND stage = 'ocr'
            """,
            ("ocr-retry-doc", 1),
        ).fetchone()
    assert row is not None
    assert int(row[0]) == 2
    assert row[1] == "completed"
    assert second.reused_pages == [2]


@pytest.mark.ci
def test_prepare_ocr_workflow_input_exhausts_candidate_models_until_success() -> None:
    scratch = _scratch("ocr_candidates")
    image_path = scratch / "images" / "page_1.png"
    _draw_test_image(image_path, lines=["Candidate page"])
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="primary-model"),
        embedding=EmbeddingProviderConfig(provider="fake", model="fake-embed", dimension=1),
    )

    def _runner(image_path: Path, page_number: int, provider_settings: WorkflowProviderSettings):
        if provider_settings.ocr.model == "primary-model":
            raise RuntimeError("primary failed")
        return _fake_ocr_response(page_number, f"winner:{provider_settings.ocr.model}")

    artifacts = prepare_ocr_workflow_input(
        document_id="ocr-candidates-doc",
        title="OCR Candidate Doc",
        output_dir=scratch / "artifacts",
        image_payloads=[OCRImagePayload(page_number=1, image_path=str(image_path))],
        provider_settings=settings,
        ocr_runner=_runner,
        ocr_candidate_models=["primary-model", "backup-model"],
    )

    with sqlite3.connect(artifacts.state_db_path) as conn:
        rows = conn.execute(
            """
            SELECT model_name, status
            FROM model_attempts
            WHERE document_id = ? AND page_number = ? AND stage = 'ocr'
            ORDER BY attempt_index
            """,
            ("ocr-candidates-doc", 1),
        ).fetchall()
    assert [tuple(row) for row in rows] == [
        ("primary-model", "failed"),
        ("backup-model", "completed"),
    ]
    assert artifacts.workflow_input.collections[0].pages[0].units[0].text == "winner:backup-model"


# Manual examples with explicit parametrized node ids:
#
# Image payload + glm-ocr:
#   .venv\Scripts\python.exe -m pytest tests/test_workflow_ingest_ocr_pipeline.py::test_workflow_first_ocr_manual_matrix[glm-ocr-image] -q
#
# PDF input + glm-ocr:
#   .venv\Scripts\python.exe -m pytest tests/test_workflow_ingest_ocr_pipeline.py::test_workflow_first_ocr_manual_matrix[glm-ocr-pdf] -q
#
# Image payload + gemma4:
#   .venv\Scripts\python.exe -m pytest tests/test_workflow_ingest_ocr_pipeline.py::test_workflow_first_ocr_manual_matrix[gemma4-image] -q
#
# Image payload + gemini:
#   .venv\Scripts\python.exe -m pytest tests/test_workflow_ingest_ocr_pipeline.py::test_workflow_first_ocr_manual_matrix[gemini-image] -q
#
# The output directory for each manual case is stable and inspectable under:
#   tests/.tmp_workflow_ingest_ocr/manual_cases/<provider>/<model>/<input_kind>
#
# Suggested manual inspection flow:
# 1. click a single case in VS Code
# 2. watch page-level OCR progress in the test output
# 3. inspect `artifacts/ocr-summary.json` for the run summary
# 4. inspect `artifacts/ocr-state.sqlite` for authoritative page state
# 5. inspect `artifacts/ocr-progress.json` for readable mirrored resume state
# 6. inspect `artifacts/legacy_split_pages/.../page_N.json` for OCR payloads
# 7. inspect `source/` and `artifacts/rendered_pages/` if the OCR looks wrong
#
# If a local model is weak or occasionally unstable:
# - delete only that case directory
# - rerun the same clicked case
# - compare the regenerated `page_N.json` files with the previous run


@pytest.mark.ci_full
@pytest.mark.parametrize(
    "provider, model",
    [
        pytest.param("ollama", "glm-ocr:latest", id="glm-ocr"),
        pytest.param("ollama", "gemma4:e2b", id="gemma4"),
        pytest.param("gemini", "gemini-2.5-flash", id="gemini"),
    ],
)
@pytest.mark.parametrize(
    "input_kind",
    [
        pytest.param("image", id="image"),
        pytest.param("pdf", id="pdf"),
    ],
)
def test_workflow_first_ocr_manual_matrix(provider: str, model: str, input_kind: str) -> None:
    """Manual OCR workflow case meant for inspection in VS Code.

    This case is intentionally artifact-heavy. Each run writes:

    - rendered page images
    - legacy-compatible page JSON files
    - a SQLite state store
    - a progress manifest
    - a short summary file
    - the normalized workflow-ingest handoff state that downstream parsing sees

    The directory is stable per case so reruns are easy to inspect. If the OCR
    model output becomes stale or corrupted, delete the case directory and rerun.

    This is a full workflow-first OCR path:

    1. source image or PDF pages
    2. per-page OCR model invocation
    3. legacy-compatible page artifact write
    4. workflow-ingest normalization
    5. workflow ingest execution with deterministic fake semantic parsing

    The fake semantic parse is intentional here: it keeps the downstream ingest
    deterministic so manual inspection can focus on the OCR and source-map
    quality rather than mixed OCR + parser variability.
    """

    if provider == "gemini" and not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY is required for the manual Gemini OCR case")

    case_dir = _manual_case_dir(input_kind=input_kind, provider=provider, model=model)
    source_dir = case_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    image_paths = [source_dir / "page_1.png", source_dir / "page_2.png"]
    _draw_test_image(image_paths[0], lines=["OCR workflow page one", "Alpha clause"])
    _draw_test_image(image_paths[1], lines=["OCR workflow page two", "Beta clause"])

    image_payloads = [
        OCRImagePayload(page_number=1, image_path=str(image_paths[0])),
        OCRImagePayload(page_number=2, image_path=str(image_paths[1])),
    ]
    pdf_path = source_dir / "manual_source.pdf"
    _write_two_page_pdf(image_paths, pdf_path)

    workflow_engine, conversation_engine, knowledge_engine = build_workflow_engine_triplet(
        case_dir / "engines",
        "in_memory",
    )
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(
            provider=provider,
            model=model,
            base_url=os.getenv("KG_DOC_OCR_BASE_URL", "http://127.0.0.1:11434") if provider == "ollama" else None,
            api_key_env="GOOGLE_API_KEY" if provider == "gemini" else None,
        ),
        embedding=EmbeddingProviderConfig(provider="fake", model="ocr-manual-embed", dimension=1),
    )

    try:
        run, bundle, artifacts = run_ocr_ingest_workflow(
            document_id=f"ocr-manual-{provider}-{input_kind}",
            title="Workflow OCR Manual Case",
            output_dir=case_dir / "artifacts",
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            knowledge_engine=knowledge_engine,
            image_payloads=image_payloads if input_kind == "image" else None,
            pdf_path=pdf_path if input_kind == "pdf" else None,
            provider_settings=settings,
            deps={"parse_semantic_fn": _fake_semantic_tree},
        )
        drain_phase1_indexes_until_idle(workflow_engine, conversation_engine, knowledge_engine)
    except Exception as exc:  # noqa: BLE001
        _skip_if_live_ocr_unavailable(exc)
        raise

    assert run.status == "succeeded"
    assert bundle is not None
    assert artifacts.summary_path.exists()
    assert artifacts.progress_path.exists()
    assert artifacts.state_db_path.exists()
    assert artifacts.legacy_dir.exists()
    assert len(list(artifacts.legacy_dir.glob("page_*.json"))) >= 2

    _LOGGER.info("OCR manual case completed")
    _LOGGER.info("  input_kind=%s provider=%s model=%s", input_kind, provider, model)
    _LOGGER.info("  source_dir=%s", source_dir)
    _LOGGER.info("  rendered_dir=%s", artifacts.rendered_dir)
    _LOGGER.info("  legacy_dir=%s", artifacts.legacy_dir)
    _LOGGER.info("  state_db_path=%s", artifacts.state_db_path)
    _LOGGER.info("  progress_path=%s", artifacts.progress_path)
    _LOGGER.info("  summary_path=%s", artifacts.summary_path)
    _LOGGER.info("  completed_pages=%s reused_pages=%s", artifacts.completed_pages, artifacts.reused_pages)
