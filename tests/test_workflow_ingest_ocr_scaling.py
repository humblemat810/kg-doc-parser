"""Deterministic checkpoint-size and OCR workflow scaling tests."""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
from io import BytesIO
from pathlib import Path
from typing import cast

import pytest
from _kogwistar_test_helpers import (
    load_kogwistar_fake_backend,
)
from kogwistar.typing_interfaces import EmbeddingFunctionLike
from ocr_workflow_support import (
    fake_ocr_response as _fake_ocr_response,
)
from PIL import Image

import kg_doc_parser.workflow_ingest.ocr_pipeline as ocr_pipeline_module
import kg_doc_parser.workflow_ingest.ocr_runtime_page as ocr_runtime_page_module
import kg_doc_parser.workflow_ingest.ocr_runtime_state as ocr_runtime_state_module
from kg_doc_parser.models import (
    OCRClusterResponse,
)
from kg_doc_parser.workflow_ingest import (
    EmbeddingProviderConfig,
    ProviderEndpointConfig,
    WorkflowProviderSettings,
)
from kg_doc_parser.workflow_ingest.adapters import OCRPageJSON
from kg_doc_parser.workflow_ingest.ocr_pipeline import (
    OCRPageProcessingContext,
    OCRSourcePlan,
    _ocr_configuration_fingerprint,
)
from kg_doc_parser.workflow_ingest.ocr_runtime import (
    run_ocr_page_workflow,
)
from kg_doc_parser.workflow_ingest.ocr_runtime_contracts import (
    OCR_PAGE_STATE_VERSION as _OCR_PAGE_STATE_VERSION,
)
from kg_doc_parser.workflow_ingest.ocr_runtime_contracts import (
    OCRWorkflowState as _OCRWorkflowState,
)
from kg_doc_parser.workflow_ingest.ocr_runtime_page_state import (
    page_workflow_initial_state as _page_workflow_initial_state,
)
from kg_doc_parser.workflow_ingest.ocr_state_store import (
    OCRWorkflowStateStore,
)
from kg_doc_parser.workflow_ingest.service import (
    _TinyEmbeddingFunction,
    build_default_engines,
)

pytestmark = [pytest.mark.workflow]


_LOGGER = logging.getLogger(__name__)


def _terminate_interrupted_worker(process: subprocess.Popen[str]) -> None:
    """Use the strongest available process-stop primitive for recovery tests."""
    if os.name == "posix":
        os.kill(process.pid, signal.SIGKILL)
    else:
        process.terminate()


@pytest.mark.ci
def test_large_page_checkpoint_stays_artifact_backed_and_linearly_bounded() -> None:
    """A long document checkpoint stores references, not cumulative OCR JSON."""

    def _checkpoint_size(page_count: int) -> int:
        source_plan = OCRSourcePlan(
            pdf_source=None,
            rendered_dir=Path("rendered"),
            legacy_dir=Path("legacy"),
            page_sources=[
                (page_number, Path(f"page-{page_number}.png"))
                for page_number in range(1, page_count + 1)
            ],
            source_kind="image",
            input_fingerprint=f"checkpoint-{page_count}",
        )
        state = _page_workflow_initial_state(source_plan)
        state.update({"page_cursor": page_count})
        assert "raw_pages" not in state
        assert "page_sources" not in state
        assert "completed_pages" not in state
        assert "reused_pages" not in state
        return len(json.dumps(state, sort_keys=True, separators=(",", ":")))

    size_100 = _checkpoint_size(100)
    size_500 = _checkpoint_size(500)
    size_1000 = _checkpoint_size(1000)

    # Page references are expected to grow linearly. A cumulative OCR payload
    # would add large per-page response data and exceed this bounded ratio.
    assert size_500 < size_100 * 6
    assert size_1000 < size_100 * 12
    assert size_1000 < size_500 * 3


@pytest.mark.slow
def test_ocr_page_workflow_scaling_is_linear(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Measure the real resolver/store/checkpoint path at fixed payload size."""
    telemetry_logger = logging.getLogger("workflow.telemetry")
    pipeline_logger = logging.getLogger(ocr_pipeline_module.__name__)
    previous_telemetry_level = telemetry_logger.level
    previous_pipeline_level = pipeline_logger.level
    previous_global_disable = logging.root.manager.disable
    telemetry_logger.setLevel(logging.WARNING)
    pipeline_logger.setLevel(logging.WARNING)
    logging.disable(logging.CRITICAL)
    request.addfinalizer(lambda: logging.disable(previous_global_disable))
    request.addfinalizer(lambda: telemetry_logger.setLevel(previous_telemetry_level))
    request.addfinalizer(lambda: pipeline_logger.setLevel(previous_pipeline_level))
    monkeypatch.setenv("KG_DOC_DISABLE_PHASE1_INDEX_JOBS", "1")
    measurements: list[dict[str, int]] = []
    original_json_pages = ocr_runtime_state_module.json_pages
    original_progress_sync = ocr_runtime_page_module.sync_progress_from_state
    original_read_bytes = Path.read_bytes

    for page_count in (100, 500, 1000):
        scratch = tmp_path / str(page_count)
        source_dir = scratch / "source"
        source_dir.mkdir(parents=True)
        sources = [
            (page_number, source_dir / f"page-{page_number}.png")
            for page_number in range(1, page_count + 1)
        ]
        with Image.new("RGB", (1200, 800), "white") as source_image:
            image_buffer = BytesIO()
            source_image.save(image_buffer, format="PNG")
        source_image_payload = image_buffer.getvalue()
        for _page_number, image_path in sources:
            image_path.write_bytes(source_image_payload)
        source_plan = OCRSourcePlan(
            pdf_source=None,
            rendered_dir=scratch / "rendered",
            legacy_dir=scratch / "legacy",
            page_sources=sources,
            source_kind="image",
            input_fingerprint=f"scaling-{page_count}",
        )
        source_plan.legacy_dir.mkdir(parents=True)
        settings = WorkflowProviderSettings(
            ocr=ProviderEndpointConfig(provider="fake", model="scaling-ocr"),
            parser=ProviderEndpointConfig(provider="fake", model="scaling-parser"),
            embedding=EmbeddingProviderConfig(
                provider="fake", model="scaling-embed", dimension=1
            ),
        )
        store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
        provider_calls = 0
        accepted_writes = 0
        acceptance_lookups = 0
        source_reads = 0
        source_bytes = 0
        decoded_pages = 0
        decoded_json_bytes = 0
        progress_calls = 0
        progress_bytes = 0

        def _runner(
            image_path: Path,
            page_number: int,
            provider_settings: WorkflowProviderSettings,
        ) -> OCRClusterResponse:
            nonlocal provider_calls
            del image_path, provider_settings
            provider_calls += 1
            return _fake_ocr_response(page_number, "x")

        original_publish = store.publish_ocr_page
        original_should_skip = store.should_skip_page

        def _count_publish(_original_publish=original_publish, **kwargs):
            nonlocal accepted_writes
            accepted_writes += 1
            return _original_publish(**kwargs)

        def _count_should_skip(_original_should_skip=original_should_skip, **kwargs):
            nonlocal acceptance_lookups
            acceptance_lookups += 1
            return _original_should_skip(**kwargs)

        context = OCRPageProcessingContext(
            document_id=f"scaling-document-{page_count}",
            title="Scaling fixture",
            source_kind="image",
            total_pages=page_count,
            input_fingerprint=source_plan.input_fingerprint,
            candidate_models=["scaling-ocr"],
            provider_settings=settings,
            ocr_page_runner=_runner,
            state_store=store,
            probe=None,
            configuration_fingerprints=(
                _ocr_configuration_fingerprint(
                    settings, candidate_models=["scaling-ocr"]
                ),
            ),
        )

        def _count_json_pages(
            state: _OCRWorkflowState,
            *,
            source_plan: OCRSourcePlan | None = None,
            _current_plan: OCRSourcePlan | None = source_plan,
        ) -> list[OCRPageJSON]:
            nonlocal decoded_pages, decoded_json_bytes
            cursor = ocr_runtime_state_module.page_cursor(state)
            decoded_pages += cursor
            legacy_dir = state.get("legacy_dir")
            if isinstance(legacy_dir, str):
                decoded_json_bytes += sum(
                    (Path(legacy_dir) / f"page_{page_number}.json").stat().st_size
                    for page_number in range(1, cursor + 1)
                )
            return original_json_pages(
                state,
                source_plan=source_plan or _current_plan,
            )

        def _count_progress_sync(**kwargs):
            nonlocal progress_calls, progress_bytes
            progress_calls += 1
            result = original_progress_sync(**kwargs)
            progress_bytes += len(
                json.dumps(result.model_dump(mode="json"), separators=(",", ":"))
            )
            return result

        source_paths = {path.resolve() for _number, path in sources}

        def _count_source_reads(
            path: Path, _source_paths: set[Path] = source_paths
        ) -> bytes:
            nonlocal source_reads, source_bytes
            payload = original_read_bytes(path)
            if path.resolve() in _source_paths:
                source_reads += 1
                source_bytes += len(payload)
            return payload

        with monkeypatch.context() as scoped:
            scoped.setattr(ocr_runtime_page_module, "json_pages", _count_json_pages)
            scoped.setattr(
                ocr_runtime_page_module,
                "sync_progress_from_state",
                _count_progress_sync,
            )
            scoped.setattr(Path, "read_bytes", _count_source_reads)
            scoped.setattr(store, "publish_ocr_page", _count_publish)
            scoped.setattr(store, "should_skip_page", _count_should_skip)
            workflow_engine, conversation_engine, _knowledge_engine = (
                build_default_engines(
                    scratch / "engines",
                    backend_factory=load_kogwistar_fake_backend(),
                    embedding_function=cast(
                        EmbeddingFunctionLike, _TinyEmbeddingFunction()
                    ),
                )
            )
            # Keep real Kogwistar step/checkpoint persistence, but omit its
            # optional observer-only SQLite event mirror from this algorithmic
            # measurement so log volume does not dominate the page counters.
            workflow_engine.persist_directory = None
            pages, completed, reused, run_id = run_ocr_page_workflow(
                workflow_engine=workflow_engine,
                conversation_engine=conversation_engine,
                context=context,
                source_plan=source_plan,
                progress_path=scratch / "ocr-progress.json",
                run_id=f"scaling-run-{page_count}",
            )
            step_nodes = list(
                conversation_engine.read.get_nodes(
                    where={
                        "$and": [
                            {"entity_type": "workflow_step_exec"},
                            {"run_id": run_id},
                        ]
                    },
                    limit=None,
                )
            )
            checkpoint_nodes = list(
                conversation_engine.read.get_nodes(
                    where={
                        "$and": [
                            {"entity_type": "workflow_checkpoint"},
                            {"run_id": run_id},
                        ]
                    },
                    limit=None,
                )
            )

        checkpoint_bytes = sum(
            len(str((node.metadata or {}).get("state_json", "")))
            for node in checkpoint_nodes
        )
        update_bytes = 0
        for node in step_nodes:
            result_json = str((node.metadata or {}).get("result_json", "{}"))
            result_payload = json.loads(result_json)
            update_bytes += len(
                json.dumps(
                    result_payload.get("state_update", []),
                    separators=(",", ":"),
                )
            )
        assert len(pages) == len(completed) == page_count
        assert not reused
        assert provider_calls == page_count
        measurements.append(
            {
                "pages": page_count,
                "page_decodes": decoded_pages,
                "provider_calls": provider_calls,
                "accepted_page_writes": accepted_writes,
                "acceptance_lookups": acceptance_lookups,
                "decoded_json_bytes": decoded_json_bytes,
                "source_image_reads": source_reads,
                "source_image_bytes": source_bytes,
                "progress_writes": progress_calls,
                "progress_bytes": progress_bytes,
                "step_update_bytes": update_bytes,
                "checkpoint_bytes": checkpoint_bytes,
                "step_count": len(step_nodes),
                "checkpoint_count": len(checkpoint_nodes),
            }
        )

    print(f"OCR scaling measurements: {json.dumps(measurements, sort_keys=True)}")
    for measurement in measurements:
        page_count = measurement["pages"]
        assert measurement["page_decodes"] == page_count
        assert measurement["provider_calls"] == page_count
        assert measurement["accepted_page_writes"] == page_count
        assert measurement["progress_writes"] == 1
        assert measurement["source_image_reads"] <= page_count * 2
        assert measurement["acceptance_lookups"] <= page_count * 3
    for key in (
        "decoded_json_bytes",
        "progress_bytes",
        "step_update_bytes",
        "checkpoint_bytes",
    ):
        assert measurements[1][key] < measurements[0][key] * 6, key
        assert measurements[2][key] < measurements[0][key] * 12, key

    # Negative control: repeat the old accepted-prefix materialization using
    # the same JSON decoder over real accepted artifacts. Its exact
    # triangular operation count must fail the linear scaling budget.
    control_plan = OCRSourcePlan(
        pdf_source=None,
        rendered_dir=tmp_path / "100" / "rendered",
        legacy_dir=tmp_path / "100" / "legacy",
        page_sources=[
            (
                page_number,
                tmp_path / "100" / "source" / f"page-{page_number}.png",
            )
            for page_number in range(1, 101)
        ],
        source_kind="image",
        input_fingerprint="scaling-100",
    )
    quadratic_decodes = 0
    for cursor in range(1, 101):
        prefix_state: _OCRWorkflowState = {
            "legacy_dir": str(control_plan.legacy_dir),
            "page_cursor": cursor,
            "raw_pages": None,
            "page_sources": [
                {"page_number": page, "image_path": str(path)}
                for page, path in control_plan.page_sources
            ],
        }
        quadratic_decodes += len(original_json_pages(prefix_state))
    assert quadratic_decodes == 100 * 101 // 2
    with pytest.raises(AssertionError):
        assert quadratic_decodes <= measurements[0]["page_decodes"] * 2
    telemetry_logger.setLevel(previous_telemetry_level)
    pipeline_logger.setLevel(previous_pipeline_level)


@pytest.mark.ci
def test_ocr_scaling_artifact_reads_and_checkpoint_state_bytes_are_linear(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise old and compact state shapes at 100/500/1000 deterministic pages."""
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    page_payload_sizes: dict[int, int] = {}
    page_payloads: dict[int, str] = {}
    payload_template = _fake_ocr_response(1, "fixed page").model_dump(
        dump_format="json"
    )
    for page_number in range(1, 1001):
        payload_template["printed_page_number"] = str(page_number)
        serialized_payload = json.dumps(payload_template, separators=(",", ":"))
        page_payloads[page_number] = serialized_payload
        page_payload_sizes[page_number] = len(serialized_payload.encode("utf-8"))

    read_count = 0
    read_bytes = 0
    original_read_text = Path.read_text

    def _count_reads(path: Path, *args, **kwargs) -> str:
        nonlocal read_count, read_bytes
        if path.parent == legacy_dir:
            page_number = int(path.stem.removeprefix("page_"))
            content = page_payloads[page_number]
            read_count += 1
            read_bytes += len(content.encode("utf-8"))
            return content
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _count_reads)

    # Execute a small legacy prefix replay, then scale its exact triangular
    # operation/byte formula to the requested document sizes.
    old_plan = OCRSourcePlan(
        pdf_source=None,
        rendered_dir=tmp_path / "rendered",
        legacy_dir=legacy_dir,
        page_sources=[
            (page_number, tmp_path / "source" / f"page-{page_number}.png")
            for page_number in range(1, 21)
        ],
        source_kind="image",
        input_fingerprint="quadratic-control",
    )
    read_count = read_bytes = 0
    baseline_page_decodes = 0
    for cursor in range(1, 21):
        old_state: _OCRWorkflowState = {
            "legacy_dir": str(legacy_dir),
            "page_sources": [
                {"page_number": number, "image_path": str(path)}
                for number, path in old_plan.page_sources
            ],
            "page_cursor": cursor,
            "raw_pages": None,
        }
        baseline_page_decodes += len(ocr_runtime_state_module.json_pages(old_state))
    baseline_read_count = read_count
    baseline_read_bytes = read_bytes
    assert baseline_page_decodes == baseline_read_count == 20 * 21 // 2
    assert baseline_read_bytes == sum(
        (21 - page_number) * page_payload_sizes[page_number]
        for page_number in range(1, 21)
    )

    measurements: list[dict[str, int]] = []
    compact_state_bytes: dict[int, int] = {}
    baseline_state_bytes: dict[int, int] = {}
    baseline_prefix_bytes: dict[int, int] = {}
    for page_count in (100, 500, 1000):
        plan = OCRSourcePlan(
            pdf_source=None,
            rendered_dir=tmp_path / "rendered",
            legacy_dir=legacy_dir,
            page_sources=[
                (page_number, tmp_path / "source" / f"page-{page_number}.png")
                for page_number in range(1, page_count + 1)
            ],
            source_kind="image",
            input_fingerprint=f"linear-{page_count}",
        )
        compact_state = _page_workflow_initial_state(plan)
        compact_state.update(
            {
                "document_id": "scaling-document",
                "ocr_configuration_fingerprints": ["fixed-config"],
            }
        )
        current_read_count = current_read_bytes = 0
        read_count = read_bytes = 0
        accepted_pages = ocr_runtime_state_module.json_pages(
            {**compact_state, "page_cursor": page_count}, source_plan=plan
        )
        current_read_count, current_read_bytes = read_count, read_bytes
        assert len(accepted_pages) == page_count
        assert current_read_count == page_count
        assert current_read_bytes == sum(
            page_payload_sizes[number] for number in range(1, page_count + 1)
        )

        old_state: _OCRWorkflowState = {
            "ocr_page_state_version": _OCR_PAGE_STATE_VERSION,
            "source_fingerprint": plan.input_fingerprint,
            "page_sources": [
                {"page_number": number, "image_path": str(path)}
                for number, path in plan.page_sources
            ],
            "legacy_dir": str(legacy_dir),
            "page_cursor": 0,
            "completed_pages": [],
            "reused_pages": [],
            "document_id": "scaling-document",
            "ocr_configuration_fingerprints": ["fixed-config"],
        }
        current_state = dict(compact_state)
        current_state["last_reused_page"] = None
        current_total = 0
        compact_base_bytes = len(
            json.dumps(current_state, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        old_base_bytes = len(
            json.dumps(old_state, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        old_completed_json_bytes = 2
        old_total = 0
        for page_number in range(1, page_count + 1):
            if page_number > 1:
                old_completed_json_bytes += 1
            old_completed_json_bytes += len(str(page_number))
            old_total += (
                old_base_bytes
                - 3  # empty completed_pages plus one-digit page_cursor
                + old_completed_json_bytes
                + len(str(page_number))
            )
            current_total += compact_base_bytes - 1 + len(str(page_number))
        baseline_state_bytes[page_count] = old_total
        compact_state_bytes[page_count] = current_total
        baseline_prefix_bytes[page_count] = sum(
            (page_count - page_number + 1) * page_payload_sizes[page_number]
            for page_number in range(1, page_count + 1)
        )
        measurements.append(
            {
                "pages": page_count,
                "accepted_page_reads": current_read_count,
                "accepted_json_bytes": current_read_bytes,
                "baseline_prefix_decodes": page_count * (page_count + 1) // 2,
                "baseline_prefix_bytes": baseline_prefix_bytes[page_count],
                "baseline_checkpoint_state_bytes": old_total,
                "compact_checkpoint_state_bytes": current_total,
            }
        )

    assert compact_state_bytes[500] < compact_state_bytes[100] * 6
    assert compact_state_bytes[1000] < compact_state_bytes[100] * 12
    assert baseline_state_bytes[500] > baseline_state_bytes[100] * 10
    assert baseline_state_bytes[1000] > baseline_state_bytes[100] * 30
    assert compact_state_bytes[1000] < baseline_state_bytes[1000]
    assert read_count == 1000
    print(
        "OCR deterministic scaling measurements: "
        + json.dumps(
            {
                "baseline_20_prefix_reads": baseline_read_count,
                "baseline_20_prefix_bytes": baseline_read_bytes,
                "baseline_100_prefix_reads": 100 * 101 // 2,
                "baseline_100_prefix_bytes": baseline_prefix_bytes[100],
                "measurements": measurements,
            },
            sort_keys=True,
        )
    )
