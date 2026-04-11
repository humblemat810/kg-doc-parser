from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest

from src.workflow_ingest import OCRWorkflowArtifacts
from src.workflow_ingest import cli as workflow_cli
from src.workflow_ingest import runners


pytestmark = [pytest.mark.workflow]


@dataclass
class _FakeRun:
    run_id: str = "run|fake"
    status: str = "succeeded"


@dataclass
class _FakeBundle:
    payload: dict[str, object] | None = None

    def model_dump(self, mode: str = "json"):
        return self.payload or {"bundle": "ok"}


def _scratch(name: str) -> Path:
    root = Path("tests") / ".tmp_workflow_ingest_cli"
    path = root / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fake_ocr_artifacts(base_dir: Path) -> OCRWorkflowArtifacts:
    return OCRWorkflowArtifacts(
        workflow_input=object(),
        ocr_pages=[],
        legacy_dir=base_dir / "legacy",
        rendered_dir=base_dir / "rendered",
        state_db_path=base_dir / "ocr-state.sqlite",
        progress_path=base_dir / "ocr-progress.json",
        summary_path=base_dir / "ocr-summary.json",
        completed_pages=[1],
        reused_pages=[],
    )


def test_ocr_source_workflow_emits_probe_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    scratch = _scratch("ocr_source")
    source = scratch / "sample.png"
    source.write_bytes(b"fake image")
    output_dir = scratch / "out"

    def _fake_run_ocr_ingest_workflow(**kwargs):
        return _FakeRun(), _FakeBundle(), _fake_ocr_artifacts(output_dir)

    monkeypatch.setattr(runners, "run_ocr_ingest_workflow", _fake_run_ocr_ingest_workflow)

    result = runners.run_ocr_source_workflow(source, output_dir=output_dir)

    assert result.status == "succeeded"
    assert result.summary_path == output_dir / "ocr-summary.json"
    events = [json.loads(line) for line in (output_dir / "workflow-events.jsonl").read_text(encoding="utf-8").splitlines()]
    kinds = [event["kind"] for event in events]
    assert "workflow.file_started" in kinds
    assert "workflow.file_finished" in kinds


def test_ocr_batch_workflow_emits_one_output_root_per_file(monkeypatch: pytest.MonkeyPatch) -> None:
    scratch = _scratch("ocr_batch")
    source_dir = scratch / "inputs"
    source_dir.mkdir()
    first = source_dir / "a.png"
    second = source_dir / "b.png"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    output_dir = scratch / "batch-out"
    seen: list[Path] = []

    def _fake_run_ocr_ingest_workflow(**kwargs):
        seen.append(Path(kwargs["output_dir"]))
        return _FakeRun(), _FakeBundle(), _fake_ocr_artifacts(Path(kwargs["output_dir"]))

    monkeypatch.setattr(runners, "run_ocr_ingest_workflow", _fake_run_ocr_ingest_workflow)

    results = runners.run_ocr_batch_workflow([source_dir], output_dir=output_dir)

    assert [path.name for path in seen] == ["a", "b"]
    assert len(results) == 2
    events = [json.loads(line) for line in (output_dir / "workflow-events.jsonl").read_text(encoding="utf-8").splitlines()]
    kinds = [event["kind"] for event in events]
    assert "workflow.batch_started" in kinds
    assert "workflow.batch_finished" in kinds
    assert "workflow.file_started" in kinds
    assert "workflow.file_finished" in kinds


def test_cli_page_index_delegates_to_runner(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    scratch = _scratch("page_index_cli")
    source = scratch / "page-index.txt"
    source.write_text("Alpha\n--- PAGE BREAK ---\nBeta", encoding="utf-8")
    output_dir = scratch / "cli-out"
    called: dict[str, object] = {}

    def _fake_run_page_index_source_workflow(source_path, **kwargs):
        called["source_path"] = Path(source_path)
        called["kwargs"] = kwargs

        @dataclass
        class _Result:
            kind: str = "page_index"
            input_path: Path = Path(source_path)
            output_dir: Path = Path(kwargs["output_dir"])
            status: str = "succeeded"
            probe_path: Path = Path(kwargs["output_dir"]) / "workflow-events.jsonl"
            summary_path: Path = Path(kwargs["output_dir"]) / "page-index-summary.json"
            extra: dict[str, object] = None  # type: ignore[assignment]

        return _Result(extra={"kind": "page_index"})

    monkeypatch.setattr(workflow_cli, "run_page_index_source_workflow", _fake_run_page_index_source_workflow)

    exit_code = workflow_cli.main([
        "page-index",
        str(source),
        "--output-dir",
        str(output_dir),
    ])

    assert exit_code == 0
    assert called["source_path"] == source
    assert Path(called["kwargs"]["output_dir"]) == output_dir
    captured = capsys.readouterr().out
    assert "\"kind\": \"page_index\"" in captured


def test_cli_ocr_smoke_assets_writes_expected_files() -> None:
    scratch = _scratch("smoke_assets")
    output_dir = scratch / "assets"

    exit_code = workflow_cli.main([
        "ocr-smoke-assets",
        "--output-dir",
        str(output_dir),
    ])

    assert exit_code == 0
    assert (output_dir / "ocr_smoke_page_1.png").exists()
    assert (output_dir / "ocr_smoke_page_2.png").exists()
    assert (output_dir / "ocr_smoke_document.pdf").exists()
