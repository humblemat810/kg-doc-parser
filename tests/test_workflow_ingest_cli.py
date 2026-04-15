from __future__ import annotations

import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path
from uuid import uuid4

import pytest

from kg_doc_parser.workflow_ingest import OCRWorkflowArtifacts, ProviderEndpointConfig, WorkflowProviderSettings
from kg_doc_parser.workflow_ingest import cli as workflow_cli
from kg_doc_parser.workflow_ingest import runners


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


def test_build_legacy_parse_semantic_fn_uses_requested_model_and_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_default_parse_semantic_fn(*, collection, parser_input_dict, parser_source_map, model_names=None):
        captured["collection"] = collection
        captured["parser_input_dict"] = parser_input_dict
        captured["parser_source_map"] = parser_source_map
        captured["model_names"] = list(model_names) if model_names is not None else None
        captured["env"] = {
            "KG_DOC_PARSER_PROVIDER": os.getenv("KG_DOC_PARSER_PROVIDER"),
            "KG_DOC_PARSER_MODEL": os.getenv("KG_DOC_PARSER_MODEL"),
        }
        return {"ok": True}

    monkeypatch.setattr(runners, "default_parse_semantic_fn", _fake_default_parse_semantic_fn)
    settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(
            provider="ollama",
            model="gemma4:e2b",
            base_url="http://127.0.0.1:11434",
        )
    )
    parse_fn = runners.build_legacy_parse_semantic_fn(provider_settings=settings, model_names=["gemma4:e2b"])
    result = parse_fn(
        collection=SimpleNamespace(collection_id="doc", title="Doc"),
        parser_input_dict={"document_filename": "Doc", "pages": []},
        parser_source_map={"doc|p1_t0": {"text": "Alpha", "participates_in_semantic_text": True}},
    )

    assert result == {"ok": True}
    assert captured["model_names"] == ["gemma4:e2b"]
    assert captured["env"]["KG_DOC_PARSER_PROVIDER"] == "ollama"
    assert captured["env"]["KG_DOC_PARSER_MODEL"] == "gemma4:e2b"


def test_cli_ocr_parser_override_builds_legacy_parse_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    scratch = _scratch("ocr_parser_override")
    source = scratch / "sample.png"
    source.write_bytes(b"fake image")
    output_dir = scratch / "out"
    captured: dict[str, object] = {}

    def _fake_build_legacy_parse_semantic_fn(*, provider_settings, model_names=None):
        captured["provider_settings"] = provider_settings
        captured["model_names"] = list(model_names) if model_names is not None else None

        def _parse_semantic_fn(**kwargs):
            captured["parse_semantic_fn_called"] = True
            return {"ok": True}

        return _parse_semantic_fn

    def _fake_run_ocr_source_workflow(source_path, **kwargs):
        captured["source_path"] = Path(source_path)
        captured["deps"] = kwargs.get("deps")

        @dataclass
        class _Result:
            kind: str = "ocr"
            input_path: Path = Path(source_path)
            output_dir: Path = Path(kwargs["output_dir"])
            status: str = "succeeded"
            probe_path: Path = Path(kwargs["output_dir"]) / "workflow-events.jsonl"
            summary_path: Path = Path(kwargs["output_dir"]) / "ocr-summary.json"
            extra: dict[str, object] = None  # type: ignore[assignment]

        return _Result(extra={"kind": "ocr"})

    monkeypatch.setattr(workflow_cli, "build_legacy_parse_semantic_fn", _fake_build_legacy_parse_semantic_fn)
    monkeypatch.setattr(workflow_cli, "run_ocr_source_workflow", _fake_run_ocr_source_workflow)

    exit_code = workflow_cli.main(
        [
            "ocr",
            str(source),
            "--output-dir",
            str(output_dir),
            "--parser-provider",
            "ollama",
            "--parser-model",
            "gemma4:e2b",
            "--parser-base-url",
            "http://127.0.0.1:11434",
        ]
    )

    assert exit_code == 0
    assert captured["source_path"] == source
    assert captured["model_names"] == ["gemma4:e2b"]
    assert captured["provider_settings"].parser.provider == "ollama"
    assert captured["provider_settings"].parser.model == "gemma4:e2b"
    assert captured["deps"]["parse_semantic_fn"] is not None


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
