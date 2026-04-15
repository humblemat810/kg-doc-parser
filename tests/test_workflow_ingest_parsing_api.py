from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.workflow_ingest import (
    OCRImagePayload,
    OCRWorkflowArtifacts,
    WorkflowProviderSettings,
    parse_document,
    parse_ocr_document,
    parse_page_index_document,
    parse_tree_document,
)
from src.workflow_ingest import cli as workflow_cli
from src.workflow_ingest import runners
from src.workflow_ingest import parsing as parsing_api


pytestmark = [pytest.mark.workflow]


def _scratch(name: str) -> Path:
    root = Path("tests") / ".tmp_workflow_ingest_parsing_api"
    path = root / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_parse_document_dispatches_to_requested_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def _fake_ocr_document(**kwargs):
        calls.append(("ocr", kwargs))
        return "ocr-result"

    def _fake_page_index_document(**kwargs):
        calls.append(("page_index", kwargs))
        return "page-index-result"

    def _fake_tree_document(**kwargs):
        calls.append(("tree", kwargs))
        return "tree-result"

    monkeypatch.setattr(parsing_api, "parse_ocr_document", _fake_ocr_document)
    monkeypatch.setattr(parsing_api, "parse_page_index_document", _fake_page_index_document)
    monkeypatch.setattr(parsing_api, "parse_tree_document", _fake_tree_document)

    ocr_result = parse_document(
        mode="ocr",
        document_id="doc-ocr",
        title="Doc OCR",
        output_dir=Path("tests") / ".tmp_workflow_ingest_parsing_api" / "ocr",
        image_payloads=[OCRImagePayload(page_number=1, image_path="page.png")],
    )
    page_index_result = parse_document(
        mode="page_index",
        document_id="doc-page",
        title="Doc Page",
        raw_text="Alpha",
    )
    tree_result = parse_document(
        mode="tree",
        doc_id="doc-tree",
        raw_doc_dict={"doc-tree": {"pages": []}},
    )

    assert ocr_result == "ocr-result"
    assert page_index_result == "page-index-result"
    assert tree_result == "tree-result"
    assert [mode for mode, _ in calls] == ["ocr", "page_index", "tree"]


def test_parse_ocr_document_resolves_explicit_provider_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    sentinel = object()
    output_dir = _scratch("ocr_explicit_provider")

    def _fake_prepare_ocr_workflow_input(**kwargs):
        captured["provider_settings"] = kwargs["provider_settings"]
        captured["output_dir"] = Path(kwargs["output_dir"])
        return sentinel

    monkeypatch.setattr(parsing_api, "prepare_ocr_workflow_input", _fake_prepare_ocr_workflow_input)
    monkeypatch.setenv("KG_DOC_OCR_PROVIDER", "gemini")
    monkeypatch.setenv("KG_DOC_OCR_MODEL", "gemini-2.5-flash")

    result = parse_ocr_document(
        document_id="ocr-doc",
        title="OCR Doc",
        output_dir=output_dir,
        image_payloads=[OCRImagePayload(page_number=1, image_path="page.png")],
        provider="ollama",
        model="qwen3:4b",
    )

    assert result is sentinel
    assert captured["output_dir"] == output_dir
    settings = captured["provider_settings"]
    assert isinstance(settings, WorkflowProviderSettings)
    assert settings.ocr.provider == "ollama"
    assert settings.ocr.model == "qwen3:4b"


def test_parse_ocr_document_uses_env_defaults_when_overrides_are_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    output_dir = _scratch("ocr_env_defaults")

    def _fake_prepare_ocr_workflow_input(**kwargs):
        captured["provider_settings"] = kwargs["provider_settings"]
        return object()

    monkeypatch.setattr(parsing_api, "prepare_ocr_workflow_input", _fake_prepare_ocr_workflow_input)
    monkeypatch.setenv("KG_DOC_OCR_PROVIDER", "gemini")
    monkeypatch.setenv("KG_DOC_OCR_MODEL", "gemini-2.5-flash")

    parse_ocr_document(
        document_id="ocr-doc",
        title="OCR Doc",
        output_dir=output_dir,
        image_payloads=[OCRImagePayload(page_number=1, image_path="page.png")],
    )

    settings = captured["provider_settings"]
    assert isinstance(settings, WorkflowProviderSettings)
    assert settings.ocr.provider == "gemini"
    assert settings.ocr.model == "gemini-2.5-flash"


def test_parse_page_index_document_resolves_explicit_provider_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def _fake_page_index_parse(**kwargs):
        captured["provider_settings"] = kwargs["provider_settings"]
        return sentinel

    monkeypatch.setattr(parsing_api, "_parse_page_index_document", _fake_page_index_parse)
    monkeypatch.setenv("KG_DOC_PARSER_PROVIDER", "gemini")
    monkeypatch.setenv("KG_DOC_PARSER_MODEL", "gemini-2.5-flash")

    result = parse_page_index_document(
        document_id="page-doc",
        title="Page Doc",
        raw_text="Alpha",
        provider="vertex",
        model="gemini-2.5-pro",
    )

    assert result is sentinel
    settings = captured["provider_settings"]
    assert isinstance(settings, WorkflowProviderSettings)
    assert settings.parser.provider == "vertex"
    assert settings.parser.model == "gemini-2.5-pro"


@pytest.mark.parametrize("provider", ["ollama", "gemini", "openai", "vertex"])
def test_parse_tree_document_honors_explicit_provider_and_model(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    captured: dict[str, object] = {}

    def _fake_parse_doc(*, doc_id, raw_doc_dict, parsing_mode, max_depth, model_names):
        captured["doc_id"] = doc_id
        captured["raw_doc_dict"] = raw_doc_dict
        captured["parsing_mode"] = parsing_mode
        captured["max_depth"] = max_depth
        captured["model_names"] = list(model_names)
        captured["env"] = {
            "KG_DOC_PARSER_PROVIDER": os.getenv("KG_DOC_PARSER_PROVIDER"),
            "KG_DOC_PARSER_MODEL": os.getenv("KG_DOC_PARSER_MODEL"),
        }
        return ("tree", {"source": "map"})

    monkeypatch.setattr("src.semantic_document_splitting_layerwise_edits.parse_doc", _fake_parse_doc)
    monkeypatch.setenv("KG_DOC_PARSER_PROVIDER", "gemini")
    monkeypatch.setenv("KG_DOC_PARSER_MODEL", "gemini-2.5-flash")

    result = parse_tree_document(
        doc_id="tree-doc",
        raw_doc_dict={"tree-doc": {"pages": []}},
        provider=provider,
        model="explicit-model",
    )

    assert result == ("tree", {"source": "map"})
    assert captured["doc_id"] == "tree-doc"
    assert captured["parsing_mode"] == "snippet"
    assert captured["max_depth"] == 10
    assert captured["model_names"] == ["explicit-model"]
    assert captured["env"]["KG_DOC_PARSER_PROVIDER"] == provider
    assert captured["env"]["KG_DOC_PARSER_MODEL"] == "explicit-model"


def test_parse_tree_document_uses_env_defaults_when_overrides_are_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_parse_doc(*, doc_id, raw_doc_dict, parsing_mode, max_depth, model_names):
        captured["model_names"] = list(model_names)
        captured["env"] = {
            "KG_DOC_PARSER_PROVIDER": os.getenv("KG_DOC_PARSER_PROVIDER"),
            "KG_DOC_PARSER_MODEL": os.getenv("KG_DOC_PARSER_MODEL"),
        }
        return ("tree", {"source": "map"})

    monkeypatch.setattr("src.semantic_document_splitting_layerwise_edits.parse_doc", _fake_parse_doc)
    monkeypatch.setenv("KG_DOC_PARSER_PROVIDER", "ollama")
    monkeypatch.setenv("KG_DOC_PARSER_MODEL", "qwen3:4b")

    parse_tree_document(
        doc_id="tree-doc",
        raw_doc_dict={"tree-doc": {"pages": []}},
    )

    assert captured["model_names"] == ["qwen3:4b"]
    assert captured["env"]["KG_DOC_PARSER_PROVIDER"] == "ollama"
    assert captured["env"]["KG_DOC_PARSER_MODEL"] == "qwen3:4b"


def test_cli_ocr_command_uses_parse_ocr_document(monkeypatch: pytest.MonkeyPatch) -> None:
    scratch = _scratch("cli_ocr")
    source = scratch / "sample.png"
    source.write_bytes(b"fake")
    output_dir = scratch / "ocr-out"
    captured: dict[str, object] = {}

    def _fake_parse_ocr_document(**kwargs):
        captured["parse_called"] = True
        captured["kwargs"] = kwargs
        return OCRWorkflowArtifacts(
            workflow_input=SimpleNamespace(),
            ocr_pages=[],
            legacy_dir=scratch / "legacy",
            rendered_dir=scratch / "rendered",
            state_db_path=scratch / "ocr-state.sqlite",
            progress_path=scratch / "ocr-progress.json",
            summary_path=scratch / "ocr-summary.json",
            completed_pages=[],
            reused_pages=[],
        )

    def _fake_build_default_engines(*args, **kwargs):
        captured["build_default_engines_called"] = True
        return object(), object(), object()

    def _fake_run_ingest_workflow(**kwargs):
        captured["run_ingest_workflow_called"] = True

        @dataclass
        class _Run:
            run_id: str = "run|fake"
            status: str = "succeeded"
            final_state: dict[str, object] = None  # type: ignore[assignment]

        @dataclass
        class _Bundle:
            payload: dict[str, object] = None  # type: ignore[assignment]

            def model_dump(self, *args, **kwargs):
                return self.payload or {"bundle": "ok"}

        return _Run(final_state={}), _Bundle()

    monkeypatch.setattr("src.workflow_ingest.parsing.parse_ocr_document", _fake_parse_ocr_document)
    monkeypatch.setattr("src.workflow_ingest.ocr_pipeline.run_ingest_workflow", _fake_run_ingest_workflow)
    monkeypatch.setattr(runners, "build_default_engines", _fake_build_default_engines)

    exit_code = workflow_cli.main(["ocr", str(source), "--output-dir", str(output_dir)])

    assert exit_code == 0
    assert captured["parse_called"] is True
    assert captured["build_default_engines_called"] is True
    assert captured["run_ingest_workflow_called"] is True


def test_cli_page_index_command_uses_parse_page_index_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = _scratch("cli_page_index")
    source = scratch / "page-index.txt"
    source.write_text("Alpha\n--- PAGE BREAK ---\nBeta", encoding="utf-8")
    output_dir = scratch / "page-index-out"
    captured: dict[str, object] = {}

    def _fake_parse_page_index_document(**kwargs):
        captured["parse_called"] = True
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            coverage={"overall": 1.0},
            workflow_input=SimpleNamespace(collections=[SimpleNamespace(pages=[1, 2])]),
            semantic_tree=SimpleNamespace(
                child_nodes=[
                    SimpleNamespace(child_nodes=[]),
                    SimpleNamespace(child_nodes=[]),
                ]
            ),
        )

    monkeypatch.setattr(runners, "parse_page_index_document", _fake_parse_page_index_document)

    exit_code = workflow_cli.main(["page-index", str(source), "--output-dir", str(output_dir)])

    assert exit_code == 0
    assert captured["parse_called"] is True
    assert captured["kwargs"]["document_id"] == source.stem
    assert captured["kwargs"]["title"] == source.stem
