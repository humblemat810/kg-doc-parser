from __future__ import annotations

import sys
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType

import pytest

from kg_doc_parser.workflow_ingest import (
    ProviderEndpointConfig,
    WorkflowProviderSettings,
)
from kg_doc_parser.workflow_ingest.ocr_providers import (
    list_ocr_provider_capabilities,
    normalize_ocr_provider,
    run_docling_ocr_page,
)
from kg_doc_parser.workflow_ingest.providers import build_chat_model


def test_capability_inventory_contains_all_docling_engines_and_codex_boundary() -> None:
    capabilities = list_ocr_provider_capabilities()
    engines = {item.engine for item in capabilities if item.provider == "docling"}
    assert engines == {"rapidocr", "easyocr", "tesseract", "tesserocr", "ocrmac", "nemotron-ocr"}
    codex = next(item for item in capabilities if item.provider == "codex")
    assert codex.vision is False


@pytest.mark.parametrize(
    ("raw", "engine"),
    [("docling:easyocr", "easyocr"), ("bonsai2", None), ("openai", None)],
)
def test_provider_names_are_normalized(raw: str, engine: str | None) -> None:
    provider, selected_engine = normalize_ocr_provider(raw)
    assert provider in {"docling", "bonsai", "openai"}
    assert selected_engine == engine


def test_unknown_docling_engine_fails_before_optional_import() -> None:
    with pytest.raises(ValueError, match="unsupported Docling OCR engine"):
        normalize_ocr_provider("docling", engine="missing")


def test_bonsai_reuses_openai_compatible_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeChat:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    module = ModuleType("langchain_openai")
    module.ChatOpenAI = FakeChat  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_openai", module)
    monkeypatch.setenv("BONSAI_API_KEY", "test-token")
    model = build_chat_model(
        ProviderEndpointConfig(
            provider="bonsai",
            model="Ternary-Bonsai-2-27B-PTQ1_0",
            base_url="http://127.0.0.1:8001/v1",
            api_key_env="BONSAI_API_KEY",
        )
    )
    assert model is not None
    assert captured["model"] == "Ternary-Bonsai-2-27B-PTQ1_0"
    assert captured["base_url"] == "http://127.0.0.1:8001/v1"
    assert captured["api_key"] == "test-token"


def test_docling_is_configurable_without_importing_docling() -> None:
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="docling", model="docling", ocr_engine="rapidocr")
    )
    assert settings.ocr.ocr_engine == "rapidocr"


def test_docling_missing_runtime_fails_at_selection_boundary() -> None:
    if find_spec("docling") is not None:
        pytest.skip("Docling is installed; missing-runtime branch is not applicable")
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="docling", model="docling", ocr_engine="rapidocr")
    )
    with pytest.raises(RuntimeError, match="optional docling package"):
        run_docling_ocr_page(Path("missing-page.png"), 1, settings, engine="rapidocr")
