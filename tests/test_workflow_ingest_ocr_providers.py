from __future__ import annotations

import sys
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType
from typing import ClassVar

import pytest

from kg_doc_parser.workflow_ingest import (
    ProviderEndpointConfig,
    WorkflowProviderSettings,
)
from kg_doc_parser.workflow_ingest.ocr_providers import (
    list_ocr_provider_capabilities,
    normalize_ocr_provider,
    run_docling_ocr_page,
    run_easyocr_page,
    run_rapidocr_page,
)
from kg_doc_parser.workflow_ingest.providers import build_chat_model


def test_capability_inventory_contains_all_docling_engines_and_codex_boundary() -> None:
    capabilities = list_ocr_provider_capabilities()
    engines = {item.engine for item in capabilities if item.provider == "docling"}
    assert engines == {"rapidocr", "easyocr", "tesseract", "tesserocr", "ocrmac", "nemotron-ocr"}
    codex = next(item for item in capabilities if item.provider == "codex")
    assert codex.vision is True
    direct = {item.provider for item in capabilities if item.provider in {"rapidocr", "easyocr"}}
    assert direct == {"rapidocr", "easyocr"}
    assert all(item.coordinates for item in capabilities if item.provider in direct)


@pytest.mark.parametrize(
    ("raw", "engine"),
    [("docling:easyocr", "easyocr"), ("rapidocr", None), ("easyocr", None), ("bonsai2", None), ("openai", None)],
)
def test_provider_names_are_normalized(raw: str, engine: str | None) -> None:
    provider, selected_engine = normalize_ocr_provider(raw)
    assert provider in {"docling", "rapidocr", "easyocr", "bonsai", "openai"}
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


def test_direct_rapidocr_adapter_normalizes_boxes_without_docling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from PIL import Image

    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 80), "white").save(image_path)

    class Output:
        boxes: ClassVar[list[list[tuple[int, int]]]] = [[(10, 20), (50, 20), (50, 30), (10, 30)]]
        txts: ClassVar[list[str]] = ["hello"]

    class FakeRapidOCR:
        def __call__(self, path: str) -> Output:
            assert path == str(image_path)
            return Output()

    module = ModuleType("rapidocr")
    module.RapidOCR = FakeRapidOCR  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rapidocr", module)
    response = run_rapidocr_page(image_path, 1, ProviderEndpointConfig(provider="rapidocr"))
    assert response.OCR_text_clusters[0].text == "hello"
    assert response.OCR_text_clusters[0].bb_x_min == 10
    assert response.OCR_text_clusters[0].bb_y_max == 30


def test_direct_easyocr_adapter_normalizes_boxes_without_docling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from PIL import Image

    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 80), "white").save(image_path)

    class FakeReader:
        def __init__(self, languages: list[str], *, gpu: bool, verbose: bool) -> None:
            assert languages == ["en"]
            assert gpu is False and verbose is False

        def readtext(self, path: str, *, detail: int) -> list[object]:
            assert path == str(image_path) and detail == 1
            return [[[(10, 20), (50, 20), (50, 30), (10, 30)], "hello", 0.99]]

    module = ModuleType("easyocr")
    module.Reader = FakeReader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "easyocr", module)
    response = run_easyocr_page(image_path, 1, ProviderEndpointConfig(provider="easyocr"))
    assert response.OCR_text_clusters[0].text == "hello"
    assert response.OCR_text_clusters[0].bb_x_max == 50


def test_docling_missing_runtime_fails_at_selection_boundary() -> None:
    if find_spec("docling") is not None:
        pytest.skip("Docling is installed; missing-runtime branch is not applicable")
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="docling", model="docling", ocr_engine="rapidocr")
    )
    with pytest.raises(RuntimeError, match="optional docling package"):
        run_docling_ocr_page(Path("missing-page.png"), 1, settings, engine="rapidocr")
