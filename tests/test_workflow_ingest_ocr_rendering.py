"""PDF render identity, cache validation, and atomic publication tests."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from ocr_workflow_support import scratch_path
from PIL import Image
from pypdf import PdfWriter

import kg_doc_parser.workflow_ingest.ocr_file_io as ocr_file_io_module
import kg_doc_parser.workflow_ingest.ocr_pdf_rendering as ocr_pdf_rendering_module
from kg_doc_parser.workflow_ingest.ocr_file_io import write_png_atomic
from kg_doc_parser.workflow_ingest.ocr_pdf_rendering import (
    _pdf_render_manifest_payload,
    _render_pdf_to_images,
    _render_pdf_with_identity,
)

pytestmark = [pytest.mark.workflow]


@pytest.mark.ci
def test_pdf_render_manifest_rejects_tampered_page_and_records_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.4\nsource\n")
    rendered = tmp_path / "rendered"
    calls: list[int] = []

    class Rasterizer:
        configuration_fingerprint = "dpi=144;rotation=0"

        def __call__(self, pdf_path: Path, rendered_dir: Path) -> list[Path]:
            del pdf_path
            calls.append(1)
            rendered_dir.mkdir(parents=True, exist_ok=True)
            path = rendered_dir / "page_1.png"
            Image.new("RGB", (11, 7), "white").save(path, "PNG")
            return [path]

    rasterizer = Rasterizer()
    first = _render_pdf_with_identity(
        pdf_path=source, rendered_dir=rendered, rasterizer=rasterizer
    )
    manifest = json.loads(
        (rendered / "render-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["rasterizer_configuration"] == "dpi=144;rotation=0"
    assert manifest["pages"][0]["dimensions"] == [11, 7]
    second = _render_pdf_with_identity(
        pdf_path=source, rendered_dir=rendered, rasterizer=rasterizer
    )
    assert second == first
    assert calls == [1]

    first[0].write_bytes(b"truncated")
    _render_pdf_with_identity(
        pdf_path=source, rendered_dir=rendered, rasterizer=rasterizer
    )
    assert calls == [1, 1]


@pytest.mark.ci
def test_pdf_render_manifest_rejects_a_missing_page_from_a_valid_pdf(
    tmp_path: Path,
) -> None:
    source = tmp_path / "two-pages.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_blank_page(width=72, height=72)
    with source.open("wb") as handle:
        writer.write(handle)

    rendered = tmp_path / "rendered"
    calls: list[int] = []

    class Rasterizer:
        def __call__(self, pdf_path: Path, rendered_dir: Path) -> list[Path]:
            del pdf_path
            calls.append(1)
            rendered_dir.mkdir(parents=True, exist_ok=True)
            paths = [rendered_dir / "page_1.png", rendered_dir / "page_2.png"]
            for path in paths:
                Image.new("RGB", (11, 7), "white").save(path, "PNG")
            return paths

    rasterizer = Rasterizer()
    _render_pdf_with_identity(
        pdf_path=source, rendered_dir=rendered, rasterizer=rasterizer
    )
    manifest_path = rendered / "render-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pages"] = manifest["pages"][:1]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _render_pdf_with_identity(
        pdf_path=source, rendered_dir=rendered, rasterizer=rasterizer
    )
    assert calls == [1, 1]


@pytest.mark.ci
def test_pdf_render_manifest_rejects_same_path_replacement(tmp_path: Path) -> None:
    source = tmp_path / "same-path.pdf"
    rendered = tmp_path / "rendered"
    calls: list[bytes] = []

    def write_pdf(title: str) -> None:
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.add_blank_page(width=72, height=72)
        writer.add_metadata({"/Title": title})
        with source.open("wb") as handle:
            writer.write(handle)

    class Rasterizer:
        def __call__(self, pdf_path: Path, rendered_dir: Path) -> list[Path]:
            source_bytes = pdf_path.read_bytes()
            calls.append(source_bytes)
            rendered_dir.mkdir(parents=True, exist_ok=True)
            paths = [rendered_dir / "page_1.png", rendered_dir / "page_2.png"]
            color = "white" if len(calls) == 1 else "black"
            for path in paths:
                Image.new("RGB", (11, 7), color).save(path, "PNG")
            return paths

    write_pdf("revision-a")
    rasterizer = Rasterizer()
    first = _render_pdf_with_identity(
        pdf_path=source, rendered_dir=rendered, rasterizer=rasterizer
    )
    write_pdf("revision-b")
    second = _render_pdf_with_identity(
        pdf_path=source, rendered_dir=rendered, rasterizer=rasterizer
    )

    assert first == second
    assert len(calls) == 2
    with Image.open(second[1]) as image:
        assert image.getpixel((0, 0)) == (0, 0, 0)


@pytest.mark.ci
def test_unidentified_pdf_rasterizer_never_reuses_render_manifest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"placeholder")
    rendered = tmp_path / "rendered"
    calls: list[int] = []

    def rasterizer(pdf_path: Path, rendered_dir: Path) -> list[Path]:
        del pdf_path
        calls.append(1)
        rendered_dir.mkdir(parents=True, exist_ok=True)
        path = rendered_dir / "page_1.png"
        Image.new("RGB", (11, 7), "white").save(path, "PNG")
        return [path]

    _render_pdf_with_identity(
        pdf_path=source, rendered_dir=rendered, rasterizer=rasterizer
    )
    _render_pdf_with_identity(
        pdf_path=source, rendered_dir=rendered, rasterizer=rasterizer
    )
    assert calls == [1, 1]


@pytest.mark.ci
def test_builtin_pdf_render_identity_includes_poppler_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"placeholder")
    first = _pdf_render_manifest_payload(
        pdf_path=source, rasterizer=_render_pdf_to_images
    )

    monkeypatch.setenv("POPPLER_PATH", str(tmp_path / "other-poppler"))
    second = _pdf_render_manifest_payload(
        pdf_path=source, rasterizer=_render_pdf_to_images
    )
    assert first["rasterizer_configuration"] != second["rasterizer_configuration"]
    assert first["render_identity"] != second["render_identity"]


@pytest.mark.ci
def test_builtin_pdf_render_identity_includes_poppler_binary_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POPPLER_PATH", "")
    monkeypatch.setattr(
        ocr_pdf_rendering_module.shutil,
        "which",
        lambda _name: "C:/tools/poppler/bin/pdfinfo.exe",
    )
    versions = ["pdfinfo version 24.02.0", "pdfinfo version 25.01.0"]

    def probe(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["pdfinfo", "-v"], returncode=0, stdout="", stderr=versions[0]
        )

    monkeypatch.setattr(ocr_pdf_rendering_module.subprocess, "run", probe)
    first = ocr_pdf_rendering_module._builtin_pdf_rasterizer_configuration()
    versions[0] = versions[1]
    second = ocr_pdf_rendering_module._builtin_pdf_rasterizer_configuration()
    assert first != second


@pytest.mark.ci
def test_builtin_pdf_render_does_not_reuse_cache_when_poppler_version_is_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "unknown-poppler.pdf"
    source.write_bytes(b"%PDF-1.4\nunknown-poppler\n")
    rendered = tmp_path / "rendered"
    calls = 0

    def render(_pdf_path: Path, rendered_dir: Path) -> list[Path]:
        nonlocal calls
        calls += 1
        rendered_dir.mkdir(parents=True, exist_ok=True)
        page = rendered_dir / "page_1.png"
        Image.new("RGB", (12, 8), "white").save(page, "PNG")
        return [page]

    monkeypatch.setattr(ocr_pdf_rendering_module, "_render_pdf_to_images", render)
    monkeypatch.setattr(
        ocr_pdf_rendering_module,
        "_builtin_pdf_rasterizer_configuration",
        lambda: json.dumps({"poppler_identity": "C:/tools/pdfinfo.exe|unavailable"}),
    )
    for _ in range(2):
        _render_pdf_with_identity(
            pdf_path=source,
            rendered_dir=rendered,
            rasterizer=ocr_pdf_rendering_module._render_pdf_to_images,
        )
    assert calls == 2


@pytest.mark.ci
def test_png_publication_failure_keeps_previous_artifact_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "page-1.png"
    target.write_bytes(b"previous")
    image = Image.new("RGB", (4, 4), "white")

    def fail_save(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("injected rasterizer failure")

    monkeypatch.setattr(image, "save", fail_save)
    with pytest.raises(OSError, match="injected rasterizer failure"):
        write_png_atomic(image, target)
    assert target.read_bytes() == b"previous"
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


@pytest.mark.ci
def test_atomic_json_publication_uses_distinct_temporary_files_for_concurrent_writers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = scratch_path("ocr_atomic_publication_race")
    target = scratch / "page.json"
    sources: list[str] = []
    replace_lock = threading.Lock()
    original_replace = os.replace

    def replace(source: str, destination: str, /) -> None:
        with replace_lock:
            sources.append(source)
            original_replace(source, destination)

    monkeypatch.setattr(ocr_file_io_module.os, "replace", replace)

    def publish(value: str) -> None:
        ocr_file_io_module.write_json_atomic(
            target, {"value": value, "padding": "x" * 100_000}
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(publish, value) for value in ("a", "b")]
        for future in futures:
            future.result()

    assert len(sources) == 2
    assert len(set(sources)) == 2
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["value"] in {"a", "b"}
    assert not list(target.parent.glob(f".{target.name}.*.tmp"))
