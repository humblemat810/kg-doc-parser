"""PDF rasterization and content-addressed rendered-page cache."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from importlib import metadata
from pathlib import Path
from types import FunctionType
from typing import TypedDict

from PIL import Image
from pypdf import PdfReader
from pypdf.errors import PdfReadError, PdfStreamError

from .ocr_file_io import image_file_is_valid, write_json_atomic, write_png_atomic
from .ocr_inputs import sha256_file
from .ocr_pipeline_contracts import PDFRasterizer, PDFRasterizerWithIdentity

_PDF_RENDER_MANIFEST_VERSION = 1
_PDF_RENDERER_VERSION = "pdf2image-pillow-v1"


class _PDFRenderPageEntry(TypedDict):
    page_number: int
    filename: str
    sha256: str
    dimensions: list[int]


class _PDFRenderManifest(TypedDict, total=False):
    manifest_version: int
    pdf_sha256: str
    renderer_version: str
    rasterizer: str
    rasterizer_configuration: str
    pdf_page_count: int | None
    render_identity: str
    pages: list[_PDFRenderPageEntry]


def _builtin_pdf_rasterizer_configuration() -> str:
    """Describe every process-level option used by the built-in rasterizer."""
    try:
        pdf2image_version = metadata.version("pdf2image")
    except metadata.PackageNotFoundError:
        pdf2image_version = "unknown"
    poppler_path = os.environ.get("POPPLER_PATH", "")
    executable_name = "pdfinfo.exe" if os.name == "nt" else "pdfinfo"
    executable = (
        str(Path(poppler_path) / executable_name)
        if poppler_path
        else shutil.which(executable_name)
    )
    poppler_identity = "unavailable"
    if executable is not None:
        try:
            version_probe = subprocess.run(
                [executable, "-v"],
                capture_output=True,
                check=False,
                text=True,
                timeout=3,
            )
            output = f"{version_probe.stdout}\n{version_probe.stderr}"
            version_line = (
                next((line.strip() for line in output.splitlines() if line.strip()), "")
                if version_probe.returncode == 0
                else ""
            )
            if not version_line:
                version_line = "unavailable"
            poppler_identity = f"{Path(executable).resolve()}|{version_line}"
        except (OSError, subprocess.TimeoutExpired):
            poppler_identity = f"{Path(executable).resolve()}|unavailable"
    return json.dumps(
        {
            "dpi": 200,
            "fmt": "png",
            "rotation": 0,
            "pdf2image": pdf2image_version,
            "poppler_path": poppler_path,
            "poppler_identity": poppler_identity,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _pdf_render_manifest_payload(
    *,
    pdf_path: Path,
    rasterizer: PDFRasterizer,
) -> _PDFRenderManifest:
    """Describe the exact PDF/rendering inputs behind cached page images."""
    rasterizer_name = type(rasterizer).__qualname__
    rasterizer_module = type(rasterizer).__module__
    if isinstance(rasterizer, FunctionType):
        rasterizer_name = rasterizer.__qualname__
        rasterizer_module = rasterizer.__module__
    if rasterizer is _render_pdf_to_images:
        configuration_fingerprint = _builtin_pdf_rasterizer_configuration()
    elif isinstance(rasterizer, PDFRasterizerWithIdentity):
        configuration_fingerprint = rasterizer.configuration_fingerprint
    else:
        configuration_fingerprint = "default"
    if not isinstance(configuration_fingerprint, str) or not configuration_fingerprint:
        configuration_fingerprint = "default"
    try:
        page_count: int | None = len(PdfReader(str(pdf_path)).pages)
    except (OSError, PdfReadError, PdfStreamError, ValueError):
        # Injected/fake rasterizers may intentionally use placeholder PDF bytes.
        page_count = None
    config: _PDFRenderManifest = {
        "manifest_version": _PDF_RENDER_MANIFEST_VERSION,
        "pdf_sha256": sha256_file(pdf_path),
        "renderer_version": _PDF_RENDERER_VERSION,
        "rasterizer": f"{rasterizer_module}.{rasterizer_name}",
        "rasterizer_configuration": configuration_fingerprint,
        "pdf_page_count": page_count,
    }
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    config["render_identity"] = hashlib.sha256(encoded).hexdigest()
    return config


def _rasterizer_has_stable_identity(rasterizer: PDFRasterizer) -> bool:
    """Return whether rendered pages may be safely reused across runs."""
    stable = rasterizer is _render_pdf_to_images or isinstance(
        rasterizer, PDFRasterizerWithIdentity
    )
    if not stable:
        return False
    if rasterizer is _render_pdf_to_images:
        try:
            configuration = json.loads(_builtin_pdf_rasterizer_configuration())
        except (TypeError, ValueError):
            return False
        if not isinstance(configuration, dict):
            return False
        poppler_identity = configuration.get("poppler_identity")
        pdf2image_identity = configuration.get("pdf2image")
        if (
            not isinstance(poppler_identity, str)
            or "unavailable" in poppler_identity.casefold()
            or "unknown" in poppler_identity.casefold()
            or not isinstance(pdf2image_identity, str)
            or "unknown" in pdf2image_identity.casefold()
        ):
            return False
    return True


def _decode_pdf_render_manifest(value: object) -> _PDFRenderManifest | None:
    """Narrow a persisted render manifest before using cache identity fields."""
    if not isinstance(value, dict):
        return None
    required_strings = (
        "pdf_sha256",
        "renderer_version",
        "rasterizer",
        "rasterizer_configuration",
        "render_identity",
    )
    if any(not isinstance(value.get(key), str) for key in required_strings):
        return None
    manifest_version = value.get("manifest_version")
    page_count = value.get("pdf_page_count")
    if not isinstance(manifest_version, int) or isinstance(manifest_version, bool):
        return None
    if page_count is not None and (
        not isinstance(page_count, int) or isinstance(page_count, bool)
    ):
        return None
    raw_pages = value.get("pages")
    if raw_pages is not None and not isinstance(raw_pages, list):
        return None
    pages: list[_PDFRenderPageEntry] = []
    for raw_page in raw_pages or []:
        if not isinstance(raw_page, dict):
            return None
        page_number = raw_page.get("page_number")
        filename = raw_page.get("filename")
        sha256 = raw_page.get("sha256")
        dimensions = raw_page.get("dimensions")
        if (
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or not isinstance(filename, str)
            or not isinstance(sha256, str)
            or not isinstance(dimensions, list)
            or any(
                not isinstance(dimension, int) or isinstance(dimension, bool)
                for dimension in dimensions
            )
        ):
            return None
        pages.append(
            {
                "page_number": page_number,
                "filename": filename,
                "sha256": sha256,
                "dimensions": dimensions,
            }
        )
    pdf_sha256 = value.get("pdf_sha256")
    renderer_version = value.get("renderer_version")
    rasterizer = value.get("rasterizer")
    rasterizer_configuration = value.get("rasterizer_configuration")
    render_identity = value.get("render_identity")
    if not isinstance(pdf_sha256, str):
        return None
    if not isinstance(renderer_version, str):
        return None
    if not isinstance(rasterizer, str):
        return None
    if not isinstance(rasterizer_configuration, str):
        return None
    if not isinstance(render_identity, str):
        return None
    manifest: _PDFRenderManifest = {
        "manifest_version": manifest_version,
        "pdf_sha256": pdf_sha256,
        "renderer_version": renderer_version,
        "rasterizer": rasterizer,
        "rasterizer_configuration": rasterizer_configuration,
        "pdf_page_count": page_count,
        "render_identity": render_identity,
    }
    if raw_pages is not None:
        manifest["pages"] = pages
    return manifest


def _render_pdf_with_identity(
    *,
    pdf_path: Path,
    rendered_dir: Path,
    rasterizer: PDFRasterizer,
) -> list[Path]:
    """Reuse rendered pages only when the source and renderer identity match."""
    rendered_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = rendered_dir / "render-manifest.json"
    expected = _pdf_render_manifest_payload(pdf_path=pdf_path, rasterizer=rasterizer)
    existing: _PDFRenderManifest | None = None
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing = _decode_pdf_render_manifest(payload)
        except (OSError, ValueError):
            existing = None
    paths: list[Path] = []
    if _rasterizer_has_stable_identity(rasterizer) and existing is not None and all(
        existing.get(key) == value for key, value in expected.items()
    ):
        entries = existing.get("pages")
        if entries is not None:
            candidate_paths: list[Path] = []
            valid = True
            for entry in entries:
                filename = entry["filename"]
                if Path(filename).name != filename:
                    valid = False
                    break
                candidate = rendered_dir / filename
                if (
                    candidate.is_symlink()
                    or not image_file_is_valid(candidate)
                    or entry["sha256"] != sha256_file(candidate)
                ):
                    valid = False
                    break
                try:
                    with Image.open(candidate) as image:
                        dimensions = [image.width, image.height]
                except (OSError, SyntaxError, ValueError):
                    valid = False
                    break
                if entry["dimensions"] != dimensions:
                    valid = False
                    break
                candidate_paths.append(candidate)
            expected_count = expected.get("pdf_page_count")
            page_numbers = [entry["page_number"] for entry in entries]
            valid = valid and (
                expected_count is None
                or (
                    len(entries) == expected_count
                    and page_numbers == list(range(1, expected_count + 1))
                )
            )
            if valid and candidate_paths:
                paths = candidate_paths
    if not paths:
        for candidate in rendered_dir.glob("page_*"):
            if candidate.suffix.lower() != ".json":
                candidate.unlink(missing_ok=True)
        paths = [Path(path) for path in rasterizer(pdf_path, rendered_dir)]
        if not paths:
            raise RuntimeError("pdf rasterizer returned no complete page set")
        for path in paths:
            if path.is_symlink() or not path.is_file() or not image_file_is_valid(path):
                raise RuntimeError(f"pdf rasterizer returned an invalid page image: {path}")
        page_entries: list[_PDFRenderPageEntry] = []
        for page_number, path in enumerate(paths, start=1):
            with Image.open(path) as image:
                dimensions = [image.width, image.height]
            page_entries.append(
                {
                    "page_number": page_number,
                    "filename": path.name,
                    "sha256": sha256_file(path),
                    "dimensions": dimensions,
                }
            )
        complete_manifest: _PDFRenderManifest = {**expected, "pages": page_entries}
        write_json_atomic(manifest_path, complete_manifest)
    return paths


def _render_pdf_to_images(pdf_path: Path, rendered_dir: Path) -> list[Path]:
    """Rasterize a PDF page-by-page into atomically published PNG files."""
    from pdf2image import convert_from_path

    rendered_dir.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(pdf_path))
    output_paths: list[Path] = []
    for page_number in range(1, len(reader.pages) + 1):
        output_path = rendered_dir / f"page_{page_number}.png"
        if output_path.is_symlink():
            raise ValueError(f"rendered PDF page is a symlink: {output_path.name}")
        if not output_path.is_file() or not image_file_is_valid(output_path):
            images = convert_from_path(
                str(pdf_path),
                first_page=page_number,
                last_page=page_number,
                fmt="png",
            )
            if not images:
                raise RuntimeError(f"pdf rasterizer returned no image for page {page_number}")
            write_png_atomic(images[0], output_path)
        output_paths.append(output_path)
    return output_paths
