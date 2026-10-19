"""Page-source discovery, materialization, and content identity helpers."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path

from .ocr_pipeline_contracts import OCRImagePayload


def scan_rendered_pages(rendered_dir: Path) -> list[tuple[int, Path]]:
    """Find numbered page files in a rendered-page directory."""
    pairs: list[tuple[int, Path]] = []
    if not rendered_dir.exists():
        return pairs
    for path in sorted(rendered_dir.glob("page_*")):
        suffix = path.stem.rsplit("_", 1)
        if len(suffix) == 2 and suffix[1].isdigit():
            pairs.append((int(suffix[1]), path))
    return pairs


def scan_legacy_json_pages(legacy_dir: Path) -> list[tuple[int, Path]]:
    """Find numbered legacy OCR JSON files."""
    pairs: list[tuple[int, Path]] = []
    if not legacy_dir.exists():
        return pairs
    for path in sorted(legacy_dir.glob("page_*.json")):
        suffix = path.stem.rsplit("_", 1)
        if len(suffix) == 2 and suffix[1].isdigit():
            pairs.append((int(suffix[1]), path))
    return pairs


def sha256_file(path: Path) -> str:
    """Hash a file incrementally so large page images stay memory-bounded."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_matching_render_hash(
    rendered_dir: Path,
    page_number: int,
    *,
    hash_file: Callable[[Path], str],
) -> str | None:
    """Return the hash only from the current revision's authoritative render."""
    for candidate in rendered_dir.glob(f"page_{page_number}.*"):
        if candidate.suffix.lower() != ".json" and candidate.exists():
            return hash_file(candidate)
    return None


def image_to_data_url(path: Path) -> str:
    """Encode an image file as a MIME-qualified data URL."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    suffix = path.suffix.lower()
    mime = (
        "image/jpeg"
        if suffix in {".jpg", ".jpeg"}
        else "image/webp"
        if suffix == ".webp"
        else "image/png"
    )
    return f"data:{mime};base64,{encoded}"


def materialize_image_payloads(
    image_payloads: Sequence[OCRImagePayload],
    rendered_dir: Path,
    *,
    write_bytes: Callable[[Path, bytes], None],
) -> list[tuple[int, Path]]:
    """Resolve image payloads to safe, ordered files under the render directory."""
    if rendered_dir.is_symlink():
        raise ValueError("OCR rendered directory must not be a symlink")
    rendered_dir.mkdir(parents=True, exist_ok=True)
    resolved: list[tuple[int, Path]] = []
    seen_page_numbers: set[int] = set()
    supplied_page_numbers: list[int] = []
    for index, payload in enumerate(image_payloads, start=1):
        page_number = index if payload.page_number is None else int(payload.page_number)
        if page_number < 1:
            raise ValueError("OCR page numbers must be positive")
        if page_number in seen_page_numbers:
            raise ValueError(f"duplicate OCR page number: {page_number}")
        seen_page_numbers.add(page_number)
        supplied_page_numbers.append(page_number)
        suffix = ".png"
        if payload.image_path:
            source = Path(payload.image_path)
            suffix = source.suffix or ".png"
            destination = rendered_dir / f"page_{page_number}{suffix}"
            if destination.is_symlink():
                raise ValueError(f"OCR rendered page is a symlink: {destination.name}")
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            else:
                destination = source
        elif payload.image_bytes_b64:
            raw = base64.b64decode(payload.image_bytes_b64)
            requested_suffix = Path(payload.filename or "").suffix.lower()
            if requested_suffix in {
                ".bmp",
                ".jpeg",
                ".jpg",
                ".png",
                ".tif",
                ".tiff",
                ".webp",
            }:
                suffix = requested_suffix
            destination = rendered_dir / f"page_{page_number}{suffix}"
            if destination.is_symlink():
                raise ValueError(f"OCR rendered page is a symlink: {destination.name}")
            write_bytes(destination, raw)
        else:
            raise ValueError("OCRImagePayload requires image_path or image_bytes_b64")
        resolved.append((page_number, destination))
    if supplied_page_numbers != sorted(supplied_page_numbers):
        raise ValueError("OCR page numbers must be supplied in ascending order")
    return resolved


def compute_input_fingerprint(
    *,
    hash_file: Callable[[Path], str],
    page_sources: Sequence[tuple[int, Path]] | None = None,
    pdf_path: Path | None = None,
) -> str:
    """Derive a stable run identity from the source PDF or ordered page files."""
    payload: dict[str, object] = {}
    if pdf_path is not None:
        payload["pdf_sha256"] = hash_file(pdf_path)
    if page_sources is not None:
        payload["pages"] = [
            {"page_number": page_number, "sha256": hash_file(path)}
            for page_number, path in page_sources
        ]
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
