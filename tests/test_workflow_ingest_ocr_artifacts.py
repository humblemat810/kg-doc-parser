"""Focused tests for OCR image staging and PDF cache artifacts."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from PIL import Image

import kg_doc_parser.workflow_ingest.ocr_pdf_rendering as ocr_pdf_rendering_module
from kg_doc_parser.workflow_ingest import OCRImagePayload
from kg_doc_parser.workflow_ingest.ocr_pipeline import (
    _materialize_image_payloads,
    _render_pdf_to_images,
)

pytestmark = [pytest.mark.workflow]


def test_materialize_image_payloads_uses_safe_atomic_page_paths(tmp_path: Path) -> None:
    rendered_dir = tmp_path / "rendered"
    payload = OCRImagePayload(
        page_number=2,
        image_bytes_b64=base64.b64encode(b"not-a-real-image").decode("ascii"),
        filename="../outside/escape.jpg",
    )

    resolved = _materialize_image_payloads([payload], rendered_dir)

    assert resolved == [(2, rendered_dir / "page_2.jpg")]
    assert resolved[0][1].read_bytes() == b"not-a-real-image"
    assert not (tmp_path / "outside" / "escape.jpg").exists()


def test_materialize_image_payloads_rejects_invalid_page_numbers_and_symlinks(
    tmp_path: Path,
) -> None:
    encoded = base64.b64encode(b"image").decode("ascii")
    with pytest.raises(ValueError, match="positive"):
        _materialize_image_payloads(
            [OCRImagePayload(page_number=0, image_bytes_b64=encoded)],
            tmp_path / "invalid",
        )
    with pytest.raises(ValueError, match="duplicate"):
        _materialize_image_payloads(
            [
                OCRImagePayload(page_number=1, image_bytes_b64=encoded),
                OCRImagePayload(page_number=1, image_bytes_b64=encoded),
            ],
            tmp_path / "duplicate",
        )

    rendered_dir = tmp_path / "symlinked"
    rendered_dir.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"original")
    page_link = rendered_dir / "page_1.png"
    try:
        page_link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is not available in this environment")
    with pytest.raises(ValueError, match="symlink"):
        _materialize_image_payloads(
            [OCRImagePayload(page_number=1, image_bytes_b64=encoded)], rendered_dir
        )
    assert outside.read_bytes() == b"original"

    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    copied_dir = tmp_path / "copied"
    copied_dir.mkdir()
    copied_link = copied_dir / "page_1.png"
    try:
        copied_link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is not available in this environment")
    with pytest.raises(ValueError, match="symlink"):
        _materialize_image_payloads(
            [OCRImagePayload(page_number=1, image_path=str(source))], copied_dir
        )
    assert outside.read_bytes() == b"original"


def test_materialize_image_payloads_rejects_out_of_order_page_numbers(
    tmp_path: Path,
) -> None:
    encoded = base64.b64encode(b"image").decode("ascii")

    with pytest.raises(ValueError, match="ascending order"):
        _materialize_image_payloads(
            [
                OCRImagePayload(page_number=2, image_bytes_b64=encoded),
                OCRImagePayload(page_number=1, image_bytes_b64=encoded),
            ],
            tmp_path / "out-of-order",
        )


def test_pdf_rasterization_regenerates_corrupt_cached_page_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rendered_dir = tmp_path / "rendered"
    rendered_dir.mkdir()
    corrupt_page = rendered_dir / "page_1.png"
    corrupt_page.write_bytes(b"partial-png")

    class _Reader:
        def __init__(self) -> None:
            self.pages = [object()]

    monkeypatch.setattr(ocr_pdf_rendering_module, "PdfReader", lambda _: _Reader())
    import pdf2image

    calls: list[int] = []

    def _convert(*args: object, **kwargs: object) -> list[Image.Image]:
        calls.append(int(kwargs["first_page"]))
        return [Image.new("RGB", (8, 8), "white")]

    monkeypatch.setattr(pdf2image, "convert_from_path", _convert)

    paths = _render_pdf_to_images(tmp_path / "source.pdf", rendered_dir)

    assert paths == [corrupt_page]
    assert calls == [1]
    with Image.open(corrupt_page) as image:
        assert image.size == (8, 8)
