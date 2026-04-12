from __future__ import annotations

"""Reusable OCR smoke asset generation for manual and CLI workflows."""

from pathlib import Path

from PIL import Image, ImageDraw


def _draw_page(path: Path, *, title: str, lines: list[str]) -> None:
    image = Image.new("RGB", (1400, 1000), "white")
    draw = ImageDraw.Draw(image)
    draw.text((80, 70), title, fill="black")
    y = 160
    for line in lines:
        draw.text((80, y), line, fill="black")
        y += 90
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG")


def _build_pdf(image_paths: list[Path], pdf_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    first, rest = images[0], images[1:]
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    first.save(pdf_path, "PDF", save_all=True, append_images=rest)
    for image in images:
        image.close()


def generate_ocr_smoke_assets(output_dir: Path) -> dict[str, str]:
    """Generate inspectable OCR smoke assets and return their file paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    page_1 = output_dir / "ocr_smoke_page_1.png"
    page_2 = output_dir / "ocr_smoke_page_2.png"
    pdf_path = output_dir / "ocr_smoke_document.pdf"

    _draw_page(
        page_1,
        title="OCR Workflow Smoke Page 1",
        lines=[
            "Section 1. Overview",
            "Alpha clause: the workflow should preserve page order.",
            "Beta clause: per-page OCR artifacts should remain inspectable.",
        ],
    )
    _draw_page(
        page_2,
        title="OCR Workflow Smoke Page 2",
        lines=[
            "Section 2. Review",
            "Gamma clause: downstream source maps should include OCR text.",
            "Delta clause: resume state should survive reruns.",
        ],
    )
    _build_pdf([page_1, page_2], pdf_path)
    return {
        "page_1_png": str(page_1),
        "page_2_png": str(page_2),
        "pdf": str(pdf_path),
    }
