"""Resolve OCR source inputs and their stable artifact identities."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .ocr_file_io import write_bytes_atomic
from .ocr_inputs import (
    compute_input_fingerprint as _compute_input_fingerprint_impl,
)
from .ocr_inputs import (
    find_matching_render_hash as _find_matching_render_hash_impl,
)
from .ocr_inputs import (
    materialize_image_payloads as _materialize_image_payloads_impl,
)
from .ocr_inputs import (
    sha256_file,
)
from .ocr_pdf_rendering import (
    _pdf_render_manifest_payload,
    _render_pdf_to_images,
    _render_pdf_with_identity,
)
from .ocr_pipeline_contracts import OCRImagePayload, PDFRasterizer


def _compute_input_fingerprint(
    *,
    page_sources: Sequence[tuple[int, Path]] | None = None,
    pdf_path: Path | None = None,
) -> str:
    return _compute_input_fingerprint_impl(
        hash_file=sha256_file,
        page_sources=page_sources,
        pdf_path=pdf_path,
    )

def _materialize_image_payloads(
    image_payloads: Sequence[OCRImagePayload], rendered_dir: Path
) -> list[tuple[int, Path]]:
    return _materialize_image_payloads_impl(
        image_payloads,
        rendered_dir,
        write_bytes=write_bytes_atomic,
    )

def _find_matching_render_hash(rendered_dir: Path, page_number: int) -> str | None:
    return _find_matching_render_hash_impl(
        rendered_dir, page_number, hash_file=sha256_file
    )
def _legacy_folder_name(*, document_id: str, pdf_path: Path | None) -> str:
    if pdf_path is not None:
        return pdf_path.name
    return document_id


def _source_artifact_folder_name(
    *, document_id: str, pdf_path: Path, input_fingerprint: str
) -> str:
    """Return a stable, collision-resistant folder name for one PDF source.

    The basename is useful when inspecting artifacts, but it is not an identity:
    two workspaces commonly contain unrelated files named ``report.pdf``.  Keep
    the human-readable components while suffixing the content fingerprint so
    cached pages cannot be reused across different sources.
    """

    def _safe_component(value: str) -> str:
        safe = "".join(
            character if character.isalnum() or character in {"-", "_", "."} else "_"
            for character in value
        ).strip("._")
        return safe[:80] or "document"

    return "-".join(
        (
            _safe_component(document_id),
            _safe_component(pdf_path.stem),
            input_fingerprint[:16],
        )
    )


@dataclass(slots=True)
class OCRSourcePlan:
    """Resolved file-system plan for one OCR ingest run.

    This captures the immutable inputs needed to process a document:
    where the pages live, where legacy JSON should be written, and how the
    page sources should be iterated.
    """

    pdf_source: Path | None
    rendered_dir: Path
    legacy_dir: Path
    page_sources: list[tuple[int, Path]]
    source_kind: str
    input_fingerprint: str
    render_configuration_fingerprint: str = "render-v1"


def _resolve_ocr_source_plan(
    *,
    document_id: str,
    output_dir: Path,
    image_payloads: Sequence[OCRImagePayload] | None,
    pdf_path: Path | None,
    pdf_rasterizer: PDFRasterizer | None,
) -> OCRSourcePlan:
    rendered_root = output_dir / "rendered_pages"
    legacy_root = output_dir / "legacy_split_pages"

    pdf_source = Path(pdf_path) if pdf_path is not None else None
    artifact_folder = _legacy_folder_name(
        document_id=document_id,
        pdf_path=pdf_source,
    )
    if pdf_source is not None:
        input_fingerprint = _compute_input_fingerprint(pdf_path=pdf_source)
        # Render once per PDF so downstream OCR can work page-by-page against
        # stable page image files and the state store can resume individual pages.
        artifact_folder = _source_artifact_folder_name(
            document_id=document_id,
            pdf_path=pdf_source,
            input_fingerprint=input_fingerprint,
        )
        rendered_dir = rendered_root / artifact_folder
        pdf_page_rasterizer = pdf_rasterizer or _render_pdf_to_images
        rendered_paths = _render_pdf_with_identity(
            pdf_path=pdf_source,
            rendered_dir=rendered_dir,
            rasterizer=pdf_page_rasterizer,
        )
        render_manifest = _pdf_render_manifest_payload(
            pdf_path=pdf_source,
            rasterizer=pdf_page_rasterizer,
        )
        render_identity = render_manifest.get("render_identity")
        if not isinstance(render_identity, str):
            raise ValueError("PDF render manifest did not produce an identity")
        render_configuration_fingerprint = render_identity
        page_sources = list(enumerate(rendered_paths, start=1))
        source_kind = "pdf"
    else:
        rendered_dir = rendered_root / document_id
        page_sources = _materialize_image_payloads(list(image_payloads or []), rendered_dir)
        source_kind = "image"
        input_fingerprint = _compute_input_fingerprint(page_sources=page_sources)
        render_configuration_fingerprint = hashlib.sha256(
            b'{"renderer":"image-materialize-v1"}'
        ).hexdigest()

    legacy_dir = legacy_root / (
        artifact_folder
        if pdf_source is not None
        else _legacy_folder_name(document_id=document_id, pdf_path=pdf_source)
    )
    legacy_dir.mkdir(parents=True, exist_ok=True)
    return OCRSourcePlan(
        pdf_source=pdf_source,
        rendered_dir=rendered_dir,
        legacy_dir=legacy_dir,
        page_sources=page_sources,
        source_kind=source_kind,
        input_fingerprint=input_fingerprint,
        render_configuration_fingerprint=render_configuration_fingerprint,
    )

