"""Typed public contracts and result models for OCR pipeline operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ..models import OCRClusterResponse
from .adapters import OCRPageJSON
from .document_models import DocumentTree, EvidenceSlice, SourceItem, Summary
from .models import WorkflowIngestInput
from .providers import WorkflowProviderSettings


class OCRImagePayload(BaseModel):
    """Single OCR page input, either file-backed or supplied as encoded bytes."""

    page_number: int | None = None
    image_path: str | None = None
    image_bytes_b64: str | None = None
    filename: str | None = None


class OCRPageProgress(BaseModel):
    page_number: int
    image_path: str
    image_sha256: str
    json_path: str
    status: str = "completed"


class OCRWorkflowProgress(BaseModel):
    document_id: str
    title: str
    source_kind: str
    total_pages: int
    pages: dict[str, OCRPageProgress] = Field(default_factory=dict)


@dataclass(slots=True)
class OCRWorkflowArtifacts:
    """Inspectable result bundle returned after OCR preparation."""

    workflow_input: WorkflowIngestInput
    ocr_pages: list[OCRPageJSON]
    legacy_dir: Path
    rendered_dir: Path
    state_db_path: Path
    progress_path: Path
    summary_path: Path
    completed_pages: list[int]
    reused_pages: list[int]
    document_tree: DocumentTree | None = None

    @property
    def document_tree_path(self) -> Path:
        return self.legacy_dir / "document-tree.json"


@dataclass(frozen=True, slots=True)
class GroupingCheckpointMetadata:
    """Durable identity for one accepted grouping checkpoint."""

    document_id: str
    source_fingerprint: str
    grouping_policy_version: str
    accepted_page_frontier: tuple[int, ...]
    tree_revision: str
    merge_revision: int
    last_runtime_step: str


class OCRRunner(Protocol):
    """Run OCR for one page using the selected provider settings."""

    def __call__(
        self,
        image_path: Path,
        page_number: int,
        provider_settings: WorkflowProviderSettings,
        /,
    ) -> OCRClusterResponse: ...


class PDFRasterizer(Protocol):
    """Materialize a PDF into ordered page images in a destination directory."""

    def __call__(self, pdf_path: Path, rendered_dir: Path, /) -> list[Path]: ...


@runtime_checkable
class PDFRasterizerWithIdentity(PDFRasterizer, Protocol):
    """Optional rasterizer extension for configuration-sensitive cache keys."""

    configuration_fingerprint: str


class SummaryGenerator(Protocol):
    """Generate one bounded, evidence-grounded summary for a tree node."""

    def __call__(
        self,
        *,
        tree: DocumentTree,
        node_id: str,
        source_items: tuple[SourceItem, ...],
        input_fingerprint: str,
    ) -> Summary: ...


class GroundedSummaryGenerator(Protocol):
    """Generate a summary from only the exact evidence slices it owns."""

    def __call__(
        self,
        *,
        tree: DocumentTree,
        node_id: str,
        evidence_slices: tuple[EvidenceSlice, ...],
        input_fingerprint: str,
    ) -> Summary: ...


@runtime_checkable
class SummaryGeneratorWithIdentity(Protocol):
    """Optional stable identity required for persistent summary cache reuse."""

    @property
    def configuration_fingerprint(self) -> str: ...


__all__ = [
    "GroundedSummaryGenerator",
    "GroupingCheckpointMetadata",
    "OCRImagePayload",
    "OCRPageProgress",
    "OCRRunner",
    "OCRWorkflowArtifacts",
    "OCRWorkflowProgress",
    "PDFRasterizer",
    "PDFRasterizerWithIdentity",
    "SummaryGenerator",
    "SummaryGeneratorWithIdentity",
]
