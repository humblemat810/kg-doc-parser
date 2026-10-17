"""Workflow-first OCR ingest helpers for image and PDF sources.

This module sits at the boundary between raw OCR and the reusable workflow
ingest pipeline. It is responsible for three ideas that are easy to conflate:

1. **Page materialization**
   - Accept already-rendered page images, or render a PDF into page images.
   - Keep the page images around on disk so humans can inspect what was OCR'd.

2. **OCR execution and resume state**
   - Run one OCR provider call per page through a pluggable callable hook.
   - Persist page-level progress in ``ocr-state.sqlite`` so reruns can skip
     completed pages and rebuild state if the DB is missing.
   - Save legacy-compatible ``page_N.json`` artifacts alongside the images.

3. **Workflow normalization**
   - Convert the serialized OCR pages into ``WorkflowIngestInput``.
   - Hand that normalized input to the downstream semantic workflow parser.

Important concepts used throughout the file:

- ``OCRRunner``: a callable that OCRs exactly one page image and returns one
  structured OCR page response.
- ``PDFRasterizer``: a callable that turns a PDF into a list of page image paths.
- ``OCRWorkflowStateStore``: the SQLite-backed resume store for page render/OCR
  state.
- ``OCRWorkflowArtifacts``: the inspectable bundle returned after OCR prep so
  tests and manual runs can open the generated files.

The public entrypoints are intentionally explicit so tests and manual runs can
inspect intermediate folders without needing to understand the legacy OCR code.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import logging
import math
import os
import shutil
import sqlite3
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from types import FunctionType
from typing import Literal, NotRequired, Protocol, TypedDict, cast, runtime_checkable

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image
from pydantic import BaseModel, Field
from pypdf import PdfReader
from pypdf.errors import PdfReadError, PdfStreamError

from ..llm_structured_output import build_structured_output_runnable
from ..models import OCRClusterResponse, SplitPage, SplitPageMeta, TextCluster
from .adapters import OCRPageJSON, _validated_ocr_page_json, normalize_ocr_pages
from .document_adapters import (
    apply_grouping_proposals,
    apply_summary_updates,
    attach_document_tree_metadata,
    build_document_tree,
    select_summary_targets,
    suggest_cross_page_continuations,
)
from .document_grouping import GroupProposal
from .document_models import (
    DocumentNode,
    DocumentTree,
    EvidenceSlice,
    SourceItem,
    Summary,
    SummaryPolicy,
)
from .models import WorkflowExportBundle, WorkflowIngestInput
from .probe import WorkflowProbe, emit_probe_event
from .providers import (
    ProviderDiagnostics,
    ProviderDiagnosticsSink,
    WorkflowProviderSettings,
    build_chat_model_for_role,
    invoke_with_timeout,
)
from .runtime_contracts import WorkflowRuntimeDeps
from .sdk_boundary import raw_content
from .serialization import JsonValue, json_safe
from .service import _RunCompat

_LOGGER = logging.getLogger(__name__)

JsonObject = dict[str, JsonValue]


def _json_object(value: object) -> JsonObject:
    if isinstance(value, dict):
        return cast(JsonObject, value)
    return {}


def _read_json_object(path: Path) -> JsonObject:
    """Read optional resume metadata without making it authoritative."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return _json_object(value)


def _is_sha256_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _ocr_configuration_fingerprint(
    settings: WorkflowProviderSettings,
    *,
    candidate_models: Sequence[str] = (),
    quality_policy: OCRQualityPolicy | None = None,
) -> str:
    """Return the non-secret identity of an OCR execution configuration."""
    config = {
        "provider": settings.ocr.provider,
        "model": settings.ocr.model,
        "base_url": settings.ocr.base_url,
        "api_version": settings.ocr.api_version,
        "project": settings.ocr.project,
        "ocr_engine": settings.ocr.ocr_engine,
        "pdf_text_mode": settings.ocr.pdf_text_mode,
        "ocr_language": settings.ocr.ocr_language,
        "candidate_models": list(candidate_models),
        "quality_policy": (quality_policy or OCRQualityPolicy()).model_dump(mode="json"),
    }
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _ocr_page_metadata(
    settings: WorkflowProviderSettings,
    *,
    candidate_models: Sequence[str] = (),
    source_image_sha256: str | None = None,
    quality_policy: OCRQualityPolicy | None = None,
) -> dict[str, str]:
    """Build stable, non-secret provenance for one accepted OCR page."""
    metadata = {
        "ocr_provider": settings.ocr.provider,
        "ocr_model_name": settings.ocr.model,
        "ocr_pdf_text_mode": settings.ocr.pdf_text_mode,
        "ocr_configuration_fingerprint": _ocr_configuration_fingerprint(
            settings,
            candidate_models=candidate_models,
            quality_policy=quality_policy,
        ),
    }
    if source_image_sha256 is not None:
        metadata["source_image_sha256"] = source_image_sha256
    return metadata


class OCRImagePayload(BaseModel):
    """Single OCR page input.

    A payload may already exist on disk, or it can be materialized from bytes
    into the working directory. The resulting file path is what the OCR model
    and the legacy artifact writer operate on.
    """

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


class _OCRArtifactMetadata(TypedDict, total=False):
    """Known metadata fields read from a persisted OCR artifact.

    Artifact JSON remains an external compatibility boundary. Keeping the
    fields we consume explicit prevents later ``dict.get`` calls from
    degrading known strings and numeric values back to ``object``.
    """

    ocr_provider: NotRequired[str]
    ocr_model_name: NotRequired[str]
    ocr_configuration_fingerprint: NotRequired[str]
    source_image_sha256: NotRequired[str]
    ocr_json_version: NotRequired[str]
    ocr_datetime: NotRequired[float]


def _ocr_artifact_metadata(payload: Mapping[str, object]) -> _OCRArtifactMetadata:
    """Narrow the metadata fields used by reuse and recovery decisions."""

    raw_metadata = payload.get("metadata")
    if not isinstance(raw_metadata, dict):
        return {}
    metadata: _OCRArtifactMetadata = {}
    ocr_provider = raw_metadata.get("ocr_provider")
    if isinstance(ocr_provider, str):
        metadata["ocr_provider"] = ocr_provider
    ocr_model_name = raw_metadata.get("ocr_model_name")
    if isinstance(ocr_model_name, str):
        metadata["ocr_model_name"] = ocr_model_name
    configuration_fingerprint = raw_metadata.get("ocr_configuration_fingerprint")
    if isinstance(configuration_fingerprint, str):
        metadata["ocr_configuration_fingerprint"] = configuration_fingerprint
    source_image_sha256 = raw_metadata.get("source_image_sha256")
    if isinstance(source_image_sha256, str):
        metadata["source_image_sha256"] = source_image_sha256
    ocr_json_version = raw_metadata.get("ocr_json_version")
    if isinstance(ocr_json_version, str):
        metadata["ocr_json_version"] = ocr_json_version
    timestamp = raw_metadata.get("ocr_datetime")
    if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool):
        metadata["ocr_datetime"] = float(timestamp)
    return metadata


@dataclass(slots=True)
class OCRWorkflowArtifacts:
    """Inspectable result bundle produced by the OCR preparation phase.

    This is the bridge object between the OCR world and the workflow ingest
    world. It keeps both the normalized workflow input and the on-disk
    artifacts that were used to produce it.
    """
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


def _summary_generator_identity(
    generator: SummaryGenerator | GroundedSummaryGenerator,
) -> str:
    """Return a stable, typed identity for the active summary implementation."""
    if isinstance(generator, FunctionType):
        return f"{generator.__module__}.{generator.__qualname__}"
    generator_type = type(generator)
    return f"{generator_type.__module__}.{generator_type.__qualname__}"


class OCRQualityPolicy(BaseModel):
    """Policy for turning provider quality signals into acceptance decisions."""

    max_data_loss_likelihood: float = Field(default=1.0, ge=0.0, le=1.0)
    reject_incomplete_text: bool = False
    reject_incomplete_words_on_edge: bool = False
    reject_empty_page: bool = False
    reject_unresolved_items: bool = False
    reject_invalid_spatial_coordinates: bool = True


class OCRQualityDiagnostic(BaseModel):
    """A bounded, persisted explanation for a flagged OCR page."""

    code: str
    severity: Literal["warning", "error"]
    message: str


def _evaluate_ocr_quality(
    response: OCRClusterResponse,
    policy: OCRQualityPolicy,
) -> tuple[OCRQualityDiagnostic, ...]:
    diagnostics: list[OCRQualityDiagnostic] = []
    if response.is_empty_page:
        diagnostics.append(
            OCRQualityDiagnostic(
                code="empty_ocr_output",
                severity="error" if policy.reject_empty_page else "warning",
                message="provider reported no OCR or non-text evidence for the page",
            )
        )
    if response.incomplete_text:
        diagnostics.append(
            OCRQualityDiagnostic(
                code="incomplete_text",
                severity="error" if policy.reject_incomplete_text else "warning",
                message="provider marked the OCR text as incomplete",
            )
        )
    if response.incomplete_words_on_edge:
        diagnostics.append(
            OCRQualityDiagnostic(
                code="incomplete_words_on_edge",
                severity="error" if policy.reject_incomplete_words_on_edge else "warning",
                message="provider marked words at a page edge as incomplete",
            )
        )
    if response.data_loss_likelihood > policy.max_data_loss_likelihood:
        diagnostics.append(
            OCRQualityDiagnostic(
                code="data_loss_likelihood",
                severity="error",
                message=(
                    "provider data-loss likelihood exceeds the configured "
                    f"limit ({response.data_loss_likelihood:.3f} > "
                    f"{policy.max_data_loss_likelihood:.3f})"
                ),
            )
        )
    if response.unresolved_item_count:
        diagnostics.append(
            OCRQualityDiagnostic(
                code="unresolved_items",
                severity="error" if policy.reject_unresolved_items else "warning",
                message=(
                    "provider reported items that could not be represented with "
                    f"grounded records ({response.unresolved_item_count})"
                ),
            )
        )
    if response.spatial_accuracy != "page":
        invalid_coordinates = [
            cluster.cluster_number
            for cluster in (*response.OCR_text_clusters, *response.non_text_objects)
            if not all(
                math.isfinite(value)
                for value in (
                    cluster.bb_x_min,
                    cluster.bb_y_min,
                    cluster.bb_x_max,
                    cluster.bb_y_max,
                )
            )
            or cluster.bb_x_min >= cluster.bb_x_max
            or cluster.bb_y_min >= cluster.bb_y_max
        ]
        if invalid_coordinates:
            diagnostics.append(
                OCRQualityDiagnostic(
                    code="invalid_spatial_coordinates",
                    severity=(
                        "error"
                        if policy.reject_invalid_spatial_coordinates
                        else "warning"
                    ),
                    message=(
                        "provider declared item-level spatial precision but emitted "
                        "invalid bounding boxes for clusters "
                        + ", ".join(str(number) for number in invalid_coordinates)
                    ),
                )
            )
    return tuple(diagnostics)

_OCR_STATE_SCHEMA_VERSION = 2
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


def _write_document_tree(path: Path, tree: DocumentTree) -> None:
    """Persist the derived tree with replace semantics for crash-safe replay."""
    _write_bytes_atomic(
        path,
        json.dumps(tree.model_dump(mode="json"), indent=2, ensure_ascii=False).encode("utf-8"),
    )


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Publish bytes through a unique same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _write_json_atomic(path: Path, payload: object) -> None:
    """Publish a JSON artifact only after serialization succeeds."""
    _write_bytes_atomic(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"),
    )


def _image_file_is_valid(path: Path) -> bool:
    """Return whether a cached image can be decoded as a complete image."""
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, SyntaxError, ValueError):
        return False
    return True


def _write_png_atomic(image: Image.Image, path: Path) -> None:
    """Encode and publish a rasterized page without exposing partial PNG data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        os.close(descriptor)
        image.save(temporary_name, "PNG")
        if not _image_file_is_valid(Path(temporary_name)):
            raise RuntimeError(f"rasterizer produced an invalid image for {path.name}")
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _coalesce_cross_page_continuations(
    proposals: Sequence[GroupProposal],
) -> tuple[GroupProposal, ...]:
    """Turn adjacent pair suggestions into disjoint continuation chains."""
    if not proposals:
        return ()
    chains: list[GroupProposal] = []
    current = proposals[0]
    for proposal in proposals[1:]:
        if (
            proposal.kind == current.kind
            and proposal.members[0] == current.members[-1]
        ):
            current = GroupProposal(
                kind=current.kind,
                members=current.members + (proposal.members[-1],),
            )
        else:
            chains.append(current)
            current = proposal
    chains.append(current)
    return tuple(chains)


def _apply_cross_page_continuations(
    tree: DocumentTree,
    *,
    grouping_policy_version: str,
) -> DocumentTree:
    """Apply only provider-signaled, adjacent cross-page continuations."""
    proposals = _coalesce_cross_page_continuations(suggest_cross_page_continuations(tree))
    if not proposals:
        return tree
    return apply_grouping_proposals(
        tree,
        proposals,
        grouping_policy_version=grouping_policy_version,
    )


def _generate_tree_summaries(
    tree: DocumentTree,
    *,
    document_id: str,
    state_store: OCRWorkflowStateStore,
    policy: SummaryPolicy | None,
    generator: SummaryGenerator | None,
    grounded_generator: GroundedSummaryGenerator | None = None,
    grouping_policy_version: str,
    expected_tree_revision: str | None = None,
    require_no_existing_tree: bool = False,
    last_runtime_step: str = "generate_grouping_summaries",
) -> DocumentTree:
    """Run optional bounded summary generation without making OCR depend on it."""
    if (generator is None and grounded_generator is None) or policy is None or not policy.enabled:
        return tree
    if generator is not None and grounded_generator is not None:
        raise ValueError("provide generator or grounded_generator, not both")
    items = {item.item_id: item for item in tree.source_items}
    current = tree
    for node_id in select_summary_targets(current, policy):
        evidence_fingerprint = current.summary_input_fingerprint(node_id)
        policy_payload = policy.model_dump(mode="json") if policy is not None else {}
        active_generator = generator if generator is not None else grounded_generator
        assert active_generator is not None
        generator_name = _summary_generator_identity(active_generator)
        input_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "evidence": evidence_fingerprint,
                    "tree_revision": current.tree_revision,
                    "grouping_policy_version": grouping_policy_version,
                    "summary_policy": policy_payload,
                    "generation_fingerprint": policy.generation_fingerprint,
                    "generator": generator_name,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        # Recompute the node's evidence through the tree contract; the
        # generator receives only source items in that subtree.
        node_map = {node.node_id: node for node in current.nodes}
        covered: set[str] = set()

        def collect(
            current_id: str,
            node_map: dict[str, DocumentNode] = node_map,
            covered: set[str] = covered,
        ) -> None:
            node = node_map[current_id]
            if node.parts:
                covered.update(part.item_id for part in node.parts)
                return
            for child in node.children:
                collect(child.node_id)

        collect(node_id)
        evidence_ids = tuple(sorted(covered))
        summary = state_store.load_completed_summary(
            document_id=document_id,
            node_id=node_id,
            input_fingerprint=input_fingerprint,
        )
        if summary is not None:
            current_node = node_map[node_id]
            if current_node.summary == summary:
                # The accepted journal row already represents this exact
                # generation. Do not create another accepted attempt or
                # advance the tree revision on a retry.
                continue
            # A crash may have committed the journal row after the tree write
            # was interrupted. Repair only the missing derived field; the
            # accepted summary remains the single canonical journal result.
            updated = apply_summary_updates(current, {node_id: summary})
            expected_tree_revision = state_store.save_document_tree(
                document_id=document_id,
                tree=updated,
                grouping_policy_version=grouping_policy_version,
                expected_tree_revision=expected_tree_revision,
                require_no_existing_tree=require_no_existing_tree,
                last_runtime_step=last_runtime_step,
            )
            require_no_existing_tree = False
            current = updated
            continue

        state_store.record_summary_attempt(
            document_id=document_id,
            node_id=node_id,
            input_fingerprint=input_fingerprint,
            status="pending",
        )
        try:
            if grounded_generator is not None:
                summary = grounded_generator(
                    tree=current,
                    node_id=node_id,
                    evidence_slices=current.evidence_slices(node_id, evidence_ids),
                    input_fingerprint=evidence_fingerprint,
                )
            else:
                assert generator is not None
                summary = generator(
                    tree=current,
                    node_id=node_id,
                    source_items=tuple(items[item_id] for item_id in evidence_ids),
                    input_fingerprint=evidence_fingerprint,
                )
            summary = summary.model_copy(
                update={
                    "evidence_slice_fingerprint": current.summary_slice_fingerprint(
                        node_id, evidence_ids
                    )
                }
            )
            updated = apply_summary_updates(current, {node_id: summary})
            expected_tree_revision = state_store.save_document_tree(
                document_id=document_id,
                tree=updated,
                grouping_policy_version=grouping_policy_version,
                expected_tree_revision=expected_tree_revision,
                require_no_existing_tree=require_no_existing_tree,
                last_runtime_step=last_runtime_step,
            )
            require_no_existing_tree = False
            state_store.record_summary_attempt(
                document_id=document_id,
                node_id=node_id,
                input_fingerprint=input_fingerprint,
                status=summary.status,
                summary=summary,
            )
            current = updated
        except GroupingCheckpointConflict:
            # A stale writer must fail the workflow so the caller can restore
            # the newer checkpoint; treating this as an optional summary error
            # would hide a lost-write race.
            raise
        except Exception as exc:  # noqa: BLE001
            state_store.record_summary_attempt(
                document_id=document_id,
                node_id=node_id,
                input_fingerprint=input_fingerprint,
                status="failed",
                error_message=str(exc),
            )
    return current


class GroupingCheckpointConflict(RuntimeError):
    """A grouping writer lost its expected checkpoint ownership."""


@dataclass(slots=True)
class OCRPageStateSnapshot:
    """One page/stage snapshot from the SQLite resume store."""
    page_number: int
    stage: str
    attempt_count: int
    status: str
    content_hash: str | None
    artifact_path: str | None
    last_error: str | None
    last_model: str | None
    last_attempted_ts: float | None
    config_fingerprint: str | None = None
    artifact_sha256: str | None = None


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


@dataclass(slots=True)
class OCRPageProcessingContext:
    """Immutable per-run OCR processing context.

    This groups the values that every page in a run needs so the per-page loop
    can stay focused on the page-specific state rather than carrying a very
    long parameter list.
    """

    document_id: str
    title: str
    source_kind: str
    total_pages: int
    input_fingerprint: str
    candidate_models: list[str]
    provider_settings: WorkflowProviderSettings
    ocr_page_runner: OCRRunner
    state_store: OCRWorkflowStateStore
    probe: WorkflowProbe | None
    configuration_fingerprints: tuple[str, ...]
    render_configuration_fingerprint: str = "render-v1"
    quality_policy: OCRQualityPolicy = field(default_factory=OCRQualityPolicy)


@dataclass(slots=True)
class OCRPagePreparation:
    """Durable inputs and injected dependencies for the page workflow.

    The preparation object deliberately contains runtime-only dependencies.  A
    workflow checkpoint receives only the JSON source plan and accepted page
    payloads; providers, SQLite handles, and callables stay in the process that
    owns the runtime resolver.
    """

    source_plan: OCRSourcePlan
    state_store: OCRWorkflowStateStore
    page_context: OCRPageProcessingContext
    state_db_path: Path
    progress_path: Path
    summary_path: Path


def _build_ocr_page_preparation(
    *,
    document_id: str,
    title: str,
    output_dir: Path,
    image_payloads: Sequence[OCRImagePayload] | None,
    pdf_path: Path | None,
    provider_settings: WorkflowProviderSettings,
    ocr_runner: OCRRunner | None,
    pdf_rasterizer: PDFRasterizer | None,
    ocr_candidate_models: Sequence[str] | None,
    probe: WorkflowProbe | None,
    quality_policy: OCRQualityPolicy | None = None,
) -> OCRPagePreparation:
    """Build the page-run context without invoking a provider."""

    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "ocr-progress.json"
    summary_path = output_dir / "ocr-summary.json"
    state_db_path = output_dir / "ocr-state.sqlite"
    source_plan = _resolve_ocr_source_plan(
        document_id=document_id,
        output_dir=output_dir,
        image_payloads=image_payloads,
        pdf_path=pdf_path,
        pdf_rasterizer=pdf_rasterizer,
    )
    state_store = OCRWorkflowStateStore.open_or_rebuild(
        db_path=state_db_path,
        document_id=document_id,
        title=title,
        source_kind=source_plan.source_kind,
        total_pages=len(source_plan.page_sources),
        input_fingerprint=source_plan.input_fingerprint,
        rendered_dir=source_plan.rendered_dir,
        legacy_dir=source_plan.legacy_dir,
        progress_path=progress_path,
        render_configuration_fingerprint=source_plan.render_configuration_fingerprint,
    )
    candidate_models = list(ocr_candidate_models or [provider_settings.ocr.model])
    if ocr_runner is None:
        def _instrumented_ocr_runner(
            image_path: Path,
            page_number: int,
            settings: WorkflowProviderSettings,
        ) -> OCRClusterResponse:
            return _run_live_ocr_page(
                image_path,
                page_number,
                settings,
                provider_diagnostics_sink=(
                    lambda diagnostics: _emit_ocr_event(probe, "ocr.provider_call", **diagnostics)
                    if probe is not None
                    else None
                ),
            )

        effective_runner: OCRRunner = _instrumented_ocr_runner
    else:
        effective_runner = ocr_runner
    native_pdf_path = source_plan.pdf_source
    if native_pdf_path is not None and provider_settings.ocr.pdf_text_mode != "disabled":
        configured_runner = effective_runner

        def _pdf_aware_runner(
            image_path: Path,
            page_number: int,
            settings: WorkflowProviderSettings,
        ) -> OCRClusterResponse:
            from .pdf_native import extract_native_pdf_page

            native = extract_native_pdf_page(native_pdf_path, page_number)
            if native is not None:
                return native
            if settings.ocr.pdf_text_mode == "native_only":
                raise ValueError(
                    f"native PDF text is unavailable for page {page_number}; "
                    "native_only mode refuses an ungrounded fallback"
                )
            return configured_runner(image_path, page_number, settings)

        effective_runner = _pdf_aware_runner
    page_context = OCRPageProcessingContext(
        document_id=document_id,
        title=title,
        source_kind=source_plan.source_kind,
        total_pages=len(source_plan.page_sources),
        input_fingerprint=source_plan.input_fingerprint,
        candidate_models=candidate_models,
        provider_settings=provider_settings,
        ocr_page_runner=effective_runner,
        state_store=state_store,
        probe=probe,
        configuration_fingerprints=tuple(
            _ocr_configuration_fingerprint(
                provider_settings.model_copy(
                    update={"ocr": provider_settings.ocr.model_copy(update={"model": model})}
                ),
                candidate_models=candidate_models,
                quality_policy=quality_policy or OCRQualityPolicy(),
            )
            for model in candidate_models
        )
        + (
            _ocr_page_metadata(
                provider_settings,
                quality_policy=quality_policy or OCRQualityPolicy(),
            )["ocr_configuration_fingerprint"],
        ),
        render_configuration_fingerprint=source_plan.render_configuration_fingerprint,
        quality_policy=quality_policy or OCRQualityPolicy(),
    )
    return OCRPagePreparation(
        source_plan=source_plan,
        state_store=state_store,
        page_context=page_context,
        state_db_path=state_db_path,
        progress_path=progress_path,
        summary_path=summary_path,
    )


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


def _process_ocr_page(
    *,
    context: OCRPageProcessingContext,
    done_count: int,
    page_number: int,
    image_path: Path,
    image_sha256: str,
    page_json_path: Path,
    page_image_path: Path,
    raw_pages: list[OCRPageJSON],
    completed_pages: list[int],
    reused_pages: list[int],
    page_failures: list[int],
) -> None:
    """Process one page image, either by resuming or by running OCR models."""

    state_store = context.state_store
    document_id = context.document_id
    title = context.title
    source_kind = context.source_kind
    total_pages = context.total_pages
    probe = context.probe

    state_store.ensure_document(
        document_id=document_id,
        title=title,
        source_kind=source_kind,
        total_pages=total_pages,
        input_fingerprint=context.input_fingerprint,
    )
    if not state_store.should_skip_page(
        document_id=document_id,
        page_number=page_number,
        stage="render",
        content_hash=image_sha256,
        artifact_path=image_path,
        config_fingerprint=context.render_configuration_fingerprint,
    ):
        _emit_ocr_event(
            probe,
            "ocr.render_started",
            document_id=document_id,
            page_number=page_number,
            image_path=str(image_path),
        )
        render_attempt = state_store.record_attempt(
            document_id=document_id,
            page_number=page_number,
            stage="render",
            content_hash=image_sha256,
            model_name="local-materialize",
            artifact_path=image_path,
            config_fingerprint=context.render_configuration_fingerprint,
        )
        state_store.record_page_completed(
            document_id=document_id,
            page_number=page_number,
            stage="render",
            content_hash=image_sha256,
            artifact_path=image_path,
            model_name="local-materialize",
            attempt_index=render_attempt,
            config_fingerprint=context.render_configuration_fingerprint,
        )
        _emit_ocr_event(
            probe,
            "ocr.render_completed",
            document_id=document_id,
            page_number=page_number,
            image_path=str(image_path),
        )

    # Reconcile an artifact-first commit before checking the SQLite completion
    # row. A fresh recovery child starts with cursor zero, so its checkpoint
    # prefix reconciliation cannot visit page one; the artifact itself must
    # still be allowed to restore the accepted result.
    state_store.reconcile_orphan_artifact(
        document_id=document_id,
        page_number=page_number,
        content_hash=image_sha256,
        artifact_path=page_json_path,
        config_fingerprints=context.configuration_fingerprints,
    )

    # Resume boundary: if the OCR page artifact already matches the image,
    # reuse the serialized page instead of calling the model again.
    if state_store.should_skip_page(
        document_id=document_id,
        page_number=page_number,
        stage="ocr",
        content_hash=image_sha256,
        artifact_path=page_json_path,
        config_fingerprint=context.configuration_fingerprints,
    ):
        persisted_page = _validated_ocr_page_json(_read_json_object(page_json_path))
        split_page = SplitPage.model_validate(persisted_page)
        raw_page = split_page.dump_supercede_parse()
        raw_page["meaningful_ordering"] = list(split_page.meaningful_ordering)
        persisted_metadata = persisted_page.get("metadata")
        raw_page["metadata"] = (
            persisted_metadata
            if persisted_metadata is not None
            else split_page.metadata.model_dump(mode="json")
        )
        raw_pages.append(_validated_ocr_page_json(raw_page, default_page_number=page_number))
        completed_pages.append(page_number)
        reused_pages.append(page_number)
        _emit_ocr_event(
            probe,
            "ocr.page_reused",
            document_id=document_id,
            page_number=page_number,
            page_json_path=str(page_json_path),
        )
        _LOGGER.info(
            "ocr resume | %s/%s | %s | reused page %s",
            done_count,
            total_pages,
            _progress_bar(done_count, total_pages),
            page_number,
        )
        return

    last_exc: Exception | None = None
    page_completed = False
    _emit_ocr_event(
        probe,
        "ocr.page_started",
        document_id=document_id,
        page_number=page_number,
        page_json_path=str(page_json_path),
    )
    # Try candidate OCR models one-by-one for this page; the first grounded
    # success wins and gets persisted as the canonical page artifact.
    for candidate_model in context.candidate_models:
        candidate_settings: WorkflowProviderSettings = context.provider_settings.model_copy(
            update={
                "ocr": context.provider_settings.ocr.model_copy(update={"model": candidate_model}),
            }
        )
        _emit_ocr_event(
            probe,
            "ocr.candidate_started",
            document_id=document_id,
            page_number=page_number,
            model_name=candidate_model,
        )
        attempt_index = state_store.record_attempt(
            document_id=document_id,
            page_number=page_number,
            stage="ocr",
            content_hash=image_sha256,
            model_name=candidate_model,
            artifact_path=page_json_path,
            config_fingerprint=_ocr_configuration_fingerprint(
                candidate_settings,
                candidate_models=context.candidate_models,
                quality_policy=context.quality_policy,
            ),
        )
        try:
            response = context.ocr_page_runner(image_path, page_number, candidate_settings)
            quality_diagnostics = _evaluate_ocr_quality(response, context.quality_policy)
            if any(item.severity == "error" for item in quality_diagnostics):
                raise ValueError(
                    "OCR quality policy rejected page "
                    f"{page_number}: "
                    + "; ".join(item.code for item in quality_diagnostics)
                )
            split_page: SplitPage = SplitPage(
                pdf_page_num=page_number,
                metadata=SplitPageMeta(
                    ocr_model_name=candidate_model,
                    ocr_datetime=0.0,
                    ocr_json_version="workflow_ingest_v1",
                ),
                **response.model_dump(dump_format="python"),
            )
            shutil.copy2(image_path, page_image_path)
            page_payload = split_page.dump_raw(dump_format="json")
            page_payload["metadata"] = {
                **split_page.metadata.model_dump(mode="json"),
                **_ocr_page_metadata(
                    candidate_settings,
                    candidate_models=context.candidate_models,
                    source_image_sha256=image_sha256,
                    quality_policy=context.quality_policy,
                ),
                "ocr_quality_status": "flagged" if quality_diagnostics else "accepted",
                "ocr_quality_diagnostics": [
                    item.model_dump(mode="json") for item in quality_diagnostics
                ],
            }
            # Validate before publication; the SQLite row is committed only
            # after the complete artifact is atomically visible. Publication
            # and acceptance are serialized so duplicate workers cannot
            # replace an already accepted page.
            SplitPage.model_validate(page_payload)
            published = state_store.publish_ocr_page(
                document_id=document_id,
                page_number=page_number,
                content_hash=image_sha256,
                artifact_path=page_json_path,
                payload=page_payload,
                model_name=candidate_model,
                attempt_index=attempt_index,
                config_fingerprint=_ocr_configuration_fingerprint(
                    candidate_settings,
                    candidate_models=context.candidate_models,
                    quality_policy=context.quality_policy,
                ),
            )
            if not published:
                # Another worker accepted the same page while this provider
                # call was running. Rehydrate the winner and report reuse;
                # never append this worker's unaccepted payload.
                if not state_store.should_skip_page(
                    document_id=document_id,
                    page_number=page_number,
                    stage="ocr",
                    content_hash=image_sha256,
                    artifact_path=page_json_path,
                    config_fingerprint=context.configuration_fingerprints,
                ):
                    raise RuntimeError("OCR page acceptance lost without a reusable winner")
                persisted_page = json.loads(page_json_path.read_text(encoding="utf-8"))
                split_page = SplitPage.model_validate(persisted_page)
                raw_page = split_page.dump_supercede_parse()
                raw_page["meaningful_ordering"] = list(split_page.meaningful_ordering)
                raw_page["metadata"] = persisted_page.get(
                    "metadata", split_page.metadata.model_dump(mode="json")
                )
                raw_pages.append(
                    _validated_ocr_page_json(raw_page, default_page_number=page_number)
                )
                completed_pages.append(page_number)
                reused_pages.append(page_number)
                page_completed = True
                break
            # The adapter consumes plain dict pages, not the Pydantic page
            # model, so we serialize the page into the legacy-compatible shape here.
            raw_page = split_page.dump_supercede_parse()
            raw_page["meaningful_ordering"] = list(split_page.meaningful_ordering)
            raw_page["metadata"] = page_payload["metadata"]
            raw_pages.append(_validated_ocr_page_json(raw_page, default_page_number=page_number))
            completed_pages.append(page_number)
            page_completed = True
            _emit_ocr_event(
                probe,
                "ocr.candidate_completed",
                document_id=document_id,
                page_number=page_number,
                model_name=candidate_model,
                page_json_path=str(page_json_path),
            )
            _LOGGER.info(
                "ocr progress | %s/%s | %s | completed page %s | model=%s",
                done_count,
                total_pages,
                _progress_bar(done_count, total_pages),
                page_number,
                candidate_model,
            )
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            state_store.record_page_failed(
                document_id=document_id,
                page_number=page_number,
                stage="ocr",
                content_hash=image_sha256,
                config_fingerprint=_ocr_configuration_fingerprint(
                    candidate_settings,
                    candidate_models=context.candidate_models,
                    quality_policy=context.quality_policy,
                ),
                error_message=str(exc),
                model_name=candidate_model,
                attempt_index=attempt_index,
            )
            _LOGGER.warning(
                "ocr candidate failed | page=%s model=%s attempt=%s error=%s",
                page_number,
                candidate_model,
                attempt_index,
                exc,
            )
            _emit_ocr_event(
                probe,
                "ocr.candidate_failed",
                document_id=document_id,
                page_number=page_number,
                model_name=candidate_model,
                attempt_index=attempt_index,
                error=str(exc),
            )
    if not page_completed:
        page_failures.append(page_number)
        _emit_ocr_event(
            probe,
            "ocr.page_failed",
            document_id=document_id,
            page_number=page_number,
            candidate_models=list(context.candidate_models),
        )
        _LOGGER.warning(
            "ocr progress | %s/%s | %s | failed page %s after %s candidate(s)",
            done_count,
            total_pages,
            _progress_bar(done_count, total_pages),
            page_number,
            len(context.candidate_models),
        )
        if last_exc is not None:
            _LOGGER.warning("last ocr error for page %s: %s", page_number, last_exc)


def _finalize_ocr_workflow_artifacts(
    *,
    document_id: str,
    title: str,
    raw_pages: list[OCRPageJSON],
    completed_pages: list[int],
    reused_pages: list[int],
    rendered_dir: Path,
    legacy_dir: Path,
    state_db_path: Path,
    progress_path: Path,
    summary_path: Path,
    provider_settings: WorkflowProviderSettings,
    state_store: OCRWorkflowStateStore,
    restore_grouping_checkpoint: bool = True,
    persist_grouping_checkpoint: bool = True,
    grouping_policy_version: str = "baseline-v1",
    summary_policy: SummaryPolicy | None = None,
    summary_generator: SummaryGenerator | None = None,
    grounded_summary_generator: GroundedSummaryGenerator | None = None,
) -> OCRWorkflowArtifacts:
    workflow_input: WorkflowIngestInput = normalize_ocr_pages(
        document_id=document_id,
        title=title,
        pages=raw_pages,
    )
    input_fingerprint = state_store.document_input_fingerprint(document_id=document_id)
    source_revision = (
        f"ocr-v1:{input_fingerprint}"
        if input_fingerprint
        else "ocr-v1"
    )
    document_tree = build_document_tree(workflow_input, source_revision=source_revision)
    if document_tree is not None:
        checkpointed_tree = (
            state_store.load_document_tree(
                document_id=document_id,
                source_fingerprint=document_tree.source_fingerprint(),
                grouping_policy_version=grouping_policy_version,
            )
            if restore_grouping_checkpoint and persist_grouping_checkpoint
            else None
        )
        if checkpointed_tree is not None:
            document_tree = checkpointed_tree
        else:
            document_tree = _apply_cross_page_continuations(
                document_tree,
                grouping_policy_version=grouping_policy_version,
            )
            if persist_grouping_checkpoint:
                state_store.save_document_tree(
                    document_id=document_id,
                    tree=document_tree,
                    grouping_policy_version=grouping_policy_version,
                    last_runtime_step="finalize_grouping_checkpoint",
                )
        if persist_grouping_checkpoint:
            existing_tree_revision = state_store.document_tree_revision(document_id=document_id)
            document_tree = _generate_tree_summaries(
                document_tree,
                document_id=document_id,
                state_store=state_store,
                policy=summary_policy,
                generator=summary_generator,
                grounded_generator=grounded_summary_generator,
                grouping_policy_version=grouping_policy_version,
                expected_tree_revision=existing_tree_revision,
                require_no_existing_tree=existing_tree_revision is None,
            )
        attach_document_tree_metadata(workflow_input, document_tree)
    artifacts = OCRWorkflowArtifacts(
        workflow_input=workflow_input,
        ocr_pages=raw_pages,
        legacy_dir=legacy_dir,
        rendered_dir=rendered_dir,
        state_db_path=state_db_path,
        progress_path=progress_path,
        summary_path=summary_path,
        completed_pages=completed_pages,
        reused_pages=reused_pages,
        document_tree=document_tree,
    )
    if document_tree is not None:
        _write_document_tree(legacy_dir / "document-tree.json", document_tree)
    _write_summary(
        artifacts,
        document_id=document_id,
        provider_settings=provider_settings,
        state_store=state_store,
    )
    return artifacts


class OCRWorkflowStateStore:
    """SQLite-backed OCR/render state for one artifact root."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @classmethod
    def open_or_rebuild(
        cls,
        *,
        db_path: Path,
        document_id: str,
        title: str,
        source_kind: str,
        total_pages: int,
        input_fingerprint: str,
        rendered_dir: Path,
        legacy_dir: Path,
        progress_path: Path,
        render_configuration_fingerprint: str = "render-v1",
    ) -> OCRWorkflowStateStore:
        store = cls(db_path)
        rebuilt = False
        if not db_path.exists() or store._is_empty():
            rebuilt = store.rebuild_from_artifacts(
                document_id=document_id,
                title=title,
                source_kind=source_kind,
                total_pages=total_pages,
                input_fingerprint=input_fingerprint,
                rendered_dir=rendered_dir,
                legacy_dir=legacy_dir,
                progress_path=progress_path,
                render_configuration_fingerprint=render_configuration_fingerprint,
            )
        else:
            store.ensure_document(
                document_id=document_id,
                title=title,
                source_kind=source_kind,
                total_pages=total_pages,
                input_fingerprint=input_fingerprint,
            )
        if rebuilt:
            _LOGGER.info("ocr state rebuilt from artifacts | db=%s", db_path)
        return store

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    @contextlib.contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._session() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS document_state (
                    document_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    input_fingerprint TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    total_pages INTEGER NOT NULL,
                    is_completed INTEGER NOT NULL DEFAULT 0,
                    last_updated_ts REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS page_state (
                    document_id TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    stage TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'pending',
                    content_hash TEXT,
                    artifact_path TEXT,
                    last_error TEXT,
                    last_model TEXT,
                    config_fingerprint TEXT,
                    artifact_sha256 TEXT,
                    last_attempted_ts REAL,
                    PRIMARY KEY (document_id, page_number, stage)
                );

                CREATE TABLE IF NOT EXISTS model_attempts (
                    document_id TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    stage TEXT NOT NULL,
                    attempt_index INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    attempted_ts REAL NOT NULL,
                    PRIMARY KEY (document_id, page_number, stage, attempt_index)
                );

                CREATE TABLE IF NOT EXISTS document_tree_state (
                    document_id TEXT PRIMARY KEY,
                    tree_revision TEXT NOT NULL,
                    source_fingerprint TEXT NOT NULL,
                    grouping_policy_version TEXT NOT NULL,
                    tree_json TEXT NOT NULL,
                    last_updated_ts REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS grouping_frontier_state (
                    document_id TEXT PRIMARY KEY,
                    source_fingerprint TEXT NOT NULL,
                    grouping_policy_version TEXT NOT NULL,
                    frontier_json TEXT NOT NULL,
                    last_updated_ts REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS grouping_checkpoint_metadata (
                    document_id TEXT PRIMARY KEY,
                    source_fingerprint TEXT NOT NULL,
                    grouping_policy_version TEXT NOT NULL,
                    accepted_page_frontier_json TEXT NOT NULL,
                    tree_revision TEXT NOT NULL,
                    merge_revision INTEGER NOT NULL DEFAULT 0,
                    last_runtime_step TEXT NOT NULL,
                    last_updated_ts REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS summary_attempts (
                    document_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    input_fingerprint TEXT NOT NULL,
                    attempt_index INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    summary_json TEXT,
                    error_message TEXT,
                    last_updated_ts REAL NOT NULL,
                    PRIMARY KEY (document_id, node_id, input_fingerprint, attempt_index)
                );
                """
            )
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(page_state)").fetchall()}
            if "config_fingerprint" not in columns:
                conn.execute("ALTER TABLE page_state ADD COLUMN config_fingerprint TEXT")
            if "artifact_sha256" not in columns:
                conn.execute("ALTER TABLE page_state ADD COLUMN artifact_sha256 TEXT")
            checkpoint_columns = {
                str(row[1])
                for row in conn.execute(
                    "PRAGMA table_info(grouping_checkpoint_metadata)"
                ).fetchall()
            }
            if "merge_revision" not in checkpoint_columns:
                conn.execute(
                    "ALTER TABLE grouping_checkpoint_metadata "
                    "ADD COLUMN merge_revision INTEGER NOT NULL DEFAULT 0"
                )

    def _is_empty(self) -> bool:
        with self._session() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM document_state").fetchone()
        return bool(row is None or int(row["count"]) == 0)

    def document_input_fingerprint(self, *, document_id: str) -> str | None:
        """Return the content identity used to build the current OCR state."""
        with self._session() as conn:
            row = conn.execute(
                "SELECT input_fingerprint FROM document_state WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return None if row is None else str(row["input_fingerprint"])

    def ensure_document(
        self,
        *,
        document_id: str,
        title: str,
        source_kind: str,
        total_pages: int,
        input_fingerprint: str,
    ) -> None:
        now = time.time()
        with self._session() as conn:
            existing = conn.execute(
                "SELECT input_fingerprint FROM document_state WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            if existing is not None and str(existing["input_fingerprint"]) != input_fingerprint:
                conn.execute("DELETE FROM model_attempts WHERE document_id = ?", (document_id,))
                conn.execute("DELETE FROM page_state WHERE document_id = ?", (document_id,))
                conn.execute("DELETE FROM document_tree_state WHERE document_id = ?", (document_id,))
                conn.execute("DELETE FROM grouping_frontier_state WHERE document_id = ?", (document_id,))
                conn.execute("DELETE FROM grouping_checkpoint_metadata WHERE document_id = ?", (document_id,))
                conn.execute("DELETE FROM summary_attempts WHERE document_id = ?", (document_id,))
                conn.execute("DELETE FROM document_state WHERE document_id = ?", (document_id,))
            conn.execute(
                """
                INSERT INTO document_state (
                    document_id, title, source_kind, input_fingerprint,
                    schema_version, total_pages, is_completed, last_updated_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    title = excluded.title,
                    source_kind = excluded.source_kind,
                    input_fingerprint = excluded.input_fingerprint,
                    schema_version = excluded.schema_version,
                    total_pages = excluded.total_pages,
                    last_updated_ts = excluded.last_updated_ts
                """,
                (
                    document_id,
                    title,
                    source_kind,
                    input_fingerprint,
                    _OCR_STATE_SCHEMA_VERSION,
                    total_pages,
                    0,
                    now,
                ),
            )

    def save_document_tree(
        self,
        *,
        document_id: str,
        tree: DocumentTree,
        grouping_policy_version: str = "baseline-v1",
        expected_tree_revision: str | None = None,
        require_no_existing_tree: bool = False,
        last_runtime_step: str = "grouping_checkpoint",
    ) -> str:
        """Persist a validated tree with an optional atomic revision fence.

        A caller that observed no checkpoint can use
        ``require_no_existing_tree`` to prevent a concurrent first writer from
        being silently overwritten.
        """
        if require_no_existing_tree and expected_tree_revision is not None:
            raise ValueError("absence and expected revision fences are mutually exclusive")
        if not last_runtime_step:
            raise ValueError("last_runtime_step must not be empty")
        tree_json = json.dumps(tree.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        tree_revision = hashlib.sha256(tree_json.encode("utf-8")).hexdigest()
        with self._session() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                """
                SELECT tree_revision, source_fingerprint, grouping_policy_version
                FROM document_tree_state
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()
            current_frontier = conn.execute(
                "SELECT document_id FROM grouping_frontier_state WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            current_metadata = conn.execute(
                "SELECT document_id, merge_revision "
                "FROM grouping_checkpoint_metadata WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            current_revision = None if current is None else str(current["tree_revision"])
            has_frontier = current_frontier is not None
            # A process can die after this SQLite commit but before the
            # surrounding Kogwistar step checkpoint.  A retry with the same
            # exact tree is safe to acknowledge; a different stale tree must
            # still fail the revision fence below.
            if (
                current_revision == tree_revision
                and has_frontier
                and current_metadata is not None
                and current is not None
                and str(current["source_fingerprint"]) == tree.source_fingerprint()
                and str(current["grouping_policy_version"]) == grouping_policy_version
            ):
                return tree_revision
            merge_revision = (
                1
                if current_metadata is None
                else int(current_metadata["merge_revision"]) + 1
            )
            if require_no_existing_tree and (current_revision is not None or has_frontier):
                raise GroupingCheckpointConflict(
                    f"grouping checkpoint create conflict for {document_id!r}: "
                    f"found existing revision {current_revision!r} or frontier"
                )
            if current_revision is None and has_frontier:
                raise GroupingCheckpointConflict(
                    f"grouping checkpoint create conflict for {document_id!r}: "
                    "found orphaned frontier without a tree"
                )
            if expected_tree_revision is not None and current_revision != expected_tree_revision:
                raise GroupingCheckpointConflict(
                    f"grouping checkpoint revision conflict for {document_id!r}: "
                    f"expected {expected_tree_revision!r}, found {current_revision!r}"
                )
            if expected_tree_revision is not None and not has_frontier:
                raise GroupingCheckpointConflict(
                    f"grouping checkpoint frontier is missing for {document_id!r}"
                )
            if current_revision is not None and expected_tree_revision is None:
                raise GroupingCheckpointConflict(
                    f"grouping checkpoint update for {document_id!r} requires "
                    "an expected tree revision"
                )
            if expected_tree_revision is None and current_revision is not None and not has_frontier:
                raise GroupingCheckpointConflict(
                    f"grouping checkpoint frontier is missing for {document_id!r}"
                )
            conn.execute(
                """
                INSERT INTO document_tree_state (
                    document_id, tree_revision, source_fingerprint,
                    grouping_policy_version, tree_json, last_updated_ts
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    tree_revision = excluded.tree_revision,
                    source_fingerprint = excluded.source_fingerprint,
                    grouping_policy_version = excluded.grouping_policy_version,
                    tree_json = excluded.tree_json,
                    last_updated_ts = excluded.last_updated_ts
                """,
                (
                    document_id,
                    tree_revision,
                    tree.source_fingerprint(),
                    grouping_policy_version,
                    tree_json,
                    time.time(),
                ),
            )
            root = next(node for node in tree.nodes if node.node_id == tree.root_id)
            frontier = (
                tuple(child.node_id for child in root.children)
                if root.children
                else (root.node_id,)
            )
            conn.execute(
                """
                INSERT INTO grouping_frontier_state (
                    document_id, source_fingerprint, grouping_policy_version,
                    frontier_json, last_updated_ts
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    source_fingerprint = excluded.source_fingerprint,
                    grouping_policy_version = excluded.grouping_policy_version,
                    frontier_json = excluded.frontier_json,
                    last_updated_ts = excluded.last_updated_ts
                """,
                (
                    document_id,
                    tree.source_fingerprint(),
                    grouping_policy_version,
                    json.dumps(frontier, separators=(",", ":")),
                    time.time(),
                ),
            )
            accepted_pages = tuple(
                int(row["page_number"])
                for row in conn.execute(
                    """
                    SELECT page_number
                    FROM page_state
                    WHERE document_id = ? AND stage = 'ocr' AND status = 'completed'
                    ORDER BY page_number
                    """,
                    (document_id,),
                ).fetchall()
            )
            conn.execute(
                """
                INSERT INTO grouping_checkpoint_metadata (
                    document_id, source_fingerprint, grouping_policy_version,
                    accepted_page_frontier_json, tree_revision, merge_revision,
                    last_runtime_step, last_updated_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    source_fingerprint = excluded.source_fingerprint,
                    grouping_policy_version = excluded.grouping_policy_version,
                    accepted_page_frontier_json = excluded.accepted_page_frontier_json,
                    tree_revision = excluded.tree_revision,
                    merge_revision = excluded.merge_revision,
                    last_runtime_step = excluded.last_runtime_step,
                    last_updated_ts = excluded.last_updated_ts
                """,
                (
                    document_id,
                    tree.source_fingerprint(),
                    grouping_policy_version,
                    json.dumps(accepted_pages, separators=(",", ":")),
                    tree_revision,
                    merge_revision,
                    last_runtime_step,
                    time.time(),
                ),
            )
        return tree_revision

    def load_grouping_checkpoint_metadata(
        self,
        *,
        document_id: str,
        source_fingerprint: str | None = None,
        grouping_policy_version: str | None = None,
    ) -> GroupingCheckpointMetadata | None:
        """Load the atomically committed identity for a grouping checkpoint."""
        with self._session() as conn:
            row = conn.execute(
                """
                SELECT source_fingerprint, grouping_policy_version,
                       accepted_page_frontier_json, tree_revision, merge_revision,
                       last_runtime_step
                FROM grouping_checkpoint_metadata
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()
        if row is None:
            return None
        stored_source = str(row["source_fingerprint"])
        stored_policy = str(row["grouping_policy_version"])
        if source_fingerprint is not None and stored_source != source_fingerprint:
            return None
        if grouping_policy_version is not None and stored_policy != grouping_policy_version:
            return None
        payload = json.loads(str(row["accepted_page_frontier_json"]))
        if not isinstance(payload, list) or any(
            type(page) is not int or page < 1 for page in payload
        ):
            raise ValueError("persisted grouping checkpoint page frontier is malformed")
        if payload != sorted(set(payload)):
            raise ValueError("persisted grouping checkpoint page frontier is not ordered")
        return GroupingCheckpointMetadata(
            document_id=document_id,
            source_fingerprint=stored_source,
            grouping_policy_version=stored_policy,
            accepted_page_frontier=tuple(payload),
            tree_revision=str(row["tree_revision"]),
            merge_revision=int(row["merge_revision"]),
            last_runtime_step=str(row["last_runtime_step"]),
        )

    def document_tree_revision(self, *, document_id: str) -> str | None:
        """Return the current tree digest without applying source/policy filters."""
        with self._session() as conn:
            row = conn.execute(
                "SELECT tree_revision FROM document_tree_state WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return None if row is None else str(row["tree_revision"])

    def load_document_tree(
        self,
        *,
        document_id: str,
        source_fingerprint: str | None = None,
        grouping_policy_version: str | None = None,
    ) -> DocumentTree | None:
        """Load only a validated tree matching the current source identity."""
        with self._session() as conn:
            row = conn.execute(
                """
                SELECT source_fingerprint, grouping_policy_version, tree_json
                FROM document_tree_state
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()
        if (
            row is None
            or (source_fingerprint is not None and str(row["source_fingerprint"]) != source_fingerprint)
            or (
                grouping_policy_version is not None
                and str(row["grouping_policy_version"]) != grouping_policy_version
            )
        ):
            return None
        return DocumentTree.model_validate_json(str(row["tree_json"]))

    def load_grouping_frontier(
        self,
        *,
        document_id: str,
        source_fingerprint: str | None = None,
        grouping_policy_version: str | None = None,
    ) -> tuple[str, ...] | None:
        """Load an ordered frontier only when its source and policy still match."""
        with self._session() as conn:
            row = conn.execute(
                """
                SELECT source_fingerprint, grouping_policy_version, frontier_json
                FROM grouping_frontier_state
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()
        if (
            row is None
            or (source_fingerprint is not None and str(row["source_fingerprint"]) != source_fingerprint)
            or (
                grouping_policy_version is not None
                and str(row["grouping_policy_version"]) != grouping_policy_version
            )
        ):
            return None
        payload = json.loads(str(row["frontier_json"]))
        if not isinstance(payload, list) or not all(isinstance(item, str) and item for item in payload):
            raise ValueError("persisted grouping frontier is malformed")
        return tuple(payload)

    def rebuild_from_artifacts(
        self,
        *,
        document_id: str,
        title: str,
        source_kind: str,
        total_pages: int,
        input_fingerprint: str,
        rendered_dir: Path,
        legacy_dir: Path,
        progress_path: Path,
        render_configuration_fingerprint: str = "render-v1",
    ) -> bool:
        self.ensure_document(
            document_id=document_id,
            title=title,
            source_kind=source_kind,
            total_pages=total_pages,
            input_fingerprint=input_fingerprint,
        )
        rebuilt_any = False
        progress_payload: OCRWorkflowProgress | None = None
        if progress_path.exists():
            raw_progress = _read_json_object(progress_path)
            try:
                progress_payload = OCRWorkflowProgress.model_validate(raw_progress)
            except ValueError:
                # A damaged progress report is advisory only. The SQLite
                # rebuild must continue without trusting unvalidated fields.
                progress_payload = None
        progress_pages: dict[str, OCRPageProgress] = (
            progress_payload.pages if progress_payload is not None else {}
        )
        expected_page_numbers = set(range(1, total_pages + 1))
        for page_number, path in _scan_rendered_pages(rendered_dir):
            # A rebuilt state store may share an artifact directory with an
            # older, longer document revision. Never resurrect removed pages.
            if page_number not in expected_page_numbers:
                continue
            rebuilt_any = True
            self._upsert_page_state(
                document_id=document_id,
                page_number=page_number,
                stage="render",
                attempt_count=1,
                status="completed",
                content_hash=_sha256_file(path),
                artifact_path=str(path),
                last_error=None,
                last_model=None,
                config_fingerprint=render_configuration_fingerprint,
            )
        for page_number, json_path in _scan_legacy_json_pages(legacy_dir):
            if page_number not in expected_page_numbers:
                continue
            content_hash = None
            record = progress_pages.get(str(page_number))
            if record is not None and _is_sha256_digest(record.image_sha256):
                content_hash = record.image_sha256
            if content_hash is None:
                content_hash = _find_matching_render_hash(rendered_dir, legacy_dir, page_number)
            config_fingerprint: str | None = None
            artifact_valid = False
            artifact_error: str | None = None
            try:
                persisted = _read_json_object(json_path)
                SplitPage.model_validate(persisted)
                artifact_valid = True
                metadata = _ocr_artifact_metadata(persisted)
                config_fingerprint = metadata.get("ocr_configuration_fingerprint")
                source_digest = metadata.get("source_image_sha256")
                if source_digest != content_hash:
                    artifact_valid = False
                    artifact_error = (
                        "OCR artifact source-image digest does not match the "
                        "current rendered page"
                    )
            except (OSError, ValueError) as exc:
                artifact_error = f"invalid OCR artifact during state rebuild: {exc}"
            if content_hash is None:
                artifact_valid = False
                artifact_error = artifact_error or "OCR artifact has no authoritative source-image digest"
            rebuilt_any = True
            self._upsert_page_state(
                document_id=document_id,
                page_number=page_number,
                stage="ocr",
                attempt_count=1,
                status="completed" if artifact_valid else "pending",
                content_hash=content_hash,
                artifact_path=str(json_path),
                last_error=artifact_error,
                last_model=None,
                config_fingerprint=config_fingerprint,
                artifact_sha256=_sha256_file(json_path) if artifact_valid else None,
            )
        self.refresh_document_completion(document_id=document_id)
        return rebuilt_any

    def _upsert_page_state(
        self,
        *,
        document_id: str,
        page_number: int,
        stage: str,
        attempt_count: int,
        status: str,
        content_hash: str | None,
        artifact_path: str | None,
        last_error: str | None,
        last_model: str | None,
        config_fingerprint: str | None = None,
        artifact_sha256: str | None = None,
    ) -> None:
        now = time.time()
        with self._session() as conn:
            conn.execute(
                """
                INSERT INTO page_state (
                    document_id, page_number, stage, attempt_count, status,
                    content_hash, artifact_path, last_error, last_model,
                    config_fingerprint, artifact_sha256, last_attempted_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id, page_number, stage) DO UPDATE SET
                    attempt_count = excluded.attempt_count,
                    status = excluded.status,
                    content_hash = excluded.content_hash,
                    artifact_path = excluded.artifact_path,
                    last_error = excluded.last_error,
                    last_model = excluded.last_model,
                    config_fingerprint = excluded.config_fingerprint,
                    artifact_sha256 = excluded.artifact_sha256,
                    last_attempted_ts = excluded.last_attempted_ts
                WHERE page_state.status <> 'completed'
                   OR (
                       page_state.content_hash IS excluded.content_hash
                       AND page_state.config_fingerprint IS excluded.config_fingerprint
                       AND page_state.artifact_sha256 IS excluded.artifact_sha256
                   )
                """,
                (
                    document_id,
                    page_number,
                    stage,
                    attempt_count,
                    status,
                    content_hash,
                    artifact_path,
                    last_error,
                    last_model,
                    config_fingerprint,
                    artifact_sha256,
                    now,
                ),
            )

    def get_page_state(self, *, document_id: str, page_number: int, stage: str) -> OCRPageStateSnapshot | None:
        with self._session() as conn:
            row = conn.execute(
                """
                SELECT page_number, stage, attempt_count, status, content_hash, artifact_path,
                       last_error, last_model, config_fingerprint, artifact_sha256,
                       last_attempted_ts
                FROM page_state
                WHERE document_id = ? AND page_number = ? AND stage = ?
                """,
                (document_id, page_number, stage),
            ).fetchone()
        if row is None:
            return None
        return OCRPageStateSnapshot(
            page_number=int(row["page_number"]),
            stage=str(row["stage"]),
            attempt_count=int(row["attempt_count"]),
            status=str(row["status"]),
            content_hash=row["content_hash"],
            artifact_path=row["artifact_path"],
            last_error=row["last_error"],
            last_model=row["last_model"],
            last_attempted_ts=row["last_attempted_ts"],
            config_fingerprint=row["config_fingerprint"],
            artifact_sha256=row["artifact_sha256"],
        )

    def should_skip_page(
        self,
        *,
        document_id: str,
        page_number: int,
        stage: str,
        content_hash: str,
        artifact_path: Path | None = None,
        config_fingerprint: str | Sequence[str] | None = None,
    ) -> bool:
        state = self.get_page_state(document_id=document_id, page_number=page_number, stage=stage)
        if state is None or state.status != "completed" or state.content_hash != content_hash:
            return False
        expected = {config_fingerprint} if isinstance(config_fingerprint, str) else set(config_fingerprint or ())
        if expected and state.config_fingerprint not in expected:
            return False
        if artifact_path is None:
            return True
        if not artifact_path.is_file():
            return False
        return state.artifact_sha256 is not None and _sha256_file(artifact_path) == state.artifact_sha256

    def reconcile_orphan_artifact(
        self,
        *,
        document_id: str,
        page_number: int,
        content_hash: str,
        artifact_path: Path,
        config_fingerprints: Sequence[str],
    ) -> bool:
        """Adopt an artifact published before its SQLite completion commit.

        The artifact must carry its source and configuration identity. A
        well-formed JSON file without those identities is never promoted from
        the orphan state, because it could belong to another source or OCR
        configuration. Existing completed rows are never repaired here; the
        normal digest check must decide whether they are still reusable.
        """
        state = self.get_page_state(document_id=document_id, page_number=page_number, stage="ocr")
        if state is not None and state.status == "completed":
            return False
        if not artifact_path.is_file():
            return False
        persisted = _read_json_object(artifact_path)
        try:
            split_page = SplitPage.model_validate(persisted)
        except ValueError:
            return False
        if split_page.pdf_page_num != page_number:
            return False
        metadata = _ocr_artifact_metadata(persisted)
        source_digest = metadata.get("source_image_sha256")
        config_fingerprint = metadata.get("ocr_configuration_fingerprint")
        if source_digest != content_hash or not isinstance(config_fingerprint, str):
            return False
        if config_fingerprint not in set(config_fingerprints):
            return False
        model_name = metadata.get("ocr_model_name")
        self._upsert_page_state(
            document_id=document_id,
            page_number=page_number,
            stage="ocr",
            attempt_count=state.attempt_count if state is not None else 1,
            status="completed",
            content_hash=content_hash,
            artifact_path=str(artifact_path),
            last_error=None,
            last_model=model_name if isinstance(model_name, str) else None,
            config_fingerprint=config_fingerprint,
            artifact_sha256=_sha256_file(artifact_path),
        )
        return True

    def record_attempt(
        self,
        *,
        document_id: str,
        page_number: int,
        stage: str,
        content_hash: str | None,
        model_name: str,
        artifact_path: Path | None = None,
        config_fingerprint: str | None = None,
    ) -> int:
        now = time.time()
        with self._session() as conn:
            # Serialize the read/allocate/insert sequence so duplicate
            # deliveries receive distinct attempt identities.
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT attempt_count, status, content_hash, artifact_path,
                       artifact_sha256, config_fingerprint
                FROM page_state
                WHERE document_id = ? AND page_number = ? AND stage = ?
                """,
                (document_id, page_number, stage),
            ).fetchone()
            if (
                row is not None
                and row["status"] == "completed"
                and row["content_hash"] == content_hash
                and row["config_fingerprint"] == config_fingerprint
                and row["artifact_path"] is not None
                and row["artifact_sha256"] is not None
                and Path(str(row["artifact_path"])).is_file()
                and _sha256_file(Path(str(row["artifact_path"]))) == row["artifact_sha256"]
            ):
                # A duplicate delivery raced past should_skip_page while the
                # winner was committing. Keep the completed row authoritative
                # instead of reopening it as pending.
                return int(row["attempt_count"])
            attempt_count = int(row["attempt_count"]) + 1 if row is not None else 1
            conn.execute(
                """
                INSERT INTO page_state (
                    document_id, page_number, stage, attempt_count, status,
                    content_hash, artifact_path, last_error, last_model,
                    config_fingerprint, artifact_sha256, last_attempted_ts
                ) VALUES (?, ?, ?, ?, 'pending', ?, ?, NULL, ?, ?, NULL, ?)
                ON CONFLICT(document_id, page_number, stage) DO UPDATE SET
                    attempt_count = excluded.attempt_count,
                    status = excluded.status,
                    content_hash = excluded.content_hash,
                    artifact_path = excluded.artifact_path,
                    last_error = excluded.last_error,
                    last_model = excluded.last_model,
                    config_fingerprint = excluded.config_fingerprint,
                    artifact_sha256 = NULL,
                    last_attempted_ts = excluded.last_attempted_ts
                """,
                (
                    document_id,
                    page_number,
                    stage,
                    attempt_count,
                    content_hash,
                    str(artifact_path) if artifact_path is not None else None,
                    model_name,
                    config_fingerprint,
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO model_attempts (
                    document_id, page_number, stage, attempt_index, model_name, status, error_message, attempted_ts
                ) VALUES (?, ?, ?, ?, ?, 'pending', NULL, ?)
                """,
                (document_id, page_number, stage, attempt_count, model_name, now),
            )
        return attempt_count

    def publish_ocr_page(
        self,
        *,
        document_id: str,
        page_number: int,
        content_hash: str,
        artifact_path: Path,
        payload: object,
        model_name: str,
        attempt_index: int,
        config_fingerprint: str,
    ) -> bool:
        """Atomically accept one OCR result for a pending page attempt.

        Provider calls remain at-least-once, but the SQLite write lock makes
        publication and acceptance one serialized decision. A later duplicate
        delivery observes the completed row and must reuse the winner rather
        than replacing its artifact.
        """
        with self._session() as conn:
            conn.execute("BEGIN IMMEDIATE")
            state = conn.execute(
                """
                SELECT status, content_hash, config_fingerprint, attempt_count
                FROM page_state
                WHERE document_id = ? AND page_number = ? AND stage = 'ocr'
                """,
                (document_id, page_number),
            ).fetchone()
            if (
                state is None
                or state["status"] != "pending"
                or state["content_hash"] != content_hash
                or state["config_fingerprint"] != config_fingerprint
            ):
                return False

            _write_json_atomic(artifact_path, payload)
            artifact_digest = _sha256_file(artifact_path)
            updated = conn.execute(
                """
                UPDATE page_state
                SET status = 'completed',
                    artifact_path = ?,
                    last_error = NULL,
                    last_model = ?,
                    config_fingerprint = ?,
                    artifact_sha256 = ?,
                    last_attempted_ts = ?
                WHERE document_id = ? AND page_number = ? AND stage = 'ocr'
                  AND status = 'pending'
                  AND content_hash = ?
                  AND config_fingerprint = ?
                """,
                (
                    str(artifact_path),
                    model_name,
                    config_fingerprint,
                    artifact_digest,
                    time.time(),
                    document_id,
                    page_number,
                    content_hash,
                    config_fingerprint,
                ),
            )
            if updated.rowcount != 1:
                return False
            # Concurrent deliveries with the same source/configuration are
            # equivalent candidates. Once one is accepted, close any older
            # pending attempt records so they cannot look like live work on
            # the next resume. Different configurations remain fenced above.
            conn.execute(
                """
                UPDATE model_attempts
                SET status = 'failed', error_message = 'superseded by accepted duplicate'
                WHERE document_id = ? AND page_number = ? AND stage = 'ocr'
                  AND status = 'pending' AND attempt_index <> ?
                """,
                (document_id, page_number, attempt_index),
            )
            conn.execute(
                """
                UPDATE model_attempts
                SET status = 'completed', error_message = NULL
                WHERE document_id = ? AND page_number = ? AND stage = 'ocr' AND attempt_index = ?
                """,
                (document_id, page_number, attempt_index),
            )
            return True

    def record_page_completed(
        self,
        *,
        document_id: str,
        page_number: int,
        stage: str,
        content_hash: str | None,
        artifact_path: Path | None,
        model_name: str | None,
        attempt_index: int | None = None,
        config_fingerprint: str | None = None,
    ) -> bool:
        # An OCR completion is not authoritative without its serialized page
        # artifact. Keep this legacy completion seam from creating a
        # database-first false positive; the canonical publisher already
        # writes the artifact before accepting the SQLite row.
        if stage == "ocr":
            if artifact_path is None or not artifact_path.is_file():
                return False
            try:
                SplitPage.model_validate(_read_json_object(artifact_path))
            except (TypeError, ValueError):
                # Compatibility callers must not turn malformed JSON into an
                # accepted page; the canonical publisher applies the same
                # schema gate before recording completion.
                return False
        now = time.time()
        with self._session() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                """
                SELECT status, content_hash, artifact_path, config_fingerprint,
                       artifact_sha256, attempt_count
                FROM page_state
                WHERE document_id = ? AND page_number = ? AND stage = ?
                """,
                (document_id, page_number, stage),
            ).fetchone()
            if existing is None:
                return False
            if existing["status"] == "completed":
                # Duplicate delivery is idempotent only when it describes the
                # already accepted result. A late writer must not overwrite it.
                return bool(
                    existing["content_hash"] == content_hash
                    and existing["config_fingerprint"] == config_fingerprint
                    and existing["artifact_path"]
                    == (str(artifact_path) if artifact_path is not None else None)
                    and (
                        attempt_index is None
                        or existing["attempt_count"] == attempt_index
                    )
                )
            if existing["status"] != "pending":
                return False
            updated = conn.execute(
                """
                UPDATE page_state
                SET status = 'completed',
                    content_hash = ?,
                    artifact_path = ?,
                    last_error = NULL,
                    last_model = ?,
                    config_fingerprint = ?,
                    artifact_sha256 = ?,
                    last_attempted_ts = ?
                WHERE document_id = ? AND page_number = ? AND stage = ?
                  AND status = 'pending'
                  AND content_hash IS ?
                  AND config_fingerprint IS ?
                  AND (? IS NULL OR attempt_count = ?)
                """,
                (
                    content_hash,
                    str(artifact_path) if artifact_path is not None else None,
                    model_name,
                    config_fingerprint,
                    _sha256_file(artifact_path) if artifact_path is not None and artifact_path.is_file() else None,
                    now,
                    document_id,
                    page_number,
                    stage,
                    content_hash,
                    config_fingerprint,
                    attempt_index,
                    attempt_index,
                ),
            )
            if updated.rowcount != 1:
                return False
            if attempt_index is not None:
                conn.execute(
                    """
                    UPDATE model_attempts
                    SET status = 'completed', error_message = NULL
                    WHERE document_id = ? AND page_number = ? AND stage = ? AND attempt_index = ?
                    """,
                    (document_id, page_number, stage, attempt_index),
                )
            return True

    def record_page_failed(
        self,
        *,
        document_id: str,
        page_number: int,
        stage: str,
        content_hash: str | None,
        config_fingerprint: str | None,
        error_message: str,
        model_name: str | None,
        attempt_index: int | None = None,
    ) -> bool:
        """Record failure only for the still-owned pending attempt.

        Provider calls are at-least-once. A late failure from an older worker
        must not downgrade an accepted page or replace a newer pending retry.
        The attempt count, source hash, configuration fingerprint, and pending
        status together form the completion ownership fence.
        """
        now = time.time()
        with self._session() as conn:
            updated = conn.execute(
                """
                UPDATE page_state
                SET status = 'failed',
                    content_hash = ?,
                    last_error = ?,
                    last_model = ?,
                    last_attempted_ts = ?
                WHERE document_id = ? AND page_number = ? AND stage = ?
                  AND status = 'pending'
                  AND content_hash IS ?
                  AND config_fingerprint IS ?
                  AND (? IS NULL OR attempt_count = ?)
                """,
                (
                    content_hash,
                    error_message,
                    model_name,
                    now,
                    document_id,
                    page_number,
                    stage,
                    content_hash,
                    config_fingerprint,
                    attempt_index,
                    attempt_index,
                ),
            )
            updated_count = updated.rowcount
            if updated_count == 1 and attempt_index is not None:
                conn.execute(
                    """
                    UPDATE model_attempts
                    SET status = 'failed', error_message = ?
                    WHERE document_id = ? AND page_number = ? AND stage = ? AND attempt_index = ?
                    """,
                    (error_message, document_id, page_number, stage, attempt_index),
                )
            return updated_count == 1

    def mark_document_completed(self, *, document_id: str, is_completed: bool) -> None:
        with self._session() as conn:
            conn.execute(
                """
                UPDATE document_state
                SET is_completed = ?, last_updated_ts = ?
                WHERE document_id = ?
                """,
                (1 if is_completed else 0, time.time(), document_id),
            )

    def refresh_document_completion(self, *, document_id: str) -> bool:
        with self._session() as conn:
            row = conn.execute(
                """
                SELECT total_pages
                FROM document_state
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()
            total_pages = int(row["total_pages"]) if row is not None else 0
            completed_rows = conn.execute(
                """
                SELECT page_number
                FROM page_state
                WHERE document_id = ? AND stage = 'ocr' AND status = 'completed'
                """,
                (document_id,),
            ).fetchall()
        completed_pages = {int(completed_row["page_number"]) for completed_row in completed_rows}
        expected_pages = set(range(1, total_pages + 1))
        is_completed = total_pages > 0 and completed_pages == expected_pages
        self.mark_document_completed(document_id=document_id, is_completed=is_completed)
        return is_completed

    def list_model_attempts(self, *, document_id: str, page_number: int, stage: str = "ocr") -> list[JsonObject]:
        with self._session() as conn:
            rows = conn.execute(
                """
                SELECT attempt_index, model_name, status, error_message, attempted_ts
                FROM model_attempts
                WHERE document_id = ? AND page_number = ? AND stage = ?
                ORDER BY attempt_index
                """,
                (document_id, page_number, stage),
            ).fetchall()
        return [dict(row) for row in rows]

    def record_summary_attempt(
        self,
        *,
        document_id: str,
        node_id: str,
        input_fingerprint: str,
        status: str,
        summary: Summary | None = None,
        error_message: str | None = None,
    ) -> int:
        """Journal one idempotent summary attempt for crash-safe retries."""
        if status not in {"pending", "generated", "verified", "failed", "skipped"}:
            raise ValueError(f"unsupported summary status: {status}")
        summary_json = (
            json.dumps(summary.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
            if summary is not None
            else None
        )
        with self._session() as conn:
            if status in {"generated", "verified"}:
                # Accepted summaries are canonical for an exact generation
                # identity. Serialize the check with the insert so duplicate
                # delivery cannot create a second accepted journal row.
                conn.execute("BEGIN IMMEDIATE")
                accepted = conn.execute(
                    """
                    SELECT attempt_index
                    FROM summary_attempts
                    WHERE document_id = ? AND node_id = ?
                      AND input_fingerprint = ?
                      AND status IN ('generated', 'verified')
                    ORDER BY attempt_index DESC
                    LIMIT 1
                    """,
                    (document_id, node_id, input_fingerprint),
                ).fetchone()
                if accepted is not None:
                    return int(accepted["attempt_index"])
            row = conn.execute(
                """
                SELECT COALESCE(MAX(attempt_index), 0) AS last_attempt
                FROM summary_attempts
                WHERE document_id = ? AND node_id = ? AND input_fingerprint = ?
                """,
                (document_id, node_id, input_fingerprint),
            ).fetchone()
            attempt_index = int(row["last_attempt"]) + 1 if row is not None else 1
            conn.execute(
                """
                INSERT INTO summary_attempts (
                    document_id, node_id, input_fingerprint, attempt_index,
                    status, summary_json, error_message, last_updated_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    node_id,
                    input_fingerprint,
                    attempt_index,
                    status,
                    summary_json,
                    error_message,
                    time.time(),
                ),
            )
        return attempt_index

    def load_completed_summary(
        self,
        *,
        document_id: str,
        node_id: str,
        input_fingerprint: str,
    ) -> Summary | None:
        """Load the latest accepted summary for one exact generation key.

        Failed and pending attempts never satisfy this lookup. The JSON is
        validated through ``Summary`` before it can be reused, so malformed
        or incompatible journal rows become cache misses rather than accepted
        graph content.
        """
        with self._session() as conn:
            row = conn.execute(
                """
                SELECT summary_json
                FROM summary_attempts
                WHERE document_id = ? AND node_id = ? AND input_fingerprint = ?
                  AND status IN ('generated', 'verified')
                  AND summary_json IS NOT NULL
                ORDER BY attempt_index DESC
                LIMIT 1
                """,
                (document_id, node_id, input_fingerprint),
            ).fetchone()
        if row is None or row["summary_json"] is None:
            return None
        try:
            payload = json.loads(str(row["summary_json"]))
            return Summary.model_validate(payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    def list_summary_attempts(self, *, document_id: str) -> list[JsonObject]:
        with self._session() as conn:
            rows = conn.execute(
                """
                SELECT node_id, input_fingerprint, attempt_index, status,
                       summary_json, error_message, last_updated_ts
                FROM summary_attempts
                WHERE document_id = ?
                ORDER BY node_id, input_fingerprint, attempt_index
                """,
                (document_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def export_progress_payload(
        self,
        *,
        document_id: str,
        title: str,
        source_kind: str,
        total_pages: int,
    ) -> OCRWorkflowProgress:
        pages: dict[str, OCRPageProgress] = {}
        with self._session() as conn:
            rows = conn.execute(
                """
                SELECT page_number, content_hash, artifact_path, status
                FROM page_state
                WHERE document_id = ? AND stage = 'ocr'
                ORDER BY page_number
                """,
                (document_id,),
            ).fetchall()
        for row in rows:
            if row["artifact_path"] is None:
                continue
            pages[str(int(row["page_number"]))] = OCRPageProgress(
                page_number=int(row["page_number"]),
                image_path="",
                image_sha256=row["content_hash"] or "",
                json_path=str(row["artifact_path"]),
                status=str(row["status"]),
            )
        return OCRWorkflowProgress(
            document_id=document_id,
            title=title,
            source_kind=source_kind,
            total_pages=total_pages,
            pages=pages,
        )

    def read_document_completed(self, *, document_id: str) -> bool:
        with self._session() as conn:
            row = conn.execute(
                "SELECT is_completed FROM document_state WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return bool(row is not None and int(row["is_completed"]) == 1)


def _scan_rendered_pages(rendered_dir: Path) -> list[tuple[int, Path]]:
    pairs: list[tuple[int, Path]] = []
    if not rendered_dir.exists():
        return pairs
    for path in sorted(rendered_dir.glob("page_*")):
        stem = path.stem
        suffix = stem.rsplit("_", 1)
        if len(suffix) != 2 or not suffix[1].isdigit():
            continue
        pairs.append((int(suffix[1]), path))
    return pairs


def _scan_legacy_json_pages(legacy_dir: Path) -> list[tuple[int, Path]]:
    pairs: list[tuple[int, Path]] = []
    if not legacy_dir.exists():
        return pairs
    for path in sorted(legacy_dir.glob("page_*.json")):
        stem = path.stem
        suffix = stem.rsplit("_", 1)
        if len(suffix) != 2 or not suffix[1].isdigit():
            continue
        pairs.append((int(suffix[1]), path))
    return pairs


def _find_matching_render_hash(rendered_dir: Path, legacy_dir: Path, page_number: int) -> str | None:
    candidates = list(rendered_dir.glob(f"page_{page_number}.*")) + list(legacy_dir.glob(f"page_{page_number}.*"))
    for candidate in candidates:
        if candidate.suffix.lower() == ".json" or not candidate.exists():
            continue
        return _sha256_file(candidate)
    return None


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _image_to_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    suffix = path.suffix.lower()
    mime = "image/png"
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    return f"data:{mime};base64,{encoded}"


def _progress_bar(done: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return "." * width
    filled = min(width, max(0, round((done / total) * width)))
    return ("#" * filled) + ("." * (width - filled))


def _materialize_image_payloads(image_payloads: Sequence[OCRImagePayload], rendered_dir: Path) -> list[tuple[int, Path]]:
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
            src = Path(payload.image_path)
            suffix = src.suffix or ".png"
            dst = rendered_dir / f"page_{page_number}{suffix}"
            if dst.is_symlink():
                raise ValueError(f"OCR rendered page is a symlink: {dst.name}")
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            else:
                dst = src
        elif payload.image_bytes_b64:
            raw = base64.b64decode(payload.image_bytes_b64)
            requested_suffix = Path(payload.filename or "").suffix.lower()
            if requested_suffix in {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}:
                suffix = requested_suffix
            dst = rendered_dir / f"page_{page_number}{suffix}"
            if dst.is_symlink():
                raise ValueError(f"OCR rendered page is a symlink: {dst.name}")
            _write_bytes_atomic(dst, raw)
        else:
            raise ValueError("OCRImagePayload requires image_path or image_bytes_b64")
        resolved.append((page_number, dst))
    if supplied_page_numbers != sorted(supplied_page_numbers):
        raise ValueError("OCR page numbers must be supplied in ascending order")
    return resolved


def _compute_input_fingerprint(*, page_sources: Sequence[tuple[int, Path]] | None = None, pdf_path: Path | None = None) -> str:
    payload: JsonObject = {}
    if pdf_path is not None:
        payload["pdf_sha256"] = _sha256_file(pdf_path)
    if page_sources is not None:
        payload["pages"] = [
            {"page_number": page_number, "sha256": _sha256_file(Path(path))}
            for page_number, path in page_sources
        ]
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    page_count: int | None = None
    try:
        page_count = len(PdfReader(str(pdf_path)).pages)
    except (OSError, PdfReadError, PdfStreamError, ValueError):
        # Injected/fake rasterizers may intentionally use placeholder PDF bytes.
        # A valid PDF gets a count; an unreadable placeholder remains compatible.
        page_count = None
    config: _PDFRenderManifest = {
        "manifest_version": _PDF_RENDER_MANIFEST_VERSION,
        "pdf_sha256": _sha256_file(pdf_path),
        "renderer_version": _PDF_RENDERER_VERSION,
        "rasterizer": f"{rasterizer_module}.{rasterizer_name}",
        "rasterizer_configuration": configuration_fingerprint,
        "pdf_page_count": page_count,
    }
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    config["render_identity"] = hashlib.sha256(encoded).hexdigest()
    return config


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


def _builtin_pdf_rasterizer_configuration() -> str:
    """Describe every process-level option used by the built-in rasterizer.

    ``pdf2image`` defaults are part of the cache contract too.  Including the
    installed version and Poppler selection prevents a changed runtime from
    reusing images produced by a different renderer configuration.
    """
    try:
        pdf2image_version = metadata.version("pdf2image")
    except metadata.PackageNotFoundError:
        pdf2image_version = "unknown"
    poppler_path = os.environ.get("POPPLER_PATH", "")
    return json.dumps(
        {
            "dpi": 200,
            "fmt": "png",
            "rotation": 0,
            "pdf2image": pdf2image_version,
            "poppler_path": poppler_path,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


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
    if existing is not None and all(existing.get(key) == value for key, value in expected.items()):
        loaded_page_entries: list[_PDFRenderPageEntry] | None = existing.get("pages")
        if loaded_page_entries is not None:
            candidate_paths: list[Path] = []
            valid = True
            for entry in loaded_page_entries:
                filename = entry.get("filename")
                if not isinstance(filename, str) or Path(filename).name != filename:
                    valid = False
                    break
                candidate = rendered_dir / filename
                if (
                    candidate.is_symlink()
                    or not _image_file_is_valid(candidate)
                    or entry.get("sha256") != _sha256_file(candidate)
                ):
                    valid = False
                    break
                try:
                    with Image.open(candidate) as image:
                        dimensions = [image.width, image.height]
                except (OSError, SyntaxError, ValueError):
                    valid = False
                    break
                if entry.get("dimensions") != dimensions:
                    valid = False
                    break
                candidate_paths.append(candidate)
            expected_page_count_value = expected.get("pdf_page_count")
            expected_page_count = (
                expected_page_count_value
                if isinstance(expected_page_count_value, int)
                and not isinstance(expected_page_count_value, bool)
                else None
            )
            page_numbers = [entry["page_number"] for entry in loaded_page_entries]
            valid = valid and (
                expected_page_count is None
                or (
                    len(loaded_page_entries) == expected_page_count
                    and len(page_numbers) == len(loaded_page_entries)
                    and page_numbers == list(range(1, expected_page_count + 1))
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
            if path.is_symlink() or not path.is_file() or not _image_file_is_valid(path):
                raise RuntimeError(f"pdf rasterizer returned an invalid page image: {path}")
        page_entries: list[_PDFRenderPageEntry] = []
        for page_number, path in enumerate(paths, start=1):
            with Image.open(path) as image:
                dimensions = [image.width, image.height]
            page_entries.append(
                {
                    "page_number": page_number,
                    "filename": path.name,
                    "sha256": _sha256_file(path),
                    "dimensions": dimensions,
                }
            )
        complete_manifest: _PDFRenderManifest = {**expected, "pages": page_entries}
        _write_json_atomic(manifest_path, complete_manifest)
    return paths


def _write_progress(progress: OCRWorkflowProgress, progress_path: Path) -> None:
    _write_json_atomic(progress_path, progress.model_dump(mode="json"))


def _sync_progress_from_state(
    *,
    state_store: OCRWorkflowStateStore,
    document_id: str,
    title: str,
    source_kind: str,
    total_pages: int,
    progress_path: Path,
) -> OCRWorkflowProgress:
    progress = state_store.export_progress_payload(
        document_id=document_id,
        title=title,
        source_kind=source_kind,
        total_pages=total_pages,
    )
    _write_progress(progress, progress_path)
    return progress


def _render_pdf_to_images(pdf_path: Path, rendered_dir: Path) -> list[Path]:
    from pdf2image import convert_from_path

    rendered_dir.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(pdf_path))
    output_paths: list[Path] = []
    for page_number in range(1, len(reader.pages) + 1):
        output_path = rendered_dir / f"page_{page_number}.png"
        if output_path.is_symlink():
            raise ValueError(f"rendered PDF page is a symlink: {output_path.name}")
        if not output_path.is_file() or not _image_file_is_valid(output_path):
            images = convert_from_path(
                str(pdf_path),
                first_page=page_number,
                last_page=page_number,
                fmt="png",
            )
            if not images:
                raise RuntimeError(f"pdf rasterizer returned no image for page {page_number}")
            _write_png_atomic(images[0], output_path)
        output_paths.append(output_path)
    return output_paths


def _coerce_ocr_response(payload: object) -> OCRClusterResponse:
    """Coerce a structured-output payload into the OCR model."""
    if isinstance(payload, OCRClusterResponse):
        return payload
    if isinstance(payload, dict):
        parsed = payload.get("parsed", payload)
        if parsed is not None:
            return OCRClusterResponse.model_validate(parsed)
        raw = payload.get("raw")
        if raw is not None:
            content = raw_content(raw)
            if isinstance(content, str):
                return OCRClusterResponse.model_validate(json.loads(content))
            if isinstance(content, list):
                text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
                if text.strip():
                    return OCRClusterResponse.model_validate(json.loads(text))
    return OCRClusterResponse.model_validate(payload)


def _extract_message_text(raw: object) -> str:
    """Extract plain text from LangChain raw message content."""
    content = raw_content(raw)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def _minimal_ocr_response_from_text(*, text: str, page_number: int, image_path: Path) -> OCRClusterResponse:
    """Create a coarse page-wide OCR result when only raw text is available."""
    normalized_text = text.strip()
    with Image.open(image_path) as image:
        width, height = image.size
    if not normalized_text or normalized_text == "{}":
        return OCRClusterResponse(
            OCR_text_clusters=[],
            non_text_objects=[],
            is_empty_page=True,
            printed_page_number=str(page_number),
            meaningful_ordering=[],
            page_x_min=0.0,
            page_x_max=float(width),
            page_y_min=0.0,
            page_y_max=float(height),
            estimated_rotation_degrees=0.0,
            incomplete_words_on_edge=False,
            incomplete_text=False,
            data_loss_likelihood=0.0,
            scan_quality="medium",
            contains_table=False,
            spatial_accuracy="page",
        )
    return OCRClusterResponse(
        OCR_text_clusters=[
            TextCluster(
                text=normalized_text,
                bb_x_min=0.0,
                bb_x_max=float(width),
                bb_y_min=0.0,
                bb_y_max=float(height),
                cluster_number=0,
            )
        ],
        non_text_objects=[],
        is_empty_page=False,
        printed_page_number=str(page_number),
        meaningful_ordering=[0],
        page_x_min=0.0,
        page_x_max=float(width),
        page_y_min=0.0,
        page_y_max=float(height),
        estimated_rotation_degrees=0.0,
        incomplete_words_on_edge=False,
        incomplete_text=False,
        data_loss_likelihood=0.0,
        scan_quality="medium",
        contains_table=False,
        spatial_accuracy="page",
    )


def _run_live_ocr_page(
    image_path: Path,
    page_number: int,
    provider_settings: WorkflowProviderSettings,
    *,
    provider_diagnostics_sink: ProviderDiagnosticsSink | None = None,
) -> OCRClusterResponse:
    """Run one page image through the configured OCR provider.

    This is the default implementation behind the OCRRunner hook. Tests can
    replace it with a fake runner, but the callable contract stays the same:
    page image in, structured OCR model out.
    """
    if provider_settings.ocr.provider == "docling":
        from .ocr_providers import run_docling_ocr_page

        return run_docling_ocr_page(
            image_path,
            page_number,
            provider_settings,
            engine=provider_settings.ocr.ocr_engine or "rapidocr",
        )
    if provider_settings.ocr.provider == "rapidocr":
        from .ocr_providers import run_rapidocr_page

        return run_rapidocr_page(image_path, page_number, provider_settings.ocr)
    if provider_settings.ocr.provider == "easyocr":
        from .ocr_providers import run_easyocr_page

        return run_easyocr_page(image_path, page_number, provider_settings.ocr)
    if provider_settings.ocr.provider == "kserve":
        from .ocr_providers import run_kserve_v2_page

        return run_kserve_v2_page(image_path, page_number, provider_settings)
    if provider_settings.ocr.provider == "custom":
        from .ocr_providers import run_registered_ocr_page

        return run_registered_ocr_page(
            image_path,
            page_number,
            provider_settings,
            provider_name=provider_settings.ocr.model,
        )
    chat = build_chat_model_for_role("ocr", provider_settings)
    structured = build_structured_output_runnable(chat, OCRClusterResponse, include_raw=True)
    prompt = (
        "You are an OCR model for workflow ingest.\n"
        "Return structured OCR for one page image.\n"
        "Preserve reading order, cluster numbering, and non-text regions.\n"
        "Do not invent missing text. If the page is empty, mark it as empty."
    )
    call_diagnostics: ProviderDiagnostics = {}
    try:
        response = invoke_with_timeout(
            lambda: structured.invoke(
                [
                    SystemMessage(content=prompt),
                    HumanMessage(
                        content=[
                            {"type": "text", "text": f"OCR page {page_number} and return the structured schema."},
                            {"type": "image_url", "image_url": {"url": _image_to_data_url(image_path)}},
                        ]
                    ),
                ]
            ),
            timeout_seconds=provider_settings.ocr.timeout_seconds,
            diagnostics=call_diagnostics,
            operation="ocr_page",
            max_in_flight=provider_settings.ocr.max_in_flight_calls,
            attempt_index=1,
            call_role="ocr",
            strategy="ocr",
        )
    except Exception as exc:
        call_diagnostics.update(
            {
                "success": False,
                "failure_type": call_diagnostics.get("failure_type", "transport/provider_exception"),
                "error_type": type(exc).__name__,
            }
        )
        if provider_diagnostics_sink is not None:
            provider_diagnostics_sink(dict(call_diagnostics))
        raise
    try:
        result = _coerce_ocr_response(response)
    except Exception as exc:
        if isinstance(response, dict):
            raw = response.get("raw")
            parsing_error = response.get("parsing_error")
            raw_text = _extract_message_text(raw) if raw is not None else ""
            if raw_text:
                _LOGGER.info(
                    "ocr structured parse fallback | provider=%s model=%s page=%s | parsing_error=%s",
                    provider_settings.ocr.provider,
                    provider_settings.ocr.model,
                    page_number,
                    parsing_error,
                )
                result = _minimal_ocr_response_from_text(
                    text=raw_text,
                    page_number=page_number,
                    image_path=image_path,
                )
                call_diagnostics.update(
                    {
                        "success": False,
                        "failure_type": "structured_output_parse_failure",
                        "fallback": "raw_text_ocr",
                        "error_type": type(exc).__name__,
                    }
                )
                if provider_diagnostics_sink is not None:
                    provider_diagnostics_sink(dict(call_diagnostics))
                return result
        call_diagnostics.update(
            {
                "success": False,
                "failure_type": "structured_output_parse_failure",
                "error_type": type(exc).__name__,
            }
        )
        if provider_diagnostics_sink is not None:
            provider_diagnostics_sink(dict(call_diagnostics))
        raise RuntimeError(
            f"ocr provider returned an unreadable response for page {page_number}: {exc}"
        ) from exc
    if provider_diagnostics_sink is not None:
        provider_diagnostics_sink(dict(call_diagnostics))
    return result


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


def _write_summary(
    artifacts: OCRWorkflowArtifacts,
    *,
    document_id: str,
    provider_settings: WorkflowProviderSettings,
    state_store: OCRWorkflowStateStore,
) -> None:
    payload: JsonObject = {
        "document_id": document_id,
        "ocr_provider": provider_settings.ocr.provider,
        "ocr_model": provider_settings.ocr.model,
        "legacy_dir": str(artifacts.legacy_dir),
        "rendered_dir": str(artifacts.rendered_dir),
        "state_db_path": str(artifacts.state_db_path),
        "progress_path": str(artifacts.progress_path),
        "document_completed": state_store.read_document_completed(document_id=document_id),
        "completed_pages": cast(list[JsonValue], list(artifacts.completed_pages)),
        "reused_pages": cast(list[JsonValue], list(artifacts.reused_pages)),
        "model_attempts": json_safe(
            {
                str(page_number): state_store.list_model_attempts(
                    document_id=document_id,
                    page_number=page_number,
                )
                for page_number in artifacts.completed_pages
            }
        ),
        "summary_attempts": json_safe(
            state_store.list_summary_attempts(document_id=document_id)
        ),
        "page_count": len(artifacts.ocr_pages),
        "document_tree_path": str(artifacts.document_tree_path) if artifacts.document_tree is not None else None,
    }
    artifacts.summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _emit_ocr_event(probe: WorkflowProbe | None, kind: str, /, **payload: object) -> None:
    emit_probe_event(probe, kind, **payload)


def prepare_ocr_workflow_input(
    *,
    document_id: str,
    title: str,
    output_dir: str | Path,
    image_payloads: Sequence[OCRImagePayload] | None = None,
    pdf_path: str | Path | None = None,
    provider_settings: WorkflowProviderSettings | None = None,
    ocr_runner: OCRRunner | None = None,
    pdf_rasterizer: PDFRasterizer | None = None,
    ocr_candidate_models: Sequence[str] | None = None,
    probe: WorkflowProbe | None = None,
    restore_grouping_checkpoint: bool = True,
    persist_grouping_checkpoint: bool = True,
    workflow_engine: GraphKnowledgeEngine | None = None,
    conversation_engine: GraphKnowledgeEngine | None = None,
    ocr_runtime_run_id: str | None = None,
    resume_from_checkpoint: bool = False,
    grouping_policy_version: str = "baseline-v1",
    summary_policy: SummaryPolicy | None = None,
    summary_generator: SummaryGenerator | None = None,
    grounded_summary_generator: GroundedSummaryGenerator | None = None,
    quality_policy: OCRQualityPolicy | None = None,
) -> OCRWorkflowArtifacts:
    """Prepare OCR pages and normalize them into workflow ingest input.

    The output directory is intentionally inspectable. It keeps:

    - rendered page images
    - legacy-compatible ``page_N.json`` files
    - ``ocr-state.sqlite`` as the authoritative state store
    - a progress manifest mirrored from SQLite for quick inspection
    - a short summary file for manual debugging

    Hook contracts:
    - ``ocr_runner`` must accept ``(image_path, page_number, provider_settings)``
      and return one structured OCR page result.
    - ``pdf_rasterizer`` must accept ``(pdf_path, rendered_dir)`` and return the
      list of rendered page image paths.
    """

    # Require exactly one input shape so the rest of the function can stay linear.
    if bool(image_payloads) == bool(pdf_path):
        raise ValueError("provide exactly one of image_payloads or pdf_path")

    # Load provider defaults only when the caller did not pass them explicitly.
    provider_settings = provider_settings or WorkflowProviderSettings.from_env()

    # Record the start of OCR preparation for probes and debug traces.
    _emit_ocr_event(
        probe,
        "ocr.prepare_started",
        document_id=document_id,
        title=title,
        output_dir=str(output_dir),
        source_kind="pdf" if pdf_path is not None else "image",
    )
    # Build the source plan and injected page context without invoking OCR.
    output_dir = Path(output_dir)
    preparation = _build_ocr_page_preparation(
        document_id=document_id,
        title=title,
        output_dir=output_dir,
        image_payloads=image_payloads,
        pdf_path=Path(pdf_path) if pdf_path is not None else None,
        provider_settings=provider_settings,
        ocr_runner=ocr_runner,
        pdf_rasterizer=pdf_rasterizer,
        ocr_candidate_models=ocr_candidate_models,
        probe=probe,
        quality_policy=quality_policy,
    )
    source_plan = preparation.source_plan
    state_store = preparation.state_store
    page_context = preparation.page_context
    state_db_path = preparation.state_db_path
    progress_path = preparation.progress_path
    summary_path = preparation.summary_path

    # Emit a trace that the state store and inspectable folders are ready.
    _emit_ocr_event(
        probe,
        "ocr.state_ready",
        document_id=document_id,
        state_db_path=str(state_db_path),
        progress_path=str(progress_path),
        rendered_dir=str(source_plan.rendered_dir),
        legacy_dir=str(source_plan.legacy_dir),
    )

    # Accumulate the serialized OCR pages that will later be normalized into workflow input.
    raw_pages: list[OCRPageJSON] = []

    # Track which pages completed, were reused, or still failed after retries.
    completed_pages: list[int] = []
    reused_pages: list[int] = []

    # Collect pages that still fail after exhausting every candidate model.
    page_failures: list[int] = []

    if workflow_engine is not None and conversation_engine is not None:
        from .ocr_runtime import run_ocr_page_workflow

        raw_pages, completed_pages, reused_pages, _ocr_runtime_run_id = run_ocr_page_workflow(
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            context=page_context,
            source_plan=source_plan,
            progress_path=progress_path,
            run_id=ocr_runtime_run_id or f"ocr-pages|{document_id}",
            resume_from_checkpoint=resume_from_checkpoint,
        )
    else:
        # Compatibility path for the standalone preparation API. The high
        # level ingest entrypoint supplies engines and uses the persisted path.
        for done_count, (page_number, image_path) in enumerate(source_plan.page_sources, start=1):
            image_path = Path(image_path)
            image_sha256 = _sha256_file(image_path)
            page_json_path = source_plan.legacy_dir / f"page_{page_number}.json"
            page_image_path = source_plan.legacy_dir / f"page_{page_number}{image_path.suffix or '.png'}"
            _process_ocr_page(
                context=page_context,
                done_count=done_count,
                page_number=page_number,
                image_path=image_path,
                image_sha256=image_sha256,
                page_json_path=page_json_path,
                page_image_path=page_image_path,
                raw_pages=raw_pages,
                completed_pages=completed_pages,
                reused_pages=reused_pages,
                page_failures=page_failures,
            )
            _sync_progress_from_state(
                state_store=state_store,
                document_id=document_id,
                title=title,
                source_kind=source_plan.source_kind,
                total_pages=len(source_plan.page_sources),
                progress_path=progress_path,
            )

    # Mark the whole document complete once every page has been processed.
    state_store.refresh_document_completion(document_id=document_id)

    # Write one final mirrored progress snapshot after completion.
    _sync_progress_from_state(
        state_store=state_store,
        document_id=document_id,
        title=title,
        source_kind=source_plan.source_kind,
        total_pages=len(source_plan.page_sources),
        progress_path=progress_path,
    )

    # Emit a final probe event for tooling and debug traces.
    _emit_ocr_event(
        probe,
        "ocr.prepare_finished",
        document_id=document_id,
        status="failed" if page_failures else "succeeded",
        completed_pages=completed_pages,
        reused_pages=reused_pages,
        failed_pages=page_failures,
        state_db_path=str(state_db_path),
    )

    # Fail the run if any page still has no successful OCR artifact.
    if page_failures:
        raise RuntimeError(
            f"OCR preparation incomplete for pages {page_failures}; rerun will retry failed pages."
        )

    # Convert the serialized OCR page dicts into the workflow ingest source model.
    return _finalize_ocr_workflow_artifacts(
        document_id=document_id,
        title=title,
        raw_pages=raw_pages,
        completed_pages=completed_pages,
        reused_pages=reused_pages,
        rendered_dir=source_plan.rendered_dir,
        legacy_dir=source_plan.legacy_dir,
        state_db_path=state_db_path,
        progress_path=progress_path,
        summary_path=summary_path,
        provider_settings=provider_settings,
        state_store=state_store,
        restore_grouping_checkpoint=restore_grouping_checkpoint,
        persist_grouping_checkpoint=persist_grouping_checkpoint,
        grouping_policy_version=grouping_policy_version,
        summary_policy=summary_policy,
        summary_generator=summary_generator,
        grounded_summary_generator=grounded_summary_generator,
    )


def run_ocr_ingest_workflow(
    *,
    document_id: str,
    title: str,
    output_dir: str | Path,
    workflow_engine: GraphKnowledgeEngine,
    conversation_engine: GraphKnowledgeEngine,
    knowledge_engine: GraphKnowledgeEngine | None = None,
    image_payloads: Sequence[OCRImagePayload] | None = None,
    pdf_path: str | Path | None = None,
    provider_settings: WorkflowProviderSettings | None = None,
    ocr_runner: OCRRunner | None = None,
    pdf_rasterizer: PDFRasterizer | None = None,
    ocr_candidate_models: Sequence[str] | None = None,
    deps: WorkflowRuntimeDeps | None = None,
    probe: WorkflowProbe | None = None,
    grouping_proposals: Sequence[GroupProposal] | None = None,
    grouping_passes: Sequence[Sequence[GroupProposal]] | None = None,
    grouping_policy_version: str = "baseline-v1",
    summary_policy: SummaryPolicy | None = None,
    summary_generator: SummaryGenerator | None = None,
    grounded_summary_generator: GroundedSummaryGenerator | None = None,
    quality_policy: OCRQualityPolicy | None = None,
    run_id: str | None = None,
    resume_from_checkpoint: bool = False,
    run_ownership_lease_ms: int | None = None,
) -> tuple[_RunCompat, WorkflowExportBundle | None, OCRWorkflowArtifacts]:
    """Run OCR, grouping, and semantic ingest under one persisted root run."""

    if grouping_proposals is not None and grouping_passes is not None:
        raise ValueError("provide grouping_proposals or grouping_passes, not both")

    if bool(image_payloads) == bool(pdf_path):
        raise ValueError("provide exactly one of image_payloads or pdf_path")
    effective_settings = provider_settings or WorkflowProviderSettings.from_env()
    preparation = _build_ocr_page_preparation(
        document_id=document_id,
        title=title,
        output_dir=Path(output_dir),
        image_payloads=image_payloads,
        pdf_path=Path(pdf_path) if pdf_path is not None else None,
        provider_settings=effective_settings,
        ocr_runner=ocr_runner,
        pdf_rasterizer=pdf_rasterizer,
        ocr_candidate_models=ocr_candidate_models,
        probe=probe,
        quality_policy=quality_policy,
    )
    from .ocr_runtime import run_ocr_root_workflow

    root_result, bundle, artifacts = run_ocr_root_workflow(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        preparation=preparation,
        provider_settings=effective_settings,
        document_id=document_id,
        title=title,
        grouping_policy_version=grouping_policy_version,
        grouping_proposals=tuple(grouping_proposals or ()),
        grouping_passes=tuple(grouping_passes or ()),
        summary_policy=summary_policy,
        summary_generator=summary_generator,
        grounded_summary_generator=grounded_summary_generator,
        deps=deps,
        run_id=run_id,
        resume_from_checkpoint=resume_from_checkpoint,
        run_ownership_lease_ms=run_ownership_lease_ms,
    )
    return (
        _RunCompat(
            run_id=str(root_result.run_id),
            final_state=root_result.final_state,
            status=str(root_result.status),
        ),
        bundle,
        artifacts,
    )
