from __future__ import annotations

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

import base64
import contextlib
import hashlib
import json
import logging
import sqlite3
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from PIL import Image
from pypdf import PdfReader

from ..models import OCRClusterResponse, SplitPage, SplitPageMeta

from .adapters import OCRPageJSON, normalize_ocr_pages
from .models import WorkflowIngestInput
from .providers import ProviderEndpointConfig, WorkflowProviderSettings, build_chat_model_for_role
from .probe import WorkflowProbe, emit_probe_event
from .service import run_ingest_workflow

_LOGGER = logging.getLogger(__name__)


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


# Pluggable hook for "OCR one page image and return the structured OCR model".
# Signature: (image_path, page_number, provider_settings) -> OCRClusterResponse
OCRRunner = Callable[[Path, int, WorkflowProviderSettings], OCRClusterResponse]
# Pluggable hook for "render a PDF into page image paths inside a destination dir".
# Signature: (pdf_path, rendered_dir) -> list[Path]
PDFRasterizer = Callable[[Path, Path], list[Path]]

_OCR_STATE_SCHEMA_VERSION = 1


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
    if pdf_source is not None:
        # Render once per PDF so downstream OCR can work page-by-page against
        # stable page image files and the state store can resume individual pages.
        rendered_dir = rendered_root / pdf_source.stem
        pdf_page_rasterizer = pdf_rasterizer or _render_pdf_to_images
        rendered_paths = pdf_page_rasterizer(pdf_source, rendered_dir)
        page_sources = list(enumerate(rendered_paths, start=1))
        source_kind = "pdf"
        input_fingerprint = _compute_input_fingerprint(pdf_path=pdf_source)
    else:
        rendered_dir = rendered_root / document_id
        page_sources = _materialize_image_payloads(list(image_payloads or []), rendered_dir)
        source_kind = "image"
        input_fingerprint = _compute_input_fingerprint(page_sources=page_sources)

    legacy_dir = legacy_root / _legacy_folder_name(document_id=document_id, pdf_path=pdf_source)
    legacy_dir.mkdir(parents=True, exist_ok=True)
    return OCRSourcePlan(
        pdf_source=pdf_source,
        rendered_dir=rendered_dir,
        legacy_dir=legacy_dir,
        page_sources=page_sources,
        source_kind=source_kind,
        input_fingerprint=input_fingerprint,
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
    ):
        _emit_ocr_event(
            probe,
            "ocr.render_started",
            document_id=document_id,
            page_number=page_number,
            image_path=str(image_path),
        )
        state_store.record_attempt(
            document_id=document_id,
            page_number=page_number,
            stage="render",
            content_hash=image_sha256,
            model_name="local-materialize",
            artifact_path=image_path,
        )
        state_store.record_page_completed(
            document_id=document_id,
            page_number=page_number,
            stage="render",
            content_hash=image_sha256,
            artifact_path=image_path,
            model_name="local-materialize",
            attempt_index=1 if state_store.get_page_state(document_id=document_id, page_number=page_number, stage="render") else None,
        )
        _emit_ocr_event(
            probe,
            "ocr.render_completed",
            document_id=document_id,
            page_number=page_number,
            image_path=str(image_path),
        )

    # Resume boundary: if the OCR page artifact already matches the image,
    # reuse the serialized page instead of calling the model again.
    if state_store.should_skip_page(
        document_id=document_id,
        page_number=page_number,
        stage="ocr",
        content_hash=image_sha256,
        artifact_path=page_json_path,
    ):
        split_page = SplitPage.model_validate(json.loads(page_json_path.read_text(encoding="utf-8")))
        raw_pages.append(split_page.dump_supercede_parse())
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
        )
        try:
            response = context.ocr_page_runner(image_path, page_number, candidate_settings)
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
            page_json_path.write_text(
                json.dumps(split_page.dump_raw(dump_format="json"), indent=2),
                encoding="utf-8",
            )
            state_store.record_page_completed(
                document_id=document_id,
                page_number=page_number,
                stage="ocr",
                content_hash=image_sha256,
                artifact_path=page_json_path,
                model_name=candidate_model,
                attempt_index=attempt_index,
            )
            # The adapter consumes plain dict pages, not the Pydantic page
            # model, so we serialize the page into the legacy-compatible shape here.
            raw_pages.append(split_page.dump_supercede_parse())
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
) -> OCRWorkflowArtifacts:
    workflow_input: WorkflowIngestInput = normalize_ocr_pages(
        document_id=document_id,
        title=title,
        pages=raw_pages,
    )
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
    )
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
    ) -> "OCRWorkflowStateStore":
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
    def _session(self):
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
                """
            )

    def _is_empty(self) -> bool:
        with self._session() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM document_state").fetchone()
        return bool(row is None or int(row["count"]) == 0)

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
    ) -> bool:
        self.ensure_document(
            document_id=document_id,
            title=title,
            source_kind=source_kind,
            total_pages=total_pages,
            input_fingerprint=input_fingerprint,
        )
        rebuilt_any = False
        progress_payload: dict[str, Any] = {}
        if progress_path.exists():
            progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
        progress_pages = progress_payload.get("pages", {})
        for page_number, path in _scan_rendered_pages(rendered_dir):
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
            )
        for page_number, json_path in _scan_legacy_json_pages(legacy_dir):
            content_hash = None
            record = progress_pages.get(str(page_number))
            if isinstance(record, dict):
                content_hash = record.get("image_sha256")
            if content_hash is None:
                content_hash = _find_matching_render_hash(rendered_dir, legacy_dir, page_number)
            rebuilt_any = True
            self._upsert_page_state(
                document_id=document_id,
                page_number=page_number,
                stage="ocr",
                attempt_count=1,
                status="completed",
                content_hash=content_hash,
                artifact_path=str(json_path),
                last_error=None,
                last_model=None,
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
    ) -> None:
        now = time.time()
        with self._session() as conn:
            conn.execute(
                """
                INSERT INTO page_state (
                    document_id, page_number, stage, attempt_count, status,
                    content_hash, artifact_path, last_error, last_model, last_attempted_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id, page_number, stage) DO UPDATE SET
                    attempt_count = excluded.attempt_count,
                    status = excluded.status,
                    content_hash = excluded.content_hash,
                    artifact_path = excluded.artifact_path,
                    last_error = excluded.last_error,
                    last_model = excluded.last_model,
                    last_attempted_ts = excluded.last_attempted_ts
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
                    now,
                ),
            )

    def get_page_state(self, *, document_id: str, page_number: int, stage: str) -> OCRPageStateSnapshot | None:
        with self._session() as conn:
            row = conn.execute(
                """
                SELECT page_number, stage, attempt_count, status, content_hash, artifact_path,
                       last_error, last_model, last_attempted_ts
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
        )

    def should_skip_page(
        self,
        *,
        document_id: str,
        page_number: int,
        stage: str,
        content_hash: str,
        artifact_path: Path | None = None,
    ) -> bool:
        state = self.get_page_state(document_id=document_id, page_number=page_number, stage=stage)
        if state is None or state.status != "completed" or state.content_hash != content_hash:
            return False
        if artifact_path is not None and not artifact_path.exists():
            return False
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
    ) -> int:
        now = time.time()
        with self._session() as conn:
            row = conn.execute(
                """
                SELECT attempt_count
                FROM page_state
                WHERE document_id = ? AND page_number = ? AND stage = ?
                """,
                (document_id, page_number, stage),
            ).fetchone()
            attempt_count = int(row["attempt_count"]) + 1 if row is not None else 1
            conn.execute(
                """
                INSERT INTO page_state (
                    document_id, page_number, stage, attempt_count, status,
                    content_hash, artifact_path, last_error, last_model, last_attempted_ts
                ) VALUES (?, ?, ?, ?, 'pending', ?, ?, NULL, ?, ?)
                ON CONFLICT(document_id, page_number, stage) DO UPDATE SET
                    attempt_count = excluded.attempt_count,
                    status = excluded.status,
                    content_hash = excluded.content_hash,
                    artifact_path = excluded.artifact_path,
                    last_error = excluded.last_error,
                    last_model = excluded.last_model,
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
    ) -> None:
        now = time.time()
        with self._session() as conn:
            conn.execute(
                """
                UPDATE page_state
                SET status = 'completed',
                    content_hash = ?,
                    artifact_path = ?,
                    last_error = NULL,
                    last_model = ?,
                    last_attempted_ts = ?
                WHERE document_id = ? AND page_number = ? AND stage = ?
                """,
                (
                    content_hash,
                    str(artifact_path) if artifact_path is not None else None,
                    model_name,
                    now,
                    document_id,
                    page_number,
                    stage,
                ),
            )
            if attempt_index is not None:
                conn.execute(
                    """
                    UPDATE model_attempts
                    SET status = 'completed', error_message = NULL
                    WHERE document_id = ? AND page_number = ? AND stage = ? AND attempt_index = ?
                    """,
                    (document_id, page_number, stage, attempt_index),
                )

    def record_page_failed(
        self,
        *,
        document_id: str,
        page_number: int,
        stage: str,
        content_hash: str | None,
        error_message: str,
        model_name: str | None,
        attempt_index: int | None = None,
    ) -> None:
        now = time.time()
        with self._session() as conn:
            conn.execute(
                """
                UPDATE page_state
                SET status = 'failed',
                    content_hash = ?,
                    last_error = ?,
                    last_model = ?,
                    last_attempted_ts = ?
                WHERE document_id = ? AND page_number = ? AND stage = ?
                """,
                (
                    content_hash,
                    error_message,
                    model_name,
                    now,
                    document_id,
                    page_number,
                    stage,
                ),
            )
            if attempt_index is not None:
                conn.execute(
                    """
                    UPDATE model_attempts
                    SET status = 'failed', error_message = ?
                    WHERE document_id = ? AND page_number = ? AND stage = ? AND attempt_index = ?
                    """,
                    (error_message, document_id, page_number, stage, attempt_index),
                )

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
            completed = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM page_state
                WHERE document_id = ? AND stage = 'ocr' AND status = 'completed'
                """,
                (document_id,),
            ).fetchone()
        is_completed = total_pages > 0 and int(completed["count"]) == total_pages
        self.mark_document_completed(document_id=document_id, is_completed=is_completed)
        return is_completed

    def list_model_attempts(self, *, document_id: str, page_number: int, stage: str = "ocr") -> list[dict[str, Any]]:
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
    rendered_dir.mkdir(parents=True, exist_ok=True)
    resolved: list[tuple[int, Path]] = []
    for index, payload in enumerate(image_payloads, start=1):
        page_number = int(payload.page_number or index)
        suffix = ".png"
        if payload.image_path:
            src = Path(payload.image_path)
            suffix = src.suffix or ".png"
            dst = rendered_dir / f"page_{page_number}{suffix}"
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            else:
                dst = src
        elif payload.image_bytes_b64:
            raw = base64.b64decode(payload.image_bytes_b64)
            filename = payload.filename or f"page_{page_number}.png"
            dst = rendered_dir / filename
            dst.write_bytes(raw)
        else:
            raise ValueError("OCRImagePayload requires image_path or image_bytes_b64")
        resolved.append((page_number, dst))
    return resolved


def _compute_input_fingerprint(*, page_sources: Sequence[tuple[int, Path]] | None = None, pdf_path: Path | None = None) -> str:
    payload: dict[str, Any] = {}
    if pdf_path is not None:
        payload["pdf_sha256"] = _sha256_file(pdf_path)
    if page_sources is not None:
        payload["pages"] = [
            {"page_number": page_number, "sha256": _sha256_file(Path(path))}
            for page_number, path in page_sources
        ]
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_progress(progress: OCRWorkflowProgress, progress_path: Path) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(
        json.dumps(progress.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )


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
        if not output_path.exists():
            images = convert_from_path(
                str(pdf_path),
                first_page=page_number,
                last_page=page_number,
                fmt="png",
            )
            if not images:
                raise RuntimeError(f"pdf rasterizer returned no image for page {page_number}")
            images[0].save(output_path, "PNG")
        output_paths.append(output_path)
    return output_paths


def _coerce_ocr_response(payload: Any) -> OCRClusterResponse:
    """Coerce a structured-output payload into the OCR model."""
    if isinstance(payload, OCRClusterResponse):
        return payload
    if isinstance(payload, dict):
        parsed = payload.get("parsed", payload)
        if parsed is not None:
            return OCRClusterResponse.model_validate(parsed)
        raw = payload.get("raw")
        if raw is not None:
            content = getattr(raw, "content", raw)
            if isinstance(content, str):
                return OCRClusterResponse.model_validate(json.loads(content))
            if isinstance(content, list):
                text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
                if text.strip():
                    return OCRClusterResponse.model_validate(json.loads(text))
    return OCRClusterResponse.model_validate(payload)


def _extract_message_text(raw: Any) -> str:
    """Extract plain text from LangChain raw message content."""
    content = getattr(raw, "content", raw)
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
        )
    return OCRClusterResponse(
        OCR_text_clusters=[
            {
                "text": normalized_text,
                "bb_x_min": 0.0,
                "bb_x_max": float(width),
                "bb_y_min": 0.0,
                "bb_y_max": float(height),
                "cluster_number": 0,
            }
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
    )


def _run_live_ocr_page(image_path: Path, page_number: int, provider_settings: WorkflowProviderSettings) -> OCRClusterResponse:
    """Run one page image through the configured OCR provider.

    This is the default implementation behind the OCRRunner hook. Tests can
    replace it with a fake runner, but the callable contract stays the same:
    page image in, structured OCR model out.
    """
    chat = build_chat_model_for_role("ocr", provider_settings)
    structured = chat.with_structured_output(OCRClusterResponse, include_raw=True)
    prompt = (
        "You are an OCR model for workflow ingest.\n"
        "Return structured OCR for one page image.\n"
        "Preserve reading order, cluster numbering, and non-text regions.\n"
        "Do not invent missing text. If the page is empty, mark it as empty."
    )
    response = structured.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(
                content=[
                    {"type": "text", "text": f"OCR page {page_number} and return the structured schema."},
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(image_path)}},
                ]
            ),
        ]
    )
    try:
        return _coerce_ocr_response(response)
    except Exception as exc:  # noqa: BLE001
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
                return _minimal_ocr_response_from_text(
                    text=raw_text,
                    page_number=page_number,
                    image_path=image_path,
                )
        raise RuntimeError(
            f"ocr provider returned an unreadable response for page {page_number}: {exc}"
        ) from exc


def _legacy_folder_name(*, document_id: str, pdf_path: Path | None) -> str:
    if pdf_path is not None:
        return pdf_path.name
    return document_id


def _write_summary(
    artifacts: OCRWorkflowArtifacts,
    *,
    document_id: str,
    provider_settings: WorkflowProviderSettings,
    state_store: OCRWorkflowStateStore,
) -> None:
    payload: dict[str, Any] = {
        "document_id": document_id,
        "ocr_provider": provider_settings.ocr.provider,
        "ocr_model": provider_settings.ocr.model,
        "legacy_dir": str(artifacts.legacy_dir),
        "rendered_dir": str(artifacts.rendered_dir),
        "state_db_path": str(artifacts.state_db_path),
        "progress_path": str(artifacts.progress_path),
        "document_completed": state_store.read_document_completed(document_id=document_id),
        "completed_pages": artifacts.completed_pages,
        "reused_pages": artifacts.reused_pages,
        "model_attempts": {
            str(page_number): state_store.list_model_attempts(document_id=document_id, page_number=page_number)
            for page_number in artifacts.completed_pages
        },
        "page_count": len(artifacts.ocr_pages),
    }
    artifacts.summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _emit_ocr_event(probe: WorkflowProbe | None, kind: str, /, **payload: Any) -> None:
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
    # Resolve the output folder and the inspectable files we will write.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "ocr-summary.json"
    progress_path = output_dir / "ocr-progress.json"
    state_db_path = output_dir / "ocr-state.sqlite"

    # Turn the source input into a concrete page-by-page plan.
    source_plan = _resolve_ocr_source_plan(
        document_id=document_id,
        output_dir=output_dir,
        image_payloads=image_payloads,
        pdf_path=Path(pdf_path) if pdf_path is not None else None,
        pdf_rasterizer=pdf_rasterizer,
    )

    # Open or rebuild the SQLite resume store for this exact document input.
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
    )

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

    # Select candidate OCR models, falling back to the configured default model.
    candidate_models: list[str] = list(ocr_candidate_models or [provider_settings.ocr.model])

    # Collect pages that still fail after exhausting every candidate model.
    page_failures: list[int] = []

    # Bundle the reusable per-run OCR context so the per-page loop stays small.
    page_context = OCRPageProcessingContext(
        document_id=document_id,
        title=title,
        source_kind=source_plan.source_kind,
        total_pages=len(source_plan.page_sources),
        input_fingerprint=source_plan.input_fingerprint,
        candidate_models=candidate_models,
        provider_settings=provider_settings,
        ocr_page_runner=ocr_runner or _run_live_ocr_page,
        state_store=state_store,
        probe=probe,
    )

    # Process every rendered page one-by-one.
    for done_count, (page_number, image_path) in enumerate(source_plan.page_sources, start=1):
        image_path = Path(image_path)
        image_sha256 = _sha256_file(image_path)
        page_json_path = source_plan.legacy_dir / f"page_{page_number}.json"
        page_image_path = source_plan.legacy_dir / f"page_{page_number}{image_path.suffix or '.png'}"

        # Run the page through the resume-aware OCR processor.
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

        # Mirror the SQLite state into the human-readable progress file.
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
    )


def run_ocr_ingest_workflow(
    *,
    document_id: str,
    title: str,
    output_dir: str | Path,
    workflow_engine: Any,
    conversation_engine: Any,
    knowledge_engine: Any | None = None,
    image_payloads: Sequence[OCRImagePayload] | None = None,
    pdf_path: str | Path | None = None,
    provider_settings: WorkflowProviderSettings | None = None,
    ocr_runner: OCRRunner | None = None,
    pdf_rasterizer: PDFRasterizer | None = None,
    ocr_candidate_models: Sequence[str] | None = None,
    deps: dict[str, Any] | None = None,
    probe: WorkflowProbe | None = None,
):
    """Run OCR preparation and then feed the normalized result into workflow ingest.

    This is the high-level two-stage orchestration:
    1. produce or reuse per-page OCR artifacts
    2. normalize those pages and hand them to the semantic workflow parser
    """

    from .parsing import parse_ocr_document

    artifacts = parse_ocr_document(
        document_id=document_id,
        title=title,
        output_dir=output_dir,
        image_payloads=image_payloads,
        pdf_path=pdf_path,
        provider_settings=provider_settings,
        ocr_runner=ocr_runner,
        pdf_rasterizer=pdf_rasterizer,
        ocr_candidate_models=ocr_candidate_models,
        probe=probe,
    )
    run, bundle = run_ingest_workflow(
        inp=artifacts.workflow_input,
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        deps=deps,
    )
    return run, bundle, artifacts
