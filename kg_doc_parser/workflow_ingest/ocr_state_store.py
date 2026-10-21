"""SQLite-backed OCR artifact, page-attempt, and grouping checkpoints."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import sqlite3
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from ..models import SplitPage
from .document_models import DocumentTree
from .ocr_artifacts import (
    artifact_metadata as _ocr_artifact_metadata,
)
from .ocr_artifacts import (
    artifact_schema_error as _ocr_artifact_schema_error,
)
from .ocr_artifacts import (
    read_json_object as _read_json_object,
)
from .ocr_file_io import write_bytes_atomic as _write_bytes_atomic
from .ocr_inputs import (
    find_matching_render_hash as _find_matching_render_hash,
)
from .ocr_inputs import (
    scan_legacy_json_pages as _scan_legacy_json_pages,
)
from .ocr_inputs import (
    scan_rendered_pages as _scan_rendered_pages,
)
from .ocr_inputs import (
    sha256_file as _sha256_file,
)
from .ocr_pipeline_contracts import (
    GroupingCheckpointMetadata,
)
from .ocr_progress_state import OCRProgressStateMixin
from .ocr_quality_policy import (
    ocr_artifact_quality_error as _ocr_artifact_quality_error,
)
from .ocr_state_schema import initialize_ocr_state_schema
from .ocr_state_types import (
    OCRPageStatus,
    OCRStage,
    decode_ocr_page_status,
    decode_ocr_stage,
)
from .ocr_summary_state import OCRSummaryStateMixin
from .serialization import JsonObject

_LOGGER = logging.getLogger(__name__)
_OCR_STATE_SCHEMA_VERSION = 2


def _grouping_text(row: sqlite3.Row, column: str) -> str:
    value: object = row[column]
    if not isinstance(value, str):
        raise ValueError(f"persisted grouping field {column!r} must be text")
    return value


def _grouping_nonnegative_int(row: sqlite3.Row, column: str) -> int:
    value: object = row[column]
    if type(value) is not int or value < 0:
        raise ValueError(
            f"persisted grouping field {column!r} must be a nonnegative integer"
        )
    return value


def _decode_grouping_pages(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError("persisted grouping checkpoint page frontier is malformed")
    pages: list[int] = []
    for page in value:
        if type(page) is not int or page < 1:
            raise ValueError("persisted grouping checkpoint page frontier is malformed")
        pages.append(page)
    if pages != sorted(set(pages)):
        raise ValueError("persisted grouping checkpoint page frontier is not ordered")
    return tuple(pages)


def _decode_grouping_frontier(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("persisted grouping frontier is malformed")
    frontier: list[str] = []
    for node_id in value:
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("persisted grouping frontier is malformed")
        frontier.append(node_id)
    return tuple(frontier)


class GroupingCheckpointConflict(RuntimeError):
    """A grouping writer lost its expected checkpoint ownership."""


@dataclass(slots=True)
class OCRPageStateSnapshot:
    """One page/stage snapshot from the SQLite resume store."""
    page_number: int
    stage: OCRStage
    attempt_count: int
    status: OCRPageStatus
    content_hash: str | None
    artifact_path: str | None
    last_error: str | None
    last_model: str | None
    last_attempted_ts: float | None
    config_fingerprint: str | None = None
    artifact_sha256: str | None = None


class OCRWorkflowStateStore(OCRProgressStateMixin, OCRSummaryStateMixin):
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
        initialize_ocr_state_schema(self._session)

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
            current_document = conn.execute(
                "SELECT input_fingerprint FROM document_state WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            if current_document is not None and tree.source_revision.startswith("ocr-v1:"):
                current_source_revision = (
                    f"ocr-v1:{current_document['input_fingerprint']}"
                )
                if tree.source_revision != current_source_revision:
                    raise GroupingCheckpointConflict(
                        f"grouping checkpoint source revision conflict for "
                        f"{document_id!r}: expected {current_source_revision!r}, "
                        f"found {tree.source_revision!r}"
                    )
            current = conn.execute(
                """
                SELECT tree_revision, source_fingerprint, grouping_policy_version
                FROM document_tree_state
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()
            current_frontier = conn.execute(
                """
                SELECT source_fingerprint, grouping_policy_version, frontier_json
                FROM grouping_frontier_state WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()
            current_metadata = conn.execute(
                """
                SELECT source_fingerprint, grouping_policy_version,
                       accepted_page_frontier_json, tree_revision,
                       merge_revision, last_runtime_step
                FROM grouping_checkpoint_metadata WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()
            completed_page_rows = conn.execute(
                """
                SELECT page_number
                FROM page_state
                WHERE document_id = ? AND stage = 'ocr' AND status = 'completed'
                ORDER BY page_number
                """,
                (document_id,),
            ).fetchall()
            accepted_pages = tuple(int(row["page_number"]) for row in completed_page_rows)
            current_revision = None if current is None else str(current["tree_revision"])
            has_frontier = current_frontier is not None
            root = next(node for node in tree.nodes if node.node_id == tree.root_id)
            expected_frontier = (
                tuple(child.node_id for child in root.children)
                if root.children
                else (root.node_id,)
            )
            if current_frontier is not None:
                try:
                    stored_frontier_payload = json.loads(
                        str(current_frontier["frontier_json"])
                    )
                except (TypeError, ValueError):
                    stored_frontier_payload = None
                frontier_matches = (
                    str(current_frontier["source_fingerprint"])
                    == tree.source_fingerprint()
                    and str(current_frontier["grouping_policy_version"])
                    == grouping_policy_version
                    and isinstance(stored_frontier_payload, list)
                    and tuple(stored_frontier_payload) == expected_frontier
                )
            else:
                frontier_matches = False
            if current_metadata is not None:
                try:
                    accepted_pages_payload = json.loads(
                        str(current_metadata["accepted_page_frontier_json"])
                    )
                except (TypeError, ValueError):
                    accepted_pages_payload = None
                metadata_matches = (
                    str(current_metadata["source_fingerprint"])
                    == tree.source_fingerprint()
                    and str(current_metadata["grouping_policy_version"])
                    == grouping_policy_version
                    and str(current_metadata["tree_revision"]) == tree_revision
                    and isinstance(accepted_pages_payload, list)
                    and all(
                        type(page) is int and page >= 1
                        for page in accepted_pages_payload
                    )
                    and accepted_pages_payload
                    == sorted(set(accepted_pages_payload))
                    and accepted_pages_payload == list(accepted_pages)
                    and int(current_metadata["merge_revision"]) >= 0
                    and bool(str(current_metadata["last_runtime_step"]))
                )
            else:
                metadata_matches = False
            # A process can die after this SQLite commit but before the
            # surrounding Kogwistar step checkpoint.  A retry with the same
            # exact tree is safe to acknowledge only when every companion
            # record still describes that exact commit.
            if current_revision == tree_revision:
                if not has_frontier:
                    raise GroupingCheckpointConflict(
                        f"grouping checkpoint frontier is missing for {document_id!r}"
                    )
                if current_metadata is None:
                    raise GroupingCheckpointConflict(
                        f"grouping checkpoint metadata is missing for {document_id!r}"
                    )
                if (
                    current is not None
                    and str(current["source_fingerprint"]) == tree.source_fingerprint()
                    and str(current["grouping_policy_version"]) == grouping_policy_version
                    and frontier_matches
                    and metadata_matches
                ):
                    return tree_revision
                raise GroupingCheckpointConflict(
                    f"idempotent grouping checkpoint replay for {document_id!r} "
                    "found mismatched companion records"
                )
            merge_revision = (
                1
                if current_metadata is None
                else int(current_metadata["merge_revision"]) + 1
            )
            if current_revision is None and (has_frontier or current_metadata is not None):
                raise GroupingCheckpointConflict(
                    f"grouping checkpoint create conflict for {document_id!r}: "
                    "found orphaned frontier or metadata without a tree"
                )
            if require_no_existing_tree and (
                current_revision is not None or has_frontier or current_metadata is not None
            ):
                raise GroupingCheckpointConflict(
                    f"grouping checkpoint create conflict for {document_id!r}: "
                    f"found existing revision {current_revision!r}, frontier, or metadata"
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
                    json.dumps(expected_frontier, separators=(",", ":")),
                    time.time(),
                ),
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
        stored_source = _grouping_text(row, "source_fingerprint")
        stored_policy = _grouping_text(row, "grouping_policy_version")
        if source_fingerprint is not None and stored_source != source_fingerprint:
            return None
        if grouping_policy_version is not None and stored_policy != grouping_policy_version:
            return None
        payload: object = json.loads(
            _grouping_text(row, "accepted_page_frontier_json")
        )
        accepted_pages = _decode_grouping_pages(payload)
        return GroupingCheckpointMetadata(
            document_id=document_id,
            source_fingerprint=stored_source,
            grouping_policy_version=stored_policy,
            accepted_page_frontier=accepted_pages,
            tree_revision=_grouping_text(row, "tree_revision"),
            merge_revision=_grouping_nonnegative_int(row, "merge_revision"),
            last_runtime_step=_grouping_text(row, "last_runtime_step"),
        )

    def document_tree_revision(self, *, document_id: str) -> str | None:
        """Return the current tree digest without applying source/policy filters."""
        with self._session() as conn:
            row = conn.execute(
                "SELECT tree_revision FROM document_tree_state WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return None if row is None else _grouping_text(row, "tree_revision")

    def load_document_tree(
        self,
        *,
        document_id: str,
        source_fingerprint: str | None = None,
        grouping_policy_version: str | None = None,
    ) -> DocumentTree | None:
        """Load a tree only when its atomically persisted checkpoint agrees."""
        with self._session() as conn:
            conn.execute("BEGIN")
            row = conn.execute(
                """
                SELECT source_fingerprint, grouping_policy_version, tree_json
                FROM document_tree_state
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()
            frontier = conn.execute(
                """
                SELECT source_fingerprint, grouping_policy_version, frontier_json
                FROM grouping_frontier_state
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()
            metadata = conn.execute(
                """
                SELECT source_fingerprint, grouping_policy_version,
                       accepted_page_frontier_json, tree_revision, last_runtime_step
                FROM grouping_checkpoint_metadata
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()
            completed_page_rows = conn.execute(
                """
                SELECT page_number
                FROM page_state
                WHERE document_id = ? AND stage = 'ocr' AND status = 'completed'
                ORDER BY page_number
                """,
                (document_id,),
            ).fetchall()
        if row is None:
            return None
        stored_source = _grouping_text(row, "source_fingerprint")
        stored_policy = _grouping_text(row, "grouping_policy_version")
        tree_json = _grouping_text(row, "tree_json")
        if (
            (source_fingerprint is not None and stored_source != source_fingerprint)
            or (
                grouping_policy_version is not None
                and stored_policy != grouping_policy_version
            )
        ):
            return None
        if frontier is None or metadata is None:
            raise ValueError("persisted grouping checkpoint is missing a companion record")
        stored_frontier_source = _grouping_text(frontier, "source_fingerprint")
        stored_frontier_policy = _grouping_text(frontier, "grouping_policy_version")
        stored_metadata_source = _grouping_text(metadata, "source_fingerprint")
        stored_metadata_policy = _grouping_text(metadata, "grouping_policy_version")
        stored_metadata_revision = _grouping_text(metadata, "tree_revision")
        last_runtime_step = _grouping_text(metadata, "last_runtime_step")
        stored_revision = hashlib.sha256(tree_json.encode("utf-8")).hexdigest()
        if (
            stored_frontier_source != stored_source
            or stored_frontier_policy != stored_policy
            or stored_metadata_source != stored_source
            or stored_metadata_policy != stored_policy
            or stored_metadata_revision != stored_revision
            or not last_runtime_step
        ):
            raise ValueError("persisted grouping checkpoint records disagree")

        tree = DocumentTree.model_validate_json(tree_json)
        if tree.source_fingerprint() != stored_source:
            raise ValueError("persisted grouping tree source fingerprint is inconsistent")
        root = next((node for node in tree.nodes if node.node_id == tree.root_id), None)
        if root is None:
            raise ValueError("persisted grouping tree root is missing")
        expected_frontier = (
            tuple(child.node_id for child in root.children)
            if root.children
            else (root.node_id,)
        )
        frontier_payload: object = json.loads(
            _grouping_text(frontier, "frontier_json")
        )
        if _decode_grouping_frontier(frontier_payload) != expected_frontier:
            raise ValueError("persisted grouping frontier does not match its tree")
        accepted_pages_payload: object = json.loads(
            _grouping_text(metadata, "accepted_page_frontier_json")
        )
        accepted_pages = _decode_grouping_pages(accepted_pages_payload)
        completed_pages = tuple(
            _grouping_nonnegative_int(completed_row, "page_number")
            for completed_row in completed_page_rows
        )
        if accepted_pages != completed_pages:
            raise ValueError(
                "persisted grouping accepted-page frontier does not match "
                "completed OCR pages"
            )
        return tree

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
        if row is None:
            return None
        stored_source = _grouping_text(row, "source_fingerprint")
        stored_policy = _grouping_text(row, "grouping_policy_version")
        if (
            (source_fingerprint is not None and stored_source != source_fingerprint)
            or (
                grouping_policy_version is not None
                and stored_policy != grouping_policy_version
            )
        ):
            return None
        payload: object = json.loads(_grouping_text(row, "frontier_json"))
        return _decode_grouping_frontier(payload)

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
            # The rendered artifact is the only authoritative source for the
            # current page identity. Progress JSON is advisory and must never
            # resurrect an OCR result after a same-path image replacement.
            content_hash = _find_matching_render_hash(
                rendered_dir, page_number, hash_file=_sha256_file
            )
            config_fingerprint: str | None = None
            artifact_valid = False
            artifact_error: str | None = None
            artifact_sha256: str | None = None
            try:
                persisted = _read_json_object(json_path)
                SplitPage.model_validate(persisted)
                artifact_valid = True
                metadata = _ocr_artifact_metadata(persisted)
                artifact_error = _ocr_artifact_schema_error(persisted)
                if artifact_error is not None:
                    artifact_valid = False
                quality_error = _ocr_artifact_quality_error(persisted)
                if quality_error is not None:
                    artifact_valid = False
                    artifact_error = quality_error
                config_fingerprint = metadata.get("ocr_configuration_fingerprint")
                source_digest = metadata.get("source_image_sha256")
                if (
                    artifact_valid
                    and content_hash is not None
                    and source_digest != content_hash
                ):
                    artifact_valid = False
                    artifact_error = (
                        "OCR artifact source-image digest does not match the "
                        "current rendered page"
                    )
                if artifact_valid:
                    artifact_sha256 = _sha256_file(json_path)
            except (OSError, ValueError) as exc:
                artifact_valid = False
                artifact_sha256 = None
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
                artifact_sha256=artifact_sha256 if artifact_valid else None,
            )
        self.refresh_document_completion(document_id=document_id)
        return rebuilt_any

    def _upsert_page_state(
        self,
        *,
        document_id: str,
        page_number: int,
        stage: OCRStage,
        attempt_count: int,
        status: OCRPageStatus,
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

    def get_page_state(
        self, *, document_id: str, page_number: int, stage: OCRStage
    ) -> OCRPageStateSnapshot | None:
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
            stage=decode_ocr_stage(row["stage"]),
            attempt_count=int(row["attempt_count"]),
            status=decode_ocr_page_status(row["status"]),
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
        stage: OCRStage,
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
            return stage != "ocr"
        if not artifact_path.is_file():
            return False
        try:
            artifact_sha256 = _sha256_file(artifact_path)
        except OSError:
            return False
        if state.artifact_sha256 is None or artifact_sha256 != state.artifact_sha256:
            return False
        if stage != "ocr":
            return True
        try:
            persisted = _read_json_object(artifact_path)
            split_page = SplitPage.model_validate(persisted)
            metadata = _ocr_artifact_metadata(persisted)
        except (OSError, TypeError, ValueError):
            return False
        return (
            split_page.pdf_page_num == page_number
            and _ocr_artifact_schema_error(persisted) is None
            and _ocr_artifact_quality_error(persisted) is None
            and metadata.get("source_image_sha256") == content_hash
            and metadata.get("ocr_configuration_fingerprint")
            == state.config_fingerprint
        )

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
        try:
            persisted = _read_json_object(artifact_path)
            split_page = SplitPage.model_validate(persisted)
            artifact_sha256 = _sha256_file(artifact_path)
        except (OSError, TypeError, ValueError):
            return False
        if split_page.pdf_page_num != page_number:
            return False
        if _ocr_artifact_schema_error(persisted) is not None:
            return False
        if _ocr_artifact_quality_error(persisted) is not None:
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
            artifact_sha256=artifact_sha256,
        )
        return True

    def record_attempt(
        self,
        *,
        document_id: str,
        page_number: int,
        stage: OCRStage,
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
        if not isinstance(payload, Mapping):
            return False
        try:
            artifact_bytes = json.dumps(
                payload, indent=2, ensure_ascii=False
            ).encode("utf-8")
            snapshot = json.loads(artifact_bytes)
        except (TypeError, ValueError):
            return False
        if not isinstance(snapshot, Mapping):
            return False
        try:
            split_page = SplitPage.model_validate(snapshot)
        except (TypeError, ValidationError):
            return False
        artifact_metadata = _ocr_artifact_metadata(snapshot)
        if (
            split_page.pdf_page_num != page_number
            or _ocr_artifact_schema_error(snapshot) is not None
            or _ocr_artifact_quality_error(snapshot) is not None
            or artifact_metadata.get("source_image_sha256") != content_hash
            or artifact_metadata.get("ocr_configuration_fingerprint")
            != config_fingerprint
        ):
            return False

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

            _write_bytes_atomic(artifact_path, artifact_bytes)
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
                SET status = 'completed', error_message = NULL, retry_feedback = NULL
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
        stage: OCRStage,
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
                persisted = _read_json_object(artifact_path)
                split_page = SplitPage.model_validate(persisted)
                metadata = _ocr_artifact_metadata(persisted)
                if _ocr_artifact_schema_error(persisted) is not None:
                    return False
                if _ocr_artifact_quality_error(persisted) is not None:
                    return False
                if split_page.pdf_page_num != page_number:
                    return False
                if metadata.get("source_image_sha256") != content_hash:
                    return False
            except (OSError, TypeError, ValueError):
                # Compatibility callers must not turn malformed JSON into an
                # accepted page; the canonical publisher applies the same
                # schema gate before recording completion.
                return False
        artifact_sha256: str | None = None
        if artifact_path is not None and artifact_path.is_file():
            try:
                artifact_sha256 = _sha256_file(artifact_path)
            except OSError:
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
                    artifact_sha256,
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
                    SET status = 'completed', error_message = NULL, retry_feedback = NULL
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
        stage: OCRStage,
        content_hash: str | None,
        config_fingerprint: str | None,
        error_message: str,
        model_name: str | None,
        retry_feedback: str | None = None,
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
                    SET status = 'failed', error_message = ?, retry_feedback = ?
                    WHERE document_id = ? AND page_number = ? AND stage = ? AND attempt_index = ?
                    """,
                    (
                        error_message,
                        retry_feedback,
                        document_id,
                        page_number,
                        stage,
                        attempt_index,
                    ),
                )
            return updated_count == 1

    def list_model_attempts(
        self, *, document_id: str, page_number: int, stage: OCRStage = "ocr"
    ) -> list[JsonObject]:
        with self._session() as conn:
            rows = conn.execute(
                """
                SELECT attempt_index, model_name, status, error_message,
                       retry_feedback, attempted_ts
                FROM model_attempts
                WHERE document_id = ? AND page_number = ? AND stage = ?
                ORDER BY attempt_index
                """,
                (document_id, page_number, stage),
            ).fetchall()
        return [dict(row) for row in rows]

