"""SQLite-backed document completion and progress projections."""

from __future__ import annotations

import sqlite3
import time
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager

from .ocr_pipeline_contracts import OCRPageProgress, OCRWorkflowProgress
from .ocr_state_types import decode_ocr_page_status


def _sqlite_integer(row: sqlite3.Row, column: str) -> int:
    value: object = row[column]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"OCR state column {column!r} must contain an integer")
    return value


def _sqlite_text(row: sqlite3.Row, column: str) -> str:
    value: object = row[column]
    if not isinstance(value, str):
        raise TypeError(f"OCR state column {column!r} must contain text")
    return value


def _sqlite_optional_text(row: sqlite3.Row, column: str) -> str | None:
    value: object = row[column]
    if value is not None and not isinstance(value, str):
        raise TypeError(f"OCR state column {column!r} must contain text or NULL")
    return value


class OCRProgressStateMixin(ABC):
    """Read and update document-level completion through the owner's session."""

    @abstractmethod
    def _session(self) -> AbstractContextManager[sqlite3.Connection]:
        """Yield a connection whose transaction commits on clean exit."""

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
            total_pages = _sqlite_integer(row, "total_pages") if row is not None else 0
            if total_pages < 0:
                raise ValueError("OCR state total_pages cannot be negative")
            completed_rows = conn.execute(
                """
                SELECT page_number
                FROM page_state
                WHERE document_id = ? AND stage = 'ocr' AND status = 'completed'
                """,
                (document_id,),
            ).fetchall()
        completed_pages = {
            _sqlite_integer(completed_row, "page_number")
            for completed_row in completed_rows
        }
        if any(page_number < 1 for page_number in completed_pages):
            raise ValueError("OCR state page numbers must be positive")
        expected_pages = set(range(1, total_pages + 1))
        is_completed = total_pages > 0 and completed_pages == expected_pages
        self.mark_document_completed(document_id=document_id, is_completed=is_completed)
        return is_completed

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
            artifact_path = _sqlite_optional_text(row, "artifact_path")
            if artifact_path is None:
                continue
            page_number = _sqlite_integer(row, "page_number")
            if page_number < 1:
                raise ValueError("OCR state page numbers must be positive")
            content_hash = _sqlite_optional_text(row, "content_hash") or ""
            pages[str(page_number)] = OCRPageProgress(
                page_number=page_number,
                image_path="",
                image_sha256=content_hash,
                json_path=artifact_path,
                status=decode_ocr_page_status(_sqlite_text(row, "status")),
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
        if row is None:
            return False
        is_completed = _sqlite_integer(row, "is_completed")
        if is_completed not in {0, 1}:
            raise ValueError("OCR state is_completed must be 0 or 1")
        return is_completed == 1
