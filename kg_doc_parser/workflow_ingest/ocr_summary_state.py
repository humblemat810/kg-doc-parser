"""SQLite-backed idempotent summary-attempt journal operations."""

from __future__ import annotations

import json
import sqlite3
import time
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager

from .document_models import Summary, SummaryStatus
from .serialization import JsonObject


class OCRSummaryStateMixin(ABC):
    """Summary journal API backed by the owner's SQLite session boundary."""

    @abstractmethod
    def _session(self) -> AbstractContextManager[sqlite3.Connection]:
        """Yield a connection whose transaction is committed on clean exit."""

    def record_summary_attempt(
        self,
        *,
        document_id: str,
        node_id: str,
        input_fingerprint: str,
        status: SummaryStatus,
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
