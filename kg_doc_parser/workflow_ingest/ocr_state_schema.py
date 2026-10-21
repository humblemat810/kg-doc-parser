"""Schema creation and additive migrations for the OCR state database."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import AbstractContextManager

OCRStateSessionFactory = Callable[
    [], AbstractContextManager[sqlite3.Connection]
]


def initialize_ocr_state_schema(session: OCRStateSessionFactory) -> None:
    """Create current OCR tables and add columns to supported legacy schemas."""
    with session() as conn:
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
                retry_feedback TEXT,
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
        page_columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(page_state)").fetchall()
        }
        if "config_fingerprint" not in page_columns:
            conn.execute("ALTER TABLE page_state ADD COLUMN config_fingerprint TEXT")
        if "artifact_sha256" not in page_columns:
            conn.execute("ALTER TABLE page_state ADD COLUMN artifact_sha256 TEXT")
        attempt_columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(model_attempts)").fetchall()
        }
        if "retry_feedback" not in attempt_columns:
            conn.execute("ALTER TABLE model_attempts ADD COLUMN retry_feedback TEXT")
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
