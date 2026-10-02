"""Configuration for the parser's durable ingest event log."""

from __future__ import annotations

import os
from collections.abc import Mapping

DEFAULT_DOCUMENT_INGEST_LOG_DB = os.path.join("logs", "document_ingest.sqlite")
DOCUMENT_INGEST_LOG_DB_ENV = "KG_DOC_PARSER_DOCUMENT_INGEST_LOG_DB"


def configured_document_ingest_log_db(
    environ: Mapping[str, str] | None = None,
) -> str:
    """Return the configured ingest log path, preserving the legacy default."""

    values = os.environ if environ is None else environ
    configured = str(values.get(DOCUMENT_INGEST_LOG_DB_ENV, "") or "").strip()
    return configured or DEFAULT_DOCUMENT_INGEST_LOG_DB


__all__ = [
    "DEFAULT_DOCUMENT_INGEST_LOG_DB",
    "DOCUMENT_INGEST_LOG_DB_ENV",
    "configured_document_ingest_log_db",
]
