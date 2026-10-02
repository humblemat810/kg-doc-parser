from kg_doc_parser.document_ingest_log_config import (
    DEFAULT_DOCUMENT_INGEST_LOG_DB,
    DOCUMENT_INGEST_LOG_DB_ENV,
    configured_document_ingest_log_db,
)


def test_document_ingest_log_path_preserves_legacy_default() -> None:
    assert configured_document_ingest_log_db({}) == DEFAULT_DOCUMENT_INGEST_LOG_DB


def test_document_ingest_log_path_accepts_instance_specific_override() -> None:
    path = "/var/lib/llm-wiki/logs/document_ingest-worker-a.sqlite"
    assert configured_document_ingest_log_db({DOCUMENT_INGEST_LOG_DB_ENV: path}) == path


def test_blank_document_ingest_log_path_uses_legacy_default() -> None:
    assert configured_document_ingest_log_db({DOCUMENT_INGEST_LOG_DB_ENV: "  "}) == DEFAULT_DOCUMENT_INGEST_LOG_DB
