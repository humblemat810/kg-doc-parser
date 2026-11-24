use anyhow::{bail, Result};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

use crate::ocr_artifacts::artifact_schema_error;

const OCR_STATE_SCHEMA_VERSION: i64 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OcrStage {
    Ocr,
    Render,
    Grouping,
    Summary,
}

impl OcrStage {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Ocr => "ocr",
            Self::Render => "render",
            Self::Grouping => "grouping",
            Self::Summary => "summary",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OcrPageStatus {
    Pending,
    Completed,
    Failed,
}

#[derive(Debug, Clone)]
pub struct OcrPageStateSnapshot {
    pub page_number: i64,
    pub stage: OcrStage,
    pub attempt_count: i64,
    pub status: OcrPageStatus,
    pub content_hash: Option<String>,
    pub artifact_path: Option<String>,
    pub last_error: Option<String>,
    pub last_model: Option<String>,
    pub last_attempted_ts: Option<f64>,
    pub config_fingerprint: Option<String>,
    pub artifact_sha256: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DocumentIdentity {
    pub document_id: String,
    pub title: String,
    pub source_kind: String,
    pub input_fingerprint: String,
    pub total_pages: i64,
}

#[derive(Debug, Clone, Default)]
pub struct OcrAttemptRequest {
    pub content_hash: Option<String>,
    pub model_name: String,
    pub artifact_path: Option<PathBuf>,
    pub config_fingerprint: Option<String>,
}

struct ExistingPageAttempt {
    attempt_count: i64,
    status: String,
    content_hash: Option<String>,
    config_fingerprint: Option<String>,
    artifact_path: Option<String>,
    artifact_sha256: Option<String>,
}

pub struct OcrPagePublication<'a> {
    pub page_number: i64,
    pub content_hash: &'a str,
    pub artifact_path: &'a Path,
    pub payload: Value,
    pub model_name: &'a str,
    pub attempt_index: i64,
    pub config_fingerprint: &'a str,
}

fn write_atomic(path: &Path, contents: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!(
        "{}tmp-{}",
        path.extension()
            .and_then(std::ffi::OsStr::to_str)
            .map(|value| format!("{value}."))
            .unwrap_or_default(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    std::fs::write(&temporary, contents)?;
    std::fs::rename(temporary, path)?;
    Ok(())
}

fn validate_publication(publication: &OcrPagePublication) -> Result<()> {
    if !publication.payload.is_object() {
        bail!("OCR page payload must be a JSON object");
    }
    if let Some(error) = artifact_schema_error(&publication.payload) {
        bail!(error);
    }
    if artifact_page_number(&publication.payload) != Some(publication.page_number) {
        bail!("OCR page payload page number does not match publication");
    }
    if metadata_field(&publication.payload, "source_image_sha256") != Some(publication.content_hash)
    {
        bail!("OCR page payload source hash does not match publication");
    }
    if metadata_field(&publication.payload, "ocr_configuration_fingerprint")
        != Some(publication.config_fingerprint)
    {
        bail!("OCR page payload configuration does not match publication");
    }
    Ok(())
}

pub struct OcrWorkflowStateStore {
    db_path: PathBuf,
}

fn unix_time() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or_default()
}

fn status_str(status: OcrPageStatus) -> &'static str {
    match status {
        OcrPageStatus::Pending => "pending",
        OcrPageStatus::Completed => "completed",
        OcrPageStatus::Failed => "failed",
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    let contents = std::fs::read(path)?;
    let mut digest = Sha256::new();
    digest.update(contents);
    Ok(digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn json_object(path: &Path) -> Result<Value> {
    let payload: Value = serde_json::from_slice(&std::fs::read(path)?)?;
    if !payload.is_object() {
        bail!("OCR state artifact must be a JSON object");
    }
    Ok(payload)
}

fn artifact_page_number(payload: &Value) -> Option<i64> {
    payload
        .get("pdf_page_num")
        .and_then(Value::as_i64)
        .or_else(|| {
            payload
                .get("pdf_page_num")
                .and_then(Value::as_str)
                .and_then(|value| value.trim().parse::<i64>().ok())
        })
}

fn metadata_field<'a>(payload: &'a Value, key: &str) -> Option<&'a str> {
    payload
        .get("metadata")
        .and_then(|metadata| metadata.get(key))
        .and_then(Value::as_str)
}

fn scan_rendered_pages(rendered_dir: &Path) -> Vec<(i64, PathBuf)> {
    let mut pages = Vec::new();
    let Ok(entries) = std::fs::read_dir(rendered_dir) else {
        return pages;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(stem) = path.file_stem().and_then(std::ffi::OsStr::to_str) else {
            continue;
        };
        let Some(page_number) = stem
            .rsplit_once('_')
            .and_then(|(_, suffix)| suffix.parse::<i64>().ok())
        else {
            continue;
        };
        pages.push((page_number, path));
    }
    pages.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));
    pages
}

fn scan_legacy_json_pages(legacy_dir: &Path) -> Vec<(i64, PathBuf)> {
    let mut pages = Vec::new();
    let Ok(entries) = std::fs::read_dir(legacy_dir) else {
        return pages;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(std::ffi::OsStr::to_str) != Some("json") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(std::ffi::OsStr::to_str) else {
            continue;
        };
        let Some(page_number) = stem
            .rsplit_once('_')
            .and_then(|(_, suffix)| suffix.parse::<i64>().ok())
        else {
            continue;
        };
        pages.push((page_number, path));
    }
    pages.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));
    pages
}

impl OcrWorkflowStateStore {
    pub fn open(db_path: &Path) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let store = Self {
            db_path: db_path.to_path_buf(),
        };
        store.initialize()?;
        Ok(store)
    }

    pub fn rebuild_from_artifacts(
        &self,
        identity: &DocumentIdentity,
        rendered_dir: &Path,
        legacy_dir: &Path,
        render_configuration_fingerprint: &str,
    ) -> Result<bool> {
        self.ensure_document(identity)?;
        let mut rebuilt_any = false;
        let rendered_pages = scan_rendered_pages(rendered_dir);
        for (page_number, rendered_path) in &rendered_pages {
            if *page_number > identity.total_pages {
                continue;
            }
            rebuilt_any = true;
            self.upsert_page_state(
                identity.document_id.as_str(),
                *page_number,
                OcrStage::Render,
                1,
                OcrPageStatus::Completed,
                Some(sha256_file(rendered_path)?),
                Some(rendered_path.to_string_lossy().into_owned()),
                None,
                None,
                Some(render_configuration_fingerprint.to_string()),
                None,
            )?;
        }
        for (page_number, json_path) in scan_legacy_json_pages(legacy_dir) {
            if page_number > identity.total_pages {
                continue;
            }
            rebuilt_any = true;
            let content_hash = rendered_pages
                .iter()
                .find(|(candidate_page, _)| *candidate_page == page_number)
                .map(|(_, path)| sha256_file(path))
                .transpose()?;
            let persisted = std::fs::read_to_string(&json_path)
                .ok()
                .and_then(|contents| serde_json::from_str::<Value>(&contents).ok());
            let mut artifact_error = if persisted.is_none() {
                Some("invalid OCR artifact during state rebuild".to_string())
            } else {
                None
            };
            let mut config_fingerprint = None;
            let mut artifact_sha256 = None;
            let mut valid = persisted.is_some();
            if let Some(payload) = persisted.as_ref() {
                if let Some(error) = artifact_schema_error(payload) {
                    valid = false;
                    artifact_error = Some(error);
                }
                config_fingerprint =
                    metadata_field(payload, "ocr_configuration_fingerprint").map(str::to_string);
                let source_digest = metadata_field(payload, "source_image_sha256");
                if valid
                    && (artifact_page_number(payload) != Some(page_number)
                        || source_digest != content_hash.as_deref())
                {
                    valid = false;
                    artifact_error = Some(
                        "OCR artifact identity does not match the current rendered page"
                            .to_string(),
                    );
                }
                if valid {
                    artifact_sha256 = Some(sha256_file(&json_path)?);
                }
            }
            if content_hash.is_none() {
                valid = false;
                artifact_error = artifact_error.or(Some(
                    "OCR artifact has no authoritative source-image digest".to_string(),
                ));
            }
            self.upsert_page_state(
                identity.document_id.as_str(),
                page_number,
                OcrStage::Ocr,
                1,
                if valid {
                    OcrPageStatus::Completed
                } else {
                    OcrPageStatus::Pending
                },
                content_hash,
                Some(json_path.to_string_lossy().into_owned()),
                artifact_error,
                None,
                config_fingerprint,
                artifact_sha256,
            )?;
        }
        Ok(rebuilt_any)
    }

    fn connection(&self) -> Result<Connection> {
        Ok(Connection::open_with_flags(
            &self.db_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE | rusqlite::OpenFlags::SQLITE_OPEN_CREATE,
        )?)
    }

    fn initialize(&self) -> Result<()> {
        let mut connection = self.connection()?;
        let transaction = connection.transaction()?;
        transaction.execute_batch(
            r#"
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
            "#,
        )?;
        transaction.commit()?;
        Ok(())
    }

    pub fn is_empty(&self) -> Result<bool> {
        let connection = self.connection()?;
        let count: i64 = connection.query_row(
            "SELECT (SELECT count(*) FROM document_state) + (SELECT count(*) FROM page_state)",
            [],
            |row| row.get(0),
        )?;
        Ok(count == 0)
    }

    pub fn document_input_fingerprint(&self, document_id: &str) -> Result<Option<String>> {
        let connection = self.connection()?;
        let mut statement = connection
            .prepare("SELECT input_fingerprint FROM document_state WHERE document_id = ?1")?;
        let mut rows = statement.query([document_id])?;
        Ok(match rows.next()? {
            Some(row) => Some(row.get(0)?),
            None => None,
        })
    }

    pub fn ensure_document(&self, identity: &DocumentIdentity) -> Result<()> {
        let mut connection = self.connection()?;
        let transaction = connection.transaction()?;
        let existing: Option<String> = transaction
            .query_row(
                "SELECT input_fingerprint FROM document_state WHERE document_id = ?1",
                [identity.document_id.as_str()],
                |row| row.get(0),
            )
            .optional()?;
        if existing.as_deref() != Some(identity.input_fingerprint.as_str()) {
            for table in [
                "model_attempts",
                "page_state",
                "document_tree_state",
                "grouping_frontier_state",
                "grouping_checkpoint_metadata",
                "summary_attempts",
                "document_state",
            ] {
                transaction.execute(
                    &format!("DELETE FROM {table} WHERE document_id = ?1"),
                    [identity.document_id.as_str()],
                )?;
            }
        }
        transaction.execute(
            r#"
            INSERT INTO document_state (
                document_id, title, source_kind, input_fingerprint, schema_version,
                total_pages, is_completed, last_updated_ts
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0, ?7)
            ON CONFLICT(document_id) DO UPDATE SET
                title = excluded.title,
                source_kind = excluded.source_kind,
                input_fingerprint = excluded.input_fingerprint,
                schema_version = excluded.schema_version,
                total_pages = excluded.total_pages,
                last_updated_ts = excluded.last_updated_ts
            "#,
            rusqlite::params![
                identity.document_id,
                identity.title,
                identity.source_kind,
                identity.input_fingerprint,
                OCR_STATE_SCHEMA_VERSION,
                identity.total_pages,
                unix_time(),
            ],
        )?;
        transaction.commit()?;
        Ok(())
    }

    pub fn record_attempt(
        &self,
        document_id: &str,
        page_number: i64,
        stage: OcrStage,
        request: &OcrAttemptRequest,
    ) -> Result<i64> {
        let mut connection = self.connection()?;
        let transaction = connection.transaction()?;
        let existing: Option<ExistingPageAttempt> = transaction
            .query_row(
                r#"
                SELECT attempt_count, status, content_hash, config_fingerprint, artifact_path, artifact_sha256
                FROM page_state WHERE document_id = ?1 AND page_number = ?2 AND stage = ?3
                "#,
                rusqlite::params![document_id, page_number, stage.as_str()],
                |row| {
                    Ok(ExistingPageAttempt {
                        attempt_count: row.get(0)?,
                        status: row.get(1)?,
                        content_hash: row.get(2)?,
                        config_fingerprint: row.get(3)?,
                        artifact_path: row.get(4)?,
                        artifact_sha256: row.get(5)?,
                    })
                },
            )
            .optional()?;
        if let Some(existing) = existing.as_ref() {
            let artifact_matches = existing
                .artifact_path
                .as_deref()
                .map(Path::new)
                .filter(|path| path.is_file())
                .and_then(|path| sha256_file(path).ok())
                .as_deref()
                == existing.artifact_sha256.as_deref();
            if existing.status == "completed"
                && existing.content_hash == request.content_hash
                && existing.config_fingerprint == request.config_fingerprint
                && existing.artifact_path.is_some()
                && existing.artifact_sha256.is_some()
                && artifact_matches
            {
                transaction.commit()?;
                return Ok(existing.attempt_count);
            }
        }
        let attempt_count = existing.as_ref().map_or(1, |state| state.attempt_count + 1);
        transaction.execute(
            r#"
            INSERT INTO page_state (
                document_id, page_number, stage, attempt_count, status, content_hash,
                artifact_path, last_error, last_model, config_fingerprint,
                artifact_sha256, last_attempted_ts
            ) VALUES (?1, ?2, ?3, ?4, 'pending', ?5, ?6, NULL, ?7, ?8, NULL, ?9)
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
            "#,
            rusqlite::params![
                document_id,
                page_number,
                stage.as_str(),
                attempt_count,
                request.content_hash,
                request.artifact_path.as_deref().and_then(Path::to_str),
                request.model_name,
                request.config_fingerprint,
                unix_time(),
            ],
        )?;
        transaction.execute(
            r#"
            INSERT INTO model_attempts (
                document_id, page_number, stage, attempt_index, model_name, status,
                error_message, attempted_ts
            ) VALUES (?1, ?2, ?3, ?4, ?5, 'pending', NULL, ?6)
            "#,
            rusqlite::params![
                document_id,
                page_number,
                stage.as_str(),
                attempt_count,
                request.model_name,
                unix_time(),
            ],
        )?;
        transaction.commit()?;
        Ok(attempt_count)
    }

    pub fn get_page_state(
        &self,
        document_id: &str,
        page_number: i64,
        stage: OcrStage,
    ) -> Result<Option<OcrPageStateSnapshot>> {
        let connection = self.connection()?;
        let mut statement = connection.prepare(
            r#"
            SELECT page_number, stage, attempt_count, status, content_hash, artifact_path,
                   last_error, last_model, config_fingerprint, artifact_sha256, last_attempted_ts
            FROM page_state WHERE document_id = ?1 AND page_number = ?2 AND stage = ?3
            "#,
        )?;
        let mut rows =
            statement.query(rusqlite::params![document_id, page_number, stage.as_str()])?;
        Ok(match rows.next()? {
            Some(row) => Some(OcrPageStateSnapshot {
                page_number: row.get(0)?,
                stage,
                attempt_count: row.get(2)?,
                status: match row.get::<_, String>(3)?.as_str() {
                    "pending" => OcrPageStatus::Pending,
                    "completed" => OcrPageStatus::Completed,
                    "failed" => OcrPageStatus::Failed,
                    invalid => bail!("unknown OCR page status: {invalid}"),
                },
                content_hash: row.get(4)?,
                artifact_path: row.get(5)?,
                last_error: row.get(6)?,
                last_model: row.get(7)?,
                last_attempted_ts: row.get(10)?,
                config_fingerprint: row.get(8)?,
                artifact_sha256: row.get(9)?,
            }),
            None => None,
        })
    }

    pub fn should_skip_page(
        &self,
        document_id: &str,
        page_number: i64,
        stage: OcrStage,
        content_hash: &str,
        artifact_path: Option<&Path>,
        config_fingerprints: &[&str],
    ) -> Result<bool> {
        let Some(state) = self.get_page_state(document_id, page_number, stage)? else {
            return Ok(false);
        };
        if state.status != OcrPageStatus::Completed
            || state.content_hash.as_deref() != Some(content_hash)
        {
            return Ok(false);
        }
        if !config_fingerprints.is_empty()
            && state
                .config_fingerprint
                .as_deref()
                .is_none_or(|value| !config_fingerprints.contains(&value))
        {
            return Ok(false);
        }
        let Some(artifact_path) = artifact_path else {
            return Ok(stage != OcrStage::Ocr);
        };
        if !artifact_path.is_file() {
            return Ok(false);
        }
        if sha256_file(artifact_path)? != state.artifact_sha256.as_deref().unwrap_or_default() {
            return Ok(false);
        }
        if stage != OcrStage::Ocr {
            return Ok(true);
        }
        let payload = json_object(artifact_path)?;
        Ok(artifact_page_number(&payload) == Some(page_number)
            && metadata_field(&payload, "source_image_sha256") == Some(content_hash)
            && metadata_field(&payload, "ocr_configuration_fingerprint")
                == state.config_fingerprint.as_deref())
    }

    #[allow(clippy::too_many_arguments)]
    fn upsert_page_state(
        &self,
        document_id: &str,
        page_number: i64,
        stage: OcrStage,
        attempt_count: i64,
        status: OcrPageStatus,
        content_hash: Option<String>,
        artifact_path: Option<String>,
        last_error: Option<String>,
        last_model: Option<String>,
        config_fingerprint: Option<String>,
        artifact_sha256: Option<String>,
    ) -> Result<()> {
        let mut connection = self.connection()?;
        let transaction = connection.transaction()?;
        transaction.execute(
            r#"
            INSERT INTO page_state (
                document_id, page_number, stage, attempt_count, status, content_hash,
                artifact_path, last_error, last_model, config_fingerprint,
                artifact_sha256, last_attempted_ts
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
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
            "#,
            rusqlite::params![
                document_id,
                page_number,
                stage.as_str(),
                attempt_count,
                status_str(status),
                content_hash,
                artifact_path,
                last_error,
                last_model,
                config_fingerprint,
                artifact_sha256,
                unix_time(),
            ],
        )?;
        transaction.commit()?;
        Ok(())
    }

    pub fn publish_ocr_page(
        &self,
        document_id: &str,
        publication: &OcrPagePublication,
    ) -> Result<bool> {
        validate_publication(publication)?;
        write_atomic(
            publication.artifact_path,
            &serde_json::to_string_pretty(&publication.payload)?,
        )?;
        let artifact_sha256 = sha256_file(publication.artifact_path)?;
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let updated = transaction.execute(
            r#"
            UPDATE page_state
            SET status = 'completed', artifact_path = ?1, last_error = NULL,
                last_model = ?2, config_fingerprint = ?3, artifact_sha256 = ?4,
                last_attempted_ts = ?5
            WHERE document_id = ?6 AND page_number = ?7 AND stage = 'ocr'
              AND status = 'pending'
              AND content_hash = ?8 AND config_fingerprint = ?3
              AND attempt_count = ?9
            "#,
            rusqlite::params![
                publication.artifact_path.to_string_lossy(),
                publication.model_name,
                publication.config_fingerprint,
                artifact_sha256,
                unix_time(),
                document_id,
                publication.page_number,
                publication.content_hash,
                publication.attempt_index,
            ],
        )?;
        if updated == 1 {
            transaction.execute(
                r#"
                UPDATE model_attempts
                SET status = 'failed', error_message = 'superseded by accepted duplicate'
                WHERE document_id = ?1 AND page_number = ?2 AND stage = 'ocr'
                  AND status = 'pending' AND attempt_index <> ?3
                "#,
                rusqlite::params![
                    document_id,
                    publication.page_number,
                    publication.attempt_index,
                ],
            )?;
            transaction.execute(
                r#"
                UPDATE model_attempts
                SET status = 'completed', error_message = NULL, retry_feedback = NULL
                WHERE document_id = ?1 AND page_number = ?2 AND stage = 'ocr' AND attempt_index = ?3
                "#,
                rusqlite::params![
                    document_id,
                    publication.page_number,
                    publication.attempt_index,
                ],
            )?;
        }
        transaction.commit()?;
        Ok(updated == 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(fingerprint: &str) -> DocumentIdentity {
        DocumentIdentity {
            document_id: "doc".to_string(),
            title: "Doc".to_string(),
            source_kind: "image".to_string(),
            input_fingerprint: fingerprint.to_string(),
            total_pages: 1,
        }
    }

    #[test]
    fn ensure_document_clears_mismatched_state() {
        let temp = tempfile::tempdir().unwrap();
        let store = OcrWorkflowStateStore::open(&temp.path().join("ocr-state.sqlite")).unwrap();
        store.ensure_document(&identity("one")).unwrap();
        store
            .record_attempt(
                "doc",
                1,
                OcrStage::Ocr,
                &OcrAttemptRequest {
                    content_hash: Some("hash".to_string()),
                    model_name: "model".to_string(),
                    config_fingerprint: Some("config".to_string()),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(
            store.document_input_fingerprint("doc").unwrap().as_deref(),
            Some("one")
        );
        store.ensure_document(&identity("two")).unwrap();
        assert_eq!(
            store.document_input_fingerprint("doc").unwrap().as_deref(),
            Some("two")
        );
        assert!(store
            .get_page_state("doc", 1, OcrStage::Ocr)
            .unwrap()
            .is_none());
    }

    #[test]
    fn pending_attempt_requires_artifact_identity_before_skip() {
        let temp = tempfile::tempdir().unwrap();
        let artifact = temp.path().join("page_1.json");
        std::fs::write(
            &artifact,
            serde_json::json!({
                "metadata": {
                    "ocr_json_version": "0.1",
                    "ocr_model_name": "model",
                    "source_image_sha256": "hash",
                    "ocr_configuration_fingerprint": "config",
                },
                "pdf_page_num": 1,
                "OCR_text_clusters": [{"text": "page"}],
                "non_text_objects": [],
            })
            .to_string(),
        )
        .unwrap();
        let store = OcrWorkflowStateStore::open(&temp.path().join("ocr-state.sqlite")).unwrap();
        store.ensure_document(&identity("one")).unwrap();
        store
            .record_attempt(
                "doc",
                1,
                OcrStage::Ocr,
                &OcrAttemptRequest {
                    content_hash: Some("hash".to_string()),
                    model_name: "model".to_string(),
                    artifact_path: Some(artifact.clone()),
                    config_fingerprint: Some("config".to_string()),
                },
            )
            .unwrap();
        let state = store
            .get_page_state("doc", 1, OcrStage::Ocr)
            .unwrap()
            .unwrap();
        assert_eq!(state.status, OcrPageStatus::Pending);
        std::fs::write(
            &artifact,
            serde_json::json!({
                "metadata": {"ocr_json_version": "0.1", "ocr_model_name": "model"},
                "pdf_page_num": 1,
                "OCR_text_clusters": [{"text": "page"}],
                "non_text_objects": [],
            })
            .to_string(),
        )
        .unwrap();
        assert!(!store
            .should_skip_page(
                "doc",
                1,
                OcrStage::Ocr,
                "hash",
                Some(&artifact),
                &["config"]
            )
            .unwrap());
    }

    #[test]
    fn publish_ocr_page_accepts_pending_identity_and_enables_skip() {
        let temp = tempfile::tempdir().unwrap();
        let artifact = temp.path().join("page_1.json");
        let store = OcrWorkflowStateStore::open(&temp.path().join("ocr-state.sqlite")).unwrap();
        store.ensure_document(&identity("one")).unwrap();
        let attempt = store
            .record_attempt(
                "doc",
                1,
                OcrStage::Ocr,
                &OcrAttemptRequest {
                    content_hash: Some("hash".to_string()),
                    model_name: "model".to_string(),
                    config_fingerprint: Some("config".to_string()),
                    ..Default::default()
                },
            )
            .unwrap();
        let accepted = store
            .publish_ocr_page(
                "doc",
                &OcrPagePublication {
                    page_number: 1,
                    content_hash: "hash",
                    artifact_path: &artifact,
                    payload: serde_json::json!({
                        "metadata": {
                            "ocr_json_version": "0.1",
                            "ocr_model_name": "model",
                            "source_image_sha256": "hash",
                            "ocr_configuration_fingerprint": "config",
                        },
                        "pdf_page_num": 1,
                        "OCR_text_clusters": [{"text": "page"}],
                        "non_text_objects": [],
                    }),
                    model_name: "model",
                    attempt_index: attempt,
                    config_fingerprint: "config",
                },
            )
            .unwrap();
        assert!(accepted);
        assert!(store
            .should_skip_page(
                "doc",
                1,
                OcrStage::Ocr,
                "hash",
                Some(&artifact),
                &["config"]
            )
            .unwrap());
    }

    #[test]
    fn rebuilds_render_and_ocr_state_from_matching_artifacts() {
        let temp = tempfile::tempdir().unwrap();
        let rendered = temp.path().join("rendered_pages");
        let legacy = temp.path().join("legacy_split_pages");
        std::fs::create_dir_all(&rendered).unwrap();
        std::fs::create_dir_all(&legacy).unwrap();
        let rendered_page = rendered.join("page_1.png");
        std::fs::write(&rendered_page, b"page image").unwrap();
        let rendered_hash = sha256_file(&rendered_page).unwrap();
        let artifact = legacy.join("page_1.json");
        std::fs::write(
            &artifact,
            serde_json::json!({
                "metadata": {
                    "ocr_json_version": "0.1",
                    "ocr_model_name": "model",
                    "source_image_sha256": rendered_hash,
                    "ocr_configuration_fingerprint": "config",
                },
                "pdf_page_num": 1,
                "OCR_text_clusters": [{"text": "page"}],
                "non_text_objects": [],
            })
            .to_string(),
        )
        .unwrap();
        let store = OcrWorkflowStateStore::open(&temp.path().join("ocr-state.sqlite")).unwrap();
        let rebuilt = store
            .rebuild_from_artifacts(&identity("one"), &rendered, &legacy, "render-v1")
            .unwrap();
        assert!(rebuilt);
        assert_eq!(
            store
                .get_page_state("doc", 1, OcrStage::Render)
                .unwrap()
                .unwrap()
                .status,
            OcrPageStatus::Completed
        );
        assert_eq!(
            store
                .get_page_state("doc", 1, OcrStage::Ocr)
                .unwrap()
                .unwrap()
                .status,
            OcrPageStatus::Completed
        );
        assert!(store
            .should_skip_page(
                "doc",
                1,
                OcrStage::Ocr,
                &rendered_hash,
                Some(&artifact),
                &["config"]
            )
            .unwrap());
    }
}
