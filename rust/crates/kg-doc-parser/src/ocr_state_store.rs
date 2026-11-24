use anyhow::{bail, Result};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

use crate::document_models::{DocumentTree, Summary, SummaryStatus};
use crate::ocr_artifacts::artifact_schema_error;
use crate::semantic_indexing::json_compact;
use std::collections::{BTreeMap, BTreeSet};

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

#[derive(Debug, Clone, Serialize)]
pub struct OcrProgressPage {
    pub page_number: i64,
    pub image_path: String,
    pub image_sha256: String,
    pub json_path: String,
    pub status: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct OcrProgressPayload {
    pub document_id: String,
    pub title: String,
    pub source_kind: String,
    pub total_pages: i64,
    pub pages: BTreeMap<String, OcrProgressPage>,
}

#[derive(Debug, Clone)]
pub struct SummaryAttemptRequest {
    pub document_id: String,
    pub node_id: String,
    pub input_fingerprint: String,
    pub status: SummaryStatus,
    pub summary: Option<Summary>,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SummaryAttemptRecord {
    pub node_id: String,
    pub input_fingerprint: String,
    pub attempt_index: i64,
    pub status: String,
    pub summary_json: Option<Value>,
    pub error_message: Option<String>,
    pub last_updated_ts: f64,
}

#[derive(Debug, Clone)]
pub struct GroupingCheckpointSave<'a> {
    pub document_id: &'a str,
    pub tree: &'a DocumentTree,
    pub grouping_policy_version: &'a str,
    pub expected_tree_revision: Option<&'a str>,
    pub require_no_existing_tree: bool,
    pub last_runtime_step: &'a str,
}

#[derive(Debug, Clone, Serialize)]
pub struct GroupingCheckpointMetadata {
    pub document_id: String,
    pub source_fingerprint: String,
    pub grouping_policy_version: String,
    pub accepted_page_frontier: Vec<i64>,
    pub tree_revision: String,
    pub merge_revision: i64,
    pub last_runtime_step: String,
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

fn sha256_hex(contents: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(contents);
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn summary_status_str(status: SummaryStatus) -> &'static str {
    match status {
        SummaryStatus::Pending => "pending",
        SummaryStatus::Generated => "generated",
        SummaryStatus::Verified => "verified",
        SummaryStatus::Failed => "failed",
        SummaryStatus::Skipped => "skipped",
    }
}

fn validate_summary(summary: Option<&Summary>, status: SummaryStatus) -> Result<()> {
    let Some(summary) = summary else {
        if matches!(status, SummaryStatus::Generated | SummaryStatus::Verified) {
            bail!("generated summaries require a Summary payload");
        }
        return Ok(());
    };
    let generated = matches!(status, SummaryStatus::Generated | SummaryStatus::Verified);
    if generated
        && (summary.text.as_deref().unwrap_or_default().is_empty()
            || summary.evidence_item_ids.is_empty()
            || summary.provider_model.is_none()
            || summary.prompt_revision.is_none()
            || summary.input_fingerprint.is_none())
    {
        bail!("generated summaries require text, evidence, and provenance");
    }
    if !generated && summary.text.is_some() {
        bail!("unavailable summaries cannot contain text");
    }
    if summary
        .evidence_item_ids
        .iter()
        .collect::<BTreeSet<_>>()
        .len()
        != summary.evidence_item_ids.len()
    {
        bail!("summary evidence items must be unique");
    }
    if summary
        .generation_fingerprint
        .as_deref()
        .is_some_and(str::is_empty)
    {
        bail!("summary generation fingerprint must not be blank");
    }
    Ok(())
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

    pub fn mark_document_completed(&self, document_id: &str, is_completed: bool) -> Result<()> {
        let mut connection = self.connection()?;
        let transaction = connection.transaction()?;
        transaction.execute(
            "UPDATE document_state SET is_completed = ?1, last_updated_ts = ?2 WHERE document_id = ?3",
            rusqlite::params![i64::from(is_completed), unix_time(), document_id],
        )?;
        transaction.commit()?;
        Ok(())
    }

    pub fn refresh_document_completion(&self, document_id: &str) -> Result<bool> {
        let connection = self.connection()?;
        let total_pages: i64 = connection
            .query_row(
                "SELECT total_pages FROM document_state WHERE document_id = ?1",
                [document_id],
                |row| row.get(0),
            )
            .optional()?
            .unwrap_or(0);
        if total_pages < 0 {
            bail!("OCR state total_pages cannot be negative");
        }
        let mut statement = connection.prepare(
            "SELECT page_number FROM page_state WHERE document_id = ?1 AND stage = 'ocr' AND status = 'completed'",
        )?;
        let mut rows = statement.query([document_id])?;
        let mut completed_pages = BTreeSet::new();
        while let Some(row) = rows.next()? {
            let page_number: i64 = row.get(0)?;
            if page_number < 1 {
                bail!("OCR state page numbers must be positive");
            }
            completed_pages.insert(page_number);
        }
        let expected_pages: BTreeSet<i64> = (1..=total_pages).collect();
        let is_completed = total_pages > 0 && completed_pages == expected_pages;
        self.mark_document_completed(document_id, is_completed)?;
        Ok(is_completed)
    }

    pub fn export_progress_payload(
        &self,
        document_id: &str,
        title: &str,
        source_kind: &str,
        total_pages: i64,
    ) -> Result<OcrProgressPayload> {
        let connection = self.connection()?;
        let mut statement = connection.prepare(
            r#"
            SELECT page_number, content_hash, artifact_path, status
            FROM page_state WHERE document_id = ?1 AND stage = 'ocr' ORDER BY page_number
            "#,
        )?;
        let mut rows = statement.query([document_id])?;
        let mut pages = BTreeMap::new();
        while let Some(row) = rows.next()? {
            let artifact_path: Option<String> = row.get(2)?;
            let Some(artifact_path) = artifact_path else {
                continue;
            };
            let page_number: i64 = row.get(0)?;
            if page_number < 1 {
                bail!("OCR state page numbers must be positive");
            }
            pages.insert(
                page_number.to_string(),
                OcrProgressPage {
                    page_number,
                    image_path: String::new(),
                    image_sha256: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                    json_path: artifact_path,
                    status: match row.get::<_, String>(3)?.as_str() {
                        "pending" => "pending",
                        "completed" => "completed",
                        "failed" => "failed",
                        invalid => bail!("unknown OCR page status: {invalid}"),
                    },
                },
            );
        }
        Ok(OcrProgressPayload {
            document_id: document_id.to_string(),
            title: title.to_string(),
            source_kind: source_kind.to_string(),
            total_pages,
            pages,
        })
    }

    pub fn read_document_completed(&self, document_id: &str) -> Result<bool> {
        let connection = self.connection()?;
        let is_completed: Option<i64> = connection
            .query_row(
                "SELECT is_completed FROM document_state WHERE document_id = ?1",
                [document_id],
                |row| row.get(0),
            )
            .optional()?;
        Ok(match is_completed {
            Some(0) => false,
            Some(1) => true,
            Some(value) => bail!("OCR state is_completed must be 0 or 1: {value}"),
            None => false,
        })
    }

    pub fn save_document_tree(&self, request: &GroupingCheckpointSave) -> Result<String> {
        request.tree.validate()?;
        if request.require_no_existing_tree && request.expected_tree_revision.is_some() {
            bail!("absence and expected revision fences are mutually exclusive");
        }
        if request.last_runtime_step.is_empty() {
            bail!("last_runtime_step must not be empty");
        }
        let tree_json = json_compact(&serde_json::to_value(request.tree)?);
        let tree_revision = sha256_hex(tree_json.as_bytes());
        let source_fingerprint = request.tree.source_fingerprint();
        let root = request
            .tree
            .nodes
            .iter()
            .find(|node| node.node_id == request.tree.root_id)
            .ok_or_else(|| anyhow::anyhow!("document root is missing"))?;
        let expected_frontier: Vec<String> = if root.children.is_empty() {
            vec![root.node_id.clone()]
        } else {
            root.children
                .iter()
                .map(|child| child.node_id.clone())
                .collect()
        };
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let current_document: Option<String> = transaction
            .query_row(
                "SELECT input_fingerprint FROM document_state WHERE document_id = ?1",
                [request.document_id],
                |row| row.get(0),
            )
            .optional()?;
        if let Some(input_fingerprint) = current_document {
            let expected_source_revision = format!("ocr-v1:{input_fingerprint}");
            if request.tree.source_revision.starts_with("ocr-v1:")
                && request.tree.source_revision != expected_source_revision
            {
                bail!(
                    "grouping checkpoint source revision conflict for {:?}: expected {:?}, found {:?}",
                    request.document_id,
                    expected_source_revision,
                    request.tree.source_revision
                );
            }
        }
        let current: Option<(String, String, String)> = transaction
            .query_row(
                r#"
                SELECT tree_revision, source_fingerprint, grouping_policy_version
                FROM document_tree_state WHERE document_id = ?1
                "#,
                [request.document_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;
        let current_frontier: Option<(String, String, String)> = transaction
            .query_row(
                r#"
                SELECT source_fingerprint, grouping_policy_version, frontier_json
                FROM grouping_frontier_state WHERE document_id = ?1
                "#,
                [request.document_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;
        let current_metadata: Option<(String, String, String, String, i64, String)> = transaction
            .query_row(
                r#"
                SELECT source_fingerprint, grouping_policy_version, accepted_page_frontier_json,
                       tree_revision, merge_revision, last_runtime_step
                FROM grouping_checkpoint_metadata WHERE document_id = ?1
                "#,
                [request.document_id],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                    ))
                },
            )
            .optional()?;
        let mut statement = transaction.prepare(
            "SELECT page_number FROM page_state WHERE document_id = ?1 AND stage = 'ocr' AND status = 'completed' ORDER BY page_number",
        )?;
        let mut rows = statement.query([request.document_id])?;
        let mut accepted_pages = Vec::new();
        while let Some(row) = rows.next()? {
            accepted_pages.push(row.get::<_, i64>(0)?);
        }
        drop(rows);
        drop(statement);
        let current_revision = current.as_ref().map(|(revision, _, _)| revision.clone());
        let has_frontier = current_frontier.is_some();
        let frontier_matches =
            current_frontier
                .as_ref()
                .is_some_and(|(source, policy, payload)| {
                    source == &source_fingerprint
                        && policy == request.grouping_policy_version
                        && serde_json::from_str::<Vec<String>>(payload)
                            .ok()
                            .map(|frontier| frontier == expected_frontier)
                            .unwrap_or(false)
                });
        let metadata_matches = current_metadata.as_ref().is_some_and(
            |(source, policy, payload, revision, merge_revision, last_step)| {
                source == &source_fingerprint
                    && policy == request.grouping_policy_version
                    && revision == &tree_revision
                    && serde_json::from_str::<Vec<i64>>(payload)
                        .ok()
                        .map(|pages| pages == accepted_pages && pages.iter().all(|page| *page >= 1))
                        .unwrap_or(false)
                    && *merge_revision >= 0
                    && !last_step.is_empty()
            },
        );
        if current_revision.as_deref() == Some(tree_revision.as_str()) {
            if !has_frontier {
                bail!(
                    "grouping checkpoint frontier is missing for {:?}",
                    request.document_id
                );
            }
            if current_metadata.is_none() {
                bail!(
                    "grouping checkpoint metadata is missing for {:?}",
                    request.document_id
                );
            }
            if current.as_ref().is_some_and(|(_, source, policy)| {
                source == &source_fingerprint && policy == request.grouping_policy_version
            }) && frontier_matches
                && metadata_matches
            {
                transaction.commit()?;
                return Ok(tree_revision);
            }
            bail!(
                "idempotent grouping checkpoint replay for {:?} found mismatched companion records",
                request.document_id
            );
        }
        let merge_revision = match current_metadata.as_ref() {
            None => 1,
            Some(metadata) => metadata.4 + 1,
        };
        if current_revision.is_none() && (has_frontier || current_metadata.is_some()) {
            bail!(
                "grouping checkpoint create conflict for {:?}: found orphaned frontier or metadata without a tree",
                request.document_id
            );
        }
        if request.require_no_existing_tree
            && (current_revision.is_some() || has_frontier || current_metadata.is_some())
        {
            bail!(
                "grouping checkpoint create conflict for {:?}: found existing revision {:?}, frontier, or metadata",
                request.document_id,
                current_revision
            );
        }
        if let Some(expected) = request.expected_tree_revision {
            if current_revision.as_deref() != Some(expected) {
                bail!(
                    "grouping checkpoint revision conflict for {:?}: expected {:?}, found {:?}",
                    request.document_id,
                    expected,
                    current_revision
                );
            }
        }
        if current_revision.is_some() && request.expected_tree_revision.is_none() {
            bail!(
                "grouping checkpoint update for {:?} requires an expected tree revision",
                request.document_id
            );
        }
        transaction.execute(
            r#"
            INSERT INTO document_tree_state (
                document_id, tree_revision, source_fingerprint, grouping_policy_version,
                tree_json, last_updated_ts
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            ON CONFLICT(document_id) DO UPDATE SET
                tree_revision = excluded.tree_revision,
                source_fingerprint = excluded.source_fingerprint,
                grouping_policy_version = excluded.grouping_policy_version,
                tree_json = excluded.tree_json,
                last_updated_ts = excluded.last_updated_ts
            "#,
            rusqlite::params![
                request.document_id,
                tree_revision,
                source_fingerprint,
                request.grouping_policy_version,
                tree_json,
                unix_time(),
            ],
        )?;
        transaction.execute(
            r#"
            INSERT INTO grouping_frontier_state (
                document_id, source_fingerprint, grouping_policy_version, frontier_json,
                last_updated_ts
            ) VALUES (?1, ?2, ?3, ?4, ?5)
            ON CONFLICT(document_id) DO UPDATE SET
                source_fingerprint = excluded.source_fingerprint,
                grouping_policy_version = excluded.grouping_policy_version,
                frontier_json = excluded.frontier_json,
                last_updated_ts = excluded.last_updated_ts
            "#,
            rusqlite::params![
                request.document_id,
                source_fingerprint,
                request.grouping_policy_version,
                serde_json::to_string(&expected_frontier)?,
                unix_time(),
            ],
        )?;
        transaction.execute(
            r#"
            INSERT INTO grouping_checkpoint_metadata (
                document_id, source_fingerprint, grouping_policy_version,
                accepted_page_frontier_json, tree_revision, merge_revision,
                last_runtime_step, last_updated_ts
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            ON CONFLICT(document_id) DO UPDATE SET
                source_fingerprint = excluded.source_fingerprint,
                grouping_policy_version = excluded.grouping_policy_version,
                accepted_page_frontier_json = excluded.accepted_page_frontier_json,
                tree_revision = excluded.tree_revision,
                merge_revision = excluded.merge_revision,
                last_runtime_step = excluded.last_runtime_step,
                last_updated_ts = excluded.last_updated_ts
            "#,
            rusqlite::params![
                request.document_id,
                source_fingerprint,
                request.grouping_policy_version,
                serde_json::to_string(&accepted_pages)?,
                tree_revision,
                merge_revision,
                request.last_runtime_step,
                unix_time(),
            ],
        )?;
        transaction.commit()?;
        Ok(tree_revision)
    }

    pub fn load_grouping_checkpoint_metadata(
        &self,
        document_id: &str,
        source_fingerprint: Option<&str>,
        grouping_policy_version: Option<&str>,
    ) -> Result<Option<GroupingCheckpointMetadata>> {
        let connection = self.connection()?;
        let Some((
            stored_source,
            stored_policy,
            payload,
            tree_revision,
            merge_revision,
            last_runtime_step,
        )) = connection
            .query_row(
                r#"
                SELECT source_fingerprint, grouping_policy_version, accepted_page_frontier_json,
                       tree_revision, merge_revision, last_runtime_step
                FROM grouping_checkpoint_metadata WHERE document_id = ?1
                "#,
                [document_id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, i64>(4)?,
                        row.get::<_, String>(5)?,
                    ))
                },
            )
            .optional()?
        else {
            return Ok(None);
        };
        if source_fingerprint.is_some_and(|value| value != stored_source)
            || grouping_policy_version.is_some_and(|value| value != stored_policy)
        {
            return Ok(None);
        }
        let accepted_pages: Vec<i64> = serde_json::from_str(&payload).map_err(|error| {
            anyhow::anyhow!("persisted grouping checkpoint page frontier is malformed: {error}")
        })?;
        if accepted_pages.iter().any(|page| *page < 1) {
            bail!("persisted grouping checkpoint page frontier is malformed");
        }
        if accepted_pages
            .windows(2)
            .any(|window| window[0] >= window[1])
        {
            bail!("persisted grouping checkpoint page frontier is not ordered");
        }
        Ok(Some(GroupingCheckpointMetadata {
            document_id: document_id.to_string(),
            source_fingerprint: stored_source,
            grouping_policy_version: stored_policy,
            accepted_page_frontier: accepted_pages,
            tree_revision,
            merge_revision,
            last_runtime_step,
        }))
    }

    pub fn document_tree_revision(&self, document_id: &str) -> Result<Option<String>> {
        let connection = self.connection()?;
        Ok(connection
            .query_row(
                "SELECT tree_revision FROM document_tree_state WHERE document_id = ?1",
                [document_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?)
    }

    pub fn load_document_tree(
        &self,
        document_id: &str,
        source_fingerprint: Option<&str>,
        grouping_policy_version: Option<&str>,
    ) -> Result<Option<DocumentTree>> {
        let connection = self.connection()?;
        let Some((stored_source, stored_policy, tree_json)) = connection
            .query_row(
                r#"
                SELECT source_fingerprint, grouping_policy_version, tree_json
                FROM document_tree_state WHERE document_id = ?1
                "#,
                [document_id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                },
            )
            .optional()?
        else {
            return Ok(None);
        };
        if source_fingerprint.is_some_and(|value| value != stored_source)
            || grouping_policy_version.is_some_and(|value| value != stored_policy)
        {
            return Ok(None);
        }
        let Some((frontier_source, frontier_policy, frontier_json)) = connection
            .query_row(
                r#"
                SELECT source_fingerprint, grouping_policy_version, frontier_json
                FROM grouping_frontier_state WHERE document_id = ?1
                "#,
                [document_id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                },
            )
            .optional()?
        else {
            bail!("persisted grouping checkpoint is missing a companion record");
        };
        let Some((metadata_source, metadata_policy, accepted_json, metadata_revision, _)) =
            connection
                .query_row(
                    r#"
                SELECT source_fingerprint, grouping_policy_version, accepted_page_frontier_json,
                       tree_revision, last_runtime_step
                FROM grouping_checkpoint_metadata WHERE document_id = ?1
                "#,
                    [document_id],
                    |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, String>(2)?,
                            row.get::<_, String>(3)?,
                            row.get::<_, String>(4)?,
                        ))
                    },
                )
                .optional()?
        else {
            bail!("persisted grouping checkpoint is missing a companion record");
        };
        if metadata_revision != sha256_hex(tree_json.as_bytes())
            || metadata_source != stored_source
            || metadata_policy != stored_policy
            || frontier_source != stored_source
            || frontier_policy != stored_policy
        {
            bail!("persisted grouping checkpoint records disagree");
        }
        let tree: DocumentTree = serde_json::from_str(&tree_json)?;
        tree.validate()?;
        if tree.source_fingerprint() != stored_source {
            bail!("persisted grouping tree source fingerprint is inconsistent");
        }
        let root = tree
            .nodes
            .iter()
            .find(|node| node.node_id == tree.root_id)
            .ok_or_else(|| anyhow::anyhow!("persisted grouping tree root is missing"))?;
        let expected_frontier: Vec<String> = if root.children.is_empty() {
            vec![root.node_id.clone()]
        } else {
            root.children
                .iter()
                .map(|child| child.node_id.clone())
                .collect()
        };
        let frontier: Vec<String> = serde_json::from_str(&frontier_json)?;
        if frontier != expected_frontier {
            bail!("persisted grouping frontier does not match its tree");
        }
        let accepted_pages: Vec<i64> = serde_json::from_str(&accepted_json)?;
        let mut statement = connection.prepare(
            "SELECT page_number FROM page_state WHERE document_id = ?1 AND stage = 'ocr' AND status = 'completed' ORDER BY page_number",
        )?;
        let mut rows = statement.query([document_id])?;
        let mut completed_pages = Vec::new();
        while let Some(row) = rows.next()? {
            completed_pages.push(row.get::<_, i64>(0)?);
        }
        if accepted_pages != completed_pages {
            bail!("persisted grouping accepted-page frontier does not match completed OCR pages");
        }
        Ok(Some(tree))
    }

    pub fn record_summary_attempt(&self, request: &SummaryAttemptRequest) -> Result<i64> {
        validate_summary(request.summary.as_ref(), request.status)?;
        let summary_json = match request.summary.as_ref() {
            Some(summary) => Some(json_compact(&serde_json::to_value(summary)?)),
            None => None,
        };
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        if matches!(
            request.status,
            SummaryStatus::Generated | SummaryStatus::Verified
        ) {
            let accepted: Option<i64> = transaction
                .query_row(
                    r#"
                    SELECT attempt_index FROM summary_attempts
                    WHERE document_id = ?1 AND node_id = ?2 AND input_fingerprint = ?3
                      AND status IN ('generated', 'verified')
                    ORDER BY attempt_index DESC LIMIT 1
                    "#,
                    rusqlite::params![
                        request.document_id,
                        request.node_id,
                        request.input_fingerprint
                    ],
                    |row| row.get(0),
                )
                .optional()?;
            if let Some(attempt_index) = accepted {
                transaction.commit()?;
                return Ok(attempt_index);
            }
        }
        let last_attempt: i64 = transaction.query_row(
            r#"
            SELECT COALESCE(MAX(attempt_index), 0) FROM summary_attempts
            WHERE document_id = ?1 AND node_id = ?2 AND input_fingerprint = ?3
            "#,
            rusqlite::params![
                request.document_id,
                request.node_id,
                request.input_fingerprint
            ],
            |row| row.get(0),
        )?;
        let attempt_index = last_attempt + 1;
        transaction.execute(
            r#"
            INSERT INTO summary_attempts (
                document_id, node_id, input_fingerprint, attempt_index, status,
                summary_json, error_message, last_updated_ts
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            "#,
            rusqlite::params![
                request.document_id,
                request.node_id,
                request.input_fingerprint,
                attempt_index,
                summary_status_str(request.status),
                summary_json,
                request.error_message,
                unix_time(),
            ],
        )?;
        transaction.commit()?;
        Ok(attempt_index)
    }

    pub fn load_completed_summary(
        &self,
        document_id: &str,
        node_id: &str,
        input_fingerprint: &str,
    ) -> Result<Option<Summary>> {
        let connection = self.connection()?;
        let summary_json: Option<Option<String>> = connection
            .query_row(
                r#"
                SELECT summary_json FROM summary_attempts
                WHERE document_id = ?1 AND node_id = ?2 AND input_fingerprint = ?3
                  AND status IN ('generated', 'verified') AND summary_json IS NOT NULL
                ORDER BY attempt_index DESC LIMIT 1
                "#,
                [document_id, node_id, input_fingerprint],
                |row| row.get(0),
            )
            .optional()?;
        let Some(Some(summary_json)) = summary_json else {
            return Ok(None);
        };
        let summary: Summary = match serde_json::from_str(&summary_json) {
            Ok(summary) => summary,
            Err(_) => return Ok(None),
        };
        Ok(validate_summary(Some(&summary), summary.status)
            .ok()
            .and(Some(summary)))
    }

    pub fn list_summary_attempts(&self, document_id: &str) -> Result<Vec<SummaryAttemptRecord>> {
        let connection = self.connection()?;
        let mut statement = connection.prepare(
            r#"
            SELECT node_id, input_fingerprint, attempt_index, status, summary_json,
                   error_message, last_updated_ts
            FROM summary_attempts WHERE document_id = ?1
            ORDER BY node_id, input_fingerprint, attempt_index
            "#,
        )?;
        let mut rows = statement.query([document_id])?;
        let mut records = Vec::new();
        while let Some(row) = rows.next()? {
            records.push(SummaryAttemptRecord {
                node_id: row.get(0)?,
                input_fingerprint: row.get(1)?,
                attempt_index: row.get(2)?,
                status: row.get(3)?,
                summary_json: row
                    .get::<_, Option<String>>(4)?
                    .and_then(|payload| serde_json::from_str(&payload).ok()),
                error_message: row.get(5)?,
                last_updated_ts: row.get(6)?,
            });
        }
        Ok(records)
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

    #[test]
    fn refreshes_completion_and_exports_progress_payload() {
        let temp = tempfile::tempdir().unwrap();
        let store = OcrWorkflowStateStore::open(&temp.path().join("ocr-state.sqlite")).unwrap();
        let mut complete_identity = identity("one");
        complete_identity.total_pages = 2;
        store.ensure_document(&complete_identity).unwrap();
        assert!(!store.refresh_document_completion("doc").unwrap());
        store
            .upsert_page_state(
                "doc",
                1,
                OcrStage::Ocr,
                1,
                OcrPageStatus::Completed,
                Some("hash".to_string()),
                Some("p1".to_string()),
                None,
                None,
                None,
                None,
            )
            .unwrap();
        store
            .upsert_page_state(
                "doc",
                2,
                OcrStage::Ocr,
                1,
                OcrPageStatus::Pending,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();
        assert!(!store.refresh_document_completion("doc").unwrap());
        store
            .upsert_page_state(
                "doc",
                2,
                OcrStage::Ocr,
                1,
                OcrPageStatus::Completed,
                Some("hash".to_string()),
                Some("p2".to_string()),
                None,
                None,
                None,
                None,
            )
            .unwrap();
        assert!(store.refresh_document_completion("doc").unwrap());
        assert!(store.read_document_completed("doc").unwrap());
        let progress = store
            .export_progress_payload("doc", "Doc", "image", 2)
            .unwrap();
        assert_eq!(progress.total_pages, 2);
        assert_eq!(progress.pages["1"].status, "completed");
        assert_eq!(progress.pages["2"].json_path, "p2");
    }

    #[test]
    fn summary_journal_is_idempotent_and_validates_payloads() {
        let temp = tempfile::tempdir().unwrap();
        let store = OcrWorkflowStateStore::open(&temp.path().join("ocr-state.sqlite")).unwrap();
        store.ensure_document(&identity("one")).unwrap();
        let summary = Summary {
            status: SummaryStatus::Generated,
            text: Some("summary".to_string()),
            evidence_item_ids: vec!["item".to_string()],
            provider_model: Some("model".to_string()),
            prompt_revision: Some("prompt".to_string()),
            generation_fingerprint: Some("generation".to_string()),
            input_fingerprint: Some("input".to_string()),
            evidence_slice_fingerprint: None,
        };
        let request = SummaryAttemptRequest {
            document_id: "doc".to_string(),
            node_id: "node".to_string(),
            input_fingerprint: "input".to_string(),
            status: SummaryStatus::Generated,
            summary: Some(summary.clone()),
            error_message: None,
        };
        assert_eq!(store.record_summary_attempt(&request).unwrap(), 1);
        assert_eq!(store.record_summary_attempt(&request).unwrap(), 1);
        let failed = SummaryAttemptRequest {
            document_id: "doc".to_string(),
            node_id: "node".to_string(),
            input_fingerprint: "input".to_string(),
            status: SummaryStatus::Failed,
            summary: None,
            error_message: Some("provider failed".to_string()),
        };
        assert_eq!(store.record_summary_attempt(&failed).unwrap(), 2);
        assert_eq!(store.record_summary_attempt(&failed).unwrap(), 3);
        let loaded = store
            .load_completed_summary("doc", "node", "input")
            .unwrap();
        assert_eq!(
            serde_json::to_value(loaded).unwrap(),
            serde_json::to_value(&summary).unwrap()
        );
        assert_eq!(store.list_summary_attempts("doc").unwrap().len(), 3);
        assert!(store
            .record_summary_attempt(&SummaryAttemptRequest {
                document_id: "doc".to_string(),
                node_id: "node".to_string(),
                input_fingerprint: "input".to_string(),
                status: SummaryStatus::Pending,
                summary: Some(summary),
                error_message: None,
            })
            .is_err());
    }

    #[test]
    fn saves_and_loads_grouping_checkpoint() {
        let temp = tempfile::tempdir().unwrap();
        let store = OcrWorkflowStateStore::open(&temp.path().join("ocr-state.sqlite")).unwrap();
        store.ensure_document(&identity("one")).unwrap();
        store
            .upsert_page_state(
                "doc",
                1,
                OcrStage::Ocr,
                1,
                OcrPageStatus::Completed,
                Some("hash".to_string()),
                Some("p1".to_string()),
                None,
                None,
                None,
                None,
            )
            .unwrap();
        let tree = DocumentTree {
            schema_version: "1.0".to_string(),
            document_id: "doc".to_string(),
            source_revision: "ocr-v1:one".to_string(),
            tree_revision: 1,
            root_id: "root".to_string(),
            source_items: vec![crate::document_models::SourceItem {
                item_id: "item".to_string(),
                kind: "text".to_string(),
                location: crate::document_models::SourceLocation {
                    page_index: Some(1),
                    printed_page_label: None,
                    bbox: None,
                    start_seconds: None,
                    end_seconds: None,
                    structural_path: None,
                    accuracy: crate::document_models::SourceAccuracy::Page,
                },
                text: Some("alpha".to_string()),
                media: None,
                ocr_metadata: None,
                annotations: BTreeMap::new(),
            }],
            nodes: vec![
                crate::document_models::DocumentNode {
                    node_id: "root".to_string(),
                    kind: "document".to_string(),
                    parts: Vec::new(),
                    children: vec![crate::document_models::ChildRef {
                        node_id: "leaf".to_string(),
                        separator_before: String::new(),
                    }],
                    summary: None,
                    annotations: BTreeMap::new(),
                },
                crate::document_models::DocumentNode {
                    node_id: "leaf".to_string(),
                    kind: "paragraph".to_string(),
                    parts: vec![crate::document_models::SourceSlice {
                        item_id: "item".to_string(),
                        start: Some(0),
                        end: Some(5),
                        separator_before: String::new(),
                    }],
                    children: Vec::new(),
                    summary: None,
                    annotations: BTreeMap::new(),
                },
            ],
            references: Vec::new(),
        };
        let request = GroupingCheckpointSave {
            document_id: "doc",
            tree: &tree,
            grouping_policy_version: "baseline-v1",
            expected_tree_revision: None,
            require_no_existing_tree: false,
            last_runtime_step: "grouping_checkpoint",
        };
        let revision = store.save_document_tree(&request).unwrap();
        assert_eq!(
            store.document_tree_revision("doc").unwrap().as_deref(),
            Some(revision.as_str())
        );
        assert_eq!(store.save_document_tree(&request).unwrap(), revision);
        let loaded = store
            .load_document_tree("doc", None, Some("baseline-v1"))
            .unwrap()
            .unwrap();
        assert_eq!(
            serde_json::to_value(loaded).unwrap(),
            serde_json::to_value(&tree).unwrap()
        );
        let metadata = store
            .load_grouping_checkpoint_metadata("doc", None, None)
            .unwrap()
            .unwrap();
        assert_eq!(metadata.accepted_page_frontier, vec![1]);
        assert_eq!(metadata.tree_revision, revision);
        let mut updated_tree = tree.clone();
        updated_tree.tree_revision = 2;
        let updated_revision = store
            .save_document_tree(&GroupingCheckpointSave {
                document_id: "doc",
                tree: &updated_tree,
                grouping_policy_version: "baseline-v1",
                expected_tree_revision: Some(revision.as_str()),
                require_no_existing_tree: false,
                last_runtime_step: "grouping_checkpoint",
            })
            .unwrap();
        assert_ne!(updated_revision, revision);
        assert!(store
            .save_document_tree(&GroupingCheckpointSave {
                document_id: "doc",
                tree: &tree,
                grouping_policy_version: "baseline-v1",
                expected_tree_revision: None,
                require_no_existing_tree: false,
                last_runtime_step: "grouping_checkpoint",
            })
            .is_err());
    }
}
