use anyhow::{bail, Result};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

pub const OCR_ROOT_STATE_VERSION: &str = "ocr-root-v3";
pub const OCR_PAGE_STATE_VERSION: &str = "ocr-page-v2";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageSourceState {
    pub page_number: u64,
    pub image_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResumeIdentity {
    String(String),
    Strings(Vec<String>),
}

pub fn page_source_state(value: &Value) -> Result<PageSourceState> {
    let object = value.as_object().ok_or_else(|| {
        anyhow::anyhow!("OCR workflow state page_sources contains a malformed entry")
    })?;
    let Some(page_number) = object.get("page_number").and_then(Value::as_u64) else {
        bail!("OCR workflow state page_sources contains a malformed entry");
    };
    let Some(image_path) = object.get("image_path").and_then(Value::as_str) else {
        bail!("OCR workflow state page_sources contains a malformed entry");
    };
    if image_path.is_empty() {
        bail!("OCR workflow state page_sources contains a malformed entry");
    }
    Ok(PageSourceState {
        page_number,
        image_path: image_path.to_string(),
    })
}

pub fn page_sources(state: &Value) -> Result<Vec<PageSourceState>> {
    let Some(items) = state.get("page_sources").and_then(Value::as_array) else {
        bail!("OCR page runtime state has no page source list");
    };
    items
        .iter()
        .map(page_source_state)
        .collect::<Result<Vec<_>>>()
        .and_then(|sources| {
            if sources.iter().any(|source| source.page_number < 1) {
                bail!("OCR page runtime source page number is malformed");
            }
            Ok(sources)
        })
}

pub fn int_list(state: &Value, key: &str) -> Result<Vec<i64>> {
    let Some(Value::Array(items)) = state.get(key) else {
        bail!("OCR page runtime state field {key:?} is malformed");
    };
    if items.iter().any(|item| !item.is_i64()) {
        bail!("OCR page runtime state field {key:?} is malformed");
    }
    Ok(items.iter().filter_map(Value::as_i64).collect())
}

pub fn state_bool(state: &Value, key: &str, default: bool) -> bool {
    state.get(key).and_then(Value::as_bool).unwrap_or(default)
}

pub fn state_string(state: &Value, key: &str, default: &str) -> String {
    state
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or(default)
        .to_string()
}

pub fn page_cursor(state: &Value) -> u64 {
    state
        .get("page_cursor")
        .and_then(Value::as_u64)
        .unwrap_or(0)
}

pub fn resume_identity_value(state: &Value, key: &str) -> Result<ResumeIdentity> {
    let Some(value) = state.get(key) else {
        bail!("OCR checkpoint identity field {key:?} is missing");
    };
    if let Some(value) = value.as_str() {
        return Ok(ResumeIdentity::String(value.to_string()));
    }
    if key == "ocr_configuration_fingerprints" {
        if let Some(values) = value.as_array() {
            if values.iter().all(Value::is_string) {
                return Ok(ResumeIdentity::Strings(
                    values
                        .iter()
                        .filter_map(Value::as_str)
                        .map(ToString::to_string)
                        .collect(),
                ));
            }
        }
    }
    bail!("OCR checkpoint identity field {key:?} has an invalid type")
}

fn validate_resume_identity(
    state: &Value,
    expected: &[(String, ResumeIdentity)],
    workflow_kind: &str,
) -> Result<()> {
    let mut missing = expected
        .iter()
        .filter(|(key, _)| state.get(key).is_none())
        .map(|(key, _)| key.as_str())
        .collect::<Vec<_>>();
    missing.sort_unstable();
    if !missing.is_empty() {
        bail!(
            "cannot resume OCR {workflow_kind} checkpoint without immutable identity fields: {}",
            missing.join(", ")
        );
    }
    let mut mismatches = expected
        .iter()
        .filter(
            |(key, expected_value)| match resume_identity_value(state, key) {
                Ok(value) => &value != expected_value,
                Err(_) => true,
            },
        )
        .map(|(key, _)| key.as_str())
        .collect::<Vec<_>>();
    mismatches.sort_unstable();
    if !mismatches.is_empty() {
        bail!(
            "OCR {workflow_kind} checkpoint identity mismatch; refusing to mix persisted state for: {}",
            mismatches.join(", ")
        );
    }
    Ok(())
}

pub struct RootResumeIdentity {
    pub document_id: String,
    pub source_fingerprint: String,
    pub source_namespace: String,
    pub configuration_fingerprints: Vec<String>,
    pub grouping_policy_version: String,
    pub parse_generation: String,
}

pub fn validate_root_resume_identity(
    state: &Value,
    identity: &RootResumeIdentity,
    require_match: bool,
) -> Result<()> {
    if !require_match {
        return Ok(());
    }
    validate_resume_identity(
        state,
        &[
            (
                "ocr_root_state_version".to_string(),
                ResumeIdentity::String(OCR_ROOT_STATE_VERSION.to_string()),
            ),
            (
                "document_id".to_string(),
                ResumeIdentity::String(identity.document_id.clone()),
            ),
            (
                "source_fingerprint".to_string(),
                ResumeIdentity::String(identity.source_fingerprint.clone()),
            ),
            (
                "source_namespace".to_string(),
                ResumeIdentity::String(identity.source_namespace.clone()),
            ),
            (
                "ocr_configuration_fingerprints".to_string(),
                ResumeIdentity::Strings(identity.configuration_fingerprints.clone()),
            ),
            (
                "grouping_policy_version".to_string(),
                ResumeIdentity::String(identity.grouping_policy_version.clone()),
            ),
            (
                "parse_generation".to_string(),
                ResumeIdentity::String(identity.parse_generation.clone()),
            ),
        ],
        "root",
    )
}

pub struct PageResumeIdentity {
    pub document_id: String,
    pub source_fingerprint: String,
    pub configuration_fingerprints: Vec<String>,
}

pub fn validate_page_resume_identity(state: &Value, identity: &PageResumeIdentity) -> Result<()> {
    validate_resume_identity(
        state,
        &[
            (
                "ocr_page_state_version".to_string(),
                ResumeIdentity::String(OCR_PAGE_STATE_VERSION.to_string()),
            ),
            (
                "document_id".to_string(),
                ResumeIdentity::String(identity.document_id.clone()),
            ),
            (
                "source_fingerprint".to_string(),
                ResumeIdentity::String(identity.source_fingerprint.clone()),
            ),
            (
                "ocr_configuration_fingerprints".to_string(),
                ResumeIdentity::Strings(identity.configuration_fingerprints.clone()),
            ),
        ],
        "page",
    )
}

pub struct PageReconciliationSources {
    pub page_sources: Vec<PageSourceState>,
    pub legacy_dir: std::path::PathBuf,
}

pub struct PageReconciliationRequest<'a> {
    pub document_id: &'a str,
    pub page_number: u64,
    pub image_digest: &'a str,
    pub artifact_path: &'a std::path::Path,
}

pub trait PageArtifactGate {
    fn accepts(&self, request: &PageReconciliationRequest) -> bool;
}

pub struct ReadableArtifactGate;

impl PageArtifactGate for ReadableArtifactGate {
    fn accepts(&self, request: &PageReconciliationRequest) -> bool {
        std::fs::read(request.artifact_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
            .map(|payload| {
                payload.get("pdf_page_num").and_then(Value::as_u64) == Some(request.page_number)
                    || payload
                        .get("pdf_page_num")
                        .and_then(Value::as_str)
                        .and_then(|value| value.trim().parse::<u64>().ok())
                        == Some(request.page_number)
            })
            .unwrap_or(false)
    }
}

fn validated_page_json(
    value: &Value,
    default_page_number: Option<u64>,
    _strict_page_number_types: bool,
) -> Result<Value> {
    let object = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("OCR page payload must be a JSON object"))?;
    let raw_page_number = object.get("pdf_page_num").unwrap_or(&Value::Null);
    if let Some(page_number) = raw_page_number.as_u64() {
        if page_number < 1 {
            bail!("OCR page pdf_page_num must be a positive integer");
        }
    } else if let Some(page_number) = raw_page_number.as_str() {
        if page_number.trim().parse::<u64>().is_err() {
            bail!("OCR page pdf_page_num must be a positive integer");
        }
    } else if raw_page_number.is_null() {
        if default_page_number.unwrap_or(0) < 1 {
            bail!("OCR page pdf_page_num must be a positive integer");
        }
    } else {
        bail!("OCR page pdf_page_num must be a positive integer");
    }
    for key in ["printed_page_number", "spatial_accuracy", "text"] {
        if let Some(value) = object.get(key) {
            if !value.is_null() && !value.is_string() {
                bail!("OCR page field {key:?} has an invalid type");
            }
        }
    }
    for key in ["contains_table", "is_blank_page"] {
        if let Some(value) = object.get(key) {
            if !value.is_null() && !value.is_boolean() {
                bail!("OCR page field {key:?} has an invalid type");
            }
        }
    }
    for key in ["unresolved_item_count", "source_item_count"] {
        if let Some(value) = object.get(key) {
            if value.as_u64().is_none() {
                bail!("OCR page field {key:?} must be a non-negative integer");
            }
        }
    }
    for key in [
        "page_x_min",
        "page_x_max",
        "page_y_min",
        "page_y_max",
        "data_loss_likelihood",
    ] {
        if let Some(value) = object.get(key) {
            if !value.is_null() && value.as_f64().map(|number| number.is_finite()) != Some(true) {
                bail!("OCR page field {key:?} must be a finite number");
            }
        }
    }
    for key in [
        "meaningful_ordering",
        "OCR_text_clusters",
        "non_text_objects",
    ] {
        if let Some(value) = object.get(key) {
            if !value.is_null() && !value.is_array() {
                bail!("OCR page field {key:?} must be a list");
            }
        }
    }
    Ok(value.clone())
}

pub fn json_pages(
    state: &Value,
    sources: Option<&[PageSourceState]>,
    legacy_dir: Option<&std::path::Path>,
) -> Result<Vec<Value>> {
    if let Some(raw_pages) = state.get("raw_pages") {
        let Some(items) = raw_pages.as_array() else {
            bail!("OCR page runtime raw page state is malformed");
        };
        return items
            .iter()
            .map(|item| validated_page_json(item, None, true))
            .collect();
    }
    let legacy_dir = state
        .get("legacy_dir")
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .or_else(|| legacy_dir.map(PathBuf::from));
    let Some(legacy_dir) = legacy_dir else {
        bail!("OCR page runtime state has no legacy artifact directory");
    };
    let Some(sources) = sources else {
        bail!("OCR page runtime state has no page source list");
    };
    let cursor = page_cursor(state) as usize;
    let mut pages = Vec::new();
    for source in sources.iter().take(cursor) {
        let path = legacy_dir.join(format!("page_{}.json", source.page_number));
        let bytes = std::fs::read(&path).map_err(|error| {
            anyhow::anyhow!("OCR page artifact is not readable: {path:?}: {error}")
        })?;
        let payload = serde_json::from_slice::<Value>(&bytes).map_err(|error| {
            anyhow::anyhow!("OCR page artifact is not readable: {path:?}: {error}")
        })?;
        pages.push(validated_page_json(
            &payload,
            Some(source.page_number),
            false,
        )?);
    }
    Ok(pages)
}

pub fn reconcile_page_checkpoint<G: PageArtifactGate>(
    state: &Value,
    sources: Option<&[PageSourceState]>,
    legacy_dir: &std::path::Path,
    gate: &G,
    document_id: &str,
    start_index: usize,
) -> Result<Option<Value>> {
    let cursor = page_cursor(state) as usize;
    let sources = if state.get("page_sources").is_some() {
        page_sources(state)?
    } else {
        sources
            .ok_or_else(|| anyhow::anyhow!("OCR page runtime state has no page source list"))?
            .to_vec()
    };
    let completed = if state.get("completed_pages").is_some() {
        Some(int_list(state, "completed_pages")?)
    } else {
        None
    };
    let reused = if state.get("reused_pages").is_some() {
        Some(int_list(state, "reused_pages")?)
    } else {
        None
    };
    if start_index > cursor {
        bail!("OCR checkpoint reconciliation start index is out of range");
    }
    for index in start_index..cursor {
        let Some(source) = sources.get(index) else {
            bail!("OCR page runtime state has no page source list");
        };
        let image_bytes = std::fs::read(&source.image_path).map_err(|error| {
            anyhow::anyhow!(
                "OCR page image is not readable: {:?}: {error}",
                source.image_path
            )
        })?;
        let mut digest = Sha256::new();
        digest.update(image_bytes);
        let image_digest = digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let artifact_path = legacy_dir.join(format!("page_{}.json", source.page_number));
        let request = PageReconciliationRequest {
            document_id,
            page_number: source.page_number,
            image_digest: &image_digest,
            artifact_path: &artifact_path,
        };
        if gate.accepts(&request) {
            continue;
        }
        let mut repair = Map::new();
        repair.insert("page_cursor".to_string(), json!(index));
        if let Some(completed) = &completed {
            repair.insert(
                "completed_pages".to_string(),
                json!(completed.iter().take(index).cloned().collect::<Vec<_>>()),
            );
        }
        if let (Some(reused), Some(completed)) = (&reused, &completed) {
            let completed_prefix = completed
                .iter()
                .take(index)
                .cloned()
                .collect::<std::collections::HashSet<i64>>();
            repair.insert(
                "reused_pages".to_string(),
                json!(reused
                    .iter()
                    .take(index)
                    .filter(|number| completed_prefix.contains(*number))
                    .cloned()
                    .collect::<Vec<_>>()),
            );
        }
        if state.get("raw_pages").is_some() {
            let pages = json_pages(state, Some(&sources), Some(legacy_dir))?;
            repair.insert(
                "raw_pages".to_string(),
                json!(pages.into_iter().take(index).collect::<Vec<_>>()),
            );
        }
        return Ok(Some(Value::Object(repair)));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn decodes_state_accessors() {
        let state = json!({
            "page_cursor": 2,
            "page_sources": [
                {"page_number": 1, "image_path": "p1.png"},
                {"page_number": 2, "image_path": "p2.png"}
            ],
            "completed_pages": [1, 2],
            "adaptive_grouping_complete": true
        });
        assert_eq!(page_cursor(&state), 2);
        assert_eq!(page_sources(&state).unwrap().len(), 2);
        assert_eq!(page_sources(&state).unwrap()[0].image_path, "p1.png");
        assert_eq!(int_list(&state, "completed_pages").unwrap(), vec![1, 2]);
        assert!(state_bool(&state, "adaptive_grouping_complete", false));
        assert_eq!(state_string(&state, "missing", "fallback"), "fallback");
    }

    #[test]
    fn rejects_malformed_state() {
        let state = json!({
            "page_sources": [{"page_number": 0, "image_path": "p0.png"}],
            "completed_pages": [1, "two"]
        });
        assert!(page_sources(&state).is_err());
        assert!(int_list(&state, "completed_pages").is_err());
        assert!(page_source_state(&json!("bad")).is_err());
    }

    #[test]
    fn validates_root_and_page_resume_identity() {
        let state = json!({
            "ocr_root_state_version": OCR_ROOT_STATE_VERSION,
            "document_id": "doc",
            "source_fingerprint": "source",
            "source_namespace": "namespace",
            "ocr_configuration_fingerprints": ["cfg"],
            "grouping_policy_version": "grouping",
            "parse_generation": "generation"
        });
        let root = RootResumeIdentity {
            document_id: "doc".to_string(),
            source_fingerprint: "source".to_string(),
            source_namespace: "namespace".to_string(),
            configuration_fingerprints: vec!["cfg".to_string()],
            grouping_policy_version: "grouping".to_string(),
            parse_generation: "generation".to_string(),
        };
        assert!(validate_root_resume_identity(&state, &root, true).is_ok());
        assert!(validate_root_resume_identity(&state, &root, false).is_ok());

        let page_state = json!({
            "ocr_page_state_version": OCR_PAGE_STATE_VERSION,
            "document_id": "doc",
            "source_fingerprint": "source",
            "ocr_configuration_fingerprints": ["cfg"]
        });
        let page = PageResumeIdentity {
            document_id: "doc".to_string(),
            source_fingerprint: "source".to_string(),
            configuration_fingerprints: vec!["cfg".to_string()],
        };
        assert!(validate_page_resume_identity(&page_state, &page).is_ok());

        let mismatched = json!({
            "ocr_page_state_version": OCR_PAGE_STATE_VERSION,
            "document_id": "other",
            "source_fingerprint": "source",
            "ocr_configuration_fingerprints": ["cfg"]
        });
        let error = validate_page_resume_identity(&mismatched, &page)
            .unwrap_err()
            .to_string();
        assert!(error.contains("document_id"));
    }

    #[test]
    fn reconciles_checkpoint_and_rewinds_invalid_prefix() {
        let temp = tempfile::tempdir().unwrap();
        let image = temp.path().join("page_1.png");
        std::fs::write(&image, b"page one").unwrap();
        let artifact = temp.path().join("page_1.json");
        std::fs::write(&artifact, r#"{"pdf_page_num":1}"#).unwrap();
        let state = json!({
            "page_cursor": 1,
            "page_sources": [{"page_number": 1, "image_path": image}],
            "completed_pages": [1],
            "reused_pages": [1],
            "raw_pages": [{"pdf_page_num": 1}]
        });
        let gate = ReadableArtifactGate;
        let repair = reconcile_page_checkpoint(&state, None, temp.path(), &gate, "doc", 0).unwrap();
        assert!(repair.is_none());

        std::fs::remove_file(&artifact).unwrap();
        let repair = reconcile_page_checkpoint(&state, None, temp.path(), &gate, "doc", 0)
            .unwrap()
            .unwrap();
        assert_eq!(repair["page_cursor"], 0);
        assert_eq!(repair["completed_pages"], json!([]));
        assert_eq!(repair["reused_pages"], json!([]));
        assert_eq!(repair["raw_pages"], json!([]));

        assert!(reconcile_page_checkpoint(&state, None, temp.path(), &gate, "doc", 2).is_err());
    }
}
