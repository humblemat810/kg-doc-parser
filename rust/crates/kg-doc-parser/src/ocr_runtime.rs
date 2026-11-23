use anyhow::{bail, Result};
use serde_json::Value;

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
}
