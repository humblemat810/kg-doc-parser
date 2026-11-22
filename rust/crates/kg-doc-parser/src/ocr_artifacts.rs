use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;

pub const SUPPORTED_OCR_JSON_VERSIONS: [&str; 2] = ["0.1", "workflow_ingest_v1"];

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OCRArtifactMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ocr_provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ocr_model_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ocr_configuration_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_image_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ocr_json_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ocr_datetime: Option<f64>,
}

pub fn json_object(value: Value) -> Value {
    if value.is_object() {
        value
    } else {
        Value::Object(serde_json::Map::new())
    }
}

pub fn read_json_object(path: &Path) -> Value {
    match std::fs::read_to_string(path)
        .ok()
        .and_then(|contents| serde_json::from_str::<Value>(&contents).ok())
    {
        Some(value) => json_object(value),
        None => Value::Object(serde_json::Map::new()),
    }
}

pub fn artifact_metadata(payload: &Value) -> OCRArtifactMetadata {
    let Some(metadata) = payload.get("metadata") else {
        return OCRArtifactMetadata::default();
    };
    let mut result = OCRArtifactMetadata::default();
    if let Some(value) = metadata.get("ocr_provider").and_then(Value::as_str) {
        result.ocr_provider = Some(value.to_string());
    }
    if let Some(value) = metadata.get("ocr_model_name").and_then(Value::as_str) {
        result.ocr_model_name = Some(value.to_string());
    }
    if let Some(value) = metadata
        .get("ocr_configuration_fingerprint")
        .and_then(Value::as_str)
    {
        result.ocr_configuration_fingerprint = Some(value.to_string());
    }
    if let Some(value) = metadata.get("source_image_sha256").and_then(Value::as_str) {
        result.source_image_sha256 = Some(value.to_string());
    }
    if let Some(value) = metadata.get("ocr_json_version").and_then(Value::as_str) {
        result.ocr_json_version = Some(value.to_string());
    }
    if let Some(value) = metadata.get("ocr_datetime").and_then(Value::as_f64) {
        result.ocr_datetime = Some(value);
    }
    result
}

pub fn artifact_schema_error(payload: &Value) -> Option<String> {
    let version = artifact_metadata(payload).ocr_json_version;
    match version {
        Some(version) if SUPPORTED_OCR_JSON_VERSIONS.contains(&version.as_str()) => None,
        Some(version) => Some(format!(
            "unsupported OCR artifact schema version: {version}"
        )),
        None => Some("unsupported OCR artifact schema version: missing".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn decodes_object_and_rejects_non_object_payloads() {
        let payload = json_object(json!({"metadata": {"ocr_provider": "fake"}}));
        assert_eq!(payload["metadata"]["ocr_provider"], "fake");
        assert!(json_object(json!("not-an-object")).is_object());
        assert!(json_object(json!(1)).as_object().unwrap().is_empty());
    }

    #[test]
    fn narrows_known_metadata_fields() {
        let payload = json!({
            "metadata": {
                "ocr_provider": "fake",
                "ocr_model_name": "fake-model",
                "ocr_configuration_fingerprint": "cfg",
                "source_image_sha256": "sha",
                "ocr_json_version": "workflow_ingest_v1",
                "ocr_datetime": 1.5,
                "unknown": true
            }
        });
        let metadata = artifact_metadata(&payload);
        assert_eq!(metadata.ocr_provider.as_deref(), Some("fake"));
        assert_eq!(metadata.ocr_model_name.as_deref(), Some("fake-model"));
        assert_eq!(
            metadata.ocr_configuration_fingerprint.as_deref(),
            Some("cfg")
        );
        assert_eq!(metadata.source_image_sha256.as_deref(), Some("sha"));
        assert_eq!(
            metadata.ocr_json_version.as_deref(),
            Some("workflow_ingest_v1")
        );
        assert_eq!(metadata.ocr_datetime, Some(1.5));
    }

    #[test]
    fn rejects_unsupported_schema_versions() {
        let payload = json!({"metadata": {"ocr_json_version": "workflow_ingest_v1"}});
        assert!(artifact_schema_error(&payload).is_none());
        let payload = json!({"metadata": {"ocr_json_version": "0.2"}});
        assert_eq!(
            artifact_schema_error(&payload).as_deref(),
            Some("unsupported OCR artifact schema version: 0.2")
        );
        let payload = json!({"metadata": {}});
        assert_eq!(
            artifact_schema_error(&payload).as_deref(),
            Some("unsupported OCR artifact schema version: missing")
        );
    }
}
