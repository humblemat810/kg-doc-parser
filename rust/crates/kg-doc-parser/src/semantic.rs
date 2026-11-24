use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HydratedTextPointer {
    pub source_cluster_id: String,
    pub start_char: i64,
    pub end_char: i64,
    pub verbatim_text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_delimiter: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_delimiter: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validation_method: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticNode {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    #[serde(default = "default_node_type")]
    pub node_type: String,
    pub title: String,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub total_content_pointers: Vec<HydratedTextPointer>,
    #[serde(default)]
    pub aggregate_content_pointers: Vec<HydratedTextPointer>,
    #[serde(default)]
    pub child_nodes: Vec<SemanticNode>,
    #[serde(default)]
    pub level_from_root: u64,
    #[serde(default)]
    pub metadata: Value,
}

fn default_node_type() -> String {
    "TEXT_FLOW".to_string()
}

impl SemanticNode {
    pub fn with_stable_id(mut self, parent_id: Option<&str>) -> Self {
        if self.node_id.is_some() {
            return self;
        }
        let pointer_fingerprint = self
            .total_content_pointers
            .iter()
            .map(|pointer| {
                format!(
                    "{}:{}:{}:{}",
                    pointer.source_cluster_id,
                    pointer.start_char,
                    pointer.end_char,
                    pointer.verbatim_text
                )
            })
            .collect::<Vec<_>>()
            .join("|");
        let mut digest = Sha256::new();
        digest.update("workflow_ingest.semantic_node".as_bytes());
        digest.update(parent_id.unwrap_or("root").as_bytes());
        digest.update(self.node_type.as_bytes());
        digest.update(self.title.as_bytes());
        digest.update(self.level_from_root.to_string().as_bytes());
        digest.update(pointer_fingerprint.as_bytes());
        self.node_id = Some(
            digest
                .finalize()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>(),
        );
        self
    }

    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or_else(|_| json!({}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_id_is_derived_from_content() {
        let first = SemanticNode {
            node_id: None,
            parent_id: None,
            node_type: "SECTION".to_string(),
            title: "Alpha".to_string(),
            summary: String::new(),
            total_content_pointers: Vec::new(),
            aggregate_content_pointers: Vec::new(),
            child_nodes: Vec::new(),
            level_from_root: 0,
            metadata: json!({}),
        }
        .with_stable_id(None);
        let second = first.clone();
        assert_eq!(first.node_id, second.node_id);
        assert_eq!(first.node_id.as_deref().map(str::len), Some(64));
    }
}
