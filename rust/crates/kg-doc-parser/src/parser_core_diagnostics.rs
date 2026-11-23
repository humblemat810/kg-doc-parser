use anyhow::Result;
use serde::Serialize;
use serde_json::{Map, Value};

use crate::parser_core::LayerChildCandidate;
use crate::semantic::HydratedTextPointer;

#[derive(Debug, Clone, Serialize)]
pub struct LayerStructuralIssue {
    pub issue_kind: String,
    pub parent_node_id: String,
    pub node_id: Option<String>,
    pub message: String,
}

pub fn normalize_text(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join("")
        .to_lowercase()
}

pub fn child_pointer_fingerprint(child: &LayerChildCandidate) -> Vec<(String, i64, i64, String)> {
    child
        .total_content_pointers
        .iter()
        .map(|pointer| {
            (
                pointer.source_cluster_id.clone(),
                pointer.start_char,
                pointer.end_char,
                normalize_text(&pointer.verbatim_text),
            )
        })
        .collect()
}

fn parent_pointer_fingerprint(pointers: &Value) -> Vec<(String, i64, i64, String)> {
    let Some(pointers) = pointers.as_array() else {
        return Vec::new();
    };
    pointers
        .iter()
        .map(|pointer| {
            (
                pointer
                    .get("source_cluster_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                pointer
                    .get("start_char")
                    .and_then(Value::as_i64)
                    .unwrap_or_default(),
                pointer
                    .get("end_char")
                    .and_then(Value::as_i64)
                    .unwrap_or_default(),
                normalize_text(
                    pointer
                        .get("verbatim_text")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                ),
            )
        })
        .collect()
}

fn parent_pointer_records(pointers: &Value) -> Result<Vec<HydratedTextPointer>> {
    serde_json::from_value::<Vec<HydratedTextPointer>>(pointers.clone())
        .map_err(|error| anyhow::anyhow!("parent content pointers must be typed: {error}"))
}

pub fn layer_structural_issues(
    current_layer_context: &Map<String, Value>,
    current_layer_result: &crate::parser_core::CurrentLayerResult,
) -> Result<Vec<LayerStructuralIssue>> {
    let parent_pointers = current_layer_context
        .get("parent_content_pointers_by_id")
        .unwrap_or(&Value::Null);
    let parent_ids = current_layer_context
        .get("parent_node_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("parent_node_ids must be an array"))?;
    let mut issues = Vec::new();
    for parent_id in parent_ids {
        let Some(parent_id) = parent_id.as_str() else {
            anyhow::bail!("parent ids must be strings");
        };
        let children = current_layer_result
            .children
            .iter()
            .filter(|child| child.parent_node_id == parent_id)
            .collect::<Vec<_>>();
        if children.len() == 1 {
            let child = children[0];
            let unchanged = child_pointer_fingerprint(child)
                == parent_pointer_fingerprint(
                    parent_pointers.get(parent_id).unwrap_or(&Value::Null),
                );
            issues.push(LayerStructuralIssue {
                issue_kind: if unchanged { "unchanged_refinement".to_string() } else { "unary_refinement".to_string() },
                parent_node_id: parent_id.to_string(),
                node_id: Some(child.node_id.clone()),
                message: if unchanged {
                    "refinement returned one child with the parent's exact source span".to_string()
                } else {
                    "refinement returned one child; a meaningful split requires at least two children".to_string()
                },
            });
        }
        for child in children {
            if child.child_candidates.len() == 1 {
                let nested = &child.child_candidates[0];
                issues.push(LayerStructuralIssue {
                    issue_kind: "nested_unary_refinement".to_string(),
                    parent_node_id: child.node_id.clone(),
                    node_id: Some(nested.node_id.clone()),
                    message: "materialized child tree contains a unary refinement".to_string(),
                });
            }
        }
    }
    Ok(issues)
}

#[allow(dead_code)]
fn parse_parent_pointers(pointers: &Value) -> Result<Vec<HydratedTextPointer>> {
    parent_pointer_records(pointers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser_core::CurrentLayerResult;
    use serde_json::json;

    fn pointer(source: &str, start: i64, end: i64, text: &str) -> HydratedTextPointer {
        serde_json::from_value(json!({
            "source_cluster_id": source,
            "start_char": start,
            "end_char": end,
            "verbatim_text": text
        }))
        .unwrap()
    }

    fn child(
        node_id: &str,
        parent_id: &str,
        pointers: Vec<HydratedTextPointer>,
        child_candidates: Vec<LayerChildCandidate>,
    ) -> LayerChildCandidate {
        LayerChildCandidate {
            node_id: node_id.to_string(),
            parent_node_id: parent_id.to_string(),
            title: "Title".to_string(),
            node_type: "TEXT_FLOW".to_string(),
            total_content_pointers: pointers,
            expandable: true,
            metadata: Map::new(),
            child_candidates,
        }
    }

    #[test]
    fn normalizes_pointer_fingerprints() {
        let fingerprint = child_pointer_fingerprint(&child(
            "child",
            "parent",
            vec![pointer("p1", 0, 4, "Alpha  Beta")],
            Vec::new(),
        ));
        assert_eq!(
            fingerprint,
            vec![("p1".to_string(), 0, 4, "alphabeta".to_string())]
        );
    }

    #[test]
    fn reports_unary_unchanged_and_nested_unary() {
        let context = json!({
            "parent_node_ids": ["parent"],
            "parent_content_pointers_by_id": {
                "parent": [{
                    "source_cluster_id": "p1",
                    "start_char": 0,
                    "end_char": 4,
                    "verbatim_text": "Alpha"
                }]
            }
        });
        let unchanged_child = child(
            "child",
            "parent",
            vec![pointer("p1", 0, 4, "alpha")],
            Vec::new(),
        );
        let nested_child = child(
            "nested",
            "child",
            vec![pointer("p1", 0, 1, "a")],
            Vec::new(),
        );
        let nested_parent = child(
            "child",
            "parent",
            vec![pointer("p1", 0, 4, "Alpha")],
            vec![nested_child.clone()],
        );
        let result = CurrentLayerResult {
            children: vec![unchanged_child, nested_parent],
            satisfied: None,
            review_rounds: 0,
            atomic_retained: false,
            atomic_parent_ids: Vec::new(),
            atomic_content_pointers_by_parent: Map::new(),
            metadata: Map::new(),
        };
        let issues = layer_structural_issues(context.as_object().unwrap(), &result).unwrap();
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].issue_kind, "nested_unary_refinement");
    }
}
