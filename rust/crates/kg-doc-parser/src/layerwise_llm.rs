use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::layerwise_edits::reconstruct_text_from_pointers;
use crate::layerwise_parser::stable_id;
use crate::parser::ParsingMode;
use crate::parser_core::{CurrentLayerContext, CurrentLayerResult, LayerChildCandidate};
use crate::semantic::HydratedTextPointer;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerwiseLlmParentSection {
    pub parent_node_id: String,
    pub parent_title: String,
    pub text_to_analyze: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerwiseLlmLevelRequest {
    pub doc_id: String,
    pub event_name: String,
    pub parsing_mode: String,
    pub full_document_json: String,
    pub parent_sections: Vec<LayerwiseLlmParentSection>,
    pub model_names: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerwiseLlmChildResponse {
    pub parent_node_id: String,
    pub node_type: String,
    pub title: String,
    pub pointers: Vec<HydratedTextPointer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerwiseLlmLevelResponse {
    pub children: Vec<LayerwiseLlmChildResponse>,
}

pub trait LayerwiseLlmInvoker {
    fn invoke(&self, request: &LayerwiseLlmLevelRequest) -> Result<Value>;
}

impl<F> LayerwiseLlmInvoker for F
where
    F: Fn(&LayerwiseLlmLevelRequest) -> Result<Value>,
{
    fn invoke(&self, request: &LayerwiseLlmLevelRequest) -> Result<Value> {
        self(request)
    }
}

pub fn build_layerwise_llm_level_request(
    doc_id: &str,
    context: &CurrentLayerContext,
    source_map: &Value,
    full_document_json: &str,
    parsing_mode: ParsingMode,
    model_names: &[String],
) -> Result<LayerwiseLlmLevelRequest> {
    anyhow::ensure!(
        !model_names.is_empty(),
        "layerwise LLM request requires at least one model"
    );
    let mut parent_sections = Vec::with_capacity(context.parent_node_ids.len());
    for (index, parent_node_id) in context.parent_node_ids.iter().enumerate() {
        let pointers = context
            .parent_content_pointers_by_id
            .get(parent_node_id)
            .cloned()
            .unwrap_or(Value::Null);
        let pointers: Vec<HydratedTextPointer> = serde_json::from_value(pointers)
            .map_err(|error| anyhow::anyhow!("parent pointers must be typed: {error}"))?;
        parent_sections.push(LayerwiseLlmParentSection {
            parent_node_id: parent_node_id.clone(),
            parent_title: context
                .parent_titles
                .get(index)
                .cloned()
                .unwrap_or_else(|| parent_node_id.clone()),
            text_to_analyze: reconstruct_text_from_pointers(&pointers, source_map),
        });
    }
    Ok(LayerwiseLlmLevelRequest {
        doc_id: doc_id.to_string(),
        event_name: "level_parsing".to_string(),
        parsing_mode: match parsing_mode {
            ParsingMode::Snippet => "snippet".to_string(),
            ParsingMode::Delimiter => "delimiter".to_string(),
        },
        full_document_json: full_document_json.to_string(),
        parent_sections,
        model_names: model_names.to_vec(),
    })
}

fn pointer_fingerprint(pointers: &[HydratedTextPointer]) -> String {
    pointers
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
        .join("|")
}

pub fn decode_layerwise_llm_level_response(
    response: &Value,
    context: &CurrentLayerContext,
    current_depth: u64,
) -> Result<CurrentLayerResult> {
    let response: LayerwiseLlmLevelResponse = serde_json::from_value(response.clone())
        .map_err(|error| anyhow::anyhow!("layerwise LLM response is malformed: {error}"))?;
    let allowed_parents = context
        .parent_node_ids
        .iter()
        .cloned()
        .collect::<std::collections::HashSet<_>>();
    let mut children = Vec::with_capacity(response.children.len());
    for child in response.children {
        if !allowed_parents.contains(&child.parent_node_id) {
            bail!("LLM child references parent outside the current layer");
        }
        if !matches!(
            child.node_type.as_str(),
            "TEXT_FLOW" | "KEY_VALUE_PAIR" | "TABLE"
        ) {
            bail!("unsupported layerwise child node type: {}", child.node_type);
        }
        let expandable = child.node_type != "KEY_VALUE_PAIR";
        let node_id = stable_id(
            "legacy.get_node",
            &[
                &child.parent_node_id,
                &child.node_type,
                &child.title,
                &(current_depth + 1).to_string(),
                &pointer_fingerprint(&child.pointers),
            ],
        );
        children.push(LayerChildCandidate {
            node_id,
            parent_node_id: child.parent_node_id,
            title: child.title,
            node_type: child.node_type,
            total_content_pointers: child.pointers,
            expandable,
            metadata: json!({}).as_object().cloned().unwrap_or_default(),
            child_candidates: Vec::new(),
        });
    }
    let atomic_retained = children.is_empty();
    let atomic_parent_ids = if atomic_retained {
        context.parent_node_ids.clone()
    } else {
        Vec::new()
    };
    Ok(CurrentLayerResult {
        children,
        satisfied: Some(true),
        review_rounds: 0,
        atomic_retained,
        atomic_parent_ids,
        atomic_content_pointers_by_parent: Map::new(),
        metadata: json!({"llm_level": true})
            .as_object()
            .cloned()
            .unwrap_or_default(),
        validation_diagnostics: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Map};

    fn context() -> CurrentLayerContext {
        CurrentLayerContext {
            depth: 0,
            parent_node_ids: vec!["root".to_string()],
            parent_titles: vec!["Document".to_string()],
            parent_content_pointers_by_id: serde_json::from_value(json!({
                "root": [{
                    "source_cluster_id": "p1_c0",
                    "start_char": 0,
                    "end_char": -1,
                    "verbatim_text": "Alpha"
                }]
            }))
            .unwrap(),
            split_strategy: crate::parser_core::SplitStrategy::ExcerptFirst,
            retry_count: 0,
            max_retries: 3,
            metadata: Map::new(),
        }
    }

    #[test]
    fn builds_level_request_with_reconstructed_text() {
        let request = build_layerwise_llm_level_request(
            "doc",
            &context(),
            &json!({"p1_c0": {"text": "Alpha"}}),
            r#"{"pages":[]}"#,
            ParsingMode::Snippet,
            &["model".to_string()],
        )
        .unwrap();
        assert_eq!(request.parent_sections[0].text_to_analyze, "Alpha");
        assert_eq!(request.event_name, "level_parsing");
        assert_eq!(request.parsing_mode, "snippet");
    }

    #[test]
    fn decodes_llm_children_into_stable_layer_candidates() {
        let result = decode_layerwise_llm_level_response(
            &json!({
                "children": [
                    {
                        "parent_node_id": "root",
                        "node_type": "TEXT_FLOW",
                        "title": "Alpha",
                        "pointers": [{
                            "source_cluster_id": "p1_c0",
                            "start_char": 0,
                            "end_char": 4,
                            "verbatim_text": "Alpha"
                        }]
                    }
                ]
            }),
            &context(),
            0,
        )
        .unwrap();
        assert_eq!(result.children.len(), 1);
        assert!(result.children[0].expandable);
        assert!(!result.children[0].node_id.is_empty());
    }
}
