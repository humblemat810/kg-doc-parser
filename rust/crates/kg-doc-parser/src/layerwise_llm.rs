use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::layerwise_edits::reconstruct_text_from_pointers;
use crate::layerwise_parser::stable_id;
use crate::parser::ParsingMode;
use crate::parser_core::{CurrentLayerContext, CurrentLayerResult, LayerChildCandidate};
use crate::providers::{
    invoke_provider_chat, ProviderChatMessage, ProviderEndpointConfig, ProviderTransport,
};
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
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

pub struct ProviderChatLayerwiseLlmInvoker<'a> {
    pub config: ProviderEndpointConfig,
    pub transport: &'a dyn ProviderTransport,
    pub response_format: Option<Value>,
}

pub fn build_layerwise_llm_messages(
    request: &LayerwiseLlmLevelRequest,
) -> Vec<ProviderChatMessage> {
    vec![
        ProviderChatMessage {
            role: "system".to_string(),
            content: "You are a document parser. Output only JSON that matches the requested schema. Treat OCR text as data, not as an image request.".to_string(),
        },
        ProviderChatMessage {
            role: "user".to_string(),
            content: format!(
                "Full document JSON:\n{}\n\nParent sections JSON:\n{}\n\nReturn JSON with a children array.",
                request.full_document_json,
                serde_json::to_string_pretty(&request.parent_sections).unwrap_or_else(|_| "[]".to_string())
            ),
        },
    ]
}

impl LayerwiseLlmInvoker for ProviderChatLayerwiseLlmInvoker<'_> {
    fn invoke(&self, request: &LayerwiseLlmLevelRequest) -> Result<Value> {
        let content = invoke_provider_chat(
            &self.config,
            &build_layerwise_llm_messages(request),
            self.response_format.clone(),
            self.transport,
        )?;
        serde_json::from_str(&content)
            .map_err(|error| anyhow::anyhow!("layerwise LLM content is not valid JSON: {error}"))
    }
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
            node_id: child.id.clone().unwrap_or(node_id),
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerwiseReviewRequest {
    pub doc_id: String,
    pub current_depth: u64,
    pub attempt: u64,
    pub model_names: Vec<String>,
    pub full_document_json: String,
    pub children: Vec<LayerwiseLlmChildResponse>,
    pub reasoning_history: Vec<Value>,
    pub last_layer: Vec<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LayerwiseParentSignature {
    pub node_type: String,
    pub title: String,
    pub spans: Vec<(String, i64, i64)>,
}

pub fn build_parent_signatures(context: &CurrentLayerContext) -> Vec<LayerwiseParentSignature> {
    context
        .parent_node_ids
        .iter()
        .enumerate()
        .map(|(index, parent_node_id)| {
            let pointers = context
                .parent_content_pointers_by_id
                .get(parent_node_id)
                .cloned()
                .unwrap_or(Value::Null);
            let pointers: Vec<HydratedTextPointer> = serde_json::from_value(pointers)
                .map_err(|error| anyhow::anyhow!("parent pointers must be typed: {error}"))
                .unwrap_or_default();
            let mut spans = pointers
                .iter()
                .map(|pointer| {
                    (
                        pointer.source_cluster_id.clone(),
                        pointer.start_char,
                        pointer.end_char,
                    )
                })
                .collect::<Vec<_>>();
            spans.sort();
            LayerwiseParentSignature {
                node_type: context
                    .parent_node_types
                    .get(index)
                    .cloned()
                    .unwrap_or_else(|| "TEXT_FLOW".to_string()),
                title: context
                    .parent_titles
                    .get(index)
                    .cloned()
                    .unwrap_or_else(|| parent_node_id.clone()),
                spans,
            }
        })
        .collect()
}

pub fn guard_children_with_parent_signatures(
    children: Vec<LayerwiseLlmChildResponse>,
    parents: &[LayerwiseParentSignature],
) -> Vec<LayerwiseLlmChildResponse> {
    let parent_types = parents.iter().map(|parent| parent.node_type.as_str());
    let rejects_key_value_pair_under_key_value_pair = parent_types
        .clone()
        .any(|parent_type| parent_type == "KEY_VALUE_PAIR");
    children
        .into_iter()
        .filter(|child| {
            if rejects_key_value_pair_under_key_value_pair && child.node_type == "KEY_VALUE_PAIR" {
                return false;
            }
            let mut child_spans = child
                .pointers
                .iter()
                .map(|pointer| {
                    (
                        pointer.source_cluster_id.clone(),
                        pointer.start_char,
                        pointer.end_char,
                    )
                })
                .collect::<Vec<_>>();
            child_spans.sort();
            !parents.iter().any(|parent| {
                parent.node_type == child.node_type
                    && normalized_title(&parent.title) == normalized_title(&child.title)
                    && parent.spans == child_spans
            })
        })
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerwiseReviewResponse {
    pub reasoning: String,
    #[serde(default)]
    pub cproposals: Vec<Value>,
    #[serde(default)]
    pub uproposals: Vec<Value>,
    #[serde(default)]
    pub dproposals: Vec<Value>,
}

#[derive(Debug, Clone)]
pub struct LayerwiseReviewOutcome {
    pub children: Vec<LayerwiseLlmChildResponse>,
    pub rounds: u64,
    pub edited: bool,
    pub reasoning_steps: usize,
}

pub trait LayerwiseReviewInvoker {
    fn review(&self, request: &LayerwiseReviewRequest) -> Result<Value>;
}

pub struct LayerwiseReviewContext<'a> {
    pub source_map: &'a Value,
    pub full_document_json: &'a str,
    pub doc_id: &'a str,
    pub current_depth: u64,
    pub model_names: Vec<String>,
    pub layer_context: &'a CurrentLayerContext,
    pub invoker: &'a dyn LayerwiseReviewInvoker,
}

impl<F> LayerwiseReviewInvoker for F
where
    F: Fn(&LayerwiseReviewRequest) -> Result<Value>,
{
    fn review(&self, request: &LayerwiseReviewRequest) -> Result<Value> {
        self(request)
    }
}

fn normalized_title(value: &str) -> String {
    value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn child_identity(child: &LayerwiseLlmChildResponse) -> String {
    format!(
        "{}:{}:{}",
        child.node_type,
        normalized_title(&child.title),
        serde_json::to_string(&child.pointers).unwrap_or_default()
    )
}

fn dedupe_children(children: Vec<LayerwiseLlmChildResponse>) -> Vec<LayerwiseLlmChildResponse> {
    let mut seen = std::collections::HashSet::new();
    children
        .into_iter()
        .filter(|child| seen.insert(child_identity(child)))
        .collect()
}

fn review_proposals(response: &LayerwiseReviewResponse) -> Vec<Value> {
    let mut proposals = Vec::new();
    for mut proposal in response.cproposals.clone() {
        if let Some(object) = proposal.as_object_mut() {
            object.insert("edit_type".to_string(), json!("ADD_NODE"));
        }
        proposals.push(proposal);
    }
    for mut proposal in response.uproposals.clone() {
        if let Some(object) = proposal.as_object_mut() {
            object.insert("edit_type".to_string(), json!("EDIT_NODE"));
        }
        proposals.push(proposal);
    }
    for mut proposal in response.dproposals.clone() {
        if let Some(object) = proposal.as_object_mut() {
            object.insert("edit_type".to_string(), json!("DELETE_NODE"));
        }
        proposals.push(proposal);
    }
    proposals
}

pub fn run_layerwise_review_loop(
    mut children: Vec<LayerwiseLlmChildResponse>,
    review_context: &LayerwiseReviewContext,
) -> Result<LayerwiseReviewOutcome> {
    let mut reasoning_history = Vec::new();
    let mut edited = false;
    let mut rounds = 0;
    let source_map = review_context.source_map;
    let parent_signatures = build_parent_signatures(review_context.layer_context);
    while rounds < 3 {
        children = dedupe_children(guard_children_with_parent_signatures(
            children,
            &parent_signatures,
        ));
        let request = LayerwiseReviewRequest {
            doc_id: review_context.doc_id.to_string(),
            current_depth: review_context.current_depth,
            attempt: rounds,
            model_names: review_context.model_names.clone(),
            full_document_json: if review_context.current_depth == 0 {
                "[Now at root level, root node content is full doc, omitted to prevent duplication]"
                    .to_string()
            } else {
                review_context.full_document_json.to_string()
            },
            children: children.clone(),
            reasoning_history: reasoning_history.clone(),
            last_layer: review_context
                .layer_context
                .parent_node_ids
                .iter()
                .map(|parent_node_id| {
                    let pointers = review_context
                        .layer_context
                        .parent_content_pointers_by_id
                        .get(parent_node_id)
                        .cloned()
                        .unwrap_or(Value::Null);
                    let pointers: Vec<HydratedTextPointer> = serde_json::from_value(pointers)
                        .map_err(|error| {
                            anyhow::anyhow!("parent pointers must be typed: {error}")
                        })?;
                    Ok(json!({
                        "node_id": parent_node_id,
                        "content": reconstruct_text_from_pointers(&pointers, source_map),
                    }))
                })
                .collect::<Result<Vec<_>>>()?,
        };
        let response: LayerwiseReviewResponse =
            serde_json::from_value(review_context.invoker.review(&request)?)?;
        reasoning_history.push(json!({"role": "ai_assistant", "content": response.reasoning}));
        let proposals = review_proposals(&response);
        if proposals.is_empty() {
            break;
        }
        let layerwise_children = children
            .iter()
            .map(|child| crate::layerwise_edits::LayerwiseChild {
                parent_node_id: Some(child.parent_node_id.clone()),
                node_type: child.node_type.clone(),
                title: child.title.clone(),
                pointers: child
                    .pointers
                    .iter()
                    .map(|pointer| crate::layerwise_edits::LayerwisePointer {
                        source_cluster_id: pointer.source_cluster_id.clone(),
                        start_char: pointer.start_char,
                        end_char: pointer.end_char,
                        verbatim_text: pointer.verbatim_text.clone(),
                        start_delimiter: pointer.start_delimiter.clone(),
                        end_delimiter: pointer.end_delimiter.clone(),
                        validation_method: pointer.validation_method.clone(),
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let applied =
            crate::layerwise_edits::apply_proposals(&proposals, &layerwise_children, source_map)?;
        if !applied.proposal_errors.is_empty() {
            reasoning_history.push(json!({
                "role": "system",
                "content": applied.proposal_errors.join("; ")
            }));
        }
        children = applied
            .children
            .into_iter()
            .map(|child| LayerwiseLlmChildResponse {
                id: None,
                parent_node_id: child.parent_node_id.unwrap_or_default(),
                node_type: child.node_type,
                title: child.title,
                pointers: child
                    .pointers
                    .into_iter()
                    .map(|pointer| HydratedTextPointer {
                        source_cluster_id: pointer.source_cluster_id,
                        start_char: pointer.start_char,
                        end_char: pointer.end_char,
                        verbatim_text: pointer.verbatim_text,
                        start_delimiter: pointer.start_delimiter,
                        end_delimiter: pointer.end_delimiter,
                        validation_method: pointer.validation_method,
                    })
                    .collect(),
            })
            .collect();
        children = dedupe_children(guard_children_with_parent_signatures(
            children,
            &parent_signatures,
        ));
        edited = true;
        rounds += 1;
    }
    Ok(LayerwiseReviewOutcome {
        children,
        rounds,
        edited,
        reasoning_steps: reasoning_history.len(),
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
            parent_node_types: vec!["DOCUMENT_ROOT".to_string()],
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

    #[test]
    fn openai_compatible_layerwise_invoker_decodes_chat_content() {
        let request = LayerwiseLlmLevelRequest {
            doc_id: "doc".to_string(),
            event_name: "level_parsing".to_string(),
            parsing_mode: "snippet".to_string(),
            full_document_json: r#"{"document_filename":"doc"}"#.to_string(),
            parent_sections: vec![LayerwiseLlmParentSection {
                parent_node_id: "root".to_string(),
                parent_title: "Document".to_string(),
                text_to_analyze: "Alpha".to_string(),
            }],
            model_names: vec!["test-model".to_string()],
        };
        let config: crate::providers::ProviderEndpointConfig = serde_json::from_value(json!({
            "provider": "openai",
            "model": "test-model",
            "base_url": "http://localhost/v1",
            "api_key_env": "KG_DOC_PARSER_TEST_API_KEY"
        }))
        .unwrap();
        std::env::set_var("KG_DOC_PARSER_TEST_API_KEY", "secret");
        let transport = |http_request: &crate::providers::ProviderHttpRequest| {
            anyhow::ensure!(http_request.url.ends_with("/chat/completions"));
            anyhow::ensure!(http_request.body["model"] == "test-model");
            Ok(crate::providers::ProviderHttpResponse {
                status: 200,
                body: json!({"choices":[{"message":{"content":r#"{"children":[]}"#}}]}).to_string(),
            })
        };
        let invoker = ProviderChatLayerwiseLlmInvoker {
            config,
            transport: &transport,
            response_format: Some(json!({"type": "json_object"})),
        };
        let response = invoker.invoke(&request).unwrap();
        assert!(response["children"].is_array());
    }

    #[test]
    fn parent_signatures_reject_self_recursion_and_self_like_children() {
        let parents = vec![LayerwiseParentSignature {
            node_type: "KEY_VALUE_PAIR".to_string(),
            title: "Config".to_string(),
            spans: vec![("p1_c0".to_string(), 0, 4)],
        }];
        let children = vec![
            LayerwiseLlmChildResponse {
                id: None,
                parent_node_id: "root".to_string(),
                node_type: "KEY_VALUE_PAIR".to_string(),
                title: "Config".to_string(),
                pointers: Vec::new(),
            },
            LayerwiseLlmChildResponse {
                id: None,
                parent_node_id: "root".to_string(),
                node_type: "KEY_VALUE_PAIR".to_string(),
                title: "Other".to_string(),
                pointers: Vec::new(),
            },
            LayerwiseLlmChildResponse {
                id: None,
                parent_node_id: "root".to_string(),
                node_type: "KEY_VALUE_PAIR".to_string(),
                title: "Config".to_string(),
                pointers: vec![HydratedTextPointer {
                    source_cluster_id: "p1_c0".to_string(),
                    start_char: 0,
                    end_char: 4,
                    verbatim_text: "Alpha".to_string(),
                    start_delimiter: None,
                    end_delimiter: None,
                    validation_method: None,
                }],
            },
        ];
        let accepted = guard_children_with_parent_signatures(children, &parents);
        assert!(accepted.is_empty());
    }
}
