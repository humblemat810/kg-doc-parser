use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::semantic::{HydratedTextPointer, SemanticNode};
use crate::source_text::source_map_entry_text;

fn source_text(record: Option<&Value>) -> String {
    match record {
        Some(record) => source_map_entry_text(record),
        None => String::new(),
    }
}

pub fn source_map_fingerprint(parser_source_map: &Value) -> String {
    let empty = Map::new();
    let records = parser_source_map
        .as_object()
        .unwrap_or(&empty)
        .iter()
        .map(|(source_cluster_id, record)| {
            json!({
                "source_cluster_id": source_cluster_id,
                "text": source_text(Some(record)),
            })
        })
        .collect::<Vec<_>>();
    let canonical = crate::semantic_indexing::json_compact(&serde_json::to_value(records).unwrap());
    let mut digest = Sha256::new();
    digest.update(canonical.as_bytes());
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitStrategy {
    #[default]
    ExcerptFirst,
    BoundaryFirst,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseSessionState {
    pub collection_id: String,
    pub root_node_id: String,
    pub current_depth: u64,
    pub max_depth: u64,
    pub allow_review: bool,
    pub split_strategy: SplitStrategy,
    pub fallback_split_strategy: SplitStrategy,
    pub strategy_history: Vec<SplitStrategy>,
    pub strategy_switch_count: u64,
    pub mode: String,
    pub layer_attempts: Map<String, Value>,
    pub last_review: Map<String, Value>,
    pub compat_full_tree: Option<Value>,
    pub metadata: Map<String, Value>,
}

impl ParseSessionState {
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.current_depth <= self.max_depth,
            "current_depth cannot exceed max_depth"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerFrontierItem {
    pub parent_node_id: String,
    pub depth: u64,
    pub order: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurrentLayerContext {
    pub depth: u64,
    pub parent_node_ids: Vec<String>,
    pub parent_titles: Vec<String>,
    pub parent_content_pointers_by_id: Map<String, Value>,
    pub split_strategy: SplitStrategy,
    pub retry_count: u64,
    pub max_retries: u64,
    pub metadata: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerChildCandidate {
    pub node_id: String,
    pub parent_node_id: String,
    pub title: String,
    pub node_type: String,
    pub total_content_pointers: Vec<HydratedTextPointer>,
    pub expandable: bool,
    pub metadata: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurrentLayerResult {
    pub children: Vec<LayerChildCandidate>,
    pub satisfied: Option<bool>,
    pub review_rounds: u64,
    pub atomic_retained: bool,
    pub atomic_parent_ids: Vec<String>,
    pub atomic_content_pointers_by_parent: Map<String, Value>,
    pub metadata: Map<String, Value>,
}

#[derive(Debug, Clone)]
pub struct SourceCollection {
    pub collection_id: String,
    pub title: String,
}

pub fn assert_source_map_matches_session(
    parse_session: &ParseSessionState,
    parser_source_map: &Value,
) -> Result<()> {
    let Some(Value::String(expected)) = parse_session.metadata.get("source_map_fingerprint") else {
        return Ok(());
    };
    anyhow::ensure!(
        *expected == source_map_fingerprint(parser_source_map),
        "parser source map does not match the parse session source revision"
    );
    Ok(())
}

pub fn initialize_parse_session(
    collection: &SourceCollection,
    parser_source_map: &Value,
    max_depth: u64,
    allow_review: bool,
    split_strategy: SplitStrategy,
    fallback_split_strategy: SplitStrategy,
) -> Result<(ParseSessionState, Vec<LayerFrontierItem>, SemanticNode)> {
    let source_map = parser_source_map
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("parser source map must be an object"))?;
    let mut pointers = Vec::new();
    for (unit_id, record) in source_map.iter() {
        let participates = record
            .get("participates_in_semantic_text")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        if !participates {
            continue;
        }
        pointers.push(HydratedTextPointer {
            source_cluster_id: unit_id.clone(),
            start_char: 0,
            end_char: -1,
            verbatim_text: source_text(Some(record)),
            start_delimiter: None,
            end_delimiter: None,
            validation_method: None,
        });
    }
    pointers.sort_by(|left, right| left.source_cluster_id.cmp(&right.source_cluster_id));
    let root_id = format!("{}|root", collection.collection_id);
    let root = SemanticNode {
        node_id: Some(root_id.clone()),
        parent_id: None,
        node_type: "DOCUMENT_ROOT".to_string(),
        title: collection.title.clone(),
        summary: String::new(),
        total_content_pointers: pointers,
        aggregate_content_pointers: Vec::new(),
        child_nodes: Vec::new(),
        level_from_root: 0,
        metadata: json!({}),
    };
    let mut metadata = Map::new();
    metadata.insert("default_split_strategy".to_string(), json!(split_strategy));
    metadata.insert(
        "default_fallback_split_strategy".to_string(),
        json!(fallback_split_strategy),
    );
    metadata.insert(
        "source_map_fingerprint".to_string(),
        json!(source_map_fingerprint(parser_source_map)),
    );
    let session = ParseSessionState {
        collection_id: collection.collection_id.clone(),
        root_node_id: root_id.clone(),
        current_depth: 0,
        max_depth,
        allow_review,
        split_strategy,
        fallback_split_strategy,
        strategy_history: vec![split_strategy],
        strategy_switch_count: 0,
        mode: "workflow_layered".to_string(),
        layer_attempts: Map::new(),
        last_review: Map::new(),
        compat_full_tree: None,
        metadata,
    };
    session.validate()?;
    let frontier = vec![LayerFrontierItem {
        parent_node_id: root_id,
        depth: 0,
        order: 0,
    }];
    Ok((session, frontier, root))
}

pub fn find_semantic_node<'a>(node: &'a SemanticNode, node_id: &str) -> Option<&'a SemanticNode> {
    if node.node_id.as_deref() == Some(node_id) {
        return Some(node);
    }
    node.child_nodes
        .iter()
        .find_map(|child| find_semantic_node(child, node_id))
}

pub fn prepare_layer_frontier(
    parse_session: &ParseSessionState,
    frontier_queue: &[LayerFrontierItem],
    semantic_tree: &SemanticNode,
    max_retries: u64,
    max_items: Option<usize>,
) -> Result<(
    CurrentLayerContext,
    Vec<LayerFrontierItem>,
    ParseSessionState,
)> {
    anyhow::ensure!(!frontier_queue.is_empty(), "frontier queue is empty");
    let mut sorted_queue = frontier_queue.to_vec();
    sorted_queue.sort_by(|left, right| (left.depth, left.order).cmp(&(right.depth, right.order)));
    let current_depth = sorted_queue[0].depth;
    let current_items = sorted_queue
        .iter()
        .filter(|item| item.depth == current_depth)
        .cloned()
        .collect::<Vec<_>>();
    let selected_items = if let Some(max_items) = max_items {
        anyhow::ensure!(max_items >= 1, "max_items must be positive");
        current_items
            .iter()
            .take(max_items)
            .cloned()
            .collect::<Vec<_>>()
    } else {
        current_items.clone()
    };
    let remaining = sorted_queue
        .into_iter()
        .filter(|item| !selected_items.contains(item))
        .collect::<Vec<_>>();
    let mut parent_titles = Vec::new();
    for item in &selected_items {
        parent_titles.push(
            find_semantic_node(semantic_tree, &item.parent_node_id)
                .map(|node| node.title.clone())
                .unwrap_or_else(|| item.parent_node_id.clone()),
        );
    }
    let mut session = parse_session.clone();
    session.current_depth = current_depth;
    session.validate()?;
    let frontier_identity = format!(
        "depth={};parents={}",
        current_depth,
        selected_items
            .iter()
            .map(|item| item.parent_node_id.as_str())
            .collect::<Vec<_>>()
            .join(",")
    );
    let mut parent_content_pointers_by_id = Map::new();
    for item in &selected_items {
        let pointers = find_semantic_node(semantic_tree, &item.parent_node_id)
            .map(|node| {
                if node.total_content_pointers.is_empty() {
                    node.aggregate_content_pointers.clone()
                } else {
                    node.total_content_pointers.clone()
                }
            })
            .unwrap_or_default();
        parent_content_pointers_by_id.insert(item.parent_node_id.clone(), json!(pointers));
    }
    let split_strategy = session
        .metadata
        .get("default_split_strategy")
        .and_then(|value| serde_json::from_value::<SplitStrategy>(value.clone()).ok())
        .unwrap_or(session.split_strategy);
    let retry_count = session
        .layer_attempts
        .get(&frontier_identity)
        .or_else(|| session.layer_attempts.get(&current_depth.to_string()))
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let mut metadata = Map::new();
    metadata.insert("frontier_identity".to_string(), json!(frontier_identity));
    let context = CurrentLayerContext {
        depth: current_depth,
        parent_node_ids: selected_items
            .iter()
            .map(|item| item.parent_node_id.clone())
            .collect(),
        parent_titles,
        parent_content_pointers_by_id,
        split_strategy,
        retry_count,
        max_retries,
        metadata,
    };
    Ok((context, remaining, session))
}

pub fn legacy_children_for_context(
    parse_session: &ParseSessionState,
    current_layer_context: &CurrentLayerContext,
) -> Result<CurrentLayerResult> {
    let Some(compat_full_tree) = &parse_session.compat_full_tree else {
        anyhow::bail!("legacy compatibility tree is missing");
    };
    let full_tree = serde_json::from_value::<SemanticNode>(compat_full_tree.clone())?;
    let mut children = Vec::new();
    let mut atomic_parent_ids = Vec::new();
    let mut atomic_content_pointers_by_parent = Map::new();
    for parent_id in &current_layer_context.parent_node_ids {
        let Some(parent) = find_semantic_node(&full_tree, parent_id) else {
            anyhow::bail!("legacy compatibility tree is missing frontier parent {parent_id:?}");
        };
        if parent.child_nodes.len() < 2 {
            atomic_parent_ids.push(parent_id.clone());
            let pointers = if parent.total_content_pointers.is_empty() {
                parent.aggregate_content_pointers.clone()
            } else {
                parent.total_content_pointers.clone()
            };
            atomic_content_pointers_by_parent.insert(parent_id.clone(), json!(pointers));
            continue;
        }
        for child in &parent.child_nodes {
            let Some(node_id) = &child.node_id else {
                anyhow::bail!("required_node_id is missing");
            };
            let mut metadata = Map::new();
            metadata.insert("source".to_string(), json!("legacy_compat"));
            children.push(LayerChildCandidate {
                node_id: node_id.clone(),
                parent_node_id: parent_id.clone(),
                title: child.title.clone(),
                node_type: child.node_type.clone(),
                total_content_pointers: child.total_content_pointers.clone(),
                expandable: !child.child_nodes.is_empty(),
                metadata,
            });
        }
    }
    atomic_parent_ids.sort();
    let mut metadata = Map::new();
    metadata.insert("source".to_string(), json!("legacy_compat"));
    Ok(CurrentLayerResult {
        children,
        satisfied: Some(true),
        review_rounds: 0,
        atomic_retained: !atomic_parent_ids.is_empty(),
        atomic_parent_ids,
        atomic_content_pointers_by_parent,
        metadata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn fingerprints_sorted_source_map() {
        let fingerprint = source_map_fingerprint(&json!({
            "a": {"text": "first"},
            "z": {"text": "second"},
        }));
        assert_eq!(
            fingerprint,
            source_map_fingerprint(&json!({
                "a": {"text": "first"},
                "z": {"text": "second"},
            }))
        );
    }

    #[test]
    fn initializes_workflow_session_and_extracts_legacy_children() {
        let source_map = json!({
            "z": {"text": "second", "participates_in_semantic_text": true},
            "a": {"text": "first"},
        });
        let collection = SourceCollection {
            collection_id: "doc".to_string(),
            title: "Document".to_string(),
        };
        let (session, frontier, root) = initialize_parse_session(
            &collection,
            &source_map,
            10,
            true,
            SplitStrategy::ExcerptFirst,
            SplitStrategy::BoundaryFirst,
        )
        .unwrap();
        assert_eq!(root.node_id.as_deref(), Some("doc|root"));
        assert_eq!(root.total_content_pointers[0].source_cluster_id, "a");
        assert_eq!(root.total_content_pointers[0].end_char, -1);
        assert_eq!(session.mode, "workflow_layered");
        assert_eq!(frontier[0].parent_node_id, "doc|root");
        assert!(assert_source_map_matches_session(&session, &source_map).is_ok());

        let mut session = session.clone();
        session.compat_full_tree = Some(serde_json::to_value(root.clone()).unwrap());
        let (context, remaining, updated) =
            prepare_layer_frontier(&session, &frontier, &root, 3, None).unwrap();
        assert_eq!(context.parent_titles, vec!["Document"]);
        assert!(remaining.is_empty());
        assert_eq!(updated.current_depth, 0);
        let result = legacy_children_for_context(&session, &context).unwrap();
        assert!(result.children.is_empty());
        assert!(result.atomic_retained);
        assert_eq!(result.atomic_parent_ids, vec!["doc|root"]);
    }
}
