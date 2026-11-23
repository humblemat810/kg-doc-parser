use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::semantic::{HydratedTextPointer, SemanticNode};
use crate::semantic_graph_import::pointer_source_validation_error;
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
    #[serde(default)]
    pub child_candidates: Vec<LayerChildCandidate>,
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
    #[serde(default)]
    pub validation_diagnostics: Vec<Value>,
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
    sorted_queue.sort_by_key(|item| (item.depth, item.order));
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
                child_candidates: Vec::new(),
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
        validation_diagnostics: Vec::new(),
    })
}

fn parse_pointer_override(pointers: &Value) -> Result<Vec<HydratedTextPointer>> {
    serde_json::from_value::<Vec<HydratedTextPointer>>(pointers.clone())
        .map_err(|error| anyhow::anyhow!("atomic pointer override must be typed: {error}"))
}

fn atomic_pointer_override_errors(
    parent_id: &str,
    current_layer_result: &CurrentLayerResult,
    parser_source_map: Option<&Value>,
) -> Result<Vec<String>> {
    let Some(raw_pointers) = current_layer_result
        .atomic_content_pointers_by_parent
        .get(parent_id)
    else {
        return Ok(Vec::new());
    };
    if raw_pointers.is_null() {
        return Ok(Vec::new());
    }
    if raw_pointers.as_array().is_some_and(Vec::is_empty) {
        return Ok(vec![format!(
            "parent {parent_id}: atomic pointer override must be a non-empty list"
        )]);
    }
    let Some(source_map) = parser_source_map else {
        return Ok(vec![format!(
            "parent {parent_id}: atomic pointer override requires an authoritative source map"
        )]);
    };
    let pointers = parse_pointer_override(raw_pointers)?;
    let errors = pointers
        .iter()
        .enumerate()
        .filter_map(|(index, pointer)| {
            pointer_source_validation_error(pointer, source_map).map(|error| {
                format!("parent {parent_id}: atomic pointer override {index}: {error}")
            })
        })
        .collect();
    Ok(errors)
}

fn ensure_unique_ids(tree: &SemanticNode) -> Result<()> {
    let mut seen = std::collections::HashSet::new();
    let mut pending = vec![tree];
    while let Some(node) = pending.pop() {
        let node_id = node
            .node_id
            .clone()
            .ok_or_else(|| anyhow::anyhow!("semantic tree contains a node without an id"))?;
        if !seen.insert(node_id.clone()) {
            anyhow::bail!("duplicate node id {node_id:?}");
        }
        pending.extend(node.child_nodes.iter());
    }
    Ok(())
}

fn apply_atomic_retention(
    node: &mut SemanticNode,
    retained_parent_ids: &std::collections::HashSet<String>,
    grouped_counts: &std::collections::HashMap<String, usize>,
    atomic_retained: bool,
    atomic_pointers_by_parent: &Map<String, Value>,
) -> Result<()> {
    let node_id = node.node_id.clone().unwrap_or_default();
    if atomic_retained
        && grouped_counts.get(&node_id).copied().unwrap_or(0) == 0
        && retained_parent_ids.contains(&node_id)
    {
        if node.metadata.get("atomic_retained") != Some(&json!(true)) {
            node.metadata["atomic_retained"] = json!(true);
        }
        if let Some(raw_pointers) = atomic_pointers_by_parent.get(&node_id) {
            if !raw_pointers.is_null() {
                node.total_content_pointers = parse_pointer_override(raw_pointers)?;
            }
        }
    }
    for child in &mut node.child_nodes {
        apply_atomic_retention(
            child,
            retained_parent_ids,
            grouped_counts,
            atomic_retained,
            atomic_pointers_by_parent,
        )?;
    }
    Ok(())
}

fn build_proposed_children(
    current_layer_result: &CurrentLayerResult,
    current_depth: u64,
) -> Result<std::collections::HashMap<String, Vec<SemanticNode>>> {
    let mut children_by_parent: std::collections::HashMap<String, Vec<SemanticNode>> =
        std::collections::HashMap::new();
    for child in &current_layer_result.children {
        let bucket = children_by_parent
            .entry(child.parent_node_id.clone())
            .or_default();
        if bucket
            .iter()
            .any(|existing| existing.node_id.as_deref() == Some(child.node_id.as_str()))
        {
            anyhow::bail!("duplicate proposed child node id {:?}", child.node_id);
        }
        let mut materialized_ids = std::collections::HashSet::new();
        let mut materialized_children = Vec::new();
        for materialized in &child.child_candidates {
            if !materialized_ids.insert(materialized.node_id.clone()) {
                anyhow::bail!(
                    "duplicate materialized child node id {:?}",
                    materialized.node_id
                );
            }
            materialized_children.push(SemanticNode {
                node_id: Some(materialized.node_id.clone()),
                parent_id: Some(child.node_id.clone()),
                node_type: materialized.node_type.clone(),
                title: materialized.title.clone(),
                summary: String::new(),
                total_content_pointers: materialized.total_content_pointers.clone(),
                aggregate_content_pointers: Vec::new(),
                child_nodes: Vec::new(),
                level_from_root: current_depth + 2,
                metadata: Value::Object(materialized.metadata.clone()),
            });
        }
        bucket.push(SemanticNode {
            node_id: Some(child.node_id.clone()),
            parent_id: Some(child.parent_node_id.clone()),
            node_type: child.node_type.clone(),
            title: child.title.clone(),
            summary: String::new(),
            total_content_pointers: child.total_content_pointers.clone(),
            aggregate_content_pointers: Vec::new(),
            child_nodes: materialized_children,
            level_from_root: current_depth + 1,
            metadata: Value::Object(child.metadata.clone()),
        });
    }
    Ok(children_by_parent)
}

fn insert_proposed_children(
    node: &mut SemanticNode,
    children_by_parent: &std::collections::HashMap<String, Vec<SemanticNode>>,
) {
    let node_id = node.node_id.clone().unwrap_or_default();
    if let Some(proposed_children) = children_by_parent.get(&node_id) {
        let mut updated_children = node.child_nodes.clone();
        for proposed in proposed_children {
            if let Some(existing) = updated_children
                .iter_mut()
                .find(|existing| existing.node_id == proposed.node_id)
            {
                if !existing.child_nodes.is_empty() && proposed.child_nodes.is_empty() {
                    let preserved_children = existing.child_nodes.clone();
                    *existing = proposed.clone();
                    existing.child_nodes = preserved_children;
                } else {
                    *existing = proposed.clone();
                }
            } else {
                updated_children.push(proposed.clone());
            }
        }
        node.child_nodes = updated_children;
    }
    for child in &mut node.child_nodes {
        insert_proposed_children(child, children_by_parent);
    }
}

pub fn commit_layer_children(
    semantic_tree: &SemanticNode,
    current_layer_result: &CurrentLayerResult,
    current_depth: u64,
    parent_node_ids: Option<&[String]>,
    parser_source_map: Option<&Value>,
    max_depth: Option<u64>,
) -> Result<SemanticNode> {
    crate::semantic_graph_import::validate_semantic_tree_integrity(
        semantic_tree,
        parser_source_map,
        max_depth,
    )?;
    let mut tree = semantic_tree.clone();
    ensure_unique_ids(&tree)?;
    if !current_layer_result.validation_diagnostics.is_empty() {
        anyhow::bail!("layer contains validation diagnostics and cannot be committed");
    }
    let mut grouped_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for child in &current_layer_result.children {
        *grouped_counts
            .entry(child.parent_node_id.clone())
            .or_default() += 1;
        if child.child_candidates.len() == 1 {
            anyhow::bail!(
                "nested unary refinement under proposed child {:?}",
                child.node_id
            );
        }
    }
    if !current_layer_result.children.is_empty() && grouped_counts.values().any(|count| *count == 1)
    {
        anyhow::bail!("unary refinement cannot be committed");
    }
    if current_layer_result.children.is_empty() && !current_layer_result.atomic_retained {
        anyhow::bail!("empty layer requires explicit atomic retention");
    }
    let parent_node_ids = parent_node_ids.unwrap_or_default();
    let children_by_parent = build_proposed_children(current_layer_result, current_depth)?;
    insert_proposed_children(&mut tree, &children_by_parent);
    if !parent_node_ids.is_empty()
        && current_layer_result.children.is_empty()
        && current_layer_result.atomic_retained
        && parser_source_map.is_none()
    {
        anyhow::bail!("atomic retention requires an authoritative source map");
    }
    let atomic_retained = current_layer_result.atomic_retained;
    if !parent_node_ids.is_empty() && atomic_retained {
        let errors = parent_node_ids
            .iter()
            .map(|parent_id| {
                atomic_pointer_override_errors(parent_id, current_layer_result, parser_source_map)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        if !errors.is_empty() {
            anyhow::bail!(
                "{}",
                errors
                    .iter()
                    .take(8)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("; ")
            );
        }
    }
    let omitted_parent_ids = parent_node_ids
        .iter()
        .filter(|parent_id| grouped_counts.get(*parent_id).copied().unwrap_or(0) == 0)
        .cloned()
        .collect::<Vec<_>>();
    if !omitted_parent_ids.is_empty()
        && atomic_retained
        && !current_layer_result.atomic_parent_ids.is_empty()
    {
        let undeclared = omitted_parent_ids
            .iter()
            .filter(|parent_id| !current_layer_result.atomic_parent_ids.contains(parent_id))
            .collect::<Vec<_>>();
        if !undeclared.is_empty() {
            anyhow::bail!(
                "layer omitted parents without an explicit atomic disposition: {}",
                undeclared
                    .iter()
                    .map(|parent_id| parent_id.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }
    if !omitted_parent_ids.is_empty() && !atomic_retained {
        anyhow::bail!(
            "layer omitted parents without atomic retention: {}",
            omitted_parent_ids.join(", ")
        );
    }
    if !omitted_parent_ids.is_empty() {
        let Some(source_map) = parser_source_map else {
            anyhow::bail!("atomic retention requires an authoritative source map");
        };
        for parent_id in &omitted_parent_ids {
            let Some(parent) = crate::parser_core::find_semantic_node(&tree, parent_id) else {
                anyhow::bail!("atomic parent does not exist: {parent_id}");
            };
            let pointers = if parent.total_content_pointers.is_empty() {
                parent.aggregate_content_pointers.clone()
            } else {
                parent.total_content_pointers.clone()
            };
            if pointers.is_empty() {
                anyhow::bail!("atomic parent {parent_id:?} has no verified source pointer");
            }
            let errors = pointers
                .iter()
                .filter_map(|pointer| pointer_source_validation_error(pointer, source_map))
                .take(4)
                .collect::<Vec<_>>();
            if !errors.is_empty() {
                anyhow::bail!(
                    "atomic parent {parent_id:?} has invalid source pointer: {}",
                    errors.join("; ")
                );
            }
        }
    }
    for parent_id in grouped_counts.keys() {
        if crate::parser_core::find_semantic_node(&tree, parent_id).is_none() {
            anyhow::bail!("proposed child parent does not exist: {parent_id}");
        }
        if let Some(allowed_parent_ids) = Some(parent_node_ids) {
            if !allowed_parent_ids.contains(parent_id) {
                anyhow::bail!("proposed child parent is outside current layer: {parent_id}");
            }
        }
    }
    let retained_parent_ids = parent_node_ids
        .iter()
        .cloned()
        .collect::<std::collections::HashSet<_>>();
    let atomic_pointers_by_parent = current_layer_result
        .atomic_content_pointers_by_parent
        .clone();
    apply_atomic_retention(
        &mut tree,
        &retained_parent_ids,
        &grouped_counts,
        atomic_retained,
        &atomic_pointers_by_parent,
    )?;
    crate::semantic_graph_import::validate_semantic_tree_integrity(
        &tree,
        parser_source_map,
        max_depth,
    )?;
    Ok(tree)
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

    #[test]
    fn commits_children_and_preserves_existing_subtrees() {
        let child_a = SemanticNode {
            node_id: Some("existing-a".to_string()),
            parent_id: Some("parent".to_string()),
            node_type: "TEXT_FLOW".to_string(),
            title: "Existing A".to_string(),
            summary: String::new(),
            total_content_pointers: Vec::new(),
            aggregate_content_pointers: Vec::new(),
            child_nodes: Vec::new(),
            level_from_root: 1,
            metadata: json!({}),
        };
        let child_b = SemanticNode {
            node_id: Some("existing-b".to_string()),
            parent_id: Some("parent".to_string()),
            node_type: "TEXT_FLOW".to_string(),
            title: "Existing B".to_string(),
            summary: String::new(),
            total_content_pointers: Vec::new(),
            aggregate_content_pointers: Vec::new(),
            child_nodes: Vec::new(),
            level_from_root: 1,
            metadata: json!({}),
        };
        let parent = SemanticNode {
            node_id: Some("parent".to_string()),
            parent_id: Some("root".to_string()),
            node_type: "SECTION".to_string(),
            title: "Parent".to_string(),
            summary: String::new(),
            total_content_pointers: Vec::new(),
            aggregate_content_pointers: Vec::new(),
            child_nodes: vec![child_a, child_b],
            level_from_root: 0,
            metadata: json!({}),
        };
        let other_parent = SemanticNode {
            node_id: Some("other-parent".to_string()),
            parent_id: Some("root".to_string()),
            node_type: "SECTION".to_string(),
            title: "Other Parent".to_string(),
            summary: String::new(),
            total_content_pointers: Vec::new(),
            aggregate_content_pointers: Vec::new(),
            child_nodes: Vec::new(),
            level_from_root: 0,
            metadata: json!({}),
        };
        let root = SemanticNode {
            node_id: Some("root".to_string()),
            parent_id: None,
            node_type: "DOCUMENT_ROOT".to_string(),
            title: "Document".to_string(),
            summary: String::new(),
            total_content_pointers: Vec::new(),
            aggregate_content_pointers: Vec::new(),
            child_nodes: vec![parent, other_parent],
            level_from_root: 0,
            metadata: json!({}),
        };
        let proposed = CurrentLayerResult {
            children: vec![
                LayerChildCandidate {
                    node_id: "new-a".to_string(),
                    parent_node_id: "parent".to_string(),
                    title: "New A".to_string(),
                    node_type: "TEXT_FLOW".to_string(),
                    total_content_pointers: Vec::new(),
                    expandable: false,
                    metadata: Map::new(),
                    child_candidates: Vec::new(),
                },
                LayerChildCandidate {
                    node_id: "new-b".to_string(),
                    parent_node_id: "parent".to_string(),
                    title: "New B".to_string(),
                    node_type: "TEXT_FLOW".to_string(),
                    total_content_pointers: Vec::new(),
                    expandable: false,
                    metadata: Map::new(),
                    child_candidates: Vec::new(),
                },
            ],
            satisfied: Some(true),
            review_rounds: 0,
            atomic_retained: false,
            atomic_parent_ids: Vec::new(),
            atomic_content_pointers_by_parent: Map::new(),
            metadata: Map::new(),
            validation_diagnostics: Vec::new(),
        };
        let parent_ids = vec!["parent".to_string()];
        let committed =
            commit_layer_children(&root, &proposed, 0, Some(&parent_ids), None, Some(2)).unwrap();
        let committed_parent = find_semantic_node(&committed, "parent").unwrap();
        assert_eq!(committed_parent.child_nodes.len(), 4);
        assert!(find_semantic_node(&committed, "new-a").is_some());
        assert!(find_semantic_node(&committed, "existing-a").is_some());
    }
}
