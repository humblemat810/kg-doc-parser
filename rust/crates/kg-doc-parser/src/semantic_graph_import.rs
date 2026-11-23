use anyhow::{bail, Result};
use serde_json::{json, Map, Value};
use std::collections::HashMap;

use crate::semantic::{HydratedTextPointer, SemanticNode};
use crate::source_text::source_map_entry_text;

fn mapping(value: &Value, label: &str) -> Result<Map<String, Value>> {
    if value.is_null() {
        return Ok(Map::new());
    }
    let object = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("legacy graph field {label:?} must be an object"))?;
    Ok(object.clone())
}

fn object_list(value: &Value, label: &str) -> Result<Vec<Map<String, Value>>> {
    let Some(items) = value.as_array() else {
        bail!("legacy graph field {label:?} must be a list of objects");
    };
    items
        .iter()
        .enumerate()
        .map(|(index, item)| mapping(item, &format!("{label}[{index}]")))
        .collect()
}

fn string_list(value: &Value, label: &str) -> Result<Vec<String>> {
    let Some(items) = value.as_array() else {
        bail!("legacy graph field {label:?} must be a list of strings");
    };
    items
        .iter()
        .map(|item| {
            item.as_str().map(ToString::to_string).ok_or_else(|| {
                anyhow::anyhow!("legacy graph field {label:?} must be a list of strings")
            })
        })
        .collect()
}

fn legacy_string(value: &Value, label: &str, default: Option<&str>) -> Result<String> {
    if value.is_null() {
        if let Some(default) = default {
            return Ok(default.to_string());
        }
    }
    value
        .as_str()
        .map(ToString::to_string)
        .ok_or_else(|| anyhow::anyhow!("legacy graph field {label:?} must be a string"))
}

fn legacy_int(value: &Value, label: &str, default: Option<i64>) -> Result<i64> {
    if value.is_null() {
        if let Some(default) = default {
            return Ok(default);
        }
    }
    value
        .as_i64()
        .ok_or_else(|| anyhow::anyhow!("legacy graph field {label:?} must be an integer"))
}

fn node_type(value: &Value) -> Result<String> {
    match value.as_str() {
        Some("DOCUMENT_ROOT" | "TEXT_FLOW" | "KEY_VALUE_PAIR" | "TABLE") => {
            Ok(value.as_str().unwrap().to_string())
        }
        _ => bail!("unsupported legacy semantic node type: {value:?}"),
    }
}

fn normalized_uuid(value: &Value, label: &str) -> Result<String> {
    let raw = legacy_string(value, label, None)?;
    let parts = raw.split('-').collect::<Vec<_>>();
    let lengths = [8usize, 4, 4, 4, 12];
    if parts.len() != lengths.len()
        || parts.iter().zip(lengths).any(|(part, length)| {
            part.len() != length || !part.chars().all(|char| char.is_ascii_hexdigit())
        })
    {
        bail!("legacy graph field {label:?} must be a UUID");
    }
    Ok(raw.to_lowercase())
}

fn cluster_id_from_url(value: &Value) -> Result<String> {
    let url = legacy_string(value, "document_page_url", Some(""))?;
    let cluster_id = url
        .split_once('#')
        .map(|(_, cluster_id)| cluster_id)
        .unwrap_or("p1_c0");
    Ok(if cluster_id.is_empty() {
        "p1_c0".to_string()
    } else {
        cluster_id.to_string()
    })
}

fn end_char(value: &Value, label: &str) -> Result<i64> {
    if value.as_i64() == Some(1_000_000_000) {
        return Ok(-1);
    }
    legacy_int(value, label, Some(-1))
}

fn pointer(
    source_cluster_id: String,
    start_char: i64,
    end_char: i64,
    verbatim_text: String,
    validation_method: Option<String>,
) -> HydratedTextPointer {
    HydratedTextPointer {
        source_cluster_id,
        start_char,
        end_char,
        verbatim_text,
        start_delimiter: None,
        end_delimiter: None,
        validation_method,
    }
}

fn pointers_from_mentions(mentions: &[Map<String, Value>]) -> Result<Vec<HydratedTextPointer>> {
    if mentions.len() != 1 {
        bail!("legacy semantic nodes must contain exactly one mention group");
    }
    let spans = object_list(
        &mentions[0].get("spans").cloned().unwrap_or(Value::Null),
        "mentions.spans",
    )?;
    spans
        .iter()
        .map(|span| {
            let source_cluster_id = cluster_id_from_url(
                &span
                    .get("document_page_url")
                    .cloned()
                    .unwrap_or(Value::Null),
            )?;
            let validation_method = match span.get("verification") {
                None | Some(Value::Null) => None,
                Some(value) => Some(
                    value
                        .as_str()
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "legacy graph field 'verification' must be a string or null"
                            )
                        })?
                        .to_string(),
                ),
            };
            Ok(pointer(
                source_cluster_id,
                legacy_int(
                    &span.get("start_char").cloned().unwrap_or(Value::Null),
                    "start_char",
                    Some(0),
                )?,
                end_char(
                    &span.get("end_char").cloned().unwrap_or(Value::Null),
                    "end_char",
                )?,
                legacy_string(
                    &span.get("excerpt").cloned().unwrap_or(Value::Null),
                    "excerpt",
                    Some(""),
                )?,
                validation_method,
            ))
        })
        .collect()
}

pub fn kge_payload_to_semantic_tree(payload: &Value) -> Result<SemanticNode> {
    kge_payload_to_semantic_tree_with_source_map(payload, None)
}

pub fn kge_payload_to_semantic_tree_with_source_map(
    payload: &Value,
    source_map: Option<&Value>,
) -> Result<SemanticNode> {
    let nodes_data = object_list(
        &payload.get("nodes").cloned().unwrap_or(Value::Null),
        "nodes",
    )?;
    let edges_data = object_list(
        &payload.get("edges").cloned().unwrap_or(Value::Null),
        "edges",
    )?;
    let mut order = Vec::new();
    let mut by_id = HashMap::new();
    for node_data in &nodes_data {
        let metadata = mapping(
            &node_data.get("metadata").cloned().unwrap_or(Value::Null),
            "metadata",
        )?;
        let level_from_root = legacy_int(
            &metadata
                .get("level_from_root")
                .cloned()
                .unwrap_or(Value::Null),
            "level_from_root",
            None,
        )?;
        let mentions = object_list(
            &node_data.get("mentions").cloned().unwrap_or(Value::Null),
            "mentions",
        )?;
        let pointers = pointers_from_mentions(&mentions)?;
        let node_id = normalized_uuid(&node_data.get("id").cloned().unwrap_or(Value::Null), "id")?;
        let parent_id = match metadata.get("parent_id") {
            None | Some(Value::Null) => None,
            Some(value) => Some(normalized_uuid(value, "parent_id")?),
        };
        let semantic_node = SemanticNode {
            node_id: Some(node_id.clone()),
            parent_id,
            node_type: node_type(
                &metadata
                    .get("semantic_node_type")
                    .cloned()
                    .unwrap_or(json!("TEXT_FLOW")),
            )?,
            title: legacy_string(
                &node_data.get("label").cloned().unwrap_or(Value::Null),
                "label",
                Some(""),
            )?,
            summary: String::new(),
            total_content_pointers: pointers,
            aggregate_content_pointers: Vec::new(),
            child_nodes: Vec::new(),
            level_from_root: level_from_root as u64,
            metadata: json!({}),
        };
        by_id.insert(node_id.clone(), semantic_node);
        order.push(node_id);
    }
    let mut children_by_parent: Map<String, Value> = Map::new();
    for edge in &edges_data {
        let relation = edge
            .get("relation")
            .and_then(Value::as_str)
            .or_else(|| edge.get("predicate").and_then(Value::as_str));
        if relation != Some("HAS_CHILD") {
            continue;
        }
        let (parent_id, child_id) =
            if edge.contains_key("source_ids") || edge.contains_key("target_ids") {
                let source_ids = string_list(
                    &edge.get("source_ids").cloned().unwrap_or(json!([])),
                    "source_ids",
                )?;
                let target_ids = string_list(
                    &edge.get("target_ids").cloned().unwrap_or(json!([])),
                    "target_ids",
                )?;
                if source_ids.is_empty() || target_ids.is_empty() {
                    continue;
                }
                (source_ids[0].clone(), target_ids[0].clone())
            } else {
                let Some(parent_id) = edge.get("subject_id").and_then(Value::as_str) else {
                    continue;
                };
                let Some(child_id) = edge.get("object_id").and_then(Value::as_str) else {
                    continue;
                };
                if parent_id.is_empty() || child_id.is_empty() {
                    continue;
                }
                (parent_id.to_string(), child_id.to_string())
            };
        children_by_parent
            .entry(parent_id)
            .or_insert_with(|| json!([]))
            .as_array_mut()
            .unwrap()
            .push(json!(child_id));
    }
    for (parent_id, child_ids) in &children_by_parent {
        let child_ids = child_ids.as_array().unwrap();
        for child_id in child_ids {
            let child = by_id.get(child_id.as_str().unwrap()).cloned();
            if let (Some(parent), Some(child)) = (by_id.get_mut(parent_id.as_str()), child) {
                parent.child_nodes.push(child.clone());
            }
        }
    }
    let mut nodes = Vec::new();
    for node_id in order {
        nodes.push(by_id.remove(&node_id).unwrap());
    }
    let root = nodes
        .iter()
        .find(|node| node.parent_id.is_none())
        .cloned()
        .or_else(|| {
            let mut child_ids = children_by_parent
                .values()
                .flat_map(Value::as_array)
                .flatten()
                .filter_map(Value::as_str)
                .map(ToString::to_string)
                .collect::<Vec<_>>();
            child_ids.sort();
            child_ids.dedup();
            nodes
                .iter()
                .filter(|node| !child_ids.contains(&node.node_id.clone().unwrap_or_default()))
                .map(Clone::clone)
                .next()
                .or_else(|| nodes.first().cloned())
        });
    let Some(root) = root else {
        bail!("list index out of range");
    };
    validate_semantic_tree_integrity(&root, source_map, None)?;
    Ok(root)
}

pub fn pointer_source_validation_error(
    pointer: &HydratedTextPointer,
    source_map: &Value,
) -> Option<String> {
    let Some(source) = source_map.get(&pointer.source_cluster_id) else {
        return Some(format!(
            "unknown source cluster {:?}",
            pointer.source_cluster_id
        ));
    };
    let text = source_map_entry_text(source);
    if pointer.start_char < 0 {
        return Some("pointer start is negative".to_string());
    }
    let end = if pointer.end_char == -1 {
        text.chars().count() as i64 - 1
    } else {
        pointer.end_char
    };
    if end < pointer.start_char {
        return Some("pointer range is reversed".to_string());
    }
    if end >= text.chars().count() as i64 {
        return Some("pointer exceeds source bounds".to_string());
    }
    let actual = text
        .chars()
        .skip(pointer.start_char as usize)
        .take((end - pointer.start_char + 1) as usize)
        .collect::<String>();
    if actual != pointer.verbatim_text {
        return Some("pointer excerpt does not match authoritative source slice".to_string());
    }
    None
}

pub fn validate_semantic_tree_integrity(
    root: &SemanticNode,
    source_map: Option<&Value>,
    max_depth: Option<u64>,
) -> Result<()> {
    validate_node(root, None, 0, &mut Vec::new(), source_map, max_depth)?;
    validate_shape(root, 0, &mut Vec::new(), max_depth)
}

fn validate_node(
    node: &SemanticNode,
    expected_parent_id: Option<&str>,
    depth: u64,
    active_ids: &mut Vec<String>,
    source_map: Option<&Value>,
    max_depth: Option<u64>,
) -> Result<()> {
    let node_id = node.node_id.clone().unwrap_or_default();
    if node_id.is_empty() {
        bail!("semantic tree contains a node without an id");
    }
    if active_ids.contains(&node_id) {
        bail!("semantic tree contains a cycle at {node_id:?}");
    }
    let actual_parent_id = node.parent_id.as_deref();
    if actual_parent_id != expected_parent_id {
        bail!(
            "semantic tree parent mismatch for {node_id:?}: expected {expected_parent_id:?}, got {actual_parent_id:?}"
        );
    }
    if max_depth.is_some_and(|maximum| depth > maximum) {
        bail!(
            "semantic tree exceeds maximum depth {maximum:?}: {node_id}",
            maximum = max_depth.unwrap()
        );
    }
    for pointer in node
        .total_content_pointers
        .iter()
        .chain(node.aggregate_content_pointers.iter())
    {
        if pointer.start_char < 0 || pointer.end_char < -1 {
            bail!("invalid pointer bounds for node {node_id:?}");
        }
        if pointer.end_char != -1 && pointer.end_char < pointer.start_char {
            bail!("reversed pointer bounds for node {node_id:?}");
        }
        if let Some(source_map) = source_map {
            if let Some(error) = pointer_source_validation_error(pointer, source_map) {
                bail!("node {node_id:?}: {error}");
            }
        }
    }
    active_ids.push(node_id.clone());
    for child in &node.child_nodes {
        validate_node(
            child,
            Some(&node_id),
            depth + 1,
            active_ids,
            source_map,
            max_depth,
        )?;
    }
    active_ids.pop();
    Ok(())
}

fn validate_shape(
    node: &SemanticNode,
    depth: u64,
    active_ids: &mut Vec<String>,
    max_depth: Option<u64>,
) -> Result<()> {
    let object_id = format!("{:p}", node);
    if active_ids.contains(&object_id) {
        bail!("semantic tree contains a cycle while checking branching");
    }
    if node.child_nodes.len() == 1 {
        bail!(
            "unary semantic node: {:?}",
            node.node_id.clone().unwrap_or_default()
        );
    }
    if max_depth.is_some_and(|maximum| depth > maximum) {
        bail!(
            "semantic tree exceeds maximum depth {maximum:?}: {node_id}",
            maximum = max_depth.unwrap(),
            node_id = node.node_id.clone().unwrap_or_default()
        );
    }
    let atomic_retained = node
        .metadata
        .get("atomic_retained")
        .and_then(Value::as_bool)
        == Some(true);
    if atomic_retained
        && node.total_content_pointers.is_empty()
        && node.aggregate_content_pointers.is_empty()
    {
        bail!(
            "atomic semantic node {node_id:?} has no verified source pointer",
            node_id = node.node_id.clone().unwrap_or_default()
        );
    }
    active_ids.push(object_id);
    for child in &node.child_nodes {
        validate_shape(child, depth + 1, active_ids, max_depth)?;
    }
    active_ids.pop();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn reconstructs_legacy_graph_tree() {
        let payload = json!({
            "nodes": [
                {
                    "id": "1e2d3586-0b30-41a7-9a95-5a4fcb23c0c1",
                    "label": "Document",
                    "mentions": [{"spans": []}],
                    "metadata": {"level_from_root": 0, "semantic_node_type": "DOCUMENT_ROOT"}
                },
                {
                    "id": "2e2d3586-0b30-41a7-9a95-5a4fcb23c0c2",
                    "label": "Section",
                    "mentions": [{"spans": []}],
                    "metadata": {"level_from_root": 1, "parent_id": "1e2d3586-0b30-41a7-9a95-5a4fcb23c0c1"}
                },
                {
                    "id": "3e2d3586-0b30-41a7-9a95-5a4fcb23c0c3",
                    "label": "Appendix",
                    "mentions": [{"spans": []}],
                    "metadata": {"level_from_root": 1, "parent_id": "1e2d3586-0b30-41a7-9a95-5a4fcb23c0c1"}
                }
            ],
            "edges": [
                {
                    "subject_id": "1e2d3586-0b30-41a7-9a95-5a4fcb23c0c1",
                    "object_id": "2e2d3586-0b30-41a7-9a95-5a4fcb23c0c2",
                    "predicate": "HAS_CHILD"
                },
                {
                    "subject_id": "1e2d3586-0b30-41a7-9a95-5a4fcb23c0c1",
                    "object_id": "3e2d3586-0b30-41a7-9a95-5a4fcb23c0c3",
                    "predicate": "HAS_CHILD"
                }
            ]
        });
        let tree = kge_payload_to_semantic_tree(&payload).unwrap();
        assert_eq!(tree.node_type, "DOCUMENT_ROOT");
        assert_eq!(tree.child_nodes[0].node_type, "TEXT_FLOW");
        assert_eq!(tree.child_nodes.len(), 2);
        assert_eq!(
            tree.child_nodes[0].parent_id.as_deref(),
            Some("1e2d3586-0b30-41a7-9a95-5a4fcb23c0c1")
        );
    }

    #[test]
    fn extracts_and_validates_pointer_grounding() {
        let node_id = "1e2d3586-0b30-41a7-9a95-5a4fcb23c0c1";
        let payload = json!({
            "nodes": [{
                "id": node_id,
                "label": "Root",
                "metadata": {"level_from_root": 0, "semantic_node_type": "DOCUMENT_ROOT"},
                "mentions": [{"spans": [{
                    "document_page_url": "doc://unit#p1_c0",
                    "start_char": 0,
                    "end_char": 4,
                    "excerpt": "Alpha",
                    "verification": "exact"
                }]}]
            }],
            "edges": []
        });
        let source_map = json!({"p1_c0": {"text": "Alpha Beta"}});
        let tree =
            kge_payload_to_semantic_tree_with_source_map(&payload, Some(&source_map)).unwrap();
        assert_eq!(tree.total_content_pointers[0].source_cluster_id, "p1_c0");
        assert_eq!(tree.total_content_pointers[0].verbatim_text, "Alpha");
        assert_eq!(
            tree.total_content_pointers[0].validation_method.as_deref(),
            Some("exact")
        );
        assert!(
            pointer_source_validation_error(&tree.total_content_pointers[0], &source_map).is_none()
        );
    }

    #[test]
    fn rejects_unary_shape_and_bad_grounding() {
        let parent = "1e2d3586-0b30-41a7-9a95-5a4fcb23c0c1";
        let child = "2e2d3586-0b30-41a7-9a95-5a4fcb23c0c2";
        let payload = json!({
            "nodes": [
                {"id": parent, "label": "Parent", "metadata": {"level_from_root": 0, "semantic_node_type": "DOCUMENT_ROOT"}},
                {"id": child, "label": "Child", "metadata": {"level_from_root": 1, "parent_id": parent}}
            ],
            "edges": [{"subject_id": parent, "object_id": child, "predicate": "HAS_CHILD"}]
        });
        assert!(kge_payload_to_semantic_tree(&payload).is_err());

        let grounding_payload = json!({
            "nodes": [{
                "id": parent,
                "label": "Root",
                "metadata": {"level_from_root": 0, "semantic_node_type": "DOCUMENT_ROOT"},
                "mentions": [{"spans": [{
                    "document_page_url": "doc://unit#p1_c0",
                    "start_char": 0,
                    "end_char": 4,
                    "excerpt": "Wrong"
                }]}]
            }],
            "edges": []
        });
        let source_map = json!({"p1_c0": {"text": "Alpha Beta"}});
        assert!(kge_payload_to_semantic_tree_with_source_map(
            &grounding_payload,
            Some(&source_map)
        )
        .is_err());
    }
}
