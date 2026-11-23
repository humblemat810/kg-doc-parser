use anyhow::{bail, Result};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashSet};

use crate::pointer_matching::{
    all_exact_occurrences, best_occurrence_by_proximity, safe_slice, soft_exact_positions,
};
use crate::source_text::source_map_entry_text;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerwiseChild {
    pub parent_node_id: Option<String>,
    pub node_type: String,
    pub title: String,
    pub pointers: Vec<LayerwisePointer>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerwisePointer {
    pub source_cluster_id: String,
    pub start_char: i64,
    pub end_char: i64,
    pub verbatim_text: String,
    pub start_delimiter: Option<String>,
    pub end_delimiter: Option<String>,
    pub validation_method: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerwiseTarget {
    pub node_id: Option<String>,
    pub node_type: Option<String>,
    pub title: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LayerwisePatch {
    pub parent_node_id: Option<String>,
    pub node_type: Option<String>,
    pub title: Option<String>,
    pub pointers: Option<Vec<LayerwisePointer>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LayerwiseProposal {
    Add {
        parent_node_id: Option<String>,
        child: LayerwiseChild,
    },
    Delete {
        target: LayerwiseTarget,
    },
    Edit {
        target: LayerwiseTarget,
        patch: LayerwisePatch,
    },
}

#[derive(Debug, Clone)]
pub struct AppliedProposals {
    pub children: Vec<LayerwiseChild>,
    pub proposal_errors: Vec<String>,
}

fn normalize_title(value: &str) -> String {
    value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn pointer_key(pointer: &LayerwisePointer) -> (String, i64, i64) {
    (
        pointer.source_cluster_id.clone(),
        pointer.start_char,
        pointer.end_char,
    )
}

fn pointer_from_value(value: &Value) -> Result<LayerwisePointer> {
    let object = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("pointer must be an object"))?;
    Ok(LayerwisePointer {
        source_cluster_id: object
            .get("source_cluster_id")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("pointer source_cluster_id must be a string"))?
            .to_string(),
        start_char: object
            .get("start_char")
            .and_then(Value::as_i64)
            .ok_or_else(|| anyhow::anyhow!("pointer start_char must be an integer"))?,
        end_char: object
            .get("end_char")
            .and_then(Value::as_i64)
            .ok_or_else(|| anyhow::anyhow!("pointer end_char must be an integer"))?,
        verbatim_text: object
            .get("verbatim_text")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        start_delimiter: object
            .get("start_delimiter")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        end_delimiter: object
            .get("end_delimiter")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        validation_method: object
            .get("validation_method")
            .and_then(Value::as_str)
            .map(ToString::to_string),
    })
}

fn pointers_from_value(value: &Value) -> Result<Vec<LayerwisePointer>> {
    let Some(items) = value.as_array() else {
        bail!("pointers must be a list");
    };
    items.iter().map(pointer_from_value).collect()
}

fn child_from_value(value: &Value) -> Result<LayerwiseChild> {
    let object = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("child must be an object"))?;
    Ok(LayerwiseChild {
        parent_node_id: object
            .get("parent_node_id")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        node_type: object
            .get("node_type")
            .and_then(Value::as_str)
            .unwrap_or("TEXT_FLOW")
            .to_string(),
        title: object
            .get("title")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        pointers: pointers_from_value(&object.get("pointers").cloned().unwrap_or(json!([])))?,
    })
}

fn target_from_value(value: &Value) -> Result<LayerwiseTarget> {
    let object = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("target must be an object"))?;
    Ok(LayerwiseTarget {
        node_id: object
            .get("node_id")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        node_type: object
            .get("node_type")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        title: object
            .get("title")
            .and_then(Value::as_str)
            .map(ToString::to_string),
    })
}

fn proposal_from_value(value: &Value) -> Result<LayerwiseProposal> {
    let object = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("proposal must be an object"))?;
    match object
        .get("edit_type")
        .and_then(Value::as_str)
        .unwrap_or_default()
    {
        "ADD_NODE" => {
            if object.get("target").is_some_and(|value| !value.is_null())
                || object.get("patch").is_some_and(|value| !value.is_null())
            {
                bail!("ADD_NODE may not include 'target' or 'patch'");
            }
            let child_value = object
                .get("add")
                .ok_or_else(|| anyhow::anyhow!("ADD_NODE requires 'add'"))?;
            let child = child_from_value(child_value)?;
            Ok(LayerwiseProposal::Add {
                parent_node_id: child.parent_node_id.clone(),
                child,
            })
        }
        "DELETE_NODE" => {
            if object.get("add").is_some_and(|value| !value.is_null())
                || object.get("patch").is_some_and(|value| !value.is_null())
            {
                bail!("DELETE_NODE may not include 'add' or 'patch'");
            }
            let target_value = object
                .get("target")
                .ok_or_else(|| anyhow::anyhow!("DELETE_NODE requires 'target'"))?;
            Ok(LayerwiseProposal::Delete {
                target: target_from_value(target_value)?,
            })
        }
        "EDIT_NODE" => {
            if object.get("add").is_some_and(|value| !value.is_null()) {
                bail!("EDIT_NODE may not include 'add'");
            }
            let target_value = object
                .get("target")
                .ok_or_else(|| anyhow::anyhow!("EDIT_NODE requires 'target'"))?;
            let patch_value = object
                .get("patch")
                .ok_or_else(|| anyhow::anyhow!("EDIT_NODE requires 'patch'"))?;
            let patch_object = patch_value
                .as_object()
                .ok_or_else(|| anyhow::anyhow!("patch must be an object"))?;
            Ok(LayerwiseProposal::Edit {
                target: target_from_value(target_value)?,
                patch: LayerwisePatch {
                    parent_node_id: patch_object
                        .get("parent_node_id")
                        .and_then(Value::as_str)
                        .map(ToString::to_string),
                    node_type: patch_object
                        .get("node_type")
                        .and_then(Value::as_str)
                        .map(ToString::to_string),
                    title: patch_object
                        .get("title")
                        .and_then(Value::as_str)
                        .map(ToString::to_string),
                    pointers: match patch_object.get("pointers") {
                        None | Some(Value::Null) => None,
                        Some(value) => Some(pointers_from_value(value)?),
                    },
                },
            })
        }
        other => bail!("unsupported proposal edit_type: {other:?}"),
    }
}

fn cluster_text(source_map: &Value, cluster_id: &str) -> Option<String> {
    source_map
        .get(cluster_id)
        .map(source_map_entry_text)
        .filter(|text| !text.is_empty())
}

fn resolve_delimiter_pointer(
    pointer: &LayerwisePointer,
    source_map: &Value,
) -> Result<LayerwisePointer> {
    let (Some(start_delimiter), Some(end_delimiter)) =
        (&pointer.start_delimiter, &pointer.end_delimiter)
    else {
        bail!("delimiters are incomplete");
    };
    let Some(text) = cluster_text(source_map, &pointer.source_cluster_id) else {
        bail!("source cluster is missing or empty");
    };
    let start_matches = all_exact_occurrences(&text, start_delimiter);
    if start_matches.len() != 1 {
        bail!(
            "Start delimiter {start_delimiter:?} is not unique (found {} times)",
            start_matches.len()
        );
    }
    let end_matches = all_exact_occurrences(&text, end_delimiter);
    if end_matches.len() != 1 {
        bail!(
            "End delimiter {end_delimiter:?} is not unique (found {} times)",
            end_matches.len()
        );
    }
    let start_idx = start_matches[0].0;
    let end_idx = end_matches[0].1;
    if end_idx < start_idx {
        bail!("End delimiter occurs before Start delimiter");
    }
    Ok(LayerwisePointer {
        source_cluster_id: pointer.source_cluster_id.clone(),
        start_char: start_idx as i64,
        end_char: end_idx as i64,
        verbatim_text: safe_slice(&text, start_idx as isize, end_idx as isize),
        start_delimiter: Some(start_delimiter.clone()),
        end_delimiter: Some(end_delimiter.clone()),
        validation_method: Some("delimiter_exact".to_string()),
    })
}

fn fuzzy_threshold_for_length(length: usize) -> Option<f64> {
    match length {
        0..=10 => None,
        11..=20 => Some(0.99),
        21..=50 => Some(0.95),
        51..=100 => Some(0.90),
        _ => Some(0.85),
    }
}

pub fn correct_and_validate_pointer(
    pointer: &LayerwisePointer,
    source_map: &Value,
) -> Option<LayerwisePointer> {
    if pointer.start_delimiter.is_some() && pointer.end_delimiter.is_some() {
        if let Ok(resolved) = resolve_delimiter_pointer(pointer, source_map) {
            return Some(resolved);
        }
    }
    let cluster = source_map.get(&pointer.source_cluster_id)?;
    let cluster_id = cluster
        .get("id")
        .and_then(Value::as_str)
        .or(Some(pointer.source_cluster_id.as_str()))?;
    let source_text = source_map_entry_text(cluster);
    if source_text.is_empty() {
        return None;
    }
    if pointer.start_char >= 0 && pointer.end_char >= pointer.start_char {
        let proposed = safe_slice(
            &source_text,
            pointer.start_char as isize,
            pointer.end_char as isize,
        );
        if !proposed.is_empty() && proposed == pointer.verbatim_text {
            return Some(LayerwisePointer {
                source_cluster_id: cluster_id.to_string(),
                start_char: pointer.start_char,
                end_char: pointer.end_char,
                verbatim_text: proposed,
                start_delimiter: pointer.start_delimiter.clone(),
                end_delimiter: pointer.end_delimiter.clone(),
                validation_method: Some("trusted_exact".to_string()),
            });
        }
    }
    let (candidates, diagnostic) = soft_exact_positions(
        &source_text,
        &pointer.verbatim_text,
        fuzzy_threshold_for_length(pointer.verbatim_text.chars().count()),
        0.25,
        0.10,
    );
    if candidates.is_empty() {
        return None;
    }
    let (start, end) =
        best_occurrence_by_proximity(&candidates, pointer.start_char.max(0) as usize)?;
    Some(LayerwisePointer {
        source_cluster_id: cluster_id.to_string(),
        start_char: start as i64,
        end_char: end as i64,
        verbatim_text: safe_slice(&source_text, start as isize, end as isize),
        start_delimiter: None,
        end_delimiter: None,
        validation_method: diagnostic.map(|diagnostic| {
            serde_json::to_string(&diagnostic).unwrap_or_else(|_| "null".to_string())
        }),
    })
}

fn validate_child_pointers(child: &LayerwiseChild, source_map: &Value) -> Option<LayerwiseChild> {
    let mut pointers = Vec::with_capacity(child.pointers.len());
    for pointer in &child.pointers {
        pointers.push(correct_and_validate_pointer(pointer, source_map)?);
    }
    Some(LayerwiseChild {
        parent_node_id: child.parent_node_id.clone(),
        node_type: child.node_type.clone(),
        title: child.title.clone(),
        pointers,
    })
}

fn child_identity(child: &LayerwiseChild) -> (String, String, Vec<(String, i64, i64)>) {
    let mut keys = child.pointers.iter().map(pointer_key).collect::<Vec<_>>();
    keys.sort_unstable();
    (child.node_type.clone(), normalize_title(&child.title), keys)
}

fn dedupe_children(children: Vec<LayerwiseChild>) -> Vec<LayerwiseChild> {
    let mut seen = HashSet::new();
    let mut result = Vec::with_capacity(children.len());
    for child in children {
        if seen.insert(child_identity(&child)) {
            result.push(child);
        }
    }
    result
}

fn locate_index(
    children: &BTreeMap<usize, LayerwiseChild>,
    target: &LayerwiseTarget,
) -> Option<usize> {
    if let Some(node_id) = &target.node_id {
        if let Ok(index) = node_id.parse::<usize>() {
            if children.contains_key(&index) {
                return Some(index);
            }
        }
    }
    if let (Some(node_type), Some(title)) = (&target.node_type, &target.title) {
        let normalized_title = normalize_title(title);
        for (index, child) in children {
            if &child.node_type == node_type && normalize_title(&child.title) == normalized_title {
                return Some(*index);
            }
        }
    }
    None
}

pub fn apply_proposals(
    proposals: &[Value],
    children: &[LayerwiseChild],
    source_map: &Value,
) -> Result<AppliedProposals> {
    let mut out = children
        .iter()
        .enumerate()
        .map(|(index, child)| (index, child.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut next_id = children.len();
    let mut proposal_errors = Vec::new();
    for raw_proposal in proposals {
        let proposal = proposal_from_value(raw_proposal)?;
        match proposal {
            LayerwiseProposal::Delete { target } => {
                let index = locate_index(&out, &target);
                if let Some(index) = index.filter(|index| out.contains_key(index)) {
                    out.remove(&index);
                }
            }
            LayerwiseProposal::Add {
                parent_node_id,
                mut child,
            } => {
                if child.parent_node_id.is_none() {
                    child.parent_node_id = parent_node_id.or_else(|| {
                        out.values()
                            .next()
                            .and_then(|first| first.parent_node_id.clone())
                    });
                }
                match validate_child_pointers(&child, source_map) {
                    Some(fixed) => {
                        out.insert(next_id, fixed);
                        next_id += 1;
                    }
                    None => proposal_errors.push(format!(
                        "child pointer validation failed for {:?}",
                        child.title
                    )),
                }
            }
            LayerwiseProposal::Edit { target, patch } => {
                let Some(index) = locate_index(&out, &target) else {
                    continue;
                };
                let Some(existing) = out.get_mut(&index) else {
                    continue;
                };
                if let Some(parent_node_id) = patch.parent_node_id {
                    existing.parent_node_id = Some(parent_node_id);
                }
                if let Some(node_type) = patch.node_type {
                    existing.node_type = node_type;
                }
                if let Some(title) = patch.title {
                    existing.title = title;
                }
                if let Some(pointers) = patch.pointers {
                    existing.pointers = pointers;
                }
                let draft = existing.clone();
                match validate_child_pointers(&draft, source_map) {
                    Some(fixed) => {
                        out.insert(index, fixed);
                    }
                    None => {
                        out.remove(&index);
                    }
                }
            }
        }
    }
    Ok(AppliedProposals {
        children: dedupe_children(out.into_values().collect()),
        proposal_errors,
    })
}

#[allow(dead_code)]
fn unused_map(value: &Value) -> Option<&Map<String, Value>> {
    value.as_object()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn pointer(start: i64, end: i64, text: &str) -> LayerwisePointer {
        LayerwisePointer {
            source_cluster_id: "p1_c0".to_string(),
            start_char: start,
            end_char: end,
            verbatim_text: text.to_string(),
            start_delimiter: None,
            end_delimiter: None,
            validation_method: None,
        }
    }

    fn child(_node_id: &str, title: &str, pointers: Vec<LayerwisePointer>) -> LayerwiseChild {
        LayerwiseChild {
            parent_node_id: Some("parent".to_string()),
            node_type: "TEXT_FLOW".to_string(),
            title: title.to_string(),
            pointers,
        }
    }

    #[test]
    fn trusts_exact_pointer_and_repairs_offset() {
        let source_map = json!({
            "p1_c0": {"id": "p1_c0", "text": "Alpha Beta"}
        });
        let trusted = correct_and_validate_pointer(&pointer(0, 4, "Alpha"), &source_map).unwrap();
        assert_eq!(trusted.start_char, 0);
        assert_eq!(trusted.end_char, 4);
        assert_eq!(trusted.validation_method.as_deref(), Some("trusted_exact"));

        let repaired = correct_and_validate_pointer(&pointer(8, 12, "Beta"), &source_map).unwrap();
        assert_eq!(repaired.start_char, 6);
        assert_eq!(repaired.end_char, 9);
    }

    #[test]
    fn resolves_unique_delimiters() {
        let source_map = json!({
            "p1_c0": {"id": "p1_c0", "text": "<<Alpha>> Beta"}
        });
        let raw = LayerwisePointer {
            source_cluster_id: "p1_c0".to_string(),
            start_char: 0,
            end_char: 0,
            verbatim_text: String::new(),
            start_delimiter: Some("<<".to_string()),
            end_delimiter: Some(">>".to_string()),
            validation_method: None,
        };
        let resolved = correct_and_validate_pointer(&raw, &source_map).unwrap();
        assert_eq!(resolved.start_char, 0);
        assert_eq!(resolved.end_char, 8);
        assert_eq!(resolved.verbatim_text, "<<Alpha>>");
    }

    #[test]
    fn applies_add_delete_and_edit_with_dedupe() {
        let source_map = json!({
            "p1_c0": {"id": "p1_c0", "text": "Alpha Beta"}
        });
        let children = vec![
            child("0", "Alpha", vec![pointer(0, 4, "Alpha")]),
            child("1", "Beta", vec![pointer(6, 9, "Beta")]),
        ];
        let proposals = vec![
            json!({
                "edit_type": "ADD_NODE",
                "add": {
                    "parent_node_id": "parent",
                    "node_type": "TEXT_FLOW",
                    "title": "Gamma",
                    "pointers": [{"source_cluster_id": "p1_c0", "start_char": 6, "end_char": 9, "verbatim_text": "Beta"}]
                }
            }),
            json!({
                "edit_type": "DELETE_NODE",
                "target": {"node_type": "TEXT_FLOW", "title": "  gamma "}
            }),
            json!({
                "edit_type": "EDIT_NODE",
                "target": {"node_id": "1"},
                "patch": {"title": "Beta"}
            }),
        ];
        let applied = apply_proposals(&proposals, &children, &source_map).unwrap();
        assert!(applied.proposal_errors.is_empty());
        assert_eq!(applied.children.len(), 2);
        assert_eq!(applied.children[0].title, "Alpha");
        assert_eq!(applied.children[1].title, "Beta");
    }

    #[test]
    fn records_invalid_add_and_repairs_edit_title() {
        let source_map = json!({
            "p1_c0": {"id": "p1_c0", "text": "Alpha Beta"}
        });
        let children = vec![child("0", "Alpha", vec![pointer(0, 4, "Alpha")])];
        let proposals = vec![
            json!({
                "edit_type": "ADD_NODE",
                "add": {
                    "parent_node_id": "parent",
                    "node_type": "TEXT_FLOW",
                    "title": "Invalid",
                    "pointers": [{"source_cluster_id": "p1_c0", "start_char": 0, "end_char": 4, "verbatim_text": "Wrong"}]
                }
            }),
            json!({
                "edit_type": "EDIT_NODE",
                "target": {"node_type": "TEXT_FLOW", "title": "alpha"},
                "patch": {"title": "Alpha"}
            }),
        ];
        let applied = apply_proposals(&proposals, &children, &source_map).unwrap();
        assert_eq!(applied.proposal_errors.len(), 1);
        assert_eq!(applied.children.len(), 1);
        assert_eq!(applied.children[0].title, "Alpha");
    }
}
