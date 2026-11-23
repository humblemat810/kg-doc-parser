use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::document_grouping::GroupProposal;
use crate::document_models::{BoundingBox, DocumentTree, SourceItem};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OCRMetadata {
    pub provider: String,
    pub model: String,
    pub configuration_fingerprint: String,
    pub confidence: Option<f64>,
}

fn first_part<'a>(
    node: &'a crate::document_models::DocumentNode,
    items: &HashMap<&str, &'a SourceItem>,
) -> Option<&'a SourceItem> {
    node.parts
        .first()
        .and_then(|part| items.get(part.item_id.as_str()).copied())
}

fn last_part<'a>(
    node: &'a crate::document_models::DocumentNode,
    items: &HashMap<&str, &'a SourceItem>,
) -> Option<&'a SourceItem> {
    node.parts
        .last()
        .and_then(|part| items.get(part.item_id.as_str()).copied())
}

fn page_range(
    node: &crate::document_models::DocumentNode,
    items: &HashMap<&str, &SourceItem>,
) -> Option<(u64, u64)> {
    let pages = node
        .parts
        .iter()
        .filter_map(|part| {
            items
                .get(part.item_id.as_str())
                .and_then(|item| item.location.page_index)
        })
        .collect::<Vec<_>>();
    let first = pages.first().copied()?;
    let last = pages.last().copied()?;
    Some((first, last))
}

fn annotation<'a>(item: &'a SourceItem, key: &str) -> Option<&'a str> {
    item.annotations.get(key).map(String::as_str)
}

fn continuation_compatible(previous_bbox: &BoundingBox, current_bbox: &BoundingBox) -> bool {
    previous_bbox.coordinate_system == current_bbox.coordinate_system
        && (previous_bbox.x0 - current_bbox.x0).abs() <= 8.0
}

pub fn suggest_cross_page_continuations(tree: &DocumentTree) -> Result<Vec<GroupProposal>> {
    tree.validate()?;
    let items = tree
        .source_items
        .iter()
        .map(|item| (item.item_id.as_str(), item))
        .collect::<HashMap<_, _>>();
    let nodes = tree
        .nodes
        .iter()
        .map(|node| (node.node_id.as_str(), node))
        .collect::<HashMap<_, _>>();
    let root = nodes
        .get(tree.root_id.as_str())
        .ok_or_else(|| anyhow::anyhow!("document root is missing"))?;
    if root.children.is_empty() {
        return Ok(Vec::new());
    }
    let mut frontier = root
        .children
        .iter()
        .filter_map(|child| nodes.get(child.node_id.as_str()).copied())
        .collect::<Vec<_>>();
    if frontier.len() != root.children.len() || frontier.iter().any(|node| node.parts.is_empty()) {
        return Ok(Vec::new());
    }
    frontier.sort_by_key(|node| {
        page_range(node, &items)
            .map(|(first, _)| first)
            .unwrap_or(0)
    });
    let mut proposals = Vec::new();
    for window in frontier.windows(2) {
        let Some(previous) = window.first().copied() else {
            continue;
        };
        let Some(current) = window.get(1).copied() else {
            continue;
        };
        if previous.kind != "text" || current.kind != "text" {
            continue;
        }
        let Some((_previous_first, previous_last)) = page_range(previous, &items) else {
            continue;
        };
        let Some((current_first, _current_last)) = page_range(current, &items) else {
            continue;
        };
        if previous_last + 1 != current_first {
            continue;
        }
        let Some(previous_item) = last_part(previous, &items) else {
            continue;
        };
        let Some(current_item) = first_part(current, &items) else {
            continue;
        };
        if annotation(previous_item, "ocr:continuation_tail_evidence") != Some("True")
            || annotation(current_item, "ocr:continuation_head_evidence") != Some("True")
        {
            continue;
        }
        let previous_continuation_id = annotation(previous_item, "ocr:continuation_id");
        if previous_continuation_id.is_none()
            || previous_continuation_id != annotation(current_item, "ocr:continuation_id")
        {
            continue;
        }
        let previous_text = previous_item.text.as_deref().unwrap_or_default().trim_end();
        let current_text = current_item
            .text
            .as_deref()
            .unwrap_or_default()
            .trim_start();
        let previous_bbox = previous_item.location.bbox.as_ref();
        let current_bbox = current_item.location.bbox.as_ref();
        let incompatible_text = previous_text.is_empty()
            || current_text.is_empty()
            || previous_text
                .chars()
                .next_back()
                .is_some_and(|char| ".?!:;。！？".contains(char))
            || (!current_text.chars().next().is_some_and(char::is_lowercase)
                && !previous_text.ends_with(['-', '\u{2010}', '\u{2011}']));
        if incompatible_text
            || previous_bbox.is_none()
            || current_bbox.is_none()
            || !continuation_compatible(previous_bbox.unwrap(), current_bbox.unwrap())
        {
            continue;
        }
        proposals.push(GroupProposal {
            kind: "layout:cross-page-continuation".to_string(),
            members: vec![previous.node_id.clone(), current.node_id.clone()],
        });
    }
    Ok(proposals)
}

pub fn spatial_accuracy(value: &serde_json::Value, has_bbox: bool) -> String {
    let Some(value) = value.as_str() else {
        return "page".to_string();
    };
    if has_bbox && matches!(value, "page" | "region" | "line" | "word" | "character") {
        value.to_string()
    } else {
        "page".to_string()
    }
}

fn non_empty_string(value: Option<&serde_json::Value>) -> Option<String> {
    value
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

pub fn ocr_metadata(metadata: &serde_json::Value) -> Option<OCRMetadata> {
    let provider = non_empty_string(metadata.get("ocr_provider"))?;
    let model = non_empty_string(
        metadata
            .get("ocr_model_name")
            .or_else(|| metadata.get("ocr_model")),
    )?;
    let configuration_fingerprint =
        non_empty_string(metadata.get("ocr_configuration_fingerprint"))?;
    let confidence = metadata
        .get("ocr_confidence")
        .and_then(serde_json::Value::as_f64)
        .filter(|confidence| confidence.is_finite());
    Some(OCRMetadata {
        provider,
        model,
        configuration_fingerprint,
        confidence,
    })
}

pub fn page_annotations(
    metadata: &serde_json::Value,
) -> std::collections::BTreeMap<String, String> {
    let mut annotations = std::collections::BTreeMap::new();
    for key in [
        "incomplete_text",
        "incomplete_words_on_edge",
        "data_loss_likelihood",
        "scan_quality",
    ] {
        if let Some(value) = metadata.get(key) {
            if value.is_string() || value.is_boolean() || value.is_number() {
                annotations.insert(
                    format!("ocr:{key}"),
                    match value {
                        serde_json::Value::String(value) => value.clone(),
                        other => other.to_string(),
                    },
                );
            }
        }
    }
    annotations
}

pub fn unit_near_page_y_edge(
    unit: &BoundingBox,
    unit_metadata: &serde_json::Value,
    page_metadata: &serde_json::Value,
    edge: &str,
) -> bool {
    if unit_metadata
        .get("coordinate_system")
        .and_then(serde_json::Value::as_str)
        != Some("pixels")
    {
        return false;
    }
    let Some(page_y_min) = page_metadata
        .get("page_y_min")
        .and_then(serde_json::Value::as_f64)
        .filter(|value| value.is_finite())
    else {
        return false;
    };
    let Some(page_y_max) = page_metadata
        .get("page_y_max")
        .and_then(serde_json::Value::as_f64)
        .filter(|value| value.is_finite())
    else {
        return false;
    };
    if page_y_max <= page_y_min {
        return false;
    }
    let page_height = page_y_max - page_y_min;
    let edge_margin = 8.0f64.max(page_height * 0.025);
    if edge == "top" {
        unit.y0 <= page_y_min + edge_margin
    } else if edge == "bottom" {
        unit.y1 >= page_y_max - edge_margin
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document_models::{
        BoundingBox, CoordinateSystem, DocumentNode, DocumentTree, SourceAccuracy, SourceLocation,
    };
    use serde_json::json;
    use std::collections::BTreeMap;

    fn annotations(values: &[(&str, &str)]) -> BTreeMap<String, String> {
        values
            .iter()
            .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
            .collect()
    }

    fn item(
        item_id: &str,
        page_index: u64,
        text: &str,
        annotations: BTreeMap<String, String>,
    ) -> SourceItem {
        SourceItem {
            item_id: item_id.to_string(),
            kind: "text".to_string(),
            location: SourceLocation {
                page_index: Some(page_index),
                printed_page_label: None,
                bbox: Some(BoundingBox {
                    x0: 10.0,
                    y0: 0.0,
                    x1: 20.0,
                    y1: 10.0,
                    coordinate_system: CoordinateSystem::Pixels,
                }),
                start_seconds: None,
                end_seconds: None,
                structural_path: None,
                accuracy: SourceAccuracy::Region,
            },
            text: Some(text.to_string()),
            media: None,
            ocr_metadata: None,
            annotations,
        }
    }

    fn node(node_id: &str, item_id: &str, text: &str) -> DocumentNode {
        DocumentNode {
            node_id: node_id.to_string(),
            kind: "text".to_string(),
            parts: vec![crate::document_models::SourceSlice {
                item_id: item_id.to_string(),
                start: Some(0),
                end: Some(text.chars().count()),
                separator_before: String::new(),
            }],
            children: Vec::new(),
            summary: None,
            annotations: BTreeMap::new(),
        }
    }

    fn tree(first: SourceItem, second: SourceItem) -> DocumentTree {
        DocumentTree {
            schema_version: "1.0".to_string(),
            document_id: "doc".to_string(),
            source_revision: "ocr-v1".to_string(),
            tree_revision: 1,
            root_id: "root".to_string(),
            source_items: vec![first.clone(), second.clone()],
            nodes: vec![
                DocumentNode {
                    node_id: "root".to_string(),
                    kind: "document".to_string(),
                    parts: Vec::new(),
                    children: vec![
                        crate::document_models::ChildRef {
                            node_id: "first".to_string(),
                            separator_before: String::new(),
                        },
                        crate::document_models::ChildRef {
                            node_id: "second".to_string(),
                            separator_before: String::new(),
                        },
                    ],
                    summary: None,
                    annotations: BTreeMap::new(),
                },
                node("first", "p1", first.text.as_deref().unwrap_or_default()),
                node("second", "p2", second.text.as_deref().unwrap_or_default()),
            ],
            references: Vec::new(),
        }
    }

    #[test]
    fn suggests_signaled_cross_page_continuation() {
        let first = item(
            "p1",
            1,
            "This sentence continues",
            annotations(&[
                ("ocr:continuation_id", "join-1"),
                ("ocr:continuation_tail_evidence", "True"),
            ]),
        );
        let second = item(
            "p2",
            2,
            "without punctuation.",
            annotations(&[
                ("ocr:continuation_id", "join-1"),
                ("ocr:continuation_head_evidence", "True"),
            ]),
        );
        let proposals = suggest_cross_page_continuations(&tree(first, second)).unwrap();
        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].kind, "layout:cross-page-continuation");
        assert_eq!(proposals[0].members, vec!["first", "second"]);
    }

    #[test]
    fn rejects_unsignaled_or_terminal_join() {
        let first = item(
            "p1",
            1,
            "Complete sentence.",
            annotations(&[("ocr:continuation_tail_evidence", "True")]),
        );
        let second = item(
            "p2",
            2,
            "continues here.",
            annotations(&[("ocr:continuation_head_evidence", "True")]),
        );
        assert!(suggest_cross_page_continuations(&tree(first, second))
            .unwrap()
            .is_empty());

        let first = item("p1", 1, "continues", annotations(&[]));
        let second = item("p2", 2, "here.", annotations(&[]));
        assert!(suggest_cross_page_continuations(&tree(first, second))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn normalizes_spatial_metadata_and_edge_evidence() {
        let metadata = json!({
            "ocr_provider": "docling",
            "ocr_model_name": "docling-v2",
            "ocr_configuration_fingerprint": "cfg-1",
            "ocr_confidence": 0.9,
            "incomplete_text": true,
            "data_loss_likelihood": 0.1,
            "scan_quality": "high"
        });
        let parsed = ocr_metadata(&metadata).unwrap();
        assert_eq!(parsed.provider, "docling");
        assert_eq!(parsed.model, "docling-v2");
        assert_eq!(parsed.configuration_fingerprint, "cfg-1");
        assert_eq!(parsed.confidence, Some(0.9));
        let annotations = page_annotations(&metadata);
        assert_eq!(annotations["ocr:incomplete_text"], "true");
        assert_eq!(annotations["ocr:data_loss_likelihood"], "0.1");
        assert_eq!(annotations["ocr:scan_quality"], "high");
        assert_eq!(spatial_accuracy(&json!("region"), true), "region");
        assert_eq!(spatial_accuracy(&json!("region"), false), "page");
        assert_eq!(spatial_accuracy(&json!("unknown"), true), "page");

        let unit = BoundingBox {
            x0: 0.0,
            y0: 0.0,
            x1: 10.0,
            y1: 4.0,
            coordinate_system: CoordinateSystem::Pixels,
        };
        let unit_metadata = json!({"coordinate_system": "pixels"});
        let page_metadata = json!({"page_y_min": 0.0, "page_y_max": 100.0});
        assert!(unit_near_page_y_edge(
            &unit,
            &unit_metadata,
            &page_metadata,
            "top"
        ));
        assert!(!unit_near_page_y_edge(
            &unit,
            &unit_metadata,
            &page_metadata,
            "bottom"
        ));
    }
}
