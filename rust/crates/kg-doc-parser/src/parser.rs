use crate::semantic::{HydratedTextPointer, SemanticNode};
use serde_json::json;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParsingMode {
    Snippet,
    Delimiter,
}

impl ParsingMode {
    pub fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "snippet" => Ok(Self::Snippet),
            "delimiter" => Ok(Self::Delimiter),
            _ => Err(anyhow::anyhow!("parsing_mode must be snippet or delimiter")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ParseResult {
    pub document_tree: SemanticNode,
    pub source_map: HashMap<String, serde_json::Value>,
}

fn pointer(cluster_id: &str, text: &str) -> HydratedTextPointer {
    HydratedTextPointer {
        source_cluster_id: cluster_id.to_string(),
        start_char: 0,
        end_char: text.chars().count().saturating_sub(1) as u64,
        verbatim_text: text.to_string(),
    }
}

pub fn text_to_ocr_format(text: &str, filename: &str) -> serde_json::Value {
    let mut envelope = serde_json::Map::new();
    envelope.insert(
        filename.to_string(),
        json!([{
            "pdf_page_num": 1,
            "OCR_text_clusters": [{
                "text": text,
                "bb_x_min": 0,
                "bb_y_min": 0,
                "bb_x_max": 1000,
                "bb_y_max": 1000,
                "cluster_number": 0
            }],
            "non_text_objects": []
        }]),
    );
    serde_json::Value::Object(envelope)
}

fn paragraph_node(cluster_id: &str, paragraph: &str, parent_id: &str, level: u64) -> SemanticNode {
    let title = paragraph
        .split_whitespace()
        .take(8)
        .collect::<Vec<_>>()
        .join(" ");
    let title = if title.is_empty() {
        "Untitled".to_string()
    } else {
        title
    };
    SemanticNode {
        node_id: None,
        parent_id: Some(parent_id.to_string()),
        node_type: "TEXT_FLOW".to_string(),
        title,
        summary: String::new(),
        total_content_pointers: vec![pointer(cluster_id, paragraph)],
        aggregate_content_pointers: Vec::new(),
        child_nodes: Vec::new(),
        level_from_root: level,
        metadata: json!({"atomic_retained": true}),
    }
    .with_stable_id(Some(parent_id))
}

pub fn parse_doc_text(
    text: &str,
    doc_id: &str,
    parsing_mode: ParsingMode,
    max_depth: u64,
) -> anyhow::Result<ParseResult> {
    if max_depth == 0 {
        anyhow::bail!("max_depth must be greater than zero");
    }
    if text.trim().is_empty() {
        anyhow::bail!("text must be non-empty");
    }
    let cluster_id = format!("{doc_id}:text:1");
    let root_id = format!("{doc_id}:root");
    let mut child_nodes = Vec::new();
    if parsing_mode == ParsingMode::Delimiter {
        let delimiters = ["\n\n\n", "\n\n", "\n"];
        let mut remaining = text;
        while let Some(delimiter) = delimiters
            .iter()
            .find(|delimiter| remaining.contains(**delimiter))
        {
            for chunk in remaining.split(delimiter) {
                let chunk = chunk.trim();
                if !chunk.is_empty() {
                    child_nodes.push(paragraph_node(&cluster_id, chunk, &root_id, 1));
                }
            }
            remaining = "";
        }
    } else {
        let mut current_section: Option<(String, Vec<String>)> = None;
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("#") {
                if let Some((title, paragraphs)) = current_section.take() {
                    let section_id = format!("{root_id}:{title}");
                    let children = paragraphs
                        .into_iter()
                        .filter(|paragraph| !paragraph.trim().is_empty())
                        .map(|paragraph| paragraph_node(&cluster_id, &paragraph, &section_id, 2))
                        .collect::<Vec<_>>();
                    child_nodes.push(
                        SemanticNode {
                            node_id: None,
                            parent_id: Some(root_id.clone()),
                            node_type: "SECTION".to_string(),
                            title,
                            summary: String::new(),
                            total_content_pointers: Vec::new(),
                            aggregate_content_pointers: Vec::new(),
                            child_nodes: children,
                            level_from_root: 1,
                            metadata: json!({}),
                        }
                        .with_stable_id(Some(&root_id)),
                    );
                }
                let title = trimmed.trim_start_matches('#').trim().to_string();
                current_section = Some((
                    if title.is_empty() {
                        "Untitled".to_string()
                    } else {
                        title
                    },
                    Vec::new(),
                ));
            } else if !trimmed.is_empty() {
                if let Some((_, paragraphs)) = current_section.as_mut() {
                    paragraphs.push(trimmed.to_string());
                }
            }
        }
        if let Some((title, paragraphs)) = current_section {
            let section_id = format!("{root_id}:{title}");
            let children = paragraphs
                .into_iter()
                .filter(|paragraph| !paragraph.trim().is_empty())
                .map(|paragraph| paragraph_node(&cluster_id, &paragraph, &section_id, 2))
                .collect::<Vec<_>>();
            child_nodes.push(
                SemanticNode {
                    node_id: None,
                    parent_id: Some(root_id.clone()),
                    node_type: "SECTION".to_string(),
                    title,
                    summary: String::new(),
                    total_content_pointers: Vec::new(),
                    aggregate_content_pointers: Vec::new(),
                    child_nodes: children,
                    level_from_root: 1,
                    metadata: json!({}),
                }
                .with_stable_id(Some(&root_id)),
            );
        }
    }
    if child_nodes.len() <= 1 {
        child_nodes = vec![paragraph_node(&cluster_id, text.trim(), &root_id, 1)];
    }
    let document_tree = SemanticNode {
        node_id: None,
        parent_id: None,
        node_type: "DOCUMENT".to_string(),
        title: doc_id.to_string(),
        summary: String::new(),
        total_content_pointers: Vec::new(),
        aggregate_content_pointers: vec![pointer(&cluster_id, text)],
        child_nodes,
        level_from_root: 0,
        metadata: json!({
            "parsing_mode": if parsing_mode == ParsingMode::Delimiter { "delimiter" } else { "snippet" },
            "max_depth": max_depth,
        }),
    }
    .with_stable_id(None);
    let mut source_map = HashMap::new();
    source_map.insert(
        cluster_id.clone(),
        json!({
            "doc_id": doc_id,
            "text": text,
        }),
    );
    Ok(ParseResult {
        document_tree,
        source_map,
    })
}

pub fn parse_file(
    path: &Path,
    parsing_mode: ParsingMode,
    max_depth: u64,
) -> anyhow::Result<ParseResult> {
    let text = std::fs::read_to_string(path)?;
    let doc_id = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("document");
    parse_doc_text(&text, doc_id, parsing_mode, max_depth)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snippet_mode_builds_sections_and_paragraphs() {
        let result = parse_doc_text(
            "# Alpha\n\nFirst paragraph.\n\n# Beta\n\nSecond paragraph.",
            "doc",
            ParsingMode::Snippet,
            10,
        )
        .unwrap();
        assert_eq!(result.document_tree.node_type, "DOCUMENT");
        assert_eq!(result.document_tree.child_nodes.len(), 2);
        assert_eq!(result.document_tree.child_nodes[0].child_nodes.len(), 1);
    }

    #[test]
    fn delimiter_mode_falls_back_to_atomic_leaf() {
        let result = parse_doc_text("one\n\ntwo", "doc", ParsingMode::Delimiter, 10).unwrap();
        assert_eq!(result.document_tree.child_nodes.len(), 2);
    }

    #[test]
    fn rejects_empty_text() {
        assert!(parse_doc_text("", "doc", ParsingMode::Snippet, 10).is_err());
    }

    #[test]
    fn wraps_raw_text_for_ocr_input() {
        let raw = text_to_ocr_format("raw text", "input_text");
        let page = &raw["input_text"][0];
        assert_eq!(page["pdf_page_num"], 1);
        let cluster = &page["OCR_text_clusters"][0];
        assert_eq!(cluster["text"], "raw text");
        assert_eq!(cluster["cluster_number"], 0);
        assert_eq!(page["non_text_objects"], serde_json::Value::Array(vec![]));
    }
}
