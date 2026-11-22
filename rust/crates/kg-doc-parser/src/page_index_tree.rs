use crate::page_index_contracts::{PageIndexBlockSpec, PageIndexSourceRole};
use crate::pointer_matching::soft_exact_positions;
use crate::semantic::{HydratedTextPointer, SemanticNode};
use serde_json::{json, Value};

pub fn page_index_normalize_text(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

pub fn make_semantic_node(
    title: &str,
    node_type: &str,
    parent_id: Option<&str>,
    level_from_root: u64,
    pointers: Vec<HydratedTextPointer>,
    summary: &str,
    metadata: Option<Value>,
) -> SemanticNode {
    SemanticNode {
        node_id: None,
        parent_id: parent_id.map(str::to_string),
        node_type: node_type.to_string(),
        title: title.to_string(),
        summary: summary.to_string(),
        total_content_pointers: pointers,
        aggregate_content_pointers: Vec::new(),
        child_nodes: Vec::new(),
        level_from_root,
        metadata: metadata.unwrap_or_else(|| json!({})),
    }
}

pub fn resolve_pointer(
    unit_id: &str,
    page_text: &str,
    excerpt: &str,
    start_at: usize,
) -> anyhow::Result<HydratedTextPointer> {
    let needle = if excerpt.trim().is_empty() {
        page_text.trim()
    } else {
        excerpt.trim()
    };
    if !needle.is_empty() {
        if let Some(found) = page_text[start_at.min(page_text.len())..].find(needle) {
            let start_char = start_at.min(page_text.len()) + found;
            let end_char = start_char + needle.chars().count() - 1;
            return Ok(HydratedTextPointer {
                source_cluster_id: unit_id.to_string(),
                start_char: start_char as u64,
                end_char: end_char as u64,
                verbatim_text: page_text[start_char..start_char + needle.len()].to_string(),
            });
        }
    }
    let (matches, diagnostic) = soft_exact_positions(page_text, needle);
    if matches.is_empty() || diagnostic.is_none() {
        anyhow::bail!("unable to resolve excerpt against page text for {unit_id:?}: {needle:?}");
    }
    let (start_char, end_char) = matches[0];
    Ok(HydratedTextPointer {
        source_cluster_id: unit_id.to_string(),
        start_char: start_char as u64,
        end_char: end_char as u64,
        verbatim_text: page_text
            .chars()
            .skip(start_char)
            .take(end_char - start_char + 1)
            .collect(),
    })
}

fn materialize_block_tree_inner(
    block_specs: &[PageIndexBlockSpec],
    page_text: &str,
    unit_id: &str,
    parent_id: &str,
    level_from_root: u64,
    start_at: usize,
) -> anyhow::Result<(Vec<SemanticNode>, usize)> {
    let mut nodes = Vec::new();
    let mut cursor = start_at;
    for spec in block_specs {
        let pointer = resolve_pointer(unit_id, page_text, &spec.excerpt, cursor)?;
        cursor = pointer.end_char as usize + 1;
        let is_heading = spec.source_role == PageIndexSourceRole::Heading;
        let mut node = make_semantic_node(
            &spec.title,
            match spec.node_type {
                crate::page_index_contracts::PageIndexNodeType::Heading => "HEADING",
                crate::page_index_contracts::PageIndexNodeType::Paragraph => "PARAGRAPH",
                crate::page_index_contracts::PageIndexNodeType::Term => "TERM",
                crate::page_index_contracts::PageIndexNodeType::Table => "TABLE",
                crate::page_index_contracts::PageIndexNodeType::TableRowGroup => "TABLE_ROW_GROUP",
                crate::page_index_contracts::PageIndexNodeType::TableRow => "TABLE_ROW",
            },
            Some(parent_id),
            level_from_root,
            if is_heading {
                Vec::new()
            } else {
                vec![pointer.clone()]
            },
            spec.summary.trim(),
            Some(json!({
                "page_index_role": if is_heading { "heading_container" } else { "content" },
                "semantic_kind": if is_heading { "heading" } else { "content" },
                "summary_unavailable": spec.summary.trim().is_empty(),
            })),
        );
        let mut heading_leaf = None;
        if is_heading {
            let mut leaf = make_semantic_node(
                &spec.title,
                "HEADING_TEXT",
                node.node_id.as_deref(),
                level_from_root + 1,
                vec![pointer.clone()],
                spec.summary.trim(),
                Some(json!({
                    "page_index_role": "heading_text",
                    "semantic_kind": "heading_text",
                    "summary_unavailable": spec.summary.trim().is_empty(),
                })),
            );
            leaf = leaf.with_stable_id(node.node_id.as_deref());
            heading_leaf = Some(leaf);
        }
        let parent_for_children = node
            .node_id
            .clone()
            .unwrap_or_else(|| parent_id.to_string());
        let (child_nodes, next_cursor) = materialize_block_tree_inner(
            &spec.child_nodes,
            page_text,
            unit_id,
            &parent_for_children,
            level_from_root + 1,
            cursor,
        )?;
        cursor = next_cursor;
        if is_heading {
            let aggregate_end = (pointer.end_char as usize).max(cursor.saturating_sub(1));
            node.aggregate_content_pointers = vec![HydratedTextPointer {
                source_cluster_id: pointer.source_cluster_id.clone(),
                start_char: pointer.start_char,
                end_char: aggregate_end as u64,
                verbatim_text: page_text
                    .chars()
                    .skip(pointer.start_char as usize)
                    .take(aggregate_end - pointer.start_char as usize + 1)
                    .collect(),
            }];
        }
        if let Some(leaf) = heading_leaf {
            if !child_nodes.is_empty() {
                node.child_nodes.push(leaf);
            } else {
                node.total_content_pointers = vec![pointer];
                if let Value::Object(metadata) = &mut node.metadata {
                    metadata.insert(
                        "page_index_role".to_string(),
                        Value::String("heading_leaf".to_string()),
                    );
                    metadata.insert("atomic_retained".to_string(), Value::Bool(true));
                }
            }
        }
        node.child_nodes.extend(child_nodes);
        node = node.with_stable_id(Some(parent_id));
        nodes.push(node);
    }
    Ok((nodes, cursor))
}

pub fn materialize_block_tree(
    block_specs: &[PageIndexBlockSpec],
    page_text: &str,
    unit_id: &str,
    parent_id: &str,
    level_from_root: u64,
    start_at: usize,
) -> anyhow::Result<(Vec<SemanticNode>, usize)> {
    materialize_block_tree_inner(
        block_specs,
        page_text,
        unit_id,
        parent_id,
        level_from_root,
        start_at,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::page_index_contracts::{PageIndexNodeType, PageIndexSourceRole};

    #[test]
    fn resolves_exact_and_whitespace_normalized_pointers() {
        let pointer = resolve_pointer("unit", "alpha\n\tbeta", "alpha beta", 0).unwrap();
        assert_eq!(pointer.start_char, 0);
        assert_eq!(pointer.end_char, 10);
        assert_eq!(pointer.verbatim_text, "alpha\n\tbeta");
    }

    #[test]
    fn materializes_heading_and_paragraph_block_specs() {
        let block_specs = vec![PageIndexBlockSpec {
            title: "Section One".to_string(),
            node_type: PageIndexNodeType::Heading,
            excerpt: "Section One".to_string(),
            display_excerpt: None,
            summary: String::new(),
            source_role: PageIndexSourceRole::Heading,
            child_nodes: vec![PageIndexBlockSpec {
                title: "Paragraph one".to_string(),
                node_type: PageIndexNodeType::Paragraph,
                excerpt: "Paragraph one.".to_string(),
                display_excerpt: None,
                summary: String::new(),
                source_role: PageIndexSourceRole::Content,
                child_nodes: Vec::new(),
            }],
        }];
        let page_text = "Section One\n\nParagraph one.";
        let (nodes, cursor) =
            materialize_block_tree(&block_specs, page_text, "unit", "root", 1, 0).unwrap();
        assert_eq!(cursor, page_text.chars().count());
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].node_type, "HEADING");
        assert_eq!(nodes[0].aggregate_content_pointers.len(), 1);
        assert_eq!(nodes[0].child_nodes.len(), 2);
    }
}
