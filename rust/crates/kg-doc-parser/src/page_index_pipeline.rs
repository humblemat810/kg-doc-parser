use crate::page_index_candidates::{
    candidate_heading_evidence, deterministic_block_assignments, extract_candidate_blocks,
};
use crate::page_index_contracts::{
    BlockAssignment, CandidateBlock, PageIndexBlockSpec, PageIndexNodeType, PageIndexSourceFormat,
    PageIndexSourceRole,
};
use crate::page_index_source_text::split_pages;
use crate::page_index_tree::{make_semantic_node, materialize_block_tree};
use crate::semantic::{HydratedTextPointer, SemanticNode};
use serde_json::{json, Value};

pub fn repair_block_assignments_for_assembly(
    candidates: &[CandidateBlock],
    assignments: &[BlockAssignment],
) -> Vec<BlockAssignment> {
    let candidate_ids = candidates
        .iter()
        .map(|candidate| candidate.block_id.clone())
        .collect::<std::collections::HashSet<_>>();
    let index_by_id = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| (candidate.block_id.clone(), index))
        .collect::<std::collections::HashMap<_, _>>();
    assignments
        .iter()
        .map(|assignment| {
            let parent_id = assignment
                .parent_id
                .as_ref()
                .filter(|parent_id| candidate_ids.contains(*parent_id))
                .and_then(|parent_id| {
                    let parent_index = index_by_id.get(parent_id).copied();
                    let child_index = index_by_id.get(&assignment.block_id).copied();
                    match (parent_index, child_index) {
                        (Some(parent_index), Some(child_index)) if parent_index < child_index => {
                            Some(parent_id.clone())
                        }
                        _ => None,
                    }
                });
            BlockAssignment {
                block_id: assignment.block_id.clone(),
                parent_id,
                node_type: assignment.node_type,
                title: assignment.title.clone(),
                summary: assignment.summary.clone(),
            }
        })
        .collect()
}

pub fn assemble_page_index_blocks(
    candidates: &[CandidateBlock],
    assignments: &[BlockAssignment],
) -> Vec<PageIndexBlockSpec> {
    let candidate_by_id = candidates
        .iter()
        .map(|candidate| (candidate.block_id.clone(), candidate))
        .collect::<std::collections::HashMap<_, _>>();
    let repaired_assignments = repair_block_assignments_for_assembly(candidates, assignments);
    let mut spec_by_id = std::collections::HashMap::new();
    for assignment in &repaired_assignments {
        let Some(candidate) = candidate_by_id.get(&assignment.block_id) else {
            continue;
        };
        let heading = candidate_heading_evidence(candidate);
        spec_by_id.insert(
            assignment.block_id.clone(),
            PageIndexBlockSpec {
                title: if assignment.title.is_empty() {
                    candidate.title_hint.clone()
                } else {
                    assignment.title.clone()
                },
                node_type: if heading {
                    PageIndexNodeType::Heading
                } else {
                    assignment.node_type
                },
                excerpt: candidate.text.clone(),
                display_excerpt: None,
                summary: assignment.summary.trim().to_string(),
                source_role: if heading {
                    PageIndexSourceRole::Heading
                } else {
                    PageIndexSourceRole::Content
                },
                child_nodes: Vec::new(),
            },
        );
    }
    for assignment in &repaired_assignments {
        let Some(spec) = spec_by_id.get(&assignment.block_id).cloned() else {
            continue;
        };
        if let Some(parent_id) = &assignment.parent_id {
            if let Some(parent_spec) = spec_by_id.get_mut(parent_id) {
                parent_spec.child_nodes.push(spec);
                continue;
            }
        }
    }
    let roots = repaired_assignments
        .iter()
        .filter(|assignment| assignment.parent_id.is_none())
        .filter_map(|assignment| spec_by_id.remove(&assignment.block_id))
        .collect::<Vec<_>>();
    roots
}

pub fn heuristic_page_outline(
    page_text: &str,
    page_number: u32,
    source_format: PageIndexSourceFormat,
) -> (Vec<PageIndexBlockSpec>, Value) {
    let candidates = extract_candidate_blocks(page_text, page_number, source_format);
    let assignments = deterministic_block_assignments(&candidates);
    let block_specs = assemble_page_index_blocks(&candidates, &assignments);
    let diagnostics = json!({
        "assignment_mode": "heuristic_deterministic",
        "candidate_count": candidates.len(),
        "assignment_count": assignments.len(),
        "validation_errors": [],
        "validation_warnings": [],
        "fallback_reason": Value::Null,
    });
    (block_specs, diagnostics)
}

#[derive(Debug, Clone)]
pub struct PageIndexParseResult {
    pub semantic_tree: SemanticNode,
    pub diagnostics: Value,
}

pub fn parse_page_index_document_heuristic(
    document_id: &str,
    title: &str,
    raw_text: &str,
    source_format: PageIndexSourceFormat,
) -> anyhow::Result<PageIndexParseResult> {
    let pages = split_pages(raw_text);
    let mut page_nodes = Vec::new();
    let mut page_diagnostics = Vec::new();
    for (page_index, page_text) in pages.iter().enumerate() {
        let page_number = page_index as u32 + 1;
        let (block_specs, diagnostics) =
            heuristic_page_outline(page_text, page_number, source_format);
        let root_pointer = HydratedTextPointer {
            source_cluster_id: format!("{document_id}:page:{page_number}"),
            start_char: 0,
            end_char: page_text.chars().count().saturating_sub(1) as u64,
            verbatim_text: page_text.clone(),
        };
        let mut page_node = make_semantic_node(
            &format!("Page {page_number}"),
            "PAGE",
            Some(document_id),
            1,
            vec![root_pointer],
            "",
            Some(json!({"page_number": page_number})),
        );
        let (child_nodes, _) = materialize_block_tree(
            &block_specs,
            page_text,
            &format!("{document_id}:page:{page_number}"),
            document_id,
            2,
            0,
        )?;
        page_node.child_nodes.extend(child_nodes);
        page_node = page_node.with_stable_id(Some(document_id));
        page_nodes.push(page_node);
        page_diagnostics.push(diagnostics);
    }
    let root = make_semantic_node(
        title,
        "DOCUMENT_ROOT",
        None,
        0,
        Vec::new(),
        "",
        Some(json!({"parse_strategy": "page_index"})),
    )
    .with_stable_id(None);
    let semantic_tree = SemanticNode {
        child_nodes: page_nodes,
        ..root
    };
    let diagnostics = json!({
        "assignment_mode": "heuristic_deterministic",
        "final_outcome": "first_pass_success",
        "page_count": page_diagnostics.len(),
        "page_diagnostics": page_diagnostics,
    });
    Ok(PageIndexParseResult {
        semantic_tree,
        diagnostics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assembles_assignments_into_nested_block_specs() {
        let page_text = "# Section One\n\nParagraph one.";
        let candidates = extract_candidate_blocks(page_text, 1, PageIndexSourceFormat::Markdown);
        dbg!(&candidates);
        let assignments = deterministic_block_assignments(&candidates);
        dbg!(&assignments);
        let block_specs = assemble_page_index_blocks(&candidates, &assignments);
        dbg!(&block_specs);
        assert_eq!(block_specs.len(), 1);
        assert_eq!(block_specs[0].title, "Section One");
        assert_eq!(block_specs[0].child_nodes.len(), 1);
    }

    #[test]
    fn parses_heuristic_page_index_end_to_end() {
        let raw_text = "# Section One\n\nParagraph one.";
        let result = parse_page_index_document_heuristic(
            "doc",
            "Document",
            raw_text,
            PageIndexSourceFormat::Markdown,
        )
        .unwrap();
        assert_eq!(result.semantic_tree.node_type, "DOCUMENT_ROOT");
        assert_eq!(result.semantic_tree.child_nodes.len(), 1);
        assert_eq!(result.semantic_tree.child_nodes[0].node_type, "PAGE");
        assert_eq!(
            result.semantic_tree.child_nodes[0].child_nodes[0].node_type,
            "HEADING"
        );
    }
}
