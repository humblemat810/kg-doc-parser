use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::page_index_contracts::{PageIndexBlockSpec, PageIndexNodeType};
use crate::page_index_tree::resolve_pointer;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExcerptRefinementAssignment {
    pub path_id: String,
    pub excerpt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExcerptRefinementBatch {
    pub assignments: Vec<ExcerptRefinementAssignment>,
}

#[derive(Debug, Clone)]
pub struct PageIndexRefinementEntry {
    pub path_id: String,
    pub node_type: PageIndexNodeType,
    pub title: String,
    pub excerpt: String,
}

#[derive(Debug, Clone)]
pub struct ExcerptRefinementRequest {
    pub page_number: u32,
    pub unit_id: String,
    pub page_text: String,
    pub entries: Vec<PageIndexRefinementEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalSummaryAssignment {
    pub path_id: String,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalSummaryBatch {
    pub assignments: Vec<HierarchicalSummaryAssignment>,
}

#[derive(Debug, Clone)]
pub struct HierarchicalSummaryBlock {
    pub path_id: String,
    pub title: String,
    pub node_type: PageIndexNodeType,
    pub excerpt: String,
    pub parent_path_id: Option<String>,
    pub parent_summary: String,
}

#[derive(Debug, Clone)]
pub struct HierarchicalSummaryRequest {
    pub page_number: u32,
    pub tree_depth: usize,
    pub blocks: Vec<HierarchicalSummaryBlock>,
}

pub trait PageIndexRefinementProvider: std::fmt::Debug {
    fn refine_excerpts(&self, request: &ExcerptRefinementRequest)
        -> Result<ExcerptRefinementBatch>;

    fn summarize_hierarchically(
        &self,
        request: &HierarchicalSummaryRequest,
    ) -> Result<HierarchicalSummaryBatch>;
}

type Diagnostics = Map<String, Value>;

fn normalize_excerpt(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn path_id(path: &[usize]) -> String {
    path.iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(".")
}

fn indexed_specs(
    specs: &[PageIndexBlockSpec],
    path: &[usize],
) -> Vec<(Vec<usize>, PageIndexBlockSpec)> {
    let mut result = Vec::new();
    for (index, spec) in specs.iter().enumerate() {
        let mut current_path = path.to_vec();
        current_path.push(index);
        result.push((current_path.clone(), spec.clone()));
        result.extend(indexed_specs(&spec.child_nodes, &current_path));
    }
    result
}

fn spec_at_path_mut<'a>(
    specs: &'a mut [PageIndexBlockSpec],
    path: &[usize],
) -> Option<&'a mut PageIndexBlockSpec> {
    let mut nodes = specs;
    for (position, index) in path.iter().enumerate() {
        let node = nodes.get_mut(*index)?;
        if position + 1 == path.len() {
            return Some(node);
        }
        nodes = node.child_nodes.as_mut_slice();
    }
    None
}

fn diagnostic_counter(diagnostics: &mut Diagnostics, key: &str) -> i64 {
    diagnostics
        .get(key)
        .and_then(Value::as_i64)
        .unwrap_or_default()
}

fn increment(diagnostics: &mut Diagnostics, key: &str, amount: i64) {
    let value = diagnostic_counter(diagnostics, key) + amount;
    diagnostics.insert(key.to_string(), json!(value));
}

fn refine_diagnostic(attempted: usize) -> Diagnostics {
    let mut diagnostics = Map::new();
    diagnostics.insert("refine_excerpts_enabled".to_string(), json!(true));
    diagnostics.insert("refine_excerpts_attempted".to_string(), json!(attempted));
    diagnostics.insert("refine_excerpts_accepted".to_string(), json!(0));
    diagnostics.insert("refine_excerpts_rejected".to_string(), json!(0));
    diagnostics.insert("refine_excerpts_fallback".to_string(), json!(false));
    diagnostics
}

fn ancestor_excerpts(specs: &[PageIndexBlockSpec], path: &[usize]) -> Vec<String> {
    let mut current = specs;
    let mut ancestors = Vec::new();
    for index in &path[..path.len().saturating_sub(1)] {
        let Some(spec) = current.get(*index) else {
            break;
        };
        let normalized = normalize_excerpt(&spec.excerpt);
        if !normalized.is_empty() {
            ancestors.push(normalized);
        }
        current = spec.child_nodes.as_slice();
    }
    ancestors
}

fn parent_nodes<'a>(
    specs: &'a mut [PageIndexBlockSpec],
    path: &[usize],
) -> &'a mut [PageIndexBlockSpec] {
    if path.len() == 1 {
        return specs;
    }
    let mut nodes = specs;
    for index in &path[..path.len() - 1] {
        nodes = nodes[*index].child_nodes.as_mut_slice();
    }
    nodes
}

fn excerpt_is_valid(
    specs: &mut [PageIndexBlockSpec],
    path: &[usize],
    original_spec: &PageIndexBlockSpec,
    proposed: &str,
    page_text: &str,
    unit_id: &str,
) -> bool {
    if proposed.trim().is_empty()
        || normalize_excerpt(proposed) == normalize_excerpt(&original_spec.excerpt)
        || normalize_excerpt(proposed) == normalize_excerpt(page_text)
    {
        return false;
    }
    let normalized_proposed = normalize_excerpt(proposed);
    if original_spec.child_nodes.is_empty()
        && normalize_excerpt(&original_spec.excerpt).contains(&normalized_proposed)
    {
        return false;
    }
    if resolve_pointer(unit_id, page_text, proposed, 0).is_err() {
        return false;
    }
    let ancestors = ancestor_excerpts(specs, path);
    if ancestors.contains(&normalized_proposed) {
        return false;
    }
    let parent = parent_nodes(specs, path);
    let siblings = parent
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != path[path.len() - 1])
        .map(|(_, sibling)| normalize_excerpt(&sibling.excerpt))
        .filter(|excerpt| !excerpt.is_empty())
        .collect::<Vec<_>>();
    if siblings.contains(&normalized_proposed) {
        let occurrences = normalize_excerpt(page_text)
            .matches(&normalized_proposed)
            .count();
        if occurrences < 2 {
            return false;
        }
    }
    true
}

pub fn refine_page_index_block_excerpts<P: PageIndexRefinementProvider>(
    block_specs: &[PageIndexBlockSpec],
    page_text: &str,
    page_number: u32,
    unit_id: &str,
    provider: &P,
) -> (Vec<PageIndexBlockSpec>, Diagnostics) {
    let entries = indexed_specs(block_specs, &[])
        .into_iter()
        .map(|(path, spec)| PageIndexRefinementEntry {
            path_id: path_id(&path),
            node_type: spec.node_type,
            title: spec.title.clone(),
            excerpt: spec.excerpt,
        })
        .collect::<Vec<_>>();
    let mut refined = block_specs.to_vec();
    let mut diagnostics = refine_diagnostic(entries.len());
    if entries.is_empty() {
        return (refined, diagnostics);
    }
    let request = ExcerptRefinementRequest {
        page_number,
        unit_id: unit_id.to_string(),
        page_text: page_text.to_string(),
        entries,
    };
    let batch = match provider.refine_excerpts(&request) {
        Ok(batch) => {
            diagnostics.insert("refine_excerpts_fallback".to_string(), json!(false));
            batch
        }
        Err(_) => {
            diagnostics.insert("refine_excerpts_fallback".to_string(), json!(true));
            return (refined, diagnostics);
        }
    };
    let original_specs = block_specs.to_vec();
    let suggestions = batch
        .assignments
        .into_iter()
        .map(|assignment| (assignment.path_id, assignment.excerpt))
        .collect::<Vec<_>>();
    for (assignment_path_id, proposed) in suggestions {
        let Some(path) = indexed_specs(&original_specs, &[])
            .into_iter()
            .find_map(|(path, _)| (path_id(&path) == assignment_path_id).then_some(path))
        else {
            increment(&mut diagnostics, "refine_excerpts_rejected", 1);
            continue;
        };
        let Some(original_spec) =
            indexed_specs(&original_specs, &[])
                .into_iter()
                .find_map(|(candidate_path, spec)| {
                    (path_id(&candidate_path) == assignment_path_id).then_some(spec)
                })
        else {
            increment(&mut diagnostics, "refine_excerpts_rejected", 1);
            continue;
        };
        if !excerpt_is_valid(
            &mut refined,
            &path,
            &original_spec,
            &proposed,
            page_text,
            unit_id,
        ) {
            increment(&mut diagnostics, "refine_excerpts_rejected", 1);
            continue;
        }
        let Some(target) = spec_at_path_mut(refined.as_mut_slice(), &path) else {
            increment(&mut diagnostics, "refine_excerpts_rejected", 1);
            continue;
        };
        target.excerpt = proposed;
        increment(&mut diagnostics, "refine_excerpts_accepted", 1);
    }
    (refined, diagnostics)
}

#[allow(clippy::items_after_test_module)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::page_index_contracts::PageIndexSourceRole;
    use serde_json::json;

    #[derive(Debug)]
    struct FixedProvider {
        refinement: Option<ExcerptRefinementBatch>,
        summary: Option<HierarchicalSummaryBatch>,
        failure: bool,
    }

    impl FixedProvider {
        fn refinement_only(assignments: &[(&str, &str)]) -> Self {
            Self {
                refinement: Some(ExcerptRefinementBatch {
                    assignments: assignments
                        .iter()
                        .map(|(path_id, excerpt)| ExcerptRefinementAssignment {
                            path_id: (*path_id).to_string(),
                            excerpt: (*excerpt).to_string(),
                        })
                        .collect(),
                }),
                summary: None,
                failure: false,
            }
        }

        fn summaries(assignments: &[(&str, &str)]) -> Self {
            Self {
                refinement: None,
                summary: Some(HierarchicalSummaryBatch {
                    assignments: assignments
                        .iter()
                        .map(|(path_id, summary)| HierarchicalSummaryAssignment {
                            path_id: (*path_id).to_string(),
                            summary: (*summary).to_string(),
                        })
                        .collect(),
                }),
                failure: false,
            }
        }
    }

    impl PageIndexRefinementProvider for FixedProvider {
        fn refine_excerpts(
            &self,
            _request: &ExcerptRefinementRequest,
        ) -> Result<ExcerptRefinementBatch> {
            if self.failure {
                anyhow::bail!("provider unavailable");
            }
            Ok(self.refinement.clone().unwrap_or(ExcerptRefinementBatch {
                assignments: Vec::new(),
            }))
        }

        fn summarize_hierarchically(
            &self,
            _request: &HierarchicalSummaryRequest,
        ) -> Result<HierarchicalSummaryBatch> {
            if self.failure {
                anyhow::bail!("provider unavailable");
            }
            Ok(self.summary.clone().unwrap_or(HierarchicalSummaryBatch {
                assignments: Vec::new(),
            }))
        }
    }

    fn sample_tree() -> Vec<PageIndexBlockSpec> {
        let child = PageIndexBlockSpec {
            title: "Paragraph".to_string(),
            node_type: PageIndexNodeType::Paragraph,
            excerpt: "Alpha intro".to_string(),
            display_excerpt: None,
            summary: String::new(),
            source_role: PageIndexSourceRole::Content,
            child_nodes: Vec::new(),
        };
        let root = PageIndexBlockSpec {
            title: "Document".to_string(),
            node_type: PageIndexNodeType::Heading,
            excerpt: "Alpha intro beta".to_string(),
            display_excerpt: None,
            summary: String::new(),
            source_role: PageIndexSourceRole::Heading,
            child_nodes: vec![child],
        };
        vec![root]
    }

    #[test]
    fn accepts_only_grounded_non_duplicate_refinement() {
        let specs = sample_tree();
        let provider = FixedProvider::refinement_only(&[("0.0", "intro beta")]);
        let (refined, diagnostics) =
            refine_page_index_block_excerpts(&specs, "Alpha intro beta", 1, "unit", &provider);
        assert_eq!(refined[0].child_nodes[0].excerpt, "intro beta");
        assert_eq!(diagnostics["refine_excerpts_accepted"], json!(1));
        assert_eq!(diagnostics["refine_excerpts_rejected"], json!(0));
    }

    #[test]
    fn rejects_leaf_substring_and_provider_failure_falls_back() {
        let specs = sample_tree();
        let provider = FixedProvider::refinement_only(&[("0.0", "Alpha")]);
        let (refined, diagnostics) =
            refine_page_index_block_excerpts(&specs, "Alpha intro beta", 1, "unit", &provider);
        assert_eq!(refined[0].child_nodes[0].excerpt, "Alpha intro");
        assert_eq!(diagnostics["refine_excerpts_rejected"], json!(1));

        let failing = FixedProvider {
            refinement: None,
            summary: None,
            failure: true,
        };
        let (refined, diagnostics) =
            refine_page_index_block_excerpts(&specs, "Alpha intro beta", 1, "unit", &failing);
        assert_eq!(refined[0].child_nodes[0].excerpt, "Alpha intro");
        assert_eq!(diagnostics["refine_excerpts_fallback"], json!(true));
    }

    #[test]
    fn applies_valid_hierarchical_summaries_in_order() {
        let specs = sample_tree();
        let provider = FixedProvider::summaries(&[
            ("0.0", "Alpha summary"),
            ("unknown", "ignored"),
            ("0.0", "   "),
        ]);
        let (refined, diagnostics) =
            refine_page_index_summaries_hierarchically(&specs, 1, &provider);
        assert_eq!(refined[0].child_nodes[0].summary, "Alpha summary");
        assert_eq!(diagnostics["hierarchical_summary_accepted"], json!(1));
        assert_eq!(diagnostics["hierarchical_summary_rejected"], json!(5));
        assert_eq!(diagnostics["hierarchical_summary_call_count"], json!(2));
    }
}

fn summary_diagnostic(attempted: usize) -> Diagnostics {
    let mut diagnostics = Map::new();
    diagnostics.insert("hierarchical_summary_enabled".to_string(), json!(true));
    diagnostics.insert(
        "hierarchical_summary_attempted".to_string(),
        json!(attempted),
    );
    diagnostics.insert("hierarchical_summary_accepted".to_string(), json!(0));
    diagnostics.insert("hierarchical_summary_rejected".to_string(), json!(0));
    diagnostics.insert("hierarchical_summary_fallback".to_string(), json!(false));
    diagnostics.insert("hierarchical_summary_call_count".to_string(), json!(0));
    diagnostics
}

pub fn refine_page_index_summaries_hierarchically<P: PageIndexRefinementProvider>(
    block_specs: &[PageIndexBlockSpec],
    page_number: u32,
    provider: &P,
) -> (Vec<PageIndexBlockSpec>, Diagnostics) {
    let paths = indexed_specs(block_specs, &[]);
    let mut refined = block_specs.to_vec();
    let mut diagnostics = summary_diagnostic(paths.len());
    if paths.is_empty() {
        return (refined, diagnostics);
    }
    let mut summaries = paths
        .iter()
        .map(|(path, spec)| (path.clone(), spec.summary.trim().to_string()))
        .collect::<Vec<_>>();
    let mut max_depth = 0usize;
    for (path, _) in &paths {
        max_depth = max_depth.max(path.len());
    }
    let mut call_count = 0usize;
    for depth in 1..=max_depth {
        let depth_paths = paths
            .iter()
            .filter(|(path, _)| path.len() == depth)
            .cloned()
            .collect::<Vec<_>>();
        for chunk in depth_paths.chunks(32) {
            call_count += 1;
            diagnostics.insert(
                "hierarchical_summary_call_count".to_string(),
                json!(call_count),
            );
            let blocks = chunk
                .iter()
                .map(|(path, spec)| {
                    let parent_summary = summaries
                        .iter()
                        .find(|(candidate_path, _)| {
                            candidate_path.as_slice() == &path[..path.len() - 1]
                        })
                        .map(|(_, summary)| summary.clone())
                        .unwrap_or_default();
                    HierarchicalSummaryBlock {
                        path_id: path_id(path),
                        title: spec.title.chars().take(240).collect(),
                        node_type: spec.node_type,
                        excerpt: spec.excerpt.chars().take(1600).collect(),
                        parent_path_id: if path.len() > 1 {
                            Some(path_id(&path[..path.len() - 1]))
                        } else {
                            None
                        },
                        parent_summary: parent_summary.chars().take(800).collect(),
                    }
                })
                .collect::<Vec<_>>();
            let request = HierarchicalSummaryRequest {
                page_number,
                tree_depth: depth,
                blocks,
            };
            let Ok(batch) = provider.summarize_hierarchically(&request) else {
                diagnostics.insert("hierarchical_summary_fallback".to_string(), json!(true));
                increment(
                    &mut diagnostics,
                    "hierarchical_summary_rejected",
                    chunk.len() as i64,
                );
                continue;
            };
            for assignment in batch.assignments {
                let Some((path, _)) = chunk
                    .iter()
                    .find(|(path, _)| path_id(path) == assignment.path_id)
                else {
                    increment(&mut diagnostics, "hierarchical_summary_rejected", 1);
                    continue;
                };
                let summary = assignment.summary.trim().to_string();
                if summary.is_empty() {
                    increment(&mut diagnostics, "hierarchical_summary_rejected", 1);
                    continue;
                }
                let Some(target) = spec_at_path_mut(refined.as_mut_slice(), path) else {
                    increment(&mut diagnostics, "hierarchical_summary_rejected", 1);
                    continue;
                };
                target.summary = summary.clone();
                if let Some(recorded) = summaries
                    .iter_mut()
                    .find(|(candidate_path, _)| candidate_path.as_slice() == path.as_slice())
                {
                    recorded.1 = summary;
                }
                increment(&mut diagnostics, "hierarchical_summary_accepted", 1);
            }
        }
    }
    (refined, diagnostics)
}
