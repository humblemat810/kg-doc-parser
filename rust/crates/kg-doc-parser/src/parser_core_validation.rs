use anyhow::Result;
use serde::Serialize;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashSet};

use crate::parser_core::{CurrentLayerContext, CurrentLayerResult, LayerChildCandidate};
use crate::parser_core_diagnostics::{child_pointer_fingerprint, normalize_text};
use crate::semantic::HydratedTextPointer;
use crate::semantic_graph_import::pointer_source_validation_error;
use crate::source_text::source_map_entry_text;

#[derive(Debug, Clone, Serialize)]
pub struct LayerSpanConflict {
    pub parent_node_id: String,
    pub left_child_id: String,
    pub right_child_id: String,
    pub source_cluster_id: String,
    pub left_span: HydratedTextPointer,
    pub right_span: HydratedTextPointer,
    pub overlap_start: i64,
    pub overlap_end: i64,
    pub conflict_kind: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct LayerCoverageGap {
    pub parent_node_id: String,
    pub source_cluster_id: String,
    pub gap_start: i64,
    pub gap_end: i64,
    pub expected_text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct LayerDuplicateChildNote {
    pub parent_node_id: String,
    pub child_node_id: String,
    pub duplicate_of_child_node_id: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct LayerInvariants {
    pub coverage_ok: bool,
    pub satisfied: bool,
    pub overlap_conflicts: Vec<LayerSpanConflict>,
    pub coverage_gaps: Vec<LayerCoverageGap>,
    pub duplicate_notes: Vec<LayerDuplicateChildNote>,
    pub review_notes: Vec<String>,
}

fn pointer_end(pointer: &HydratedTextPointer, source_map: Option<&Value>) -> i64 {
    if pointer.end_char != -1 {
        return pointer.end_char;
    }
    if let Some(source_map) = source_map {
        let text = source_map
            .get(&pointer.source_cluster_id)
            .map(source_map_entry_text)
            .unwrap_or_default();
        if !text.is_empty() {
            return (text.chars().count() as i64 - 1).max(0);
        }
    }
    pointer.start_char
}

fn merge_intervals(mut intervals: Vec<(i64, i64)>) -> Vec<(i64, i64)> {
    if intervals.is_empty() {
        return Vec::new();
    }
    intervals.sort_unstable();
    let mut merged = Vec::with_capacity(intervals.len());
    let (mut start, mut end) = intervals[0];
    for (next_start, next_end) in intervals.into_iter().skip(1) {
        if next_start <= end + 1 {
            end = end.max(next_end);
        } else {
            merged.push((start, end));
            start = next_start;
            end = next_end;
        }
    }
    merged.push((start, end));
    merged
}

fn meaningful(text: &str) -> bool {
    !text.chars().all(char::is_whitespace)
}

fn slice_chars(text: &str, start: i64, end: i64) -> String {
    text.chars()
        .skip(start.max(0) as usize)
        .take((end - start + 1).max(0) as usize)
        .collect()
}

fn gap(
    parent_id: &str,
    cluster_id: &str,
    start: i64,
    end: i64,
    expected_text: String,
) -> LayerCoverageGap {
    LayerCoverageGap {
        parent_node_id: parent_id.to_string(),
        source_cluster_id: cluster_id.to_string(),
        gap_start: start,
        gap_end: end,
        expected_text,
    }
}

fn typed_parent_pointers(
    context: &CurrentLayerContext,
) -> Result<BTreeMap<String, Vec<HydratedTextPointer>>> {
    let mut result = BTreeMap::new();
    for (parent_id, pointers) in &context.parent_content_pointers_by_id {
        let pointers = serde_json::from_value::<Vec<HydratedTextPointer>>(pointers.clone())
            .map_err(|error| anyhow::anyhow!("parent content pointers must be typed: {error}"))?;
        result.insert(parent_id.clone(), pointers);
    }
    Ok(result)
}

pub fn detect_layer_invariants(
    current_layer_context: &CurrentLayerContext,
    current_layer_result: &CurrentLayerResult,
    parser_source_map: Option<&Value>,
) -> Result<LayerInvariants> {
    let mut overlap_conflicts = Vec::new();
    let mut coverage_gaps = Vec::new();
    let mut duplicate_notes = Vec::new();
    let mut review_notes = Vec::new();
    let mut seen_overlap_pairs = HashSet::new();
    let mut seen_child_ids = BTreeMap::new();
    let mut children_by_parent: BTreeMap<String, Vec<&LayerChildCandidate>> = BTreeMap::new();
    for child in &current_layer_result.children {
        children_by_parent
            .entry(child.parent_node_id.clone())
            .or_default()
            .push(child);
        if let Some(previous_parent) = seen_child_ids.get(&child.node_id) {
            duplicate_notes.push(LayerDuplicateChildNote {
                parent_node_id: child.parent_node_id.clone(),
                child_node_id: child.node_id.clone(),
                duplicate_of_child_node_id: child.node_id.clone(),
                reason: format!(
                    "duplicate child node id; first occurrence was under parent {previous_parent}"
                ),
            });
            review_notes.push(format!(
                "duplicate child node id: {} under parents {previous_parent} and {}",
                child.node_id, child.parent_node_id
            ));
        } else {
            seen_child_ids.insert(child.node_id.clone(), child.parent_node_id.clone());
        }
    }
    let parent_pointers = typed_parent_pointers(current_layer_context)?;
    for parent_id in &current_layer_context.parent_node_ids {
        let parent_children = children_by_parent
            .get(parent_id)
            .cloned()
            .unwrap_or_default();
        if parent_children.is_empty() {
            let explicitly_atomic = current_layer_result.atomic_retained
                && (current_layer_result.atomic_parent_ids.is_empty()
                    || current_layer_result
                        .atomic_parent_ids
                        .iter()
                        .any(|atomic_parent_id| atomic_parent_id == parent_id));
            let atomic_pointers = parent_pointers.get(parent_id).cloned().unwrap_or_default();
            if explicitly_atomic {
                if atomic_pointers.is_empty() {
                    coverage_gaps.push(gap(
                        parent_id,
                        "<missing-parent-pointer>",
                        0,
                        0,
                        "atomic retention requires a verified parent pointer".to_string(),
                    ));
                    review_notes
                        .push(format!("parent {parent_id} requested atomic retention without a verified parent pointer"));
                } else if let Some(source_map) = parser_source_map {
                    for parent_pointer in atomic_pointers {
                        if let Some(pointer_error) =
                            pointer_source_validation_error(&parent_pointer, source_map)
                        {
                            coverage_gaps.push(gap(
                                parent_id,
                                &parent_pointer.source_cluster_id,
                                parent_pointer.start_char.max(0),
                                pointer_end(&parent_pointer, Some(source_map)),
                                pointer_error.clone(),
                            ));
                            review_notes.push(format!(
                                "atomic parent pointer for {parent_id} is invalid: {pointer_error}"
                            ));
                        }
                    }
                } else {
                    coverage_gaps.push(gap(
                        parent_id,
                        "<missing-source-map>",
                        0,
                        0,
                        "atomic retention requires an authoritative source map".to_string(),
                    ));
                    review_notes.push(format!(
                        "parent {parent_id} cannot be atomically retained without a source map"
                    ));
                }
                if !coverage_gaps
                    .iter()
                    .any(|item| &item.parent_node_id == parent_id)
                {
                    review_notes.push(format!(
                        "parent {parent_id} retained as an explicit atomic unit"
                    ));
                }
            } else {
                for parent_pointer in atomic_pointers {
                    let parent_end = pointer_end(&parent_pointer, parser_source_map);
                    let cluster_text = parser_source_map
                        .and_then(|source_map| source_map.get(&parent_pointer.source_cluster_id))
                        .map(source_map_entry_text)
                        .unwrap_or_default();
                    let expected_text =
                        slice_chars(&cluster_text, parent_pointer.start_char, parent_end);
                    if meaningful(&expected_text) {
                        coverage_gaps.push(gap(
                            parent_id,
                            &parent_pointer.source_cluster_id,
                            parent_pointer.start_char,
                            parent_end,
                            expected_text,
                        ));
                    }
                }
                review_notes.push(format!(
                    "parent {parent_id} returned no children without explicit atomic retention"
                ));
            }
            continue;
        }
        for (left_index, left_child) in parent_children.iter().enumerate() {
            for right_child in &parent_children[left_index + 1..] {
                for left_pointer in &left_child.total_content_pointers {
                    for right_pointer in &right_child.total_content_pointers {
                        if left_pointer.source_cluster_id != right_pointer.source_cluster_id {
                            continue;
                        }
                        let left_end = pointer_end(left_pointer, parser_source_map);
                        let right_end = pointer_end(right_pointer, parser_source_map);
                        let overlap_start = left_pointer.start_char.max(right_pointer.start_char);
                        let overlap_end = left_end.min(right_end);
                        if overlap_start > overlap_end {
                            continue;
                        }
                        let conflict_kind = if left_pointer.start_char == right_pointer.start_char
                            && left_end == right_end
                            && normalize_text(&left_pointer.verbatim_text)
                                == normalize_text(&right_pointer.verbatim_text)
                        {
                            "duplicate".to_string()
                        } else {
                            "overlap".to_string()
                        };
                        let pair_key = format!(
                            "{}|{}|{}|{}|{}|{}",
                            parent_id,
                            left_child.node_id,
                            right_child.node_id,
                            left_pointer.source_cluster_id,
                            overlap_start,
                            overlap_end,
                        );
                        if seen_overlap_pairs.insert(pair_key) {
                            overlap_conflicts.push(LayerSpanConflict {
                                parent_node_id: parent_id.clone(),
                                left_child_id: left_child.node_id.clone(),
                                right_child_id: right_child.node_id.clone(),
                                source_cluster_id: left_pointer.source_cluster_id.clone(),
                                left_span: left_pointer.clone(),
                                right_span: right_pointer.clone(),
                                overlap_start,
                                overlap_end,
                                conflict_kind: conflict_kind.clone(),
                            });
                            review_notes.push(format!(
                                "{conflict_kind} between {} and {} on {}:{overlap_start}-{overlap_end}",
                                left_child.node_id,
                                right_child.node_id,
                                left_pointer.source_cluster_id
                            ));
                        }
                    }
                }
            }
        }
        for parent_pointer in &parent_pointers.get(parent_id).cloned().unwrap_or_default() {
            let parent_pointer_error = parser_source_map
                .and_then(|source_map| pointer_source_validation_error(parent_pointer, source_map));
            if let Some(pointer_error) = parent_pointer_error {
                coverage_gaps.push(gap(
                    parent_id,
                    &parent_pointer.source_cluster_id,
                    parent_pointer.start_char.max(0),
                    pointer_end(parent_pointer, parser_source_map),
                    pointer_error.clone(),
                ));
                review_notes.push(format!(
                    "invalid parent pointer for {parent_id}: {pointer_error}"
                ));
                continue;
            }
            let parent_end = pointer_end(parent_pointer, parser_source_map);
            let parent_start = parent_pointer.start_char.max(0);
            if parent_end < parent_start {
                continue;
            }
            let mut child_intervals = Vec::new();
            for child in &parent_children {
                if child.total_content_pointers.is_empty() {
                    review_notes.push(format!(
                        "child {} under {parent_id} has no content pointer",
                        child.node_id
                    ));
                    continue;
                }
                for child_pointer in &child.total_content_pointers {
                    let pointer_error = parser_source_map.and_then(|source_map| {
                        pointer_source_validation_error(child_pointer, source_map)
                    });
                    if let Some(pointer_error) = pointer_error {
                        coverage_gaps.push(gap(
                            parent_id,
                            &child_pointer.source_cluster_id,
                            child_pointer.start_char.max(0),
                            pointer_end(child_pointer, parser_source_map),
                            pointer_error.clone(),
                        ));
                        review_notes.push(format!(
                            "invalid child pointer {} under {parent_id}: {pointer_error}",
                            child.node_id
                        ));
                        continue;
                    }
                    if child_pointer.source_cluster_id != parent_pointer.source_cluster_id {
                        continue;
                    }
                    let child_end = pointer_end(child_pointer, parser_source_map);
                    if child_pointer.start_char < parent_start || child_end > parent_end {
                        coverage_gaps.push(gap(
                            parent_id,
                            &child_pointer.source_cluster_id,
                            child_pointer.start_char.max(0),
                            child_end,
                            "child pointer is outside parent extent".to_string(),
                        ));
                        review_notes.push(format!(
                            "child pointer {} is outside parent {parent_id}",
                            child.node_id
                        ));
                        continue;
                    }
                    child_intervals.push((child_pointer.start_char.max(0), child_end));
                }
            }
            let merged = merge_intervals(child_intervals);
            let cluster_text = parser_source_map
                .and_then(|source_map| source_map.get(&parent_pointer.source_cluster_id))
                .map(source_map_entry_text)
                .unwrap_or_default();
            let mut cursor = parent_start;
            for (start, end) in merged {
                if start > cursor {
                    let gap_start = cursor;
                    let gap_end = (start - 1).min(parent_end);
                    if gap_start <= gap_end {
                        let gap_text = slice_chars(&cluster_text, gap_start, gap_end);
                        if !meaningful(&gap_text) {
                            cursor = cursor.max(gap_end + 1);
                        } else {
                            coverage_gaps.push(gap(
                                parent_id,
                                &parent_pointer.source_cluster_id,
                                gap_start,
                                gap_end,
                                gap_text,
                            ));
                            review_notes.push(format!(
                                "gap in parent {parent_id} for {}: {gap_start}-{gap_end}",
                                parent_pointer.source_cluster_id
                            ));
                        }
                    }
                }
                cursor = cursor.max(end + 1);
                if cursor > parent_end {
                    break;
                }
            }
            if cursor <= parent_end {
                let gap_text = slice_chars(&cluster_text, cursor, parent_end);
                if meaningful(&gap_text) {
                    coverage_gaps.push(gap(
                        parent_id,
                        &parent_pointer.source_cluster_id,
                        cursor,
                        parent_end,
                        gap_text,
                    ));
                    review_notes.push(format!(
                        "gap in parent {parent_id} for {}: {cursor}-{parent_end}",
                        parent_pointer.source_cluster_id
                    ));
                }
            }
        }
    }
    let coverage_ok = coverage_gaps.is_empty();
    let has_unary_nested_child = current_layer_result
        .children
        .iter()
        .any(|child| child.child_candidates.len() == 1);
    let branching_ok = current_layer_context
        .parent_node_ids
        .iter()
        .all(|parent_id| {
            children_by_parent.get(parent_id).map(Vec::len).unwrap_or(0) != 1
                && !has_unary_nested_child
        });
    let satisfied =
        coverage_ok && overlap_conflicts.is_empty() && duplicate_notes.is_empty() && branching_ok;
    Ok(LayerInvariants {
        coverage_ok,
        satisfied,
        overlap_conflicts,
        coverage_gaps,
        duplicate_notes,
        review_notes,
    })
}

#[allow(dead_code)]
fn unused_fingerprint(child: &LayerChildCandidate) -> Vec<(String, i64, i64, String)> {
    child_pointer_fingerprint(child)
}

#[allow(dead_code)]
fn unused_json() -> Map<String, Value> {
    json!({}).as_object().unwrap().clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn pointer(source: &str, start: i64, end: i64, text: &str) -> HydratedTextPointer {
        serde_json::from_value(json!({
            "source_cluster_id": source,
            "start_char": start,
            "end_char": end,
            "verbatim_text": text
        }))
        .unwrap()
    }

    fn child(
        node_id: &str,
        parent_id: &str,
        pointers: Vec<HydratedTextPointer>,
    ) -> LayerChildCandidate {
        LayerChildCandidate {
            node_id: node_id.to_string(),
            parent_node_id: parent_id.to_string(),
            title: "Title".to_string(),
            node_type: "TEXT_FLOW".to_string(),
            total_content_pointers: pointers,
            expandable: true,
            metadata: Map::new(),
            child_candidates: Vec::new(),
        }
    }

    fn context(pointers: Value) -> CurrentLayerContext {
        serde_json::from_value(json!({
            "depth": 0,
            "parent_node_ids": ["parent"],
            "parent_titles": ["Parent"],
            "parent_content_pointers_by_id": {"parent": pointers},
            "split_strategy": "excerpt_first",
            "retry_count": 0,
            "max_retries": 3,
            "metadata": {}
        }))
        .unwrap()
    }

    fn result(children: Vec<LayerChildCandidate>) -> CurrentLayerResult {
        CurrentLayerResult {
            children,
            satisfied: None,
            review_rounds: 0,
            atomic_retained: false,
            atomic_parent_ids: Vec::new(),
            atomic_content_pointers_by_parent: Map::new(),
            metadata: Map::new(),
        }
    }

    #[test]
    fn accepts_complete_nonoverlapping_coverage() {
        let context = context(json!([{
            "source_cluster_id": "p1",
            "start_char": 0,
            "end_char": 8,
            "verbatim_text": "AlphaBeta"
        }]));
        let current = result(vec![
            child("left", "parent", vec![pointer("p1", 0, 4, "Alpha")]),
            child("right", "parent", vec![pointer("p1", 5, 8, "Beta")]),
        ]);
        let invariants = detect_layer_invariants(
            &context,
            &current,
            Some(&json!({"p1": {"text": "AlphaBeta"}})),
        )
        .unwrap();
        assert!(invariants.coverage_ok);
        assert!(invariants.satisfied);
        assert!(invariants.overlap_conflicts.is_empty());
        assert!(invariants.coverage_gaps.is_empty());
    }

    #[test]
    fn detects_overlap_duplicate_and_gap() {
        let context = context(json!([{
            "source_cluster_id": "p1",
            "start_char": 0,
            "end_char": 8,
            "verbatim_text": "AlphaBeta"
        }]));
        let overlapping = result(vec![
            child("left", "parent", vec![pointer("p1", 0, 5, "AlphaB")]),
            child("right", "parent", vec![pointer("p1", 5, 8, "Beta")]),
        ]);
        let invariants = detect_layer_invariants(
            &context,
            &overlapping,
            Some(&json!({"p1": {"text": "AlphaBeta"}})),
        )
        .unwrap();
        assert!(!invariants.satisfied);
        assert_eq!(invariants.overlap_conflicts.len(), 1);
        assert_eq!(invariants.overlap_conflicts[0].conflict_kind, "overlap");

        let first_duplicate = child("left", "parent", vec![pointer("p1", 0, 4, "alpha")]);
        let second_duplicate = child("right", "parent", vec![pointer("p1", 0, 4, "alpha")]);
        let invariants = detect_layer_invariants(
            &context,
            &result(vec![first_duplicate, second_duplicate]),
            Some(&json!({"p1": {"text": "AlphaBeta"}})),
        )
        .unwrap();
        assert_eq!(invariants.overlap_conflicts[0].conflict_kind, "duplicate");

        let gapped = result(vec![
            child("left", "parent", vec![pointer("p1", 0, 2, "Alp")]),
            child("right", "parent", vec![pointer("p1", 8, 8, "a")]),
        ]);
        let invariants = detect_layer_invariants(
            &context,
            &gapped,
            Some(&json!({"p1": {"text": "AlphaBeta"}})),
        )
        .unwrap();
        assert!(!invariants.coverage_ok);
        assert_eq!(invariants.coverage_gaps.len(), 1);
        assert_eq!(invariants.coverage_gaps[0].gap_start, 3);
        assert_eq!(invariants.coverage_gaps[0].gap_end, 7);
    }

    #[test]
    fn requires_atomic_pointer_and_rejects_unary_branching() {
        let context = context(json!([{
            "source_cluster_id": "p1",
            "start_char": 0,
            "end_char": 4,
            "verbatim_text": "Alpha"
        }]));
        let mut current = result(Vec::new());
        current.atomic_retained = true;
        current.atomic_parent_ids = vec!["parent".to_string()];
        let invariants =
            detect_layer_invariants(&context, &current, Some(&json!({"p1": {"text": "Alpha"}})))
                .unwrap();
        assert!(invariants.satisfied);

        let mut current = result(Vec::new());
        current.atomic_retained = true;
        let invariants =
            detect_layer_invariants(&context, &current, Some(&json!({"p1": {"text": "Wrong"}})))
                .unwrap();
        assert!(!invariants.satisfied);

        let unary = result(vec![child(
            "left",
            "parent",
            vec![pointer("p1", 0, 4, "Alpha")],
        )]);
        let invariants = detect_layer_invariants(&context, &unary, None).unwrap();
        assert!(!invariants.satisfied);
        assert!(bail_on_missing_id(&current).is_ok());
    }

    fn bail_on_missing_id(_value: &CurrentLayerResult) -> Result<()> {
        Ok(())
    }
}
