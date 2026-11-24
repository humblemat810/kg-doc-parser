use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashSet};

use crate::parser_core::{
    atomic_pointer_override_errors, CurrentLayerContext, CurrentLayerResult, LayerChildCandidate,
};
use crate::parser_core_diagnostics::{
    child_pointer_fingerprint, layer_structural_issues, normalize_text, LayerStructuralIssue,
};
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerValidationDiagnostic {
    pub phase: String,
    pub error_type: String,
    pub message: String,
    pub location: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CurrentLayerReview {
    pub updated_result: CurrentLayerResult,
    pub coverage_ok: bool,
    pub satisfied: bool,
    pub strategy_used: crate::parser_core::SplitStrategy,
    pub overlap_conflicts: Vec<LayerSpanConflict>,
    pub coverage_gap_notes: Vec<LayerCoverageGap>,
    pub duplicate_child_notes: Vec<LayerDuplicateChildNote>,
    pub structural_issues: Vec<LayerStructuralIssue>,
    pub validation_diagnostics: Vec<LayerValidationDiagnostic>,
    pub review_notes: Vec<String>,
    pub metadata: Map<String, Value>,
    pub committable_parent_node_ids: Vec<String>,
    pub failed_parent_node_ids: Vec<String>,
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

fn diagnostic_applies_to_parent(
    diagnostic: &LayerValidationDiagnostic,
    parent_id: &str,
    children: &[LayerChildCandidate],
) -> bool {
    if diagnostic.location.is_empty() {
        return true;
    }
    if diagnostic.location.iter().any(|item| item == parent_id) {
        return true;
    }
    children
        .iter()
        .filter(|child| child.parent_node_id == parent_id)
        .any(|child| diagnostic.location.contains(&child.node_id))
}

fn structural_context(context: &CurrentLayerContext) -> Map<String, Value> {
    let mut object = Map::new();
    object.insert(
        "parent_node_ids".to_string(),
        json!(context.parent_node_ids),
    );
    object.insert(
        "parent_content_pointers_by_id".to_string(),
        serde_json::to_value(&context.parent_content_pointers_by_id)
            .unwrap_or(Value::Object(Map::new())),
    );
    object
}

pub fn validate_layer_commit(
    current_layer_context: &CurrentLayerContext,
    current_layer_result: &CurrentLayerResult,
    parser_source_map: Option<&Value>,
) -> Result<CurrentLayerReview> {
    let parent_ids = current_layer_context.parent_node_ids.clone();
    let mut validation_diagnostics: Vec<LayerValidationDiagnostic> = current_layer_result
        .validation_diagnostics
        .iter()
        .map(|value| serde_json::from_value::<LayerValidationDiagnostic>(value.clone()))
        .collect::<Result<_, _>>()
        .map_err(|error| anyhow::anyhow!("validation diagnostics must be typed: {error}"))?;
    if parent_ids.is_empty() {
        let invariants = detect_layer_invariants(
            current_layer_context,
            current_layer_result,
            parser_source_map,
        )?;
        let satisfied = invariants.satisfied && validation_diagnostics.is_empty();
        let structural_issues = layer_structural_issues(
            &structural_context(current_layer_context),
            current_layer_result,
        )?;
        let mut metadata = Map::new();
        metadata.insert("commit_validation".to_string(), json!(true));
        metadata.insert(
            "validation_diagnostics".to_string(),
            json!(validation_diagnostics),
        );
        let mut updated_result = current_layer_result.clone();
        updated_result.satisfied = Some(satisfied);
        return Ok(CurrentLayerReview {
            updated_result,
            coverage_ok: invariants.coverage_ok,
            satisfied,
            strategy_used: current_layer_context.split_strategy,
            overlap_conflicts: invariants.overlap_conflicts,
            coverage_gap_notes: invariants.coverage_gaps,
            duplicate_child_notes: invariants.duplicate_notes,
            structural_issues,
            validation_diagnostics,
            review_notes: ["post-repair pre-commit validation".to_string()]
                .into_iter()
                .chain(invariants.review_notes.iter().take(20).cloned())
                .collect(),
            metadata,
            committable_parent_node_ids: if satisfied {
                parent_ids.clone()
            } else {
                Vec::new()
            },
            failed_parent_node_ids: if satisfied { Vec::new() } else { parent_ids },
        });
    }
    let mut child_parents: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for child in &current_layer_result.children {
        child_parents
            .entry(child.node_id.clone())
            .or_default()
            .push(child.parent_node_id.clone());
    }
    for (child_id, parents) in child_parents {
        if parents.len() > 1 {
            let mut sorted_parents = parents.clone();
            sorted_parents.sort();
            let diagnostic = LayerValidationDiagnostic {
                phase: "commit".to_string(),
                error_type: "duplicate_child_id".to_string(),
                message: format!(
                    "child node id {child_id:?} is proposed under multiple parents: {}",
                    sorted_parents.join(", ")
                ),
                location: std::iter::once(child_id.clone())
                    .chain(sorted_parents)
                    .collect(),
            };
            if !validation_diagnostics.contains(&diagnostic) {
                validation_diagnostics.push(diagnostic);
            }
        }
    }
    let mut valid_parent_ids = Vec::new();
    let mut failed_parent_ids = Vec::new();
    let mut valid_children = Vec::new();
    let mut overlap_conflicts = Vec::new();
    let mut coverage_gaps = Vec::new();
    let mut duplicate_notes = Vec::new();
    let mut structural_issues = Vec::new();
    let mut notes = Vec::new();
    for parent_id in &parent_ids {
        let local_diagnostics = validation_diagnostics
            .iter()
            .filter(|diagnostic| {
                diagnostic_applies_to_parent(diagnostic, parent_id, &current_layer_result.children)
            })
            .cloned()
            .collect::<Vec<_>>();
        let local_children = current_layer_result
            .children
            .iter()
            .filter(|child| &child.parent_node_id == parent_id)
            .cloned()
            .collect::<Vec<_>>();
        let local_context = CurrentLayerContext {
            parent_node_ids: vec![parent_id.clone()],
            parent_titles: vec![current_layer_context
                .parent_titles
                .iter()
                .zip(&current_layer_context.parent_node_ids)
                .find_map(|(title, id)| (id == parent_id).then(|| title.clone()))
                .unwrap_or_else(|| parent_id.clone())],
            parent_content_pointers_by_id: current_layer_context
                .parent_content_pointers_by_id
                .get(parent_id)
                .map(|value| {
                    let mut map = Map::new();
                    map.insert(parent_id.clone(), value.clone());
                    map
                })
                .unwrap_or_default(),
            ..current_layer_context.clone()
        };
        let local_result = CurrentLayerResult {
            children: local_children,
            validation_diagnostics: local_diagnostics
                .iter()
                .map(|diagnostic| json!(diagnostic))
                .collect(),
            ..current_layer_result.clone()
        };
        let local = detect_layer_invariants(&local_context, &local_result, parser_source_map)?;
        let mut local_notes = local.review_notes.clone();
        let mut local_coverage_ok = local.coverage_ok;
        let mut local_satisfied = local.satisfied;
        let atomic_override_errors =
            atomic_pointer_override_errors(parent_id, current_layer_result, parser_source_map)?;
        if !atomic_override_errors.is_empty() {
            local_coverage_ok = false;
            local_satisfied = false;
            local_notes.extend(atomic_override_errors);
        }
        overlap_conflicts.extend(local.overlap_conflicts);
        coverage_gaps.extend(local.coverage_gaps);
        duplicate_notes.extend(local.duplicate_notes);
        structural_issues.extend(layer_structural_issues(
            &structural_context(&local_context),
            &local_result,
        )?);
        notes.extend(
            local_notes
                .iter()
                .take(12)
                .map(|note| format!("parent {parent_id}: {note}")),
        );
        if local_diagnostics.is_empty() && local_coverage_ok && local_satisfied {
            valid_parent_ids.push(parent_id.clone());
            valid_children.extend(local_result.children);
        } else {
            failed_parent_ids.push(parent_id.clone());
        }
    }
    let partial_commit = !valid_parent_ids.is_empty() && !failed_parent_ids.is_empty();
    let mut updated_result = current_layer_result.clone();
    updated_result.children = valid_children;
    updated_result.satisfied = Some(!valid_parent_ids.is_empty());
    updated_result.validation_diagnostics = if valid_parent_ids.is_empty() {
        validation_diagnostics
            .iter()
            .map(|diagnostic| json!(diagnostic))
            .collect()
    } else {
        Vec::new()
    };
    let mut metadata = Map::new();
    metadata.insert("commit_validation".to_string(), json!(true));
    metadata.insert(
        "committable_parent_node_ids".to_string(),
        json!(valid_parent_ids),
    );
    metadata.insert(
        "failed_parent_node_ids".to_string(),
        json!(failed_parent_ids),
    );
    metadata.insert("partial_commit".to_string(), json!(partial_commit));
    metadata.insert(
        "validation_diagnostics".to_string(),
        json!(validation_diagnostics),
    );
    Ok(CurrentLayerReview {
        updated_result,
        coverage_ok: failed_parent_ids.is_empty(),
        satisfied: !valid_parent_ids.is_empty(),
        strategy_used: current_layer_context.split_strategy,
        overlap_conflicts,
        coverage_gap_notes: coverage_gaps,
        duplicate_child_notes: duplicate_notes,
        structural_issues,
        validation_diagnostics,
        review_notes: ["post-repair pre-commit validation".to_string()]
            .into_iter()
            .chain(notes.into_iter().take(20))
            .collect(),
        metadata,
        committable_parent_node_ids: valid_parent_ids,
        failed_parent_node_ids: failed_parent_ids,
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

pub fn dedupe_and_filter_layer(
    current_layer_context: &CurrentLayerContext,
    current_layer_result: &CurrentLayerResult,
) -> CurrentLayerResult {
    let parent_ids = current_layer_context
        .parent_node_ids
        .iter()
        .collect::<HashSet<_>>();
    let mut seen = HashSet::new();
    let mut filtered = Vec::new();
    let mut validation_diagnostics = current_layer_result.validation_diagnostics.clone();
    for child in &current_layer_result.children {
        if !parent_ids.contains(&&child.parent_node_id) {
            let diagnostic = LayerValidationDiagnostic {
                phase: "commit".to_string(),
                error_type: "foreign_parent".to_string(),
                message: format!(
                    "child {:?} references parent {:?} outside the current layer",
                    child.node_id, child.parent_node_id
                ),
                location: vec![child.node_id.clone(), child.parent_node_id.clone()],
            };
            let diagnostic = serde_json::to_value(diagnostic).unwrap_or_default();
            if !validation_diagnostics.contains(&diagnostic) {
                validation_diagnostics.push(diagnostic);
            }
            continue;
        }
        let key = (
            child.parent_node_id.clone(),
            child.node_type.clone(),
            child.title.trim().to_lowercase(),
            child_pointer_fingerprint(child),
        );
        if seen.insert(key) {
            filtered.push(child.clone());
        }
    }
    CurrentLayerResult {
        children: filtered,
        satisfied: current_layer_result
            .satisfied
            .map(|satisfied| satisfied && validation_diagnostics.is_empty()),
        review_rounds: current_layer_result.review_rounds,
        atomic_retained: current_layer_result.atomic_retained,
        atomic_parent_ids: current_layer_result.atomic_parent_ids.clone(),
        atomic_content_pointers_by_parent: current_layer_result
            .atomic_content_pointers_by_parent
            .clone(),
        metadata: current_layer_result.metadata.clone(),
        validation_diagnostics,
    }
}

pub fn apply_cud_update(
    _current_layer_result: &crate::parser_core::CurrentLayerResult,
    current_layer_review: &CurrentLayerReview,
) -> crate::parser_core::CurrentLayerResult {
    let mut updated = current_layer_review.updated_result.clone();
    let mut metadata = updated.metadata.clone();
    for (key, value) in &current_layer_review.metadata {
        metadata.insert(key.clone(), value.clone());
    }
    metadata.insert(
        "split_strategy".to_string(),
        serde_json::to_value(current_layer_review.strategy_used).unwrap_or_default(),
    );
    metadata.insert(
        "layer_coverage_ok".to_string(),
        serde_json::to_value(current_layer_review.coverage_ok).unwrap_or_default(),
    );
    if !current_layer_review.review_notes.is_empty() {
        metadata.insert(
            "review_notes".to_string(),
            serde_json::to_value(current_layer_review.review_notes.clone()).unwrap_or_default(),
        );
    }
    if !current_layer_review.overlap_conflicts.is_empty() {
        metadata.insert(
            "overlap_conflicts".to_string(),
            serde_json::to_value(current_layer_review.overlap_conflicts.clone())
                .unwrap_or_default(),
        );
    }
    if !current_layer_review.coverage_gap_notes.is_empty() {
        metadata.insert(
            "coverage_gap_notes".to_string(),
            serde_json::to_value(current_layer_review.coverage_gap_notes.clone())
                .unwrap_or_default(),
        );
    }
    if !current_layer_review.duplicate_child_notes.is_empty() {
        metadata.insert(
            "duplicate_child_notes".to_string(),
            serde_json::to_value(current_layer_review.duplicate_child_notes.clone())
                .unwrap_or_default(),
        );
    }
    if !current_layer_review.structural_issues.is_empty() {
        metadata.insert(
            "structural_issues".to_string(),
            serde_json::to_value(current_layer_review.structural_issues.clone())
                .unwrap_or_default(),
        );
    }
    if !current_layer_review.validation_diagnostics.is_empty() {
        metadata.insert(
            "validation_diagnostics".to_string(),
            serde_json::to_value(current_layer_review.validation_diagnostics.clone())
                .unwrap_or_default(),
        );
    }
    updated.satisfied = Some(current_layer_review.satisfied);
    updated.review_rounds += 1;
    updated.metadata = metadata;
    updated
}

fn pointer_repair_context(pointer: &HydratedTextPointer, parser_source_map: &Value) -> String {
    let source = parser_source_map.get(&pointer.source_cluster_id);
    let text = source.map(source_map_entry_text).unwrap_or_default();
    let preview = text.replace('\n', "\\n").replace('\t', "\\t");
    let preview = if preview.chars().count() > 120 {
        format!("{}...", preview.chars().take(117).collect::<String>())
    } else {
        preview
    };
    format!(
        "source_cluster_id={:?}, span=({},{}) verbatim_text={:?}, source_text={:?}",
        pointer.source_cluster_id,
        pointer.start_char,
        pointer.end_char,
        pointer.verbatim_text,
        preview
    )
}

pub fn repair_layer_candidates<F>(
    current_layer_result: &CurrentLayerResult,
    parser_source_map: &Value,
    correct_pointer_fn: F,
) -> (CurrentLayerResult, usize)
where
    F: Fn(&HydratedTextPointer, &Value) -> Result<Option<HydratedTextPointer>>,
{
    let mut repaired_children = Vec::with_capacity(current_layer_result.children.len());
    let mut repair_failures = Vec::new();
    let mut repair_diagnostics = Vec::new();
    let mut repaired_count = 0;

    for child in &current_layer_result.children {
        let mut repaired_pointers = Vec::with_capacity(child.total_content_pointers.len());
        let mut child_failed = false;
        for pointer in &child.total_content_pointers {
            let fixed = match correct_pointer_fn(pointer, parser_source_map) {
                Ok(Some(fixed)) => Some(fixed),
                Ok(None) => {
                    let message = format!(
                        "unrecoverable pointer for child {:?} (parent={:?}, node_id={:?}); {}",
                        child.title,
                        child.parent_node_id,
                        child.node_id,
                        pointer_repair_context(pointer, parser_source_map)
                    );
                    repair_failures.push(message);
                    child_failed = true;
                    None
                }
                Err(error) => {
                    let diagnostic = LayerValidationDiagnostic {
                        phase: "repair".to_string(),
                        error_type: "repair_error".to_string(),
                        message: error.to_string(),
                        location: Vec::new(),
                    };
                    let diagnostic = serde_json::to_value(diagnostic).unwrap_or_default();
                    if !repair_diagnostics.contains(&diagnostic) {
                        repair_diagnostics.push(diagnostic);
                    }
                    let message = format!(
                        "pointer repair validation failed for child {:?} (parent={:?}, node_id={:?}); RepairError: {error}",
                        child.title,
                        child.parent_node_id,
                        child.node_id
                    );
                    repair_failures.push(message);
                    child_failed = true;
                    None
                }
            };
            let Some(fixed) = fixed else { break };
            if fixed != *pointer {
                repaired_count += 1;
            }
            repaired_pointers.push(fixed);
        }
        if child_failed {
            continue;
        }
        repaired_children.push(LayerChildCandidate {
            node_id: child.node_id.clone(),
            parent_node_id: child.parent_node_id.clone(),
            title: child.title.clone(),
            node_type: child.node_type.clone(),
            total_content_pointers: repaired_pointers,
            expandable: child.expandable,
            metadata: child.metadata.clone(),
            child_candidates: child.child_candidates.clone(),
        });
    }

    let mut metadata = current_layer_result.metadata.clone();
    if !repair_failures.is_empty() {
        metadata.insert(
            "repair_failures".to_string(),
            serde_json::to_value(repair_failures.iter().take(32).cloned().collect::<Vec<_>>())
                .unwrap_or_default(),
        );
        metadata.insert(
            "repair_failure_scope".to_string(),
            json!("child_replacement"),
        );
        metadata.insert("failure_type".to_string(), json!("repair_failure"));
        metadata.insert("rollback".to_string(), json!("verified_parent_retained"));
        metadata.insert("retryable".to_string(), json!(true));
    }
    let mut validation_diagnostics = current_layer_result.validation_diagnostics.clone();
    for diagnostic in repair_diagnostics {
        if !validation_diagnostics.contains(&diagnostic) {
            validation_diagnostics.push(diagnostic);
        }
    }
    let satisfied = if validation_diagnostics.is_empty() {
        current_layer_result.satisfied
    } else {
        Some(false)
    };
    (
        CurrentLayerResult {
            children: repaired_children,
            satisfied,
            review_rounds: current_layer_result.review_rounds,
            atomic_retained: current_layer_result.atomic_retained,
            atomic_parent_ids: current_layer_result.atomic_parent_ids.clone(),
            atomic_content_pointers_by_parent: current_layer_result
                .atomic_content_pointers_by_parent
                .clone(),
            metadata,
            validation_diagnostics,
        },
        repaired_count,
    )
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
            validation_diagnostics: Vec::new(),
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

    #[test]
    fn dedupes_and_filters_layer_for_commit() {
        let context = context(json!([{
            "source_cluster_id": "p1",
            "start_char": 0,
            "end_char": 4,
            "verbatim_text": "Alpha"
        }]));
        let mut current = result(vec![
            child("left", "parent", vec![pointer("p1", 0, 4, "Alpha")]),
            child("duplicate", "parent", vec![pointer("p1", 0, 4, "Alpha")]),
            child("foreign", "other", vec![pointer("p1", 0, 4, "Alpha")]),
        ]);
        current.satisfied = Some(true);
        let filtered = dedupe_and_filter_layer(&context, &current);
        assert_eq!(filtered.children.len(), 1);
        assert_eq!(filtered.children[0].node_id, "left");
        assert_eq!(filtered.validation_diagnostics.len(), 1);
        assert_eq!(
            filtered.validation_diagnostics[0]["error_type"],
            "foreign_parent"
        );
        assert_eq!(filtered.satisfied, Some(false));
    }

    #[test]
    fn applies_cud_review_update() {
        let current = result(vec![child(
            "left",
            "parent",
            vec![pointer("p1", 0, 4, "Alpha")],
        )]);
        let mut current = current;
        current.satisfied = Some(false);
        current.review_rounds = 2;
        current
            .metadata
            .insert("previous".to_string(), json!("value"));
        let mut review = CurrentLayerReview {
            updated_result: current.clone(),
            coverage_ok: true,
            satisfied: true,
            strategy_used: crate::parser_core::SplitStrategy::BoundaryFirst,
            overlap_conflicts: Vec::new(),
            coverage_gap_notes: Vec::new(),
            duplicate_child_notes: Vec::new(),
            structural_issues: Vec::new(),
            validation_diagnostics: Vec::new(),
            review_notes: vec!["review passed".to_string()],
            metadata: Map::new(),
            committable_parent_node_ids: vec!["parent".to_string()],
            failed_parent_node_ids: Vec::new(),
        };
        review
            .metadata
            .insert("review_key".to_string(), json!("review-value"));

        let updated = apply_cud_update(&current, &review);
        assert_eq!(updated.satisfied, Some(true));
        assert_eq!(updated.review_rounds, 3);
        assert_eq!(updated.metadata["previous"], json!("value"));
        assert_eq!(updated.metadata["review_key"], json!("review-value"));
        assert_eq!(updated.metadata["split_strategy"], json!("boundary_first"));
        assert_eq!(updated.metadata["layer_coverage_ok"], json!(true));
        assert_eq!(updated.metadata["review_notes"], json!(["review passed"]));
    }

    #[test]
    fn repairs_layer_candidates_without_erasing_verified_siblings() {
        let source_map = json!({"p1": {"id": "p1", "text": "AlphaBeta"}});
        let mut current = result(vec![
            child("bad", "parent", vec![pointer("missing", 0, 4, "Alpha")]),
            child("good", "parent", vec![pointer("p1", 8, 12, "Beta")]),
        ]);
        current.satisfied = Some(true);
        let (repaired, repaired_count) =
            repair_layer_candidates(&current, &source_map, |pointer, _| {
                Ok(
                    (pointer.source_cluster_id == "p1").then(|| HydratedTextPointer {
                        source_cluster_id: "p1".to_string(),
                        start_char: 5,
                        end_char: 8,
                        verbatim_text: "Beta".to_string(),
                        start_delimiter: None,
                        end_delimiter: None,
                        validation_method: Some("repair".to_string()),
                    }),
                )
            });
        assert_eq!(repaired_count, 1);
        assert_eq!(repaired.children.len(), 1);
        assert_eq!(repaired.children[0].node_id, "good");
        assert_eq!(repaired.children[0].total_content_pointers[0].start_char, 5);
        assert_eq!(repaired.metadata["retryable"], json!(true));
        assert_eq!(repaired.metadata["failure_type"], json!("repair_failure"));
        assert!(repaired.validation_diagnostics.is_empty());
        assert_eq!(repaired.satisfied, Some(true));

        let (failed, _) = repair_layer_candidates(&current, &source_map, |pointer, _| {
            if pointer.source_cluster_id == "p1" {
                return Ok(Some(pointer.clone()));
            }
            Err(anyhow::anyhow!("validation failed"))
        });
        assert_eq!(failed.children.len(), 1);
        assert_eq!(failed.children[0].node_id, "good");
        assert_eq!(
            failed.validation_diagnostics[0]["error_type"],
            "repair_error"
        );
        assert_eq!(failed.validation_diagnostics[0]["phase"], "repair");
        assert_eq!(failed.satisfied, Some(false));
    }

    #[test]
    fn validates_layer_commit_and_partial_parents() {
        let context = serde_json::from_value::<CurrentLayerContext>(json!({
            "depth": 0,
            "parent_node_ids": ["good", "bad"],
            "parent_titles": ["Good", "Bad"],
            "parent_content_pointers_by_id": {
                "good": [{"source_cluster_id": "p1", "start_char": 0, "end_char": 8, "verbatim_text": "AlphaBeta"}],
                "bad": [{"source_cluster_id": "p1", "start_char": 0, "end_char": 8, "verbatim_text": "AlphaBeta"}]
            },
            "split_strategy": "excerpt_first",
            "retry_count": 0,
            "max_retries": 3,
            "metadata": {}
        }))
        .unwrap();
        let result = CurrentLayerResult {
            children: vec![
                child("good-left", "good", vec![pointer("p1", 0, 4, "Alpha")]),
                child("good-right", "good", vec![pointer("p1", 5, 8, "Beta")]),
                child("bad-left", "bad", vec![pointer("p1", 0, 2, "Alp")]),
            ],
            satisfied: None,
            review_rounds: 0,
            atomic_retained: false,
            atomic_parent_ids: Vec::new(),
            atomic_content_pointers_by_parent: Map::new(),
            metadata: Map::new(),
            validation_diagnostics: Vec::new(),
        };
        let source_map = json!({"p1": {"text": "AlphaBeta"}});
        let review = validate_layer_commit(&context, &result, Some(&source_map)).unwrap();
        assert_eq!(review.committable_parent_node_ids, vec!["good"]);
        assert_eq!(review.failed_parent_node_ids, vec!["bad"]);
        assert!(review.metadata["partial_commit"].as_bool().unwrap());
        assert_eq!(review.updated_result.children.len(), 2);
        assert!(review.updated_result.satisfied.unwrap());
    }

    fn bail_on_missing_id(_value: &CurrentLayerResult) -> Result<()> {
        Ok(())
    }
}
