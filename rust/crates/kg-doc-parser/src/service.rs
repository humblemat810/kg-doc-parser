use serde_json::{json, Map, Value};

type State = Value;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkflowEdgeInfo {
    pub dst: String,
}

fn mapping(value: &Value) -> Map<String, Value> {
    value.as_object().cloned().unwrap_or_default()
}

fn string_list(value: Option<&Value>) -> Vec<String> {
    match value {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(Value::as_str)
            .map(ToString::to_string)
            .collect(),
        _ => Vec::new(),
    }
}

fn string_set(value: Option<&Value>) -> Vec<String> {
    let mut values = string_list(value);
    values.sort();
    values.dedup();
    values
}

fn metadata_mapping(value: Option<&Value>) -> Map<String, Value> {
    let raw = mapping(value.unwrap_or(&Value::Null));
    let mut metadata = Map::new();
    if let Some(parse_strategy) = raw.get("parse_strategy").and_then(Value::as_str) {
        metadata.insert("parse_strategy".to_string(), json!(parse_strategy));
    }
    let disabled_strategies = string_list(raw.get("disabled_strategies"));
    if !disabled_strategies.is_empty() {
        metadata.insert(
            "disabled_strategies".to_string(),
            json!(disabled_strategies),
        );
    }
    if let Some(commit_validation) = raw.get("commit_validation").and_then(Value::as_bool) {
        metadata.insert("commit_validation".to_string(), json!(commit_validation));
    }
    if let Some(review_failure) = raw.get("review_failure").and_then(Value::as_str) {
        metadata.insert("review_failure".to_string(), json!(review_failure));
    }
    metadata
}

fn context(state: &State) -> Map<String, Value> {
    let raw = mapping(state.get("current_layer_context").unwrap_or(&Value::Null));
    let mut context = Map::new();
    let parent_node_ids = string_list(raw.get("parent_node_ids"));
    if !parent_node_ids.is_empty() {
        context.insert("parent_node_ids".to_string(), json!(parent_node_ids));
    }
    let metadata = metadata_mapping(raw.get("metadata"));
    if !metadata.is_empty() {
        context.insert("metadata".to_string(), Value::Object(metadata));
    }
    context
}

fn state_metadata(state: &State) -> Map<String, Value> {
    metadata_mapping(context(state).get("metadata"))
}

fn review(state: &State) -> Map<String, Value> {
    let raw = mapping(state.get("current_layer_review").unwrap_or(&Value::Null));
    let mut review = Map::new();
    let metadata = metadata_mapping(raw.get("metadata"));
    if !metadata.is_empty() {
        review.insert("metadata".to_string(), Value::Object(metadata));
    }
    for field_name in ["satisfied", "coverage_ok"] {
        if let Some(value) = raw.get(field_name).and_then(Value::as_bool) {
            review.insert(field_name.to_string(), json!(value));
        }
    }
    for field_name in [
        "overlap_conflicts",
        "coverage_gap_notes",
        "duplicate_child_notes",
        "structural_issues",
        "validation_diagnostics",
    ] {
        if let Some(Value::Array(value)) = raw.get(field_name) {
            review.insert(field_name.to_string(), json!(value));
        }
    }
    review
}

fn result(state: &State) -> Map<String, Value> {
    let raw = mapping(state.get("current_layer_result").unwrap_or(&Value::Null));
    let mut result = Map::new();
    if let Some(satisfied) = raw.get("satisfied").and_then(Value::as_bool) {
        result.insert("satisfied".to_string(), json!(satisfied));
    }
    let children = children(raw.get("children"));
    if !children.is_empty() {
        result.insert("children".to_string(), json!(children));
    }
    result
}

fn children(value: Option<&Value>) -> Vec<Map<String, Value>> {
    let Some(Value::Array(items)) = value else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|child| {
            let parent_node_id = child.get("parent_node_id")?.as_str()?.to_string();
            let mut narrowed = Map::new();
            narrowed.insert("parent_node_id".to_string(), json!(parent_node_id));
            Some(narrowed)
        })
        .collect()
}

fn strategy_name(edge: &WorkflowEdgeInfo) -> String {
    let target = edge.dst.rsplit('|').next().unwrap_or_default().to_string();
    match target.as_str() {
        "layer_excerpt_method" => "layer_excerpt".to_string(),
        "layer_boundary_method" => "layer_boundary".to_string(),
        "page_index_layer" => "page_index".to_string(),
        _ => String::new(),
    }
}

fn selected_strategy(edge: &WorkflowEdgeInfo, state: &State) -> bool {
    state_metadata(state)
        .get("parse_strategy")
        .and_then(Value::as_str)
        .unwrap_or_default()
        == strategy_name(edge)
}

fn failed_with_remaining(_edge: &WorkflowEdgeInfo, state: &State) -> bool {
    let disabled = string_set(state_metadata(state).get("disabled_strategies"));
    !disabled.is_empty() && disabled.len() < 3
}

fn exhausted(_edge: &WorkflowEdgeInfo, state: &State) -> bool {
    string_set(state_metadata(state).get("disabled_strategies")).len() >= 3
}

fn strategy_selection_failed(_edge: &WorkflowEdgeInfo, state: &State) -> bool {
    state
        .get("strategy_selection_error")
        .is_some_and(|value| !value.is_null())
}

fn commit_candidates_valid(_edge: &WorkflowEdgeInfo, state: &State) -> bool {
    let current = review(state);
    metadata_mapping(current.get("metadata"))
        .get("commit_validation")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        && current.get("satisfied").and_then(Value::as_bool) == Some(true)
}

fn commit_candidates_invalid(_edge: &WorkflowEdgeInfo, state: &State) -> bool {
    let current = review(state);
    metadata_mapping(current.get("metadata"))
        .get("commit_validation")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        && current.get("satisfied").and_then(Value::as_bool) == Some(false)
}

fn batch_has_repair_candidates(_edge: &WorkflowEdgeInfo, state: &State) -> bool {
    let current_context = context(state);
    let current_review = review(state);
    let current_result = result(state);
    if metadata_mapping(current_review.get("metadata"))
        .get("review_failure")
        .and_then(Value::as_str)
        .is_some()
    {
        return false;
    }
    let parent_ids = string_set(current_context.get("parent_node_ids"));
    let current_children = children(current_result.get("children"));
    if parent_ids.len() < 2 || current_children.is_empty() {
        return false;
    }
    current_children.iter().any(|child| {
        child
            .get("parent_node_id")
            .and_then(Value::as_str)
            .is_some_and(|parent_node_id| parent_ids.contains(&parent_node_id.to_string()))
    })
}

fn satisfied(_edge: &WorkflowEdgeInfo, state: &State) -> bool {
    let current_review = review(state);
    let current_result = result(state);
    let metadata = metadata_mapping(current_review.get("metadata"));
    current_review.get("coverage_ok").and_then(Value::as_bool) == Some(true)
        && metadata
            .get("review_failure")
            .and_then(Value::as_str)
            .is_none()
        && current_result
            .get("satisfied")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        && current_review
            .get("overlap_conflicts")
            .and_then(Value::as_array)
            .unwrap_or(&Vec::new())
            .is_empty()
        && current_review
            .get("coverage_gap_notes")
            .and_then(Value::as_array)
            .unwrap_or(&Vec::new())
            .is_empty()
        && current_review
            .get("duplicate_child_notes")
            .and_then(Value::as_array)
            .unwrap_or(&Vec::new())
            .is_empty()
        && current_review
            .get("structural_issues")
            .and_then(Value::as_array)
            .unwrap_or(&Vec::new())
            .is_empty()
        && current_review
            .get("validation_diagnostics")
            .and_then(Value::as_array)
            .unwrap_or(&Vec::new())
            .is_empty()
}

pub fn workflow_predicate_names() -> Vec<&'static str> {
    vec![
        "parse_strategy_layer_excerpt",
        "parse_strategy_layer_boundary",
        "parse_strategy_page_index",
        "strategy_attempted",
        "layer_satisfied",
        "strategy_failed_with_remaining",
        "all_strategies_exhausted",
        "strategy_selection_failed",
        "commit_candidates_valid",
        "commit_candidates_invalid",
        "batch_has_repair_candidates",
    ]
}

pub fn evaluate_workflow_predicate(name: &str, edge: &WorkflowEdgeInfo, state: &State) -> bool {
    match name {
        "parse_strategy_layer_excerpt"
        | "parse_strategy_layer_boundary"
        | "parse_strategy_page_index" => selected_strategy(edge, state),
        "strategy_attempted" => true,
        "layer_satisfied" => satisfied(edge, state),
        "strategy_failed_with_remaining" => failed_with_remaining(edge, state),
        "all_strategies_exhausted" => exhausted(edge, state),
        "strategy_selection_failed" => strategy_selection_failed(edge, state),
        "commit_candidates_valid" => commit_candidates_valid(edge, state),
        "commit_candidates_invalid" => commit_candidates_invalid(edge, state),
        "batch_has_repair_candidates" => batch_has_repair_candidates(edge, state),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn routes_selected_strategy_and_exhaustion() {
        let edge = WorkflowEdgeInfo {
            dst: "workflow|layer_excerpt_method".to_string(),
        };
        let state = json!({
            "current_layer_context": {
                "metadata": {"parse_strategy": "layer_excerpt", "disabled_strategies": []}
            }
        });
        assert!(evaluate_workflow_predicate(
            "parse_strategy_layer_excerpt",
            &edge,
            &state
        ));
        let page_index_edge = WorkflowEdgeInfo {
            dst: "workflow|page_index_layer".to_string(),
        };
        assert!(!evaluate_workflow_predicate(
            "parse_strategy_page_index",
            &page_index_edge,
            &state
        ));

        let exhausted_state = json!({
            "current_layer_context": {
                "metadata": {"disabled_strategies": ["a", "b", "c"]}
            }
        });
        assert!(evaluate_workflow_predicate(
            "all_strategies_exhausted",
            &edge,
            &exhausted_state
        ));
        assert!(!evaluate_workflow_predicate(
            "strategy_failed_with_remaining",
            &edge,
            &exhausted_state
        ));
    }

    #[test]
    fn guards_commit_and_repair_candidates() {
        let edge = WorkflowEdgeInfo {
            dst: "workflow|any".to_string(),
        };
        let valid_state = json!({
            "current_layer_review": {
                "metadata": {"commit_validation": true},
                "satisfied": true
            },
            "current_layer_context": {
                "parent_node_ids": ["p1", "p2"]
            },
            "current_layer_result": {
                "children": [{"parent_node_id": "p1"}]
            }
        });
        assert!(evaluate_workflow_predicate(
            "commit_candidates_valid",
            &edge,
            &valid_state
        ));
        assert!(evaluate_workflow_predicate(
            "batch_has_repair_candidates",
            &edge,
            &valid_state
        ));

        let invalid_state = json!({
            "current_layer_review": {
                "metadata": {"commit_validation": true},
                "satisfied": false
            }
        });
        assert!(evaluate_workflow_predicate(
            "commit_candidates_invalid",
            &edge,
            &invalid_state
        ));
    }

    #[test]
    fn validates_layer_satisfaction() {
        let edge = WorkflowEdgeInfo {
            dst: "workflow|any".to_string(),
        };
        let state = json!({
            "current_layer_review": {
                "metadata": {},
                "coverage_ok": true,
                "satisfied": true
            },
            "current_layer_result": {"satisfied": true}
        });
        assert!(evaluate_workflow_predicate(
            "layer_satisfied",
            &edge,
            &state
        ));

        let conflicting = json!({
            "current_layer_review": {
                "metadata": {},
                "coverage_ok": true,
                "overlap_conflicts": [{"left": "a"}]
            },
            "current_layer_result": {"satisfied": true}
        });
        assert!(!evaluate_workflow_predicate(
            "layer_satisfied",
            &edge,
            &conflicting
        ));
    }
}
