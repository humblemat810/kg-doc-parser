use serde_json::{Map, Value};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexMode {
    FlatVerbatim,
    BottomUpDigest,
    Both,
}

impl IndexMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FlatVerbatim => "flat_verbatim",
            Self::BottomUpDigest => "bottom_up_digest",
            Self::Both => "both",
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct IdMapping {
    pub forward_map: HashMap<String, String>,
    pub backward_map: HashMap<String, String>,
}

impl IdMapping {
    pub fn to_uuid(&self, short_id: &str) -> Option<&String> {
        self.backward_map.get(short_id)
    }

    pub fn to_short_id(&mut self, node_id: &str, title: &Value) -> String {
        if !self.forward_map.contains_key(node_id) {
            let rendered_title = title
                .as_str()
                .map(ToString::to_string)
                .unwrap_or_else(|| title.to_string());
            let short_id = format!("nid:{}:{}", self.forward_map.len(), rendered_title);
            self.forward_map
                .insert(node_id.to_string(), short_id.clone());
            self.backward_map
                .insert(short_id.clone(), node_id.to_string());
        }
        self.forward_map[node_id].clone()
    }
}

pub fn default_token_estimate(text: &str) -> usize {
    text.chars().count().div_ceil(4).max(1)
}

fn sort_value(value: &Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.iter().map(sort_value).collect()),
        Value::Object(map) => {
            let mut sorted: Map<String, Value> = Map::new();
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            for key in keys {
                sorted.insert(key.clone(), sort_value(&map[key]));
            }
            Value::Object(sorted)
        }
        value => value.clone(),
    }
}

pub fn json_compact(value: &Value) -> String {
    serde_json::to_string(&sort_value(value)).unwrap_or_default()
}

pub fn render_nodes_payload(nodes_payload: &Value) -> String {
    json_compact(&serde_json::json!({ "nodes": nodes_payload }))
}

pub struct BatchNodesConfig<'a, T> {
    pub max_nodes: Option<usize>,
    pub max_input_tokens: Option<usize>,
    pub token_estimator: Box<dyn Fn(&str) -> usize + 'a>,
    pub base_prompt_tokens: usize,
    pub per_node_overhead_tokens: usize,
    pub render_item_for_estimate: Box<dyn Fn(&T) -> String + 'a>,
}

impl<'a, T> BatchNodesConfig<'a, T> {
    pub fn new(
        token_estimator: impl Fn(&str) -> usize + 'a,
        render_item_for_estimate: impl Fn(&T) -> String + 'a,
    ) -> Self {
        Self {
            max_nodes: Some(25),
            max_input_tokens: None,
            token_estimator: Box::new(token_estimator),
            base_prompt_tokens: 450,
            per_node_overhead_tokens: 15,
            render_item_for_estimate: Box::new(render_item_for_estimate),
        }
    }

    pub fn with_node_limit(mut self, max_nodes: usize) -> Self {
        self.max_nodes = Some(max_nodes);
        self
    }

    pub fn with_token_limit(mut self, max_input_tokens: usize) -> Self {
        self.max_input_tokens = Some(max_input_tokens);
        self
    }

    pub fn with_base_prompt_tokens(mut self, base_prompt_tokens: usize) -> Self {
        self.base_prompt_tokens = base_prompt_tokens;
        self
    }

    pub fn with_per_node_overhead_tokens(mut self, per_node_overhead_tokens: usize) -> Self {
        self.per_node_overhead_tokens = per_node_overhead_tokens;
        self
    }
}

pub fn batch_nodes<T: Clone>(items: &[T], config: BatchNodesConfig<T>) -> Vec<Vec<T>> {
    let max_nodes = if config.max_nodes.is_none() && config.max_input_tokens.is_none() {
        Some(25)
    } else {
        config.max_nodes
    };
    let mut batches: Vec<Vec<T>> = Vec::new();
    let mut batch: Vec<T> = Vec::new();
    let mut batch_tokens = config.base_prompt_tokens;
    for item in items {
        let item_text = (config.render_item_for_estimate)(item);
        let item_tokens = (config.token_estimator)(&item_text) + config.per_node_overhead_tokens;
        let would_exceed_tokens = config.max_input_tokens.is_some_and(|max_input_tokens| {
            !batch.is_empty() && batch_tokens + item_tokens > max_input_tokens
        });
        let would_exceed_nodes =
            max_nodes.is_some_and(|max_nodes| !batch.is_empty() && batch.len() + 1 > max_nodes);
        if would_exceed_tokens || would_exceed_nodes {
            batches.push(std::mem::take(&mut batch));
            batch_tokens = config.base_prompt_tokens;
        }
        batch.push(item.clone());
        batch_tokens += item_tokens;
    }
    if !batch.is_empty() {
        batches.push(batch);
    }
    batches
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn short_ids_keep_parentage_and_title() {
        let mut mapping = IdMapping::default();
        assert_eq!(
            mapping.to_short_id("root", &json!("Document")),
            "nid:0:Document"
        );
        assert_eq!(
            mapping.to_short_id("child", &json!("Section")),
            "nid:1:Section"
        );
        assert_eq!(
            mapping.to_short_id("root", &json!("Document")),
            "nid:0:Document"
        );
        assert_eq!(mapping.to_uuid("nid:0:Document").unwrap(), "root");
        assert_eq!(mapping.to_uuid("nid:1:Section").unwrap(), "child");
    }

    #[test]
    fn default_token_estimate_matches_python_quotient() {
        assert_eq!(default_token_estimate(""), 1);
        assert_eq!(default_token_estimate("abcd"), 1);
        assert_eq!(default_token_estimate("abcde"), 2);
    }

    #[test]
    fn json_compact_sorts_object_keys() {
        assert_eq!(
            json_compact(&json!({"b": 1, "a": {"d": 2, "c": 3}})),
            r#"{"a":{"c":3,"d":2},"b":1}"#
        );
    }

    #[test]
    fn render_nodes_payload_uses_compact_json() {
        assert_eq!(
            render_nodes_payload(&json!([{"b": 1, "a": 2}])),
            r#"{"nodes":[{"a":2,"b":1}]}"#
        );
    }

    #[test]
    fn batch_nodes_preserves_order_and_node_limit() {
        let items = vec![1, 2, 3, 4, 5];
        let config = BatchNodesConfig::new(|_text: &str| 1, |item: &i32| item.to_string())
            .with_node_limit(2)
            .with_base_prompt_tokens(0)
            .with_per_node_overhead_tokens(0);
        let batches = batch_nodes(&items, config);
        assert_eq!(batches, vec![vec![1, 2], vec![3, 4], vec![5]]);
    }

    #[test]
    fn batch_nodes_allows_one_oversized_item_without_losing_order() {
        let items = vec!["large", "next"];
        let config = BatchNodesConfig::new(|_text: &str| 5, |item: &&str| item.to_string())
            .with_token_limit(1)
            .with_base_prompt_tokens(0)
            .with_per_node_overhead_tokens(0);
        let batches = batch_nodes(&items, config);
        assert_eq!(batches, vec![vec!["large"], vec!["next"]]);
    }
}
