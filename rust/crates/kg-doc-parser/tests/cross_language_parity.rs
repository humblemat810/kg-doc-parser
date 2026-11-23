use kg_doc_parser::{semantic_indexing as indexing, source_text::source_map_entry_text};
use serde_json::Value;
use std::path::PathBuf;

fn fixture() -> Value {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/semantic_contracts.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn source_map_entry_text_matches_python() {
    let payload = fixture();
    for case in payload["source_map_entry_text"].as_array().unwrap() {
        assert_eq!(
            source_map_entry_text(&case["entry"]),
            case["expected"].as_str().unwrap(),
        );
    }
}

#[test]
fn semantic_indexing_contracts_match_python() {
    let payload = fixture();
    for case in payload["default_token_estimate"].as_array().unwrap() {
        assert_eq!(
            indexing::default_token_estimate(case["text"].as_str().unwrap()),
            case["expected"].as_u64().unwrap() as usize,
        );
    }
    for case in payload["render_nodes_payload"].as_array().unwrap() {
        assert_eq!(
            indexing::render_nodes_payload(&case["nodes"]),
            case["expected"].as_str().unwrap(),
        );
    }
    for case in payload["batch_nodes"].as_array().unwrap() {
        let items = case["items"].as_array().unwrap().to_vec();
        let config =
            indexing::BatchNodesConfig::new(indexing::default_token_estimate, |item: &Value| {
                indexing::json_compact(item)
            })
            .with_node_limit(case["max_nodes"].as_u64().unwrap() as usize)
            .with_base_prompt_tokens(case["base_prompt_tokens"].as_u64().unwrap() as usize)
            .with_per_node_overhead_tokens(
                case["per_node_overhead_tokens"].as_u64().unwrap() as usize,
            );
        let batches = indexing::batch_nodes(&items, config);
        assert_eq!(
            serde_json::to_value(&batches).unwrap(),
            case["expected"],
            "batch ordering must match Python"
        );
    }
}
