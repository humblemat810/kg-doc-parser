use kg_doc_parser::ocr_artifacts::artifact_schema_error;
use kg_doc_parser::page_index_classification::{classify_block, ClassifyBlockOptions};
use kg_doc_parser::page_index_contracts::PageIndexSourceFormat;
use kg_doc_parser::page_index_pipeline::heuristic_page_outline;
use kg_doc_parser::page_index_tree::page_index_normalize_text;
use kg_doc_parser::pointer_matching::{
    all_exact_occurrences, best_occurrence_by_proximity, safe_slice, soft_exact_positions,
    whitespace_collapse,
};
use kg_doc_parser::{
    page_index_source_text::split_pages, semantic_indexing as indexing,
    source_text::source_map_entry_text,
};
use serde_json::Value;
use std::path::PathBuf;

fn fixture() -> Value {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/semantic_contracts.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn page_index_normalize_text_matches_python() {
    let payload = fixture();
    for case in payload["page_index_normalize_text"].as_array().unwrap() {
        assert_eq!(
            page_index_normalize_text(case["text"].as_str().unwrap()),
            case["expected"].as_str().unwrap(),
        );
    }
}

#[test]
fn ocr_artifact_schema_errors_match_python() {
    let payload = fixture();
    for case in payload["ocr_artifact_schema"].as_array().unwrap() {
        let actual = artifact_schema_error(case);
        let expected = case["expected"].as_str().map(str::to_string);
        assert_eq!(actual, expected);
    }
}

#[test]
fn pointer_matching_contracts_match_python() {
    let payload = fixture()["pointer_matching"].clone();
    for case in payload["safe_slice"].as_array().unwrap() {
        assert_eq!(
            safe_slice(
                case["text"].as_str().unwrap(),
                case["start"].as_i64().unwrap() as isize,
                case["end_inclusive"].as_i64().unwrap() as isize,
            ),
            case["expected"].as_str().unwrap(),
        );
    }
    for case in payload["all_exact_occurrences"].as_array().unwrap() {
        let actual = all_exact_occurrences(
            case["haystack"].as_str().unwrap(),
            case["needle"].as_str().unwrap(),
        );
        assert_eq!(serde_json::to_value(actual).unwrap(), case["expected"]);
    }
    for case in payload["best_occurrence_by_proximity"].as_array().unwrap() {
        let occurrences = case["occurrences"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| {
                let items = value.as_array().unwrap();
                (
                    items[0].as_u64().unwrap() as usize,
                    items[1].as_u64().unwrap() as usize,
                )
            })
            .collect::<Vec<_>>();
        let actual = best_occurrence_by_proximity(
            &occurrences,
            case["proposed_start"].as_u64().unwrap() as usize,
        );
        assert_eq!(serde_json::to_value(actual).unwrap(), case["expected"]);
    }
    for case in payload["whitespace_collapse"].as_array().unwrap() {
        assert_eq!(
            whitespace_collapse(case["text"].as_str().unwrap()),
            case["expected"].as_str().unwrap(),
        );
    }
    for case in payload["soft_exact_positions"].as_array().unwrap() {
        let (actual, diagnostic) = soft_exact_positions(
            case["source_text"].as_str().unwrap(),
            case["verbatim"].as_str().unwrap(),
            None,
            0.25,
            0.10,
        );
        assert_eq!(serde_json::to_value(&actual).unwrap(), case["expected"]);
        assert_eq!(
            serde_json::to_value(diagnostic).unwrap(),
            case["diagnostic"]
        );
    }
}

#[test]
fn page_index_heuristic_outlines_match_python() {
    let payload = fixture();
    for case in payload["page_index_outlines"].as_array().unwrap() {
        let source_format = match case["source_format"].as_str().unwrap() {
            "markdown" => PageIndexSourceFormat::Markdown,
            _ => PageIndexSourceFormat::Text,
        };
        let (block_specs, diagnostics) = heuristic_page_outline(
            case["page_text"].as_str().unwrap(),
            case["page_number"].as_u64().unwrap() as u32,
            source_format,
        );
        assert_eq!(
            serde_json::to_value(&block_specs).unwrap(),
            case["expected_block_specs"]
        );
        assert_eq!(diagnostics, case["expected_diagnostics"]);
    }
}

#[test]
fn page_index_classify_blocks_matches_python() {
    let payload = fixture();
    for case in payload["page_index_classify_blocks"].as_array().unwrap() {
        let options = &case["options"];
        let source_format = match options["source_format"].as_str().unwrap() {
            "markdown" => PageIndexSourceFormat::Markdown,
            _ => PageIndexSourceFormat::Text,
        };
        let actual = classify_block(
            case["text"].as_str().unwrap(),
            case["start_char"].as_u64().unwrap() as usize,
            case["end_char"].as_u64().unwrap() as usize,
            ClassifyBlockOptions {
                source_format,
                is_first: options["is_first"].as_bool().unwrap_or(false),
                ..Default::default()
            },
        );
        let expected = &case["expected"];
        assert_eq!(
            serde_json::to_value(actual.node_type).unwrap(),
            expected["node_type"]
        );
        assert_eq!(actual.title, expected["title"]);
        assert_eq!(actual.kind_hint, expected["kind_hint"]);
        if let Some(heading_level) = expected["heading_level"].as_u64() {
            assert_eq!(actual.heading_level, Some(heading_level as u8));
        } else {
            assert_eq!(actual.heading_level, None);
        }
    }
}

#[test]
fn page_index_split_pages_matches_python() {
    let payload = fixture();
    for case in payload["page_index_split_pages"].as_array().unwrap() {
        let actual = split_pages(case["raw_text"].as_str().unwrap());
        let expected = case["expected"].as_array().unwrap();
        let expected = expected
            .iter()
            .map(Value::as_str)
            .collect::<Option<Vec<_>>>()
            .unwrap();
        assert_eq!(actual, expected);
    }
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
