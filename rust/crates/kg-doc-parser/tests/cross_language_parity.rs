use kg_doc_parser::document_grouping::{group_node_id, grouping_policy_identity, GroupProposal};
use kg_doc_parser::document_models::AdaptiveGroupingPolicy;
use kg_doc_parser::document_models::{
    BoundingBox, CoordinateSystem, SourceAccuracy, SourceLocation,
};
use kg_doc_parser::layerwise_edits::{
    apply_proposals, semantic_tree_to_kge_payload, LayerwiseChild, LayerwisePointer,
};
use kg_doc_parser::ocr_artifacts::artifact_schema_error;
use kg_doc_parser::ocr_quality::ocr_artifact_quality_error;
use kg_doc_parser::page_index_candidates::extract_candidate_blocks;
use kg_doc_parser::page_index_classification::{classify_block, ClassifyBlockOptions};
use kg_doc_parser::page_index_contracts::PageIndexSourceFormat;
use kg_doc_parser::page_index_pipeline::heuristic_page_outline;
use kg_doc_parser::page_index_tree::page_index_normalize_text;
use kg_doc_parser::pointer_matching::{
    all_exact_occurrences, best_occurrence_by_proximity, safe_slice, soft_exact_positions,
    whitespace_collapse,
};
use kg_doc_parser::semantic::SemanticNode;
use kg_doc_parser::service::evaluate_workflow_predicate;
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
fn document_grouping_identities_match_python() {
    let policy = AdaptiveGroupingPolicy {
        enabled: true,
        maximum_rounds: 6,
        window_size: 64,
        window_overlap: 31,
        maximum_evidence_items_per_candidate: 4,
        maximum_text_chars_per_candidate: 300,
    };
    let proposal = GroupProposal {
        kind: "section".to_string(),
        members: vec!["a".to_string(), "b".to_string()],
    };
    let identity =
        grouping_policy_identity("grouping-v1", &policy, Some("proposer"), Some("summary"));
    assert_eq!(identity.len(), "grouping-v1:adaptive-".len() + 16);
    assert_eq!(
        identity,
        grouping_policy_identity("grouping-v1", &policy, Some("proposer"), Some("summary"))
    );
    let node_id = group_node_id("doc", "grouping-v1", &proposal);
    assert_eq!(node_id.len(), "merge:".len() + 24);
    assert_eq!(node_id, group_node_id("doc", "grouping-v1", &proposal));
}

#[test]
fn semantic_tree_kge_export_matches_python() {
    let payload = fixture();
    for case in payload["semantic_tree_kge_export"].as_array().unwrap() {
        let tree: SemanticNode = serde_json::from_value(case["tree"].clone()).unwrap();
        let actual = semantic_tree_to_kge_payload(
            &tree,
            Some(&case["parser_source_map"]),
            Some(case["doc_id"].as_str().unwrap()),
            case["insertion_method"].as_str().unwrap(),
        )
        .unwrap();
        assert_eq!(actual, case["expected"]);
    }
}

#[test]
fn document_model_validation_matches_python() {
    let bounding_box = BoundingBox {
        x0: 0.0,
        y0: 0.0,
        x1: 1.0,
        y1: 1.0,
        coordinate_system: CoordinateSystem::Relative,
    };
    assert!(bounding_box.validate().is_ok());
    let invalid_bounding_box = BoundingBox {
        x0: 0.0,
        y0: 0.0,
        x1: 0.0,
        y1: 1.0,
        coordinate_system: CoordinateSystem::Relative,
    };
    assert!(invalid_bounding_box.validate().is_err());
    let source_location = SourceLocation {
        page_index: Some(1),
        printed_page_label: Some("1".to_string()),
        bbox: Some(bounding_box),
        start_seconds: None,
        end_seconds: None,
        structural_path: None,
        accuracy: SourceAccuracy::Region,
    };
    assert!(source_location.validate().is_ok());
    let missing_bbox = SourceLocation {
        page_index: Some(1),
        printed_page_label: Some("1".to_string()),
        bbox: None,
        start_seconds: None,
        end_seconds: None,
        structural_path: None,
        accuracy: SourceAccuracy::Region,
    };
    assert!(missing_bbox.validate().is_err());
}

#[test]
fn layerwise_cud_matches_python() {
    let source_map = serde_json::json!({
        "p1_c0": {"id": "p1_c0", "text": "Alpha Beta"}
    });
    let pointer = LayerwisePointer {
        source_cluster_id: "p1_c0".to_string(),
        start_char: 0,
        end_char: 4,
        verbatim_text: "Alpha".to_string(),
        start_delimiter: None,
        end_delimiter: None,
        validation_method: None,
    };
    let beta_pointer = LayerwisePointer {
        source_cluster_id: "p1_c0".to_string(),
        start_char: 6,
        end_char: 9,
        verbatim_text: "Beta".to_string(),
        start_delimiter: None,
        end_delimiter: None,
        validation_method: None,
    };
    let children = vec![
        LayerwiseChild {
            parent_node_id: Some("parent".to_string()),
            node_type: "TEXT_FLOW".to_string(),
            title: "Alpha".to_string(),
            pointers: vec![pointer],
        },
        LayerwiseChild {
            parent_node_id: Some("parent".to_string()),
            node_type: "TEXT_FLOW".to_string(),
            title: "Beta".to_string(),
            pointers: vec![beta_pointer.clone()],
        },
    ];
    let proposals = vec![
        serde_json::json!({
            "edit_type": "ADD_NODE",
            "add": {
                "parent_node_id": "parent",
                "node_type": "TEXT_FLOW",
                "title": "Gamma",
                "pointers": [{
                    "source_cluster_id": "p1_c0",
                    "start_char": 6,
                    "end_char": 9,
                    "verbatim_text": "Beta"
                }]
            },
            "reasoning_create": "test"
        }),
        serde_json::json!({
            "edit_type": "DELETE_NODE",
            "target": {"node_id": "1"},
            "reasoning_delete": "test"
        }),
        serde_json::json!({
            "edit_type": "EDIT_NODE",
            "target": {"node_id": "0"},
            "patch": {"title": "Alpha"},
            "reasoning_update": "test"
        }),
    ];
    let applied = apply_proposals(&proposals, &children, &source_map).unwrap();
    assert!(applied.proposal_errors.is_empty());
    assert_eq!(applied.children.len(), 2);
    assert_eq!(applied.children[0].title, "Alpha");
    assert_eq!(applied.children[1].title, "Gamma");
    assert_eq!(
        applied.children[1].pointers[0].verbatim_text,
        beta_pointer.verbatim_text
    );
}

#[test]
fn page_index_candidate_extraction_matches_python() {
    let payload = fixture();
    for case in payload["page_index_candidates"].as_array().unwrap() {
        let actual = extract_candidate_blocks(
            case["page_text"].as_str().unwrap(),
            case["page_number"].as_u64().unwrap() as u32,
            if case["source_format"].as_str().unwrap() == "markdown" {
                PageIndexSourceFormat::Markdown
            } else {
                PageIndexSourceFormat::Text
            },
        );
        let texts = actual
            .iter()
            .map(|candidate| candidate.text.clone())
            .collect::<Vec<_>>();
        let kinds = actual
            .iter()
            .map(|candidate| candidate.kind_hint.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            serde_json::to_value(&texts).unwrap(),
            case["expected_texts"]
        );
        assert_eq!(
            serde_json::to_value(&kinds).unwrap(),
            case["expected_kinds"]
        );
    }
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
fn ocr_artifact_quality_errors_match_python() {
    let payload = fixture();
    for case in payload["ocr_artifact_quality"].as_array().unwrap() {
        let actual = ocr_artifact_quality_error(&case["payload"]);
        let expected = case["expected"].as_str().map(str::to_string);
        assert_eq!(actual, expected);
    }
}

#[test]
fn service_workflow_predicates_match_python() {
    let payload = fixture();
    for case in payload["service_workflow_predicates"].as_array().unwrap() {
        let edge = kg_doc_parser::service::WorkflowEdgeInfo {
            dst: case["edge"]["dst"].as_str().unwrap().to_string(),
        };
        let actual =
            evaluate_workflow_predicate(case["name"].as_str().unwrap(), &edge, &case["state"]);
        assert_eq!(actual, case["expected"].as_bool().unwrap());
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
    for case in payload["soft_exact_positions_fuzzy"].as_array().unwrap() {
        let (actual, diagnostic) = soft_exact_positions(
            case["source_text"].as_str().unwrap(),
            case["verbatim"].as_str().unwrap(),
            Some(case["fuzzy_threshold"].as_f64().unwrap()),
            case["fuzzy_len_stretch"].as_f64().unwrap(),
            case["fuzzy_stride_frac"].as_f64().unwrap(),
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
