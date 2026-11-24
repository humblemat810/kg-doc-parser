use kg_doc_parser::layerwise_llm::{
    run_layerwise_review_loop, LayerwiseLlmChildResponse, LayerwiseReviewResponse,
};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn review_loop_applies_delete_then_stops_on_empty_round() {
    let source_map = json!({
        "p1_c0": {"id": "p1_c0", "text": "Alpha page"},
        "p1_c1": {"id": "p1_c1", "text": "Beta page"}
    });
    let children = vec![
        LayerwiseLlmChildResponse {
            id: None,
            parent_node_id: "root".to_string(),
            node_type: "TEXT_FLOW".to_string(),
            title: "Alpha".to_string(),
            pointers: vec![kg_doc_parser::semantic::HydratedTextPointer {
                source_cluster_id: "p1_c0".to_string(),
                start_char: 0,
                end_char: 4,
                verbatim_text: "Alpha".to_string(),
                start_delimiter: None,
                end_delimiter: None,
                validation_method: None,
            }],
        },
        LayerwiseLlmChildResponse {
            id: None,
            parent_node_id: "root".to_string(),
            node_type: "TEXT_FLOW".to_string(),
            title: "Beta".to_string(),
            pointers: vec![kg_doc_parser::semantic::HydratedTextPointer {
                source_cluster_id: "p1_c1".to_string(),
                start_char: 0,
                end_char: 3,
                verbatim_text: "Beta".to_string(),
                start_delimiter: None,
                end_delimiter: None,
                validation_method: None,
            }],
        },
    ];
    let calls = AtomicUsize::new(0);
    let invoker = |request: &kg_doc_parser::layerwise_llm::LayerwiseReviewRequest| -> Result<Value, anyhow::Error> {
        let invocation = calls.fetch_add(1, Ordering::SeqCst) + 1;
        anyhow::ensure!(request.doc_id == "doc");
        anyhow::ensure!(request.current_depth == 1);
        if invocation == 1 {
            anyhow::ensure!(request.children.len() == 2);
            return Ok(json!(LayerwiseReviewResponse {
                reasoning: "delete duplicate scope".to_string(),
                cproposals: vec![],
                uproposals: vec![],
                dproposals: vec![json!({
                    "edit_type": "DELETE_NODE",
                    "target": {"node_type": "TEXT_FLOW", "title": "Alpha"},
                    "reasoning_delete": "outside parent"
                })],
            }));
        }
        anyhow::ensure!(request.children.len() == 1);
        anyhow::ensure!(request.children[0].title == "Beta");
        Ok(json!(LayerwiseReviewResponse {
            reasoning: "review complete".to_string(),
            cproposals: vec![],
            uproposals: vec![],
            dproposals: vec![],
        }))
    };
    let outcome = run_layerwise_review_loop(
        children,
        &source_map,
        r#"{"pages":[]}"#,
        "doc",
        1,
        &["model".to_string()],
        &invoker,
    )
    .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(outcome.rounds, 1);
    assert!(outcome.edited);
    assert_eq!(outcome.reasoning_steps, 2);
    assert_eq!(outcome.children.len(), 1);
    assert_eq!(outcome.children[0].title, "Beta");
}
