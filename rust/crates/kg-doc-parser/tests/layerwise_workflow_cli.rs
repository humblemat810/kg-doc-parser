use std::process::Command;

#[test]
fn layerwise_workflow_cli_writes_artifacts() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("legacy-doc");
    std::fs::create_dir_all(&input).unwrap();
    std::fs::write(
        input.join("page_1.json"),
        serde_json::json!({
            "pdf_page_num": 1,
            "OCR_text_clusters": [{"text": "Alpha page"}],
            "non_text_objects": []
        })
        .to_string(),
    )
    .unwrap();
    std::fs::write(
        input.join("page_2.json"),
        serde_json::json!({
            "pdf_page_num": 2,
            "OCR_text_clusters": [{"text": "Beta page"}],
            "non_text_objects": []
        })
        .to_string(),
    )
    .unwrap();
    let output_dir = temp.path().join("workflow");
    let status = Command::new(env!("CARGO_BIN_EXE_kg-doc-parser"))
        .args([
            "layerwise-workflow",
            "--input",
            input.to_str().unwrap(),
            "--output-dir",
            output_dir.to_str().unwrap(),
            "--mode",
            "snippet",
        ])
        .status()
        .unwrap();
    assert!(status.success());
    assert!(output_dir.join("layerwise-summary.json").exists());
    assert!(output_dir.join("layerwise-graph.json").exists());
    assert!(output_dir.join("workflow-events.jsonl").exists());
}
