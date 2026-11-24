use std::process::Command;

#[test]
fn page_index_workflow_cli_writes_artifacts() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("page-index.md");
    std::fs::write(&input, "# Section One\n\nParagraph one.").unwrap();
    let output_dir = temp.path().join("workflow");
    let status = Command::new(env!("CARGO_BIN_EXE_kg-doc-parser"))
        .args([
            "page-index-workflow",
            "--input",
            input.to_str().unwrap(),
            "--output-dir",
            output_dir.to_str().unwrap(),
            "--mode",
            "markdown",
        ])
        .status()
        .unwrap();
    assert!(status.success());
    assert!(output_dir.join("page-index-summary.json").exists());
    assert!(output_dir.join("workflow-events.jsonl").exists());
}

#[test]
fn page_index_workflow_cli_runs_batch() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("page-index.md");
    std::fs::write(&input, "# Section One\n\nParagraph one.").unwrap();
    let output_dir = temp.path().join("workflow-batch");
    let status = Command::new(env!("CARGO_BIN_EXE_kg-doc-parser"))
        .args([
            "page-index-workflow",
            "--input",
            temp.path().to_str().unwrap(),
            "--output-dir",
            output_dir.to_str().unwrap(),
            "--mode",
            "markdown",
            "--batch",
        ])
        .status()
        .unwrap();
    assert!(status.success());
    let events = output_dir.join("workflow-events.jsonl");
    let text = std::fs::read_to_string(events).unwrap();
    assert!(text.contains("workflow.batch_started"));
    assert!(text.contains("workflow.batch_finished"));
}
