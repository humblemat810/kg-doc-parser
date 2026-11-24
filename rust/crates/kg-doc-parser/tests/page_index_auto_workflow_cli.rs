use std::process::Command;

#[test]
fn page_index_auto_infers_markdown() {
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
            "auto",
        ])
        .status()
        .unwrap();
    assert!(status.success());
    let summary: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(output_dir.join("page-index-summary.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(summary["source_format"], "markdown");
}
