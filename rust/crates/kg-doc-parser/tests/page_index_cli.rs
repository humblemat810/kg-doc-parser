use std::process::Command;

#[test]
fn page_index_cli_emits_semantic_tree() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("page-index.md");
    std::fs::write(&input, "# Section One\n\nParagraph one.").unwrap();
    let output = temp.path().join("page-index.json");
    let status = Command::new(env!("CARGO_BIN_EXE_kg-doc-parser"))
        .args([
            "page-index",
            "--input",
            input.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success());
    let payload: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(output).unwrap()).unwrap();
    assert_eq!(payload["semantic_tree"]["node_type"], "DOCUMENT_ROOT");
    assert_eq!(
        payload["semantic_tree"]["child_nodes"]
            .as_array()
            .map(Vec::len),
        Some(1)
    );
}
