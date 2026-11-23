use std::process::Command;

#[test]
fn demo_summary_cli_writes_artifacts() {
    let temp = tempfile::tempdir().unwrap();
    let output_dir = temp.path().join("demo");
    let input = temp.path().join("demo-doc.txt");
    std::fs::write(&input, "").unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_kg-doc-parser"))
        .args([
            "demo-summary",
            "--input",
            input.to_str().unwrap(),
            "--output-dir",
            output_dir.to_str().unwrap(),
            "--mode",
            "fake_layered",
        ])
        .status()
        .unwrap();
    assert!(status.success());
    assert!(output_dir.join("demo-summary.json").exists());
    assert!(output_dir.join("engines").exists());
    assert!(output_dir.join("server-data").exists());
}

#[test]
fn demo_summary_cli_uses_custom_title_and_text() {
    let temp = tempfile::tempdir().unwrap();
    let output_dir = temp.path().join("demo");
    let status = Command::new(env!("CARGO_BIN_EXE_kg-doc-parser"))
        .args([
            "demo-summary",
            "--input",
            temp.path().join("doc").to_str().unwrap(),
            "--output-dir",
            output_dir.to_str().unwrap(),
            "--title",
            "Custom Demo",
            "--mode",
            "fake_layered",
            "--text",
            "Custom alpha\nCustom beta",
        ])
        .status()
        .unwrap();
    assert!(status.success());
    let summary: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(output_dir.join("demo-summary.json")).unwrap(),
    )
    .unwrap();
    assert!(summary.is_object());
}
