use std::process::Command;

#[test]
fn ocr_workflow_cli_writes_source_artifacts() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("sample.png");
    std::fs::write(&input, b"fake image").unwrap();
    let output_dir = temp.path().join("workflow");
    let status = Command::new(env!("CARGO_BIN_EXE_kg-doc-parser"))
        .args([
            "ocr-workflow",
            "--input",
            input.to_str().unwrap(),
            "--output-dir",
            output_dir.to_str().unwrap(),
            "--provider",
            "fake",
        ])
        .status()
        .unwrap();
    assert!(status.success());
    assert!(output_dir.join("ocr-summary.json").exists());
    assert!(output_dir.join("ocr-progress.json").exists());
    assert!(output_dir.join("legacy_split_pages/page_1.json").exists());
}

#[test]
fn ocr_workflow_cli_rejects_unsupported_provider() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("sample.png");
    std::fs::write(&input, b"fake image").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_kg-doc-parser"))
        .args([
            "ocr-workflow",
            "--input",
            input.to_str().unwrap(),
            "--output-dir",
            temp.path().join("workflow").to_str().unwrap(),
            "--provider",
            "unknown",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("unsupported OCR provider"));
}
