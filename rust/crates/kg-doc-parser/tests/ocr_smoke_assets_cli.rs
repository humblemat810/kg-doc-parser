use std::process::Command;
use tempfile::tempdir;

#[test]
fn ocr_smoke_assets_cli_writes_expected_files() {
    let temp = tempdir().unwrap();
    let output_dir = temp.path().join("assets");
    let executable = env!("CARGO_BIN_EXE_kg-doc-parser");

    let output = Command::new(executable)
        .args(["ocr-smoke-assets", "--output-dir"])
        .arg(&output_dir)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output_dir.join("ocr_smoke_page_1.png").is_file());
    assert!(output_dir.join("ocr_smoke_page_2.png").is_file());
    let pdf = output_dir.join("ocr_smoke_document.pdf");
    assert!(pdf.is_file());
    assert!(std::fs::read(&pdf).unwrap().starts_with(b"%PDF-1.4"));
}
