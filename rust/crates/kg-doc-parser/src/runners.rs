use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::page_index_contracts::PageIndexSourceFormat;
use crate::page_index_pipeline::parse_page_index_document_heuristic;

pub fn infer_page_index_source_format(
    source_path: &Path,
    requested: &str,
) -> Result<PageIndexSourceFormat> {
    match requested {
        "text" => Ok(PageIndexSourceFormat::Text),
        "markdown" => Ok(PageIndexSourceFormat::Markdown),
        "auto" => {
            let extension = source_path
                .extension()
                .and_then(std::ffi::OsStr::to_str)
                .map(|value| value.to_lowercase())
                .unwrap_or_default();
            if extension == "md" || extension == "markdown" {
                Ok(PageIndexSourceFormat::Markdown)
            } else {
                Ok(PageIndexSourceFormat::Text)
            }
        }
        other => {
            anyhow::bail!("page-index source format must be auto, text, or markdown: {other:?}")
        }
    }
}

fn walk_directory(
    directory: &Path,
    allowed_suffixes: &std::collections::HashSet<String>,
    recursive: bool,
    files: &mut Vec<PathBuf>,
) -> Result<()> {
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() && recursive {
            walk_directory(&path, allowed_suffixes, recursive, files)?;
        } else if path.is_file() {
            let suffix = path
                .extension()
                .and_then(std::ffi::OsStr::to_str)
                .map(|value| format!(".{}", value.to_lowercase()))
                .unwrap_or_default();
            if allowed_suffixes.contains(&suffix) {
                files.push(path);
            }
        }
    }
    Ok(())
}

pub fn discover_input_files(
    paths: &[PathBuf],
    allowed_suffixes: &[&str],
    recursive: bool,
) -> Result<Vec<PathBuf>> {
    let allowed_suffixes = allowed_suffixes
        .iter()
        .map(|suffix| suffix.to_lowercase())
        .collect::<std::collections::HashSet<_>>();
    let mut files = Vec::new();
    for raw_path in paths {
        let path = raw_path.as_path();
        if path.is_dir() {
            walk_directory(path, &allowed_suffixes, recursive, &mut files)?;
        } else {
            let suffix = path
                .extension()
                .and_then(std::ffi::OsStr::to_str)
                .map(|value| format!(".{}", value.to_lowercase()))
                .unwrap_or_default();
            if allowed_suffixes.contains(&suffix) {
                files.push(path.to_path_buf());
            }
        }
    }
    let mut unique = BTreeMap::new();
    for path in files {
        let key = path.canonicalize().unwrap_or_else(|_| path.clone());
        unique.insert(key, path);
    }
    let mut result = unique.into_values().collect::<Vec<_>>();
    result.sort_by_key(|path| path.to_string_lossy().to_string());
    Ok(result)
}

pub fn collision_safe_batch_identity(
    source_path: &Path,
    stem_counts: &BTreeMap<String, usize>,
) -> String {
    let stem = source_path
        .file_stem()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or_default()
        .to_string();
    if stem_counts.get(&stem).copied().unwrap_or(0) <= 1 {
        return stem;
    }
    let resolved = source_path
        .canonicalize()
        .unwrap_or_else(|_| source_path.to_path_buf());
    let mut digest = Sha256::new();
    digest.update(resolved.to_string_lossy().as_bytes());
    let fingerprint = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("{stem}-{}", &fingerprint[..16])
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowCommandResult {
    pub kind: String,
    pub input_path: PathBuf,
    pub output_dir: PathBuf,
    pub status: String,
    pub probe_path: PathBuf,
    pub summary_path: PathBuf,
    pub extra: Value,
}

fn probe_event(path: &Path, kind: &str, payload: Value) -> Result<()> {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(
        file,
        "{}",
        serde_json::to_string(&serde_json::json!({"event": kind, "payload": payload}))?
    )?;
    Ok(())
}

fn document_stem(path: &Path) -> String {
    path.file_stem()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or("document")
        .to_string()
}

pub fn run_page_index_source_workflow(
    source_path: &Path,
    output_dir: &Path,
    source_format: PageIndexSourceFormat,
) -> Result<WorkflowCommandResult> {
    let output_dir = output_dir.to_path_buf();
    std::fs::create_dir_all(&output_dir)?;
    let probe_path = output_dir.join("workflow-events.jsonl");
    let raw_text = std::fs::read_to_string(source_path)?;
    probe_event(
        &probe_path,
        "workflow.file_started",
        serde_json::json!({
            "workflow_kind": "page_index",
            "source_path": source_path,
            "output_dir": output_dir,
        }),
    )?;
    let stem = document_stem(source_path);
    let result = parse_page_index_document_heuristic(&stem, &stem, &raw_text, source_format)?;
    let page_count = result.semantic_tree.child_nodes.len();
    let max_depth = result
        .semantic_tree
        .child_nodes
        .iter()
        .map(|page| page.child_nodes.len())
        .max()
        .unwrap_or(0);
    let summary = serde_json::json!({
        "kind": "page_index",
        "source_path": source_path,
        "mode": "heuristic",
        "source_format": source_format,
        "page_count": page_count,
        "max_depth": max_depth,
    });
    let summary_path = output_dir.join("page-index-summary.json");
    std::fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;
    probe_event(
        &probe_path,
        "workflow.file_finished",
        serde_json::json!({
            "workflow_kind": "page_index",
            "source_path": source_path,
            "output_dir": output_dir,
            "status": "succeeded",
            "summary_path": summary_path,
        }),
    )?;
    Ok(WorkflowCommandResult {
        kind: "page_index".to_string(),
        input_path: source_path.to_path_buf(),
        output_dir,
        status: "succeeded".to_string(),
        probe_path,
        summary_path,
        extra: summary,
    })
}

pub fn run_page_index_batch_workflow(
    paths: &[PathBuf],
    output_dir: &Path,
    source_format: PageIndexSourceFormat,
) -> Result<Vec<WorkflowCommandResult>> {
    let output_dir = output_dir.to_path_buf();
    std::fs::create_dir_all(&output_dir)?;
    let probe_path = output_dir.join("workflow-events.jsonl");
    let files = discover_input_files(paths, &[".md", ".txt", ".markdown"], true)?;
    probe_event(
        &probe_path,
        "workflow.batch_started",
        serde_json::json!({
            "workflow_kind": "page_index",
            "file_count": files.len(),
            "output_dir": output_dir,
        }),
    )?;
    let results = files
        .iter()
        .map(|source_path| {
            run_page_index_source_workflow(
                source_path,
                &output_dir.join(document_stem(source_path)),
                source_format,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    probe_event(
        &probe_path,
        "workflow.batch_finished",
        serde_json::json!({
            "workflow_kind": "page_index",
            "file_count": results.len(),
            "output_dir": output_dir,
        }),
    )?;
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn discovers_and_deduplicates_input_files() {
        let temp = tempfile::tempdir().unwrap();
        let nested = temp.path().join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        let alpha = temp.path().join("alpha.md");
        let beta = nested.join("beta.txt");
        std::fs::write(&alpha, "alpha").unwrap();
        std::fs::write(&beta, "beta").unwrap();
        let files = discover_input_files(
            &[temp.path().to_path_buf(), alpha.clone()],
            &[".md", ".txt"],
            true,
        )
        .unwrap();
        assert_eq!(files.len(), 2);
        assert!(files.contains(&alpha));
        assert!(files.contains(&beta));
        let shallow =
            discover_input_files(&[temp.path().to_path_buf()], &[".md", ".txt"], false).unwrap();
        assert_eq!(shallow, vec![alpha]);
    }

    #[test]
    fn preserves_unique_stems_and_hashes_collisions() {
        let unique = collision_safe_batch_identity(
            Path::new("inputs/unique.md"),
            &BTreeMap::from([("unique".to_string(), 1)]),
        );
        assert_eq!(unique, "unique");
        let identity = collision_safe_batch_identity(
            Path::new("inputs/one/duplicate.md"),
            &BTreeMap::from([("duplicate".to_string(), 2)]),
        );
        assert_ne!(identity, "duplicate");
        assert!(identity.starts_with("duplicate-"));
        assert_eq!(identity.len(), "duplicate-".len() + 16);
    }

    #[test]
    fn runs_page_index_source_and_batch_workflows() {
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("page-index.md");
        std::fs::write(&input, "# Section One\n\nParagraph one.").unwrap();
        let result = run_page_index_source_workflow(
            &input,
            &temp.path().join("single"),
            PageIndexSourceFormat::Markdown,
        )
        .unwrap();
        assert_eq!(result.status, "succeeded");
        assert_eq!(result.extra["page_count"], 1);
        assert!(result.summary_path.exists());
        assert!(result.probe_path.exists());

        let results = run_page_index_batch_workflow(
            &[temp.path().to_path_buf()],
            &temp.path().join("batch"),
            PageIndexSourceFormat::Markdown,
        )
        .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].input_path, input);
        assert!(results[0].summary_path.exists());
    }
}
