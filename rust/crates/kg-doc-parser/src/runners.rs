use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Map;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::layerwise_edits::semantic_tree_to_kge_payload;
use crate::layerwise_parser::{prepare_document_for_llm, stable_id};
use crate::page_index_contracts::PageIndexSourceFormat;
use crate::page_index_pipeline::parse_page_index_document_heuristic;
use crate::parser::ParsingMode;
use crate::semantic::{HydratedTextPointer, SemanticNode};
use crate::source_text::source_map_entry_text;

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

pub fn run_layerwise_batch_workflow(
    source_paths: &[PathBuf],
    output_dir: &Path,
    parsing_mode: ParsingMode,
    max_depth: u64,
) -> Result<Vec<WorkflowCommandResult>> {
    let output_dir = output_dir.to_path_buf();
    std::fs::create_dir_all(&output_dir)?;
    let probe_path = output_dir.join("workflow-events.jsonl");
    let mut directories = source_paths
        .iter()
        .filter(|path| path.is_dir())
        .cloned()
        .collect::<Vec<PathBuf>>();
    directories.sort_by_key(|path| path.to_string_lossy().to_string());
    probe_event(
        &probe_path,
        "workflow.batch_started",
        serde_json::json!({
            "workflow_kind": "layerwise",
            "file_count": directories.len(),
            "output_dir": output_dir,
        }),
    )?;
    let results = directories
        .iter()
        .map(|source_path| {
            run_layerwise_source_workflow(
                source_path,
                &output_dir.join(
                    source_path
                        .file_name()
                        .and_then(std::ffi::OsStr::to_str)
                        .unwrap_or("document"),
                ),
                parsing_mode,
                max_depth,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    probe_event(
        &probe_path,
        "workflow.batch_finished",
        serde_json::json!({
            "workflow_kind": "layerwise",
            "file_count": results.len(),
            "output_dir": output_dir,
        }),
    )?;
    Ok(results)
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

fn read_legacy_ocr_pages(source_path: &Path) -> Result<(String, Vec<Value>)> {
    if !source_path.is_dir() {
        anyhow::bail!("layerwise workflow expects a directory of legacy OCR page artifacts");
    }
    let mut page_files = Vec::new();
    for entry in std::fs::read_dir(source_path)? {
        let path = entry?.path();
        let Some(filename) = path.file_name().and_then(std::ffi::OsStr::to_str) else {
            continue;
        };
        let Some(stem) = filename.strip_suffix(".json") else {
            continue;
        };
        let Some(page_number) = stem.strip_prefix("page_") else {
            continue;
        };
        page_files.push((page_number.parse::<u32>()?, path));
    }
    if page_files.is_empty() {
        anyhow::bail!("legacy OCR folder contains no page_*.json files");
    }
    page_files.sort_by(|(left, _), (right, _)| left.cmp(right));
    let pages = page_files
        .iter()
        .map(|(_, path)| Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?))
        .collect::<Result<Vec<Value>>>()?;
    let name = source_path
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or("legacy-document")
        .to_string();
    Ok((name, pages))
}

fn offline_layerwise_tree(name: &str, raw_document: &Value) -> Result<(SemanticNode, Value)> {
    let prepared = prepare_document_for_llm(raw_document)?;
    let source_map = serde_json::to_value(prepared.source_map.clone())?;
    let cluster_ids = prepared.source_map.keys().cloned().collect::<Vec<String>>();
    let root_id = stable_id("legacy.root_node", &[name, &cluster_ids.join("|")]);
    let mut child_nodes = Vec::new();
    for cluster_id in &cluster_ids {
        let source = prepared.source_map.get(cluster_id).expect("cluster exists");
        let text = source_map_entry_text(source);
        if text.is_empty() {
            continue;
        }
        let end_char = text.chars().count().saturating_sub(1) as i64;
        let title = text
            .split_whitespace()
            .take(8)
            .collect::<Vec<_>>()
            .join(" ");
        child_nodes.push(
            SemanticNode {
                node_id: None,
                parent_id: Some(root_id.clone()),
                node_type: "TEXT_FLOW".to_string(),
                title: if title.is_empty() {
                    "Untitled".to_string()
                } else {
                    title
                },
                summary: String::new(),
                total_content_pointers: vec![HydratedTextPointer {
                    source_cluster_id: cluster_id.clone(),
                    start_char: 0,
                    end_char,
                    verbatim_text: text,
                    start_delimiter: None,
                    end_delimiter: None,
                    validation_method: None,
                }],
                aggregate_content_pointers: Vec::new(),
                child_nodes: Vec::new(),
                level_from_root: 1,
                metadata: serde_json::json!({"offline_fallback": true}),
            }
            .with_stable_id(Some(&root_id)),
        );
    }
    let root = SemanticNode {
        node_id: Some(root_id),
        parent_id: None,
        node_type: "DOCUMENT_ROOT".to_string(),
        title: name.to_string(),
        summary: String::new(),
        total_content_pointers: child_nodes
            .iter()
            .flat_map(|child| child.total_content_pointers.clone())
            .collect(),
        aggregate_content_pointers: Vec::new(),
        child_nodes,
        level_from_root: 0,
        metadata: serde_json::json!({"offline_fallback": true}),
    };
    Ok((root, source_map))
}

pub fn run_layerwise_source_workflow(
    source_path: &Path,
    output_dir: &Path,
    parsing_mode: ParsingMode,
    max_depth: u64,
) -> Result<WorkflowCommandResult> {
    let _ = (parsing_mode, max_depth);
    let output_dir = output_dir.to_path_buf();
    std::fs::create_dir_all(&output_dir)?;
    let probe_path = output_dir.join("workflow-events.jsonl");
    let (name, pages) = read_legacy_ocr_pages(source_path)?;
    let raw_document = serde_json::json!({name.clone(): pages});
    probe_event(
        &probe_path,
        "workflow.file_started",
        serde_json::json!({
            "workflow_kind": "layerwise",
            "source_path": source_path,
            "output_dir": output_dir,
            "parsing_mode": match parsing_mode {
                ParsingMode::Snippet => "snippet",
                ParsingMode::Delimiter => "delimiter",
            },
        }),
    )?;
    let (tree, source_map) = offline_layerwise_tree(&name, &raw_document)?;
    let graph_payload = semantic_tree_to_kge_payload(
        &tree,
        Some(&source_map),
        Some(&name),
        "semantic_document_parser_v1",
    )?;
    let graph_path = output_dir.join("layerwise-graph.json");
    std::fs::write(&graph_path, serde_json::to_string_pretty(&graph_payload)?)?;
    let summary = serde_json::json!({
        "kind": "layerwise",
        "source_path": source_path,
        "node_count": graph_payload["nodes"].as_array().map(Vec::len).unwrap_or(0),
        "edge_count": graph_payload["edges"].as_array().map(Vec::len).unwrap_or(0),
        "source_count": source_map.as_object().map(Map::len).unwrap_or(0),
        "graph_path": graph_path,
    });
    let summary_path = output_dir.join("layerwise-summary.json");
    std::fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;
    probe_event(
        &probe_path,
        "workflow.file_finished",
        serde_json::json!({
            "workflow_kind": "layerwise",
            "source_path": source_path,
            "output_dir": output_dir,
            "status": "succeeded",
            "summary_path": summary_path,
        }),
    )?;
    Ok(WorkflowCommandResult {
        kind: "layerwise".to_string(),
        input_path: source_path.to_path_buf(),
        output_dir,
        status: "succeeded".to_string(),
        probe_path,
        summary_path,
        extra: summary,
    })
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

    #[test]
    fn infers_page_index_source_format() {
        let markdown = Path::new("inputs/page-index.md");
        let text = Path::new("inputs/page-index.txt");
        assert_eq!(
            infer_page_index_source_format(markdown, "auto").unwrap(),
            PageIndexSourceFormat::Markdown
        );
        assert_eq!(
            infer_page_index_source_format(text, "auto").unwrap(),
            PageIndexSourceFormat::Text
        );
        assert_eq!(
            infer_page_index_source_format(text, "markdown").unwrap(),
            PageIndexSourceFormat::Markdown
        );
        assert!(infer_page_index_source_format(text, "json").is_err());
    }

    #[test]
    fn discovers_nested_page_index_files_for_batch_workflows() {
        let temp = tempfile::tempdir().unwrap();
        let nested = temp.path().join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        let alpha = temp.path().join("alpha.md");
        let beta = nested.join("beta.txt");
        std::fs::write(&alpha, "# Alpha\n\nAlpha paragraph.").unwrap();
        std::fs::write(&beta, "Beta\n\nBeta paragraph.").unwrap();
        let results = run_page_index_batch_workflow(
            &[temp.path().to_path_buf()],
            &temp.path().join("output"),
            PageIndexSourceFormat::Text,
        )
        .unwrap();
        assert_eq!(results.len(), 2);
        let input_paths = results
            .iter()
            .map(|result| result.input_path.clone())
            .collect::<Vec<_>>();
        assert!(input_paths.contains(&alpha));
        assert!(input_paths.contains(&beta));
    }

    #[test]
    fn runs_layerwise_source_and_batch_workflows() {
        let temp = tempfile::tempdir().unwrap();
        let document = temp.path().join("legacy-doc");
        std::fs::create_dir_all(&document).unwrap();
        std::fs::write(
            document.join("page_2.json"),
            serde_json::json!({"pdf_page_num": 2, "OCR_text_clusters": [{"text": "Beta page"}], "non_text_objects": []}).to_string(),
        )
        .unwrap();
        std::fs::write(
            document.join("page_1.json"),
            serde_json::json!({"pdf_page_num": 1, "OCR_text_clusters": [{"text": "Alpha page"}], "non_text_objects": []}).to_string(),
        )
        .unwrap();
        let result = run_layerwise_source_workflow(
            &document,
            &temp.path().join("single"),
            ParsingMode::Snippet,
            10,
        )
        .unwrap();
        assert_eq!(result.status, "succeeded");
        assert_eq!(result.extra["source_count"], 2);
        let graph = serde_json::from_str::<Value>(
            &std::fs::read_to_string(result.extra["graph_path"].as_str().unwrap()).unwrap(),
        )
        .unwrap();
        assert_eq!(graph["nodes"].as_array().unwrap().len(), 3);
        assert_eq!(graph["edges"].as_array().unwrap().len(), 2);

        let results = run_layerwise_batch_workflow(
            std::slice::from_ref(&document),
            &temp.path().join("batch"),
            ParsingMode::Snippet,
            10,
        )
        .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].input_path, document);
        assert!(results[0].summary_path.exists());
    }
}
