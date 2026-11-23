use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Map;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::layerwise_edits::semantic_tree_to_kge_payload;
use crate::layerwise_parser::{prepare_document_for_llm, stable_id};
use crate::ocr_artifacts::artifact_schema_error;
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

pub const OCR_FILE_SUFFIXES: [&str; 8] = [
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".pdf",
];
pub const OCR_IMAGE_SUFFIXES: [&str; 7] =
    [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"];

pub trait OcrRunner {
    fn provider_name(&self) -> &str {
        "provider"
    }

    fn run_page(&self, image_path: &Path, page_number: u64) -> Result<Value>;
}

impl<F> OcrRunner for F
where
    F: Fn(&Path, u64) -> Result<Value>,
{
    fn run_page(&self, image_path: &Path, page_number: u64) -> Result<Value> {
        self(image_path, page_number)
    }
}

pub trait PdfRasterizer {
    fn identity(&self) -> &str {
        "render-v1"
    }

    fn render_pdf(&self, pdf_path: &Path, rendered_dir: &Path) -> Result<Vec<PathBuf>>;
}

impl<F> PdfRasterizer for F
where
    F: Fn(&Path, &Path) -> Result<Vec<PathBuf>>,
{
    fn render_pdf(&self, pdf_path: &Path, rendered_dir: &Path) -> Result<Vec<PathBuf>> {
        self(pdf_path, rendered_dir)
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    let contents = std::fs::read(path)?;
    let mut digest = Sha256::new();
    digest.update(contents);
    Ok(digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn normalize_page_number(value: u64) -> Result<i64> {
    let value = i64::try_from(value)?;
    if value <= 0 {
        anyhow::bail!("OCR page numbers must be positive");
    }
    Ok(value)
}

fn validate_ocr_artifact(
    mut artifact: Value,
    image_path: &Path,
    page_number: u64,
    provider_name: &str,
) -> Result<Value> {
    let object = artifact
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("OCR page artifact must be a JSON object"))?;
    let page_number = normalize_page_number(page_number)?;
    if !object.contains_key("pdf_page_num") {
        object.insert("pdf_page_num".to_string(), serde_json::json!(page_number));
    }
    let metadata = object
        .entry("metadata")
        .or_insert_with(|| serde_json::json!({}));
    let metadata = metadata
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("OCR page metadata must be a JSON object"))?;
    metadata
        .entry("ocr_provider".to_string())
        .or_insert_with(|| serde_json::json!(provider_name));
    if !metadata.contains_key("source_image_sha256") {
        metadata.insert(
            "source_image_sha256".to_string(),
            serde_json::json!(sha256_file(image_path)?),
        );
    }
    if let Some(error) = artifact_schema_error(&artifact) {
        anyhow::bail!(error);
    }
    Ok(artifact)
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

pub fn run_ocr_source_workflow(
    source_path: &Path,
    output_dir: &Path,
    runner: &dyn OcrRunner,
    rasterizer: Option<&dyn PdfRasterizer>,
    probe_path: &Path,
) -> Result<WorkflowCommandResult> {
    std::fs::create_dir_all(output_dir)?;
    let document_id = document_stem(source_path);
    let extension = source_path
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .map(|extension| format!(".{}", extension.to_lowercase()))
        .unwrap_or_default();
    let mut page_sources = Vec::new();
    let (source_kind, render_configuration_fingerprint) = if extension == ".pdf" {
        if !source_path.is_file() {
            anyhow::bail!("OCR source must be an existing image or PDF");
        }
        let Some(rasterizer) = rasterizer else {
            anyhow::bail!("PDF OCR workflow requires a PDF rasterizer");
        };
        let rendered_dir = output_dir.join("rendered_pages");
        std::fs::create_dir_all(&rendered_dir)?;
        let rendered_paths = rasterizer.render_pdf(source_path, &rendered_dir)?;
        if rendered_paths.is_empty() {
            anyhow::bail!("PDF rasterizer produced no pages");
        }
        for (index, image_path) in rendered_paths.iter().enumerate() {
            let image_extension = image_path
                .extension()
                .and_then(std::ffi::OsStr::to_str)
                .map(|extension| format!(".{}", extension.to_lowercase()))
                .unwrap_or_default();
            if !OCR_IMAGE_SUFFIXES.contains(&image_extension.as_str()) {
                anyhow::bail!("unsupported rendered OCR image: {}", image_path.display());
            }
            page_sources.push(((index + 1) as u64, image_path.clone()));
        }
        ("pdf", rasterizer.identity().to_string())
    } else {
        if !OCR_IMAGE_SUFFIXES.contains(&extension.as_str()) {
            anyhow::bail!("OCR source must be an image or PDF");
        }
        if !source_path.is_file() {
            anyhow::bail!("OCR source must be an existing image or PDF");
        }
        page_sources.push((1, source_path.to_path_buf()));
        ("image", "image-materialize-v1".to_string())
    };
    let rendered_dir = output_dir.join("rendered_pages");
    let input_fingerprint = sha256_file(source_path)?;
    let legacy_dir = output_dir.join("legacy_split_pages");
    std::fs::create_dir_all(&legacy_dir)?;
    probe_event(
        probe_path,
        "workflow.file_started",
        serde_json::json!({
            "workflow_kind": "ocr",
            "source_path": source_path,
            "output_dir": output_dir,
            "document_id": document_id,
        }),
    )?;
    let mut pages = Vec::with_capacity(page_sources.len());
    for (page_number, image_path) in page_sources {
        let artifact = runner.run_page(&image_path, page_number)?;
        let artifact =
            validate_ocr_artifact(artifact, &image_path, page_number, runner.provider_name())?;
        let artifact_path = legacy_dir.join(format!("page_{page_number}.json"));
        std::fs::write(&artifact_path, serde_json::to_string_pretty(&artifact)?)?;
        pages.push(serde_json::json!({
            "page_number": page_number,
            "image_path": image_path,
            "json_path": artifact_path,
            "status": "completed",
        }));
    }
    let progress = serde_json::json!({
        "document_id": document_id,
        "title": document_id,
        "source_kind": source_kind,
        "input_fingerprint": input_fingerprint,
        "render_configuration_fingerprint": render_configuration_fingerprint,
        "total_pages": pages.len(),
        "pages": pages,
    });
    let progress_path = output_dir.join("ocr-progress.json");
    std::fs::write(&progress_path, serde_json::to_string_pretty(&progress)?)?;
    let summary = serde_json::json!({
        "kind": "ocr",
        "source_path": source_path,
        "document_id": document_id,
        "title": document_id,
        "source_kind": source_kind,
        "input_fingerprint": input_fingerprint,
        "page_count": pages.len(),
        "render_configuration_fingerprint": render_configuration_fingerprint,
        "legacy_dir": legacy_dir,
        "rendered_dir": rendered_dir,
        "progress_path": progress_path,
    });
    let summary_path = output_dir.join("ocr-summary.json");
    std::fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;
    probe_event(
        probe_path,
        "workflow.file_finished",
        serde_json::json!({
            "workflow_kind": "ocr",
            "source_path": source_path,
            "output_dir": output_dir,
            "document_id": document_id,
            "status": "succeeded",
            "summary_path": summary_path,
        }),
    )?;
    Ok(WorkflowCommandResult {
        kind: "ocr".to_string(),
        input_path: source_path.to_path_buf(),
        output_dir: output_dir.to_path_buf(),
        status: "succeeded".to_string(),
        probe_path: probe_path.to_path_buf(),
        summary_path,
        extra: summary,
    })
}

pub fn run_ocr_batch_workflow(
    source_paths: &[PathBuf],
    output_dir: &Path,
    runner: &dyn OcrRunner,
    rasterizer: Option<&dyn PdfRasterizer>,
) -> Result<Vec<WorkflowCommandResult>> {
    std::fs::create_dir_all(output_dir)?;
    let probe_path = output_dir.join("workflow-events.jsonl");
    let files = discover_input_files(source_paths, &OCR_FILE_SUFFIXES, true)?;
    let mut stem_counts: BTreeMap<String, usize> = BTreeMap::new();
    for source_path in &files {
        let stem = document_stem(source_path);
        *stem_counts.entry(stem).or_insert(0) += 1;
    }
    probe_event(
        &probe_path,
        "workflow.batch_started",
        serde_json::json!({
            "workflow_kind": "ocr",
            "file_count": files.len(),
            "output_dir": output_dir,
        }),
    )?;
    let results = files
        .iter()
        .map(|source_path| {
            let batch_identity = collision_safe_batch_identity(source_path, &stem_counts);
            run_ocr_source_workflow(
                source_path,
                &output_dir.join(&batch_identity),
                runner,
                rasterizer,
                &probe_path,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    probe_event(
        &probe_path,
        "workflow.batch_finished",
        serde_json::json!({
            "workflow_kind": "ocr",
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
    page_files.sort_by_key(|(left, _)| *left);
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

    #[test]
    fn runs_ocr_image_source_workflow() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("sample.png");
        std::fs::write(&source, b"fake image").unwrap();
        let runner = |_image_path: &Path, page_number: u64| -> Result<Value> {
            Ok(serde_json::json!({
                "metadata": {"ocr_json_version": "0.1", "ocr_model_name": "fake"},
                "pdf_page_num": page_number,
                "OCR_text_clusters": [{"text": format!("page {page_number}")}],
                "non_text_objects": []
            }))
        };
        let output = temp.path().join("single");
        let result = run_ocr_source_workflow(
            &source,
            &output,
            &runner,
            None,
            &output.join("workflow-events.jsonl"),
        )
        .unwrap();
        assert_eq!(result.status, "succeeded");
        assert_eq!(result.extra["page_count"], 1);
        assert!(output.join("legacy_split_pages/page_1.json").exists());
        assert!(output.join("ocr-summary.json").exists());
        assert!(output.join("ocr-progress.json").exists());
    }

    #[test]
    fn runs_ocr_pdf_and_batch_workflows() {
        let temp = tempfile::tempdir().unwrap();
        let pdf = temp.path().join("report.pdf");
        std::fs::write(&pdf, b"fake pdf").unwrap();
        let runner = |_image_path: &Path, page_number: u64| -> Result<Value> {
            Ok(serde_json::json!({
                "metadata": {"ocr_json_version": "0.1", "ocr_model_name": "fake"},
                "pdf_page_num": page_number,
                "OCR_text_clusters": [{"text": format!("page {page_number}")}],
                "non_text_objects": []
            }))
        };
        let rasterizer = |_pdf_path: &Path, rendered_dir: &Path| -> Result<Vec<PathBuf>> {
            let first = rendered_dir.join("page_1.png");
            let second = rendered_dir.join("page_2.png");
            std::fs::write(&first, b"one").unwrap();
            std::fs::write(&second, b"two").unwrap();
            Ok(vec![first, second])
        };
        let output = temp.path().join("single");
        let result = run_ocr_source_workflow(
            &pdf,
            &output,
            &runner,
            Some(&rasterizer),
            &output.join("workflow-events.jsonl"),
        )
        .unwrap();
        assert_eq!(result.extra["page_count"], 2);
        assert!(output.join("legacy_split_pages/page_1.json").exists());
        assert!(output.join("legacy_split_pages/page_2.json").exists());

        let batch_input = temp.path().join("batch-inputs");
        std::fs::create_dir_all(&batch_input).unwrap();
        std::fs::write(batch_input.join("sample.png"), b"one").unwrap();
        std::fs::write(batch_input.join("sample.jpg"), b"two").unwrap();
        let batch_output = temp.path().join("batch");
        let results = run_ocr_batch_workflow(&[batch_input], &batch_output, &runner, None).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|result| result.status == "succeeded"));
    }
}
