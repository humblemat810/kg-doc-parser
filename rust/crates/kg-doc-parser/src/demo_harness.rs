use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

use crate::semantic::{HydratedTextPointer, SemanticNode};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DemoParserMode {
    #[default]
    FakeLayered,
    LegacyCached,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DemoServerMode {
    #[default]
    TestClient,
    SubprocessHttp,
    ExternalHttp,
}

#[derive(Debug, Clone)]
pub struct DemoHarnessConfig {
    pub output_dir: PathBuf,
    pub document_id: String,
    pub title: String,
    pub text: String,
    pub workflow_id: String,
    pub parser_mode: DemoParserMode,
    pub server_mode: DemoServerMode,
    pub external_base_url: Option<String>,
    pub enable_sys_monitoring: bool,
    pub probe_filename: String,
    pub summary_filename: String,
    pub cache_dirname: String,
}

impl Default for DemoHarnessConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("."),
            document_id: "demo-doc".to_string(),
            title: "Workflow Ingest Demo".to_string(),
            text: "Alpha clause\nBeta clause\nGamma clause".to_string(),
            workflow_id: "kg_doc_parser.ingest.v1".to_string(),
            parser_mode: DemoParserMode::FakeLayered,
            server_mode: DemoServerMode::TestClient,
            external_base_url: None,
            enable_sys_monitoring: true,
            probe_filename: "probe-events.jsonl".to_string(),
            summary_filename: "demo-summary.json".to_string(),
            cache_dirname: "llm-cache".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DemoHarnessArtifacts {
    pub output_dir: PathBuf,
    pub probe_path: PathBuf,
    pub summary_path: PathBuf,
    pub cache_dir: PathBuf,
    pub engine_dir: PathBuf,
    pub server_data_dir: PathBuf,
    pub run_id: Option<String>,
    pub status: Option<String>,
    pub canonical_write_confirmed: bool,
    pub persistence_mode: Option<String>,
    pub kg_authority: Option<String>,
}

pub const DEMO_JWT_SECRET: &str = "kg-doc-parser-demo-test-secret";

pub fn validate_demo_config(config: &DemoHarnessConfig) -> Result<()> {
    if config.output_dir.as_os_str().is_empty() {
        bail!("output_dir is required");
    }
    if config.document_id.trim().is_empty() {
        bail!("document_id must be non-empty");
    }
    if config.workflow_id.trim().is_empty() {
        bail!("workflow_id must be non-empty");
    }
    if config.server_mode == DemoServerMode::ExternalHttp
        && config
            .external_base_url
            .as_deref()
            .map(str::trim)
            .is_none_or(str::is_empty)
    {
        bail!("external_base_url is required when server_mode='external_http'");
    }
    Ok(())
}

pub fn prepare_demo_artifacts(config: &DemoHarnessConfig) -> Result<DemoHarnessArtifacts> {
    validate_demo_config(config)?;
    let output_dir = config.output_dir.clone();
    std::fs::create_dir_all(&output_dir)?;
    let cache_dir = output_dir.join(&config.cache_dirname);
    let engine_dir = output_dir.join("engines");
    let server_data_dir = output_dir.join("server-data");
    std::fs::create_dir_all(&cache_dir)?;
    std::fs::create_dir_all(&engine_dir)?;
    std::fs::create_dir_all(&server_data_dir)?;
    Ok(DemoHarnessArtifacts {
        output_dir: output_dir.clone(),
        probe_path: output_dir.join(&config.probe_filename),
        summary_path: output_dir.join(&config.summary_filename),
        cache_dir,
        engine_dir,
        server_data_dir,
        run_id: None,
        status: None,
        canonical_write_confirmed: false,
        persistence_mode: None,
        kg_authority: None,
    })
}

pub fn seed_demo_document_payload(document_id: &str, text: &str) -> Value {
    json!({
        "doc_id": document_id,
        "doc_type": "text",
        "insertion_method": "demo_harness",
        "content": text,
    })
}

pub fn demo_summary_payload(
    artifacts: &DemoHarnessArtifacts,
    config: &DemoHarnessConfig,
    proposal_mode: &str,
) -> Value {
    json!({
        "run_id": artifacts.run_id,
        "status": artifacts.status,
        "canonical_write_confirmed": artifacts.canonical_write_confirmed,
        "persistence_mode": artifacts.persistence_mode,
        "kg_authority": artifacts.kg_authority,
        "probe_path": artifacts.probe_path,
        "cache_dir": artifacts.cache_dir,
        "engine_dir": artifacts.engine_dir,
        "server_data_dir": artifacts.server_data_dir,
        "proposal_mode": proposal_mode,
        "parser_mode": config.parser_mode,
        "server_mode": config.server_mode,
    })
}

pub fn write_demo_summary(
    artifacts: &DemoHarnessArtifacts,
    config: &DemoHarnessConfig,
    proposal_mode: &str,
) -> Result<()> {
    let payload = demo_summary_payload(artifacts, config, proposal_mode);
    std::fs::write(
        &artifacts.summary_path,
        serde_json::to_string_pretty(&payload)?,
    )?;
    Ok(())
}

fn pointer(text: &str, unit_id: &str, fragment: &str) -> Result<HydratedTextPointer> {
    let start = text
        .find(fragment)
        .ok_or_else(|| anyhow::anyhow!("demo fragment is missing from source text"))?;
    let end = start + fragment.chars().count() - 1;
    Ok(HydratedTextPointer {
        source_cluster_id: unit_id.to_string(),
        start_char: start as i64,
        end_char: end as i64,
        verbatim_text: fragment.to_string(),
        start_delimiter: None,
        end_delimiter: None,
        validation_method: None,
    })
}

fn semantic_child(
    node_id: &str,
    title: &str,
    pointers: Vec<HydratedTextPointer>,
    children: Vec<SemanticNode>,
    level_from_root: u64,
) -> SemanticNode {
    SemanticNode {
        node_id: Some(node_id.to_string()),
        parent_id: None,
        node_type: "TEXT_FLOW".to_string(),
        title: title.to_string(),
        summary: String::new(),
        total_content_pointers: pointers,
        aggregate_content_pointers: Vec::new(),
        child_nodes: children,
        level_from_root,
        metadata: json!({}),
    }
}

pub fn fake_layered_tree(request_id: &str, title: &str, text: &str) -> Result<SemanticNode> {
    let unit_id = format!("{request_id}|p1_t0");
    let lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    let overview_pointers = vec![pointer(text, &unit_id, text)?];
    let clauses = lines
        .iter()
        .enumerate()
        .map(|(index, line)| {
            Ok::<_, anyhow::Error>(semantic_child(
                &format!("{request_id}|clause|{index}"),
                line,
                vec![pointer(text, &unit_id, line)?],
                Vec::new(),
                2,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let overview = semantic_child(
        &format!("{request_id}|section|overview"),
        "Overview",
        overview_pointers,
        clauses,
        1,
    );
    Ok(SemanticNode {
        node_id: Some(format!("{request_id}|root")),
        parent_id: None,
        node_type: "DOCUMENT_ROOT".to_string(),
        title: title.to_string(),
        summary: String::new(),
        total_content_pointers: Vec::new(),
        aggregate_content_pointers: Vec::new(),
        child_nodes: vec![overview],
        level_from_root: 0,
        metadata: json!({}),
    })
}

pub fn demo_event(kind: &str, payload: Value) -> Value {
    json!({"event": kind, "payload": payload})
}

pub fn append_demo_probe_event(path: &Path, kind: &str, payload: Value) -> Result<()> {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(
        file,
        "{}",
        serde_json::to_string(&demo_event(kind, payload))?
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn validates_demo_config_and_prepares_artifacts() {
        let temp = tempfile::tempdir().unwrap();
        let mut config = DemoHarnessConfig {
            output_dir: temp.path().join("demo"),
            server_mode: DemoServerMode::ExternalHttp,
            ..Default::default()
        };
        assert!(validate_demo_config(&config).is_err());
        config.external_base_url = Some("http://127.0.0.1:9000".to_string());
        let artifacts = prepare_demo_artifacts(&config).unwrap();
        assert!(artifacts.cache_dir.exists());
        assert!(artifacts.engine_dir.exists());
        assert!(artifacts.server_data_dir.exists());
    }

    #[test]
    fn builds_fake_layered_tree_with_grounding() {
        let tree = fake_layered_tree("request", "Demo", "Alpha clause\nBeta clause").unwrap();
        assert_eq!(tree.node_type, "DOCUMENT_ROOT");
        assert_eq!(tree.child_nodes.len(), 1);
        let overview = &tree.child_nodes[0];
        assert_eq!(overview.title, "Overview");
        assert_eq!(overview.child_nodes.len(), 2);
        assert_eq!(overview.child_nodes[0].title, "Alpha clause");
        assert_eq!(
            overview.child_nodes[0].total_content_pointers[0].verbatim_text,
            "Alpha clause"
        );
    }

    #[test]
    fn writes_probe_and_summary_artifacts() {
        let temp = tempfile::tempdir().unwrap();
        let config = DemoHarnessConfig {
            output_dir: temp.path().join("demo"),
            ..Default::default()
        };
        let mut artifacts = prepare_demo_artifacts(&config).unwrap();
        append_demo_probe_event(
            &artifacts.probe_path,
            "demo.started",
            json!({"document_id": config.document_id}),
        )
        .unwrap();
        artifacts.run_id = Some("run-1".to_string());
        artifacts.status = Some("succeeded".to_string());
        artifacts.canonical_write_confirmed = true;
        write_demo_summary(&artifacts, &config, "children").unwrap();
        let probe = std::fs::read_to_string(&artifacts.probe_path).unwrap();
        let summary: Value =
            serde_json::from_str(&std::fs::read_to_string(&artifacts.summary_path).unwrap())
                .unwrap();
        assert!(probe.contains("demo.started"));
        assert_eq!(summary["run_id"], "run-1");
        assert_eq!(summary["canonical_write_confirmed"], true);
        assert_eq!(seed_demo_document_payload("doc", "text")["doc_id"], "doc");
    }
}
