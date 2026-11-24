use anyhow::{bail, Result};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

use crate::document_models::DocumentTree;
use crate::ocr_state_store::{OcrStage, OcrWorkflowStateStore};

pub const OCR_ROOT_STATE_VERSION: &str = "ocr-root-v3";
pub const OCR_PAGE_STATE_VERSION: &str = "ocr-page-v2";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageSourceState {
    pub page_number: u64,
    pub image_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResumeIdentity {
    String(String),
    Strings(Vec<String>),
}

pub fn page_source_state(value: &Value) -> Result<PageSourceState> {
    let object = value.as_object().ok_or_else(|| {
        anyhow::anyhow!("OCR workflow state page_sources contains a malformed entry")
    })?;
    let Some(page_number) = object.get("page_number").and_then(Value::as_u64) else {
        bail!("OCR workflow state page_sources contains a malformed entry");
    };
    let Some(image_path) = object.get("image_path").and_then(Value::as_str) else {
        bail!("OCR workflow state page_sources contains a malformed entry");
    };
    if image_path.is_empty() {
        bail!("OCR workflow state page_sources contains a malformed entry");
    }
    Ok(PageSourceState {
        page_number,
        image_path: image_path.to_string(),
    })
}

pub fn page_sources(state: &Value) -> Result<Vec<PageSourceState>> {
    let Some(items) = state.get("page_sources").and_then(Value::as_array) else {
        bail!("OCR page runtime state has no page source list");
    };
    items
        .iter()
        .map(page_source_state)
        .collect::<Result<Vec<_>>>()
        .and_then(|sources| {
            if sources.iter().any(|source| source.page_number < 1) {
                bail!("OCR page runtime source page number is malformed");
            }
            Ok(sources)
        })
}

pub fn int_list(state: &Value, key: &str) -> Result<Vec<i64>> {
    let Some(Value::Array(items)) = state.get(key) else {
        bail!("OCR page runtime state field {key:?} is malformed");
    };
    if items.iter().any(|item| !item.is_i64()) {
        bail!("OCR page runtime state field {key:?} is malformed");
    }
    Ok(items.iter().filter_map(Value::as_i64).collect())
}

pub fn state_bool(state: &Value, key: &str, default: bool) -> bool {
    state.get(key).and_then(Value::as_bool).unwrap_or(default)
}

pub fn state_string(state: &Value, key: &str, default: &str) -> String {
    state
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or(default)
        .to_string()
}

pub fn page_cursor(state: &Value) -> u64 {
    state
        .get("page_cursor")
        .and_then(Value::as_u64)
        .unwrap_or(0)
}

pub fn page_workflow_initial_state(
    source_fingerprint: &str,
    legacy_dir: &std::path::Path,
) -> Value {
    json!({
        "ocr_page_state_version": OCR_PAGE_STATE_VERSION,
        "source_fingerprint": source_fingerprint,
        "legacy_dir": legacy_dir,
        "page_cursor": 0,
    })
}

pub fn resume_identity_value(state: &Value, key: &str) -> Result<ResumeIdentity> {
    let Some(value) = state.get(key) else {
        bail!("OCR checkpoint identity field {key:?} is missing");
    };
    if let Some(value) = value.as_str() {
        return Ok(ResumeIdentity::String(value.to_string()));
    }
    if key == "ocr_configuration_fingerprints" {
        if let Some(values) = value.as_array() {
            if values.iter().all(Value::is_string) {
                return Ok(ResumeIdentity::Strings(
                    values
                        .iter()
                        .filter_map(Value::as_str)
                        .map(ToString::to_string)
                        .collect(),
                ));
            }
        }
    }
    bail!("OCR checkpoint identity field {key:?} has an invalid type")
}

fn validate_resume_identity(
    state: &Value,
    expected: &[(String, ResumeIdentity)],
    workflow_kind: &str,
) -> Result<()> {
    let mut missing = expected
        .iter()
        .filter(|(key, _)| state.get(key).is_none())
        .map(|(key, _)| key.as_str())
        .collect::<Vec<_>>();
    missing.sort_unstable();
    if !missing.is_empty() {
        bail!(
            "cannot resume OCR {workflow_kind} checkpoint without immutable identity fields: {}",
            missing.join(", ")
        );
    }
    let mut mismatches = expected
        .iter()
        .filter(
            |(key, expected_value)| match resume_identity_value(state, key) {
                Ok(value) => &value != expected_value,
                Err(_) => true,
            },
        )
        .map(|(key, _)| key.as_str())
        .collect::<Vec<_>>();
    mismatches.sort_unstable();
    if !mismatches.is_empty() {
        bail!(
            "OCR {workflow_kind} checkpoint identity mismatch; refusing to mix persisted state for: {}",
            mismatches.join(", ")
        );
    }
    Ok(())
}

pub struct RootResumeIdentity {
    pub document_id: String,
    pub source_fingerprint: String,
    pub source_namespace: String,
    pub configuration_fingerprints: Vec<String>,
    pub grouping_policy_version: String,
    pub parse_generation: String,
}

pub fn validate_root_resume_identity(
    state: &Value,
    identity: &RootResumeIdentity,
    require_match: bool,
) -> Result<()> {
    if !require_match {
        return Ok(());
    }
    validate_resume_identity(
        state,
        &[
            (
                "ocr_root_state_version".to_string(),
                ResumeIdentity::String(OCR_ROOT_STATE_VERSION.to_string()),
            ),
            (
                "document_id".to_string(),
                ResumeIdentity::String(identity.document_id.clone()),
            ),
            (
                "source_fingerprint".to_string(),
                ResumeIdentity::String(identity.source_fingerprint.clone()),
            ),
            (
                "source_namespace".to_string(),
                ResumeIdentity::String(identity.source_namespace.clone()),
            ),
            (
                "ocr_configuration_fingerprints".to_string(),
                ResumeIdentity::Strings(identity.configuration_fingerprints.clone()),
            ),
            (
                "grouping_policy_version".to_string(),
                ResumeIdentity::String(identity.grouping_policy_version.clone()),
            ),
            (
                "parse_generation".to_string(),
                ResumeIdentity::String(identity.parse_generation.clone()),
            ),
        ],
        "root",
    )
}

pub struct PageResumeIdentity {
    pub document_id: String,
    pub source_fingerprint: String,
    pub configuration_fingerprints: Vec<String>,
}

pub fn validate_page_resume_identity(state: &Value, identity: &PageResumeIdentity) -> Result<()> {
    validate_resume_identity(
        state,
        &[
            (
                "ocr_page_state_version".to_string(),
                ResumeIdentity::String(OCR_PAGE_STATE_VERSION.to_string()),
            ),
            (
                "document_id".to_string(),
                ResumeIdentity::String(identity.document_id.clone()),
            ),
            (
                "source_fingerprint".to_string(),
                ResumeIdentity::String(identity.source_fingerprint.clone()),
            ),
            (
                "ocr_configuration_fingerprints".to_string(),
                ResumeIdentity::Strings(identity.configuration_fingerprints.clone()),
            ),
        ],
        "page",
    )
}

pub struct PageReconciliationSources {
    pub page_sources: Vec<PageSourceState>,
    pub legacy_dir: std::path::PathBuf,
}

pub struct PageReconciliationRequest<'a> {
    pub document_id: &'a str,
    pub page_number: u64,
    pub image_digest: &'a str,
    pub artifact_path: &'a std::path::Path,
}

pub trait PageArtifactGate {
    fn accepts(&self, request: &PageReconciliationRequest) -> bool;
}

pub struct ReadableArtifactGate;

impl PageArtifactGate for ReadableArtifactGate {
    fn accepts(&self, request: &PageReconciliationRequest) -> bool {
        std::fs::read(request.artifact_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
            .map(|payload| {
                payload.get("pdf_page_num").and_then(Value::as_u64) == Some(request.page_number)
                    || payload
                        .get("pdf_page_num")
                        .and_then(Value::as_str)
                        .and_then(|value| value.trim().parse::<u64>().ok())
                        == Some(request.page_number)
            })
            .unwrap_or(false)
    }
}

fn validated_page_json(
    value: &Value,
    default_page_number: Option<u64>,
    _strict_page_number_types: bool,
) -> Result<Value> {
    let object = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("OCR page payload must be a JSON object"))?;
    let raw_page_number = object.get("pdf_page_num").unwrap_or(&Value::Null);
    if let Some(page_number) = raw_page_number.as_u64() {
        if page_number < 1 {
            bail!("OCR page pdf_page_num must be a positive integer");
        }
    } else if let Some(page_number) = raw_page_number.as_str() {
        if page_number.trim().parse::<u64>().is_err() {
            bail!("OCR page pdf_page_num must be a positive integer");
        }
    } else if raw_page_number.is_null() {
        if default_page_number.unwrap_or(0) < 1 {
            bail!("OCR page pdf_page_num must be a positive integer");
        }
    } else {
        bail!("OCR page pdf_page_num must be a positive integer");
    }
    for key in ["printed_page_number", "spatial_accuracy", "text"] {
        if let Some(value) = object.get(key) {
            if !value.is_null() && !value.is_string() {
                bail!("OCR page field {key:?} has an invalid type");
            }
        }
    }
    for key in ["contains_table", "is_blank_page"] {
        if let Some(value) = object.get(key) {
            if !value.is_null() && !value.is_boolean() {
                bail!("OCR page field {key:?} has an invalid type");
            }
        }
    }
    for key in ["unresolved_item_count", "source_item_count"] {
        if let Some(value) = object.get(key) {
            if value.as_u64().is_none() {
                bail!("OCR page field {key:?} must be a non-negative integer");
            }
        }
    }
    for key in [
        "page_x_min",
        "page_x_max",
        "page_y_min",
        "page_y_max",
        "data_loss_likelihood",
    ] {
        if let Some(value) = object.get(key) {
            if !value.is_null() && value.as_f64().map(|number| number.is_finite()) != Some(true) {
                bail!("OCR page field {key:?} must be a finite number");
            }
        }
    }
    for key in [
        "meaningful_ordering",
        "OCR_text_clusters",
        "non_text_objects",
    ] {
        if let Some(value) = object.get(key) {
            if !value.is_null() && !value.is_array() {
                bail!("OCR page field {key:?} must be a list");
            }
        }
    }
    Ok(value.clone())
}

pub fn json_pages(
    state: &Value,
    sources: Option<&[PageSourceState]>,
    legacy_dir: Option<&std::path::Path>,
) -> Result<Vec<Value>> {
    if let Some(raw_pages) = state.get("raw_pages") {
        let Some(items) = raw_pages.as_array() else {
            bail!("OCR page runtime raw page state is malformed");
        };
        return items
            .iter()
            .map(|item| validated_page_json(item, None, true))
            .collect();
    }
    let legacy_dir = state
        .get("legacy_dir")
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .or_else(|| legacy_dir.map(PathBuf::from));
    let Some(legacy_dir) = legacy_dir else {
        bail!("OCR page runtime state has no legacy artifact directory");
    };
    let Some(sources) = sources else {
        bail!("OCR page runtime state has no page source list");
    };
    let cursor = page_cursor(state) as usize;
    let mut pages = Vec::new();
    for source in sources.iter().take(cursor) {
        let path = legacy_dir.join(format!("page_{}.json", source.page_number));
        let bytes = std::fs::read(&path).map_err(|error| {
            anyhow::anyhow!("OCR page artifact is not readable: {path:?}: {error}")
        })?;
        let payload = serde_json::from_slice::<Value>(&bytes).map_err(|error| {
            anyhow::anyhow!("OCR page artifact is not readable: {path:?}: {error}")
        })?;
        pages.push(validated_page_json(
            &payload,
            Some(source.page_number),
            false,
        )?);
    }
    Ok(pages)
}

pub fn reconcile_page_checkpoint<G: PageArtifactGate>(
    state: &Value,
    sources: Option<&[PageSourceState]>,
    legacy_dir: &std::path::Path,
    gate: &G,
    document_id: &str,
    start_index: usize,
) -> Result<Option<Value>> {
    let cursor = page_cursor(state) as usize;
    let sources = if state.get("page_sources").is_some() {
        page_sources(state)?
    } else {
        sources
            .ok_or_else(|| anyhow::anyhow!("OCR page runtime state has no page source list"))?
            .to_vec()
    };
    let completed = if state.get("completed_pages").is_some() {
        Some(int_list(state, "completed_pages")?)
    } else {
        None
    };
    let reused = if state.get("reused_pages").is_some() {
        Some(int_list(state, "reused_pages")?)
    } else {
        None
    };
    if start_index > cursor {
        bail!("OCR checkpoint reconciliation start index is out of range");
    }
    for index in start_index..cursor {
        let Some(source) = sources.get(index) else {
            bail!("OCR page runtime state has no page source list");
        };
        let image_bytes = std::fs::read(&source.image_path).map_err(|error| {
            anyhow::anyhow!(
                "OCR page image is not readable: {:?}: {error}",
                source.image_path
            )
        })?;
        let mut digest = Sha256::new();
        digest.update(image_bytes);
        let image_digest = digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let artifact_path = legacy_dir.join(format!("page_{}.json", source.page_number));
        let request = PageReconciliationRequest {
            document_id,
            page_number: source.page_number,
            image_digest: &image_digest,
            artifact_path: &artifact_path,
        };
        if gate.accepts(&request) {
            continue;
        }
        let mut repair = Map::new();
        repair.insert("page_cursor".to_string(), json!(index));
        if let Some(completed) = &completed {
            repair.insert(
                "completed_pages".to_string(),
                json!(completed.iter().take(index).cloned().collect::<Vec<_>>()),
            );
        }
        if let (Some(reused), Some(completed)) = (&reused, &completed) {
            let completed_prefix = completed
                .iter()
                .take(index)
                .cloned()
                .collect::<std::collections::HashSet<i64>>();
            repair.insert(
                "reused_pages".to_string(),
                json!(reused
                    .iter()
                    .take(index)
                    .filter(|number| completed_prefix.contains(*number))
                    .cloned()
                    .collect::<Vec<_>>()),
            );
        }
        if state.get("raw_pages").is_some() {
            let pages = json_pages(state, Some(&sources), Some(legacy_dir))?;
            repair.insert(
                "raw_pages".to_string(),
                json!(pages.into_iter().take(index).collect::<Vec<_>>()),
            );
        }
        return Ok(Some(Value::Object(repair)));
    }
    Ok(None)
}

fn expected_page_numbers(sources: &[PageSourceState]) -> Vec<u64> {
    sources.iter().map(|source| source.page_number).collect()
}

fn root_page_checkpoint_incomplete<G: PageArtifactGate>(
    state: &Value,
    sources: &[PageSourceState],
    legacy_dir: &std::path::Path,
    gate: &G,
    document_id: &str,
) -> Result<bool> {
    let cursor = page_cursor(state);
    let expected_pages = expected_page_numbers(sources);
    let legacy_completed = if state.get("completed_pages").is_some() {
        int_list(state, "completed_pages")?
            .into_iter()
            .map(|page| page as u64)
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let completed = if legacy_completed.is_empty() {
        expected_pages
            .iter()
            .take(cursor as usize)
            .cloned()
            .collect::<Vec<_>>()
    } else {
        legacy_completed
    };
    let incomplete = cursor != sources.len() as u64 || completed != expected_pages;
    let checkpoint_broken =
        reconcile_page_checkpoint(state, Some(sources), legacy_dir, gate, document_id, 0)?
            .is_some();
    Ok(incomplete || checkpoint_broken)
}

pub fn validate_root_page_result<G: PageArtifactGate>(
    state: &Value,
    sources: &[PageSourceState],
    legacy_dir: &std::path::Path,
    gate: &G,
    document_id: &str,
) -> Result<()> {
    if root_page_checkpoint_incomplete(state, sources, legacy_dir, gate, document_id)? {
        bail!("OCR root page child completed with an incomplete page checkpoint");
    }
    Ok(())
}

pub fn page_result_needs_recovery<G: PageArtifactGate>(
    state: &Value,
    sources: &[PageSourceState],
    legacy_dir: &std::path::Path,
    gate: &G,
    document_id: &str,
) -> Result<bool> {
    root_page_checkpoint_incomplete(state, sources, legacy_dir, gate, document_id)
}

pub fn validate_existing_page_identities(
    page_states: &std::collections::BTreeMap<u64, Option<String>>,
    configuration_fingerprints: &[String],
) -> Result<()> {
    for (page_number, config_fingerprint) in page_states {
        let Some(config_fingerprint) = config_fingerprint else {
            continue;
        };
        if configuration_fingerprints.contains(config_fingerprint) {
            continue;
        }
        bail!(
            "OCR root checkpoint identity mismatch; refusing to mix persisted OCR configuration for page {page_number}"
        );
    }
    Ok(())
}

pub struct RootArtifactRecoveryRequest<'a> {
    pub state_store: &'a OcrWorkflowStateStore,
    pub document_id: &'a str,
    pub sources: &'a [PageSourceState],
    pub legacy_dir: &'a std::path::Path,
    pub configuration_fingerprints: &'a [String],
}

pub fn root_page_artifacts_need_recovery(
    request: &RootArtifactRecoveryRequest<'_>,
) -> Result<bool> {
    let configuration_fingerprints = request
        .configuration_fingerprints
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    for source in request.sources {
        let image_digest = Sha256::digest(std::fs::read(&source.image_path)?)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let page_path = request
            .legacy_dir
            .join(format!("page_{}.json", source.page_number));
        request.state_store.reconcile_orphan_artifact(
            request.document_id,
            source.page_number as i64,
            &image_digest,
            &page_path,
            &configuration_fingerprints,
        )?;
        if !request.state_store.should_skip_page(
            request.document_id,
            source.page_number as i64,
            OcrStage::Ocr,
            &image_digest,
            Some(&page_path),
            &configuration_fingerprints,
        )? {
            return Ok(true);
        }
    }
    Ok(false)
}

pub fn recovery_generation(state: &Value) -> Result<u64> {
    match state.get("ocr_page_recovery_generation") {
        None | Some(Value::Null) => Ok(0),
        Some(value) => {
            let Some(generation) = value.as_u64() else {
                bail!("OCR page recovery generation must be a non-negative integer");
            };
            Ok(generation)
        }
    }
}

pub fn semantic_parse_generation(recovery_generation: u64) -> String {
    if recovery_generation == 0 {
        "semantic-v1".to_string()
    } else {
        format!("semantic-recovery-v{recovery_generation}")
    }
}

pub fn next_root_recovery_run_id<F>(
    base_run_id: &str,
    mut completed_exists: F,
) -> Result<(String, u64)>
where
    F: FnMut(&str) -> Result<bool>,
{
    let mut generation = 1;
    loop {
        let candidate = format!("{base_run_id}|recovery-{generation}");
        if !completed_exists(&candidate)? {
            return Ok((candidate, generation));
        }
        generation += 1;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvocationLineage {
    pub workflow_id: String,
    pub invocation_key: String,
    pub child_run_id: String,
}

pub fn expected_invocation_lineages(
    root_run_id: &str,
    recovery_generation: u64,
) -> Vec<InvocationLineage> {
    let suffix = if recovery_generation == 0 {
        String::new()
    } else {
        format!("|recovery-{recovery_generation}")
    };
    let invocation_key = |base: &str| {
        if recovery_generation == 0 {
            base.to_string()
        } else {
            format!("{base}-recovery-v{recovery_generation}")
        }
    };
    vec![
        InvocationLineage {
            workflow_id: "kg_doc_parser.ocr_pages.v1".to_string(),
            invocation_key: invocation_key("ocr-pages-v1"),
            child_run_id: format!("{root_run_id}|ocr-pages{suffix}"),
        },
        InvocationLineage {
            workflow_id: "kg_doc_parser.ocr_grouping.v1".to_string(),
            invocation_key: invocation_key("grouping-v1"),
            child_run_id: format!("{root_run_id}|grouping{suffix}"),
        },
        InvocationLineage {
            workflow_id: "kg_doc_parser.ingest.v1".to_string(),
            invocation_key: invocation_key("semantic-v1"),
            child_run_id: format!("{root_run_id}|semantic{suffix}"),
        },
    ]
}

fn invocation_lineage_key(lineage: &InvocationLineage) -> (String, String, String) {
    (
        lineage.workflow_id.clone(),
        lineage.invocation_key.clone(),
        lineage.child_run_id.clone(),
    )
}

pub fn validate_invocation_lineages(
    edges: &[Value],
    root_run_id: &str,
    recovery_generation: u64,
) -> Result<()> {
    let expected = expected_invocation_lineages(root_run_id, recovery_generation);
    let expected_keys = expected
        .iter()
        .map(invocation_lineage_key)
        .collect::<std::collections::HashSet<_>>();
    let mut counts: std::collections::HashMap<(String, String, String), usize> =
        std::collections::HashMap::new();
    for edge in edges {
        let Some(metadata) = edge.get("metadata").and_then(Value::as_object) else {
            continue;
        };
        if metadata.get("relation").and_then(Value::as_str) != Some("wf_invoked") {
            continue;
        }
        if metadata.get("parent_run_id").and_then(Value::as_str) != Some(root_run_id) {
            continue;
        }
        let Some(workflow_id) = metadata.get("workflow_id").and_then(Value::as_str) else {
            continue;
        };
        let Some(invocation_key) = metadata.get("invocation_key").and_then(Value::as_str) else {
            continue;
        };
        let Some(child_run_id) = metadata.get("child_run_id").and_then(Value::as_str) else {
            continue;
        };
        let key = (
            workflow_id.to_string(),
            invocation_key.to_string(),
            child_run_id.to_string(),
        );
        if expected_keys.contains(&key) {
            *counts.entry(key).or_default() += 1;
        }
    }
    let mut missing = expected_keys
        .difference(&counts.keys().cloned().collect())
        .cloned()
        .collect::<Vec<_>>();
    let mut duplicate = counts
        .iter()
        .filter(|(_key, count)| **count != 1)
        .map(|(key, _count)| key.clone())
        .collect::<Vec<_>>();
    missing.sort();
    duplicate.sort();
    if !missing.is_empty() || !duplicate.is_empty() {
        let mut details = Vec::new();
        if !missing.is_empty() {
            details.push(format!("missing={missing:?}"));
        }
        if !duplicate.is_empty() {
            details.push(format!("duplicate={duplicate:?}"));
        }
        bail!(
            "OCR root persisted invocation lineage is invalid for {root_run_id:?}: {}",
            details.join("; ")
        );
    }
    Ok(())
}

pub fn expected_grouping_frontier(tree: &DocumentTree) -> Result<Vec<String>> {
    tree.validate()?;
    let Some(root) = tree.nodes.iter().find(|node| node.node_id == tree.root_id) else {
        bail!("grouping checkpoint root is missing");
    };
    if root.children.is_empty() {
        Ok(vec![root.node_id.clone()])
    } else {
        Ok(root
            .children
            .iter()
            .map(|child| child.node_id.clone())
            .collect())
    }
}

pub fn validated_grouping_checkpoint(
    tree: &DocumentTree,
    grouping_policy_version: &str,
    restored: Option<DocumentTree>,
    revision: Option<String>,
    persisted_frontier: Option<Vec<String>>,
) -> Result<(Option<DocumentTree>, Option<String>)> {
    let expected_source_fingerprint = tree.source_fingerprint();
    let Some(restored) = restored else {
        if revision.is_some() {
            bail!("grouping checkpoint exists but does not match the current source or policy");
        }
        return Ok((None, None));
    };
    if restored.source_fingerprint() != expected_source_fingerprint {
        bail!("grouping checkpoint exists but does not match the current source or policy");
    }
    let Some(persisted_frontier) = persisted_frontier else {
        bail!("grouping checkpoint exists but its frontier is missing or mismatched");
    };
    let expected_frontier = expected_grouping_frontier(&restored)?;
    if persisted_frontier != expected_frontier {
        bail!("grouping checkpoint frontier does not match the restored tree");
    }
    Ok((
        Some(restored),
        revision.map(|revision| format!("{grouping_policy_version}:{revision}")),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ocr_state_store::{sha256_hex, DocumentIdentity, OcrAttemptRequest};
    use serde_json::json;

    #[test]
    fn decodes_state_accessors() {
        let state = json!({
            "page_cursor": 2,
            "page_sources": [
                {"page_number": 1, "image_path": "p1.png"},
                {"page_number": 2, "image_path": "p2.png"}
            ],
            "completed_pages": [1, 2],
            "adaptive_grouping_complete": true
        });
        assert_eq!(page_cursor(&state), 2);
        assert_eq!(page_sources(&state).unwrap().len(), 2);
        assert_eq!(page_sources(&state).unwrap()[0].image_path, "p1.png");
        assert_eq!(int_list(&state, "completed_pages").unwrap(), vec![1, 2]);
        assert!(state_bool(&state, "adaptive_grouping_complete", false));
        assert_eq!(state_string(&state, "missing", "fallback"), "fallback");
    }

    #[test]
    fn rejects_malformed_state() {
        let state = json!({
            "page_sources": [{"page_number": 0, "image_path": "p0.png"}],
            "completed_pages": [1, "two"]
        });
        assert!(page_sources(&state).is_err());
        assert!(int_list(&state, "completed_pages").is_err());
        assert!(page_source_state(&json!("bad")).is_err());
    }

    #[test]
    fn validates_root_and_page_resume_identity() {
        let state = json!({
            "ocr_root_state_version": OCR_ROOT_STATE_VERSION,
            "document_id": "doc",
            "source_fingerprint": "source",
            "source_namespace": "namespace",
            "ocr_configuration_fingerprints": ["cfg"],
            "grouping_policy_version": "grouping",
            "parse_generation": "generation"
        });
        let root = RootResumeIdentity {
            document_id: "doc".to_string(),
            source_fingerprint: "source".to_string(),
            source_namespace: "namespace".to_string(),
            configuration_fingerprints: vec!["cfg".to_string()],
            grouping_policy_version: "grouping".to_string(),
            parse_generation: "generation".to_string(),
        };
        assert!(validate_root_resume_identity(&state, &root, true).is_ok());
        assert!(validate_root_resume_identity(&state, &root, false).is_ok());

        let page_state = json!({
            "ocr_page_state_version": OCR_PAGE_STATE_VERSION,
            "document_id": "doc",
            "source_fingerprint": "source",
            "ocr_configuration_fingerprints": ["cfg"]
        });
        let page = PageResumeIdentity {
            document_id: "doc".to_string(),
            source_fingerprint: "source".to_string(),
            configuration_fingerprints: vec!["cfg".to_string()],
        };
        assert!(validate_page_resume_identity(&page_state, &page).is_ok());

        let mismatched = json!({
            "ocr_page_state_version": OCR_PAGE_STATE_VERSION,
            "document_id": "other",
            "source_fingerprint": "source",
            "ocr_configuration_fingerprints": ["cfg"]
        });
        let error = validate_page_resume_identity(&mismatched, &page)
            .unwrap_err()
            .to_string();
        assert!(error.contains("document_id"));
    }

    #[test]
    fn reconciles_checkpoint_and_rewinds_invalid_prefix() {
        let temp = tempfile::tempdir().unwrap();
        let image = temp.path().join("page_1.png");
        std::fs::write(&image, b"page one").unwrap();
        let artifact = temp.path().join("page_1.json");
        std::fs::write(&artifact, r#"{"pdf_page_num":1}"#).unwrap();
        let state = json!({
            "page_cursor": 1,
            "page_sources": [{"page_number": 1, "image_path": image}],
            "completed_pages": [1],
            "reused_pages": [1],
            "raw_pages": [{"pdf_page_num": 1}]
        });
        let gate = ReadableArtifactGate;
        let repair = reconcile_page_checkpoint(&state, None, temp.path(), &gate, "doc", 0).unwrap();
        assert!(repair.is_none());

        std::fs::remove_file(&artifact).unwrap();
        let repair = reconcile_page_checkpoint(&state, None, temp.path(), &gate, "doc", 0)
            .unwrap()
            .unwrap();
        assert_eq!(repair["page_cursor"], 0);
        assert_eq!(repair["completed_pages"], json!([]));
        assert_eq!(repair["reused_pages"], json!([]));
        assert_eq!(repair["raw_pages"], json!([]));

        assert!(reconcile_page_checkpoint(&state, None, temp.path(), &gate, "doc", 2).is_err());
    }

    #[test]
    fn chooses_first_unused_recovery_run_id() {
        let used = ["root|recovery-1", "root|recovery-2"];
        let (run_id, generation) =
            next_root_recovery_run_id("root", |candidate| Ok(used.contains(&candidate))).unwrap();
        assert_eq!(run_id, "root|recovery-3");
        assert_eq!(generation, 3);
    }

    #[test]
    fn detects_root_artifact_recovery_before_terminal_root() {
        let temp = tempfile::tempdir().unwrap();
        let image = temp.path().join("page_1.png");
        std::fs::write(&image, b"page one").unwrap();
        let artifact = temp.path().join("page_1.json");
        let store = OcrWorkflowStateStore::open(&temp.path().join("ocr-state.sqlite")).unwrap();
        store
            .ensure_document(&DocumentIdentity {
                document_id: "doc".to_string(),
                title: "Doc".to_string(),
                source_kind: "image".to_string(),
                input_fingerprint: "source".to_string(),
                total_pages: 1,
            })
            .unwrap();
        store
            .record_attempt(
                "doc",
                1,
                OcrStage::Ocr,
                &OcrAttemptRequest {
                    content_hash: Some(sha256_hex(&std::fs::read(&image).unwrap())),
                    model_name: "model".to_string(),
                    config_fingerprint: Some("config".to_string()),
                    ..Default::default()
                },
            )
            .unwrap();
        let sources = vec![PageSourceState {
            page_number: 1,
            image_path: image.to_string_lossy().to_string(),
        }];
        let configuration_fingerprints = vec!["config".to_string()];
        let request = RootArtifactRecoveryRequest {
            state_store: &store,
            document_id: "doc",
            sources: &sources,
            legacy_dir: temp.path(),
            configuration_fingerprints: &configuration_fingerprints,
        };
        assert!(root_page_artifacts_need_recovery(&request).unwrap());

        std::fs::write(
            &artifact,
            serde_json::json!({
                "metadata": {
                    "ocr_json_version": "0.1",
                    "ocr_model_name": "model",
                    "source_image_sha256": sha256_hex(&std::fs::read(&image).unwrap()),
                    "ocr_configuration_fingerprint": "config",
                },
                "pdf_page_num": 1,
                "OCR_text_clusters": [{"text": "page one"}],
                "non_text_objects": [],
            })
            .to_string(),
        )
        .unwrap();
        assert!(!root_page_artifacts_need_recovery(&request).unwrap());
    }

    #[test]
    fn validates_recovery_generation_and_invocation_lineage() {
        assert_eq!(recovery_generation(&json!({})).unwrap(), 0);
        assert_eq!(
            recovery_generation(&json!({"ocr_page_recovery_generation": 2})).unwrap(),
            2
        );
        assert!(recovery_generation(&json!({"ocr_page_recovery_generation": -1})).is_err());
        assert_eq!(semantic_parse_generation(0), "semantic-v1");
        assert_eq!(semantic_parse_generation(2), "semantic-recovery-v2");

        let expected = expected_invocation_lineages("root", 0);
        let edges = expected
            .iter()
            .map(|lineage| {
                json!({
                    "metadata": {
                        "relation": "wf_invoked",
                        "parent_run_id": "root",
                        "workflow_id": lineage.workflow_id,
                        "invocation_key": lineage.invocation_key,
                        "child_run_id": lineage.child_run_id,
                    }
                })
            })
            .collect::<Vec<_>>();
        assert!(validate_invocation_lineages(&edges, "root", 0).is_ok());

        let mut incomplete = edges.clone();
        incomplete.pop();
        let error = validate_invocation_lineages(&incomplete, "root", 0)
            .unwrap_err()
            .to_string();
        assert!(error.contains("missing="));

        let mut duplicate = edges.clone();
        duplicate.push(edges[0].clone());
        let error = validate_invocation_lineages(&duplicate, "root", 0)
            .unwrap_err()
            .to_string();
        assert!(error.contains("duplicate="));
    }

    #[test]
    fn validates_grouping_checkpoint_frontier() {
        let tree = DocumentTree {
            schema_version: "1.0".to_string(),
            document_id: "doc".to_string(),
            source_revision: "source".to_string(),
            tree_revision: 1,
            root_id: "root".to_string(),
            source_items: vec![crate::document_models::SourceItem {
                item_id: "p1".to_string(),
                kind: "text".to_string(),
                location: crate::document_models::SourceLocation {
                    page_index: Some(1),
                    printed_page_label: None,
                    bbox: None,
                    start_seconds: None,
                    end_seconds: None,
                    structural_path: None,
                    accuracy: crate::document_models::SourceAccuracy::Page,
                },
                text: Some("Alpha Beta".to_string()),
                media: None,
                ocr_metadata: None,
                annotations: std::collections::BTreeMap::new(),
            }],
            nodes: vec![
                crate::document_models::DocumentNode {
                    node_id: "root".to_string(),
                    kind: "document".to_string(),
                    parts: Vec::new(),
                    children: vec![
                        crate::document_models::ChildRef {
                            node_id: "left".to_string(),
                            separator_before: String::new(),
                        },
                        crate::document_models::ChildRef {
                            node_id: "right".to_string(),
                            separator_before: String::new(),
                        },
                    ],
                    summary: None,
                    annotations: std::collections::BTreeMap::new(),
                },
                crate::document_models::DocumentNode {
                    node_id: "left".to_string(),
                    kind: "text".to_string(),
                    parts: vec![crate::document_models::SourceSlice {
                        item_id: "p1".to_string(),
                        start: Some(0),
                        end: Some(5),
                        separator_before: String::new(),
                    }],
                    children: Vec::new(),
                    summary: None,
                    annotations: std::collections::BTreeMap::new(),
                },
                crate::document_models::DocumentNode {
                    node_id: "right".to_string(),
                    kind: "text".to_string(),
                    parts: vec![crate::document_models::SourceSlice {
                        item_id: "p1".to_string(),
                        start: Some(5),
                        end: Some(10),
                        separator_before: String::new(),
                    }],
                    children: Vec::new(),
                    summary: None,
                    annotations: std::collections::BTreeMap::new(),
                },
            ],
            references: Vec::new(),
        };
        assert_eq!(
            expected_grouping_frontier(&tree).unwrap(),
            vec!["left".to_string(), "right".to_string()]
        );
        assert!(validated_grouping_checkpoint(
            &tree,
            "grouping-v1",
            Some(tree.clone()),
            Some("2".to_string()),
            Some(vec!["left".to_string(), "right".to_string()])
        )
        .is_ok());
        assert!(validated_grouping_checkpoint(
            &tree,
            "grouping-v1",
            Some(tree.clone()),
            Some("2".to_string()),
            Some(vec!["wrong".to_string()])
        )
        .is_err());
        assert!(validated_grouping_checkpoint(
            &tree,
            "grouping-v1",
            None,
            Some("2".to_string()),
            None
        )
        .is_err());
        assert!(
            validated_grouping_checkpoint(&tree, "grouping-v1", None, None, None)
                .unwrap()
                .0
                .is_none()
        );
    }

    #[test]
    fn validates_root_page_result_and_recovery() {
        let temp = tempfile::tempdir().unwrap();
        let image = temp.path().join("page_1.png");
        std::fs::write(&image, b"page one").unwrap();
        let artifact = temp.path().join("page_1.json");
        std::fs::write(&artifact, r#"{"pdf_page_num":1}"#).unwrap();
        let sources = vec![PageSourceState {
            page_number: 1,
            image_path: image.to_string_lossy().to_string(),
        }];
        let state = page_workflow_initial_state("source", temp.path());
        assert_eq!(state["ocr_page_state_version"], OCR_PAGE_STATE_VERSION);
        assert_eq!(state["page_cursor"], 0);

        let gate = ReadableArtifactGate;
        assert!(validate_root_page_result(&state, &sources, temp.path(), &gate, "doc").is_err());

        let mut complete = state.clone();
        complete["page_cursor"] = json!(1);
        complete["completed_pages"] = json!([1]);
        assert!(validate_root_page_result(&complete, &sources, temp.path(), &gate, "doc").is_ok());
        std::fs::remove_file(&artifact).unwrap();
        assert!(
            page_result_needs_recovery(&complete, &sources, temp.path(), &gate, "doc").unwrap()
        );
    }

    #[test]
    fn validates_existing_page_identities() {
        let mut states = std::collections::BTreeMap::new();
        states.insert(1, Some("cfg-1".to_string()));
        states.insert(2, None);
        assert!(validate_existing_page_identities(&states, &["cfg-1".to_string()]).is_ok());
        states.insert(2, Some("cfg-2".to_string()));
        assert!(validate_existing_page_identities(&states, &["cfg-1".to_string()]).is_err());
    }
}
