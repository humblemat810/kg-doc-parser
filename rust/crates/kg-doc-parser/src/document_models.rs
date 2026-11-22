use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CoordinateSystem {
    Pixels,
    Relative,
    Points,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceAccuracy {
    Character,
    Word,
    Line,
    Region,
    Page,
    Time,
    Path,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
    pub coordinate_system: CoordinateSystem,
}

impl BoundingBox {
    pub fn validate(&self) -> Result<()> {
        if ![self.x0, self.y0, self.x1, self.y1]
            .iter()
            .all(|value| value.is_finite())
        {
            bail!("bounding-box coordinates must be finite");
        }
        if self.x0 >= self.x1 || self.y0 >= self.y1 {
            bail!("bounding box must have positive area");
        }
        if self.coordinate_system == CoordinateSystem::Relative
            && ![self.x0, self.y0, self.x1, self.y1]
                .iter()
                .all(|value| (0.0..=1.0).contains(value))
        {
            bail!("relative coordinates must lie in [0, 1]");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceLocation {
    #[serde(default)]
    pub page_index: Option<u64>,
    #[serde(default)]
    pub printed_page_label: Option<String>,
    #[serde(default)]
    pub bbox: Option<BoundingBox>,
    #[serde(default)]
    pub start_seconds: Option<f64>,
    #[serde(default)]
    pub end_seconds: Option<f64>,
    #[serde(default)]
    pub structural_path: Option<String>,
    pub accuracy: SourceAccuracy,
}

impl SourceLocation {
    pub fn validate(&self) -> Result<()> {
        if let Some(bbox) = &self.bbox {
            bbox.validate()?;
            if self.page_index.is_none() {
                bail!("bbox requires page_index");
            }
        }
        let has_time = self.start_seconds.is_some() || self.end_seconds.is_some();
        if let (Some(start), Some(end)) = (self.start_seconds, self.end_seconds) {
            if !start.is_finite() || !end.is_finite() {
                bail!("time coordinates must be finite");
            }
            if end <= start {
                bail!("time range must have positive endpoints");
            }
        }
        if self.page_index.is_none() && !has_time && self.structural_path.is_none() {
            bail!("location requires a page, time range, or path");
        }
        if matches!(
            self.accuracy,
            SourceAccuracy::Character
                | SourceAccuracy::Word
                | SourceAccuracy::Line
                | SourceAccuracy::Region
        ) && self.bbox.is_none()
        {
            bail!("spatial precision requires a bounding box");
        }
        if self.accuracy == SourceAccuracy::Time && !has_time {
            bail!("time accuracy requires a time range");
        }
        if self.accuracy == SourceAccuracy::Path && self.structural_path.is_none() {
            bail!("path accuracy requires structural_path");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaRef {
    pub uri: String,
    pub mime_type: String,
    pub sha256_hex: String,
    pub byte_length: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OCRMetadata {
    pub provider: String,
    pub model: String,
    pub configuration_fingerprint: String,
    #[serde(default)]
    pub confidence: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceItem {
    pub item_id: String,
    pub kind: String,
    pub location: SourceLocation,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub media: Option<MediaRef>,
    #[serde(default)]
    pub ocr_metadata: Option<OCRMetadata>,
    #[serde(default)]
    pub annotations: BTreeMap<String, String>,
}

impl SourceItem {
    pub fn validate(&self) -> Result<()> {
        if (self.text.is_none()) == (self.media.is_none()) {
            bail!("source item requires exactly one text or media payload");
        }
        if self.text.as_deref() == Some("") {
            bail!("empty OCR text is not an evidence item");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceSlice {
    pub item_id: String,
    #[serde(default)]
    pub start: Option<usize>,
    #[serde(default)]
    pub end: Option<usize>,
    #[serde(default)]
    pub separator_before: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceSlice {
    pub item_id: String,
    pub kind: String,
    pub location: SourceLocation,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub media: Option<MediaRef>,
    #[serde(default)]
    pub start: Option<usize>,
    #[serde(default)]
    pub end: Option<usize>,
    #[serde(default)]
    pub separator_before: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChildRef {
    pub node_id: String,
    #[serde(default)]
    pub separator_before: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SummaryStatus {
    Pending,
    Generated,
    Verified,
    Failed,
    Skipped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Summary {
    pub status: SummaryStatus,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub evidence_item_ids: Vec<String>,
    #[serde(default)]
    pub provider_model: Option<String>,
    #[serde(default)]
    pub prompt_revision: Option<String>,
    #[serde(default)]
    pub generation_fingerprint: Option<String>,
    #[serde(default)]
    pub input_fingerprint: Option<String>,
    #[serde(default)]
    pub evidence_slice_fingerprint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryPolicy {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub generation_fingerprint: Option<String>,
    #[serde(default = "default_eligible_kinds")]
    pub eligible_kinds: Vec<String>,
    #[serde(default = "default_two")]
    pub minimum_source_items: usize,
    #[serde(default = "default_one")]
    pub minimum_source_chars: usize,
    #[serde(default = "default_one")]
    pub minimum_information_gain_chars: usize,
    #[serde(default = "default_2048")]
    pub maximum_input_tokens: usize,
    #[serde(default = "default_32")]
    pub maximum_targets: usize,
}

fn default_true() -> bool {
    true
}

fn default_one() -> usize {
    1
}

fn default_two() -> usize {
    2
}

fn default_32() -> usize {
    32
}

fn default_2048() -> usize {
    2048
}

fn default_eligible_kinds() -> Vec<String> {
    ["table", "figure", "section", "chapter", "document"]
        .iter()
        .map(|kind| kind.to_string())
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveGroupingPolicy {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_six")]
    pub maximum_rounds: usize,
    #[serde(default = "default_64")]
    pub window_size: usize,
    #[serde(default = "default_31")]
    pub window_overlap: usize,
    #[serde(default = "default_four")]
    pub maximum_evidence_items_per_candidate: usize,
    #[serde(default = "default_300")]
    pub maximum_text_chars_per_candidate: usize,
}

fn default_four() -> usize {
    4
}

fn default_six() -> usize {
    6
}

fn default_31() -> usize {
    31
}

fn default_64() -> usize {
    64
}

fn default_300() -> usize {
    300
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReferenceKind {
    Page,
    Section,
    Figure,
    Table,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReferenceStatus {
    Pending,
    Resolved,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentReferenceMention {
    pub mention_id: String,
    pub source: SourceSlice,
    #[serde(default)]
    pub additional_sources: Vec<SourceSlice>,
    pub target_kind: ReferenceKind,
    pub target_label: String,
    #[serde(default)]
    pub target_node_id: Option<String>,
    #[serde(default = "reference_status_pending")]
    pub status: ReferenceStatus,
}

fn reference_status_pending() -> ReferenceStatus {
    ReferenceStatus::Pending
}

impl DocumentReferenceMention {
    pub fn sources(&self) -> Vec<&SourceSlice> {
        std::iter::once(&self.source)
            .chain(self.additional_sources.iter())
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentNode {
    pub node_id: String,
    pub kind: String,
    #[serde(default)]
    pub parts: Vec<SourceSlice>,
    #[serde(default)]
    pub children: Vec<ChildRef>,
    #[serde(default)]
    pub summary: Option<Summary>,
    #[serde(default)]
    pub annotations: BTreeMap<String, String>,
}

impl DocumentNode {
    pub fn validate(&self) -> Result<()> {
        if self.parts.is_empty() == self.children.is_empty() {
            bail!("node must contain parts or children, exclusively");
        }
        if !self.children.is_empty() && self.children.len() < 2 && self.kind != "document" {
            bail!("unary grouping nodes are not allowed");
        }
        let child_ids: HashSet<&str> = self.children.iter().map(|c| c.node_id.as_str()).collect();
        if child_ids.len() != self.children.len() {
            bail!("duplicate child in grouping node");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentTree {
    #[serde(default = "schema_version")]
    pub schema_version: String,
    pub document_id: String,
    pub source_revision: String,
    pub tree_revision: u64,
    pub root_id: String,
    pub source_items: Vec<SourceItem>,
    pub nodes: Vec<DocumentNode>,
    #[serde(default)]
    pub references: Vec<DocumentReferenceMention>,
}

fn schema_version() -> String {
    "1.0".to_string()
}

impl DocumentTree {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != "1.0" {
            bail!("unsupported document tree schema version");
        }
        let items: HashMap<&str, &SourceItem> = self
            .source_items
            .iter()
            .map(|item| (item.item_id.as_str(), item))
            .collect();
        let nodes: HashMap<&str, &DocumentNode> = self
            .nodes
            .iter()
            .map(|node| (node.node_id.as_str(), node))
            .collect();
        if items.len() != self.source_items.len()
            || nodes.len() != self.nodes.len()
            || items.keys().any(|item_id| nodes.contains_key(item_id))
        {
            bail!("duplicate or overlapping source and node identities");
        }
        let root = nodes
            .get(self.root_id.as_str())
            .ok_or_else(|| anyhow::anyhow!("document root is missing or has the wrong kind"))?;
        if root.kind != "document" {
            bail!("document root is missing or has the wrong kind");
        }
        let mut reference_ids = HashSet::new();
        for reference in &self.references {
            if !reference_ids.insert(reference.mention_id.as_str()) {
                bail!("duplicate document reference mention");
            }
            let mut source_keys = HashSet::new();
            for source in reference.sources() {
                let key = (source.item_id.as_str(), source.start, source.end);
                if !source_keys.insert(key) {
                    bail!("reference mention contains duplicate source slice");
                }
                let item = items.get(source.item_id.as_str()).ok_or_else(|| {
                    anyhow::anyhow!("reference mention must point to a text source item")
                })?;
                if item.text.is_none() {
                    bail!("reference mention must point to a text source item");
                }
                if let Some(end) = source.end {
                    if end > item.text.as_ref().unwrap_or(&String::new()).chars().count() {
                        bail!("reference mention exceeds source item");
                    }
                }
            }
            if let Some(target_node_id) = &reference.target_node_id {
                if !nodes.contains_key(target_node_id.as_str()) {
                    bail!("reference mention target node does not exist");
                }
            }
        }

        let mut parents: HashMap<&str, usize> = HashMap::new();
        let mut intervals: HashMap<&str, Vec<(usize, usize)>> = HashMap::new();
        let mut media_counts: HashMap<&str, usize> = HashMap::new();
        let mut visiting = HashSet::new();
        let mut seen = HashSet::new();
        let covered = visit_tree_node(
            self.root_id.as_str(),
            &nodes,
            &items,
            &mut parents,
            &mut intervals,
            &mut media_counts,
            &mut visiting,
            &mut seen,
        )?;
        if let Some(summary) = &root.summary {
            if summary
                .evidence_item_ids
                .iter()
                .any(|item_id| !covered.contains(item_id.as_str()))
            {
                bail!("summary cites evidence outside its subtree");
            }
        }

        if seen.len() != nodes.len()
            || nodes.keys().any(|node_id| {
                *node_id != self.root_id && parents.get(*node_id).copied().unwrap_or(0) != 1
            })
        {
            bail!("orphan or multiply-parented node");
        }
        for (item_id, item) in &items {
            if item.text.is_none() {
                if media_counts.get(*item_id).copied().unwrap_or(0) != 1 {
                    bail!("media item must be covered exactly once: {item_id}");
                }
                continue;
            }
            let mut cursor = 0usize;
            let mut source_intervals = intervals.get(*item_id).cloned().unwrap_or_default();
            source_intervals.sort_unstable();
            for (start, end) in source_intervals {
                if start != cursor {
                    bail!("source gap or overlap in item {item_id}");
                }
                cursor = end;
            }
            if cursor != item.text.as_ref().unwrap_or(&String::new()).chars().count() {
                bail!("incomplete source coverage: {item_id}");
            }
        }
        Ok(())
    }

    pub fn source_fingerprint(&self) -> String {
        let mut payload = format!("{}|{}|", self.document_id, self.source_revision);
        for item in &self.source_items {
            payload.push_str(&serde_json::to_string(item).unwrap_or_default());
            payload.push('|');
        }
        let mut digest = Sha256::new();
        digest.update(payload.as_bytes());
        digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }
}

#[allow(clippy::too_many_arguments)]
fn visit_tree_node<'a>(
    node_id: &'a str,
    nodes: &HashMap<&'a str, &'a DocumentNode>,
    items: &HashMap<&'a str, &'a SourceItem>,
    parents: &mut HashMap<&'a str, usize>,
    intervals: &mut HashMap<&'a str, Vec<(usize, usize)>>,
    media_counts: &mut HashMap<&'a str, usize>,
    visiting: &mut HashSet<&'a str>,
    seen: &mut HashSet<&'a str>,
) -> Result<HashSet<&'a str>> {
    if visiting.contains(node_id) {
        bail!("cycle in document tree");
    }
    if seen.contains(node_id) {
        return Ok(HashSet::new());
    }
    let node = nodes
        .get(node_id)
        .ok_or_else(|| anyhow::anyhow!("dangling node reference: {node_id}"))?;
    visiting.insert(node_id);
    let mut covered = HashSet::new();
    if !node.parts.is_empty() {
        for part in &node.parts {
            let item = items
                .get(part.item_id.as_str())
                .ok_or_else(|| anyhow::anyhow!("unknown source item: {}", part.item_id))?;
            covered.insert(part.item_id.as_str());
            if item.text.is_none() {
                if part.start.is_some() || part.end.is_some() {
                    bail!("media slices cannot have text offsets");
                }
                *media_counts.entry(part.item_id.as_str()).or_insert(0) += 1;
            } else {
                let (start, end) = match (part.start, part.end) {
                    (Some(start), Some(end)) => (start, end),
                    _ => bail!("text slice exceeds source item"),
                };
                if end > item.text.as_ref().unwrap_or(&String::new()).chars().count() {
                    bail!("text slice exceeds source item");
                }
                intervals
                    .entry(part.item_id.as_str())
                    .or_default()
                    .push((start, end));
            }
        }
    } else {
        for child in &node.children {
            *parents.entry(child.node_id.as_str()).or_insert(0) += 1;
            covered.extend(visit_tree_node(
                child.node_id.as_str(),
                nodes,
                items,
                parents,
                intervals,
                media_counts,
                visiting,
                seen,
            )?);
        }
    }
    if let Some(summary) = &node.summary {
        if summary
            .evidence_item_ids
            .iter()
            .any(|item_id| !covered.contains(item_id.as_str()))
        {
            bail!("summary cites evidence outside its subtree");
        }
    }
    visiting.remove(node_id);
    seen.insert(node_id);
    Ok(covered)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn location(page_index: u64) -> SourceLocation {
        SourceLocation {
            page_index: Some(page_index),
            printed_page_label: None,
            bbox: None,
            start_seconds: None,
            end_seconds: None,
            structural_path: None,
            accuracy: SourceAccuracy::Page,
        }
    }

    fn text_item(item_id: &str, text: &str) -> SourceItem {
        SourceItem {
            item_id: item_id.to_string(),
            kind: "paragraph".to_string(),
            location: location(0),
            text: Some(text.to_string()),
            media: None,
            ocr_metadata: None,
            annotations: BTreeMap::new(),
        }
    }

    fn leaf(node_id: &str, item_id: &str, text: &str) -> DocumentNode {
        DocumentNode {
            node_id: node_id.to_string(),
            kind: "paragraph".to_string(),
            parts: vec![SourceSlice {
                item_id: item_id.to_string(),
                start: Some(0),
                end: Some(text.chars().count()),
                separator_before: String::new(),
            }],
            children: Vec::new(),
            summary: None,
            annotations: BTreeMap::new(),
        }
    }

    #[test]
    fn validates_minimal_tree() {
        let tree = DocumentTree {
            schema_version: "1.0".to_string(),
            document_id: "doc".to_string(),
            source_revision: "rev-1".to_string(),
            tree_revision: 1,
            root_id: "root".to_string(),
            source_items: vec![text_item("a", "alpha")],
            nodes: vec![
                DocumentNode {
                    node_id: "root".to_string(),
                    kind: "document".to_string(),
                    parts: Vec::new(),
                    children: vec![ChildRef {
                        node_id: "leaf".to_string(),
                        separator_before: String::new(),
                    }],
                    summary: None,
                    annotations: BTreeMap::new(),
                },
                leaf("leaf", "a", "alpha"),
            ],
            references: Vec::new(),
        };
        tree.validate().unwrap();
        assert!(!tree.source_fingerprint().is_empty());
    }

    #[test]
    fn rejects_incomplete_source_coverage() {
        let tree = DocumentTree {
            schema_version: "1.0".to_string(),
            document_id: "doc".to_string(),
            source_revision: "rev-1".to_string(),
            tree_revision: 1,
            root_id: "root".to_string(),
            source_items: vec![text_item("a", "alpha")],
            nodes: vec![
                DocumentNode {
                    node_id: "root".to_string(),
                    kind: "document".to_string(),
                    parts: Vec::new(),
                    children: vec![ChildRef {
                        node_id: "leaf".to_string(),
                        separator_before: String::new(),
                    }],
                    summary: None,
                    annotations: BTreeMap::new(),
                },
                leaf("leaf", "a", "alph"),
            ],
            references: Vec::new(),
        };
        assert!(tree.validate().is_err());
    }

    #[test]
    fn rejects_relative_bbox_outside_unit_range() {
        let bbox = BoundingBox {
            x0: 0.0,
            y0: 0.0,
            x1: 1.1,
            y1: 1.0,
            coordinate_system: CoordinateSystem::Relative,
        };
        assert!(bbox.validate().is_err());
    }
}
