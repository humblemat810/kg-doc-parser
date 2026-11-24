use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OCRGroundTruthModality {
    Text,
    #[serde(rename = "non_text")]
    NonText,
}

impl OCRGroundTruthModality {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::NonText => "non_text",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OCRGroundTruthItem {
    pub item_id: String,
    pub modality: OCRGroundTruthModality,
    pub x_min: f64,
    pub y_min: f64,
    pub x_max: f64,
    pub y_max: f64,
    #[serde(default)]
    pub text: Option<String>,
}

impl OCRGroundTruthItem {
    pub fn validate(&self) -> Result<()> {
        if ![self.x_min, self.y_min, self.x_max, self.y_max]
            .iter()
            .all(|value| value.is_finite())
        {
            bail!("ground-truth item bounds must be finite");
        }
        if self.x_min >= self.x_max || self.y_min >= self.y_max {
            bail!("ground-truth item bounds must have positive area");
        }
        match self.modality {
            OCRGroundTruthModality::Text if self.text.as_deref().is_none_or(str::is_empty) => {
                bail!("text ground-truth items require text");
            }
            OCRGroundTruthModality::NonText if self.text.is_some() => {
                bail!("non-text ground-truth items cannot carry text");
            }
            _ => {}
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OCRGroundTruth {
    #[serde(default)]
    pub items: Vec<OCRGroundTruthItem>,
}

impl OCRGroundTruth {
    pub fn validate(&self) -> Result<()> {
        let ids: HashSet<&String> = self.items.iter().map(|item| &item.item_id).collect();
        if ids.len() != self.items.len() {
            bail!("ground-truth item IDs must be unique");
        }
        for item in &self.items {
            item.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OCRCluster {
    pub cluster_number: u32,
    pub bb_x_min: f64,
    pub bb_y_min: f64,
    pub bb_x_max: f64,
    pub bb_y_max: f64,
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OCRClusterResponse {
    #[serde(default, rename = "OCR_text_clusters")]
    pub text_clusters: Vec<OCRCluster>,
    #[serde(default, rename = "non_text_objects")]
    pub non_text_objects: Vec<OCRCluster>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OCRQualityDiagnosticSeverity {
    Warning,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OCRQualityDiagnostic {
    pub code: String,
    pub severity: OCRQualityDiagnosticSeverity,
    pub message: String,
}

pub fn ocr_artifact_quality_error(payload: &serde_json::Value) -> Option<String> {
    let metadata = payload
        .as_object()
        .and_then(|payload| payload.get("metadata"))
        .and_then(|value| value.as_object())?;
    let has_status = metadata.contains_key("ocr_quality_status");
    let has_diagnostics = metadata.contains_key("ocr_quality_diagnostics");
    if !has_status && !has_diagnostics {
        return None;
    }
    let status = metadata
        .get("ocr_quality_status")
        .and_then(|value| value.as_str());
    if has_status
        && !matches!(
            status,
            Some("accepted") | Some("flagged") | Some("rejected")
        )
    {
        return Some("invalid OCR quality status".to_string());
    }
    if status == Some("rejected") {
        return Some("OCR quality status is rejected".to_string());
    }
    if !has_diagnostics {
        return None;
    }
    let Some(raw_diagnostics) = metadata
        .get("ocr_quality_diagnostics")
        .and_then(|value| value.as_array())
    else {
        return Some("OCR quality diagnostics must be a list".to_string());
    };
    let Ok(diagnostics) = raw_diagnostics
        .iter()
        .map(|value| serde_json::from_value::<OCRQualityDiagnostic>(value.clone()))
        .collect::<std::result::Result<Vec<_>, _>>()
    else {
        return Some("OCR quality diagnostics are malformed".to_string());
    };
    if diagnostics
        .iter()
        .any(|diagnostic| diagnostic.severity == OCRQualityDiagnosticSeverity::Error)
    {
        return Some("OCR artifact contains error-severity quality diagnostics".to_string());
    }
    if status == Some("accepted") && !diagnostics.is_empty() {
        return Some("accepted OCR quality status cannot include diagnostics".to_string());
    }
    if status == Some("flagged") && diagnostics.is_empty() {
        return Some("flagged OCR quality status requires diagnostics".to_string());
    }
    None
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OCRGroundTruthReport {
    pub expected_item_count: usize,
    pub matched_item_count: usize,
    pub item_recall: f64,
    pub mean_spatial_iou: f64,
    #[serde(default)]
    pub missing_item_ids: Vec<String>,
    #[serde(default)]
    pub extra_cluster_numbers: Vec<u32>,
    #[serde(default)]
    pub text_mismatch_item_ids: Vec<String>,
    pub minimum_iou: f64,
    #[serde(default)]
    pub item_recall_by_modality: BTreeMap<String, Option<f64>>,
}

#[derive(Debug, Clone)]
struct Candidate {
    modality: OCRGroundTruthModality,
    cluster_number: u32,
    bounds: [f64; 4],
    text: Option<String>,
}

fn normalize_text(value: &str) -> String {
    value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn intersection_over_union(left: &OCRGroundTruthItem, right: &[f64; 4]) -> f64 {
    let x_min = left.x_min.max(right[0]);
    let y_min = left.y_min.max(right[1]);
    let x_max = left.x_max.min(right[2]);
    let y_max = left.y_max.min(right[3]);
    let intersection = (x_max - x_min).max(0.0) * (y_max - y_min).max(0.0);
    let left_area = (left.x_max - left.x_min) * (left.y_max - left.y_min);
    let right_area = (right[2] - right[0]).max(0.0) * (right[3] - right[1]).max(0.0);
    let union = left_area + right_area - intersection;
    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

fn assign_matches(
    expected_items: &[OCRGroundTruthItem],
    candidates: &[Candidate],
    minimum_iou: f64,
) -> HashMap<usize, (usize, f64)> {
    let mut eligible: Vec<Vec<(usize, f64)>> = expected_items
        .iter()
        .map(|expected| {
            candidates
                .iter()
                .enumerate()
                .filter(|(_, candidate)| candidate.modality == expected.modality)
                .map(|(index, candidate)| {
                    (index, intersection_over_union(expected, &candidate.bounds))
                })
                .filter(|(_, iou)| *iou > 0.0 && *iou >= minimum_iou)
                .collect()
        })
        .collect();
    for options in &mut eligible {
        options.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(left.0.cmp(&right.0))
        });
    }
    let mut candidate_owner: HashMap<usize, usize> = HashMap::new();
    for expected_index in 0..expected_items.len() {
        visit_match(
            expected_index,
            &eligible,
            &mut candidate_owner,
            &mut HashSet::new(),
        );
    }
    let mut assignments = HashMap::new();
    for (expected_index, options) in eligible.iter().enumerate() {
        if let Some(candidate_index) =
            candidate_owner.iter().find_map(|(candidate_index, owner)| {
                (*owner == expected_index).then_some(*candidate_index)
            })
        {
            if let Some((_, iou)) = options.iter().find(|(index, _)| *index == candidate_index) {
                assignments.insert(expected_index, (candidate_index, *iou));
            }
        }
    }
    assignments
}

fn visit_match(
    expected_index: usize,
    eligible: &[Vec<(usize, f64)>],
    candidate_owner: &mut HashMap<usize, usize>,
    seen: &mut HashSet<usize>,
) -> bool {
    for (candidate_index, _) in &eligible[expected_index] {
        if !seen.insert(*candidate_index) {
            continue;
        }
        match candidate_owner.get(candidate_index) {
            None => {
                candidate_owner.insert(*candidate_index, expected_index);
                return true;
            }
            Some(owner) => {
                if visit_match(*owner, eligible, candidate_owner, seen) {
                    candidate_owner.insert(*candidate_index, expected_index);
                    return true;
                }
            }
        }
    }
    false
}

pub fn evaluate_ocr_ground_truth(
    response: &OCRClusterResponse,
    ground_truth: &OCRGroundTruth,
    minimum_iou: f64,
) -> Result<OCRGroundTruthReport> {
    ground_truth.validate()?;
    if !(0.0..=1.0).contains(&minimum_iou) {
        bail!("minimum_iou must be between 0.0 and 1.0");
    }
    let mut candidates = response
        .text_clusters
        .iter()
        .map(|cluster| Candidate {
            modality: OCRGroundTruthModality::Text,
            cluster_number: cluster.cluster_number,
            bounds: [
                cluster.bb_x_min,
                cluster.bb_y_min,
                cluster.bb_x_max,
                cluster.bb_y_max,
            ],
            text: cluster.text.clone(),
        })
        .collect::<Vec<_>>();
    candidates.extend(response.non_text_objects.iter().map(|cluster| Candidate {
        modality: OCRGroundTruthModality::NonText,
        cluster_number: cluster.cluster_number,
        bounds: [
            cluster.bb_x_min,
            cluster.bb_y_min,
            cluster.bb_x_max,
            cluster.bb_y_max,
        ],
        text: None,
    }));
    for candidate in &candidates {
        if !candidate.bounds.iter().all(|value| value.is_finite())
            || candidate.bounds[0] >= candidate.bounds[2]
            || candidate.bounds[1] >= candidate.bounds[3]
        {
            bail!(
                "OCR candidate {} must have finite, positive-area bounds",
                candidate.cluster_number
            );
        }
    }
    let assignments = assign_matches(&ground_truth.items, &candidates, minimum_iou);
    let mut recall_by_modality = BTreeMap::new();
    for modality in [
        OCRGroundTruthModality::Text,
        OCRGroundTruthModality::NonText,
    ] {
        let expected_indices: Vec<usize> = ground_truth
            .items
            .iter()
            .enumerate()
            .filter(|(_, item)| item.modality == modality)
            .map(|(index, _)| index)
            .collect();
        let matched_count = expected_indices
            .iter()
            .filter(|index| assignments.contains_key(index))
            .count();
        recall_by_modality.insert(
            modality.as_str().to_string(),
            if expected_indices.is_empty() {
                None
            } else {
                Some(matched_count as f64 / expected_indices.len() as f64)
            },
        );
    }
    let mut missing = Vec::new();
    let mut mismatches = Vec::new();
    let mut spatial_ious = Vec::new();
    let mut matched = 0;
    for (expected_index, expected) in ground_truth.items.iter().enumerate() {
        let Some((candidate_index, iou)) = assignments.get(&expected_index) else {
            missing.push(expected.item_id.clone());
            continue;
        };
        matched += 1;
        spatial_ious.push(*iou);
        let candidate_text = candidates[*candidate_index]
            .text
            .as_deref()
            .unwrap_or_default();
        if let Some(expected_text) = &expected.text {
            if normalize_text(candidate_text) != normalize_text(expected_text) {
                mismatches.push(expected.item_id.clone());
            }
        }
    }
    let assigned_candidates: HashSet<usize> =
        assignments.values().map(|(index, _)| *index).collect();
    Ok(OCRGroundTruthReport {
        expected_item_count: ground_truth.items.len(),
        matched_item_count: matched,
        item_recall: if ground_truth.items.is_empty() {
            1.0
        } else {
            matched as f64 / ground_truth.items.len() as f64
        },
        mean_spatial_iou: if spatial_ious.is_empty() {
            0.0
        } else {
            spatial_ious.iter().sum::<f64>() / spatial_ious.len() as f64
        },
        missing_item_ids: missing,
        extra_cluster_numbers: candidates
            .iter()
            .enumerate()
            .filter(|(index, _)| !assigned_candidates.contains(index))
            .map(|(_, candidate)| candidate.cluster_number)
            .collect(),
        text_mismatch_item_ids: mismatches,
        minimum_iou,
        item_recall_by_modality: recall_by_modality,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ground_truth() -> OCRGroundTruth {
        serde_json::from_value(json!({
            "items": [
                {
                    "item_id": "text-1",
                    "modality": "text",
                    "x_min": 0.0,
                    "y_min": 0.0,
                    "x_max": 10.0,
                    "y_max": 10.0,
                    "text": "Alpha  Beta"
                },
                {
                    "item_id": "image-1",
                    "modality": "non_text",
                    "x_min": 20.0,
                    "y_min": 20.0,
                    "x_max": 30.0,
                    "y_max": 30.0
                }
            ]
        }))
        .unwrap()
    }

    #[test]
    fn matches_text_and_non_text_items() {
        let response: OCRClusterResponse = serde_json::from_value(json!({
            "OCR_text_clusters": [{
                "cluster_number": 0,
                "bb_x_min": 1.0,
                "bb_y_min": 1.0,
                "bb_x_max": 10.0,
                "bb_y_max": 10.0,
                "text": "alpha beta"
            }],
            "non_text_objects": [{
                "cluster_number": 1,
                "bb_x_min": 20.0,
                "bb_y_min": 20.0,
                "bb_x_max": 30.0,
                "bb_y_max": 30.0
            }]
        }))
        .unwrap();
        let report = evaluate_ocr_ground_truth(&response, &ground_truth(), 0.5).unwrap();
        assert_eq!(report.matched_item_count, 2);
        assert_eq!(report.item_recall, 1.0);
        assert!(report.missing_item_ids.is_empty());
        assert!(report.extra_cluster_numbers.is_empty());
        assert!(report.text_mismatch_item_ids.is_empty());
        assert_eq!(report.item_recall_by_modality.get("text"), Some(&Some(1.0)));
        assert_eq!(
            report.item_recall_by_modality.get("non_text"),
            Some(&Some(1.0))
        );
    }

    #[test]
    fn reports_missing_and_mismatched_items() {
        let response: OCRClusterResponse = serde_json::from_value(json!({
            "OCR_text_clusters": [{
                "cluster_number": 0,
                "bb_x_min": 1.0,
                "bb_y_min": 1.0,
                "bb_x_max": 10.0,
                "bb_y_max": 10.0,
                "text": "wrong text"
            }]
        }))
        .unwrap();
        let report = evaluate_ocr_ground_truth(&response, &ground_truth(), 0.5).unwrap();
        assert_eq!(report.matched_item_count, 1);
        assert_eq!(report.item_recall, 0.5);
        assert_eq!(report.missing_item_ids, vec!["image-1"]);
        assert!(report.extra_cluster_numbers.is_empty());
        assert_eq!(report.text_mismatch_item_ids, vec!["text-1"]);
    }

    #[test]
    fn validates_artifact_quality_metadata() {
        assert_eq!(ocr_artifact_quality_error(&json!({"text": "Alpha"})), None);
        assert_eq!(
            ocr_artifact_quality_error(&json!({"metadata": {"ocr_quality_status": "bad"}})),
            Some("invalid OCR quality status".to_string())
        );
        assert_eq!(
            ocr_artifact_quality_error(&json!({"metadata": {"ocr_quality_status": "rejected"}})),
            Some("OCR quality status is rejected".to_string())
        );
        assert_eq!(
            ocr_artifact_quality_error(&json!({
                "metadata": {
                    "ocr_quality_status": "accepted",
                    "ocr_quality_diagnostics": [{
                        "code": "unresolved_items",
                        "severity": "warning",
                        "message": "provider unresolved item"
                    }]
                }
            })),
            Some("accepted OCR quality status cannot include diagnostics".to_string())
        );
        assert_eq!(
            ocr_artifact_quality_error(&json!({
                "metadata": {
                    "ocr_quality_status": "flagged",
                    "ocr_quality_diagnostics": [{
                        "code": "empty_ocr_output",
                        "severity": "warning",
                        "message": "provider reported no OCR evidence"
                    }]
                }
            })),
            None
        );
        assert_eq!(
            ocr_artifact_quality_error(&json!({
                "metadata": {
                    "ocr_quality_status": "flagged",
                    "ocr_quality_diagnostics": [{
                        "code": "invalid",
                        "severity": "error",
                        "message": "invalid evidence"
                    }]
                }
            })),
            Some("OCR artifact contains error-severity quality diagnostics".to_string())
        );
    }
}
