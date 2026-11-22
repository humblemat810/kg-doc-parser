use crate::document_models::{AdaptiveGroupingPolicy, DocumentNode, DocumentTree};
use crate::semantic_indexing::json_compact;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupProposal {
    pub kind: String,
    pub members: Vec<String>,
}

impl GroupProposal {
    pub fn validate(&self) -> Result<()> {
        if self.members.len() < 2 || self.members.len() > 32 {
            bail!("a merge proposal must contain between 2 and 32 members");
        }
        let unique: HashSet<&String> = self.members.iter().collect();
        if unique.len() != self.members.len() {
            bail!("a merge proposal cannot repeat a member");
        }
        if self.kind == "document" {
            bail!("document roots are created by the orchestrator");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupProposalBatch {
    #[serde(default)]
    pub proposals: Vec<GroupProposal>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GroupingModality {
    Text,
    Image,
    Audio,
    Video,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupingEvidence {
    pub item_id: String,
    pub kind: String,
    pub modality: GroupingModality,
    #[serde(default)]
    pub page_index: Option<u64>,
    #[serde(default)]
    pub printed_page_label: Option<String>,
    #[serde(default)]
    pub bbox: Option<crate::document_models::BoundingBox>,
    #[serde(default)]
    pub text_excerpt: Option<String>,
    #[serde(default)]
    pub media_mime_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupingCandidate {
    pub node_id: String,
    pub kind: String,
    pub source_item_count: usize,
    pub omitted_evidence_count: usize,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub evidence: Vec<GroupingEvidence>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupingPassContext {
    pub document_id: String,
    pub source_revision: String,
    pub pass_index: usize,
    pub frontier_ids: Vec<String>,
    pub candidates: Vec<GroupingCandidate>,
}

impl GroupingPassContext {
    pub fn validate(&self) -> Result<()> {
        if self.frontier_ids.len() < 2 || self.frontier_ids.len() > 64 {
            bail!("grouping context frontier must contain between 2 and 64 nodes");
        }
        if self.frontier_ids
            != self
                .candidates
                .iter()
                .map(|candidate| candidate.node_id.clone())
                .collect::<Vec<_>>()
        {
            bail!("grouping context candidates must exactly match frontier order");
        }
        if self.frontier_ids.iter().collect::<HashSet<_>>().len() != self.frontier_ids.len() {
            bail!("grouping context frontier IDs must be unique");
        }
        Ok(())
    }
}

pub fn grouping_policy_identity(
    policy_version: &str,
    policy: &AdaptiveGroupingPolicy,
    proposer_fingerprint: Option<&str>,
    summary_fingerprint: Option<&str>,
) -> String {
    let payload = json!([
        policy_version,
        proposer_fingerprint,
        summary_fingerprint,
        serde_json::to_value(policy).unwrap_or(Value::Null),
    ]);
    let serialized = json_compact(&payload);
    let mut digest = Sha256::new();
    digest.update(serialized.as_bytes());
    let hex: String = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    format!("{policy_version}:adaptive-{}", &hex[..16])
}

pub fn grouping_frontier_windows(
    frontier: &[String],
    policy: &AdaptiveGroupingPolicy,
) -> Vec<Vec<String>> {
    if !policy.enabled
        || frontier.len() < 2
        || frontier.iter().collect::<HashSet<_>>().len() != frontier.len()
    {
        return Vec::new();
    }
    let stride = policy.window_size - policy.window_overlap;
    let mut windows = Vec::new();
    let mut start = 0;
    while start < frontier.len() - 1 {
        let end = (start + policy.window_size).min(frontier.len());
        let window = frontier[start..end].to_vec();
        let window_len = window.len();
        if window.len() < 2 {
            break;
        }
        windows.push(window);
        if start + window_len >= frontier.len() {
            break;
        }
        start += stride;
    }
    windows
}

pub fn validate_grouping_window_proposals(
    frontier: &[String],
    window: &[String],
    proposals: &[GroupProposal],
) -> Result<Vec<GroupProposal>> {
    if window.is_empty()
        || !frontier
            .windows(window.len())
            .any(|candidate| candidate == window)
    {
        bail!("grouping proposal window is not a contiguous frontier slice");
    }
    let positions: HashMap<&String, usize> = frontier
        .iter()
        .enumerate()
        .map(|(index, node_id)| (node_id, index))
        .collect();
    let allowed_ids: HashSet<&String> = window.iter().collect();
    for proposal in proposals {
        if proposal
            .members
            .iter()
            .any(|member| !allowed_ids.contains(member))
        {
            bail!("grouping proposal references a candidate outside its window");
        }
        let mut indexes = Vec::new();
        for member in &proposal.members {
            indexes.push(positions.get(member).copied().unwrap_or(usize::MAX));
        }
        let first = indexes[0];
        if first == usize::MAX
            || indexes
                .iter()
                .enumerate()
                .any(|(offset, index)| *index != first + offset)
        {
            bail!("adaptive grouping proposal must cover adjacent frontier nodes");
        }
    }
    Ok(proposals.to_vec())
}

pub fn resolve_grouping_proposals(
    frontier: &[String],
    proposals: &[GroupProposal],
) -> Result<Vec<GroupProposal>> {
    if frontier.iter().collect::<HashSet<_>>().len() != frontier.len() {
        bail!("grouping frontier IDs must be unique");
    }
    let positions: HashMap<&String, usize> = frontier
        .iter()
        .enumerate()
        .map(|(index, node_id)| (node_id, index))
        .collect();
    let mut proposals_by_key: HashMap<(String, Vec<String>), GroupProposal> = HashMap::new();
    for proposal in proposals {
        let mut indexes = Vec::new();
        for member in &proposal.members {
            indexes.push(positions.get(member).copied().unwrap_or(usize::MAX));
        }
        let first = indexes[0];
        if first == usize::MAX
            || indexes
                .iter()
                .enumerate()
                .any(|(offset, index)| *index != first + offset)
        {
            bail!("adaptive grouping proposal must cover adjacent frontier nodes");
        }
        proposals_by_key.insert(
            (proposal.kind.clone(), proposal.members.clone()),
            proposal.clone(),
        );
    }
    let mut ordered: Vec<GroupProposal> = proposals_by_key.into_values().collect();
    ordered.sort_by_key(|proposal| {
        let start = positions
            .get(&proposal.members[0])
            .copied()
            .unwrap_or(usize::MAX);
        (
            start,
            std::cmp::Reverse(proposal.members.len()),
            proposal.kind.clone(),
            proposal.members.clone(),
        )
    });
    let mut occupied = HashSet::new();
    let mut selected = Vec::new();
    for proposal in ordered {
        let indexes: Vec<usize> = proposal
            .members
            .iter()
            .map(|member| positions.get(member).copied().unwrap_or(usize::MAX))
            .collect();
        if indexes.iter().any(|index| occupied.contains(index)) {
            continue;
        }
        occupied.extend(indexes);
        selected.push(proposal);
    }
    Ok(selected)
}

pub fn group_node_id(
    document_id: &str,
    grouping_policy_version: &str,
    proposal: &GroupProposal,
) -> String {
    let payload = json!([
        document_id,
        grouping_policy_version,
        proposal.kind,
        proposal.members,
    ]);
    let serialized = json_compact(&payload);
    let mut digest = Sha256::new();
    digest.update(serialized.as_bytes());
    let hex: String = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    format!("merge:{}", &hex[..24])
}

pub fn group_node_annotations(
    grouping_policy_version: &str,
    proposal: &GroupProposal,
) -> BTreeMap<String, String> {
    let mut annotations = BTreeMap::new();
    annotations.insert("grouping:format".to_string(), "grouping-v1".to_string());
    annotations.insert(
        "grouping:policy_version".to_string(),
        grouping_policy_version.to_string(),
    );
    annotations.insert(
        "grouping:members".to_string(),
        serde_json::to_string(&proposal.members).unwrap_or_default(),
    );
    annotations.insert(
        "grouping:summary_state".to_string(),
        "invalidated".to_string(),
    );
    annotations
}

pub fn merge_groups(
    document_id: &str,
    frontier: &[String],
    proposals: &[GroupProposal],
    grouping_policy_version: &str,
) -> Result<(Vec<String>, Vec<DocumentNode>)> {
    if frontier.iter().collect::<HashSet<_>>().len() != frontier.len()
        || document_id.is_empty()
        || grouping_policy_version.is_empty()
    {
        bail!("frontier and grouping identifiers must be unique and nonempty");
    }
    let positions: HashMap<&String, usize> = frontier
        .iter()
        .enumerate()
        .map(|(index, node_id)| (node_id, index))
        .collect();
    let mut occupied = HashSet::new();
    let mut groups: HashMap<usize, (usize, GroupProposal, String)> = HashMap::new();
    for proposal in proposals {
        let mut indexes = Vec::new();
        for member in &proposal.members {
            indexes.push(positions.get(member).copied().unwrap_or(usize::MAX));
        }
        let first = indexes[0];
        if first == usize::MAX
            || indexes
                .iter()
                .enumerate()
                .any(|(offset, index)| *index != first + offset)
            || indexes.iter().any(|index| occupied.contains(index))
        {
            bail!("merge must use adjacent, non-overlapping frontier members");
        }
        occupied.extend(indexes.clone());
        groups.insert(
            first,
            (
                *indexes.last().unwrap(),
                proposal.clone(),
                group_node_id(document_id, grouping_policy_version, proposal),
            ),
        );
    }
    let mut result = Vec::new();
    let mut created = Vec::new();
    let mut index = 0;
    while index < frontier.len() {
        if let Some((last, proposal, node_id)) = groups.get(&index) {
            result.push(node_id.clone());
            created.push(DocumentNode {
                node_id: node_id.clone(),
                kind: proposal.kind.clone(),
                parts: Vec::new(),
                children: proposal
                    .members
                    .iter()
                    .map(|member| crate::document_models::ChildRef {
                        node_id: member.clone(),
                        separator_before: String::new(),
                    })
                    .collect(),
                summary: None,
                annotations: group_node_annotations(grouping_policy_version, proposal),
            });
            index = last + 1;
        } else {
            result.push(frontier[index].clone());
            index += 1;
        }
    }
    Ok((result, created))
}

pub fn build_grouping_pass_context(
    tree: &DocumentTree,
    pass_index: usize,
    frontier_ids: &[String],
    maximum_evidence_items_per_candidate: usize,
    maximum_text_chars_per_candidate: usize,
) -> Result<GroupingPassContext> {
    if maximum_evidence_items_per_candidate < 1 || maximum_text_chars_per_candidate < 1 {
        bail!("grouping evidence limits must be positive");
    }
    if frontier_ids.iter().collect::<HashSet<_>>().len() != frontier_ids.len() {
        bail!("grouping context frontier IDs must be unique");
    }
    let nodes: HashMap<&str, &DocumentNode> = tree
        .nodes
        .iter()
        .map(|node| (node.node_id.as_str(), node))
        .collect();
    let items: HashMap<&str, &crate::document_models::SourceItem> = tree
        .source_items
        .iter()
        .map(|item| (item.item_id.as_str(), item))
        .collect();
    let mut candidates = Vec::new();
    for candidate_id in frontier_ids {
        let node = nodes.get(candidate_id.as_str()).ok_or_else(|| {
            anyhow::anyhow!("grouping context contains a node outside the current frontier")
        })?;
        let owned = node_owned_source_slices(node, &nodes, &items);
        let mut remaining_chars = maximum_text_chars_per_candidate;
        let mut evidence = Vec::new();
        for (item, source_slice) in owned.iter().take(maximum_evidence_items_per_candidate) {
            let mut excerpt = None;
            if let Some(text) = &item.text {
                let start_offset = source_slice.start.unwrap_or(0);
                let end_offset = source_slice.end.unwrap_or_else(|| text.chars().count());
                let source_text: String = text
                    .chars()
                    .skip(start_offset)
                    .take(end_offset - start_offset)
                    .collect();
                if source_text.chars().count() > remaining_chars {
                    if remaining_chars > 3 {
                        let head = (remaining_chars - 3) / 2;
                        let tail = remaining_chars - 3 - head;
                        let head_text: String = source_text.chars().take(head).collect();
                        let tail_text: String = source_text
                            .chars()
                            .skip(source_text.chars().count() - tail)
                            .collect();
                        excerpt = Some(format!("{head_text}...{tail_text}"));
                    } else {
                        excerpt = Some(source_text.chars().take(remaining_chars).collect());
                    }
                } else {
                    excerpt = Some(source_text);
                }
                remaining_chars = remaining_chars
                    .saturating_sub(excerpt.as_ref().map_or(0, |value| value.chars().count()));
            }
            evidence.push(GroupingEvidence {
                item_id: item.item_id.clone(),
                kind: item.kind.clone(),
                modality: item_modality(item),
                page_index: item.location.page_index,
                printed_page_label: item.location.printed_page_label.clone(),
                bbox: item.location.bbox.clone(),
                text_excerpt: excerpt,
                media_mime_type: item.media.as_ref().map(|media| media.mime_type.clone()),
            });
        }
        candidates.push(GroupingCandidate {
            node_id: node.node_id.clone(),
            kind: node.kind.clone(),
            source_item_count: owned.len(),
            omitted_evidence_count: owned.len().saturating_sub(evidence.len()),
            summary: node.summary.as_ref().and_then(|summary| {
                if matches!(
                    summary.status,
                    crate::document_models::SummaryStatus::Generated
                        | crate::document_models::SummaryStatus::Verified
                ) {
                    summary.text.as_ref().map(|text| {
                        text.chars()
                            .take(maximum_text_chars_per_candidate)
                            .collect()
                    })
                } else {
                    None
                }
            }),
            evidence,
        });
    }
    let context = GroupingPassContext {
        document_id: tree.document_id.clone(),
        source_revision: tree.source_revision.clone(),
        pass_index,
        frontier_ids: frontier_ids.to_vec(),
        candidates,
    };
    context.validate()?;
    Ok(context)
}

fn item_modality(item: &crate::document_models::SourceItem) -> GroupingModality {
    if item.text.is_some() {
        return GroupingModality::Text;
    }
    if let Some(media) = &item.media {
        let mime_type = media.mime_type.to_lowercase();
        if mime_type.starts_with("image/") {
            return GroupingModality::Image;
        }
        if mime_type.starts_with("audio/") {
            return GroupingModality::Audio;
        }
        if mime_type.starts_with("video/") {
            return GroupingModality::Video;
        }
    }
    GroupingModality::Other
}

fn node_owned_source_slices<'a>(
    node: &'a DocumentNode,
    nodes: &HashMap<&str, &'a DocumentNode>,
    items: &HashMap<&str, &'a crate::document_models::SourceItem>,
) -> Vec<(
    &'a crate::document_models::SourceItem,
    crate::document_models::SourceSlice,
)> {
    let mut result = Vec::new();
    let mut visiting = HashSet::new();
    let mut seen = HashSet::new();
    visit_node_source_slices(node, nodes, items, &mut visiting, &mut seen, &mut result);
    result
}

fn visit_node_source_slices<'a>(
    node: &'a DocumentNode,
    nodes: &HashMap<&str, &'a DocumentNode>,
    items: &HashMap<&str, &'a crate::document_models::SourceItem>,
    visiting: &mut HashSet<&'a str>,
    seen: &mut HashSet<&'a str>,
    result: &mut Vec<(
        &'a crate::document_models::SourceItem,
        crate::document_models::SourceSlice,
    )>,
) {
    if visiting.contains(node.node_id.as_str()) || seen.contains(node.node_id.as_str()) {
        return;
    }
    visiting.insert(node.node_id.as_str());
    if !node.parts.is_empty() {
        for part in &node.parts {
            if let Some(item) = items.get(part.item_id.as_str()) {
                result.push((item, part.clone()));
            }
        }
    } else {
        for child in &node.children {
            if let Some(child_node) = nodes.get(child.node_id.as_str()) {
                visit_node_source_slices(child_node, nodes, items, visiting, seen, result);
            }
        }
    }
    visiting.remove(node.node_id.as_str());
    seen.insert(node.node_id.as_str());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> AdaptiveGroupingPolicy {
        serde_json::from_value(json!({
            "enabled": true,
            "maximum_rounds": 6,
            "window_size": 4,
            "window_overlap": 2,
            "maximum_evidence_items_per_candidate": 4,
            "maximum_text_chars_per_candidate": 300
        }))
        .unwrap()
    }

    fn proposal(kind: &str, members: &[&str]) -> GroupProposal {
        GroupProposal {
            kind: kind.to_string(),
            members: members.iter().map(|member| member.to_string()).collect(),
        }
    }

    #[test]
    fn partitions_frontier_into_overlapping_windows() {
        let frontier = [
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
            "e".to_string(),
        ];
        let windows = grouping_frontier_windows(&frontier, &policy());
        assert_eq!(windows.len(), 2);
        assert_eq!(
            windows[0],
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string()
            ]
        );
        assert_eq!(
            windows[1],
            vec!["c".to_string(), "d".to_string(), "e".to_string()]
        );
    }

    #[test]
    fn resolves_overlapping_groups_left_to_right() {
        let frontier = [
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ];
        let proposals = vec![
            proposal("paragraph", &["a", "b"]),
            proposal("section", &["a", "b", "c"]),
        ];
        let resolved = resolve_grouping_proposals(&frontier, &proposals).unwrap();
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].kind, "section");
        assert_eq!(resolved[0].members, vec!["a", "b", "c"]);
    }

    #[test]
    fn merges_adjacent_frontier_members() {
        let frontier = ["a".to_string(), "b".to_string(), "c".to_string()];
        let proposal = proposal("section", &["a", "b"]);
        let (result, created) =
            merge_groups("doc", &frontier, std::slice::from_ref(&proposal), "v1").unwrap();
        assert_eq!(
            result,
            vec![group_node_id("doc", "v1", &proposal), "c".to_string()]
        );
        assert_eq!(created.len(), 1);
        assert_eq!(created[0].children.len(), 2);
    }
}
