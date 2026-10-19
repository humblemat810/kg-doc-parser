"""Deterministic validation of bottom-up grouping proposals."""
from __future__ import annotations

import json
from collections.abc import Sequence
from hashlib import sha256
from typing import Literal, Protocol

from pydantic import Field, model_validator

from .document_models import (
    AdaptiveGroupingPolicy,
    BoundingBox,
    ChildRef,
    DocumentNode,
    DocumentTree,
    FrozenModel,
    SourceItem,
    SourceSlice,
)


class GroupProposal(FrozenModel):
    kind: str = Field(min_length=1, max_length=128)
    members: tuple[str, ...] = Field(min_length=2, max_length=32)

    @model_validator(mode="after")
    def valid(self) -> GroupProposal:
        if len(set(self.members)) != len(self.members):
            raise ValueError("a merge proposal cannot repeat a member")
        if self.kind == "document":
            raise ValueError("document roots are created by the orchestrator")
        return self


class GroupProposalBatch(FrozenModel):
    """Strict structured-output envelope returned by one grouping pass."""

    proposals: tuple[GroupProposal, ...] = Field(default=(), max_length=32)


class GroupingEvidence(FrozenModel):
    """Bounded source-grounded evidence shown for one frontier candidate."""

    item_id: str
    kind: str
    modality: Literal["text", "image", "audio", "video", "other"]
    page_index: int | None = None
    printed_page_label: str | None = None
    bbox: BoundingBox | None = None
    text_excerpt: str | None = None
    media_mime_type: str | None = None


class GroupingCandidate(FrozenModel):
    """One current frontier node and a bounded view of its owned evidence."""

    node_id: str
    kind: str
    source_item_count: int = Field(ge=0)
    omitted_evidence_count: int = Field(ge=0)
    summary: str | None = None
    evidence: tuple[GroupingEvidence, ...] = ()


class GroupingPassContext(FrozenModel):
    """Immutable input to a proposal round; proposals never carry source text."""

    document_id: str
    source_revision: str
    pass_index: int = Field(ge=0)
    frontier_ids: tuple[str, ...] = Field(min_length=2, max_length=64)
    candidates: tuple[GroupingCandidate, ...] = Field(min_length=2, max_length=64)

    @model_validator(mode="after")
    def valid(self) -> GroupingPassContext:
        if self.frontier_ids != tuple(candidate.node_id for candidate in self.candidates):
            raise ValueError("grouping context candidates must exactly match frontier order")
        if len(set(self.frontier_ids)) != len(self.frontier_ids):
            raise ValueError("grouping context frontier IDs must be unique")
        return self


class GroupProposalGenerator(Protocol):
    """Generate structured grouping proposals from a bounded frontier window."""

    @property
    def configuration_fingerprint(self) -> str: ...

    def __call__(self, context: GroupingPassContext, /) -> GroupProposalBatch: ...


def grouping_policy_identity(
    *,
    policy_version: str,
    policy: AdaptiveGroupingPolicy,
    proposer_fingerprint: str | None,
    summary_fingerprint: str | None = None,
) -> str:
    """Bind persisted group identities to the exact proposer policy/config."""
    serialized = json.dumps(
        [
            policy_version,
            proposer_fingerprint,
            summary_fingerprint,
            policy.model_dump(mode="json"),
        ],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = sha256(serialized.encode("utf-8")).hexdigest()[:16]
    return f"{policy_version}:adaptive-{digest}"


def _modality(item: SourceItem) -> Literal["text", "image", "audio", "video", "other"]:
    if item.text is not None:
        return "text"
    media_type = item.media.mime_type.lower() if item.media is not None else ""
    if media_type.startswith("image/"):
        return "image"
    if media_type.startswith("audio/"):
        return "audio"
    if media_type.startswith("video/"):
        return "video"
    return "other"


def _node_evidence(
    node_id: str,
    *,
    nodes: dict[str, DocumentNode],
    items: dict[str, SourceItem],
) -> list[tuple[SourceItem, SourceSlice]]:
    """Return leaf-owned source slices in stable document traversal order."""
    found: list[tuple[SourceItem, SourceSlice]] = []
    pending = [node_id]
    while pending:
        node = nodes[pending.pop()]
        if node.parts:
            found.extend((items[part.item_id], part) for part in node.parts)
        else:
            pending.extend(child.node_id for child in reversed(node.children))
    return found


def build_grouping_pass_context(
    tree: DocumentTree,
    *,
    pass_index: int,
    frontier_ids: Sequence[str] | None = None,
    maximum_evidence_items_per_candidate: int = 8,
    maximum_text_chars_per_candidate: int = 2400,
) -> GroupingPassContext:
    """Build a bounded prompt view without changing or deduplicating evidence.

    The caller may pass one contiguous frontier window. Any proposal still
    refers only to frontier node IDs; exact ownership and adjacency are checked
    against the full tree before a mutation is accepted.
    """
    root = next(node for node in tree.nodes if node.node_id == tree.root_id)
    full_frontier = tuple(child.node_id for child in root.children)
    selected = tuple(frontier_ids) if frontier_ids is not None else full_frontier
    if not selected or len(selected) > 64:
        raise ValueError("grouping context window must contain between 1 and 64 nodes")
    if len(selected) < 2:
        raise ValueError("grouping proposals require at least two frontier nodes")
    try:
        start = full_frontier.index(selected[0])
    except ValueError as exc:
        raise ValueError("grouping context contains a node outside the current frontier") from exc
    if full_frontier[start : start + len(selected)] != selected:
        raise ValueError("grouping context must be a contiguous ordered frontier window")
    if maximum_evidence_items_per_candidate < 1 or maximum_text_chars_per_candidate < 1:
        raise ValueError("grouping evidence limits must be positive")

    nodes = {node.node_id: node for node in tree.nodes}
    items = {item.item_id: item for item in tree.source_items}
    candidates: list[GroupingCandidate] = []
    for candidate_id in selected:
        node = nodes[candidate_id]
        owned = _node_evidence(candidate_id, nodes=nodes, items=items)
        remaining_chars = maximum_text_chars_per_candidate
        evidence: list[GroupingEvidence] = []
        for item, source_slice in owned[:maximum_evidence_items_per_candidate]:
            excerpt: str | None = None
            if item.text is not None:
                start_offset = source_slice.start or 0
                end_offset = source_slice.end if source_slice.end is not None else len(item.text)
                source_text = item.text[start_offset:end_offset]
                if len(source_text) > remaining_chars:
                    if remaining_chars > 3:
                        head = (remaining_chars - 3) // 2
                        tail = remaining_chars - 3 - head
                        excerpt = f"{source_text[:head]}...{source_text[-tail:]}" if tail else source_text[:head]
                    else:
                        excerpt = source_text[:remaining_chars]
                else:
                    excerpt = source_text
                remaining_chars = max(0, remaining_chars - len(excerpt))
            evidence.append(
                GroupingEvidence(
                    item_id=item.item_id,
                    kind=item.kind,
                    modality=_modality(item),
                    page_index=item.location.page_index,
                    printed_page_label=item.location.printed_page_label,
                    bbox=item.location.bbox,
                    text_excerpt=excerpt,
                    media_mime_type=item.media.mime_type if item.media is not None else None,
                )
            )
        summary = (
            node.summary.text[:maximum_text_chars_per_candidate]
            if node.summary is not None
            and node.summary.status in {"generated", "verified"}
            and node.summary.text is not None
            else None
        )
        candidates.append(
            GroupingCandidate(
                node_id=node.node_id,
                kind=node.kind,
                source_item_count=len(owned),
                omitted_evidence_count=max(0, len(owned) - len(evidence)),
                summary=summary,
                evidence=tuple(evidence),
            )
        )
    return GroupingPassContext(
        document_id=tree.document_id,
        source_revision=tree.source_revision,
        pass_index=pass_index,
        frontier_ids=selected,
        candidates=tuple(candidates),
    )


def grouping_frontier_windows(
    frontier: Sequence[str], policy: AdaptiveGroupingPolicy
) -> tuple[tuple[str, ...], ...]:
    """Partition an ordered frontier into bounded, overlapping prompt windows."""
    if not policy.enabled:
        raise ValueError("cannot create grouping windows when policy is disabled")
    if len(frontier) != len(set(frontier)):
        raise ValueError("grouping frontier IDs must be unique")
    if len(frontier) < 2:
        return ()
    stride = policy.window_size - policy.window_overlap
    windows: list[tuple[str, ...]] = []
    start = 0
    while start < len(frontier) - 1:
        window = tuple(frontier[start : start + policy.window_size])
        if len(window) < 2:
            break
        windows.append(window)
        if start + len(window) >= len(frontier):
            break
        start += stride
    return tuple(windows)


def validate_grouping_window_proposals(
    *,
    frontier: Sequence[str],
    window: Sequence[str],
    proposals: Sequence[GroupProposal],
) -> tuple[GroupProposal, ...]:
    """Validate model references and adjacency before persisting one window."""
    full_frontier = tuple(frontier)
    window_ids = tuple(window)
    if not window_ids or not any(
        full_frontier[start : start + len(window_ids)] == window_ids
        for start in range(len(full_frontier) - len(window_ids) + 1)
    ):
        raise ValueError("grouping proposal window is not a contiguous frontier slice")
    positions = {node_id: index for index, node_id in enumerate(full_frontier)}
    allowed_ids = set(window_ids)
    for proposal in proposals:
        if not set(proposal.members).issubset(allowed_ids):
            raise ValueError("grouping proposal references a candidate outside its window")
        indexes = [positions.get(member, -1) for member in proposal.members]
        first = indexes[0]
        if first < 0 or indexes != list(range(first, first + len(indexes))):
            raise ValueError("adaptive grouping proposal must cover adjacent frontier nodes")
    return tuple(proposals)


def resolve_grouping_proposals(
    *, frontier: Sequence[str], proposals: Sequence[GroupProposal]
) -> tuple[GroupProposal, ...]:
    """Deduplicate and resolve overlapping, valid adjacent group suggestions.

    Conflicts resolve left-to-right, preferring the widest group at a shared
    start position. Ties use kind and member IDs, never provider ordering.
    """
    if len(frontier) != len(set(frontier)):
        raise ValueError("grouping frontier IDs must be unique")
    positions = {node_id: index for index, node_id in enumerate(frontier)}
    proposals_by_key: dict[tuple[str, tuple[str, ...]], GroupProposal] = {}
    for proposal in proposals:
        indexes = [positions.get(member, -1) for member in proposal.members]
        first = indexes[0]
        if first < 0 or indexes != list(range(first, first + len(indexes))):
            raise ValueError("adaptive grouping proposal must cover adjacent frontier nodes")
        proposals_by_key[(proposal.kind, proposal.members)] = proposal

    ordered = sorted(
        proposals_by_key.values(),
        key=lambda proposal: (
            positions[proposal.members[0]],
            -len(proposal.members),
            proposal.kind,
            proposal.members,
        ),
    )
    selected: list[GroupProposal] = []
    occupied: set[int] = set()
    for proposal in ordered:
        indexes = [positions[member] for member in proposal.members]
        if occupied.intersection(indexes):
            continue
        occupied.update(indexes)
        selected.append(proposal)
    return tuple(selected)


def propose_grouping_round(
    tree: DocumentTree,
    *,
    pass_index: int,
    policy: AdaptiveGroupingPolicy,
    generator: GroupProposalGenerator,
) -> tuple[GroupProposal, ...]:
    """Collect and deterministically resolve proposals over the current frontier."""
    root = next(node for node in tree.nodes if node.node_id == tree.root_id)
    frontier = tuple(child.node_id for child in root.children)
    windows = grouping_frontier_windows(frontier, policy)
    all_proposals: list[GroupProposal] = []
    for window in windows:
        context = build_grouping_pass_context(
            tree,
            pass_index=pass_index,
            frontier_ids=window,
            maximum_evidence_items_per_candidate=policy.maximum_evidence_items_per_candidate,
            maximum_text_chars_per_candidate=policy.maximum_text_chars_per_candidate,
        )
        batch = generator(context)
        if not isinstance(batch, GroupProposalBatch):
            raise TypeError("grouping proposal generator must return GroupProposalBatch")
        all_proposals.extend(
            validate_grouping_window_proposals(
                frontier=frontier,
                window=window,
                proposals=batch.proposals,
            )
        )
    return resolve_grouping_proposals(frontier=frontier, proposals=all_proposals)


def group_node_id(
    *,
    document_id: str,
    grouping_policy_version: str,
    proposal: GroupProposal,
) -> str:
    """Return the stable identity for one accepted grouping proposal."""
    identity = json.dumps(
        [document_id, grouping_policy_version, proposal.kind, list(proposal.members)],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return "merge:" + sha256(identity.encode()).hexdigest()[:24]


def group_node_annotations(
    *, grouping_policy_version: str, proposal: GroupProposal
) -> dict[str, str]:
    """Return stable provenance for a derived grouping node.

    Group identity is already content-addressed by the proposal. These
    annotations make the transformation and summary invalidation explicit in
    exported trees without copying mutable provider metadata into the group.
    """
    return {
        "grouping:format": "grouping-v1",
        "grouping:policy_version": grouping_policy_version,
        "grouping:members": json.dumps(
            list(proposal.members), separators=(",", ":"), ensure_ascii=False
        ),
        "grouping:summary_state": "invalidated",
    }


def merge_groups(*, document_id: str, frontier: Sequence[str], proposals: Sequence[GroupProposal], grouping_policy_version: str) -> tuple[tuple[str, ...], tuple[DocumentNode, ...]]:
    if len(frontier) != len(set(frontier)) or not document_id or not grouping_policy_version:
        raise ValueError("frontier and grouping identifiers must be unique and nonempty")
    positions = {node_id: index for index, node_id in enumerate(frontier)}
    occupied: set[int] = set()
    groups: dict[int, tuple[int, GroupProposal, str]] = {}
    for proposal in proposals:
        indexes = [positions.get(member, -1) for member in proposal.members]
        first = indexes[0]
        if first < 0 or indexes != list(range(first, first + len(indexes))) or any(index in occupied for index in indexes):
            raise ValueError("merge must use adjacent, non-overlapping frontier members")
        occupied.update(indexes)
        groups[first] = (
            indexes[-1],
            proposal,
            group_node_id(
                document_id=document_id,
                grouping_policy_version=grouping_policy_version,
                proposal=proposal,
            ),
        )
    result: list[str] = []
    created: list[DocumentNode] = []
    index = 0
    while index < len(frontier):
        group = groups.get(index)
        if group is None:
            result.append(frontier[index])
            index += 1
            continue
        last, proposal, node_id = group
        result.append(node_id)
        created.append(
            DocumentNode(
                node_id=node_id,
                kind=proposal.kind,
                children=tuple(ChildRef(node_id=member) for member in proposal.members),
                annotations=group_node_annotations(
                    grouping_policy_version=grouping_policy_version,
                    proposal=proposal,
                ),
            )
        )
        index = last + 1
    return tuple(result), tuple(created)


__all__ = [
    "GroupProposal",
    "GroupProposalBatch",
    "GroupProposalGenerator",
    "GroupingCandidate",
    "GroupingEvidence",
    "GroupingPassContext",
    "build_grouping_pass_context",
    "group_node_annotations",
    "group_node_id",
    "grouping_frontier_windows",
    "grouping_policy_identity",
    "merge_groups",
    "propose_grouping_round",
    "resolve_grouping_proposals",
    "validate_grouping_window_proposals",
]
