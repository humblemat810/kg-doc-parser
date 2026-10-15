"""Deterministic validation of bottom-up grouping proposals."""
from __future__ import annotations

import json
from collections.abc import Sequence
from hashlib import sha256

from pydantic import Field, model_validator

from .document_models import ChildRef, DocumentNode, FrozenModel


class GroupProposal(FrozenModel):
    kind: str = Field(min_length=1, max_length=128)
    members: tuple[str, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def valid(self) -> GroupProposal:
        if len(set(self.members)) != len(self.members):
            raise ValueError("a merge proposal cannot repeat a member")
        if self.kind == "document":
            raise ValueError("document roots are created by the orchestrator")
        return self


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
        created.append(DocumentNode(node_id=node_id, kind=proposal.kind, children=tuple(ChildRef(node_id=member) for member in proposal.members)))
        index = last + 1
    return tuple(result), tuple(created)


__all__ = ["GroupProposal", "group_node_id", "merge_groups"]
