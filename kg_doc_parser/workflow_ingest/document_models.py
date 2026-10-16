"""Standalone, source-grounded document grouping contracts.

These models deliberately do not inherit from Kogwistar persistence models.
They are the immutable evidence view produced by OCR before semantic parsing.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from hashlib import sha256
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BoundingBox(FrozenModel):
    x0: float
    y0: float
    x1: float
    y1: float
    coordinate_system: Literal["pixels", "relative", "points"]

    @model_validator(mode="after")
    def valid(self) -> BoundingBox:
        if not (self.x0 < self.x1 and self.y0 < self.y1):
            raise ValueError("bounding box must have positive area")
        if self.coordinate_system == "relative" and not all(0 <= v <= 1 for v in (self.x0, self.y0, self.x1, self.y1)):
            raise ValueError("relative coordinates must lie in [0, 1]")
        return self


class SourceLocation(FrozenModel):
    page_index: int | None = Field(default=None, ge=0)
    printed_page_label: str | None = None
    bbox: BoundingBox | None = None
    start_seconds: float | None = Field(default=None, ge=0)
    end_seconds: float | None = Field(default=None, ge=0)
    structural_path: str | None = None
    accuracy: Literal["character", "word", "line", "region", "page", "time", "path"]

    @model_validator(mode="after")
    def valid(self) -> SourceLocation:
        has_time = self.start_seconds is not None or self.end_seconds is not None
        if has_time and (self.start_seconds is None or self.end_seconds is None or self.end_seconds <= self.start_seconds):
            raise ValueError("time range must have positive endpoints")
        if self.bbox is not None and self.page_index is None:
            raise ValueError("bbox requires page_index")
        if self.page_index is None and not has_time and not self.structural_path:
            raise ValueError("location requires a page, time range, or path")
        if self.accuracy in {"character", "word", "line", "region"} and self.bbox is None:
            raise ValueError("spatial precision requires a bounding box")
        if self.accuracy == "time" and not has_time:
            raise ValueError("time accuracy requires a time range")
        if self.accuracy == "path" and not self.structural_path:
            raise ValueError("path accuracy requires structural_path")
        return self


class MediaRef(FrozenModel):
    uri: str = Field(min_length=1)
    mime_type: str = Field(min_length=3)
    sha256_hex: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_length: int = Field(ge=0)


class OCRMetadata(FrozenModel):
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    configuration_fingerprint: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)


class SourceItem(FrozenModel):
    item_id: str = Field(min_length=1)
    kind: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z][A-Za-z0-9_.:/-]*$")
    location: SourceLocation
    text: str | None = None
    media: MediaRef | None = None
    ocr_metadata: OCRMetadata | None = None
    annotations: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def one_payload(self) -> SourceItem:
        if (self.text is None) == (self.media is None):
            raise ValueError("source item requires exactly one text or media payload")
        if self.text == "":
            raise ValueError("empty OCR text is not an evidence item")
        return self


class SourceSlice(FrozenModel):
    item_id: str = Field(min_length=1)
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)
    separator_before: str = ""

    @model_validator(mode="after")
    def valid(self) -> SourceSlice:
        if self.start is None or self.end is None:
            if self.start is not None or self.end is not None:
                raise ValueError("text slices require a nonempty half-open interval")
        elif self.end <= self.start:
            raise ValueError("text slices require a nonempty half-open interval")
        if self.separator_before and not self.separator_before.isspace():
            raise ValueError("slice separators must be whitespace")
        return self


class EvidenceSlice(FrozenModel):
    """The exact source-owned payload exposed to a grounded summarizer."""

    item_id: str = Field(min_length=1)
    kind: str = Field(min_length=1, max_length=128)
    location: SourceLocation
    text: str | None = None
    media: MediaRef | None = None
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)
    separator_before: str = ""

    @model_validator(mode="after")
    def valid(self) -> EvidenceSlice:
        if self.separator_before and not self.separator_before.isspace():
            raise ValueError("evidence slice separators must be whitespace")
        if (self.text is None) == (self.media is None):
            raise ValueError("evidence slice requires exactly one text or media payload")
        if self.text is None:
            if self.start is not None or self.end is not None:
                raise ValueError("media evidence slices cannot have text offsets")
        elif self.start is None or self.end is None or self.end <= self.start:
            raise ValueError("text evidence slices require a nonempty interval")
        return self


class ChildRef(FrozenModel):
    node_id: str = Field(min_length=1)
    separator_before: str = ""

    @model_validator(mode="after")
    def valid(self) -> ChildRef:
        if self.separator_before and not self.separator_before.isspace():
            raise ValueError("child separators must be whitespace")
        return self


class Summary(FrozenModel):
    status: Literal["pending", "generated", "verified", "failed", "skipped"]
    text: str | None = None
    evidence_item_ids: tuple[str, ...] = ()
    provider_model: str | None = None
    prompt_revision: str | None = None
    input_fingerprint: str | None = None
    evidence_slice_fingerprint: str | None = None

    @model_validator(mode="after")
    def grounded(self) -> Summary:
        if self.status in {"generated", "verified"} and not (self.text and self.evidence_item_ids and self.provider_model and self.prompt_revision and self.input_fingerprint):
            raise ValueError("generated summaries require text, evidence, and provenance")
        if self.status not in {"generated", "verified"} and self.text is not None:
            raise ValueError("unavailable summaries cannot contain text")
        if len(set(self.evidence_item_ids)) != len(self.evidence_item_ids):
            raise ValueError("summary evidence items must be unique")
        return self


class DocumentReferenceMention(FrozenModel):
    """A source-grounded reference phrase kept outside the source partition.

    This is deliberately not a graph edge.  Resolution to a document node is
    optional and belongs to a later projection or application layer.
    """

    mention_id: str = Field(min_length=1)
    source: SourceSlice
    additional_sources: tuple[SourceSlice, ...] = ()
    target_kind: Literal["page", "section", "figure", "table"]
    target_label: str = Field(min_length=1, max_length=128)
    target_node_id: str | None = None
    status: Literal["pending", "resolved"] = "pending"

    @property
    def sources(self) -> tuple[SourceSlice, ...]:
        """Return all exact source slices in their declared reading order."""
        return (self.source, *self.additional_sources)

    @model_validator(mode="after")
    def valid(self) -> DocumentReferenceMention:
        if self.source.start is None or self.source.end is None:
            raise ValueError("reference mentions require a text source interval")
        if self.status == "resolved" and not self.target_node_id:
            raise ValueError("resolved reference mentions require a target node")
        return self


class SummaryPolicy(FrozenModel):
    """Bound optional derived-summary work independently of OCR."""

    enabled: bool = True
    # Change this whenever the summary provider, model, prompt, or other
    # output-affecting generation configuration changes. It is cache identity.
    generation_fingerprint: str = Field(default="summary-v1", min_length=1)
    eligible_kinds: tuple[str, ...] = (
        "table",
        "figure",
        "section",
        "chapter",
        "document",
    )
    minimum_source_items: int = Field(default=2, ge=1)
    minimum_source_chars: int = Field(default=1, ge=0)
    minimum_information_gain_chars: int = Field(default=1, ge=0)
    maximum_input_tokens: int = Field(default=2048, ge=1)
    maximum_targets: int = Field(default=32, ge=1)

    @model_validator(mode="after")
    def valid(self) -> SummaryPolicy:
        if len(set(self.eligible_kinds)) != len(self.eligible_kinds):
            raise ValueError("summary policy kinds must be unique")
        return self


class DocumentNode(FrozenModel):
    node_id: str = Field(min_length=1)
    kind: str = Field(min_length=1, max_length=128)
    parts: tuple[SourceSlice, ...] = ()
    children: tuple[ChildRef, ...] = ()
    summary: Summary | None = None
    annotations: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def shape(self) -> DocumentNode:
        if bool(self.parts) == bool(self.children):
            raise ValueError("node must contain parts or children, exclusively")
        # A document root is a structural wrapper and may retain one final
        # meaningful group.  Semantic grouping nodes remain strictly
        # branching: they must have at least two children.
        if self.children and len(self.children) < 2 and self.kind != "document":
            raise ValueError("unary grouping nodes are not allowed")
        if len({c.node_id for c in self.children}) != len(self.children):
            raise ValueError("duplicate child in grouping node")
        return self


class DocumentTree(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    document_id: str = Field(min_length=1)
    source_revision: str = Field(min_length=1)
    tree_revision: int = Field(ge=1)
    root_id: str = Field(min_length=1)
    source_items: tuple[SourceItem, ...]
    nodes: tuple[DocumentNode, ...]
    references: tuple[DocumentReferenceMention, ...] = ()

    @model_validator(mode="after")
    def validate_tree(self) -> DocumentTree:
        items = {item.item_id: item for item in self.source_items}
        nodes = {node.node_id: node for node in self.nodes}
        if len(items) != len(self.source_items) or len(nodes) != len(self.nodes) or set(items) & set(nodes):
            raise ValueError("duplicate or overlapping source and node identities")
        root = nodes.get(self.root_id)
        if root is None or root.kind != "document":
            raise ValueError("document root is missing or has the wrong kind")
        reference_ids = [reference.mention_id for reference in self.references]
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError("duplicate document reference mention")
        for reference in self.references:
            source_keys: set[tuple[str, int | None, int | None]] = set()
            for source in reference.sources:
                key = (source.item_id, source.start, source.end)
                if key in source_keys:
                    raise ValueError("reference mention contains duplicate source slice")
                source_keys.add(key)
                item = items.get(source.item_id)
                if item is None or item.text is None:
                    raise ValueError("reference mention must point to a text source item")
                if source.end is None or source.end > len(item.text):
                    raise ValueError("reference mention exceeds source item")
            if reference.target_node_id is not None and reference.target_node_id not in nodes:
                raise ValueError("reference mention target node does not exist")
        parents: dict[str, int] = defaultdict(int)
        intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
        media: dict[str, int] = defaultdict(int)
        visiting: set[str] = set()
        seen: set[str] = set()

        def visit(node_id: str) -> set[str]:
            if node_id in visiting:
                raise ValueError("cycle in document tree")
            if node_id in seen:
                return set()
            node = nodes.get(node_id)
            if node is None:
                raise ValueError(f"dangling node reference: {node_id}")
            visiting.add(node_id)
            covered: set[str] = set()
            if node.parts:
                for part in node.parts:
                    item = items.get(part.item_id)
                    if item is None:
                        raise ValueError(f"unknown source item: {part.item_id}")
                    covered.add(part.item_id)
                    if item.text is None:
                        if part.start is not None or part.end is not None:
                            raise ValueError("media slices cannot have text offsets")
                        media[item.item_id] += 1
                    else:
                        if part.start is None or part.end is None or part.end > len(item.text):
                            raise ValueError("text slice exceeds source item")
                        intervals[item.item_id].append((part.start, part.end))
            else:
                for child in node.children:
                    parents[child.node_id] += 1
                    covered.update(visit(child.node_id))
            if node.summary and set(node.summary.evidence_item_ids) - covered:
                raise ValueError("summary cites evidence outside its subtree")
            visiting.remove(node_id)
            seen.add(node_id)
            return covered

        visit(self.root_id)
        if seen != set(nodes) or any(parents[node_id] != 1 for node_id in nodes if node_id != self.root_id):
            raise ValueError("orphan or multiply-parented node")
        for item_id, item in items.items():
            if item.text is None:
                if media[item_id] != 1:
                    raise ValueError(f"media item must be covered exactly once: {item_id}")
                continue
            cursor = 0
            for start, end in sorted(intervals[item_id]):
                if start != cursor:
                    raise ValueError(f"source gap or overlap in item {item_id}")
                cursor = end
            if cursor != len(item.text):
                raise ValueError(f"incomplete source coverage: {item_id}")
        for node_id, node in nodes.items():
            if node.summary is None or node.summary.status not in {"generated", "verified"}:
                continue
            evidence = node.summary.evidence_item_ids
            payload = [items[item_id].model_dump(mode="json") for item_id in sorted(evidence)]
            expected = sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if node.summary.input_fingerprint != expected:
                raise ValueError(f"summary fingerprint is stale for node {node_id}")
            if node.summary.evidence_slice_fingerprint is not None:
                expected_slices = self.summary_slice_fingerprint(node_id, evidence)
                if node.summary.evidence_slice_fingerprint != expected_slices:
                    raise ValueError(f"summary slice fingerprint is stale for node {node_id}")
        return self

    def source_fingerprint(self) -> str:
        payload = [item.model_dump(mode="json") for item in self.source_items]
        return sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def summary_input_fingerprint(self, node_id: str, evidence_item_ids: tuple[str, ...] | None = None) -> str:
        """Return the fingerprint a generated summary must record for a node."""
        nodes = {node.node_id: node for node in self.nodes}
        items = {item.item_id: item for item in self.source_items}
        node = nodes.get(node_id)
        if node is None:
            raise KeyError(node_id)
        covered: set[str] = set()

        def visit(current_id: str) -> None:
            current = nodes[current_id]
            if current.parts:
                covered.update(part.item_id for part in current.parts)
            else:
                for child in current.children:
                    visit(child.node_id)

        visit(node_id)
        selected = tuple(sorted(evidence_item_ids if evidence_item_ids is not None else covered))
        if set(selected) - covered:
            raise ValueError("summary evidence must be within the node subtree")
        payload = [items[item_id].model_dump(mode="json") for item_id in selected]
        return sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def summary_slice_fingerprint(
        self,
        node_id: str,
        evidence_item_ids: tuple[str, ...] | None = None,
    ) -> str:
        """Fingerprint the exact ordered source slices in a subtree.

        ``summary_input_fingerprint`` remains the compatibility fingerprint
        for existing persisted summaries. New summaries additionally record
        this boundary so two nodes that use different slices of the same
        source item cannot share an accepted summary accidentally.
        """
        nodes = {node.node_id: node for node in self.nodes}
        node = nodes.get(node_id)
        if node is None:
            raise KeyError(node_id)
        allowed = set(evidence_item_ids) if evidence_item_ids is not None else None
        payload: list[dict[str, object]] = []

        def visit(current_id: str) -> None:
            current = nodes[current_id]
            if current.parts:
                for part_index, part in enumerate(current.parts):
                    if allowed is None or part.item_id in allowed:
                        payload.append(
                            {
                                "node_id": current_id,
                                "part_index": part_index,
                                "item_id": part.item_id,
                                "start": part.start,
                                "end": part.end,
                                "separator_before": part.separator_before,
                            }
                        )
                return
            for child in current.children:
                visit(child.node_id)

        visit(node_id)
        return sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()

    def evidence_slices(
        self,
        node_id: str,
        evidence_item_ids: tuple[str, ...] | None = None,
    ) -> tuple[EvidenceSlice, ...]:
        """Return exact ordered source slices owned by a node subtree."""
        nodes = {node.node_id: node for node in self.nodes}
        items = {item.item_id: item for item in self.source_items}
        if node_id not in nodes:
            raise KeyError(node_id)
        allowed = set(evidence_item_ids) if evidence_item_ids is not None else None
        result: list[EvidenceSlice] = []

        def visit(current_id: str) -> None:
            current = nodes[current_id]
            if current.parts:
                for part in current.parts:
                    if allowed is not None and part.item_id not in allowed:
                        continue
                    item = items[part.item_id]
                    if item.text is None:
                        result.append(
                            EvidenceSlice(
                                item_id=item.item_id,
                                kind=item.kind,
                                location=item.location,
                                media=item.media,
                                separator_before=part.separator_before,
                            )
                        )
                    else:
                        if part.start is None or part.end is None:
                            raise ValueError("text evidence requires explicit source bounds")
                        result.append(
                            EvidenceSlice(
                                item_id=item.item_id,
                                kind=item.kind,
                                location=item.location,
                                text=item.text[part.start : part.end],
                                start=part.start,
                                end=part.end,
                                separator_before=part.separator_before,
                            )
                        )
                return
            for child in current.children:
                visit(child.node_id)

        visit(node_id)
        return tuple(result)

    def render_preview(self) -> str:
        """Render source text for inspection without changing canonical evidence."""
        items = {item.item_id: item for item in self.source_items}
        nodes = {node.node_id: node for node in self.nodes}

        def render(node_id: str) -> str:
            node = nodes[node_id]
            if node.parts:
                output = ""
                for part in node.parts:
                    item = items[part.item_id]
                    output += part.separator_before
                    output += item.text[part.start:part.end] if item.text is not None else f"[media:{item.item_id}]"
                return output
            return "".join(link.separator_before + render(link.node_id) for link in node.children)

        return render(self.root_id)


_REFERENCE_PATTERN = re.compile(
    r"\b(?:see|refer\s+to|referenced\s+in)\s+"
    r"(?P<target_kind>page|section|figure|table)\s+"
    r"(?P<target_label>[A-Za-z0-9][A-Za-z0-9._-]*)",
    re.IGNORECASE,
)


def extract_reference_mentions(tree: DocumentTree) -> tuple[DocumentReferenceMention, ...]:
    """Extract only explicit, source-grounded document-reference phrases.

    The result is metadata outside the tree's primary coverage partition.  It
    never assigns a target node or creates a semantic relation.
    """

    mentions: list[DocumentReferenceMention] = []
    text_items = tuple(item for item in tree.source_items if item.text is not None)
    if not text_items:
        return ()

    # Search one deterministic reading-order stream, while retaining the item
    # and character offset for every source character. Separators are synthetic
    # and therefore never become part of an authoritative source slice.
    stream_parts: list[str] = []
    stream_locations: list[tuple[str, int] | None] = []
    for item_index, item in enumerate(text_items):
        assert item.text is not None
        if item_index:
            stream_parts.append(" ")
            stream_locations.append(None)
        stream_parts.append(item.text)
        stream_locations.extend((item.item_id, offset) for offset in range(len(item.text)))
    stream = "".join(stream_parts)

    for match in _REFERENCE_PATTERN.finditer(stream):
        locations = stream_locations[match.start() : match.end()]
        grouped: list[SourceSlice] = []
        for location in locations:
            if location is None:
                continue
            item_id, offset = location
            if grouped and grouped[-1].item_id == item_id and grouped[-1].end == offset:
                grouped[-1] = grouped[-1].model_copy(update={"end": offset + 1})
            else:
                grouped.append(SourceSlice(item_id=item_id, start=offset, end=offset + 1))
        if not grouped:
            continue
        target_kind = cast(
            Literal["page", "section", "figure", "table"],
            match.group("target_kind").lower(),
        )
        target_label = match.group("target_label")
        source_keys = ",".join(
            f"{source.item_id}:{source.start}:{source.end}" for source in grouped
        )
        digest = sha256(f"{source_keys}:{target_kind}:{target_label}".encode()).hexdigest()[:24]
        mentions.append(
            DocumentReferenceMention(
                mention_id=f"mention:{digest}",
                source=grouped[0],
                additional_sources=tuple(grouped[1:]),
                target_kind=target_kind,
                target_label=target_label,
            )
        )
    return tuple(mentions)


__all__ = [
    "BoundingBox",
    "ChildRef",
    "DocumentNode",
    "DocumentReferenceMention",
    "DocumentTree",
    "FrozenModel",
    "MediaRef",
    "OCRMetadata",
    "SourceItem",
    "SourceLocation",
    "SourceSlice",
    "Summary",
    "extract_reference_mentions",
]
