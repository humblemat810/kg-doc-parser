"""Standalone, source-grounded document grouping contracts.

These models deliberately do not inherit from Kogwistar persistence models.
They are the immutable evidence view produced by OCR before semantic parsing.
"""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from typing import Literal

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
    def valid(self) -> "BoundingBox":
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
    def valid(self) -> "SourceLocation":
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
    kind: str = Field(min_length=1, max_length=128)
    location: SourceLocation
    text: str | None = None
    media: MediaRef | None = None
    ocr_metadata: OCRMetadata | None = None

    @model_validator(mode="after")
    def one_payload(self) -> "SourceItem":
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
    def valid(self) -> "SourceSlice":
        if (self.start is None) != (self.end is None) or (self.start is not None and self.end <= self.start):
            raise ValueError("text slices require a nonempty half-open interval")
        if self.separator_before and not self.separator_before.isspace():
            raise ValueError("slice separators must be whitespace")
        return self


class ChildRef(FrozenModel):
    node_id: str = Field(min_length=1)
    separator_before: str = ""


class Summary(FrozenModel):
    status: Literal["pending", "generated", "verified", "failed", "skipped"]
    text: str | None = None
    evidence_item_ids: tuple[str, ...] = ()
    provider_model: str | None = None
    prompt_revision: str | None = None
    input_fingerprint: str | None = None

    @model_validator(mode="after")
    def grounded(self) -> "Summary":
        if self.status in {"generated", "verified"} and not (self.text and self.evidence_item_ids and self.provider_model and self.prompt_revision and self.input_fingerprint):
            raise ValueError("generated summaries require text, evidence, and provenance")
        if self.status not in {"generated", "verified"} and self.text is not None:
            raise ValueError("unavailable summaries cannot contain text")
        return self


class DocumentNode(FrozenModel):
    node_id: str = Field(min_length=1)
    kind: str = Field(min_length=1, max_length=128)
    parts: tuple[SourceSlice, ...] = ()
    children: tuple[ChildRef, ...] = ()
    summary: Summary | None = None

    @model_validator(mode="after")
    def shape(self) -> "DocumentNode":
        if bool(self.parts) == bool(self.children):
            raise ValueError("node must contain parts or children, exclusively")
        if self.children and len(self.children) < 2:
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

    @model_validator(mode="after")
    def validate_tree(self) -> "DocumentTree":
        items = {item.item_id: item for item in self.source_items}
        nodes = {node.node_id: node for node in self.nodes}
        if len(items) != len(self.source_items) or len(nodes) != len(self.nodes) or set(items) & set(nodes):
            raise ValueError("duplicate or overlapping source and node identities")
        root = nodes.get(self.root_id)
        if root is None or root.kind != "document":
            raise ValueError("document root is missing or has the wrong kind")
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
        return self

    def source_fingerprint(self) -> str:
        payload = [item.model_dump(mode="json") for item in sorted(self.source_items, key=lambda value: value.item_id)]
        return sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

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


__all__ = ["BoundingBox", "ChildRef", "DocumentNode", "DocumentTree", "FrozenModel", "MediaRef", "OCRMetadata", "SourceItem", "SourceLocation", "SourceSlice", "Summary"]
