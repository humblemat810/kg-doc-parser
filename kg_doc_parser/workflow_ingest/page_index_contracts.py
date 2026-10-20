"""Typed data contracts shared by PageIndex parsing stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .models import GroundedSourceRecord, WorkflowIngestInput
from .semantics import SemanticNode
from .serialization import JsonObject, JsonValue

PageIndexMode = Literal[
    "heuristic",
    "fake",
    "ollama",
    "openai",
    "azure",
    "gemini",
    "vertex",
    "codex",
]
PageIndexSourceFormat = Literal["text", "markdown"]
# SECTION/SUBSECTION remain accepted for old provider payloads; materialized
# trees use HEADING plus parent links for shape.
PageIndexNodeType = Literal[
    "HEADING",
    "SECTION",
    "SUBSECTION",
    "PARAGRAPH",
    "TERM",
    "TABLE",
    "TABLE_ROW_GROUP",
    "TABLE_ROW",
]
PageIndexSourceRole = Literal["heading", "content"]
PageIndexDiagnostics = dict[str, JsonValue]


class PageIndexBlockSpec(BaseModel):
    """Recursive structural block emitted by the page-index parser."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        strict=True,
        description="Short label for this block, usually the heading text or a concise paragraph label.",
    )
    node_type: PageIndexNodeType = Field(
        description=(
            "One of HEADING, PARAGRAPH, TERM, TABLE, TABLE_ROW_GROUP, or TABLE_ROW. "
            "Preserve hierarchy through parent links; do not encode depth in the kind."
        )
    )
    excerpt: str = Field(
        strict=True,
        description="A short verbatim excerpt from the source page that grounds this block. Do not use the whole page text; keep it tight and exact.",
    )
    display_excerpt: str | None = Field(
        default=None,
        description=(
            "Optional shorter exact display excerpt. The grounded excerpt remains authoritative for source pointers; "
            "this field is only a bounded presentation hint and must be an exact substring of excerpt."
        ),
    )
    summary: str = Field(
        strict=True,
        default="",
        description="A concise factual summary of this block, grounded only in its excerpt. Leave empty when summaries are disabled.",
    )
    source_role: PageIndexSourceRole = Field(
        default="content",
        description="Whether this block is a source-bearing heading or ordinary content.",
    )
    child_nodes: list[PageIndexBlockSpec] = Field(
        default_factory=list,
        description="Direct children only. Use nested children for substructure; do not duplicate the same excerpt across siblings.",
    )


PageIndexBlockSpec.model_rebuild()


@dataclass(slots=True)
class CandidateBlock:
    block_id: str
    page_number: int
    order: int
    start_char: int
    end_char: int
    line_start: int
    line_end: int
    indent: int
    kind_hint: str
    confidence: float
    text: str
    node_type_hint: PageIndexNodeType
    title_hint: str
    heading_level: int | None = None


class BlockAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    block_id: str = Field(strict=True, description="Stable candidate block identifier.")
    parent_id: str | None = Field(
        strict=True,
        default=None,
        description="Parent block identifier or null when the block is a root.",
    )
    node_type: PageIndexNodeType = Field(description="Assigned node type for the block.")
    title: str = Field(strict=True, description="Assigned display title for the block.")
    summary: str = Field(
        strict=True,
        default="",
        description="A concise factual summary of the block, grounded only in the supplied candidate text.",
    )


class BlockAssignmentBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assignments: list[BlockAssignment] = Field(default_factory=list)


class ExcerptRefinementSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path_id: str = Field(strict=True, description="Stable tree path identifier for the block to refine.")
    excerpt: str = Field(strict=True, description="Refined excerpt proposed for the block.")


class ExcerptRefinementBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suggestions: list[ExcerptRefinementSuggestion] = Field(default_factory=list)


class HierarchicalSummaryAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path_id: str = Field(strict=True, description="Stable tree path identifier for the block to summarize.")
    summary: str = Field(strict=True, description="A concise factual summary grounded in the block excerpt.")


class HierarchicalSummaryBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assignments: list[HierarchicalSummaryAssignment] = Field(default_factory=list)


@dataclass(slots=True)
class PageIndexValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]
    fallback_reason: str | None = None


@dataclass(slots=True)
class PageIndexParseResult:
    mode: PageIndexMode
    source_format: PageIndexSourceFormat
    workflow_input: WorkflowIngestInput
    authoritative_source_map: dict[str, GroundedSourceRecord]
    parser_input_dict: JsonObject
    parser_source_map: dict[str, JsonObject]
    semantic_tree: SemanticNode
    coverage: dict[str, JsonValue]
    diagnostics: PageIndexDiagnostics


@dataclass(slots=True)
class PageUnit:
    page_number: int
    unit_id: str
    text: str


@dataclass(slots=True)
class BlockSpan:
    start_char: int
    end_char: int
    text: str
    node_type: PageIndexNodeType
    title: str
    line_start: int = 1
    line_end: int = 1
    indent: int = 0
    kind_hint: str = "paragraph"
    confidence: float = 0.8
    heading_level: int | None = None
