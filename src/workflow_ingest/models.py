from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal, Optional

from pydantic import BaseModel, Field, model_validator
from pydantic_extension.model_slicing import BackendField, FrontendField
from pydantic_extension.model_slicing.mixin import DtoField, ExcludeMode, LLMField, ModeSlicingMixin

from .semantics import HydratedTextPointer


class BoundingBox(ModeSlicingMixin, BaseModel):
    default_include_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}
    include_unmarked_for_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}

    y_min: Annotated[float, DtoField(), BackendField(), FrontendField(), LLMField()]
    x_min: Annotated[float, DtoField(), BackendField(), FrontendField(), LLMField()]
    y_max: Annotated[float, DtoField(), BackendField(), FrontendField(), LLMField()]
    x_max: Annotated[float, DtoField(), BackendField(), FrontendField(), LLMField()]


class SourceUnit(ModeSlicingMixin, BaseModel):
    default_include_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}
    include_unmarked_for_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}

    unit_id: Annotated[Optional[str], DtoField(), BackendField(), FrontendField()] = None
    modality: Annotated[
        Literal["text", "ocr_text", "non_text", "image_region", "pure_image"],
        DtoField(),
        BackendField(),
        FrontendField(),
        LLMField(),
    ]
    page_number: Annotated[Optional[int], DtoField(), BackendField(), FrontendField(), LLMField()] = None
    cluster_number: Annotated[Optional[int], DtoField(), BackendField(), FrontendField()] = None
    text: Annotated[Optional[str], DtoField(), BackendField(), FrontendField(), LLMField()] = None
    description: Annotated[Optional[str], DtoField(), BackendField(), FrontendField(), LLMField()] = None
    bbox: Annotated[Optional[BoundingBox], DtoField(), BackendField(), FrontendField(), LLMField()] = None
    source_uri: Annotated[Optional[str], DtoField(), BackendField(), FrontendField()] = None
    embedding_space: Annotated[str, DtoField(), BackendField(), FrontendField(), ExcludeMode("llm")] = "default_text"
    parser_hint_text: Annotated[
        Optional[str],
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = None
    metadata: Annotated[
        dict[str, Any],
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_content(self) -> "SourceUnit":
        if self.modality in {"text", "ocr_text"} and not (self.text and self.text.strip()):
            raise ValueError(f"{self.modality} units require non-empty text")
        if self.modality in {"non_text", "image_region", "pure_image"} and not (
            (self.description and self.description.strip()) or self.source_uri
        ):
            raise ValueError(f"{self.modality} units require description or source_uri")
        return self

    @property
    def parser_text(self) -> str:
        if self.parser_hint_text:
            return self.parser_hint_text
        if self.text:
            return self.text
        if self.description:
            return self.description
        return ""


class NormalizedPage(ModeSlicingMixin, BaseModel):
    default_include_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}
    include_unmarked_for_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}

    page_number: Annotated[int, DtoField(), BackendField(), FrontendField(), LLMField()]
    units: Annotated[list[SourceUnit], DtoField(), BackendField(), FrontendField(), LLMField()] = Field(default_factory=list)
    metadata: Annotated[
        dict[str, Any],
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = Field(default_factory=dict)


class NormalizedSourceCollection(ModeSlicingMixin, BaseModel):
    default_include_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}
    include_unmarked_for_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}

    collection_id: Annotated[str, DtoField(), BackendField(), FrontendField(), LLMField()]
    title: Annotated[str, DtoField(), BackendField(), FrontendField(), LLMField()]
    modality: Annotated[Literal["text", "ocr", "multimodal"], DtoField(), BackendField(), FrontendField(), LLMField()]
    pages: Annotated[list[NormalizedPage], DtoField(), BackendField(), FrontendField(), LLMField()] = Field(default_factory=list)
    embedding_spaces: Annotated[
        list[str],
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = Field(default_factory=lambda: ["default_text"])
    metadata: Annotated[
        dict[str, Any],
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_pages(self) -> "NormalizedSourceCollection":
        if not self.pages:
            raise ValueError("collection must include at least one page")
        return self


class WorkflowIngestInput(ModeSlicingMixin, BaseModel):
    default_include_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}
    include_unmarked_for_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}

    request_id: Annotated[str, DtoField(), BackendField(), FrontendField(), LLMField()] = "ingest-request"
    collections: Annotated[list[NormalizedSourceCollection], DtoField(), BackendField(), FrontendField(), LLMField()] = Field(default_factory=list)
    metadata: Annotated[
        dict[str, Any],
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_collections(self) -> "WorkflowIngestInput":
        if not self.collections:
            raise ValueError("at least one collection is required")
        return self

    @classmethod
    def from_text(cls, *, document_id: str, text: str, title: Optional[str] = None) -> "WorkflowIngestInput":
        return cls(
            request_id=document_id,
            collections=[
                NormalizedSourceCollection(
                    collection_id=document_id,
                    title=title or document_id,
                    modality="text",
                    pages=[
                        NormalizedPage(
                            page_number=1,
                            units=[SourceUnit(modality="text", text=text, page_number=1, cluster_number=0)],
                        )
                    ],
                )
            ],
        )


class GroundedSourceRecord(ModeSlicingMixin, BaseModel):
    default_include_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}
    include_unmarked_for_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}

    unit_id: Annotated[str, DtoField(), BackendField(), FrontendField(), LLMField()]
    collection_id: Annotated[str, DtoField(), BackendField(), FrontendField(), LLMField()]
    modality: Annotated[str, DtoField(), BackendField(), FrontendField(), LLMField()]
    page_number: Annotated[int, DtoField(), BackendField(), FrontendField(), LLMField()]
    cluster_number: Annotated[Optional[int], DtoField(), BackendField(), FrontendField()] = None
    text: Annotated[str, DtoField(), BackendField(), FrontendField(), LLMField()]
    parser_text: Annotated[str, DtoField(), BackendField(), FrontendField(), LLMField()]
    source_uri: Annotated[Optional[str], DtoField(), BackendField(), FrontendField()] = None
    embedding_space: Annotated[str, DtoField(), BackendField(), FrontendField(), ExcludeMode("llm")] = "default_text"
    participates_in_semantic_text: Annotated[
        bool,
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = True
    bbox: Annotated[Optional[BoundingBox], DtoField(), BackendField(), FrontendField(), LLMField()] = None
    metadata: Annotated[
        dict[str, Any],
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = Field(default_factory=dict)


class ValidationReport(ModeSlicingMixin, BaseModel):
    default_include_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}
    include_unmarked_for_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}

    overall_text_coverage: Annotated[float, DtoField(), BackendField(), FrontendField(), LLMField()]
    per_cluster_coverage: Annotated[dict[str, float], DtoField(), BackendField(), FrontendField(), LLMField()] = Field(default_factory=dict)
    corrected_pointer_count: Annotated[int, DtoField(), BackendField(), FrontendField(), LLMField()] = 0
    validation_notes: Annotated[list[str], DtoField(), BackendField(), FrontendField(), LLMField()] = Field(default_factory=list)


class ParseSessionState(ModeSlicingMixin, BaseModel):
    default_include_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}
    include_unmarked_for_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}

    collection_id: Annotated[str, DtoField(), BackendField(), FrontendField(), LLMField()]
    root_node_id: Annotated[str, DtoField(), BackendField(), FrontendField(), LLMField()]
    current_depth: Annotated[int, DtoField(), BackendField(), FrontendField(), LLMField()] = 0
    max_depth: Annotated[int, DtoField(), BackendField(), FrontendField(), LLMField()] = 10
    allow_review: Annotated[bool, DtoField(), BackendField(), FrontendField(), LLMField()] = True
    mode: Annotated[
        Literal["workflow_layered", "legacy_compat"],
        DtoField(),
        BackendField(),
        FrontendField(),
        LLMField(),
    ] = "workflow_layered"
    layer_attempts: Annotated[
        dict[str, int],
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = Field(default_factory=dict)
    compat_full_tree: Annotated[
        Optional[dict[str, Any]],
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = None
    metadata: Annotated[
        dict[str, Any],
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = Field(default_factory=dict)


class LayerFrontierItem(ModeSlicingMixin, BaseModel):
    default_include_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}
    include_unmarked_for_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}

    parent_node_id: Annotated[str, DtoField(), BackendField(), FrontendField(), LLMField()]
    depth: Annotated[int, DtoField(), BackendField(), FrontendField(), LLMField()]
    order: Annotated[int, DtoField(), BackendField(), FrontendField(), LLMField()] = 0


class LayerChildCandidate(ModeSlicingMixin, BaseModel):
    default_include_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}
    include_unmarked_for_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}

    node_id: Annotated[str, DtoField(), BackendField(), FrontendField(), LLMField()]
    parent_node_id: Annotated[str, DtoField(), BackendField(), FrontendField(), LLMField()]
    title: Annotated[str, DtoField(), BackendField(), FrontendField(), LLMField()]
    node_type: Annotated[str, DtoField(), BackendField(), FrontendField(), LLMField()]
    total_content_pointers: Annotated[
        list[HydratedTextPointer],
        DtoField(),
        BackendField(),
        FrontendField(),
        LLMField(),
    ] = Field(default_factory=list)
    expandable: Annotated[bool, DtoField(), BackendField(), FrontendField(), LLMField()] = True
    metadata: Annotated[
        dict[str, Any],
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = Field(default_factory=dict)


class CurrentLayerContext(ModeSlicingMixin, BaseModel):
    default_include_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}
    include_unmarked_for_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}

    depth: Annotated[int, DtoField(), BackendField(), FrontendField(), LLMField()]
    parent_node_ids: Annotated[list[str], DtoField(), BackendField(), FrontendField(), LLMField()] = Field(default_factory=list)
    parent_titles: Annotated[list[str], DtoField(), BackendField(), FrontendField(), LLMField()] = Field(default_factory=list)
    retry_count: Annotated[int, DtoField(), BackendField(), FrontendField(), LLMField()] = 0
    max_retries: Annotated[int, DtoField(), BackendField(), FrontendField(), LLMField()] = 3
    metadata: Annotated[
        dict[str, Any],
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = Field(default_factory=dict)


class CurrentLayerResult(ModeSlicingMixin, BaseModel):
    default_include_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}
    include_unmarked_for_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}

    children: Annotated[list[LayerChildCandidate], DtoField(), BackendField(), FrontendField(), LLMField()] = Field(default_factory=list)
    satisfied: Annotated[Optional[bool], DtoField(), BackendField(), FrontendField(), LLMField()] = None
    reasoning_history: Annotated[list[dict[str, Any]], DtoField(), BackendField(), FrontendField(), LLMField()] = Field(default_factory=list)
    review_rounds: Annotated[int, DtoField(), BackendField(), FrontendField(), LLMField()] = 0
    metadata: Annotated[
        dict[str, Any],
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = Field(default_factory=dict)


class CurrentLayerReview(ModeSlicingMixin, BaseModel):
    default_include_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}
    include_unmarked_for_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}

    updated_result: Annotated[
        Optional[CurrentLayerResult],
        DtoField(),
        BackendField(),
        FrontendField(),
        LLMField(),
    ] = None
    coverage_ok: Annotated[Optional[bool], DtoField(), BackendField(), FrontendField(), LLMField()] = None
    satisfied: Annotated[Optional[bool], DtoField(), BackendField(), FrontendField(), LLMField()] = None
    review_notes: Annotated[list[str], DtoField(), BackendField(), FrontendField(), LLMField()] = Field(
        default_factory=list
    )
    metadata: Annotated[
        dict[str, Any],
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = Field(default_factory=dict)


class CanonicalGraphWriteResult(ModeSlicingMixin, BaseModel):
    default_include_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}
    include_unmarked_for_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}

    persistence_mode: Annotated[
        Literal["local_debug", "server_canonical"],
        DtoField(),
        BackendField(),
        FrontendField(),
        LLMField(),
    ]
    kg_authority: Annotated[
        Literal["local", "server"],
        DtoField(),
        BackendField(),
        FrontendField(),
        LLMField(),
    ]
    canonical_write_confirmed: Annotated[bool, DtoField(), BackendField(), FrontendField(), LLMField()] = False
    nodes_written: Annotated[int, DtoField(), BackendField(), FrontendField(), LLMField()] = 0
    edges_written: Annotated[int, DtoField(), BackendField(), FrontendField(), LLMField()] = 0
    transport: Annotated[str, DtoField(), BackendField(), FrontendField(), LLMField()] = "direct_runtime"
    server_parser_used: Annotated[bool, DtoField(), BackendField(), FrontendField(), LLMField()] = False
    error: Annotated[Optional[str], DtoField(), BackendField(), FrontendField(), LLMField()] = None


class WorkflowExportBundle(ModeSlicingMixin, BaseModel):
    default_include_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}
    include_unmarked_for_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}

    graph_payload: Annotated[dict[str, Any], DtoField(), BackendField(), FrontendField(), LLMField()]
    authoritative_source_map: Annotated[
        dict[str, GroundedSourceRecord],
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ]
    embedding_spaces: Annotated[list[str], DtoField(), BackendField(), FrontendField(), ExcludeMode("llm")] = Field(default_factory=list)
    consolidation_candidates: Annotated[
        list[dict[str, Any]],
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = Field(default_factory=list)
    retrieval_metadata: Annotated[
        dict[str, Any],
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = Field(default_factory=dict)
    persistence_mode: Annotated[
        Literal["local_debug", "server_canonical"],
        DtoField(),
        BackendField(),
        FrontendField(),
        LLMField(),
    ] = "local_debug"
    kg_authority: Annotated[
        Literal["local", "server"],
        DtoField(),
        BackendField(),
        FrontendField(),
        LLMField(),
    ] = "local"
    canonical_write_confirmed: Annotated[
        bool,
        DtoField(),
        BackendField(),
        FrontendField(),
        LLMField(),
    ] = False
    parser_owner: Annotated[
        Literal["local"],
        DtoField(),
        BackendField(),
        FrontendField(),
        LLMField(),
    ] = "local"
    server_parser_used: Annotated[
        bool,
        DtoField(),
        BackendField(),
        FrontendField(),
        LLMField(),
    ] = False
    canonical_write_result: Annotated[
        Optional[CanonicalGraphWriteResult],
        DtoField(),
        BackendField(),
        FrontendField(),
        LLMField(),
    ] = None
    persisted_to_knowledge_engine: Annotated[
        bool,
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = False


class IngestRunHandle(ModeSlicingMixin, BaseModel):
    default_include_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}
    include_unmarked_for_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}

    run_id: Annotated[str, DtoField(), BackendField(), FrontendField(), LLMField()]
    workflow_id: Annotated[str, DtoField(), BackendField(), FrontendField(), LLMField()]
    execution_mode: Annotated[
        Literal["direct_runtime", "server_canonical_client"],
        DtoField(),
        BackendField(),
        FrontendField(),
        LLMField(),
    ]


class IngestRunResult(ModeSlicingMixin, BaseModel):
    default_include_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}
    include_unmarked_for_modes: ClassVar[set[str]] = {"dto", "backend", "frontend", "llm"}

    handle: Annotated[IngestRunHandle, DtoField(), BackendField(), FrontendField(), LLMField()]
    status: Annotated[
        Literal["succeeded", "failed", "failure", "suspended"],
        DtoField(),
        BackendField(),
        FrontendField(),
        LLMField(),
    ]
    bundle: Annotated[Optional[WorkflowExportBundle], DtoField(), BackendField(), FrontendField(), LLMField()] = None
    final_state: Annotated[
        dict[str, Any],
        DtoField(),
        BackendField(),
        FrontendField(),
        ExcludeMode("llm"),
    ] = Field(default_factory=dict)
