"""
Data Models & Schema Definitions
================================

This module defines the core data models for the Optical Character Recognition (OCR) layer
and documents the implicit schema of the Resultant Knowledge Graph.

Physical Layer (OCR)
--------------------
- SplitPage: Represents a single page of a document.
- OCRClusterResponse: Contains detected text clusters and non-text objects.
- TextCluster: A bounding box with text content.
  - text: str (The recognized text content)
  - cluster_number: int (Unique ID on the page)
  - bb_x_min, bb_x_max, bb_y_min, bb_y_max: float (Bounding box coordinates)

    [PDF/Image] -> [OCR Service] -> [SplitPage] -> [TextCluster]

Resultant Knowledge Graph Structure
-----------------------------------
The semantic parsing pipeline (`kg_doc_parser.semantic_document_splitting_layerwise_edits`) transforms
these physical models into a Knowledge Graph.

    [SemanticNode] (Entity)
          |
          +--- mentions ---+
          |                |
          |        [TextCluster] (Grounding)
          |
          +--- HAS_CHILD --> [SemanticNode] (Sub-entity)

Graph Schema (Nodes & Edges)
----------------------------
Nodes (Entities):
  - id: UUID
  - label: Title/Heading
  - type: "entity"
  - subtype: "TEXT_FLOW" | "KEY_VALUE_PAIR" | "TABLE"
  - mentions: Link to source `TextCluster`s (provenance)

Edges (Relationships):
  - id: UUID
  - source_id: Parent Node UUID
  - target_id: Child Node UUID
  - relation: "HAS_CHILD"
  - type: "relationship"
"""
from collections import Counter

if True:
    import logging
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.NullHandler())
    logger.debug("loading models")
from typing import Literal, Self, TypeAlias

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator
from pydantic_extension.model_slicing import (
    DtoType,
    ModeSlicingMixin,
)
from pydantic_extension.model_slicing.mixin import DtoField


def _meaningful_ordering_error(expected_ids: set[int], ordering: list[int]) -> str:
    """Describe an invalid reading-order permutation for retry feedback."""

    missing_ids = sorted(expected_ids - set(ordering))
    unknown_ids = sorted(set(ordering) - expected_ids)
    duplicate_ids = sorted(
        cluster_id
        for cluster_id, count in Counter(ordering).items()
        if count > 1
    )
    return (
        "meaningful_ordering must be the authoritative reading-order permutation "
        "of every text and non-text cluster exactly once; "
        f"expected_cluster_ids={sorted(expected_ids)!r}; "
        f"returned_ordering={ordering!r}; "
        f"missing_cluster_ids={missing_ids!r}; "
        f"unknown_cluster_ids={unknown_ids!r}; "
        f"duplicate_ordering_ids={duplicate_ids!r}"
    )

JsonPrimitive = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]  # noqa: UP040 - PyPy 3.11 support
JsonObject: TypeAlias = dict[str, JsonValue]  # noqa: UP040 - PyPy 3.11 support


class ModelInvariantError(ValueError):
    """Raised when an OCR model cannot satisfy its structural contract."""


#========================= OCR DOC

# pre-validation model
class box_2d(BaseModel): # text box for ocr, the id here is not a database unique id but a unique id for identified object
    box_2d: list[int] = Field(description = 'box y min, x min, y max and x max')
    label : str = Field(description = 'text in the box')
    id: int  = Field(description = 'id of the text box in the page, autoincrement from 0')
class NonText_box_2d(BaseModel): # text box for ocr, the id here is not a database unique id but a unique id for identified object
    """Recognised meaningful objects other than OCR characters, include image, figures. """
    description: str = Field(description='the description or summary of the non-OCR object')
    box_2d: list[int] = Field(description = 'box y min, x min, y max and x max')
    id: int = Field(description="per page unique number of the cluster, starting from 0")

# post validation model
class TextCluster(ModeSlicingMixin, BaseModel):
    """a text cluster along with spatial information"""
    text: DtoType[str] = Field(description='the text content of the text cluster')
    bb_x_min: DtoType[float]  = Field(description='the bounding box x min in pixel coordinate of the text_cluster. ')
    bb_x_max: DtoType[float]  = Field(description='the bounding box x max in pixel coordinate of the text_cluster. ')
    bb_y_min: DtoType[float]  = Field(description='the bounding box y min in pixel coordinate of the text_cluster. ')
    bb_y_max: DtoType[float]  = Field(description='the bounding box y max in pixel coordinate of the text_cluster. ')
    cluster_number: int = Field(description="per page unique number of the cluster, starting from 0")
class NonTextCluster(ModeSlicingMixin, BaseModel):
    """Recognised meaningful objects other than OCR characters, include image, figures. """
    description: DtoType[str] = Field(description='the description or summary of the non-OCR object')
    bb_x_min: DtoType[float]  = Field(description='the bounding box x min in pixel coordinate of the non-OCR object. ')
    bb_x_max: DtoType[float]  = Field(description='the bounding box x max in pixel coordinate of the non-OCR object. ')
    bb_y_min: DtoType[float]  = Field(description='the bounding box y min in pixel coordinate of the non-OCR object. ')
    bb_y_max: DtoType[float]  = Field(description='the bounding box y max in pixel coordinate of the non-OCR object. ')
    cluster_number: int = Field(description="per page unique number of the cluster, starting from 0")

class OCRClusterResponse(ModeSlicingMixin, BaseModel):
    """id/ cluster number must be all unique, for example, one of the ocr boxes_2d used id='1', 
       the first image box id (cluster numebr) will be '2', the next signature will be '3' """
    OCR_text_clusters: DtoType[list[TextCluster]] = Field(description="the OCR text results. Share cluster number uniqueness with non-OCR objects. Include emoji or unicode text")
    non_text_objects:  DtoType[list[NonTextCluster]] = Field(description="the non-OCR object results. Share cluster number uniqueness with OCR texts. ")
    is_empty_page: DtoType[bool | None] = Field(default = False, description="true if the whole page is empty without recognisable text.")
    printed_page_number: DtoType[str | None] = Field(description='the page number identified from OCR texts, can be in form of roman numerals such as "i", "ii", "iii", "iv"...; ' 
                    'Arabic numeral such as 1, 2, 3... or letter such as "a", "b", "c"...\n'
                    'Sometimes the are surrounded by symbols such as "- 1 -", "- 2 -"'
                    r"Can be null/none if there is no page order assigned and printed and found in the scanned texts. Do not assign page number. Only use page number found.")
    meaningful_ordering : DtoType[list[int]] = Field(description="The authoritative reading order of all text and non-text clusters, from first to last meaningful item. Must contain every cluster number exactly once and no others. ")
    page_x_min : DtoType[float]=Field(description='the page x min in pixel coordinate. ')
    page_x_max : DtoType[float]=Field(description='the page x max in pixel coordinate. ')
    page_y_min : DtoType[float]=Field(description='the page y min in pixel coordinate. ')
    page_y_max : DtoType[float]=Field(description='the page y max in pixel coordinate. ')
    estimated_rotation_degrees : DtoType[float]=Field(description='the page estimated rotation degree using right hand rule. ')
    incomplete_words_on_edge: DtoType[bool] = Field(description='If there is any text being incomplete due to the scan does not scan the edges properly. ')
    incomplete_text: DtoType[bool]  = Field(description='Any incomplete text')
    data_loss_likelihood: DtoType[float] = Field(description='The likelihood (range from 0.0 to 1.0 inclusive) that the page has lost information by missing the scan data on the edges of the page.' )
    scan_quality: DtoType[Literal['low', 'medium', 'high']] = Field(description='The image quality of the scan. All qualities exclude signatures. '
                                                                                      '"low", "medium" or "high". '
                                'low: text barely legible. medium: Legible with non smooth due to pixelation. high: texts are easily and highly identifiable. ' )
    contains_table: DtoType[bool] = Field(description='Whether this page contains table. ')
    unresolved_item_count: DtoType[int] = Field(
        default=0,
        ge=0,
        description=(
            'Number of provider-recognised items that could not be represented '
            'with a usable bounding box or modality record.'
        ),
    )
    source_item_count: DtoType[int | None] = Field(
        default=None,
        ge=0,
        description=(
            'Total number of source items observed by the provider. When set, '
            'the sum of represented text/non-text clusters and unresolved items '
            'must equal this count.'
        ),
    )
    spatial_accuracy: DtoType[Literal['page', 'region', 'line', 'word', 'character']] = Field(
        default='page',
        description=(
            'The declared precision of the bounding boxes. Use page for a '
            'synthetic page-wide fallback; providers with item-level boxes '
            'must explicitly declare region or finer precision.'
        ),
    )

    @model_validator(mode='after')
    def check_cluster_meaningful_ordering_agreement(self) -> Self:
        all_clusters = self.OCR_text_clusters + self.non_text_objects
        if bool(self.is_empty_page) == (len(all_clusters) > 0):
            raise ValueError(
                f"is_empty_page value {self.is_empty_page} disagree with OCR clusters "
                f"len={len(all_clusters)}"
            )
        all_ids = [item.cluster_number for item in all_clusters]
        overlap_id = {
            cluster_id
            for cluster_id, count in Counter(all_ids).items()
            if count > 1
        }
        if overlap_id:
            raise ValueError(f"cluster number from non_text_objects block and ocr text blocks must be ALL distinct. Overlapped ids: {list(overlap_id)}")
        all_ids_set = set(all_ids)
        if len(self.meaningful_ordering) != len(all_ids_set) or set(self.meaningful_ordering) != all_ids_set:
            raise ValueError(_meaningful_ordering_error(all_ids_set, self.meaningful_ordering))
        if self.source_item_count is not None:
            accounted_item_count = len(all_clusters) + self.unresolved_item_count
            if self.source_item_count != accounted_item_count:
                raise ValueError(
                    "OCR source item accounting does not reconcile "
                    f"({self.source_item_count} observed versus "
                    f"{accounted_item_count} represented or unresolved)"
                )
        return self

    

class OCRClusterResponseBc(OCRClusterResponse):
    # backward compability layer, can use by overriding fields with union of previous versions
    
    pass
    # OCR_text_clusters: list[TextCluster | TextCluster_yolo_bb] = Field(description="the OCR text results, " # type: ignore
    #                                                                    "prefer min max bounding box to centre width height style")
    
class SplitPageMeta(BaseModel):
    ocr_model_name: str = Field(description="model that perform OCR")
    ocr_datetime: float = Field(description="unix timestamp when ocr is performed")
    ocr_json_version: str = Field(description = "the model does the OCR") 
    @field_validator('ocr_json_version', mode = "before")
    def version_to_str(cls, v: object) -> str:
        return str(v)
class SplitPage(OCRClusterResponseBc):
    # model not for LLM response
    pdf_page_num: int
    metadata: SplitPageMeta
    refined_version: OCRClusterResponse[DtoField] | None = Field(default = None, description = "refined processed/ grouped/ merged version of ocr text clusters. ")
    def model_dump(self, *arg: object, **kwarg: object) -> JsonObject:
        return self.to_doc()
    def dump_raw(self, *arg: object, **kwarg: object) -> JsonObject:
        options = {**kwarg, "exclude": ["refined_version"]}
        return super().model_dump(*arg, **options)
    def dump_supercede_parse(self, *arg: object, **kwarg: object) -> JsonObject:
        options = {**kwarg, "exclude": ["refined_version", "metadata"]}
        return super().model_dump(*arg, **options)
    @model_validator(mode="after")
    def roundtrip_invariant(self, info: ValidationInfo) -> "SplitPage":
        # Context may be None if caller didn't pass it
        ctx = info.context or {}
        # Re-entrancy guard
        if ctx.get("_roundtrip_active", False):
            return self
        # First time from here
        self.to_doc()
        # Mark active for the nested validation
        nested_ctx = dict(ctx)
        nested_ctx["_roundtrip_active"] = True

        dumped = self.dump_raw()

        # IMPORTANT: pass context so nested validation sees the flag
        again = self.__class__.model_validate(dumped, context=nested_ctx)
        
        
        # Optional: assert equivalence (pick your definition)
        if again != self:
            raise ValueError("Roundtrip invariant failed: dump->validate changed the model")

        return self
    def to_doc(self) -> JsonObject:
        """Model to llm one-way serializer with manual slicing logic, can refactor using sliced view
        with some token saving logic. 
        """
        target = self.refined_version or self
        ocr_cluster = target.OCR_text_clusters
        non_ocr_cluster = target.non_text_objects
        if target.contains_table:
            id_sorted_text_cluster = []
            cluster_numbers = {i.cluster_number for i in ocr_cluster}
            if len(cluster_numbers) != len(ocr_cluster):
                raise ValueError("OCR cluster numbers must be unique")
            cluster_lookup_by_number = {i.cluster_number : i for i in (ocr_cluster + non_ocr_cluster#+ target.signature_blocks
                                                                       )}
            for i in target.meaningful_ordering:
                c_p = cluster_lookup_by_number.get(i)
                if c_p is None:
                    raise KeyError(f"{i} does not exist")
                cluster_dump: JsonObject = c_p.model_dump()
                cluster_dump.pop("cluster_number")
                id_sorted_text_cluster.append(cluster_dump)
            others = cluster_numbers - set(target.meaningful_ordering)
            for i in others:
                cluster_dump: JsonObject
                cluster_dump = cluster_lookup_by_number[i].model_dump()
                cluster_dump.pop("cluster_number")
                id_sorted_text_cluster.append(cluster_dump)
            c_return: JsonObject = {}
            c_return['pdf_page_num'] = self.pdf_page_num
            c_return['printed_page_number'] = self.printed_page_number
            c_return['OCR_text_clusters'] = id_sorted_text_cluster
            c_return['contains_table'] = self.contains_table
            return c_return
        else:
            # This legacy projection contains text only. The raw SplitPage
            # retains every modality and its full authoritative ordering.
            text_by_number = {cluster.cluster_number: cluster for cluster in ocr_cluster}
            texts = '\n'.join(
                text_by_number[cluster_number].text
                for cluster_number in self.meaningful_ordering
                if cluster_number in text_by_number
            )
            c_return: JsonObject = {}
            c_return['pdf_page_num'] = self.pdf_page_num
            c_return['printed_page_number'] = self.printed_page_number
            c_return['text'] = texts
            return c_return
