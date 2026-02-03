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

    [PDF/Image] -> [OCR Service] -> [SplitPage] -> [TextCluster]

Resultant Knowledge Graph Structure
-----------------------------------
The semantic parsing pipeline (`src.semantic_document_splitting_layerwise_edits`) transforms
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
if True:
    import logging
    import os
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.NullHandler())
    logger.debug("loading models")
from typing import List, Literal, Optional, Dict, Any, Type, TypeAlias, Union, Annotated, ClassVar
from pydantic import BaseModel, Field, model_validator, field_validator, ValidationInfo

from pydantic_extension.model_slicing import (ModeSlicingMixin, NotMode, FrontendField, BackendField, LLMField,
                DtoType,
                BackendType,
                FrontendType,
                LLMType,
                use_mode)
from pydantic_extension.model_slicing.mixin import ExcludeMode, DtoField
JsonPrimitive = Union[str, int, float, bool, None]
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
    is_empty_page: DtoType[Optional[bool]] = Field(default = False, description="true if the whole page is empty without recognisable text.")
    printed_page_number: DtoType[Optional[str]] = Field(description='the page number identified from OCR texts, can be in form of roman numerals such as "i", "ii", "iii", "iv"...; ' 
                    'Arabic numeral such as 1, 2, 3... or letter such as "a", "b", "c"...\n'
                    'Sometimes the are surrounded by symbols such as "- 1 -", "- 2 -"'
                    r"Can be null/none if there is no page order assigned and printed and found in the scanned texts. Do not assign page number. Only use page number found.")
    meaningful_ordering : DtoType[list[int]] = Field(description="The correct meaningful ordering of the identified text clusters. Must cover all OCR_text_clusters once and only once. ")
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

    @model_validator(mode='after')
    def check_cluster_meaningful_ordering_agreement(self):
        assert bool(self.is_empty_page) ^ (len(self.OCR_text_clusters) > 0), f"is_empty_page value {self.is_empty_page} disagree with OCR_text_clusters len={len(self.OCR_text_clusters)}"
        overlap_id = set(i.cluster_number for i in self.OCR_text_clusters).intersection(set(i.cluster_number for i in self.non_text_objects))
        if overlap_id:
            raise ValueError(f"cluster number from non_text_objects block and ocr text blocks must be ALL distinct. Overlapped ids: {list(overlap_id)}")
        try:
            if not (len(self.meaningful_ordering) == len(set(self.meaningful_ordering))): # <= len(self.OCR_text_clusters)):
                raise ValueError("meaningful_order must cover each text cluster at most once")
        except Exception as e:
            raise e
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
    def version_to_str(cls, v):
        return str(v)
class SplitPage(OCRClusterResponseBc):
    # model not for LLM response
    pdf_page_num: int
    metadata: SplitPageMeta
    refined_version: Optional[OCRClusterResponse[DtoField]] = Field(default = None, description = "refined processed/ grouped/ merged version of ocr text clusters. ")
    def model_dump(self, *arg, **kwarg):
        return self.to_doc()
    def dump_raw(self, *arg, **kwarg):
        return super(SplitPage, self).model_dump(exclude = ["refined_version"], *arg, **kwarg)
    def dump_supercede_parse(self, *arg, **kwarg):
        return super(SplitPage, self).model_dump(exclude = ["refined_version", "metadata"], *arg, **kwarg)
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
    def to_doc(self):
        """Model to llm one-way serializer with manual slicing logic, can refactor using sliced view
        with some token saving logic. 
        """
        target = self.refined_version or self
        ocr_cluster = target.OCR_text_clusters
        non_ocr_cluster = target.non_text_objects
        if target.contains_table:
            id_sorted_text_cluster = []
            cluster_numbers = (i.cluster_number for i in ocr_cluster)
            assert len(set(cluster_numbers)) == len(ocr_cluster)
            cluster_lookup_by_number = {i.cluster_number : i for i in (ocr_cluster + non_ocr_cluster#+ target.signature_blocks
                                                                       )}
            for i in target.meaningful_ordering:
                c_p = cluster_lookup_by_number.get(i)
                if c_p is None:
                    raise KeyError(f"{i} does not exist")
                cluster_dump: dict = c_p.model_dump()
                cluster_dump.pop("cluster_number")
                id_sorted_text_cluster.append(cluster_dump)
            others = (set(cluster_numbers) - set(target.meaningful_ordering))
            for i in others:
                cluster_dump: dict
                cluster_dump = cluster_lookup_by_number[i].model_dump()
                cluster_dump.pop("cluster_number")
                id_sorted_text_cluster.append(cluster_dump)
            c_return = {}
            c_return['pdf_page_num'] = self.pdf_page_num
            c_return['printed_page_number'] = self.printed_page_number
            c_return['OCR_text_clusters'] = id_sorted_text_cluster
            c_return['contains_table'] = self.contains_table
            return c_return
        else:
            # isSorted = True
            expected_next = 0
            i_ocr = 0
            # i_sig = 0
            i_non_ocr = 0
            ocr_clus_nums = sorted([i.cluster_number for i in ocr_cluster])
            non_clus_nums = sorted([i.cluster_number for i in non_ocr_cluster])
            # sig_clus_nums = sorted([i.cluster_number for i in target.signature_blocks])
            # expected_next = min(ocr_clus_nums + sig_clus_nums)
            if ocr_clus_nums:
                start_num = min(ocr_clus_nums + non_clus_nums)
                assert start_num in [0, 1], "only allow 0-indexed based or 1-indexed based cluster numbers"
                expected_next = start_num
                while expected_next < start_num + (len(ocr_clus_nums) + len(non_clus_nums)):
                    if (i_ocr <len(ocr_cluster)) and expected_next == ocr_cluster[i_ocr].cluster_number:
                        i_ocr += 1
                    # elif (i_sig < len(target.signature_blocks)) and expected_next == target.signature_blocks[i_sig].cluster_number:
                        # i_sig += 1
                    elif (i_non_ocr < len(non_ocr_cluster)) and expected_next == non_ocr_cluster[i_non_ocr].cluster_number:
                        i_non_ocr += 1
                    else:
                        raise Exception(f'expected index {expected_next} not found in both ocr_cluster nor signature_blocks')
                    expected_next += 1

            
            id_sorted_text_cluster = sorted(ocr_cluster, key=lambda x : x.cluster_number )
            assert len(id_sorted_text_cluster) == len(set(i.cluster_number for i in id_sorted_text_cluster))
            is_normal = True
            shift = 0 # try zero indexing sanity
            c : TextCluster #| TextCluster_yolo_bb
            for i, c in enumerate(id_sorted_text_cluster):
                
                if c.cluster_number != i:
                    is_normal = False
                    break
            c2: TextCluster #| TextCluster_yolo_bb
            if not is_normal:
                is_normal = True
                shift = -1 # try one-indexing sanity
                for i, c2 in enumerate(id_sorted_text_cluster):
                    if c2.cluster_number != i+1:
                        is_normal = False
                        break
            # sig_num = set([i.cluster_number for i in self.signature_blocks])
            assert set(self.meaningful_ordering) <= set(i.cluster_number for i in (self.OCR_text_clusters))
            if is_normal:
                # is sorted list where index is order + shift
                texts = '\n'.join(id_sorted_text_cluster[i+shift].text for i in self.meaningful_ordering)
            else:
                # general case
                tcd = {x.cluster_number:x   for x in id_sorted_text_cluster}
                texts = '\n'.join(tcd[i].text for i in self.meaningful_ordering)
            c_return = {}
            c_return['pdf_page_num'] = self.pdf_page_num
            c_return['printed_page_number'] = self.printed_page_number
            c_return['text'] = texts
            return c_return
