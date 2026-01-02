
# ==============================================================================
# PHASE 1: SETUP - MODELS AND IMPORTS
# ==============================================================================
import json
import re
from typing import Annotated, List, TypeAlias, Union, Literal, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field, ValidationError, validator, field_validator
from uuid import UUID, uuid1
from collections import deque
from pydantic import BaseModel, Field, model_validator
import math
from rapidfuzz.distance import LCSseq
from datetime import datetime
from typing import Callable, TypeVar, ParamSpec, cast
from joblib import Memory

P = ParamSpec("P")
R = TypeVar("R")
import functools
def joblib_memory_cached(memory: Memory, *arg, **kwarg):
    def wrapper(fn: Callable[P, R]) -> Callable[P, R]:
        return cast(Callable[P, R], memory.cache(fn, *arg, **kwarg))
    return wrapper


def partition(iterable, predicate):
    t1, t2 = [], []
    for x in iterable:
        (t1 if predicate(x) else t2).append(x)
    return t1, t2
from pydantic_extension.model_slicing import DtoType, BackendField, BackendType, FrontendField, FrontendType
from pydantic_extension.model_slicing.mixin import DtoField, LLMField, LLMType, ExcludeMode, ModeSlicingMixin
class HydratedTextPointer(ModeSlicingMixin, BaseModel):
    source_cluster_id: Annotated[str, FrontendField(), BackendField(), DtoField(), LLMField()] = Field(description="The unique ID of the source text block (e.g., 'p1_c0').")
    start_char: Annotated[int, FrontendField(), BackendField(), DtoField(), LLMField()] = Field(description="The starting character index within the source text block.")
    end_char: Annotated[int, FrontendField(), BackendField(), DtoField(), LLMField()] = Field(description="The inclusive ending character index. Use -1 for 'to the end'.")
    verbatim_text: Annotated[str, FrontendField(), BackendField(), DtoField(), LLMField()] = Field(description="The exact text of this fragment. This MUST match the text at the specified pointer location.")
    validation_method: Optional[Annotated[str, BackendField(), LLMField(), ExcludeMode("llm")]] = Field(None, description="The exact text of this fragment. This MUST match the text at the specified pointer location.")
    # backend and llm used, default to dump include, but ExcludeMode("llm") specified must be excluded when dumping to llm mode
    # --------------------------
    # pointer -> ref dict
    # --------------------------
    def to_ref_dict(
        self,
        *,
        doc_id: str,
        insertion_method: str = "semantic_document_parser_v1",
        base_doc_url: Optional[str] = None,
        page_num: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Convert this pointer to a plain dict reference, no external model required.
        We encode the cluster id in document_page_url as ...#{source_cluster_id},
        so we can reverse later.
        """
        if not base_doc_url:
            base_doc_url = f"doc://{doc_id}"

        start_page = page_num or 1
        end_page = start_page

        # keep the -1 sentinel reversible by using a large number
        end_char = self.end_char if self.end_char != -1 else 10**9

        return {
            "doc_id": doc_id,
            "collection_page_url": base_doc_url,
            "document_page_url": f"{base_doc_url}#{self.source_cluster_id}",
            "insertion_method": insertion_method,
            "start_page": start_page,
            "end_page": end_page,
            "start_char": self.start_char,
            "end_char": end_char,
            "snippet": self.verbatim_text[:400],
        }

    # --------------------------
    # ref dict -> pointer
    # --------------------------
    @classmethod
    def from_ref_dict(cls, ref: Dict[str, Any]) -> "HydratedTextPointer":
        """
        Rebuild a pointer from a plain dict ref.
        We rely on document_page_url ending with '#{cluster_id}'.
        If missing, we fallback to UNKNOWN_CLUSTER.
        """
        doc_url = ref.get("document_page_url") or ""
        if ref.get("source_cluster_id"):
            source_cluster_id = ref.get("source_cluster_id") or "UNKNOWN_CLUSTER"
        elif "#" in doc_url:
            source_cluster_id = doc_url.split("#", 1)[1]
        else:
            source_cluster_id = "UNKNOWN_CLUSTER"

        start_char = int(ref.get("start_char", 0))
        end_char_raw = int(ref.get("end_char", 0))
        end_char = -1 if end_char_raw >= 10**8 else end_char_raw

        return cls(
            source_cluster_id=source_cluster_id,
            start_char=start_char,
            end_char=end_char,
            verbatim_text=ref.get("snippet") or "",
        )
HydratedTextPointerLLM : TypeAlias = HydratedTextPointer['llm']
from typing import ClassVar
class LLMChildNodeResponse(ModeSlicingMixin, BaseModel):
    default_include_modes:  ClassVar= {"dto"}
    default_exclude_modes: ClassVar = {"llm"}
    include_unmarked_for_modes: ClassVar = {"dto", "frontend", "backend"}
    parent_node_id: Annotated[str, DtoField(),BackendField(),LLMField()] = Field(description="The UUID string of the parent node this child belongs to.")
    node_type: BackendField[LLMField[DtoField[Literal["TEXT_FLOW", "KEY_VALUE_PAIR", "TABLE"]]]] = Field(description="The semantic type of the child node.")
    title: BackendField[LLMField[DtoField[str]]] = Field(description="The title, key, or a concise summary of the child section.")
    pointers: BackendField[LLMField[DtoField[List[HydratedTextPointerLLM]]]] = Field(description="A list of rich, hydrated pointers that physically constitute this logical child node."
                                                            "For key value pair, can split the key and value into 2 different pointers. "
                                                            "If a sentence has been broken down into multiple text_clusters, leave them separatedly included in this list of pointers. ")
    # value_pointers: Optional[List[HydratedTextPointer]] = Field(None, description="(Used for KEY_VALUE_PAIR only, Null/None otherwise) Pointers to the value part of the node.")
    @field_validator("parent_node_id")
    def _check_parent_in_prev_layer(cls, v):
        allowed_ids = [str(i.node_id) for i in current_level_nodes.get()]
        if str(v) not in allowed_ids:
            raise ValueError("parent_node_id not in current allowed parent layer")
        return v
    def to_BE(self):
        return LLMChildNodeResponseBE.model_validate(self.model_dump())
    # @model_validator(mode="after")
    # def _check_consistency(self):
    #     if self.node_type == "KEY_VALUE_PAIR":
    #         if self.value_pointers is None:
    #             raise ValueError('value_pointers cannot be None if node type is KEY_VALUE_PAIR')
    #         else:
    #             assert len(self.pointers) == len(self.value_pointers), 'pointer length must be same as value pointer for KEY_VALUE_PAIR'
    #     else:
    #         assert not self.value_pointers, "value_pointers must be None if node type is not KEY_VALUE_PAIR"
    #     return self
class LLMChildNodeResponseBE(ModeSlicingMixin, BaseModel):
    default_include_modes:  ClassVar= {"dto"}
    default_exclude_modes: ClassVar = {"llm"}
    include_unmarked_for_modes: ClassVar = {"dto", "frontend", "backend"}
    id: Annotated[UUID, BackendField()] = Field(default_factory = uuid1, description="The UUID string of the child belongs to.") # no Field to avoid being accidentally passed to LLM
    parent_node_id: Annotated[str, DtoField(), BackendField()] = Field(description="The UUID string of the parent node this child belongs to.")
    node_type: Annotated[Literal["TEXT_FLOW", "KEY_VALUE_PAIR", "TABLE"], DtoField(), BackendField()] = Field(description="The semantic type of the child node.")
    title: Annotated[str, DtoField(), BackendField()] = Field(description="The title, key, or a concise summary of the child section.")
    pointers: Annotated[List[HydratedTextPointer], DtoField(), BackendField()] = Field(description="A list of rich, hydrated pointers that physically constitute this logical child node.")
    # value_pointers: Optional[List[HydratedTextPointer]] = Field(None, description="(For KEY_VALUE_PAIR only) Pointers to the value part of the node.")

class LevelParsingResponse(ModeSlicingMixin, BaseModel):
    include_unmarked_for_modes: ClassVar = {"dto", "frontend", "backend", "llm"}
    children: List[LLMChildNodeResponse] = Field(description='a list of parsing response, prefer gently narrow down scope.')
LLMLevelResponse: TypeAlias = LevelParsingResponse["llm"]
LLMLevelResponseBE: TypeAlias = LevelParsingResponse["backend"]
# class LLMLevelResponse(BaseModel):
#     children: List[LLMChildNodeResponse] = Field(description='a list of parsing response, prefer gently narrow down scope.')

# class LLMLevelResponseBE(BaseModel):
    # children: List[LLMChildNodeResponseBE]


class SemanticNode(BaseModel):
    node_id: UUID = Field(default_factory=uuid1)
    parent_id: Optional[UUID] = None
    node_type: Literal["DOCUMENT_ROOT", "TEXT_FLOW", "KEY_VALUE_PAIR", "TABLE"] = Field("TEXT_FLOW")
    title: str
    total_content_pointers: List[HydratedTextPointer]
    child_nodes: List['SemanticNode'] = Field([])

    # -------------------------------------------------
    # 🔍 Search descriptor builder (unchanged)
    # -------------------------------------------------
    def _build_search_descriptors(self, source_map: Dict[str, Dict]) -> Dict[str, Any]:
        text = "".join(p.verbatim_text for p in self.total_content_pointers)
        title = self.title or ""
        lowered = title.lower()
        tokens = re.findall(r"[a-zA-Z0-9_/.-]+", lowered)
        numbers = re.findall(r"\b\d[\d,./:-]*\b", text)
        dates = re.findall(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", text)
        return {
            "search": {
                "title_terms": tokens,
                "numbers": numbers,
                "dates": dates,
                "fulltext": text[:3000],
            }
        }

    # -------------------------------------------------
    # 🪶 to_kg_node — emit real Node dict with references
    # -------------------------------------------------
    def to_kg_node(
        self,
        source_map: Dict[str, Dict],
        doc_id: str,
        *,
        insertion_method: str = "semantic_document_parser_v1",
        namespace: str = "docs",
        base_doc_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Serialize this SemanticNode into a well-formed KGE Node dictionary."""

        # 1️ Convert pointers → ref dicts using built-in reversible mapping
        references = [
            p.to_ref_dict(
                doc_id=doc_id,
                insertion_method=insertion_method,
                base_doc_url=base_doc_url,
            )
            for p in self.total_content_pointers
        ]
        if not references:
            references = [{
                "doc_id": doc_id,
                "collection_page_url": base_doc_url or f"doc://{doc_id}",
                "document_page_url": f"{base_doc_url or 'doc://'+doc_id}#UNKNOWN",
                "insertion_method": insertion_method,
                "start_page": 1,
                "end_page": 1,
                "start_char": 0,
                "end_char": 0,
                "snippet": self.title[:400],
            }]

        # # 2 Metadata block (LLM/trace info)
        # metadata = {
        #     "pointers": [p.model_dump() for p in self.total_content_pointers],
        #     "doc_id": doc_id,
        #     "node_type_src": self.node_type,
        #     "created_at": datetime.utcnow().isoformat() + "Z",
        #     "insertion_method": insertion_method,
        # }
        # metadata.update(self._build_search_descriptors(source_map))

        # 3 Text summary
        display_text = "".join(p.verbatim_text for p in self.total_content_pointers)[:1000]

        # 4 Compose Node dict (aligned with your KGE models)
        return {
            "id": str(self.node_id),
            "label": self.title,
            "type": "entity",          # for persistence
            "subtype": self.node_type, # semantic subtype
            "namespace": namespace,
            "summary": display_text,
            "references": references,
            # "metadata": metadata,
        }

    # -------------------------------------------------
    #  from_kg_node — reconstruct from Node dict
    # -------------------------------------------------
    @classmethod
    def from_kg_node(cls, kg_node: Dict[str, Any]) -> 'SemanticNode':
        """Rebuild a SemanticNode from a KGE Node dictionary."""
        # meta = {} # kg_node.get("metadata") or {}
        # pointers_raw = meta.get("pointers") or []

        # if not present, reconstruct pointers from references
        # if not pointers_raw and (refs := kg_node.get("references")):
        refs = kg_node.get("references")
        if refs is None:
            raise ValueError('kg_node must have field "refs"')
        pointers_raw = [
            HydratedTextPointer.from_ref_dict(r).model_dump() for r in refs
        ]

        return cls(
            node_id=UUID(kg_node["id"]),
            parent_id=None,  # re-linked later from edges
            node_type=kg_node.get("subtype", "TEXT_FLOW"),
            title=kg_node.get("label", ""),
            total_content_pointers=[
                HydratedTextPointer.model_validate(p) for p in pointers_raw
            ],
            child_nodes=[],
        )

    # -------------------------------------------------
    #  flatten_tree_to_kge_payload — recursive traversal
    # -------------------------------------------------
    def flatten_tree_to_kge_payload(
        self,
        source_map: Dict[str, Dict],
        doc_id: str,
        *,
        insertion_method: str = "semantic_document_parser_v1",
        namespace: str = "docs",
        base_doc_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Flatten this SemanticNode hierarchy into a full KGE upsert payload.
        Returns:
            {
              "nodes": [...],
              "edges": [...]
            }
        """
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        def walk(node: 'SemanticNode'):
            # add node
            nodes.append(
                node.to_kg_node(
                    source_map,
                    doc_id,
                    insertion_method=insertion_method,
                    namespace=namespace,
                    base_doc_url=base_doc_url,
                )
            )
            # add edges
            for child in node.child_nodes:
                ref = (
                    child.total_content_pointers[0].to_ref_dict(
                        doc_id=doc_id,
                        insertion_method=insertion_method,
                        base_doc_url=base_doc_url,
                    )
                    if child.total_content_pointers else None
                )
                edges.append(
                    {
                        "id": str(uuid1()),
                        "subject_id": str(node.node_id),
                        "predicate": "HAS_CHILD",
                        "object_id": str(child.node_id),
                        "type": "relationship",
                        "namespace": namespace,
                        "references": [ref] if ref else [],
                        "metadata": {
                            "doc_id": doc_id,
                            "insertion_method": insertion_method,
                        },
                    }
                )
                walk(child)

        walk(self)
        return {
            "doc_id": doc_id,
            "namespace": namespace,
            "insertion_method": insertion_method,
            "nodes": nodes,
            "edges": edges,
        }
SemanticNode.model_rebuild()
from contextvars import ContextVar
current_level_nodes = ContextVar("allowed_choices", default=[])

def reject_self_recursion(parent_type: str, child: LLMChildNodeResponse) -> bool:
    """
    Return True if the child should be rejected for self-recursion.
    Current rule: forbid KEY_VALUE_PAIR directly under KEY_VALUE_PAIR.
    """
    return parent_type == "KEY_VALUE_PAIR" and child.node_type == "KEY_VALUE_PAIR"

def not_self_like(parent: SemanticNode, child: LLMChildNodeResponse) -> bool:
    """
    Reject if child mirrors the parent (same type, same title, same spans).
    """
    same_title = _normalize_title(child.title) == _normalize_title(parent.title)
    same_type  = child.node_type == parent.node_type
    parent_spans = {_pkey(p) for p in parent.total_content_pointers}
    child_spans  = {_pkey(p) for p in child.pointers}
    same_spans = child_spans == parent_spans
    return not (same_title and same_type and same_spans)
# ==============================================================================
# PHASE 2: PRE-PROCESSING
# ==============================================================================
def prepare_document_for_llm(doc_dict: Dict) -> Tuple[Dict, Dict[str, Dict]]:
    # Simple restructure of input format
    filename = list(doc_dict.keys())[0]
    pages_data = doc_dict[filename]
    source_cluster_map = {}
    for page in pages_data:
        if 'pdf_page_num' not in page or 'OCR_text_clusters' not in page:
            continue
        page_num = page['pdf_page_num']
        for i, cluster in enumerate(page['OCR_text_clusters']):
            cluster_id = f"p{page_num}_c{i}"
            cluster['id'] = cluster_id
            source_cluster_map[cluster_id] = cluster
        for i, cluster in enumerate(page['non_text_objects']):
            cluster_id = f"p{page_num}_c{i}"
            cluster['id'] = cluster_id
            source_cluster_map[cluster_id] = cluster
    return {"document_filename": filename, "pages": pages_data}, source_cluster_map

# ==============================================================================
# PHASE 3: LLM PROCESSING - ROBUST LAYER-WISE (BFS)
# ==============================================================================

PROMPT_POINTER_CORRECTION = """
**ROLE:**
You are an expert AI data validation specialist. Your task is to correct errors in previously extracted data.

**TASK:**
You will be given a list of child nodes that failed validation. The `pointers` for these nodes likely do not match their `verbatim_text`.
For EACH node in the list, re-analyze the full document context and provide a corrected list of `HydratedTextPointer` objects.

**INSTRUCTIONS:**
1.  **Focus on Correction:** Your primary goal is to fix the `start_char` and `end_char` for each pointer so that it perfectly matches its `verbatim_text`.
2.  **Maintain Structure:** Do not change the `parent_node_id`, `node_type`, or `title`. Only correct the pointers.
3.  **Return Only Corrected Nodes:** If you successfully correct a node, include it in your output. If you cannot fix a node, omit it.

**OUTPUT FORMAT:**
You MUST output a valid JSON object conforming to the `LLMLevelResponse` schema, containing only the nodes you were able to successfully correct.

**FULL DOCUMENT JSON (for context):**
```json
{full_document_json}
```
**NODES REQUIRING CORRECTION:**
```json
{nodes_to_correct_json}
"""

# PROMPT_BATCH_SUBDIVIDER = """
# **ROLE:**
# You are a meticulous AI document analyst. Your task is to physically locate and report on all text fragments that make up the immediate children of the parent sections provided.

# **TASK:**
# For EACH parent section in the list, identify its IMMEDIATE children. For each child, you must identify every single text fragment that constitutes it.

# **OUTPUT FORMAT:**
# You MUST output a valid JSON object conforming to the `LLMLevelResponse` schema.
# For each child you find, you must provide a list of `HydratedTextPointer` objects in the `pointers` field.
# Each `HydratedTextPointer` object MUST contain:
# 1.  `source_cluster_id`: The ID of the text block where the fragment is located.
# 2.  `start_char` and `end_char`: The precise start and end character indices of the fragment, relative to the start of its source text block.
# 3.  `verbatim_text`: The exact text of that fragment. This MUST match the text at the specified pointer location.

# **EXAMPLE for a child "Clause A.1" that is split across two text blocks:**
# ```json
# {
#   "parent_node_id": "uuid-of-parent",
#   "node_type": "TEXT_FLOW",
#   "title": "Clause A.1",
#   "pointers": [
#     {
#       "source_cluster_id": "p1_c5",
#       "start_char": 50,
#       "end_char": 150,
#       "verbatim_text": "This is the first part of the clause..."
#     },
#     {
#       "source_cluster_id": "p2_c1",
#       "start_char": 0,
#       "end_char": 80,
#       "verbatim_text": "...and this is the second part of the clause."
#     }
#   ]
# **FULL DOCUMENT JSON (for context):**
# ```json
# {full_document_json}
# ```

# parent layer nodes:
# ```json
# {parent_sections_json}
# ```
# }"""
from string import Template
import json
from uuid import UUID

PROMPT_BATCH_SUBDIVIDER = Template(
r"""
**ROLE:**
You are a meticulous AI document analyst. Your task is to physically locate and report on all text fragments that make up the immediate children of the parent sections provided.
Some sections might be broken down in other iterations. Do not attempt to add them back. Only consider current given parent nodes should be broken down or not.
**TASK:**
For EACH parent section in the list, identify its IMMEDIATE children. For each child, you must identify every single text fragment that constitutes it.
Prioritize at breaking into textflows. Key-value pairs can then be discovered when the test flow is granular later.
For example, if a node contains section-A and section B, each contain sub-section, being coarser is correct and preferred. Because next iteration will further breakdown from section level to subsection level.
If a node is already very fine-grained and handlable, you can then start parsing it into key value pairs.
Leave small node unparsed when it convey a single meaning.
When breaking down, all children combined together must preserve their parent's meaning.
Each parent must have more than 1 child or no child. Explanation: Having 1 child mean that child has same meaning and granularity as parent and it is not break down.
If a parent has no children, it just mean it is granular enough. It is still preserved, not discarded.
If all nodes are granular enough. Just give no children in response to indicate completion

**OUTPUT FORMAT:**
You MUST output a valid JSON object conforming to the `LLMLevelResponse` schema.
For each child you find, provide a list of `HydratedTextPointer` objects in `pointers`.

**EXAMPLE for a child "Clause A.1" that is split across two text blocks:**
```json
{
  "parent_node_id": "uuid-of-parent",
  "node_type": "TEXT_FLOW",
  "title": "Clause A.1",
  "pointers": [
    {
      "source_cluster_id": "p1_c5",
      "start_char": 50,
      "end_char": 150,
      "verbatim_text": "This is the first part of the clause..."
    },
    {
      "source_cluster_id": "p2_c1",
      "start_char": 0,
      "end_char": 80,
      "verbatim_text": "...and this is the second part of the clause."
    }
  ]
}


FULL DOCUMENT JSON (for context):

```json

$full_document_json
```
PARENT SECTIONS TO PROCESS:

```json

$parent_sections_json
```
"""
)
# PROMPT_SUBDIVIDER = """
# **ROLE:**
# You are a micro-analyst specializing in deconstructing text sections.
# **TASK:**
# Analyze ONLY the provided `text_to_analyze`. Your goal is to identify its IMMEDIATE children. Do not find grandchildren. If there are no clear sub-sections, return an empty list.
# **CONTEXT:**
# The parent section is titled: "{parent_title}"
# **OUTPUT FORMAT:**
# Provide a JSON list, where each object conforms to the `LLMChildNodeResponse` schema.
# - For each child, provide its `node_type`, `title`, `verbatim_text`, and relative `pointers`.
# - For `KEY_VALUE_PAIR`, also provide the `value_text`.
# - The pointers' `start_char` and `end_char` MUST be relative to the beginning of the `text_to_analyze`.
# **TEXT TO ANALYZE:**
# {text_to_analyze}
# """
# MASTER_PROMPT_HIERARCHICAL = """
# **ROLE:**
# You are an expert AI document analyst that understands document structure as a hierarchy.

# **TASK:**
# Analyze the provided document JSON. Your task is to identify ALL semantic sections and sub-sections, from the highest-level clauses down to individual points. You will represent this entire hierarchy as a single FLAT LIST of nodes.

# **INSTRUCTIONS:**
# 1.  **Discover the Hierarchy:** Read the entire document to understand its structure (sections, sub-sections, key-value pairs).
# 2.  **Generate Unique IDs:** For every semantic part you identify (e.g., "Section 1", "Clause 1.1", "Fee Amount"), create a `FlatSemanticNode` object and assign it a new, unique `node_id`.
# 3.  **Establish Parent-Child Links:** For each node, set its `parent_id` to the `node_id` of the section that contains it. Top-level sections should have a `parent_id` of `null`.
# 4.  **Define Pointers:** For every node, provide the `direct_content_pointers` that correspond ONLY to its title or key text (e.g., for "1.1 Definitions", the pointers cover just that heading text).

# **OUTPUT FORMAT:**
# You MUST output a single, valid JSON object that conforms to the `DocumentParseResult` schema. The output MUST be a flat list of nodes, not a nested tree.
# """
# PROMPT_BATCH_SUBDIVIDER = """
# **ROLE:**
# You are a parallel-processing AI document analyst with full contextual awareness.

# **TASK:**
# You will be given the complete JSON of a source document and a specific list of parent sections to analyze. For EACH parent section in the list, your task is to identify its IMMEDIATE children.

# **INSTRUCTIONS:**
# 1.  **Use the Full Document for Context:** Refer to the `full_document_json` to understand definitions, cross-references, and the overall purpose of the document. This is your knowledge base.
# 2.  **Focus on Your Assigned Task:** Your primary goal is to analyze ONLY the text corresponding to the `parent_sections_to_analyze`. Do not analyze or return children for any other part of the document.
# 3.  **Return a Map:** Your output must be a map where the keys are the `parent_node_id`s you were asked to process, and the values are the lists of children you found for each.

# **OUTPUT FORMAT:**
# You MUST output a single, valid JSON object that conforms to the `LLMLevelResponse` schema.
# The `child_map` keys MUST be the `parent_node_id`s from the input.

# **FULL DOCUMENT JSON (for context):**
# ```json
# {full_document_json}
# ```

# parent layer nodes:
# ```json
# {parent_sections_json}
# ```
# """


from joblib import Memory
memory = Memory(location = '.joblib')

# @memory.cache(ignore = ['model_names'])
@joblib_memory_cached(memory, ignore = ['model_names'])
def level_node_llm_parsing(
    nodes_at_level: List[dict],  # type: ignore
    source_map: Dict,
    full_document_json_str: str, # <-- NEW ARGUMENT
    model_names: List[str]
) -> LLMLevelResponse:
    """
    Processes an entire level of parent nodes in a single, batched, context-aware LLM call.
    """
    if not nodes_at_level:
        return LLMLevelResponse.model_validate({"children":[]})
    nodes_at_level: list[SemanticNode] = [SemanticNode.model_validate (i) for i in nodes_at_level]
    # 1. Prepare the list of tasks for the LLM
    parent_sections_for_prompt = []
    for node in nodes_at_level:
        parent_sections_for_prompt.append({
            "parent_node_id": node.node_id,
            "parent_title": node.title,
            "text_to_analyze": reconstruct_text_from_pointers(node.total_content_pointers, source_map)
        })
    parent_node_id_set = set(str(i.node_id) for i in nodes_at_level)

    # 2. Construct the full, context-aware prompt
    # final_prompt = PROMPT_BATCH_SUBDIVIDER.format(
    #     full_document_json=full_document_json_str,
    #     parent_sections_json=json.dumps([
    #         {k: str(v) if isinstance(v, UUID) else v for k, v in p.items()} 
    #         for p in parent_sections_for_prompt
    #     ], indent=2)
    # )
    final_prompt = PROMPT_BATCH_SUBDIVIDER.substitute(
        full_document_json=full_document_json_str,
        parent_sections_json=json.dumps([
            {k: str(v) if isinstance(v, UUID) else v for k, v in p.items()} 
            for p in parent_sections_for_prompt
        ], indent=2)
        )
    # 3. Use your robust LangChain invoker
    from langchain_core.messages import HumanMessage,SystemMessage,BaseMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    messages: list[BaseMessage] = [HumanMessage(final_prompt)]
    i_model = 0
    while True:
        model_name = model_names[i_model]
        try:    
            print(f"\n--- Calling LLM ({model_name}) for {len(nodes_at_level)} nodes at this level ---")
            llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.1)
            
            # Use with_structured_output with our new batch response model
            response: dict = llm.with_structured_output(LLMLevelResponse, include_raw=True).invoke(messages) # type: ignore
            
            if response.get('parsing_error'):
                raise response['parsing_error']
            assert all(i.parent_node_id in parent_node_id_set for i in response['parsed'].children), "llm generated non existed parent id"
            return response['parsed'].model_dump()

        except Exception as e:
            print(f"⚠️ Model {model_name} failed: {e}")
            i_model += 1
            err_msg = str(e)
            messages.append(SystemMessage((("error: " + err_msg[:10000] + '...' + err_msg[-2000:]) if len(err_msg)>=12000 else err_msg)))
            if i_model >= len(model_names):
                raise Exception(f"All models ({model_names}) failed for this batch.") from e
@memory.cache
def get_node(pid, child_def):
    # child_def: Union[LLMChildNodeResponse, LLMChildNodeResponseBE].model_dump()
    child_def_obj: LLMChildNodeResponseBE = LLMChildNodeResponseBE.model_validate(child_def)
    absolute_pointers = child_def_obj.pointers
    child_node = SemanticNode(
        node_id=uuid1(), # App-generated UUID
        parent_id=pid,
        title=child_def_obj.title,
        node_type=child_def_obj.node_type,
        total_content_pointers=absolute_pointers,
        child_nodes = []
        # value_pointers=None # You would add logic to handle this
    )
    return child_node.model_dump()            
@memory.cache
def get_root_node(title, source_map):
    # root_node = SemanticNode(
    #     title=title,
    #     node_type="DOCUMENT_ROOT",
    #     total_content_pointers=[TextPointer(source_cluster_id=cid, start_char=0, end_char=-1) for cid in sorted(source_map.keys())]
    #     )
    root_node = SemanticNode(
        node_id=uuid1(),
        title=title,
        node_type="DOCUMENT_ROOT",
        total_content_pointers=[
            HydratedTextPointer(
                source_cluster_id=cid, 
                start_char=0, 
                end_char=-1, 
                verbatim_text=source_map[cid]['text'],
                validation_method = None
            ) for cid in sorted(source_map.keys())
        ],
        child_nodes = []
    )
    return root_node.model_dump()

def _schema_guard(parent, child) -> bool:
    # Disallow KEY_VALUE_PAIR directly under KEY_VALUE_PAIR
    if parent.node_type == "KEY_VALUE_PAIR" and child.node_type == "KEY_VALUE_PAIR":
        return False
    return True

def _normalize_child_type(parent, child):
    if parent.node_type == "KEY_VALUE_PAIR" and child.node_type == "KEY_VALUE_PAIR":
        child.node_type = "TEXT_FLOW"  # coerce value to text fragment
    return child

def build_document_tree(
                llm_input_dict: Dict,
                source_map: Dict,
                max_depth: int = 10,
                allow_review = True
                ) -> SemanticNode:
    """Builds the hierarchy using an efficient, batched, layer-wise (BFS) approach.
    Initial breakdown -> check pointers/ spans validated
    -> CUD pass round get updated nodes
    next level
    output structure
        layers of nodes from coarse to fine grained
    """
    root_node : SemanticNode= SemanticNode.model_validate(get_root_node(title=llm_input_dict['document_filename'], source_map=source_map))
    # SemanticNode(
    #     title=llm_input_dict['document_filename'],
    #     node_type="DOCUMENT_ROOT",
    #     total_content_pointers=[TextPointer(source_cluster_id=cid, start_char=0, end_char=-1) for cid in sorted(source_map.keys())]
    #     )
    nodes_for_next_level: list[SemanticNode] = [root_node]
    current_depth = 0
    fixed_children: list[LLMChildNodeResponseBE]
    model_names = ["gemini-3-flash-preview", "gemini-2.5-flash", "gemini-2.5-pro", 
                   "gemini-2.5-flash-lite", ] # Example models
    full_document_json_str = json.dumps(llm_input_dict)
    while nodes_for_next_level and current_depth < max_depth:
        
        print(f"\nProcessing Level {current_depth} with {len(nodes_for_next_level)} nodes...")
        
        nodes_at_this_level = nodes_for_next_level
        current_level_node_context_reset_token = current_level_nodes.set(nodes_at_this_level)
        nodes_for_next_level = []
        node_this_level_lookup_by_id = {str(node.node_id): node for node in nodes_at_this_level}
        # This is the single, batched call for the entire level
        llm_response_json = level_node_llm_parsing(
            [i.model_dump() for i in nodes_at_this_level], 
            source_map, 
            full_document_json_str,  # <-- PASSING GLOBAL CONTEXT
            model_names
        )
        @joblib_memory_cached(memory)
        def get_level_response(llm_response_json) -> Dict[str, Any]:
            response_cacheable = LLMLevelResponseBE.model_validate(llm_response_json).model_dump() # only dumped version cacheable by joblib
            return response_cacheable
        response_cacheable = get_level_response(llm_response_json)
        level_response : LLMLevelResponseBE= LLMLevelResponseBE.model_validate(response_cacheable)
        # [{i.title + "|" + i.node_type: [j.verbatim_text for j in i.pointers]} for i in level_response.children]
        # correct excerpts
        corrected_children, unfixed_children = correct_level_children_with_iterative_pipeline(
            level_response_json=level_response.model_dump(),
            source_map=source_map,
            full_document_json=llm_input_dict,   # same dict you pass to the LLM
            # model_names=["gpt-4.1", "gpt-4o-mini"]  # or keep your Gemini list; it’s pluggable
        )

        corrected_children : list[LLMChildNodeResponseBE]
        if unfixed_children:
            raise NotImplementedError("Not implemented for the case unfixed_children")
        fixed_children = corrected_children
        
        # fixed_children -> for next level iteration use as root
        fe_children, layer_parent_types, layer_parent_sigs = prepare_frontend_children(nodes_at_this_level, level_response, fixed_children) # for next level of LLM
        
        if allow_review:
            fixed_children, _reasoning_history= iterative_review_loop(fe_children, layer_parent_types, layer_parent_sigs, source_map,
                                                    model_names, full_document_json_str, current_depth, llm_input_dict, nodes_at_this_level)
        for ch in fixed_children:
            ch: LLMChildNodeResponseBE
            child_def = ch.model_dump()
            child_def.pop("id")
            child_node = SemanticNode.model_validate(get_node(pid = ch.parent_node_id, child_def = child_def))
            parent_node = node_this_level_lookup_by_id[str(ch.parent_node_id)]
            parent_node.child_nodes.append(child_node)
            if child_node.node_type == 'KEY_VALUE_PAIR':
                pass
            else:
                nodes_for_next_level.append(child_node)
        current_level_nodes.reset(current_level_node_context_reset_token)
        current_depth += 1
        
    return root_node    
def prepare_frontend_children(nodes_at_this_level, level_response, fixed_children):
        # RUN LLM loop make sure missing content will be guarded by LLM
        corrected_level_response = LLMLevelResponse.model_validate(level_response.model_dump())
        
        # llm_response_json['children'] = [i.model_dump() for i in fixed_children]
        corrected_level_response.children = [LLMChildNodeResponse.model_validate(i.model_dump()) for i in fixed_children]
        child_map : dict[str, list[LLMChildNodeResponse]] = {}
        for child in corrected_level_response.children:
            if child_map.get(child.parent_node_id):
                child_map[child.parent_node_id].append(child)
            else:
                child_map[child.parent_node_id] = [child]
        
        # Process the results and build the next level
        layer_parent_sigs = [] # data for simple sanity, non exhausitive non perfect check for duplication
        layer_parent_types = [] # data for simple sanity, non exhausitive non perfect check for duplication
        child_definitions: list[LLMChildNodeResponse]  = []
        fe_children: List[LLMChildNodeResponse] = []
        for parent_node in nodes_at_this_level:
            child_definitions = child_map.get(str(parent_node.node_id), [])

            # Convert backend -> frontend models for CUD
            fe_children.extend([
                LLMChildNodeResponse.model_validate(c.model_dump())
                for c in child_definitions
            ])
            # Build layer signatures. If you truly want the whole layer, pass nodes_at_this_level.
            # If you only want the current parent, pass [parent_node].
            layer_parent_sigs.extend(build_parent_signatures([parent_node]))
            layer_parent_types.extend([t for (t, _, _) in layer_parent_sigs])

        # Guards BEFORE CUD
        # [(not reject_self_recursion_multi(layer_parent_types, ch), not_self_like_multi(layer_parent_sigs, ch)) for ch in fe_children]
        fe_children = [
            ch for ch in fe_children
            if not reject_self_recursion_multi(layer_parent_types, ch)
            and not_self_like_multi(layer_parent_sigs, ch)
        ]
        fe_children = dedupe_children_level(fe_children)
        return fe_children, layer_parent_types, layer_parent_sigs
        
def iterative_review_loop(fe_children: List[LLMChildNodeResponse], layer_parent_types, layer_parent_sigs, source_map,
                        model_names, 
                        full_document_json_str, 
                        current_depth, 
                        llm_input_dict, 
                        nodes_at_this_level):
    """_summary_

    Args:
        fe_children (_type_): _description_         =>   the children parsed so far
        layer_parent_types (_type_): _description_  =>   the layer text heading signature for quick dedupe detection / printing
        layer_parent_sigs (_type_): _description_   =>   the layer text heading signature for quick dedupe detection / printing
        source_map (_type_): _description_     => p1c1 p2c2 etc usually refer to the bounding box id
        model_names (_type_): _description_    => what model to LLM
        full_document_json_str (_type_): _description_  => the doc_or_docgroup ocr data
        current_depth (_type_): _description_   => how many layers of parsing (tree levels) so far
        llm_input_dict (_type_): _description_  => LLM model specific parameters

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """
    # --- NEW: CUD iterative refinement loop ---
    # iteratively edit to cover most content if missing or remove duplicate/ conflicting ideas, 
    # after the loop, at the end, just like the initial run, have to check the verbatim/excepts really exists 
    # and correct check the except really exists again

    edited = False
    proposals = True
    retries = 0
    max_retry = 3
    CUD_reasoning_history = []
    while proposals and retries < max_retry:
        # (1) per-level dedupe
        fe_children = [
            ch for ch in fe_children
            if not reject_self_recursion_multi(layer_parent_types, ch)
            and not_self_like_multi(layer_parent_sigs, ch)
        ]
        fe_children = dedupe_children_level(fe_children)

        # (2) request one CUD round
        # token = current_level_nodes.set(nodes_at_this_level)

        
        proposals_response = CUD_proposal(
            # parent_id=str(parent_node.node_id),
            children=fe_children,
            source_map=source_map,
            model_names=model_names,
            reasoning_history = CUD_reasoning_history,
            full_document_json_str = full_document_json_str if current_depth > 0 else "[Now at root level, root node content is full doc, omitted to prevent duplication]",
            last_layer = [{"node_id": str(node.node_id), 
                           "content": reconstruct_text_from_pointers(node.total_content_pointers, source_map)} 
                          for node in nodes_at_this_level],
            attempt = retries
        )
        CUD_reasoning_history.append(proposals_response.reasoning)
        # current_level_nodes.reset(token)
        if not proposals_response.proposals: # 
            break 
        edited = True
        # (3) apply proposals (ADD/DELETE/EDIT → pointer revalidation)
        fe_children = apply_proposal(
            proposals=proposals_response.proposals,
            children=fe_children,
            source_map=source_map,
        )
        fe_children_tft = [{i.title + "|" + i.node_type: [j.verbatim_text for j in i.pointers]} for i in fe_children]
        fe_children = [
            ch for ch in fe_children
            if not reject_self_recursion_multi(layer_parent_types, ch)
            and not_self_like_multi(layer_parent_sigs, ch)
        ]
        fe_children = dedupe_children_level(fe_children)
        retries += 1
    # --- END CUD loop ---   then post CUD validate below
    if not fe_children or (not edited): 
        return  [LLMChildNodeResponseBE.model_validate(i) for i in fe_children] , CUD_reasoning_history# no children even after re-check in iterative pipeline, time to early stop
    # Create SemanticNode children for this parent
    else:
        be_children = []
        for ch in fe_children:
            temp = ch.model_dump()
            temp['node_id'] = get_node(pid = ch.parent_node_id, child_def = ch.model_dump())['node_id']
            be_children.append(LLMChildNodeResponseBE.model_validate(temp))
        corrected_level_response2 : LLMLevelResponseBE= LLMLevelResponseBE.model_validate({'children': be_children})
        # corrected_level_response2 : LLMLevelResponseBE= LLMLevelResponseBE.model_validate(get_level_response(llm_response_json))

        # corrected_level_response2 : LLMLevelResponseBE= LLMLevelResponseBE.model_validate({'children': fe_children})

        # correct excerpts again, this focus on edited indeed
        corrected_children, unfixed_children, *_ = correct_level_children_with_iterative_pipeline(
            level_response_json=corrected_level_response2.model_dump(),
            source_map=source_map,
            full_document_json=llm_input_dict,   # same dict you pass to the LLM
            # model_names=["gpt-4.1", "gpt-4o-mini"]  # or keep your Gemini list; it’s pluggable
        )

        corrected_children : list[LLMChildNodeResponseBE]
        if unfixed_children:
            raise NotImplementedError("Not implemented for the case unfixed_children")
        # ids = [str(c.id) for c in corrected_children]
        # corrected_children_map: dict[str, LLMChildNodeResponseBE] = {str(i.id) : i for i in corrected_children}
        fixed_children: list[LLMChildNodeResponseBE] = corrected_children #[corrected_children_map[str(i)] for i in ids]
    return fixed_children, CUD_reasoning_history
        
from typing import Dict, List, Optional, Tuple, Callable, Iterable
import re, json
from uuid import UUID

from pydantic import BaseModel, ValidationError


# ============================================================================
# Utilities — deterministic, no‑LLM fixes first
# ============================================================================

def _safe_slice(text: str, start: int, end_inclusive: int) -> str:
    end_excl = len(text) if end_inclusive == -1 else end_inclusive + 1
    if start < 0:
        start = 0
    if end_excl < 0:
        end_excl = 0
    return text[start:end_excl]


def _all_exact_occurrences(haystack: str, needle: str) -> List[Tuple[int, int]]:
    """Return all (start, end_inclusive) exact matches for `needle` in `haystack`.
    Uses Python's find() loop for speed and determinism. Empty needle => none.
    """
    if not needle:
        return []
    out: List[Tuple[int, int]] = []
    i = 0
    L = len(needle)
    while True:
        i = haystack.find(needle, i)
        if i == -1:
            break
        out.append((i, i + L - 1))
        i += max(1, L)
    return out


def _best_occurrence_by_proximity(
    occurrences: List[Tuple[int, int]], proposed_start: int
) -> Optional[Tuple[int, int]]:
    if not occurrences:
        return None
    return min(occurrences, key=lambda ab: abs(ab[0] - (proposed_start or 0)))


def _whitespace_collapse(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def _soft_exact_positions(
    source_text: str,
    verbatim: str,
    *,
    fuzzy_threshold: Optional[float] = None,   # 0..100 if rapidfuzz; 0..1 with difflib (we normalize to 0..100)
    fuzzy_len_stretch: float = 0.25,          # allow window length to vary ±25%
    fuzzy_stride_frac: float = 0.10,          # stride as a fraction of |verbatim|
) -> List[Tuple[int, int]]:
    """Exact match with minimal sanitation; optional fuzzy fallback.
    Returns list of candidate (start, end_incl) positions.
    """
    # 1) raw exact
    occ = _all_exact_occurrences(source_text, verbatim)
    if occ:
        return occ

    # 2) whitespace-collapsed exact (build collapsed and span map)
    v2 = _whitespace_collapse(verbatim)
    if not v2:
        return []
    spans: List[Tuple[int, int]] = []  # (orig_start, orig_end_incl) per collapsed char
    collapsed_chars = []
    i = 0
    N = len(source_text)
    while i < N:
        if source_text[i].isspace():
            j = i
            while j < N and source_text[j].isspace():
                j += 1
            if collapsed_chars and collapsed_chars[-1] == " ":
                # already collapsed previous WS run; just advance
                pass
            else:
                collapsed_chars.append(" ")
                spans.append((i, j - 1))
            i = j
        else:
            collapsed_chars.append(source_text[i])
            spans.append((i, i))
            i += 1

    collapsed_text = "".join(collapsed_chars)
    occ2 = _all_exact_occurrences(collapsed_text, v2)
    if occ2:
        mapped: List[Tuple[int, int]] = []
        for s_idx, e_idx in occ2:
            mapped.append((spans[s_idx][0], spans[e_idx][1]))
        return mapped

    # 3) Optional fuzzy fallback on collapsed_text
    if fuzzy_threshold is None:
        return []

    # --- helper to map collapsed [s,e] -> original inclusive span
    def _map_back(s_idx: int, e_idx: int) -> Tuple[int, int]:
        return spans[s_idx][0], spans[e_idx][1]

    Lq = len(v2)
    if Lq == 0 or len(collapsed_text) == 0:
        return []

    # Normalize threshold to 0..100
    thr = float(fuzzy_threshold)
    if thr <= 1.0:  # user may pass 0..1 when thinking "difflib ratio"
        thr *= 100.0

    best_score = -1.0
    best_span = None  # (s_idx, e_idx)

    # --- RapidFuzz path (fast): use LCS ratio as a cheap, positionable proxy ---
    # if _HAVE_RAPIDFUZZ:
        # We search windows near the query length (±stretch).
    Lmin = max(1, int(math.floor(Lq * (1.0 - fuzzy_len_stretch))))
    Lmax = max(Lmin, int(math.ceil(Lq * (1.0 + fuzzy_len_stretch))))
    stride = max(1, int(max(1, round(Lq * fuzzy_stride_frac))))

    for wlen in (Lmin, Lq, Lmax):
        if wlen > len(collapsed_text):
            continue
        for s in range(0, len(collapsed_text) - wlen + 1, stride):
            e = s + wlen  # exclusive
            # LCSseq.ratio returns 0..100
            score = LCSseq.normalized_similarity(v2, collapsed_text[s:e]) * 100.0
            if score > best_score:
                best_score = score
                best_span = (s, e - 1)

    if best_span and best_score >= thr:
        orig_s, orig_e = _map_back(*best_span)
        return [(orig_s, orig_e)]
    # --- True RapidFuzz path (fast): use LCS ratio as a cheap, positionable proxy ---
    import difflib
    def locate_span(query: str, text: str):
        sm = difflib.SequenceMatcher(
            None,
            text.lower(),
            query.lower()
        )
        match = max(sm.get_matching_blocks(), key=lambda m: m.size)
        if match.size == 0:
            return None
        return match.a, match.a + match.size
    best_span = locate_span(verbatim, collapsed_text)
    from rapidfuzz import fuzz
    if best_span:
        best_score = fuzz.ratio(verbatim, collapsed_text[best_span[0]: best_span[1]])
    if best_span and best_score >= thr:
        orig_s, orig_e = _map_back(*best_span)
        return [(orig_s, orig_e)]
    best_span = locate_span(verbatim, source_text)
    from rapidfuzz import fuzz
    if best_span:
        best_score = fuzz.ratio(verbatim, collapsed_text[best_span[0]: best_span[1]])
    if best_span and best_score >= thr:
        return [best_span]

    return []


# ============================================================================
# Pointer correction — deterministic tier
# ============================================================================

def correct_and_validate_pointer(
    proposed_pointer: HydratedTextPointer,
    source_map: Dict,
) -> Optional[HydratedTextPointer]:
    """Deterministic multi‑step correction. Returns fixed pointer or None.

    Steps:
      TIER 1: Trust‑but‑verify using proposed indices.
      TIER 2: Exact search for verbatim (raw, then whitespace‑collapsed).
               If multiple matches, pick the one closest to proposed start.
    """
    ids = list(source_map.keys())
    ids_same_page, id_dif_page = partition(ids, predicate = lambda x: x.split("_")[0] == proposed_pointer.source_cluster_id.split('_')[0])
    _, ids_same_page_dif_cluster =  partition(ids_same_page, predicate = lambda x: x == proposed_pointer.source_cluster_id)
    # Try some heuristic possible hallucinated cluster ids
    for _i_source_cluster, source_cluster in enumerate([source_map.get(proposed_pointer.source_cluster_id)] + \
            [source_map.get(i) for i in ids_same_page_dif_cluster + id_dif_page]):
        if not source_cluster:
            # print(
            #     f"⚠️ REJECTED: Pointer references non‑existent cluster '{proposed_pointer.source_cluster_id}'."
            # )
            # return None
            continue

        source_text: str = source_cluster["text"]

        # --- TIER 1: verify current indices
        try:
            actual = _safe_slice(
                source_text, proposed_pointer.start_char, proposed_pointer.end_char
            )
            if normalize_text(actual) == normalize_text(proposed_pointer.verbatim_text):
                return proposed_pointer
        except Exception:
            pass
        # try ast
        import ast
        candidates  = []
        try:
            quotes = ["'''", "'", '"""', '"']
            for quote in quotes:
                try:
                    verbatim =ast.literal_eval(quote + proposed_pointer.verbatim_text + quote)
                except:
                    continue
                if verbatim in source_cluster['text']:
                    candidates = _soft_exact_positions(source_text, verbatim)
                    pass
        except:
            pass
        # --- TIER 2: search for verbatim (raw then WS‑collapsed)
        if not candidates:
            verbatim = proposed_pointer.verbatim_text
            # case len(verbatim):
            vlen = len(verbatim)
            match vlen:
                case _ if 0 <= vlen <= 10:
                    fuzzy_threshold = None
                case _ if 11 <= vlen <= 20:
                    fuzzy_threshold = 0.99
                case _ if 21 <= vlen <= 50:
                    fuzzy_threshold = 0.95
                case _ if 51 <= vlen <= 100:
                    fuzzy_threshold = 0.90
                case _:
                    fuzzy_threshold = 0.85
            candidates = _soft_exact_positions(source_text, verbatim, fuzzy_threshold = fuzzy_threshold)
            
        if candidates:
            best = _best_occurrence_by_proximity(candidates, proposed_pointer.start_char)
            if best:
                s, e = best
                return HydratedTextPointer(
                    source_cluster_id= source_cluster['id'], #proposed_pointer.source_cluster_id,
                    start_char=s,
                    end_char=e,
                    verbatim_text=_safe_slice(source_text, s, e),
                )

    print(
        f"🚨 REJECTED: Unrecoverable pointer in cluster '{proposed_pointer.source_cluster_id}' for text '{proposed_pointer.verbatim_text[:140]}...'"
    )
    return None


# ============================================================================
# Child correction — run deterministic tier for every pointer in a child
# ============================================================================

def _correct_child_deterministic(
    child: LLMChildNodeResponseBE, source_map: Dict, with_coverage_check = True
) -> Tuple[Optional[LLMChildNodeResponseBE], List[HydratedTextPointer]]:
    """Attempt to fix all pointers deterministically. Returns (fixed_child, unresolved_pointers).
    If at least one pointer is unrecoverable deterministically, include it in unresolved list.
    If *all* pointers are fixed, returns the fully corrected child and empty unresolved list.
    """
    fixed_pointers: List[HydratedTextPointer] = []
    unresolved: List[HydratedTextPointer] = []

    for p in child.pointers:
        ok = correct_and_validate_pointer(p, source_map)
        if ok is None:
            unresolved.append(p)
        else:
            fixed_pointers.append(ok)

    if unresolved:
        return None, unresolved
    
    # all good
    return (
        LLMChildNodeResponseBE(
            id = child.id,
            parent_node_id=child.parent_node_id,
            node_type=child.node_type,
            title=child.title,
            pointers=fixed_pointers,
            # value_pointers=None,  # adjust if you also want to fix value_pointers
        ),
        [],
    )

from collections import namedtuple
from typing import NamedTuple

class ChildrenCorrectionResult(NamedTuple):
    fixed_children: list[LLMChildNodeResponseBE]
    pending_fix_children: list[LLMChildNodeResponseBE]

# ============================================================================
# LLM batch correction wiring (pluggable)
# ============================================================================
from typing import Protocol, Type, Any, List, TypeVar

T = TypeVar("T")

class StructuredLLMCaller(Protocol):
    def __call__(
        self, 
        prompt: str, 
        model_names: List[str], 
        schema: Type[T] | dict[str, Any], 
        model_json_schema: dict, 
        i_attempt: int
    ) -> T:
        ...

@memory.cache(ignore = ['model_names', 'schema'])
def _default_call_llm_structured(
    prompt: str, model_names: List[str], schema: type[T] | dict[str, Any], model_json_schema : dict, i_attempt: int
) -> T:
    """Default implementation using LangChain Google Generative AI stack.
    Swap this out if you prefer OpenAI or another provider.
    """
    from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    messages: List[BaseMessage] = [HumanMessage(prompt)]
    last_err = None
    max_retry_per_model = 2
    for name in model_names:
        for i in range(max_retry_per_model):
            try:
                llm = ChatGoogleGenerativeAI(model=name, temperature=0.1)
                resp: dict = llm.with_structured_output(schema, include_raw=True).invoke(messages) # type: ignore
                if resp.get("parsing_error"):
                    raise resp["parsing_error"]
                return resp["parsed"]
            except Exception as e:  # noqa: BLE001
                last_err = e
                messages.append(SystemMessage(f"previous_error: {e}"))
                continue
    raise RuntimeError(f"All models failed. Last error: {last_err}")


# ============================================================================
# Iterative level correction orchestrator
# ============================================================================
T2 = TypeVar("T2", bound=BaseModel)
from typing import Any, TypeVar, overload


@memory.cache(ignore = ['call_llm_structured', 'max_rounds', 'model_names'])
def iterative_correct_children_for_level(
    children: List[LLMChildNodeResponseBE],
    source_map: Dict,
    full_document_json: Dict,
    model_names: List[str] | None = None,
    max_rounds: int = 3,
    call_llm_structured: StructuredLLMCaller = _default_call_llm_structured,
) -> ChildrenCorrectionResult: # List[List[LLMChildNodeResponseBE]]:
    """Main entry: fix child pointers at a layer.

    Strategy:
      1) Deterministic pass: try to fix each child locally.
      2) Batch LLM pass for *only* the unresolved children.
      Repeat up to `max_rounds` until convergence or no unresolved.
    """
    model_names = model_names or [
        "gemini-3-preview"
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]

    fixed: Dict[str, LLMChildNodeResponseBE] = {}
    pending: Dict[str, LLMChildNodeResponseBE] = {f"{i.parent_node_id}|{i.title}": i for i in children}
    still_unsolved_same_cnt = 0
    pending_length_history = []
    for round_idx in range(max_rounds):
        if not pending:
            break
        pending_length_history.append({'round_idx': round_idx, "stage":'pre-deterministic', 'length_pending': len(pending), "still_unsolved_same_cnt": still_unsolved_same_cnt})
        # ----- 1) deterministic pass
        still_unresolved: Dict[str, LLMChildNodeResponseBE] = {}
        for key, child in list(pending.items()):
            ok_child, unresolved_pointers = _correct_child_deterministic(child, source_map, with_coverage_check = True)
            if ok_child is not None:
                fixed[key] = ok_child
            else:
                still_unresolved[key] = child
        if len(pending) == len(still_unresolved):
            still_unsolved_same_cnt += 1
        else:
            still_unsolved_same_cnt = 0
        pending = still_unresolved
        if not pending:
            break # short circuit if all resovled correctly
        pending_length_history.append({'round_idx': round_idx, "stage":'pre-llm-correct', 'length_pending': len(pending), "still_unsolved_same_cnt": still_unsolved_same_cnt})
        # ----- 2) LLM batch pass over *only* unresolved children
        nodes_to_correct = [
            {
                "parent_node_id": c.parent_node_id,
                "node_type": c.node_type,
                "title": c.title,
                "pointers": [p.model_dump() for p in c.pointers],
            }
            for c in pending.values()
        ]
        prompt = PROMPT_POINTER_CORRECTION.format(
            full_document_json=json.dumps(full_document_json, ensure_ascii=False),
            nodes_to_correct_json=json.dumps(nodes_to_correct, ensure_ascii=False),
        )
        try:
            schema = LLMLevelResponse.model_json_schema()
            parsed: LLMLevelResponse = call_llm_structured(
                prompt, model_names, LLMLevelResponse, schema , still_unsolved_same_cnt
            )
            parsed_be = LLMLevelResponseBE.model_validate(parsed.model_dump())
            # validate each returned child again deterministically (trust but verify)
            returned_by_key: Dict[str, LLMChildNodeResponseBE] = {}
            for ch in parsed_be.children:
                key = f"{ch.parent_node_id}|{ch.title}"
                ok_child, unresolved_pointers = _correct_child_deterministic(ch, source_map)
                if ok_child is not None and not unresolved_pointers:
                    returned_by_key[key] = ok_child
            # merge fixes
            for k, v in returned_by_key.items():
                fixed[k] = v
                pending.pop(k, None)
        except Exception as e:  # noqa: BLE001
            # LLM failed this round; keep items pending for next round or exit
            print(f"LLM correction round {round_idx+1} failed: {e}")
            # fall through; next round will retry or terminate

    # Final set = fixed + whatever remains pending (keep originals for transparency)
    
    # ChildrenCorrectionResult = namedtuple("ChildrenCorrectionResult", ["fixed_children", "pending_fix_children"])
    # 1) add fixed children first
    # 2) add unresolved originals (so caller can decide whether to drop/flag)

    out = ChildrenCorrectionResult(fixed_children = list(fixed.values()) ,
                                    pending_fix_children= list(pending.values()))
    return out
# ======== Minimal additions to support your CUD loop (matching your usage) ========
from string import Template as _CUDTemplate

# ---------- Pydantic models for CUD ----------

from typing import Optional, List, Literal, Dict, Any, Tuple, Union

# --- how to select an existing child in THIS layer ---
class CUDTarget(BaseModel):
    node_id: Optional[str] = Field(default=None, description="Optional direct child id if present in your FE objects.")
    node_type: Optional[Literal["TEXT_FLOW", "KEY_VALUE_PAIR", "TABLE"]] = None
    title: Optional[str] = None

    @model_validator(mode="after")
    def _at_least_one_selector(self):
        if not (self.node_id or (self.node_type and self.title is not None)):
            raise ValueError("CUDTarget requires either node_id OR (node_type AND title).")
        return self


# --- partial edit payload (only provided fields are changed) ---
class LLMChildNodePatch(BaseModel):
    parent_node_id: Optional[str] = None
    node_type: Optional[Literal["TEXT_FLOW", "KEY_VALUE_PAIR", "TABLE"]] = None
    title: Optional[str] = None
    pointers: Optional[List[HydratedTextPointer]] = None
    # value_pointers: Optional[List[HydratedTextPointer]] = None


# --- strictly typed proposal ---
class CUDProposal(BaseModel):
    edit_type: Literal["ADD_NODE", "DELETE_NODE", "EDIT_NODE"]
    target: Optional[CUDTarget] = Field(default=None, description="Target required for DELETE/EDIT.")
    add: Optional[LLMChildNodeResponse] = Field(default=None, description="Strict child for ADD.")
    patch: Optional[LLMChildNodePatch] = Field(default=None, description="Partial patch for EDIT.")
    reasoning: str = Field(description="Reasoning for each proposal")
    @model_validator(mode="after")
    def _check_consistency(self):
        if self.edit_type == "ADD_NODE":
            if self.add is None:
                raise ValueError("ADD_NODE requires 'add'.")
            if self.target is not None or self.patch is not None:
                raise ValueError("ADD_NODE may not include 'target' or 'patch'.")
        elif self.edit_type == "DELETE_NODE":
            if self.target is None:
                raise ValueError("DELETE_NODE requires 'target'.")
            if self.add is not None or self.patch is not None:
                raise ValueError("DELETE_NODE may not include 'add' or 'patch'.")
        elif self.edit_type == "EDIT_NODE":
            if self.target is None or self.patch is None:
                raise ValueError("EDIT_NODE requires both 'target' and 'patch'.")
            if self.add is not None:
                raise ValueError("EDIT_NODE may not include 'add'.")
        return self
class DProposal(BaseModel):
    edit_type: Literal["DELETE_NODE"]
    target: Optional[CUDTarget] = Field(default=None, description="Target required for DELETE/EDIT.")
    reasoning_delete : str = Field(..., description = "reason for delete")
    @model_validator(mode="after")
    def _check_consistency(self):
        if self.edit_type == "DELETE_NODE":
            if self.target is None:
                raise ValueError("DELETE_NODE requires 'target'.")
        else:
            raise(ValueError('unrecognized mode'))
        return self
class UProposal(BaseModel):
    edit_type: Literal["EDIT_NODE"]
    target: Optional[CUDTarget] = Field(default=None, description="Target required for DELETE/EDIT.")
    patch: Optional[LLMChildNodePatch] = Field(default=None, description="Partial patch for EDIT.")
    reasoning_update : str = Field(..., description = "reason for Update")
    @model_validator(mode="after")
    def _check_consistency(self):
        if self.edit_type == "EDIT_NODE":
            if self.target is None or self.patch is None:
                raise ValueError("EDIT_NODE requires both 'target' and 'patch'.")
        else:
            raise(ValueError('unrecognized mode'))
            
        return self    
class CProposal(BaseModel):
    edit_type: Literal["ADD_NODE"]
    add: Optional[LLMChildNodeResponse] = Field(default=None, description="Strict child for ADD.")
    reasoning_create : str = Field(..., description = "reason for Create")
    @model_validator(mode="after")
    def _check_consistency(self):
        if self.edit_type == "ADD_NODE":
            if self.add is None:
                raise ValueError("ADD_NODE requires 'add'.")
        else:
            raise(ValueError('unrecognized mode'))
        return self

class CUDResponse(BaseModel):
    reasoning:str = Field(description = 'reasoning at top level')
    proposals: List[CUDProposal] = Field(default_factory=list, description = 'a list of update proposals, empty if existing is good. ')
    
class CResponse(BaseModel):
    reasoning:str = Field(description = 'reasoning at top level')
    proposals: List[CProposal] = Field(default_factory=list)
# ---------- small helpers ----------
def _normalize_title(s: str) -> str:
    return " ".join((s or "").lower().split())

def _pkey(p: HydratedTextPointer) -> tuple[str, int, int]:
    return (p.source_cluster_id, p.start_char, p.end_char)

def dedupe_children_level(children: List[LLMChildNodeResponse]) -> List[LLMChildNodeResponse]:
    """
    Per-level structural dedupe: same node_type + normalized title + identical pointer set.
    """
    seen, out = set(), []
    for ch in children:
        key = (
            ch.node_type,
            _normalize_title(ch.title),
            tuple(sorted(_pkey(p) for p in ch.pointers)),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(ch)
    return out
# ----- LAYER-AWARE GUARDS -----
def build_parent_signatures(parents: List[SemanticNode]) -> List[tuple[str, str, set]]:
    """
    For a layer of parents, return [(type, norm_title, span_set)] for each.
    span_set = set of (cluster, start, end).
    """
    sigs: List[tuple[str,str,set]] = []
    for p in parents:
        spans = {_pkey(ptr) for ptr in p.total_content_pointers}
        sigs.append((p.node_type, _normalize_title(p.title), spans))
    return sigs

def reject_self_recursion_multi(parent_types: List[str], child: LLMChildNodeResponse) -> bool:
    """
    Forbid KEY_VALUE_PAIR directly under any KEY_VALUE_PAIR parent in the layer.
    Extend with other schema rules if needed.
    """
    return child.node_type == "KEY_VALUE_PAIR" and ("KEY_VALUE_PAIR" in parent_types)

def not_self_like_multi(parent_sigs: List[tuple[str,str,set]], child: LLMChildNodeResponse) -> bool:
    """
    Reject if child is a mirror of ANY parent in the layer:
      same type AND same normalized title AND identical span set.
    """
    child_title = _normalize_title(child.title)
    child_spans = {_pkey(p) for p in child.pointers}
    for p_type, p_title, p_spans in parent_sigs:
        if (child.node_type == p_type) and (child_title == p_title) and (child_spans == p_spans):
            return False
    return True

def _validate_child_pointers(child: LLMChildNodeResponse, source_map: Dict) -> Optional[LLMChildNodeResponse]:
    """
    Reuse your pointer correction: return a fixed child or None if any pointer can't be validated.
    Called after ADD/EDIT proposals to ensure trust-but-verify.
    """
    fixed_ptrs: List[HydratedTextPointer] = []
    for p in child.pointers:
        ok = correct_and_validate_pointer(p, source_map)
        if not ok:
            return None
        fixed_ptrs.append(ok)

    # fixed_values: Optional[List[HydratedTextPointer]] = None
    # if getattr(child, "value_pointers", None):
    #     fixed_values = []
    #     for p in child.value_pointers:
    #         ok = correct_and_validate_pointer(p, source_map)
    #         if not ok:
    #             return None
    #         fixed_values.append(ok)

    return child.model_copy(update={"pointers": fixed_ptrs, 
                                    # "value_pointers": fixed_values
                                    })

# ---------- CUD prompting ----------

"""

- When breaking down, such as breaking down [SCHEDULE 1] into its children, do not just breakdown into [SUB-SCHEDULE], but break into [SCHEDULE1]-[SUB-SCHEDULE],
  Unless the parent is already containing the overall heading of SCHEDULE 1. in this way the heading/ title for look up will not miss out useful search information.
  example:   
    `{Item XX section yy section zz}` if it is very light weight item, you may keep it as it is, but if yy and zz are long, you need to break it into
    `item xx - section yy` and `item xx - section zz`. keep the bracking heading `item xx` when breaking down. It extend to not only item, can be schedule-subschedule, section-paragraph
    or even other structure substructure.
    if the previous already has a parent node just Item xx (not item xx, item aa together), in this case, you can omit the bracketing header and just put section yy and section zz in children.


; 
  or break down emulated by 1. edit the too-broad node to include part of broken down data and 2. generate multiple nodes to contain the remaining break down data.  
"""

_CUD_PROMPT = _CUDTemplate(r"""
ROLE: You are revising ONE layer (the immediate children of a single parent) during an iterative editing process.
Some sections might be broken down in other iterations. Do not attempt to add them back. Only consider current given parent nodes should be broken down or not.
The child is to destructure any parent node that is too broad in meaning. If a parent is narrow enough, just do not output children of the parent.
Allowed edits: ADD_NODE, DELETE_NODE, EDIT_NODE.

Rules:
- Edit ONLY the current layer (no ancestors/descendants).
- Pointers must match the source text exactly; the engine will validate.
- DELETE only strict duplicates WITHIN this layer or it is duplicating the combined effect of other existing nodes; prefer EDIT if fixable.
- Partial duplication within a new node can be corrected by editing to that the edit no longer contain the duplicated information.
- Do not replace a existing break down just because of a slightly higher or lower granularity, even though coarser breakdown is preferred. The reasson is that we are iteratively editing in layerwise manner. 
-- Coarser/less granular parsing in each layer with more iterations will preserve better structure than single flattened parsing.
-- For example, if a node contains section-A and section B, each contain sub-section, being coarser is correct and preferred. Because next iteration will further breakdown from section level to subsection level.
- If a node in current layer is still too broad, containing multiple children ideas, keep it unchanged.
- All current layer nodes at any time must be the result of breaking down `Previous layer`. DO NOT ATTEMPT to break down current layer.
- Since you are destructuring. Do NOT combine short verbatims to long even if they form a single sentence. But group them in a list instead. i.e. large groups -> small groups and no recombine.
- Make sure the children together convey the parent meaning
- If a single node has incorrect data/error, edit it with correction.
- When breaking down, all children combined together must preserve their parent's meaning.
- Each parent must have more than 1 child or no child. Explanation: Having 1 child mean that child has same meaning and granularity as parent and it is not break down.
- If a parent has no children, it just mean it is granular enough. It is still preserved, not discarded.

- Idealy, you need to avoid the following 2 situations:
    1. breakdown too deep, or 
    2. no breakdown (only duplicate previous layer)
- If the current layer is breaking down the last layer in a sweetspot manner, you can choose to return no change proposals to indicate satisfactory evaluation.
- If a parent need not broken down, do not keep them in output in next layer. Leave it as a leaf in the data structure.
Return JSON that conforms to this schema:
{
  "proposals": [
    {"edit_type": "ADD_NODE", "add": <LLMChildNodeResponse JSON>},
    {"edit_type": "DELETE_NODE", "target": {"node_id": "..."} OR {"node_type": "...", "title": "..."}},
    {"edit_type": "EDIT_NODE", "target": {...}, "patch": {"title": "...", "pointers": [...]} }
  ]
}
Full document context:
$full_document_json_str

Previous layer for parent_node_id refernece:
$ancestors

CURRENT LAYER SCHEMA (Not the update/ changes proposal schema, remember):
$current_layer_schema

CURRENT editing/drafting LAYER (children JSON):
$current_layer_json
""")

# use when layer first round, less error
_C_PROMPT = _CUDTemplate(r"""
ROLE: You are revising ONE layer (the immediate children of a single parent).
Allowed edits: ADD_NODE

Rules:
- Pointers must match the source text exactly; the engine will validate.
- If previous layer is already the most granular, do not force yourself into repeating the same granular result. Just give no edit in this case, leave the current layer empty.
- Do not replace a existing break down just because of a slightly higher or lower granularity, even though coarser breakdown is preferred. The reasson is that we are iteratively editing in layerwise manner. 
-- Coarser/less granular parsing in each layer with more iterations will preserve better structure than single flattened parsing.
-- For example, if a node contains section-A and section B, each contain sub-section, being coarser is correct and preferred. Because next iteration will further breakdown from section level to subsection level.
- When breaking down, all children combined together must preserve their parent's meaning.
- Each parent must have more than 1 child or no child. Explanation: Having 1 child mean that child has same meaning and granularity as parent and it is not break down.
- If a parent has no children, it just mean it is granular enough. It is still preserved, not discarded.

Return JSON that conforms to this schema:
{
  "proposals": [
    {"edit_type": "ADD_NODE", "add": <LLMChildNodeResponse JSON>, "reasoning_create": <CREATION REASONING>},
  ]
}

Previous iteration reasoning history:
$reasoning_history

Full document context:
$full_document_json_str

Ancestors for refernece:
$ancestors

CURRENT LAYER (children JSON):
$current_layer_json
""")
# _CUD_PROMPT = _CUDTemplate(r"""
# ROLE: You are revising ONE layer (the immediate children of a single parent).
# Allowed edits: ADD_NODE, DELETE_NODE, EDIT_NODE.

# Rules:
# - Edit ONLY the current layer (no ancestors/descendants).
# - Pointers must match the source text exactly; the engine will validate.
# - DELETE only strict duplicates WITHIN this layer; prefer EDIT if fixable.

# Return JSON that conforms to this schema:
# {
#   "proposals": [
#     {"edit_type": "ADD_NODE", "child": <LLMChildNodeResponse JSON>},
#     {"edit_type": "DELETE_NODE", "match": {"node_type": "...", "title": "..."}},
#     {"edit_type": "EDIT_NODE", "match": {"node_type": "...", "title": "..."}, "child": <LLMChildNodeResponse JSON>}
#   ]
# }

# Full document context:
# $full_document_json_str

# Ancestors for refernece:
# $ancestors

# CURRENT LAYER (children JSON):
# $current_layer_json
# """)

def _serialize_children_for_prompt(children: List[LLMChildNodeResponse]) -> str:
    slim = []
    for c in children:
        slim.append({
            "node_type": c.node_type,
            "title": c.title,
            "pointers": [{
                "source_cluster_id": p.source_cluster_id,
                "start_char": p.start_char,
                "end_char": p.end_char,
                "verbatim_text": (p.verbatim_text[:120] + ("…" if len(p.verbatim_text) > 120 else "")),
            } for p in c.pointers[:2]],
        })
    return json.dumps(slim, ensure_ascii=False, indent=2)

# ---------- CUD_proposal + apply_proposal ----------
@memory.cache(ignore = ['model_names', 'source_map'])
def CUD_proposal(
    # parent_id: str,
    children: List[LLMChildNodeResponse],
    source_map: Dict,
    model_names: List[str],
    full_document_json_str: str,
    last_layer: List,
    attempt : int,
    reasoning_history: list[str]
) -> CUDResponse|CResponse:
    """
    Ask LLM for CUD proposals (single round) for the CURRENT LAYER.
    Returns a list of CUDProposal, or [] if none.
    """
    # build small prompt from current layer only
    if children:
        prompt = _CUD_PROMPT.substitute(current_layer_json=_serialize_children_for_prompt(children),
                                        full_document_json_str = full_document_json_str, 
                                        current_layer_schema = LLMChildNodeResponse.model_json_schema(),
                                        ancestors = last_layer, reasoning_history = str(reasoning_history))
        ResponseModel = CUDResponse
    else:
        prompt = _C_PROMPT.substitute(current_layer_json=_serialize_children_for_prompt(children),
                                        full_document_json_str = full_document_json_str,
                                        ancestors = last_layer, reasoning_history = str(reasoning_history))
        ResponseModel = CResponse
    # Prefer your structured invoker if available

    try:
        resp = _default_call_llm_structured(
            prompt=prompt,
            model_names=model_names,
            schema=ResponseModel,
            model_json_schema=ResponseModel.model_json_schema(),
            i_attempt=attempt,
        )
        return resp
    except Exception as e:
        print("error " + str(e))
        # logger.error(e)
        # fail-safe: no proposals means exit loop on your side
        raise e
        return []
from typing import Sequence
def apply_proposal(
    proposals: Sequence[CUDProposal | CProposal],
    children: List[LLMChildNodeResponse],
    source_map: Dict,
) -> List[LLMChildNodeResponse]:
    """
    Apply proposals to THIS layer only.
    - ADD_NODE: add typed child (validated pointers)
    - DELETE_NODE: remove targeted node (by id or (type,title))
    - EDIT_NODE: patch targeted node (typed partial), then revalidate pointers
    Returns updated, deduped children list.
    """
    out = list(children)

    def _locate_index(tgt: CUDTarget) -> Optional[int]:
        if tgt.node_id:
            for i, c in enumerate(out):
                # Your FE objects don't carry id by default — fallback to (type,title) path.
                # If you do have ids in FE later, wire here.
                pass
        if tgt.node_type and tgt.title is not None:
            nt = tgt.node_type
            tt = _normalize_title(tgt.title)
            for i, c in enumerate(out):
                if c.node_type == nt and _normalize_title(c.title) == tt:
                    return i
        return None

    for p in proposals or []:
        if p.edit_type == "DELETE_NODE":
            idx = _locate_index(p.target) if p.target else None
            if idx is not None:
                out.pop(idx)
            continue

        if p.edit_type == "ADD_NODE":
            # Ensure parent id exists in payload
            if p.add is None:
                # skip the proposal as if it never return and get detected next round
                continue
            add = p.add.model_copy()
            if not add.parent_node_id:
                # fall back to first child’s parent if present
                add.parent_node_id = out[0].parent_node_id if out else add.parent_node_id
            fixed = _validate_child_pointers(add, source_map)
            if fixed:
                out.append(fixed)
            continue

        if p.edit_type == "EDIT_NODE":
            idx = _locate_index(p.target) if p.target else None
            if idx is None:
                continue
            base = out[idx].model_dump()
            # Merge patch fields (only provided)
            patch = p.patch
            if patch is None:
                # skip the proposal as if it never return and get detected next round
                continue
            
            if patch.parent_node_id is not None:
                base["parent_node_id"] = patch.parent_node_id
            if patch.node_type is not None:
                base["node_type"] = patch.node_type
            if patch.title is not None:
                base["title"] = patch.title
            if patch.pointers is not None:
                base["pointers"] = [hp.model_dump() for hp in patch.pointers]
            # if patch.value_pointers is not None:
            #     base["value_pointers"] = [hp.model_dump() for hp in patch.value_pointers]

            try:
                draft = LLMChildNodeResponse.model_validate(base)
            except Exception:
                continue
            fixed = _validate_child_pointers(draft, source_map)
            if fixed:
                out[idx] = fixed

    return dedupe_children_level(out)

# ============================================================================
# Convenience: correct one level from your existing `build_document_tree` loop
# ============================================================================
@memory.cache
def correct_level_children_with_iterative_pipeline(
    level_response_json: dict,
    source_map: Dict,
    full_document_json: Dict,
    model_names: List[str] | None = None,
) -> ChildrenCorrectionResult:
    """Helper to be used right after a level LLM call in your BFS.
    This layer is a cacheable layer
    Example integration:
        level_response = LLMLevelResponse.model_validate(llm_response_json)
        corrected_children = correct_level_children_with_iterative_pipeline(
            level_response.model_dump(), source_map, llm_input_dict
        )
    """
    level = LLMLevelResponseBE.model_validate(level_response_json)
    children = [LLMChildNodeResponseBE.model_validate(c.model_dump()) for c in level.children]
    to_return = iterative_correct_children_for_level(
        children=children,
        source_map=source_map,
        full_document_json=full_document_json,
        model_names=model_names,
    )
    
    return to_return

# ==============================================================================
# PHASE 4: TREE RECONSTRUCTION & VALIDATION
# ==============================================================================

def _merge_child_ranges_for_cluster(
    child_ranges: List[Tuple[int, int, str]],
    allowed_overlap: int = 0,
) -> bool:
    """
    child_ranges: list of (start, end_incl, child_id)
    returns True if OK, False if overlap violation.
    `allowed_overlap` = number of chars we allow two different children to clash on.
    """
    # sort by start
    child_ranges.sort(key=lambda x: x[0])
    prev_start, prev_end, prev_child = child_ranges[0]

    for i in range(1, len(child_ranges)):
        cur_start, cur_end, cur_child = child_ranges[i]

        if cur_start <= prev_end:  # overlap
            # compute actual overlap span
            overlap_len = min(prev_end, cur_end) - cur_start + 1
            # if from different children and overlap is too big -> fail
            if cur_child != prev_child and overlap_len > allowed_overlap:
                return False
            # merge for next step
            prev_end = max(prev_end, cur_end)
            # if same child, we just widen it
        else:
            # no overlap, advance
            prev_start, prev_end, prev_child = cur_start, cur_end, cur_child

    return True

from rapidfuzz import fuzz

from typing import Dict, List, Tuple

class CoverageResponse(BaseModel):
    per_cluster: dict[str, float]
    overall: float
    
def compute_pointer_coverage(
    root_node: SemanticNode,
    source_map: Dict,
    *,
    clamp_to_cluster: bool = True,
) ->CoverageResponse:
    """
    Compute how much of each source_cluster_id is covered by pointers in the tree,
    using range-merging (no per-character sets).

    Returns:
        {
          "per_cluster": { "p1_c0": 0.98, "p1_c1": 1.0, ... },
          "overall": 0.995
        }

    Notes:
    - clamp_to_cluster=True: if a pointer uses end_char == -1 or runs past the text,
      we clamp to the actual cluster text length.
    - If a cluster has zero length (empty OCR), it is ignored in overall calc.
    """
    # 1) collect all pointers in the tree
    all_pointers: List[HydratedTextPointer] = []

    def _walk(node: SemanticNode):
        # a node may have multiple pointers
        if node.node_type != 'DOCUMENT_ROOT':
            all_pointers.extend(node.total_content_pointers or [])
        for ch in node.child_nodes or []:
            _walk(ch)

    _walk(root_node)

    # 2) group by cluster
    cluster_ranges: Dict[str, List[Tuple[int, int]]] = {}
    for ptr in all_pointers:
        cid = ptr.source_cluster_id
        src = source_map.get(cid)
        if not src:
            continue
        text = src.get("text", "")
        max_idx = max(0, len(text) - 1)
        start = max(0, ptr.start_char)
        end = ptr.end_char
        if end == -1:
            end = max_idx
        if clamp_to_cluster:
            end = min(end, max_idx)
        if start > end:
            continue
        cluster_ranges.setdefault(cid, []).append((start, end))

    # 3) merge per cluster and compute coverage
    per_cluster_cov: Dict[str, float] = {}
    total_len = 0
    total_covered = 0

    for cid, ranges in cluster_ranges.items():
        src = source_map.get(cid)
        if not src:
            continue
        text = src.get("text", "")
        cluster_len = len(text)
        if cluster_len == 0:
            continue

        # merge
        ranges.sort(key=lambda x: x[0])
        merged: List[Tuple[int, int]] = []
        cur_s, cur_e = ranges[0]
        for s, e in ranges[1:]:
            if s <= cur_e + 1:
                # overlap or contiguous
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))

        covered_len = sum((e - s + 1) for s, e in merged)
        cov = covered_len / cluster_len
        per_cluster_cov[cid] = cov

        total_len += cluster_len
        total_covered += covered_len

    overall = (total_covered / total_len) if total_len else 1.0
    
    return CoverageResponse.model_validate({
        "per_cluster": per_cluster_cov,
        "overall": overall,
    })

def analyze_and_validate_tree(
    root_node: SemanticNode,
    source_map: Dict,
    *,
    allowed_overlap_per_cluster: int = 0,   # e.g. 2–5 chars to forgive punctuation/WS
    completeness_fuzz_threshold: int = 98,  # % similarity to still pass completeness
) -> bool:
    """Performs validation: sibling overlaps (range-based) and leaf completeness (fuzzy)."""

    def check_sibling_overlaps(node: SemanticNode) -> bool:
        if not node.child_nodes:
            return True

        # cluster_id -> list[(start, end_incl, child_label)]
        cluster_ranges: Dict[str, List[Tuple[int, int, str]]] = {}

        for idx, child in enumerate(node.child_nodes):
            child_label = f"{child.title}-{idx}"
            for pointer in child.total_content_pointers:
                cid = pointer.source_cluster_id
                start = pointer.start_char
                end = pointer.end_char
                if end == -1:
                    # expand to cluster length
                    src = source_map.get(cid)
                    if src:
                        end = len(src["text"]) - 1
                cluster_ranges.setdefault(cid, []).append((start, end, child_label))

        # now check per-cluster
        for cid, ranges in cluster_ranges.items():
            if not ranges:
                continue
            ok = _merge_child_ranges_for_cluster(
                ranges,
                allowed_overlap=allowed_overlap_per_cluster,
            )
            if not ok:
                print(f"🚨 OVERLAP ERROR in node '{node.title}' on cluster '{cid}'")
                return False

        # recurse
        return all(check_sibling_overlaps(child) for child in node.child_nodes)

    def check_leaf_completeness(root: SemanticNode, source: Dict) -> bool:
        leaf_pointers: List[HydratedTextPointer] = []

        def collect_leaves(n: SemanticNode):
            if not n.child_nodes and n.node_type != "DOCUMENT_ROOT":
                leaf_pointers.extend(n.total_content_pointers)
            else:
                for ch in n.child_nodes:
                    collect_leaves(ch)

        collect_leaves(root)

        reconstructed_text = reconstruct_text_from_pointers(leaf_pointers, source)
        source_text = "".join([cluster["text"] for cid, cluster in sorted(source.items())])

        recon_norm = normalize_text(reconstructed_text)
        source_norm = normalize_text(source_text)

        if recon_norm == source_norm:
            return True

        # fuzzy allow
        score = fuzz.partial_ratio(recon_norm, source_norm)
        if score >= completeness_fuzz_threshold:
            print(f"⚠️ COMPLETENESS WARN: fuzzy={score} >= {completeness_fuzz_threshold}, accepting.")
            return True
        else:
            print(f"⚠️ INCOMPLETENESS WARN: fuzzy={score} < {completeness_fuzz_threshold}, rejected.")

        print("🚨 COMPLETENESS ERROR: The leaf nodes do not fully cover the source document.")
        print(f"   fuzzy={score} < {completeness_fuzz_threshold}")
        return False

    print("\n--- Phase 4: Running Tree Validation ---")
    if not check_sibling_overlaps(root_node):
        return False
    if not check_leaf_completeness(root_node, source_map):
        return False

    print("✅ SUCCESS: Document tree is valid, complete, and has no overlaps.")
    return True

# ==============================================================================
# PHASE 5: UTILITIES & EXECUTION
# ==============================================================================
def normalize_text(text: str) -> str:
    """Removes all whitespace characters for a clean comparison."""
    return re.sub(r'\s+', '', text)

def reconstruct_text_from_pointers(pointers: List[HydratedTextPointer], source_map: Dict) -> str:
    """
    Reconstructs text from pointers. If `relative_to` is provided, assumes pointers
    are relative to that text content. Otherwise, assumes they are absolute.
    """
    full_text = ""
    pointers.sort(key=lambda p: (p.source_cluster_id, p.start_char))
    for pointer in pointers:
        source_cluster = source_map.get(pointer.source_cluster_id)
        if not source_cluster: continue
        source_text = source_cluster['text']
        end = len(source_text) if pointer.end_char == -1 else pointer.end_char + 1
        start = pointer.start_char
        if start < len(source_text):
            full_text += source_text[start:end]
    return full_text


def print_tree(node: SemanticNode, indent=""):
    """Visualizes the hydrated tree. No longer needs source_map."""
    reconstructed_text = "".join([p.verbatim_text for p in node.total_content_pointers])
    print(f"{indent} L- {node.title} ({node.node_type}) | Text: '{reconstructed_text[:150].strip()}...'")
    for child in node.child_nodes:
        print_tree(child, indent + "  ")
@memory.cache()
def parse_doc(raw_doc_dict):
    
    
    try:
        print("--- Phase 2: Preparing Document ---")
        llm_input_dict, source_map = prepare_document_for_llm(raw_doc_dict)

        print("\n--- Phase 3: Building Document Tree (Layer-wise) ---")
        document_tree = build_document_tree(llm_input_dict, source_map)
        cov: CoverageResponse = compute_pointer_coverage(document_tree, source_map)
        print("Overall coverage:", cov.overall)
        for cid, r in cov.per_cluster.items():
            if r < 0.99:
                print(f"⚠️ {cid} only {r:.1%} covered")
        
        print("\n--- Phase 4: Reconstructing and Validating Tree ---")

        # is_valid = analyze_and_validate_tree(
        #                 document_tree,
        #                 source_map,
        #                 allowed_overlap_per_cluster=4,    # tolerate small clashes
        #                 completeness_fuzz_threshold=97,   # tolerate tiny OCR drift
        #             )
        
        # if is_valid:
        print("\n--- Phase 5: Visualizing the Reconstructed Tree ---")
        print_tree(document_tree)
        
    except (ValidationError, json.JSONDecodeError) as e:
        print("\n--- ERROR: Failed to parse or validate LLM response. ---")
        print(f"Details: {e}")
        raise e
    except Exception as e:
        print("\n--- ERROR: A critical error occurred. ---")
        print(f"Details: {e}")
        raise e
    return document_tree, source_map


def semantic_tree_to_kge_payload(
    root: "SemanticNode",
    *,
    doc_id: str | None = None,
    insertion_method: str = "semantic_document_parser_v1",
) -> Dict[str, Any]:
    if doc_id is None:
        doc_id = str(uuid1())

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    def _pointers_to_references(ptrs: List["HydratedTextPointer"]) -> List[Dict[str, Any]]:
        return [
            {
                "doc_id": doc_id,
                "collection_page_url": f"doc://{doc_id}",
                "document_page_url": f"doc://{doc_id}#{p.source_cluster_id}",
                "insertion_method": insertion_method,
                "start_page": 1,
                "end_page": 1,
                "start_char": p.start_char,
                "end_char": p.end_char,
                "snippet": p.verbatim_text,
                "source_cluster_id": p.source_cluster_id
            }
            for p in ptrs
        ]

    def walk(node: "SemanticNode"):
        ptrs = list(node.total_content_pointers)
        pointers_payload = [p.model_dump() for p in ptrs]

        node_dict = {
            "id": str(node.node_id),
            "label": node.title,
            "type": "entity",
            "summary": "".join(p.verbatim_text for p in ptrs)[:4000],
            "metadata": {
                "semantic_node_type": node.node_type,
                "doc_id": doc_id,
                "parent_id": str(node.parent_id) if node.parent_id else None,
                "pointers": pointers_payload,
                "insertion_method": insertion_method,
            },
            
            "references": _pointers_to_references(ptrs),
        }
        nodes.append(node_dict)

        for child in node.child_nodes:
            edge_ptr_refs = _pointers_to_references(ptrs)  # parent’s pointers as provenance
            edge_dict = {
                "id": str(uuid1()),
                "label": "parent-child",
                "type": "relationship",
                "summary": f"{node.node_id} -> {child.node_id} (HAS_CHILD)",
                "relation": "HAS_CHILD",
                "source_ids": [str(node.node_id)],
                "target_ids": [str(child.node_id)],
                "source_edge_ids": [],
                "target_edge_ids": [],
                "references": edge_ptr_refs,
                "metadata": {
                    "doc_id": doc_id,
                    "insertion_method": insertion_method,
                },
            }
            edges.append(edge_dict)
            walk(child)

    walk(root)
    nodes[0].update({"properties": {"kind": "document_root"}}) # document root
    return {
        "doc_id": doc_id,
        "insertion_method": insertion_method,
        "nodes": nodes,
        "edges": edges,
    }
from collections import defaultdict

def _extract_pointers_from_references(refs: List[Dict[str, Any]]):
    # turn MCP ref → HydratedTextPointer-like
    results = []
    for r in refs or []:
        # document_page_url: "doc://{doc_id}#{source_cluster_id}"
        doc_page = r.get("document_page_url") or ""
        source_cluster_id = None
        if "#" in doc_page:
            source_cluster_id = doc_page.split("#", 1)[1]
        # fallback
        if not source_cluster_id:
            source_cluster_id = "p1_c0"
        results.append(
            HydratedTextPointer(
                source_cluster_id=source_cluster_id,
                start_char=r.get("start_char", 0),
                end_char=( -1 if r.get("end_char") == 10**9 else r.get("end_char", -1) ),
                verbatim_text=r.get("snippet", ""),
            )
        )
    return results


def kge_payload_to_semantic_tree(payload: Dict[str, Any]) -> "SemanticNode":
    nodes_data: List[Dict[str, Any]] = payload.get("nodes", [])
    edges_data: List[Dict[str, Any]] = payload.get("edges", [])

    sem_nodes: Dict[str, SemanticNode] = {}
    for n in nodes_data:
        md = n.get("metadata") or {}
        # old path
        # pointers = _extract_pointers_from_metadata(md)
        # new path (MCP-style references)
        # if not pointers and n.get("references"):
        pointers = _extract_pointers_from_references(n["references"])

        sem = SemanticNode(
            node_id=UUID(n["id"]),
            parent_id=UUID(md["parent_id"]) if md.get("parent_id") else None,
            node_type=md.get("semantic_node_type", "TEXT_FLOW"),
            title=n.get("label") or "",
            total_content_pointers=pointers,
            child_nodes=[],
        )
        sem_nodes[n["id"]] = sem

    children_by_parent: Dict[str, List[str]] = defaultdict(list)
    for e in edges_data:
        relation = e.get("relation") or e.get("predicate")
        if relation != "HAS_CHILD":
            continue

        if "source_ids" in e or "target_ids" in e:
            src_ids = e.get("source_ids") or []
            tgt_ids = e.get("target_ids") or []
            if not src_ids or not tgt_ids:
                continue
            parent_id = src_ids[0]
            child_id = tgt_ids[0]
        else:
            parent_id = e.get("subject_id")
            child_id = e.get("object_id")
            if not parent_id or not child_id:
                continue

        children_by_parent[parent_id].append(child_id)

    for pid, child_ids in children_by_parent.items():
        parent_sem = sem_nodes.get(pid)
        if not parent_sem:
            continue
        for cid in child_ids:
            child_sem = sem_nodes.get(cid)
            if child_sem:
                parent_sem.child_nodes.append(child_sem)

    root_candidates = [s for s in sem_nodes.values() if s.parent_id is None]
    if root_candidates:
        return root_candidates[0]
    else:
        all_child_ids = {cid for cids in children_by_parent.values() for cid in cids}
        all_ids = set(sem_nodes.keys())
        root_ids = list(all_ids - all_child_ids)
        return sem_nodes[root_ids[0]] if root_ids else list(sem_nodes.values())[0]
    
from langchain_google_genai import ChatGoogleGenerativeAI

def all_child_from_root(root: SemanticNode, results = None):
    if results is None:
        results = []
    results.extend(root.child_nodes)
    for node in root.child_nodes:
        all_child_from_root(node, results)
    return results

available_node_ids = ContextVar("indexing_nodes", default=set())
class IndexingResponse(BaseModel):
    "Represent a single indexing result of a node"
    node_id : str = Field(description = "The node id this index result is representing")
    canonical_title: str = Field(description = "best searchable title")
    keywords: list[str] = Field(description = "5-12 short keywords")
    aliases: list[str] = Field(description = "0-5 alternative phrasings")
    provision: str = Field(description = "clauses, terms, sections, and schedules. Example: 'Schedule 3.1', 'Term 5a', 'Clause 3.12.2(a)'")
    @model_validator(mode='after')
    def _check_consistency(self):
        node_set = available_node_ids.get()
        try:
            node_set.remove(str(self.node_id))
        except KeyError:
            raise Exception(f'node_id {self.node_id} duplicated or not exist')
        return self
class BatchIndexResponse(BaseModel):
    "a list of batch run of indexing"
    index : list[IndexingResponse] = Field(description = "a list of indexing result. Each response member/element must have 1 and only 1 corresponding index. Each key is the input node's id")
    # @model_validator(mode='after')
    # def _check_used_all_node_id(self):
    #     node_set = available_node_ids.get()
    #     assert (set(str(i.node_id) for i in self.index) == node_set)
    #     return self    
class IdMapping():
    forward_map = {}
    backward_map = {}
    def to_uuid(self, short_id):
        return self.backward_map.get(short_id)
    def to_short_id(self, id, title):
        if id not in self.forward_map:
            self.forward_map[id] = f"nid:{len(self.forward_map)}:{title}"
            self.backward_map[self.forward_map[id]] = id
        
        return self.forward_map[id]
    pass
def build_index_terms_for_semantic_node(
    sem_nodes: list["SemanticNode"],
) -> list[IndexingResponse]:
    from langchain_core.messages import HumanMessage, SystemMessage
    # reconstruct text for better indexing
    # text = reconstruct_text_from_pointers(sem_node.total_content_pointers, source_map)
    # title = sem_node.title
    
    id_map = IdMapping()
    dumped = []
    for n in sem_nodes:
        temp = n.model_dump()
        temp['node_id'] = id_map.to_short_id(temp['node_id'], temp['title'])
        dumped.append(temp)
    for dumped_n in dumped:
        dumped_n['parent_id'] = id_map.to_short_id(dumped_n['parent_id'], dumped_n['title'])
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    def iter_minibatches(data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size]
    def get_node_verbatim(node_dict):
        return {"node_id":node_dict["node_id"], "contents": [i["verbatim_text"] for i in node_dict["total_content_pointers"]] }
    batch_result = []
    @memory.cache
    def get_minibatch_result(messages):
        retries = 0
        retry_max = 3
        
        while True:
            token = available_node_ids.set(set(all_ids))
            try:
                res: dict = llm.with_structured_output(BatchIndexResponse, include_raw = True).invoke(messages) # type: ignore
                if not res.get('parsing_error'):
                    break
                else:
                    messages.append(SystemMessage(f'Result Parsing error: {str(res.get("parsing_error"))}'))
                out_node_set = set(str(i.node_id) for i in res['parsed'].index)
                in_node_set = set(all_ids)
                if not (in_node_set == out_node_set):
                    raise Exception (f"Extra output nodes: {(out_node_set - in_node_set) or None}, unsatisfied input nodes: {(in_node_set- out_node_set) or None}")
            finally:
                available_node_ids.reset(token)
                retries += 1
                if retries > retry_max:
                    raise Exception(f"Retry max {retry_max} reached")
        return res
    for batch in iter_minibatches(dumped, 25):
        print(batch)
        all_ids = [str(n["node_id"]) for n in batch]
        
        messages = [SystemMessage("""You will be sent some nodes data. Generate some search terms for them. Need not go too deep to investigate the nodes relationship.
this is a superficial task. Each node must return an index
"""), HumanMessage(f"""
Nodes to index:
{[get_node_verbatim(i) for i in batch]}
""")]
        res = get_minibatch_result(messages)
        batch_result.extend(res['parsed'].index)
    for n in batch_result:
        n.node_id = id_map.to_uuid(n.node_id)
    return batch_result