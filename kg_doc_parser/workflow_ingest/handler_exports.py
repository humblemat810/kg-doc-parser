from __future__ import annotations

from typing import Literal

from kogwistar.runtime import StepContext

from .adapters import select_primary_collection
from .document_adapters import project_multimodal_source_evidence
from .document_models import DocumentTree
from .handler_state import parser_source_map, state_object_map
from .model_validation import validate_backend_payload
from .models import (
    GroundedSourceRecord,
    WorkflowExportBundle,
    WorkflowIngestInput,
)
from .runtime_contracts import WorkflowRuntimeDeps
from .semantics import SemanticNode, semantic_tree_to_kge_payload
from .serialization import JsonValue, decode_json_object


def build_export_bundle(
    *,
    ctx: StepContext,
    runtime_deps: WorkflowRuntimeDeps | None = None,
) -> WorkflowExportBundle:
    normalized = validate_backend_payload(
        WorkflowIngestInput, ctx.state_view["normalized_input"]
    )
    collection = select_primary_collection(normalized)
    tree = SemanticNode.model_validate(ctx.state_view["semantic_tree"])
    graph_payload = semantic_tree_to_kge_payload(
        tree,
        doc_id=collection.collection_id,
        parser_source_map=parser_source_map(ctx.state_view["parser_source_map"]),
    )
    raw_grouping_tree = collection.metadata.get("document_tree")
    grouping_tree: dict[str, JsonValue] | None = None
    grouping_document: DocumentTree | None = None
    source_revision: str | None = None
    if raw_grouping_tree is not None:
        grouping_tree = decode_json_object(
            raw_grouping_tree,
            label="document_tree metadata",
        )
        tree_fields = {
            "document_id",
            "tree_revision",
            "root_id",
            "source_items",
            "nodes",
        }
        if tree_fields.intersection(grouping_tree):
            grouping_document = DocumentTree.model_validate(grouping_tree)
        revision_value = grouping_tree.get("source_revision")
        if isinstance(revision_value, str) and revision_value:
            source_revision = revision_value
    if source_revision is not None:
        graph_payload = semantic_tree_to_kge_payload(
            tree,
            doc_id=collection.collection_id,
            parser_source_map=parser_source_map(ctx.state_view["parser_source_map"]),
            source_revision=source_revision,
        )
    if grouping_document is not None:
        graph_payload = project_multimodal_source_evidence(
            graph_payload,
            grouping_document,
            source_namespace=collection.source_namespace,
        )
    authoritative_source_map = {
        unit_id: GroundedSourceRecord.model_validate(record)
        for unit_id, record in state_object_map(
            ctx.state_view["authoritative_source_map"]
        ).items()
    }
    deps = runtime_deps or {}
    grouping_contract: Literal["optional", "required"] = (
        "required" if deps.get("require_grouping_tree") is True else "optional"
    )
    persistence_mode = str(
        ctx.state_view.get(
            "persistence_mode", deps.get("persistence_mode", "local_debug")
        )
    )
    kg_authority = str(
        ctx.state_view.get("kg_authority", deps.get("kg_authority", "local"))
    )
    return WorkflowExportBundle(
        graph_payload=graph_payload,
        grouping_tree=grouping_tree,
        grouping_contract=grouping_contract,
        authoritative_source_map=authoritative_source_map,
        embedding_spaces=collection.embedding_spaces,
        consolidation_candidates=[],
        retrieval_metadata=decode_json_object(
            {
                "embedding_spaces": collection.embedding_spaces,
                "supports_hybrid_retrieval": True,
                "supports_split_embedding_spaces": True,
                "collection_modality": collection.modality,
            },
            label="retrieval metadata",
        ),
        persistence_mode="server_canonical"
        if persistence_mode == "server_canonical"
        else "local_debug",
        kg_authority="server" if kg_authority == "server" else "local",
        canonical_write_confirmed=False,
        parser_owner="local",
        server_parser_used=False,
        persisted_to_knowledge_engine=False,
    )
