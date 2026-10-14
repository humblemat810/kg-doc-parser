"""Adapters from the existing normalized OCR records to document evidence."""
from __future__ import annotations

from .adapters import _allocated_cluster_numbers
from .document_grouping import GroupProposal, merge_groups
from .document_models import (
    BoundingBox,
    ChildRef,
    DocumentNode,
    DocumentTree,
    SourceItem,
    SourceLocation,
    SourceSlice,
)
from .models import WorkflowIngestInput


def build_document_tree(
    workflow_input: WorkflowIngestInput,
    *,
    source_revision: str = "ocr-v1",
) -> DocumentTree | None:
    """Build the deterministic, lossless baseline tree for OCR observations.

    This is intentionally a baseline: it creates source-backed leaves and a
    document root, but never invents semantic groups. Non-text OCR entries are
    represented by their provider description until a real content-addressed
    media reference is available.
    """
    collection = workflow_input.collections[0]
    items: list[SourceItem] = []
    leaves: list[DocumentNode] = []
    for page in collection.pages:
        allocated = _allocated_cluster_numbers(page.units)
        for unit, cluster_number in zip(page.units, allocated, strict=True):
            if unit.unit_id:
                item_id = unit.unit_id
            else:
                prefix = "t" if unit.modality in {"text", "ocr_text"} else "i"
                item_id = f"{collection.collection_id}|p{page.page_number}_{prefix}{cluster_number}"
            text = unit.text if unit.modality in {"text", "ocr_text"} else unit.description
            if not text:
                continue
            precise_bbox = (
                unit.bbox
                if unit.bbox is not None
                and unit.bbox.x_min < unit.bbox.x_max
                and unit.bbox.y_min < unit.bbox.y_max
                else None
            )
            printed_page_label = page.metadata.get("printed_page_number")
            if not isinstance(printed_page_label, str):
                printed_page_label = None
            location = SourceLocation(
                page_index=page.page_number - 1,
                printed_page_label=printed_page_label,
                bbox=(
                    None
                    if precise_bbox is None
                    else BoundingBox(
                        x0=precise_bbox.x_min,
                        y0=precise_bbox.y_min,
                        x1=precise_bbox.x_max,
                        y1=precise_bbox.y_max,
                        coordinate_system="pixels",
                    )
                ),
                accuracy="region" if precise_bbox is not None else "page",
            )
            kind = "text" if unit.modality in {"text", "ocr_text"} else "non_text:description"
            items.append(SourceItem(item_id=item_id, kind=kind, text=text, location=location))
            leaves.append(
                DocumentNode(
                    node_id=f"leaf:{item_id}",
                    kind=kind,
                    parts=(SourceSlice(item_id=item_id, start=0, end=len(text)),),
                )
            )
    if not items:
        return None
    if len(leaves) == 1:
        root = DocumentNode(node_id="root", kind="document", parts=leaves[0].parts)
        nodes = (root,)
    else:
        root = DocumentNode(
            node_id="root",
            kind="document",
            children=tuple(ChildRef(node_id=leaf.node_id) for leaf in leaves),
        )
        nodes = (*leaves, root)
    return DocumentTree(
        document_id=collection.collection_id,
        source_revision=source_revision,
        tree_revision=1,
        root_id="root",
        source_items=tuple(items),
        nodes=tuple(nodes),
    )


def attach_document_tree_metadata(workflow_input: WorkflowIngestInput, tree: DocumentTree) -> None:
    """Expose validated lineage through the normalized input metadata."""
    workflow_input.collections[0].metadata["document_tree"] = tree.model_dump(mode="json")


def apply_grouping_proposals(
    tree: DocumentTree,
    proposals: tuple[GroupProposal, ...],
    *,
    grouping_policy_version: str,
) -> DocumentTree:
    """Apply one validated frontier pass without rewriting source evidence."""
    root = next(node for node in tree.nodes if node.node_id == tree.root_id)
    if not root.children:
        if proposals:
            raise ValueError("grouping proposals require an internal document frontier")
        return tree
    frontier = tuple(child.node_id for child in root.children)
    new_frontier, created = merge_groups(
        document_id=tree.document_id,
        frontier=frontier,
        proposals=proposals,
        grouping_policy_version=grouping_policy_version,
    )
    by_id = {node.node_id: node for node in tree.nodes}
    for node in created:
        by_id[node.node_id] = node
    if len(new_frontier) == 1:
        only = by_id[new_frontier[0]]
        replacement = DocumentNode(node_id=tree.root_id, kind="document", children=only.children)
        by_id.pop(only.node_id, None)
        by_id[tree.root_id] = replacement
    else:
        by_id[tree.root_id] = DocumentNode(
            node_id=tree.root_id,
            kind="document",
            children=tuple(ChildRef(node_id=node_id) for node_id in new_frontier),
        )
    return tree.model_copy(update={"nodes": tuple(by_id.values()), "tree_revision": tree.tree_revision + 1})


def apply_grouping_passes(
    tree: DocumentTree,
    passes: tuple[tuple[GroupProposal, ...], ...],
    *,
    grouping_policy_version: str,
) -> DocumentTree:
    """Apply a bounded sequence of adaptive grouping frontier passes."""
    current = tree
    for pass_number, proposals in enumerate(passes, start=1):
        current = apply_grouping_proposals(
            current,
            proposals,
            grouping_policy_version=f"{grouping_policy_version}:{pass_number}",
        )
    return current


__all__ = [
    "apply_grouping_passes",
    "apply_grouping_proposals",
    "attach_document_tree_metadata",
    "build_document_tree",
]
