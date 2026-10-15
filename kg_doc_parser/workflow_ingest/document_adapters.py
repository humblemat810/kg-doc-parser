"""Adapters from the existing normalized OCR records to document evidence."""
from __future__ import annotations

import re
from collections.abc import Mapping
from itertools import pairwise
from typing import Literal, cast

from .adapters import _allocated_cluster_numbers
from .document_grouping import GroupProposal, merge_groups
from .document_models import (
    BoundingBox,
    ChildRef,
    DocumentNode,
    DocumentTree,
    MediaRef,
    OCRMetadata,
    SourceItem,
    SourceLocation,
    SourceSlice,
    Summary,
    SummaryPolicy,
    extract_reference_mentions,
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
            media_digest = unit.metadata.get("media_sha256")
            media_size = unit.metadata.get("media_byte_length")
            media_mime = unit.metadata.get("mime_type")
            has_media = (
                unit.source_uri is not None
                and isinstance(media_digest, str)
                and re.fullmatch(r"[0-9a-f]{64}", media_digest) is not None
                and isinstance(media_size, int)
                and media_size >= 0
                and isinstance(media_mime, str)
            )
            if not text and not has_media:
                continue
            precise_bbox = (
                unit.bbox
                if unit.bbox is not None
                and unit.bbox.x_min < unit.bbox.x_max
                and unit.bbox.y_min < unit.bbox.y_max
                else None
            )
            coordinate_system = unit.metadata.get("coordinate_system", "pixels")
            if coordinate_system not in {"pixels", "relative", "points"}:
                precise_bbox = None
                coordinate_system = "pixels"
            coordinate_system_literal = cast(
                Literal["pixels", "relative", "points"], coordinate_system
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
                        coordinate_system=coordinate_system_literal,
                    )
                ),
                accuracy="region" if precise_bbox is not None else "page",
            )
            ocr_metadata = _ocr_metadata(unit.metadata)
            page_annotations = _page_annotations(page.metadata)
            kind = "text" if unit.modality in {"text", "ocr_text"} else "non_text:description"
            if has_media:
                assert unit.source_uri is not None
                assert isinstance(media_digest, str)
                assert isinstance(media_size, int)
                assert isinstance(media_mime, str)
                items.append(
                    SourceItem(
                        item_id=item_id,
                        kind=unit.modality,
                        location=location,
                        media=MediaRef(
                            uri=unit.source_uri,
                            mime_type=media_mime,
                            sha256_hex=media_digest,
                            byte_length=media_size,
                        ),
                        ocr_metadata=ocr_metadata,
                        annotations={
                            **page_annotations,
                            **({"description": unit.description} if unit.description else {}),
                        },
                    )
                )
                leaf_kind = unit.modality
                leaf_part = SourceSlice(item_id=item_id)
            else:
                if text is None:
                    raise ValueError("non-media source item requires provider text or description")
                leaf_kind = kind
                leaf_part = SourceSlice(item_id=item_id, start=0, end=len(text))
                items.append(
                    SourceItem(
                        item_id=item_id,
                        kind=kind,
                        text=text,
                        location=location,
                        ocr_metadata=ocr_metadata,
                        annotations=page_annotations,
                    )
                )
            leaves.append(
                DocumentNode(
                    node_id=f"leaf:{item_id}",
                    kind=leaf_kind,
                    parts=(leaf_part,),
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
    tree = DocumentTree(
        document_id=collection.collection_id,
        source_revision=source_revision,
        tree_revision=1,
        root_id="root",
        source_items=tuple(items),
        nodes=tuple(nodes),
    )
    return tree.model_copy(update={"references": extract_reference_mentions(tree)})


def _ocr_metadata(metadata: Mapping[str, object]) -> OCRMetadata | None:
    """Preserve provider identity only when the source declares it completely."""
    provider = metadata.get("ocr_provider")
    model = metadata.get("ocr_model_name") or metadata.get("ocr_model")
    configuration = metadata.get("ocr_configuration_fingerprint")
    if not isinstance(provider, str) or not provider.strip():
        return None
    if not isinstance(model, str) or not model.strip():
        return None
    if not isinstance(configuration, str) or not configuration.strip():
        return None
    confidence = metadata.get("ocr_confidence")
    return OCRMetadata(
        provider=provider,
        model=model,
        configuration_fingerprint=configuration,
        confidence=confidence if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) else None,
    )


def _page_annotations(metadata: Mapping[str, object]) -> dict[str, str]:
    annotations: dict[str, str] = {}
    for key in ("incomplete_text", "incomplete_words_on_edge", "data_loss_likelihood", "scan_quality"):
        value = metadata.get(key)
        if isinstance(value, (str, int, float, bool)):
            annotations[f"ocr:{key}"] = str(value)
    return annotations


def suggest_cross_page_continuations(tree: DocumentTree) -> tuple[GroupProposal, ...]:
    """Suggest only OCR-signaled cross-page joins; never infer semantic links."""
    nodes = {node.node_id: node for node in tree.nodes}
    items = {item.item_id: item for item in tree.source_items}
    root = nodes[tree.root_id]
    if not root.children:
        return ()
    frontier = [nodes[child.node_id] for child in root.children]
    if any(not node.parts for node in frontier):
        return ()
    leaves = frontier
    ordered_leaves = sorted(
        leaves,
        key=lambda node: min(
            items[part.item_id].location.page_index or 0 for part in node.parts
        ),
    )
    proposals: list[GroupProposal] = []
    for previous, current in pairwise(ordered_leaves):
        previous_pages = [
            page
            for part in previous.parts
            if (page := items[part.item_id].location.page_index) is not None
        ]
        current_pages = [
            page
            for part in current.parts
            if (page := items[part.item_id].location.page_index) is not None
        ]
        if not previous_pages or not current_pages or max(previous_pages) + 1 != min(current_pages):
            continue
        previous_item = items[previous.parts[-1].item_id]
        if previous_item.annotations.get("ocr:incomplete_text") != "True" and previous_item.annotations.get(
            "ocr:incomplete_words_on_edge"
        ) != "True":
            continue
        proposals.append(
            GroupProposal(
                kind="layout:cross-page-continuation",
                members=(previous.node_id, current.node_id),
            )
        )
    return tuple(proposals)


def attach_document_tree_metadata(workflow_input: WorkflowIngestInput, tree: DocumentTree) -> None:
    """Expose validated lineage through the normalized input metadata."""
    workflow_input.collections[0].metadata["document_tree"] = tree.model_dump(mode="json")


def apply_summary_updates(
    tree: DocumentTree,
    updates: Mapping[str, Summary],
) -> DocumentTree:
    """Apply grounded summaries through the tree validator before persistence."""
    if not updates:
        return tree
    node_ids = {node.node_id for node in tree.nodes}
    unknown = set(updates) - node_ids
    if unknown:
        raise ValueError(f"summary update references unknown node(s): {sorted(unknown)}")
    nodes = tuple(
        node.model_copy(update={"summary": updates[node.node_id]}) if node.node_id in updates else node
        for node in tree.nodes
    )
    payload = tree.model_dump(mode="json")
    payload["nodes"] = nodes
    payload["tree_revision"] = tree.tree_revision + 1
    return DocumentTree.model_validate(payload)


def select_summary_targets(
    tree: DocumentTree,
    policy: SummaryPolicy | None = None,
) -> tuple[str, ...]:
    """Select bounded, evidence-bearing grouping nodes for optional summaries.

    This deterministic planner does not call a provider or attach a summary,
    so it cannot change source evidence. Token cost is conservatively
    estimated from text size before a downstream provider is invoked.
    """

    policy = policy or SummaryPolicy()
    if not policy.enabled:
        return ()
    nodes = {node.node_id: node for node in tree.nodes}
    items = {item.item_id: item for item in tree.source_items}

    def descendants(node_id: str) -> set[str]:
        node = nodes[node_id]
        if node.parts:
            return {part.item_id for part in node.parts}
        result: set[str] = set()
        for child in node.children:
            result.update(descendants(child.node_id))
        return result

    def depth(node_id: str) -> int:
        result = 0
        current = nodes[node_id]
        while current.node_id != tree.root_id:
            parent = next(
                parent
                for parent in tree.nodes
                if any(child.node_id == current.node_id for child in parent.children)
            )
            result += 1
            current = parent
        return result

    candidates: list[tuple[int, str, int]] = []
    eligible = set(policy.eligible_kinds)
    for node in tree.nodes:
        if node.parts or node.summary is not None or node.kind not in eligible:
            continue
        item_ids = descendants(node.node_id)
        text_chars = sum(len(items[item_id].text or "") for item_id in item_ids)
        if len(item_ids) < policy.minimum_source_items or text_chars < policy.minimum_source_chars:
            continue
        child_sizes = [
            sum(len(items[item_id].text or "") for item_id in descendants(child.node_id))
            for child in node.children
        ]
        information_gain = text_chars - max(child_sizes, default=0)
        if information_gain < policy.minimum_information_gain_chars:
            continue
        estimated_tokens = max(1, (text_chars + 3) // 4)
        candidates.append((depth(node.node_id), node.node_id, estimated_tokens))

    selected: list[str] = []
    used_tokens = 0
    for _depth, node_id, estimated_tokens in sorted(candidates):
        if len(selected) >= policy.maximum_targets:
            break
        bounded_tokens = min(estimated_tokens, policy.maximum_input_tokens)
        if used_tokens + bounded_tokens > policy.maximum_input_tokens:
            continue
        selected.append(node_id)
        used_tokens += bounded_tokens
    return tuple(selected)


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
    # Keep the final meaningful group as a node.  Replacing it with the
    # document root loses its identity and any derived metadata attached to
    # the group.  The root is a structural wrapper, so it may point to the
    # single accepted group while the group retains its real kind and source
    # ownership.
    by_id[tree.root_id] = DocumentNode(
        node_id=tree.root_id,
        kind="document",
        children=tuple(ChildRef(node_id=node_id) for node_id in new_frontier),
    )
    payload = tree.model_dump(mode="json")
    payload["nodes"] = tuple(by_id.values())
    payload["tree_revision"] = tree.tree_revision + 1
    return DocumentTree.model_validate(payload)


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
    "apply_summary_updates",
    "attach_document_tree_metadata",
    "build_document_tree",
    "extract_reference_mentions",
    "select_summary_targets",
    "suggest_cross_page_continuations",
]
