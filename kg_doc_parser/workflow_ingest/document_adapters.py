"""Adapters from the existing normalized OCR records to document evidence."""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from math import ceil, floor, isfinite
from typing import Literal, cast

from kogwistar.engine_core.multimodal import (
    LegacyLocator,
    MultimodalSpan,
    SpatialRegionLocator,
    TemporalIntervalLocator,
)
from kogwistar.id_provider import stable_id

from .adapters import _allocated_cluster_numbers
from .document_grouping import GroupProposal, group_node_id, merge_groups
from .document_models import (
    BoundingBox,
    ChildRef,
    DocumentNode,
    DocumentTree,
    EvidenceSlice,
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
from .serialization import JsonObject, JsonValue, decode_json_object


@dataclass(frozen=True, slots=True)
class UnrepresentedSourceDiagnostic:
    """Describe an OCR unit that cannot be represented losslessly."""

    unit_id: str | None
    page_number: int
    modality: str
    reason: str


class UnrepresentedSourceUnitError(ValueError):
    """Raised instead of silently dropping an OCR observation."""

    def __init__(self, diagnostic: UnrepresentedSourceDiagnostic) -> None:
        self.diagnostic = diagnostic
        super().__init__(
            "unrepresented OCR source unit "
            f"{diagnostic.unit_id or '<generated>'} on page {diagnostic.page_number}: "
            f"{diagnostic.reason}"
        )


class UnrepresentedSourceUnitsError(UnrepresentedSourceUnitError):
    """Report every source unit rejected before tree construction.

    The singular exception remains the compatibility type for callers that
    catch it.  The aggregate retains the complete loss report so a provider
    or caller can repair all rejected non-text units in one pass.
    """

    def __init__(self, diagnostics: tuple[UnrepresentedSourceDiagnostic, ...]) -> None:
        if not diagnostics:
            raise ValueError("at least one unrepresented source diagnostic is required")
        self.diagnostics = diagnostics
        self.diagnostic = diagnostics[0]
        ValueError.__init__(
            self,
            "unrepresented OCR source units: "
            + "; ".join(
                f"{item.unit_id or '<generated>'} on page {item.page_number}: {item.reason}"
                for item in diagnostics
            ),
        )


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
    rejected: list[UnrepresentedSourceDiagnostic] = []
    seen_unit_ids: set[str] = set()
    for page in collection.pages:
        allocated = _allocated_cluster_numbers(page.units)
        for unit, cluster_number in zip(page.units, allocated, strict=True):
            if unit.unit_id is not None:
                if unit.unit_id in seen_unit_ids:
                    raise ValueError(f"duplicate OCR source unit id: {unit.unit_id}")
                seen_unit_ids.add(unit.unit_id)
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
                # Do not let a structurally valid but unrepresentable OCR unit
                # disappear before source-coverage validation can see it.
                rejected.append(
                    UnrepresentedSourceDiagnostic(
                        unit_id=unit.unit_id,
                        page_number=page.page_number,
                        modality=unit.modality,
                        reason=(
                            "non-text evidence requires a description or a verified "
                            "content-addressed media reference"
                        ),
                    )
                )
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
            declared_accuracy = unit.metadata.get("spatial_accuracy", "page")
            accuracy = _spatial_accuracy(
                declared_accuracy,
                has_bbox=precise_bbox is not None,
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
                accuracy=accuracy,
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
    if rejected:
        if len(rejected) == 1:
            raise UnrepresentedSourceUnitError(rejected[0])
        raise UnrepresentedSourceUnitsError(tuple(rejected))
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


def project_multimodal_source_evidence(
    graph_payload: dict[str, JsonValue],
    tree: DocumentTree,
    *,
    source_namespace: str | None = None,
) -> dict[str, JsonValue]:
    """Add first-class, source-grounded media evidence to a semantic graph.

    The document grouping tree is authoritative for media identity and reading
    order. Media is represented using Kogwistar's existing multimodal grounding
    contract, not as OCR text or an invented semantic description.
    """
    ordered_media: list[SourceItem] = []
    items = {item.item_id: item for item in tree.source_items}
    nodes_by_id = {node.node_id: node for node in tree.nodes}
    pending = [tree.root_id]
    while pending:
        node = nodes_by_id[pending.pop()]
        if node.parts:
            for part in node.parts:
                item = items[part.item_id]
                if item.media is not None:
                    ordered_media.append(item)
        else:
            pending.extend(child.node_id for child in reversed(node.children))
    if not ordered_media:
        return graph_payload
    if (
        not isinstance(source_namespace, str)
        or not source_namespace.strip()
        or source_namespace != source_namespace.strip()
    ):
        raise ValueError(
            "multimodal graph projection requires an explicit source namespace"
        )

    raw_nodes = graph_payload.get("nodes")
    raw_edges = graph_payload.get("edges")
    if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
        raise TypeError("multimodal graph projection requires node and edge lists")
    nodes = [decode_json_object(record, label="graph node") for record in raw_nodes]
    edges = [decode_json_object(record, label="graph edge") for record in raw_edges]

    semantic_roots: list[tuple[str, JsonObject]] = []
    for record in nodes:
        node_id = record.get("id")
        metadata = record.get("metadata")
        if (
            isinstance(node_id, str)
            and isinstance(metadata, dict)
            and metadata.get("semantic_node_type") is not None
            and metadata.get("parent_id") is None
        ):
            semantic_roots.append((node_id, record))
    if len(semantic_roots) != 1:
        raise ValueError(
            "multimodal graph projection requires exactly one semantic root"
        )
    semantic_root_id = semantic_roots[0][0]

    existing_node_ids = {
        record_id
        for record in nodes
        if isinstance((record_id := record.get("id")), str)
    }
    existing_edge_ids = {
        record_id
        for record in edges
        if isinstance((record_id := record.get("id")), str)
    }
    for reading_order, item in enumerate(ordered_media):
        media = item.media
        if media is None:
            continue
        mime_type = media.mime_type.partition(";")[0].strip().lower()
        kind = item.kind.casefold()
        if kind == "table" or kind.startswith("table:"):
            modality: Literal[
                "image", "audio", "video", "pdf_page", "table", "chart", "webpage"
            ] = "table"
        elif kind == "chart" or kind.startswith("chart:"):
            modality = "chart"
        elif mime_type.startswith("image/"):
            modality = "image"
        elif mime_type.startswith("audio/"):
            modality = "audio"
        elif mime_type.startswith("video/"):
            modality = "video"
        elif mime_type == "application/pdf":
            modality = "pdf_page"
        elif mime_type in {"text/html", "application/xhtml+xml"}:
            modality = "webpage"
        else:
            raise ValueError(
                f"unsupported multimodal source MIME type {media.mime_type!r} "
                f"for item {item.item_id!r}"
            )

        location = item.location
        bbox = location.bbox
        locator: SpatialRegionLocator | TemporalIntervalLocator | LegacyLocator
        if (
            bbox is not None
            and location.accuracy in {"region", "line", "word", "character"}
            and bbox.coordinate_system in {"pixels", "relative"}
        ):
            locator = SpatialRegionLocator(
                coordinate_system=(
                    "normalized_0_1"
                    if bbox.coordinate_system == "relative"
                    else "pixels"
                ),
                x=bbox.x0,
                y=bbox.y0,
                width=bbox.x1 - bbox.x0,
                height=bbox.y1 - bbox.y0,
                page_number=(
                    location.page_index + 1
                    if location.page_index is not None
                    else None
                ),
                frame_index=None,
                timestamp_ms=None,
            )
        elif (
            modality in {"audio", "video"}
            and location.start_seconds is not None
            and location.end_seconds is not None
            and isfinite(location.start_seconds)
            and isfinite(location.end_seconds)
        ):
            start_ms = max(0, floor(location.start_seconds * 1000))
            locator = TemporalIntervalLocator(
                start_ms=start_ms,
                end_ms=max(start_ms + 1, ceil(location.end_seconds * 1000)),
            )
        else:
            # Legacy locators preserve page-only, point-based, and otherwise
            # non-normalizable coordinates without claiming spatial accuracy.
            locator = LegacyLocator(
                payload={
                    "source_item_id": item.item_id,
                    "kind": item.kind,
                    "location": cast(
                        JsonObject,
                        decode_json_object(
                            location.model_dump(mode="json"),
                            label="source item location",
                        ),
                    ),
                }
            )
        span = MultimodalSpan(
            source_namespace=source_namespace,
            resource_id=media.uri,
            resource_revision_id=tree.source_revision,
            content_sha256=media.sha256_hex,
            modality=modality,
            locator=locator,
        )
        span_payload = cast(JsonObject, span.model_dump(mode="json"))
        source_node_id = str(
            stable_id(
                "workflow_ingest.multimodal_source_item",
                tree.document_id,
                tree.source_revision,
                item.item_id,
            )
        )
        edge_id = str(
            stable_id(
                "workflow_ingest.edge",
                "HAS_SOURCE_EVIDENCE",
                semantic_root_id,
                source_node_id,
                tree.document_id,
            )
        )
        if source_node_id in existing_node_ids or edge_id in existing_edge_ids:
            raise ValueError("multimodal source evidence identity collides with graph")
        description = item.annotations.get("description", "").strip()
        label = description or item.kind
        grounding: JsonObject = {"spans": [], "multimodal_spans": [span_payload]}
        nodes.append(
            {
                "id": source_node_id,
                "label": label,
                "type": "entity",
                "doc_id": tree.document_id,
                "summary": description or f"{item.kind} source evidence",
                "metadata": {
                    "doc_id": tree.document_id,
                    "entity_type": "multimodal_source_evidence",
                    "source_item_id": item.item_id,
                    "source_item_kind": item.kind,
                    "source_revision": tree.source_revision,
                    "reading_order": reading_order,
                    "mime_type": media.mime_type,
                    "media_sha256": media.sha256_hex,
                    "media_byte_length": media.byte_length,
                    "location": cast(
                        JsonValue,
                        decode_json_object(
                            location.model_dump(mode="json"),
                            label="source item location",
                        ),
                    ),
                    "annotations": cast(JsonValue, item.annotations),
                    "insertion_method": "workflow_ingest",
                },
                "mentions": [grounding],
                "aggregate_mentions": [],
            }
        )
        edges.append(
            {
                "id": edge_id,
                "label": "source evidence",
                "type": "relationship",
                "doc_id": tree.document_id,
                "summary": f"{semantic_root_id}->{source_node_id}",
                "relation": "HAS_SOURCE_EVIDENCE",
                "source_ids": [semantic_root_id],
                "target_ids": [source_node_id],
                "source_edge_ids": [],
                "target_edge_ids": [],
                "mentions": [grounding],
                "metadata": {
                    "doc_id": tree.document_id,
                    "source_item_id": item.item_id,
                    "reading_order": reading_order,
                    "insertion_method": "workflow_ingest",
                },
            }
        )
        existing_node_ids.add(source_node_id)
        existing_edge_ids.add(edge_id)

    result = dict(graph_payload)
    result["nodes"] = cast(JsonValue, nodes)
    result["edges"] = cast(JsonValue, edges)
    return result


def _spatial_accuracy(
    value: object,
    *,
    has_bbox: bool,
) -> Literal["page", "region", "line", "word", "character"]:
    """Keep spatial precision at the provider-declared level.

    A box alone is not evidence that the provider located text or media at
    region precision: page-wide fallback boxes are common in OCR adapters.
    Unknown or unusable declarations fail closed to page accuracy.
    """
    allowed = {"page", "region", "line", "word", "character"}
    if not has_bbox or not isinstance(value, str) or value not in allowed:
        return "page"
    return cast(
        Literal["page", "region", "line", "word", "character"],
        value,
    )


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
        # A page-level continuation flag is only meaningful for text-backed
        # leaves. It must never turn a figure, table, or caption into the tail
        # of the preceding paragraph merely because it starts on the next page.
        if previous.kind != "text" or current.kind != "text":
            continue
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


def _evidence_size(evidence: EvidenceSlice) -> int:
    """Estimate one evidence slice without losing its exact source bounds."""
    if evidence.text is not None:
        return len(evidence.text)
    media = evidence.media
    if media is None:
        raise ValueError("evidence slice has neither text nor media")
    return len(media.model_dump_json())


def _evidence_payload_size(evidence: EvidenceSlice) -> int:
    """Estimate the serialized provider payload for one evidence slice.

    Summary providers receive the complete typed slice, not only its text.
    Counting the serialized record keeps source metadata, media references,
    and grounding bounds inside the same conservative budget check.
    """
    return len(evidence.model_dump_json().encode("utf-8"))


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
    completed_targets = 0
    used_tokens = 0
    eligible = set(policy.eligible_kinds)
    for node in tree.nodes:
        summary = node.summary
        has_accepted_summary = summary is not None and summary.status in {
            "generated",
            "verified",
        }
        summary_matches_policy = False
        if summary is not None and summary.status in {"generated", "verified"}:
            generation_fingerprint = summary.generation_fingerprint
            summary_matches_policy = generation_fingerprint == policy.generation_fingerprint
        if summary_matches_policy and summary is not None:
            # The policy bounds accepted work for the whole tree, not just one
            # invocation. This keeps resume from resetting target/token caps.
            completed_targets += 1
            used_tokens += sum(
                max(1, (_evidence_payload_size(slice_) + 3) // 4)
                for slice_ in tree.evidence_slices(node.node_id)
                if slice_.item_id in summary.evidence_item_ids
            )
            continue
        # A completed summary from a different generation is stale and must be
        # eligible for replacement; non-completed states remain non-candidates.
        if node.parts or (summary is not None and not has_accepted_summary) or node.kind not in eligible:
            continue
        evidence = tree.evidence_slices(node.node_id)
        item_ids = {slice_.item_id for slice_ in evidence}
        text_chars = sum(_evidence_size(slice_) for slice_ in evidence)
        if len(item_ids) < policy.minimum_source_items or text_chars < policy.minimum_source_chars:
            continue
        child_sizes = [
            sum(_evidence_size(slice_) for slice_ in tree.evidence_slices(child.node_id))
            for child in node.children
        ]
        information_gain = text_chars - max(child_sizes, default=0)
        if information_gain < policy.minimum_information_gain_chars:
            continue
        payload_bytes = sum(_evidence_payload_size(slice_) for slice_ in evidence)
        estimated_tokens = max(1, (payload_bytes + 3) // 4)
        candidates.append((depth(node.node_id), node.node_id, estimated_tokens))

    selected: list[str] = []
    for _depth, node_id, estimated_tokens in sorted(candidates):
        if completed_targets + len(selected) >= policy.maximum_targets:
            break
        # The planner's estimate is also the provider boundary.  Clamping the
        # accounting while still sending the full evidence payload makes the
        # configured context limit decorative for one oversized node.
        if estimated_tokens > policy.maximum_input_tokens:
            continue
        if used_tokens + estimated_tokens > policy.maximum_input_tokens:
            continue
        selected.append(node_id)
        used_tokens += estimated_tokens
    return tuple(selected)


def apply_grouping_proposals(
    tree: DocumentTree,
    proposals: tuple[GroupProposal, ...],
    *,
    grouping_policy_version: str,
) -> DocumentTree:
    """Apply one validated frontier pass without rewriting source evidence."""
    if not proposals:
        # An empty provider pass carries no accepted mutation.  Treating it as
        # a new tree revision would make replay non-idempotent and can trigger
        # unnecessary summary invalidation.
        return tree
    root = next(node for node in tree.nodes if node.node_id == tree.root_id)
    if not root.children:
        if proposals:
            raise ValueError("grouping proposals require an internal document frontier")
        return tree
    by_id = {node.node_id: node for node in tree.nodes}
    pending_proposals: list[GroupProposal] = []
    for proposal in proposals:
        existing = by_id.get(
            group_node_id(
                document_id=tree.document_id,
                grouping_policy_version=grouping_policy_version,
                proposal=proposal,
            )
        )
        if existing is None:
            pending_proposals.append(proposal)
            continue
        if (
            existing.kind != proposal.kind
            or tuple(child.node_id for child in existing.children) != proposal.members
        ):
            raise ValueError("existing grouping node conflicts with replayed proposal")
        # A duplicate delivery of an accepted proposal is already committed.
        # Filter it before merging so a retry can safely contain both old and
        # new proposals even though the old members are no longer frontier IDs.
    if not pending_proposals:
        return tree
    frontier = tuple(child.node_id for child in root.children)
    new_frontier, created = merge_groups(
        document_id=tree.document_id,
        frontier=frontier,
        proposals=tuple(pending_proposals),
        grouping_policy_version=grouping_policy_version,
    )
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
    "UnrepresentedSourceDiagnostic",
    "UnrepresentedSourceUnitError",
    "UnrepresentedSourceUnitsError",
    "apply_grouping_passes",
    "apply_grouping_proposals",
    "apply_summary_updates",
    "attach_document_tree_metadata",
    "build_document_tree",
    "extract_reference_mentions",
    "select_summary_targets",
    "suggest_cross_page_continuations",
]
