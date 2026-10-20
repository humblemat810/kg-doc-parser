"""Validation of persisted OCR grouping artifacts before workflow resume."""

from __future__ import annotations

from .document_models import DocumentTree
from .ocr_pipeline import OCRWorkflowStateStore


def validated_grouping_checkpoint(
    *,
    state_store: OCRWorkflowStateStore,
    document_id: str,
    tree: DocumentTree,
    grouping_policy_version: str,
) -> tuple[DocumentTree | None, str | None]:
    """Validate persisted grouping state before a workflow can resume.

    A terminal runtime checkpoint may return without executing the parser's
    restore step. Validate the SQLite tree/frontier seam before requesting
    that resume so terminal state cannot bypass corruption detection.
    """
    try:
        restored = state_store.load_document_tree(
            document_id=document_id,
            source_fingerprint=tree.source_fingerprint(),
            grouping_policy_version=grouping_policy_version,
        )
    except ValueError as exc:
        if "frontier" in str(exc):
            raise RuntimeError(
                "grouping checkpoint frontier does not match the restored tree"
            ) from exc
        raise RuntimeError(f"persisted grouping checkpoint is invalid: {exc}") from exc
    revision = state_store.document_tree_revision(document_id=document_id)
    if restored is None:
        if revision is not None:
            raise RuntimeError(
                "grouping checkpoint exists but does not match the current source or policy"
            )
        return None, None

    persisted_frontier = state_store.load_grouping_frontier(
        document_id=document_id,
        source_fingerprint=restored.source_fingerprint(),
        grouping_policy_version=grouping_policy_version,
    )
    if persisted_frontier is None:
        raise RuntimeError(
            "grouping checkpoint exists but its frontier is missing or mismatched"
        )
    restored_root = next(
        node for node in restored.nodes if node.node_id == restored.root_id
    )
    expected_frontier = (
        tuple(child.node_id for child in restored_root.children)
        if restored_root.children
        else (restored_root.node_id,)
    )
    if persisted_frontier != expected_frontier:
        raise RuntimeError("grouping checkpoint frontier does not match the restored tree")
    return restored, revision
