"""Recovery-generation and persisted child-lineage validation."""

from __future__ import annotations

from kogwistar.engine_core.engine import GraphKnowledgeEngine

from .ocr_runtime_contracts import OCRWorkflowState


def recovery_generation(state: OCRWorkflowState) -> int:
    """Read a validated root recovery generation from checkpoint state."""
    value = state.get("ocr_page_recovery_generation", 0)
    if value is None:
        return 0
    if type(value) is not int or value < 0:
        raise ValueError("OCR page recovery generation must be a non-negative integer")
    return value


def semantic_parse_generation(recovery_generation: int) -> str:
    """Return the immutable semantic-parser generation for one root run."""
    return (
        "semantic-v1"
        if recovery_generation == 0
        else f"semantic-recovery-v{recovery_generation}"
    )


def next_root_recovery_run_id(
    *, conversation_engine: GraphKnowledgeEngine, base_run_id: str
) -> tuple[str, int]:
    """Choose the first unused deterministic root recovery incarnation."""
    generation = 1
    while True:
        candidate = f"{base_run_id}|recovery-{generation}"
        completed = conversation_engine.read.get_nodes(
            where={
                "$and": [
                    {"entity_type": "workflow_completed"},
                    {"run_id": candidate},
                ]
            },
            limit=1,
        )
        if not completed:
            return candidate, generation
        generation += 1


def validate_root_invocation_lineage(
    conversation_engine: GraphKnowledgeEngine,
    *,
    root_run_id: str,
    recovery_generation: int,
) -> None:
    """Require exactly one persisted edge for every accepted child workflow."""
    suffix = f"|recovery-{recovery_generation}" if recovery_generation else ""
    expected = {
        (
            "kg_doc_parser.ocr_pages.v1",
            "ocr-pages-v1" if not recovery_generation else f"ocr-pages-recovery-v{recovery_generation}",
            f"{root_run_id}|ocr-pages{suffix}",
        ),
        (
            "kg_doc_parser.ocr_grouping.v1",
            "grouping-v1" if not recovery_generation else f"grouping-recovery-v{recovery_generation}",
            f"{root_run_id}|grouping{suffix}",
        ),
        (
            "kg_doc_parser.ingest.v1",
            "semantic-v1" if not recovery_generation else f"semantic-recovery-v{recovery_generation}",
            f"{root_run_id}|semantic{suffix}",
        ),
    }
    expected_edge_ids = [
        f"wf_invoked|{root_run_id}|{child_run_id}"
        for _workflow_id, _invocation_key, child_run_id in expected
    ]
    counts: dict[tuple[str, str, str], int] = {}
    for edge in conversation_engine.read.get_edges(
        ids=expected_edge_ids,
        limit=len(expected_edge_ids),
    ):
        metadata = edge.metadata or {}
        if metadata.get("relation") != "wf_invoked":
            continue
        if metadata.get("parent_run_id") != root_run_id:
            continue
        key = (
            str(metadata.get("workflow_id")),
            str(metadata.get("invocation_key")),
            str(metadata.get("child_run_id")),
        )
        counts[key] = counts.get(key, 0) + 1
    missing = expected - counts.keys()
    duplicate = sorted(key for key, count in counts.items() if key in expected and count != 1)
    if missing or duplicate:
        details: list[str] = []
        if missing:
            details.append(f"missing={sorted(missing)!r}")
        if duplicate:
            details.append(f"duplicate={duplicate!r}")
        raise RuntimeError(
            f"OCR root persisted invocation lineage is invalid for {root_run_id!r}: "
            + "; ".join(details)
        )
