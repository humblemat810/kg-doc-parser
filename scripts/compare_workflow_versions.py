"""Compare deterministic workflow-ingest metrics across two source checkouts.

Run this script from each checkout with the same corpus and compare the JSON
records. The semantic provider is deliberately deterministic and offline; this
script is not a live-provider quality benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def _fake_semantic_tree(*, collection: Any, parser_source_map: dict[str, dict[str, Any]], **_: Any) -> Any:
    from kg_doc_parser.workflow_ingest.semantics import (
        HydratedTextPointer,
        SemanticNode,
    )

    root = SemanticNode(
        title=collection.title,
        node_type="DOCUMENT_ROOT",
        total_content_pointers=[],
        child_nodes=[],
        level_from_root=0,
    )
    for unit_id, record in parser_source_map.items():
        if not record.get("participates_in_semantic_text", True):
            continue
        text = str(record.get("text", ""))
        root.child_nodes.append(
            SemanticNode(
                title=f"section:{unit_id}",
                node_type="TEXT_FLOW",
                total_content_pointers=[
                    HydratedTextPointer(
                        source_cluster_id=unit_id,
                        start_char=0,
                        end_char=max(0, len(text) - 1),
                        verbatim_text=text,
                    )
                ],
                child_nodes=[],
                level_from_root=1,
                parent_id=root.node_id,
            )
        )
    return root


def _max_tree_depth(node: Any) -> int:
    if isinstance(node, dict):
        children = list(node.get("child_nodes", []) or [])
    else:
        children = list(getattr(node, "child_nodes", []) or [])
    return max((1 + _max_tree_depth(child) for child in children), default=0)


def _coverage(final_state: dict[str, Any]) -> float | None:
    report = final_state.get("validation_report")
    if hasattr(report, "model_dump"):
        report = report.model_dump(mode="json")
    if isinstance(report, dict):
        for key in ("overall", "coverage", "final_coverage"):
            value = report.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        for nested_key in ("coverage", "coverage_report"):
            nested = report.get(nested_key)
            if isinstance(nested, dict) and isinstance(nested.get("overall"), (int, float)):
                return float(nested["overall"])
    return None


def _graph_coverage(graph_payload: dict[str, Any], *, source_length: int) -> float | None:
    if source_length <= 0:
        return None
    intervals: list[tuple[int, int]] = []
    for node in graph_payload.get("nodes", []):
        if not isinstance(node, dict):
            continue
        pointers = list(node.get("total_content_pointers", []) or [])
        for mention in node.get("mentions", []) or []:
            if isinstance(mention, dict):
                pointers.extend(mention.get("spans", []) or [])
        for pointer in pointers:
            if not isinstance(pointer, dict):
                continue
            start = pointer.get("start_char")
            end = pointer.get("end_char")
            if isinstance(start, int) and isinstance(end, int) and end >= start:
                intervals.append((max(0, start), min(source_length - 1, end)))
    if not intervals:
        return None
    intervals.sort()
    covered = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end + 1:
            current_end = max(current_end, end)
        else:
            covered += current_end - current_start + 1
            current_start, current_end = start, end
    covered += current_end - current_start + 1
    return covered / source_length


def run_comparison(*, source: Path, output_dir: Path, version: str) -> dict[str, Any]:
    # Keep the checkout root before adding its test helpers to the import path.
    checkout_root = Path.cwd()
    sys.path.insert(0, str(checkout_root))
    sys.path.insert(0, str(checkout_root / "tests"))

    from _kogwistar_test_helpers import build_workflow_engine_triplet
    from kg_doc_parser.workflow_ingest.models import WorkflowIngestInput
    from kg_doc_parser.workflow_ingest.service import run_ingest_workflow

    text = source.read_text(encoding="utf-8")
    inp = WorkflowIngestInput.from_text(
        document_id="version-comparison-corpus",
        text=text,
        title="Version Comparison Corpus",
    )
    workflow_engine, conversation_engine, knowledge_engine = build_workflow_engine_triplet(
        output_dir / "engines",
        "in_memory",
    )
    started = time.perf_counter()
    run, bundle = run_ingest_workflow(
        inp=inp,
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        deps={"parse_semantic_fn": _fake_semantic_tree},
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    final_state = dict(run.final_state)
    parse_session = final_state.get("parse_session")
    parse_session = parse_session if isinstance(parse_session, dict) else {}
    tree = final_state.get("semantic_tree")
    graph_payload = bundle.graph_payload if bundle is not None else {}
    nodes = graph_payload.get("nodes", []) if isinstance(graph_payload, dict) else []
    strategy_history = parse_session.get("strategy_history", [])
    strategy_history = list(strategy_history) if isinstance(strategy_history, list) else []
    return {
        "version": version,
        "status": run.status,
        "elapsed_ms": elapsed_ms,
        "node_count": len(nodes),
        "max_depth": _max_tree_depth(tree) if tree is not None else None,
        "strategy_history": strategy_history,
        "timeout_count": 0,
        "final_coverage": _coverage(final_state)
        if _coverage(final_state) is not None
        else _graph_coverage(graph_payload, source_length=len(text)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(".tmp_version_comparison"))
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    print(json.dumps(run_comparison(source=args.source, output_dir=args.output_dir, version=args.version), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
