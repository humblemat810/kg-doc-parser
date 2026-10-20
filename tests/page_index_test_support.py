"""Shared deterministic PageIndex fixture helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from kg_doc_parser.workflow_ingest.semantics import SemanticNode

NodeSignature: TypeAlias = tuple[str, str, tuple["NodeSignature", ...]]  # noqa: UP040 - PyPy 3.11 support


def fixture_text(name: str) -> str:
    return (Path(__file__).parent / "fixtures" / "page_index" / name).read_text(
        encoding="utf-8"
    )


def normalized_node_signature(node: SemanticNode) -> NodeSignature:
    node_type = "HEADING" if node.node_type in {"SECTION", "SUBSECTION"} else node.node_type
    return (
        node_type,
        node.title,
        tuple(normalized_node_signature(child) for child in node.child_nodes),
    )
