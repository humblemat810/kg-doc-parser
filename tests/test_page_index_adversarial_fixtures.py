from __future__ import annotations

from pathlib import Path

import pytest
from kg_doc_parser.workflow_ingest.page_index import parse_page_index_document

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "page_index" / "adversarial"


def _walk(node):
    yield node
    for child in node.child_nodes:
        yield from _walk(child)


@pytest.mark.workflow
@pytest.mark.parametrize("fixture_path", sorted(FIXTURE_DIR.glob("*.md")), ids=lambda p: p.stem)
def test_adversarial_markdown_fixtures_remain_losslessly_grounded(fixture_path: Path) -> None:
    raw_text = fixture_path.read_text(encoding="utf-8")
    result = parse_page_index_document(
        document_id=f"adversarial-{fixture_path.stem}",
        title=fixture_path.stem,
        raw_text=raw_text,
        source_format="markdown",
        mode="heuristic",
        summary_enabled=False,
    )

    assert result.coverage["overall"] == pytest.approx(1.0)
    assert not result.diagnostics.get("validation_errors")
    assert result.semantic_tree.child_nodes

    for node in _walk(result.semantic_tree):
        for pointer in node.total_content_pointers:
            source = result.authoritative_source_map[pointer.source_cluster_id].text
            end_char = len(source) if pointer.end_char < 0 else pointer.end_char + 1
            assert 0 <= pointer.start_char <= end_char <= len(source)
            assert source[pointer.start_char:end_char] == pointer.verbatim_text

