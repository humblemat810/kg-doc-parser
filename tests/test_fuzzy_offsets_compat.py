from __future__ import annotations

from kogwistar.utils.fuzzy_offsets import FuzzySpanHit, find_best_fuzzy_span


def test_fuzzy_offsets_import_and_match():
    hit = find_best_fuzzy_span(
        content="0123 Proof step one 7890",
        excerpt="Proof-step one",
        origin_start=5,
        candidate_filter=lambda candidate: candidate != "0123 Proof-step one 7890",
    )

    assert isinstance(hit, FuzzySpanHit)
    assert hit.start == 5
    assert hit.end == 19
    assert hit.score > 0
