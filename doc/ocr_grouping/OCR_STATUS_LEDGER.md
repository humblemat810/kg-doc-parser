# OCR Implementation Status Ledger

This is the only live status checklist for the OCR runtime and seam work.
Update existing rows in place. Do not append dated progress sections or create
new checklist items for individual test runs.

Historical implementation evidence is represented by Git:

```text
git log --oneline --follow -- doc/ocr_grouping/OCR_STATUS_LEDGER.md
git show <commit> -- doc/ocr_grouping/OCR_STATUS_LEDGER.md
```

## Current Status

| Area | Status | Release-blocking remainder |
| --- | --- | --- |
| Persisted Kogwistar runtime integration | PARTIAL | Parser CI run `39015963660` (#492) on `4f54000` verifies published-core fresh-process lineage, duplicate resume, single graph publication, the deterministic 100-page PDF acceptance fixture, and the Linux recovery gate. Persistent graph-backend publication across restart remains a Kogwistar backend-specific gate, so it is not duplicated here. |
| Artifact/checkpoint crash consistency | PASS (parser gate) | Parser CI run `39011917164` passes artifact-first, checkpoint-first, grouping-commit-before-child-checkpoint, and semantic-graph-before-checkpoint recovery tests. |
| Grouping and summary durability | PARTIAL | Monotonic frontier fencing, large-frontier identity stability, bounded linear frontier access, summary journaling, and all parser writer call sites are covered. Long-run size-growth benchmarking and durable graph-backend publication remain separate gates. |
| Semantic hierarchy publication | PARTIAL | Accepted grouping is consumed by the semantic callback and preserved in graph export. Fresh-process publication against a durable graph backend remains open and is Kogwistar-owned. |
| OCR completeness and positional fidelity | PARTIAL | Coordinate precision and provider loss diagnostics are enforced; `unresolved_item_count` now records Docling items without usable provenance and `OCRQualityPolicy(reject_unresolved_items=True)` fails closed. Independent completeness for non-text regions, tables, and figures remains open. |
| Typing and supported interpreters | PASS (source matrix) | Production-package Pyright reports 0 errors, 0 warnings, and parser CI run `39011917164` passed CPython 3.12-3.14, PyPy 3.11, and the Linux recovery gate on exact commit `e57fd43`. PyPI installation under PyPy remains unavailable while published Kogwistar `0.6.12` declares Python >=3.12; this is a dependency-release gate, not a parser source failure. |
| Release | OPEN | Do not bump or publish until the release-blocking rows have evidence. |

## Evidence Standard

For every row promoted to `PASS`, record the implementation path, exact test
node, before/after result, runtime type, and CI evidence in the commit that
changes this ledger. A local adapter test cannot close a persisted-runtime
release gate.

## Source Documents

- [Runtime and seam requirements](SEAM_AND_RUNTIME_IMPLEMENTATION_CHECKLIST.md)
- [Review findings and acceptance criteria](REVIEW_FIX_CHECKLIST.md)
- [Broad OCR audit](kg_doc_parser_ocr_broad_audit_2026-10-15.md)

The source documents contain the fixed requirements and historical context;
their status prose is not a second live checklist. Use this file for the
current answer to "what remains?".
