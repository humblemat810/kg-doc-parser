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
| Persisted Kogwistar runtime integration | PARTIAL | Parser CI run `39034957448` completed successfully on commit `7afb56c`. The installed Kogwistar `0.6.12` local acceptance run passed `test_ocr_root_resumes_after_page_63_and_matches_uninterrupted_reference` in 850.22s, comparing uninterrupted and interrupted/resumed canonical graph and grouping payloads with accepted-page reuse. Persistent graph-backend publication across restart remains a Kogwistar backend-specific gate, so it is not duplicated here; this parser test uses the in-memory knowledge backend and injects a provider interruption rather than an OS kill. |
| Artifact/checkpoint crash consistency | PASS (parser gate) | Parser CI run `39011917164` passes artifact-first, checkpoint-first, grouping-commit-before-child-checkpoint, and semantic-graph-before-checkpoint recovery tests. |
| Grouping and summary durability | PARTIAL | Monotonic frontier fencing, large-frontier identity stability, bounded linear frontier access, summary journaling, all parser writer call sites, document/source-revision-namespaced summary fingerprints, and the 100-page interrupted/resumed comparison are covered. Durable graph-backend publication remains a separate Kogwistar gate. |
| Semantic hierarchy publication | PARTIAL | Accepted grouping is consumed by the semantic callback and preserved in graph export. Fresh-process publication against a durable graph backend remains open and is Kogwistar-owned. |
| OCR completeness and positional fidelity | PARTIAL | Coordinate precision and provider loss diagnostics are enforced; `unresolved_item_count` records Docling items without usable provenance, `OCRQualityPolicy(reject_unresolved_items=True)` fails closed, and provider `source_item_count` must reconcile at both the quality and normalized-page boundaries. Render-manifest identity and duplicate input-unit rejection are CI-verified by Parser run `39034957448`. Independent completeness for non-text regions, tables, and figures remains open. |
| Typing and supported interpreters | PASS (source matrix) | Production-package Pyright reports 0 errors, 0 warnings. Parser CI run `39034957448` completed successfully on commit `7afb56c`; the supported CPython 3.12-3.14 and PyPy 3.11 matrix passed. |
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
