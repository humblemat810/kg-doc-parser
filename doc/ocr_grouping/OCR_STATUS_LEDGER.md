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
| Persisted Kogwistar runtime integration | PARTIAL | Verify published-core fresh-process lineage, duplicate resume, and single graph publication. |
| Artifact/checkpoint crash consistency | PARTIAL | Reconcile artifact-first and checkpoint-first crashes across SQLite and runtime checkpoints. |
| Grouping and summary durability | PARTIAL | Prove monotonic frontier/generation behavior and bounded long-document work. |
| Semantic hierarchy publication | PARTIAL | Prove accepted grouping is consumed and preserved by semantic parsing and graph export. |
| OCR completeness and positional fidelity | PARTIAL | Coordinate precision and provider loss diagnostics are enforced; `unresolved_item_count` now records Docling items without usable provenance and `OCRQualityPolicy(reject_unresolved_items=True)` fails closed. Independent completeness for non-text regions, tables, and figures remains open. |
| Typing and supported interpreters | PARTIAL | Keep the explicit Pyright CPython 3.12 gate and verify the full CPython 3.12-3.14/PyPy 3.11 matrix. |
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
