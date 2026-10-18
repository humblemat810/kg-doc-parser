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
| Persisted Kogwistar runtime integration | PASS locally and CI | Installed `kogwistar==0.6.12` resumes ordinary interruptions on the same run; recovery-versus-repair review confirms only repair after artifact loss from terminal success needs a new root incarnation. `test_ocr_root_fresh_process_resume_after_page_63` passes locally (`1 passed, 1293.43s`) and in Parser CI run `39135756441`: two real process terminations, fresh-process recovery, persistent Chroma reopen, v1 canonical tree/graph comparison, and a changed-source revision that reprocesses all 100 pages and verifies changed page-40 OCR against the artifact matching its rendered image digest while preserving the v1 artifact. OCR/grouping responses are deterministic fixtures, not live accuracy evidence. No redundant parser retry/resume mechanism was identified; no additional checkpoint layer is justified. |
| Artifact/checkpoint crash consistency | PASS (parser gate) | Linux recovery release gate in Parser CI run `39135756441` passes artifact-first, checkpoint-first, grouping-commit-before-child-checkpoint, and semantic-graph-before-checkpoint recovery cases. |
| Adaptive grouping and summary durability | IMPLEMENTED; live-provider quality PARTIAL | The default non-fake path constructs a structured proposer and grounded summarizer. The Kogwistar workflow performs bounded iterative bottom-up grouping over ordered frontier windows; accepted groups feed later rounds. Proposal context carries modality, page, location, MIME, and available OCR text/summaries; prompts prohibit claims about unseen media. Deterministic-provider tests cover iteration, resume, default wiring, grounding, and semantic/export consumption. The updated 100-page local test verifies summary caps and accepted-summary reuse over two fresh-process interruptions. The review's "generic primitives only" assessment is stale for this branch; live-provider grouping quality and visual understanding beyond OCR evidence remain unverified. Do not duplicate grouping or Kogwistar recovery mechanisms. |
| Semantic hierarchy publication | PASS locally and CI | Static/custom and default adaptive grouping plus grounded summaries are consumed by semantic parsing and exported as grounded graph records. The 100-page gate compares recovered grouping-tree and canonical graph payload digests/IDs to an uninterrupted persistent reference after two process terminations; both executions use identical source PDF bytes, durable Chroma is reopened, and exact graph records are verified. Its OCR responses are deterministic synthetic fixtures, not live OCR. Parser CI run `39135756441` passed the Linux recovery and durable graph publication gates. Keep recovery coordinated with Kogwistar; do not add parser-owned checkpoint infrastructure. |
| OCR completeness and positional fidelity | PARTIAL (independent evidence) | Coordinate precision and provider loss diagnostics are enforced; `unresolved_item_count` records items without usable provenance, strict quality policy fails closed, and source-item accounting is reconciled at both quality and normalized-page boundaries. Invalid `meaningful_ordering` reports missing, unknown, and duplicate cluster IDs with bounded retry diagnostics. The typed `OCRGroundTruth` evaluator measures item recall by modality, spatial IoU, and text mismatch; unannotated modalities remain `None`. It rejects invalid boxes and disjoint matches (`test_ground_truth_evaluator_rejects_invalid_provider_boxes`, `test_zero_iou_threshold_does_not_match_spatially_disjoint_items`). The independently annotated 1600x1200 Codex table fixture now has 16 text items plus one table-region item. Live fixture results: Codex 16/17 total, 16/16 text, 0/1 non-text, mean IoU 0.6152; direct RapidOCR 15/17, 15/16 text, 0/1 non-text, mean IoU 0.7147; Docling+RapidOCR 5/17, 4/16 text, 1/1 table-region, mean IoU 0.8244, but misses 12 table-cell texts. Provider suites pass (`42 passed, 1 skipped`). Direct RapidOCR now materializes NumPy-style iterables, counts unmatched detection arrays as unresolved, and Docling normalizes bottom-left coordinates and binds markdown export to its document. This one-page table evaluation does not establish figure/corpus-wide completeness; live Codex figure output still had zero non-text objects, so C06 remains open. |
| Typing and supported interpreters | COMMITTED BASE PASS; CURRENT DELTA PENDING | The last committed code passed full-package Pyright (`0 errors, 0 warnings, 0 informations`) and Parser CI run `39135756441` across CPython 3.12-3.14 and PyPy 3.11. This uncommitted summary-cache delta passes Ruff, `git diff --check`, and nine focused summary tests; rerunning package-wide Pyright is currently blocked by the Windows Node launcher `EPERM` while resolving `C:\Users\chanh`. The delta still needs CI verification. |
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
