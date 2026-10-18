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
| Persisted Kogwistar runtime integration | PARTIAL (remote verification) | Installed `kogwistar==0.6.12` resumes ordinary interruptions on the same run; recovery-versus-repair review confirms only repair after artifact loss from terminal success needs a new root incarnation. `test_ocr_root_fresh_process_resume_after_page_63` now passes locally with two real process terminations, fresh-process resumes, a persistent Chroma backend, grounded summary reuse, semantic publication, and an uninterrupted 100-page reference. The recovered and reference tree digests and canonical graph payload digests/IDs match; accepted OCR and summary calls are not replayed. Remote CI for this expanded gate is pending. No redundant parser retry/resume mechanism was identified; no additional checkpoint layer is justified. |
| Artifact/checkpoint crash consistency | PASS (parser gate) | Linux recovery release gate in parser CI run `39096834736` passes the artifact-first, checkpoint-first, grouping-commit-before-child-checkpoint, and semantic-graph-before-checkpoint recovery suite. |
| Adaptive grouping and summary durability | PARTIAL (live-provider quality) | The default non-fake path constructs a structured proposer and grounded summarizer. The Kogwistar workflow performs bounded iterative bottom-up grouping over ordered frontier windows; accepted groups feed later rounds. Proposal context carries modality, page, location, MIME, and available source text/summaries; prompts forbid claims about unseen media. Deterministic-provider tests cover iteration, resume, default wiring, grounding, and semantic/export consumption. The 100-page dual-interruption test now verifies summary caps survive resume and accepted summaries are not regenerated. This validates workflow behavior, not live-provider grouping quality or visual understanding of media absent from OCR evidence. Do not duplicate grouping or Kogwistar recovery mechanisms. |
| Semantic hierarchy publication | PARTIAL (remote verification) | Static/custom and default adaptive grouping plus grounded summaries are consumed by semantic parsing and exported as grounded graph records. The expanded 100-page local gate compares the recovered grouping-tree digest and canonical graph payload digest/IDs to an uninterrupted persistent reference after two process terminations; durable Chroma is reopened and exact records verified. Remote CI for this expanded comparison is pending. Keep recovery coordinated with Kogwistar; do not add parser-owned checkpoint infrastructure. |
| OCR completeness and positional fidelity | PARTIAL (independent evidence) | Coordinate precision and provider loss diagnostics are enforced; `unresolved_item_count` records Docling and direct-provider items without usable provenance, `OCRQualityPolicy(reject_unresolved_items=True)` fails closed, and provider `source_item_count` must reconcile at both the quality and normalized-page boundaries. Invalid `meaningful_ordering` reports missing, unknown, and duplicate cluster IDs, with bounded diagnostics supplied to retry. The typed `OCRGroundTruth` evaluator independently measures aggregate and per-modality item recall, spatial IoU, and text mismatch; a modality with no annotations is `None` rather than a misleading 1.0. `test_ground_truth_report_exposes_missing_non_text_recall_separately` demonstrates text recall of 1.0 while a missing non-text region has recall 0.0. Its auditable Codex table fixture includes the exact 1600x1200 source page and SHA-256/dimension assertions for 16 annotated text/table regions; this is controlled fixture evidence, not independent provider evidence for figure/non-text recall. The recorded live Codex figure result contains zero provider-emitted non-text objects. Independent annotated visual-region/provider completeness remains open. |
| Typing and supported interpreters | PASS | Production-package Pyright reports 0 errors, 0 warnings. Parser CI run `39085988324` passed commit `d693e06` on CPython 3.12-3.14 and PyPy 3.11, including the Linux recovery gate; `test_ocr_checkpoint_repair_contract_requires_cursor` now guards the required `page_cursor` access contract. |
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
