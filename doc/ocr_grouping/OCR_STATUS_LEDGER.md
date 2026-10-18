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
| Persisted Kogwistar runtime integration | PARTIAL | Installed `kogwistar==0.6.12` resumes ordinary interrupted runs from their latest checkpoint. The reviewer-requested recovery-versus-repair simplification review is complete: ordinary interruptions resume the same run; only repair after artifact loss from a succeeded terminal run uses a new persisted root incarnation because Kogwistar correctly does not re-execute terminal runs. `test_ocr_root_resume_replays_checkpointed_page_when_artifact_is_missing` and `test_ocr_workflow_resumes_cached_pages_and_terminal_semantic_run` cover these separate paths and lineage. The Linux recovery release gate passed in parser CI run `39096834736`. No redundant parser retry/resume mechanism was identified, and no additional generation/checkpoint layer is justified. Persistent graph-backend publication across fresh-process recovery remains open and Kogwistar-owned; the 100-page comparison still uses an in-memory knowledge backend and provider interruption, not durable graph storage or OS termination. |
| Artifact/checkpoint crash consistency | PASS (parser gate) | Linux recovery release gate in parser CI run `39096834736` passes the artifact-first, checkpoint-first, grouping-commit-before-child-checkpoint, and semantic-graph-before-checkpoint recovery suite. |
| Grouping and summary durability | PARTIAL | The adaptive bottom-up grouping behavior is implemented, not just exposed as primitives: non-fake parser settings automatically enable the bounded iterative proposer and grounded summarizer; accepted groups enter subsequent frontier rounds, and the proposal context carries ordered modality, page, location, MIME, and available text/summary evidence. `test_adaptive_grouping_iterates_in_kogwistar_and_reuses_completed_tree`, `test_adaptive_grouping_resume_continues_at_uncheckpointed_window`, and `test_default_adaptive_root_wires_grounded_summary_and_semantic_export` cover iterative rounds, resume, default wiring, summary grounding, and semantic/export consumption with deterministic injected providers. Provider fingerprints include non-secret endpoint, reasoning, and fallback configuration. Thus the review's claim that the default path lacks an adaptive assembler is stale as an implementation claim. The remaining work is evidence, not a duplicate grouping implementation: live-provider grouping quality, annotated multimodal completeness (especially figures/non-text), and fresh-process summary recovery against durable graph storage are not established. |
| Semantic hierarchy publication | PARTIAL | `test_custom_grouping_policy_reaches_checkpoint_cache_and_export` and `test_default_adaptive_root_wires_grounded_summary_and_semantic_export` verify static/custom and default adaptive groups and grounded summaries are consumed by semantic parsing and exported as grounded graph records. `test_ocr_root_graph_publication_survives_durable_engine_reopen` now keeps workflow/conversation engines in memory and exercises a real persistent Chroma knowledge engine; it passed locally on pinned Kogwistar `0.6.12` (`1 passed, 46.53s`) and the dedicated durable-publication CI job passed in run `39096834736`. The earlier CI failure (`39091524022`) came from persisting unrelated workflow-design graphs and is superseded. Fresh-process recovery with durable graph publication and independent OCR completeness/recall remain open. Do not add parser-owned checkpoint infrastructure for a Kogwistar durability gap. |
| OCR completeness and positional fidelity | PARTIAL | Coordinate precision and provider loss diagnostics are enforced; `unresolved_item_count` records Docling and direct-provider items without usable provenance, `OCRQualityPolicy(reject_unresolved_items=True)` fails closed, and provider `source_item_count` must reconcile at both the quality and normalized-page boundaries. Invalid `meaningful_ordering` reports missing, unknown, and duplicate cluster IDs, with bounded diagnostics supplied to retry. Multimodal order is tested at normalization, grouping input, and graph projection; non-text graph projection and provider-backed multimodal fixtures remain open. The typed `OCRGroundTruth` evaluator independently measures item recall, spatial IoU, and text mismatch. Its Codex table fixture now includes the exact 1600x1200 source page and a SHA-256/dimension assertion, making the 16/16 region annotations auditable in clean CI; it remains a controlled text/table fixture, not figure-recognition evidence. These checks do not establish independent OCR recall for visual regions or guarantee durable graph acceptance. An audit found no provider-emitted figure/non-text objects in available live response artifacts, so annotated figure/non-text corpora remain open. |
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
