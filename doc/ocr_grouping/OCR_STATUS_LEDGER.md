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
| Persisted Kogwistar runtime integration | PARTIAL | Installed `kogwistar==0.6.12` resumes ordinary interrupted runs from their latest checkpoint. The recovery-versus-repair review is complete: ordinary interruptions resume the same run; only repair after artifact loss from a succeeded terminal run uses a new persisted root incarnation because Kogwistar correctly does not re-execute terminal runs. `test_ocr_root_fresh_process_resume_after_page_63` proves real process termination after page 63, fresh-process same-run resume, and exact 100-page Chroma node/edge identity after reopen; Parser CI run `39100319567` passed this gate and the supported interpreter matrix. `test_ocr_root_fresh_process_resume_publishes_durable_graph` now compares the recovered two-page canonical graph and grouping tree with an uninterrupted persistent reference; local test passed, new CI run pending. A single test combining durable 100-page uninterrupted-equivalence with SIGKILL and multiple interruptions remains open. No redundant parser retry/resume mechanism was identified, and no additional generation/checkpoint layer is justified. |
| Artifact/checkpoint crash consistency | PASS (parser gate) | Linux recovery release gate in parser CI run `39096834736` passes the artifact-first, checkpoint-first, grouping-commit-before-child-checkpoint, and semantic-graph-before-checkpoint recovery suite. |
| Grouping and summary durability | PARTIAL | The review's claim that the default path lacks adaptive assembly is stale as an implementation claim: non-fake parser settings automatically enable bounded iterative grouping and grounded summaries; accepted groups enter later frontier rounds. Proposal context carries ordered modality, page, location, MIME, and available text/summary evidence. `test_adaptive_grouping_iterates_in_kogwistar_and_reuses_completed_tree`, `test_adaptive_grouping_resume_continues_at_uncheckpointed_window`, and `test_default_adaptive_root_wires_grounded_summary_and_semantic_export` cover iteration, resume, default wiring, grounding, and semantic/export consumption with deterministic injected providers. `test_ocr_summary_is_not_regenerated_after_fresh_process_resume` proves a grounded summary survives process termination before semantic publication and is not regenerated on fresh-process recovery; the recovered graph is reopened from Chroma and compared by exact IDs. The summary and durable graph gates passed in Parser CI run `39100319567`. This establishes workflow behavior, not live-provider grouping quality or visual understanding of unseen figures. Do not duplicate the existing grouping or Kogwistar recovery mechanisms. |
| Semantic hierarchy publication | PARTIAL | `test_custom_grouping_policy_reaches_checkpoint_cache_and_export` and `test_default_adaptive_root_wires_grounded_summary_and_semantic_export` verify static/custom and default adaptive groups and grounded summaries are consumed by semantic parsing and exported as grounded graph records. `test_ocr_root_graph_publication_survives_durable_engine_reopen` verifies durable Chroma publication on a completed run. `test_ocr_root_fresh_process_resume_after_page_63` terminates a 100-page worker after page 63, resumes the same Kogwistar root in a fresh process, and compares unique exact graph node/edge IDs after reopening Chroma; Parser CI run `39100319567` passed on commit `27981f9`. The two-page durable recovery test additionally verifies canonical graph and grouping equivalence to an uninterrupted persistent reference, locally passed with CI pending. The combined 100-page equivalence gate remains open. Keep recovery coordinated with Kogwistar; do not add parser-owned checkpoint infrastructure. |
| OCR completeness and positional fidelity | PARTIAL | Coordinate precision and provider loss diagnostics are enforced; `unresolved_item_count` records Docling and direct-provider items without usable provenance, `OCRQualityPolicy(reject_unresolved_items=True)` fails closed, and provider `source_item_count` must reconcile at both the quality and normalized-page boundaries. Invalid `meaningful_ordering` reports missing, unknown, and duplicate cluster IDs, with bounded diagnostics supplied to retry. The typed `OCRGroundTruth` evaluator independently measures aggregate and per-modality item recall, spatial IoU, and text mismatch; a modality with no annotations is `None` rather than a misleading 1.0. `test_ground_truth_report_exposes_missing_non_text_recall_separately` demonstrates text recall of 1.0 while a missing non-text region has recall 0.0. Its auditable Codex table fixture includes the exact 1600x1200 source page and SHA-256/dimension assertions for 16 annotated text/table regions; this is not independent provider evidence for figure/non-text recall. The recorded live Codex figure result contains zero provider-emitted non-text objects. Independent annotated visual-region completeness remains open. |
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
