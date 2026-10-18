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
| Persisted Kogwistar runtime integration | PASS locally and CI | Installed `kogwistar==0.6.12` resumes ordinary interruptions on the same run; recovery-versus-repair review confirms only repair after artifact loss from terminal success needs a new root incarnation. `test_ocr_root_fresh_process_resume_after_page_63` passed locally in 1093.68s with two real process terminations, same-run fresh-process resumes, persistent Chroma reopen, accepted OCR/summary reuse, semantic publication, and canonical tree/graph equality against an uninterrupted reference using identical PDF bytes/source revision. The rendered 100-page fixture contains text, a chart, a 15-row table, duplicate paragraphs, and a page-62/63 continuation. Deterministic OCR payloads validate recovery, not live provider accuracy. Parser CI run `39114468766` passed CPython 3.12-3.14, PyPy 3.11, Linux recovery, and durable graph publication on fixture commit `8ca0d2e`. No redundant parser retry/resume mechanism was identified; no additional checkpoint layer is justified. |
| Artifact/checkpoint crash consistency | PASS (parser gate) | Linux recovery release gate in parser CI run `39096834736` passes the artifact-first, checkpoint-first, grouping-commit-before-child-checkpoint, and semantic-graph-before-checkpoint recovery suite. |
| Adaptive grouping and summary durability | IMPLEMENTED; live-provider quality PARTIAL | The default non-fake path constructs a structured proposer and grounded summarizer. The Kogwistar workflow performs bounded iterative bottom-up grouping over ordered frontier windows; accepted groups feed later rounds. Proposal context carries modality, page, location, MIME, and available OCR text/summaries; prompts prohibit claims about unseen media. Deterministic-provider tests cover iteration, resume, default wiring, grounding, and semantic/export consumption. The updated 100-page local test verifies summary caps and accepted-summary reuse over two fresh-process interruptions. The review's "generic primitives only" assessment is stale for this branch; live-provider grouping quality and visual understanding beyond OCR evidence remain unverified. Do not duplicate grouping or Kogwistar recovery mechanisms. |
| Semantic hierarchy publication | PASS locally and CI | Static/custom and default adaptive grouping plus grounded summaries are consumed by semantic parsing and exported as grounded graph records. The updated 100-page gate compares recovered grouping-tree and canonical graph payload digests/IDs to an uninterrupted persistent reference after two process terminations; both executions use identical source PDF bytes, durable Chroma is reopened, and exact graph records are verified. Its OCR responses are deterministic synthetic fixtures, not live OCR. Parser CI run `39114468766` passed the Linux recovery and durable graph publication gates. Keep recovery coordinated with Kogwistar; do not add parser-owned checkpoint infrastructure. |
| OCR completeness and positional fidelity | PARTIAL (independent evidence) | Coordinate precision and provider loss diagnostics are enforced; `unresolved_item_count` records Docling and direct-provider items without usable provenance, `OCRQualityPolicy(reject_unresolved_items=True)` fails closed, and provider `source_item_count` must reconcile at both the quality and normalized-page boundaries. Invalid `meaningful_ordering` reports missing, unknown, and duplicate cluster IDs, with bounded diagnostics supplied to retry. The typed `OCRGroundTruth` evaluator independently measures aggregate and per-modality item recall, spatial IoU, and text mismatch; a modality with no annotations is `None` rather than a misleading 1.0. It rejects non-finite/zero-area provider boxes and refuses to count disjoint boxes as matches when `minimum_iou=0` (`test_ground_truth_evaluator_rejects_invalid_provider_boxes`, `test_zero_iou_threshold_does_not_match_spatially_disjoint_items`). `test_ground_truth_report_exposes_missing_non_text_recall_separately` demonstrates text recall of 1.0 while a missing non-text region has recall 0.0. Graph export preserves accepted media evidence as ordered Kogwistar multimodal spans with digest/revision and spatial or temporal locators; the span takes an explicit host `source_namespace`, and media export fails closed when it is missing (`test_multimodal_projection_requires_explicit_graph_namespace`). The auditable Codex table fixture includes the exact 1600x1200 source page and SHA-256/dimension assertions for 16 annotated text/table regions; this is controlled fixture evidence, not independent provider evidence for figure/non-text recall. The recorded live Codex figure result contains zero provider-emitted non-text objects. Independent annotated visual-region/provider completeness remains open. |
| Typing and supported interpreters | Last pushed revision PASS; local delta PASS | Parser CI run `39119735435` passed CPython 3.12-3.14, PyPy 3.11, Linux recovery, and durable graph publication for `fb47167`. The current local namespace delta passes Ruff, Pyright (0 diagnostics), and all 63 CI-marked tests across document adapters and workflow contracts; it still needs a CI run after a code-bearing push. |
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
