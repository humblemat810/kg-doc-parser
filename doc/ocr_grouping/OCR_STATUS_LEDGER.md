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
| OCR completeness and positional fidelity | PARTIAL (independent evidence) | Coordinate precision, provider-loss diagnostics, strict rejection, and source-item reconciliation are enforced. On the independently annotated table page (16 text plus one table region), Codex matched 16/17 text and missed the region; direct RapidOCR matched 15/17 (15/16 text) but missed the `8` cell and region; current-adapter Docling+RapidOCR matched 5/17 (4/16 text, one transcription mismatch, 1/1 region). On the chart page, direct RapidOCR matched 11/12 (11/11 text, 0/1 region); direct EasyOCR emitted nine detections but only two matched annotations at IoU 0.5 (2/11 text, 0/1 region); Docling+RapidOCR matched 2/12 (1/11 text, 1/1 region). Exact images, annotations, and provider responses are offline regressions. Two controlled pages strengthen modality evidence but do not establish corpus-wide completeness. |
| Typing and supported interpreters | PASS ON `b50b0fe` | Parser CI #563 (`39147218276`) passed Linux recovery, durable graph publication, and the full CPython 3.12, 3.13, 3.14, and PyPy 3.11 matrix. The OCR-ground-truth slice also passes 15 focused tests, targeted Pyright (0 errors), Ruff, and `git diff --check`. |
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
