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
| Persisted Kogwistar runtime integration and page scaling | Runtime/recovery PASS locally and CI; deterministic scaling PASS locally; full graph-backed scale run unverified | Installed `kogwistar==0.6.12` resumes ordinary interruptions on the same run; recovery-versus-repair review confirms only repair after artifact loss from terminal success needs a new root incarnation. `test_ocr_root_fresh_process_resume_after_page_63` passes locally (`1 passed, 1293.43s`) and in Parser CI run `39135756441`: two real process terminations, fresh-process recovery, persistent Chroma reopen, v1 canonical tree/graph comparison, and a changed-source revision that reprocesses all 100 pages and verifies changed page-40 OCR against the artifact matching its rendered image digest while preserving the v1 artifact. The page loop now keeps its immutable source manifest outside repeated checkpoints, writes only cursor/reuse deltas, avoids per-page accepted-prefix JSON/progress materialization, and reuses the one validated terminal snapshot. SQLite remains the live page-status source; the detailed JSON progress mirror is published once after the page loop, while per-page progress remains in logs. Reuse reconstruction reads all matching Kogwistar step records (not the default 200) and uses the latest per-page outcome, covered by `test_reused_page_trace_query_is_not_truncated_at_default_read_limit`. `test_ocr_scaling_artifact_reads_and_checkpoint_state_bytes_are_linear` (100/500/1000) measures exact parser-state/artifact payload behavior: old prefix decodes 5,050/125,250/500,500 vs 100/500/1,000 after; parser-state snapshot payload totals 1,900,879/47,285,279/189,068,781 bytes vs 35,092/175,892/352,893 bytes. This models JSON state payload bytes and deterministic artifact reads, not the full Kogwistar database envelope or a graph-backed 1000-page run. The actual full-runtime scaling harness is marked slow and was not completed locally. OCR/grouping responses are deterministic fixtures, not live accuracy evidence. No redundant parser retry/resume mechanism was identified; no additional checkpoint layer is justified. |
| Artifact/checkpoint crash consistency | PASS (parser gate) | Linux recovery release gate in Parser CI run `39135756441` passes artifact-first, checkpoint-first, grouping-commit-before-child-checkpoint, and semantic-graph-before-checkpoint recovery cases. |
| Adaptive grouping and summary durability | IMPLEMENTED; live-provider quality PARTIAL | The default non-fake path constructs a structured proposer and grounded summarizer. The Kogwistar workflow performs bounded iterative bottom-up grouping over ordered frontier windows; accepted groups feed later rounds. Proposal context carries modality, page, location, MIME, and available OCR text/summaries; prompts prohibit claims about unseen media. Deterministic-provider tests cover iteration, resume, default wiring, grounding, and semantic/export consumption. Tree/frontier/metadata inserts are centralized in the fenced `save_document_tree()` transaction; source-revision invalidation deletes the companion rows together. Tree loads and identical-tree replays validate tree/frontier/metadata identity, including accepted-page frontier against completed OCR rows. `test_grouping_checkpoint_concurrent_updates_accept_one_revision`, `test_grouping_checkpoint_load_rejects_mismatched_companion_records`, and `test_grouping_checkpoint_rejects_stale_writer_after_source_revision_changes` cover competing revisions, tampered companions, and stale source writes. A separate monotonic counter is not needed as the write guard because expected tree revisions are the CAS authority. CI for these latest additions is pending; live-provider grouping quality remains open. Do not duplicate Kogwistar recovery mechanisms. |
| Semantic hierarchy publication | PASS locally and CI | Static/custom and default adaptive grouping plus grounded summaries are consumed by semantic parsing and exported as grounded graph records. The 100-page gate compares recovered grouping-tree and canonical graph payload digests/IDs to an uninterrupted persistent reference after two process terminations; both executions use identical source PDF bytes, durable Chroma is reopened, and exact graph records are verified. Its OCR responses are deterministic synthetic fixtures, not live OCR. Parser CI run `39135756441` passed the Linux recovery and durable graph publication gates. Keep recovery coordinated with Kogwistar; do not add parser-owned checkpoint infrastructure. |
| OCR completeness and positional fidelity | PARTIAL (independent evidence) | Coordinate precision, provider-loss diagnostics, strict rejection, and source-item reconciliation are enforced. Quality metadata is checked at legacy completion, direct publication, completed-page reuse, orphan adoption, and SQLite rebuild from artifacts. `test_legacy_ocr_completion_rejects_quality_rejected_artifact`, `test_direct_ocr_publication_rejects_quality_rejected_artifact`, `test_direct_ocr_publication_commits_the_validated_payload_snapshot`, and `test_artifact_rebuild_does_not_resurrect_quality_rejected_ocr` cover those paths, including stale completed-row reuse and mutation after validation. These and the full 29-test OCR state-store selection pass locally. Parser CI run `39160013729` passed the full CPython 3.12-3.14 and PyPy 3.11 matrix plus Linux recovery and durable graph gates for the quality-metadata changes. Production Pyright reported 0 errors and Ruff passes. The fresh-process strict-rejection retry test also passes locally. On the independently annotated table page (16 text plus one table region), Codex matched 16/17 text and missed the region; direct RapidOCR matched 15/17 (15/16 text) but missed the `8` cell and region; current-adapter Docling+RapidOCR matched 5/17 (4/16 text, one transcription mismatch, 1/1 region). On the chart page, direct RapidOCR matched 11/12 (11/11 text, 0/1 region); direct EasyOCR emitted nine detections but only two matched annotations at IoU 0.5 (2/11 text, 0/1 region); Docling+RapidOCR matched 2/12 (1/11 text, 1/1 region). Exact images, annotations, and provider responses are offline regressions. Two controlled pages strengthen modality evidence but do not establish corpus-wide completeness. |
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
