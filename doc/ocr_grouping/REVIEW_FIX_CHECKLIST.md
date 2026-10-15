# OCR Review Fix Checklist

This checklist tracks the six findings from the OCR provider abstraction review.
`PASS` requires code, a focused regression test, and evidence from the actual
runtime boundary. `PARTIAL` means containment is improved but recovery,
reconciliation, or end-to-end evidence is still missing. A checklist mark is
not a substitute for a test ID and runtime result.

The 2026-10-15 audit corrected two previously over-optimistic S02 claims:
atomic JSON publication and stored artifact digests are not release-complete
until their reuse and crash-window tests execute against the real persistence
boundary. The exact reuse-path test now covers both a well-formed tampered
artifact and a fresh-process provider/configuration change. S02 remains
PARTIAL because the two-store crash protocol is not yet proven.

The audit also rejects treating the following as release proof: a digest field
that is stored but not checked by the exact completed-page reuse path; an atomic
file replacement without a two-store reconciliation protocol; a same-process
retry without reopening the persisted parent invocation; a root/child function
call that is not represented in the persisted Kogwistar execution graph; or a
grouping checkpoint write without monotonic generation/CAS ownership. These
remain `PARTIAL` or `OPEN` below until a reusable regression test demonstrates
the actual boundary.

Evidence must be recorded as a ledger, not inferred from a green CI job. Each
item records the requirement ID, implementation file/function, exact regression
test node, the pre-fix failure or missing behavior, the post-fix assertion,
runtime classification (standalone, mocked runtime, or persisted Kogwistar),
and current status. `PASS` is reserved for the complete acceptance gate;
passing a narrow subtest remains `PARTIAL` when crash, restart, concurrency, or
provider evidence is absent.

## Latest audit corrections

These S02 subclaims are intentionally tracked separately. A green file-write
test must not be counted as a green two-store commit protocol:

| Subclaim | Status | Required evidence before release |
| --- | --- | --- |
| Provider/configuration fingerprint is part of page identity | PARTIAL | Fresh-process reuse rejection for every output-affecting provider, model, engine, and policy field, plus a persisted reopen assertion. |
| OCR artifact SHA-256 is recorded | PASS (implementation) | Artifact metadata contains the digest after validated publication; this alone does not prove reuse safety. |
| OCR artifact SHA-256 is enforced on completed-page reuse | PASS (focused boundary) | `test_ocr_state_rejects_stale_provider_config_or_tampered_artifact` rejects a well-formed but modified JSON artifact through the actual reuse predicate. |
| JSON publication uses a unique temporary file and atomic replacement | PASS (file boundary) | `test_atomic_json_publication_uses_distinct_temporary_files_for_concurrent_writers` proves distinct same-directory temporary files and replacement. It does not prove a two-store commit. |
| Artifact and SQLite/Kogwistar checkpoint commits are crash-consistent | PARTIAL | `test_ocr_page_artifact_publish_crash_is_recoverable` injects a crash in the real `_process_ocr_page` path after atomic rename and before the SQLite completion update, then reopens SQLite and adopts only the source/config-matching artifact. The inverse checkpoint-before-artifact order and full Kogwistar checkpoint reconciliation remain open. |
| Grouping checkpoint writes are fenced against late writers | PARTIAL | Expected revision/absence CAS and retryable conflict propagation are covered. Resume now also rejects a persisted frontier that disagrees with the restored tree (`tests/test_workflow_ingest_ocr_pipeline.py::test_grouping_resume_rejects_frontier_that_disagrees_with_tree`); compatibility writers, monotonic generation, and crash/restart ordering remain open. |

The realistic recovery guarantee is at-least-once provider execution with
exactly one accepted canonical result. The implementation must not claim
exactly-once OCR or LLM calls across arbitrary process termination.

| Finding | Status | Evidence / remaining work |
| --- | --- | --- |
| P0 grouping checkpoint overwrite on resume | PARTIAL | Prefix pages no longer overwrite `document_tree_state`; grouping and summary persistence now atomically fence both expected revisions and expected absence. The preparation path still has a baseline SQLite compatibility path outside the grouping child, and crash recovery remains open. |
| P0 OCR and grouping are persisted Kogwistar workflow nodes | PARTIAL | A versioned root workflow now invokes persisted page, grouping, and semantic child workflows with stable child run IDs. Process-stop checkpoint resume is covered; installed Kogwistar `0.6.8` terminal-failed-child retry is not claimed, and new-process lineage/no-double-publication evidence remain required before PASS. |
| P1 OCR model/config changes invalidate reuse | PARTIAL | `page_state.config_fingerprint` includes provider, model, endpoint, engine, language, and candidate list, and the reuse predicate compares it. `test_ocr_root_resume_rejects_changed_provider_before_replaying_pages` proves the public root rejects a changed model before replay, while `test_ocr_state_rejects_stale_provider_config_or_tampered_artifact` proves the changed configuration is rejected in a fresh process. The item remains partial pending full crash/reopen evidence. |
| P1 page artifact/database publication is crash-safe | PARTIAL | The current write path validates JSON, atomically replaces the artifact through unique same-directory temporary files, records an artifact SHA-256, rejects a well-formed tampered artifact on reuse, and adopts a valid rename-before-DB orphan. `test_ocr_two_store_recovery_predicates_hold_in_a_fresh_process` proves the direct restart predicates, `test_ocr_page_artifact_publish_crash_is_recoverable` injects the crash at the real publication boundary, and `test_duplicate_ocr_page_workers_accept_only_one_artifact` proves duplicate page delivery cannot replace the accepted artifact. These are still separate sub-properties, not crash-safe publication as a whole: checkpoint-before-artifact and full runtime/checkpoint reconciliation remain before PASS. |
| P1 grouping is not quadratic for long documents | PARTIAL | The root path avoids the old per-page semantic rebuild and uses a runtime grouping child. `test_grouping_incremental_resume_matches_uninterrupted_reference` proves a multi-pass interrupted/resumed frontier has the same canonical result as an uninterrupted reference, but bounded frontier cost, monotonic ownership, and a 100-page performance comparison remain open. |
| P1 Docling preserves item positions | PARTIAL | Item provenance boxes are exported for text, tables, and pictures when Docling exposes them. Full-page Markdown remains a compatibility fallback for adapters with no positional items; live multi-item Docling coverage is still required. |

## Required follow-up before release

- [~] Add an OCR preparation/grouping workflow design and execute those stages
  through the existing Kogwistar `WorkflowRuntime`, not a second runtime.
  Root and child designs are present; release evidence remains open.
- [~] Add a runtime trace test proving OCR and grouping stage nodes are persisted
  and resume without replaying completed page work. The current focused test
  proves terminal child reuse; interruption, new-process resume, and duplicate
  resume remain open.
- [ ] Add a mocked Docling item/provenance test with separate text, table, and
  figure boxes, including ordering and coordinate clipping.
- [~] Add crash-window reconciliation tests for artifact rename/database commit
  ordering in both directions. The exact completed-page reuse test for a
  tampered-but-well-formed JSON artifact, fresh-process configuration test,
  fresh-process two-store predicate test, and real artifact-first crash
  injection test are now present; checkpoint-first crash recovery and full
  runtime/checkpoint reconciliation remain open.
- [~] Add evidence-ledger rows for C01-C06 with implementation locations,
  exact test IDs, before/after behavior, and runtime classification.
- [~] Verify every grouping checkpoint writer uses the same revision/absence
  fence. The grouping child and summary persistence paths now fence both the
  tree and its frontier, reject orphaned/missing-frontier checkpoints, and the
  stale-summary regression proves a checkpoint conflict propagates for retry;
  the baseline compatibility writer and crash/restart behavior remain open.
- [~] Record C06 OCR quality diagnostics as typed persisted metadata and include
  the quality policy in page cache identity. Add a public strict-policy test;
  still require persisted reopen evidence and provider-side non-text coverage
  signals before marking the gate complete.
- [x] Run the full parser CI matrix after the P0 runtime work. Parser CI run
  `38663529209` passed CPython 3.12-3.14 and PyPy 3.11; slow/manual provider
  experiments remain separate gates.
