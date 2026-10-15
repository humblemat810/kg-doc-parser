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

| Finding | Status | Evidence / remaining work |
| --- | --- | --- |
| P0 grouping checkpoint overwrite on resume | PARTIAL | Prefix pages no longer overwrite `document_tree_state`, and the checkpoint store now atomically fences both expected revisions and expected absence. The preparation path still has a baseline SQLite compatibility path outside the grouping child, and crash recovery remains open. |
| P0 OCR and grouping are persisted Kogwistar workflow nodes | PARTIAL | A versioned root workflow now invokes persisted page, grouping, and semantic child workflows with stable child run IDs. Same-process failed-child resume is now covered; new-process lineage and no-double-publication evidence remain required before PASS. |
| P1 OCR model/config changes invalidate reuse | PARTIAL | `page_state.config_fingerprint` includes provider, model, endpoint, engine, language, and candidate list, and the reuse predicate compares it. `test_ocr_root_resume_rejects_changed_provider_before_replaying_pages` proves the public root rejects a changed model before replay, while `test_ocr_state_rejects_stale_provider_config_or_tampered_artifact` proves the changed configuration is rejected in a fresh process. The item remains partial pending full crash/reopen evidence. |
| P1 page artifact/database publication is crash-safe | PARTIAL | The current write path validates JSON, atomically replaces the artifact, records an artifact SHA-256, rejects a well-formed tampered artifact on reuse, and adopts a valid rename-before-DB orphan. This does not by itself prove crash consistency: both rename-before-DB and DB-before-rename reconciliation must be exercised in a fresh-process crash harness before PASS. |
| P1 grouping is not quadratic for long documents | PARTIAL | The root path avoids the old per-page semantic rebuild and uses a runtime grouping child, but bounded incremental frontier merges, monotonic ownership, and batch/incremental equivalence are not yet proven. |
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
  tampered-but-well-formed JSON artifact and the fresh-process configuration
  test are now present; fresh-process crash injection and two-store recovery
  remain open.
- [~] Add evidence-ledger rows for C01-C06 with implementation locations,
  exact test IDs, before/after behavior, and runtime classification.
- [~] Record C06 OCR quality diagnostics as typed persisted metadata and include
  the quality policy in page cache identity. Add a public strict-policy test;
  still require persisted reopen evidence and provider-side non-text coverage
  signals before marking the gate complete.
- [x] Run the full parser CI matrix after the P0 runtime work. Parser CI run
  `38655395569` passed CPython 3.12-3.14 and PyPy 3.11; slow/manual provider
  experiments remain separate gates.
