# OCR Review Fix Checklist

This checklist tracks the six findings from the OCR provider abstraction review.
`PASS` requires code, a focused regression test, and evidence from the actual
runtime boundary. `PARTIAL` means containment is improved but recovery,
reconciliation, or end-to-end evidence is still missing. A checklist mark is
not a substitute for a test ID and runtime result.

| Finding | Status | Evidence / remaining work |
| --- | --- | --- |
| P0 grouping checkpoint overwrite on resume | PARTIAL | Prefix pages no longer overwrite `document_tree_state`, but the preparation path still creates a baseline SQLite tree before the runtime-owned grouping checkpoint. Monotonic generation/CAS ownership and crash recovery remain open. |
| P0 OCR and grouping are persisted Kogwistar workflow nodes | PARTIAL | A versioned root workflow now invokes persisted page, grouping, and semantic child workflows with stable child run IDs. Same-process failed-child resume is now covered; new-process lineage and no-double-publication evidence remain required before PASS. |
| P1 OCR model/config changes invalidate reuse | PARTIAL | `page_state.config_fingerprint` includes provider, model, endpoint, engine, language, and candidate list, and the reuse predicate compares it. A fresh-process regression proving stale completed pages cannot be reused after configuration changes remains required. |
| P1 page artifact/database publication is crash-safe | PARTIAL | The current write path validates JSON, atomically replaces the artifact, and records an artifact SHA-256. Rename-before-DB and DB-before-rename crash-window reconciliation, plus tamper-reuse tests, remain required before PASS. |
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
- [ ] Add crash-window reconciliation tests for artifact rename/database commit
  ordering and a tampered-but-well-formed JSON artifact.
- [~] Add evidence-ledger rows for C01-C06 with implementation locations,
  exact test IDs, before/after behavior, and runtime classification.
- [~] Record C06 OCR quality diagnostics as typed persisted metadata and include
  the quality policy in page cache identity. Add a public strict-policy test;
  still require persisted reopen evidence and provider-side non-text coverage
  signals before marking the gate complete.
- [ ] Run the full parser CI matrix after the P0 runtime work.
