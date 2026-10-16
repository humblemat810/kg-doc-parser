# OCR Review Fix Checklist

## Latest verified release evidence (2026-10-16)

### Recovery seam update (commit `5a95c19`)

The fresh page-recovery child now calls `reconcile_orphan_artifact()` before
the OCR reuse predicate. This is required when recovery starts with cursor zero:
the checkpoint prefix loop cannot otherwise inspect page one, and a valid
artifact-first result could be replayed unnecessarily. The local regressions
`test_ocr_root_concurrent_resume_has_one_owner_and_one_result` and
`test_grouping_checkpoint_reopens_after_process_termination` pass, as does the
focused six-test grouping/runtime run. Parser CI run `#280`
(`38883695001`) is still pending, so the release-gate statuses remain partial.

Latest evidence correction: parser CI run `38873761447` passed all four
interpreter jobs (CPython 3.12-3.14 and PyPy 3.11) on the exact branch head
`cb22da2`. That run includes the typed OCR-runtime state seam and artifact
metadata reader; it is matrix evidence for this commit, not proof that the
remaining crash-window, Linux SIGKILL, or OCR completeness gates are closed.

The grouping frontier mismatch is now validated before terminal resume, not
only inside the runtime restore step. This prevents a terminal Kogwistar
checkpoint from bypassing parser corruption detection. The focused regression
`tests/test_workflow_ingest_ocr_pipeline.py::test_grouping_resume_rejects_frontier_that_disagrees_with_tree`
passes locally against Kogwistar `0.6.12`; broader dual-store and crash/restart
gates remain partial.

PyPI now exposes Kogwistar `0.6.12`, and the parser manifest and lockfile are
updated to consume it. The post-pin installed-release matrix is green on the
current branch; runs that installed `0.6.10` remain historical evidence only.
Before that pin, the installed matrix
`test_ocr_root_concurrent_resume_has_one_owner_and_one_result` receives a
nested-workflow failure instead of a `WorkflowRunOwnershipError`, and
`test_grouping_checkpoint_reopens_after_process_termination` exits with the
child worker's recovery error. The corrected published runtime must now be
retested before this gate can pass.

Dual-store crash-order, Linux SIGKILL, and OCR completeness gates remain open
independently.

The current branch also adds reusable regressions for oversized summary
evidence, built-in Poppler cache identity, and failed PNG publication. These
prove the individual seams; they do not promote the broader release gates to
PASS without their required end-to-end evidence.

## Latest evidence correction (2026-10-16)

The terminal-root recovery case is now covered locally. A completed root with
a deleted accepted page artifact is detected before terminal resume; the parser
starts a deterministic `|recovery-N` root incarnation, reuses valid pages,
retries the invalid page, and preserves the original logical lineage. The
public result keeps the stable logical root ID while persisted recovery steps
retain their incarnation suffix. The deleted-artifact regression and the
focused fresh-process matrix pass locally, including concurrent resume, real
process termination, and grouping termination. Ruff and `git diff --check`
pass. This closes the specific same-process terminal-root short-circuit gap,
but does not close the dual-store crash ordering, Linux SIGKILL, or independent
OCR completeness gates.

The parser root-finalization seam now has an additional fail-closed guard in
`ocr_runtime.py` (`c173057`).  It rechecks page completion, accepted-page
coverage, and artifact/SQLite reconciliation immediately before root
publication.  The existing real end-to-end regression passes locally after
deleting an accepted page artifact, proving that stale terminal checkpoint
data is rejected rather than published.

The grouping checkpoint writer now also rejects an orphaned frontier when no
tree revision exists, including callers that do not request an explicit
absence fence.  This closes a previously permissive compatibility path that
could have replaced a frontier left by an interrupted tree write.  The
regression `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_state_store_round_trips_grouped_tree_and_invalidates_changed_source`
passes with the orphaned-frontier assertion.  S06 remains partial because
writer-wide monotonic-generation and crash/restart evidence are still open.

The earlier parser CI run `38777972861` failed before Kogwistar `0.6.10`
publication completed. A clean environment now verifies the published package,
and the selected parser CI suite passes with `429 passed, 1 skipped, 327
deselected`. S09, C03, C05, and OCR completeness remain partial only for the
specific recovery and evidence gaps listed below.

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

### External review disposition (2026-10-15)

The latest downstream review is accepted as a correction to checklist status,
not as evidence that the implementation is complete. Two items that may look
green when reading only the file-writing code are explicitly **PARTIAL**:

The Kogwistar `0.6.10` tag is on `main`, is published on PyPI, and is the
parser's locked dependency. Clean installed-package checks and the selected
parser CI suite pass. This closes the dependency publication blocker; it does
not close the independent crash-order, Linux SIGKILL, or completeness gates.

- **S02 artifact SHA-256 verification:** the completed-page reuse predicate
  rejects a well-formed tampered artifact, but the complete root path has not
  yet proven both SQLite/artifact commit orders and runtime-checkpoint
  reconciliation.
- **S02 atomic JSON publication:** unique temporary files and `os.replace()`
  protect a single file publication, but they do not form a transaction with
  SQLite or the persisted Kogwistar checkpoint. The checkpoint-before-artifact
  and artifact-before-checkpoint crash windows remain open.

The same review confirms that the provider/configuration fingerprint is
implemented but not release-complete until reuse is proven across every
output-affecting field after a fresh-process reopen. S06 remains **PARTIAL**:
expected-revision/absence fencing is useful containment, not proof of
monotonic generation, complete writer coverage, or crash/restart safety.

The installed dependency gate is now evidenced as PASS. C02-C06 remain
PARTIAL because their remaining crash-order, signal, completeness, or
long-document evidence is still open; those are documented release risks, not
the former Kogwistar publication blocker.

### New recovery evidence (2026-10-15)

The real persisted 100-page regression
`tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_root_resumes_after_page_63_and_matches_uninterrupted_reference`
passed in 1030.06 seconds with merged Kogwistar source commit `c0eb92c` on
`PYTHONPATH`. The uninterrupted and interrupted/resumed canonical graph
payloads matched, and the injected page failure was retried without rebuilding
accepted page work. This improves C05 evidence but does not close it: the
test used a coordinated source checkout. Kogwistar `v0.6.10` is now published
on PyPI. A true
fresh-process recovery is now covered by the coordinated source-checkout
termination test below, but Linux SIGKILL evidence and the duplicate-root-resume
race remain open.

The focused test
`tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_page_resume_reopens_state_in_a_fresh_process`
passes in 33.32 seconds and proves accepted OCR page reuse across two Python
processes. It deliberately uses the in-memory graph backend only for the
workflow step wrapper, so it is page-state evidence rather than proof that a
runtime checkpoint survives process loss; persistent graph-backend recovery
remains an open gate.

### Process-boundary recovery evidence (2026-10-15)

`tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_root_reopens_after_real_process_termination`
passes in 117.12 seconds. It terminates a separate worker after page one has
been accepted and page two has started, then launches a fresh worker with the
same persistent state. The resumed worker calls only page two, preserves the
same root run ID, and accepts one grouping result. This is real process
termination evidence on the host platform, not a simulated callback failure.
It uses coordinated parser/core source checkouts, so installed-package recovery
still depends on verifying Kogwistar `0.6.10` in the current matrix. Linux `SIGKILL`,
duplicate-root-resume exclusion, and dual-store crash reconciliation remain
release gates.

The grouping checkpoint regression
`tests/test_workflow_ingest_ocr_pipeline.py::test_grouping_checkpoint_reopens_after_process_termination`
passes in 65.52 seconds. It terminates a separate grouping worker after the
checkpoint is persisted but while summary generation is in progress, then
resumes the same run after the short ownership lease expires and recovers the
accepted section and summary. This closes the host-platform grouping
checkpoint subclaim; Linux `SIGKILL`, installed-package, and dual-store
reconciliation gates remain open.

The concurrent-resume regression
`tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_root_concurrent_resume_has_one_owner_and_one_result`
passes in 120.70 seconds. It starts two fresh workers against the same
persisted root after process termination; exactly one exits successfully with
the accepted page-two result and the other exits with an ownership failure.
This closes the source-checkout duplicate-resume race subclaim, but not the
installed-package or Linux `SIGKILL` release gates.

## Review-adjusted execution checklist

This is the short operational list for the two implementations. The detailed
S01-S12 and C01-C06 rows in
`SEAM_AND_RUNTIME_IMPLEMENTATION_CHECKLIST.md` remain normative.

### A. Kogwistar runtime integration

- [~] Root/page/grouping/semantic/summary stages use one persisted ownership
  model with stable parent/child IDs.
- [~] Typed checkpoint state contains source/config identity, cursor, accepted
  artifacts, grouping frontier, summary identity, and final input reference.
- [~] Artifact-first and checkpoint-first crash recovery are reconciled and
  tested against the same accepted-result rules.
- [~] Grouping and summary writers use revision/generation fencing; all legacy
  and compatibility writers are audited.
- [~] Incremental grouping is bounded by an accepted frontier and matches an
  uninterrupted canonical reference.
- [~] A fresh process reopens the same root run after real interruption and
  does not create a second semantic generation or duplicate graph publication.
  Coordinated source-checkout tests cover process termination and concurrent
  ownership fencing; installed-package, Linux SIGKILL, and full graph
  publication verification remain open.
- [ ] The installed released Kogwistar version supports retrying a durably
  failed child; the parser does not substitute a private child-ID workaround.

### B. Strict seam validation

- [~] S01-S02 source/render/provider/config/artifact identities are persisted,
  checked on reuse, and protected by atomic single-file publication.
- [~] S02 has tamper, stale-config, orphan-adoption, and late-writer tests;
  two-store crash-order reconciliation remains open.
- [~] S03-S04 preserve declared spatial precision, media provenance, and
  deterministic mixed-object ordering; live multi-object provider evidence is
  still required.
- [~] S05 proves accepted hierarchy reaches semantic parsing and graph export,
  rather than surviving only as ignored metadata. The export metamorphic test
  now proves parent-child structure is consumed; persisted semantic-parser
  publication evidence remains open.
- [~] S06-S08 protect grouping checkpoints, preserve atomic objects, and
  invalidate summaries; S08's final-group identity loss is fixed and covered,
  while monotonic generation and the complete writer audit remain open.
- [~] S09-S10 verify runtime checkpoint ownership and collision-safe identity
  separation for same-stem, changed-path, and changed-content inputs.
- [~] S11-S12 cover cross-page references and reject conflicting public versus
  internal policy values. Multiple reference slices now round-trip and validate
  at the document-model boundary; downstream export and policy-conflict
  rejection remain open.
- [~] C06 has typed provider-loss diagnostics and strict rejection; source-side
  completeness for figures, tables, captions, and unsupported coordinates is
  not yet proven.

### Release decision

- [x] Do not count the two previously over-optimistic S02 marks as complete.
- [x] Do not count same-process tests as fresh-process runtime evidence.
- [x] Do not count a stored digest as reuse enforcement without an exact reuse
  test.
- [x] Do not count atomic file replacement as a two-store transaction.
- [ ] Do not bump the parser version or update downstream pins until all
  release-blocking runtime and seam rows above are green with evidence.

## Latest audit corrections

These S02 subclaims are intentionally tracked separately. A green file-write
test must not be counted as a green two-store commit protocol:

| Subclaim | Status | Required evidence before release |
| --- | --- | --- |
| Provider/configuration fingerprint is part of page identity | PARTIAL | Fresh-process reuse rejection for every output-affecting provider, model, engine, and policy field, plus a persisted reopen assertion. |
| OCR artifact SHA-256 is recorded | PARTIAL (implementation evidence) | Artifact metadata contains the digest after validated publication; this is not, by itself, proof that every reuse path enforces it. |
| OCR artifact SHA-256 is enforced on completed-page reuse | PARTIAL (focused boundary) | `test_ocr_state_rejects_stale_provider_config_or_tampered_artifact` rejects a well-formed but modified JSON artifact through the actual reuse predicate. Fresh-process, all-provider-field, and dual-store recovery evidence remain release gates. |
| JSON publication uses a unique temporary file and atomic replacement | PARTIAL (file boundary) | `test_atomic_json_publication_uses_distinct_temporary_files_for_concurrent_writers` proves distinct same-directory temporary files and replacement. It does not prove a two-store commit or recovery after either commit order. |
| Artifact and SQLite/Kogwistar checkpoint commits are crash-consistent | PARTIAL | `test_ocr_page_artifact_publish_crash_is_recoverable` injects a crash in the real `_process_ocr_page` path after atomic rename and before the SQLite completion update, then reopens SQLite and adopts only the source/config-matching artifact. `test_prepare_ocr_workflow_retries_checkpoint_without_artifact` proves the public preparation path reruns a database-first incomplete page. Full root/runtime checkpoint reconciliation remains open. |
| Grouping/checkpoint and page-state writers are fenced against late writers | PARTIAL | Expected revision/absence CAS and retryable conflict propagation are covered. Page completion and failure require pending status, source/configuration identity, and attempt ownership; same-identity duplicate OCR deliveries use first-valid-result-wins and close superseded pending attempt rows (`tests/test_workflow_ingest_ocr_pipeline.py::test_duplicate_ocr_page_workers_accept_only_one_artifact`, `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_page_completion_rejects_a_late_writer_after_acceptance`, `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_page_failure_cannot_downgrade_an_accepted_result`, `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_page_failure_cannot_replace_a_newer_pending_attempt`). Resume rejects a persisted frontier that disagrees with the restored tree (`tests/test_workflow_ingest_ocr_pipeline.py::test_grouping_resume_rejects_frontier_that_disagrees_with_tree`); compatibility writers, monotonic generation, and crash/restart ordering remain open. |

The realistic recovery guarantee is at-least-once provider execution with
exactly one accepted canonical result. The implementation must not claim
exactly-once OCR or LLM calls across arbitrary process termination.

## Reviewer reconciliation ledger (2026-10-15)

The latest review does not establish new passing gates. It corrects the
interpretation of existing evidence:

| Review finding | Correct status | What is proven | What remains open |
| --- | --- | --- | --- |
| S02 artifact SHA-256 verification | PARTIAL | The completed-page reuse predicate rejects a well-formed tampered artifact. | Both SQLite/artifact commit orders, runtime checkpoint reconciliation, and fresh-process recovery through the complete root remain unproven. |
| S02 atomic JSON publication | PARTIAL | Unique same-directory temporary files and atomic replacement are used. | Atomic replacement of one file does not make the artifact and SQLite/Kogwistar checkpoint a transaction; both crash orders need recovery tests. |
| S05 accepted hierarchy consumption | PARTIAL | The semantic callback receives and consumes the accepted `document_tree` to derive semantic output, `WorkflowExportBundle.grouping_tree` preserves it, and the OCR root rejects a missing or changed grouping artifact. Regression: `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_workflow_resumes_cached_pages_and_terminal_semantic_run`. | Explicit compatibility behavior for consumers that ignore hierarchy remains open. |
| S06 grouping checkpoint safety | PARTIAL | Expected-revision/absence fencing, frontier consistency checks, stable proposal identities, and `test_ocr_state_store_round_trips_grouped_tree_and_invalidates_changed_source` now reject unfenced updates to an existing checkpoint; duplicate accepted proposals are idempotent. | Every writer must be audited; late-writer, monotonic generation, and crash/restart behavior remain release blockers. |
| Runtime ownership | PARTIAL | Stable root/page/grouping/semantic child identities and persisted CAS ownership leases are exercised by the installed Kogwistar `0.6.10` environment. Fresh-process lineage, real termination, and concurrent resume tests pass locally. | Linux SIGKILL and release-CI evidence remain open. |
| 100-page resume | PARTIAL | The persisted reference/resume comparison and installed-package fresh-process tests pass locally; the selected parser CI suite also passes. | A full 100-page run through the installed release, Linux SIGKILL, and complete dual-store reconciliation remain open. |
| OCR completeness | PARTIAL | Typed provider diagnostics and strict policy can reject provider loss signals. | Interval coverage alone does not detect missing figures/tables/coordinates; provider-side completeness and recovery evidence remain open. |
| Same-basename PDF artifact isolation | FIXED (focused boundary) | `_resolve_ocr_source_plan` now derives both PDF artifact directories from document identity, PDF content fingerprint, and a safe source stem. `tests/test_workflow_ingest_ocr_pipeline.py::test_pdf_artifact_directories_do_not_collide_for_same_basename` proves two different `report.pdf` inputs cannot share rendered or legacy artifact paths. | Existing artifact directories using the old basename-only layout are intentionally not adopted as new-source identities; migration/cleanup policy remains operational work. |

The checklist therefore intentionally contains no green S02, S06, C02, C03,
C04, C05, or C06 release mark. A passing unit test for a sub-property is
recorded as evidence for that sub-property only.

## Required reviewer acceptance fields

For each A, S, C, and P item, the implementation ledger must contain:

1. Implementation: file, function, and stable source location.
2. Regression test: exact pytest node ID and the assertion that proves the
   contract.
3. Before fix: reproduced failure or demonstrated missing behavior.
4. After fix: passing result and the boundary it proves.
5. Runtime: standalone, mocked Kogwistar, installed Kogwistar, or persisted
   multi-process runtime.
6. Status: verified pass, partial, blocked, failing, not implemented, or not
   verified.

Do not infer a multi-process, crash-consistency, or provider-completeness
claim from a same-process or mocked test.

| Finding | Status | Evidence / remaining work |
| --- | --- | --- |
| P0 grouping checkpoint overwrite on resume | PARTIAL | Prefix pages no longer overwrite `document_tree_state`; grouping and summary persistence now atomically fence both expected revisions and expected absence. The preparation path still has a baseline SQLite compatibility path outside the grouping child, and a late compatibility writer or crash can still violate the intended ownership boundary until every writer is audited. |
| P0 OCR and grouping are persisted Kogwistar workflow nodes | PARTIAL | A versioned root workflow now invokes persisted page, grouping, and semantic child workflows with stable child run IDs. Same-process test coverage exists, but it is not evidence of a real process stop; new-process lineage/no-double-publication evidence remains required before PASS. |
| P1 OCR model/config changes invalidate reuse | PARTIAL | `page_state.config_fingerprint` includes provider, model, endpoint, engine, language, and candidate list, and the reuse predicate compares it. `test_ocr_root_resume_rejects_changed_provider_before_replaying_pages` proves the public root rejects a changed model before replay, while `test_ocr_state_rejects_stale_provider_config_or_tampered_artifact` proves the changed configuration is rejected in a fresh process. The item remains partial pending full crash/reopen evidence. |
| P1 page artifact/database publication is crash-safe | PARTIAL | The current write path validates JSON, atomically replaces the artifact through unique same-directory temporary files, records an artifact SHA-256, rejects a well-formed tampered artifact on reuse, and adopts a valid rename-before-DB orphan. The legacy `record_page_completed()` path is now also fenced against late writers by source/config identity and pending status, with `test_ocr_page_completion_rejects_a_late_writer_after_acceptance`. `test_ocr_two_store_recovery_predicates_hold_in_a_fresh_process` proves the direct restart predicates, `test_ocr_page_artifact_publish_crash_is_recoverable` injects the crash at the real publication boundary, and `test_duplicate_ocr_page_workers_accept_only_one_artifact` proves duplicate page delivery cannot replace the accepted artifact. These are still separate sub-properties, not crash-safe publication as a whole: checkpoint-before-artifact and full runtime/checkpoint reconciliation remain before PASS. |
| P1 grouping is not quadratic for long documents | PARTIAL | The root path avoids the old per-page semantic rebuild and uses a runtime grouping child. `test_grouping_incremental_resume_matches_uninterrupted_reference` covers a controlled multi-pass case, but bounded frontier cost, monotonic ownership, a real interruption/reopen path, and a 100-page performance comparison remain open. |
| P1 Docling preserves item positions | PARTIAL | Item provenance boxes are exported for text, tables, and pictures when Docling exposes them. `tests/test_workflow_ingest_ocr_providers.py::test_docling_adapter_preserves_text_table_figure_provenance` proves mocked multi-item ordering and clipping; full-page Markdown remains a compatibility fallback for adapters with no positional items, and live multi-item Docling coverage is still required. |

## Required follow-up before release

- [~] Add an OCR preparation/grouping workflow design and execute those stages
  through the existing Kogwistar `WorkflowRuntime`, not a second runtime.
  Root and child designs are present; release evidence remains open.
- [~] Add a runtime trace test proving OCR and grouping stage nodes are persisted
  and resume without replaying completed page work. The current focused test
  proves terminal child reuse; interruption, new-process resume, and duplicate
  resume remain open.
- [x] Add a mocked Docling item/provenance test with separate text, table, and
  figure boxes, including ordering and coordinate clipping. Evidence:
  `tests/test_workflow_ingest_ocr_providers.py::test_docling_adapter_preserves_text_table_figure_provenance`.
- [~] Add crash-window reconciliation tests for artifact rename/database commit
  ordering in both directions. The exact completed-page reuse test for a
  tampered-but-well-formed JSON artifact, fresh-process configuration test,
  fresh-process two-store predicate test, and real artifact-first crash
  injection test are now present; checkpoint-first crash recovery and full
  runtime/checkpoint reconciliation remain open.
- [x] Prove a failed summary can be retried without replaying accepted OCR.
  `tests/test_workflow_ingest_ocr_pipeline.py::test_failed_summary_is_retryable_without_replaying_ocr`
  verifies two OCR calls total and one failed plus one accepted summary
  generation for the retried target.
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
- [x] Include an explicit summary generation identity in cache keys. `SummaryPolicy.generation_fingerprint`
  is part of the persisted summary key; `tests/test_workflow_ingest_ocr_pipeline.py::test_summary_generation_fingerprint_invalidates_persisted_cache`
  proves a provider/model/prompt identity change does not reuse an accepted summary.
  Callers must change the fingerprint whenever output-affecting generation configuration changes.
- [~] Run the full parser CI matrix after the P0 runtime work. Earlier runs
  passed CPython 3.12-3.14 and PyPy 3.11, but the latest pushed run
  `38757822459` failed all four jobs at the installed-Kogwistar version
  contract because PyPI `0.6.9` still exposes
  `kogwistar.__version__ == "0.6.8"`. The parser contract remains strict; the
  matrix is being rerun against published Kogwistar `0.6.10`. Slow/manual
  provider experiments remain separate gates.

## Reviewer-required acceptance additions

These require an implementation location, an exact pytest node, before/after
behavior, and a runtime classification. C01 was previously evidenced, but the
current released-package check found a Kogwistar `0.6.9` public-version
mismatch. Core PR #61 is merged and tagged as `v0.6.10`; C01 is published but
the current parser matrix rerun is still pending. C02-C06 have partial implementation
evidence, but none is a release-gate pass yet:

- [~] Verify the installed Kogwistar package and public workflow persistence
  API, rather than relying on a source checkout or import-only probe. The
  contract test caught that Kogwistar `0.6.9` metadata reported `0.6.9` while
  its public module reported `0.6.8` in parser CI run `38720363271`. Re-run
  the same test and CPython 3.12-3.14/PyPy 3.11 matrix after tagged Kogwistar
  `v0.6.10` is visible on PyPI.

The checklist-only rerun `38729861407` failed in all four matrix jobs before
the OCR acceptance suite because the installed `0.6.9` package still exposes
`kogwistar.__version__ == "0.6.8"`. The same failure is reproduced locally by
`tests/test_pypy_ci_contract.py::test_installed_kogwistar_exposes_the_persisted_workflow_api`.
This is historical release-gate evidence, not a reason to relax the version
contract; the current run must verify the published `0.6.10` package.
- [~] Document and test one root/child ownership model, including semantic
  parser checkpoint and graph publication identities. Stable root/page/grouping/
  semantic IDs and same-process ownership tests exist; fresh-process lineage
  and no-second-publication evidence remain open.
- [~] Reconcile both artifact/SQLite and Kogwistar/checkpoint commit orders in
  a fresh process. Artifact-first adoption and direct predicates exist; the
  inverse checkpoint-first order and complete runtime reconciliation remain
  open.
- [~] Compare incremental and uninterrupted grouping using canonical content,
  excluding volatile run metadata; prove completed subtrees remain stable. A
  controlled multi-pass canonical comparison exists; bounded-frontier and
  real interruption/reopen evidence remain open.
- [~] Prove at-least-once execution with exactly one accepted canonical result
  after process interruption, including duplicate-resume behavior. Root and
  grouping fresh-process recovery plus the duplicate-root ownership race are
  covered with coordinated source checkouts; Linux SIGKILL, installed-package,
  and dual-store reconciliation evidence remain open.
- [~] Add a quality/completeness gate that detects provider omissions beyond
  source interval coverage, including non-text and coordinate-loss diagnostics.
  Typed provider-loss diagnostics and strict rejection exist; source-side
  completeness for figures, tables, captions, and unsupported coordinates is
  not yet proven.

### Evidence update (2026-10-15)

- The source-checkout fresh-process root recovery test now passes:
  `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_root_resume_reopens_same_lineage_in_a_fresh_process`.
  It uses persistent SQLite metadata, waits for the short test lease to expire,
  resumes the same root ID, retries only the failed page, and completes graph
  publication. This improves S09/C05 evidence but does not replace the
  installed Kogwistar `0.6.10` verification gate.

- S08's final-group identity loss is fixed in
  `workflow_ingest/document_adapters.py::apply_grouping_proposals` and covered
  by `tests/test_document_adapters.py::test_final_group_keeps_identity_and_kind_under_document_root`.
  The root's one-child exception is structural only; semantic grouping nodes
  remain non-unary.
- Kogwistar `0.6.10` is merged, tagged, and publicly installable. The parser
  lockfile targets that released version. The installed-package recovery gate
  still needs the current parser matrix and remaining crash-order evidence
  before C05 can be promoted.

## Broad OCR Audit Ledger (2026-10-16)

Source: `kg_doc_parser_ocr_broad_audit_2026-10-15.md`. The source review was
static and did not prove runtime behavior. The statuses below are deliberately
conservative and apply to the current parser branch HEAD `e1fb47a`. A row is
not release-complete until its
focused test, full matrix, and exact HEAD are verified.

| ID | Status | Current evidence or remaining work |
| --- | --- | --- |
| B01 | [~] Fixed locally; CI pending | `_materialize_image_payloads` now derives page paths, rejects rendered-directory/page symlinks, and writes byte payloads atomically. Tests: `test_materialize_image_payloads_uses_safe_atomic_page_paths`, `test_materialize_image_payloads_rejects_invalid_page_numbers_and_symlinks`. |
| B02 | [~] Fixed locally; CI pending | `build_document_tree` now raises typed `UnrepresentedSourceUnitError` with a structured diagnostic instead of silently dropping a non-text unit that lacks a description and verified content-addressed media. Blank pages with no units remain empty. Regression: `test_build_document_tree_rejects_unrepresented_nontext_unit`; `tests/test_document_adapters.py` passes 19 tests locally. Input-unit accounting and full CI remain open. |
| B03 | [~] Fixed locally; CI pending | `suggest_cross_page_continuations` now requires both frontier nodes to be canonical text leaves, so page-level incomplete-text flags cannot join a paragraph to a figure/table/caption description or media item. Regressions: `test_cross_page_continuation_never_merges_text_into_nontext` and `test_cross_page_continuation_suggestion_requires_ocr_boundary_signal`. Geometry and stronger paragraph-boundary evidence remain open. |
| B04 | [~] Fixed locally; CI pending | Added immutable `EvidenceSlice` records and a typed `GroundedSummaryGenerator` boundary. The public preparation, grouping, and root workflows thread this boundary through; `test_grounded_summary_generator_receives_only_owned_source_slices` proves two summaries over one source item receive only their ordered, exact ranges. Legacy `SummaryGenerator` remains available for compatibility; provider adoption and full CI remain open. |
| B05 | [~] Fixed locally; CI pending | `OCRClusterResponse.spatial_accuracy` and normalized `SourceUnit.metadata` now carry an explicit page/region/line/word/character declaration. Unknown or absent precision fails closed to page accuracy; direct OCR and item-level Docling paths declare region precision. Tests: `test_synthetic_page_box_does_not_claim_region_accuracy`, `test_explicit_region_precision_is_preserved`, and the full `tests/test_document_adapters.py` file (19 passed). |
| B06 | [~] Fixed locally; CI pending | Render and legacy artifact paths use collision-resistant document/source identities. `test_pdf_artifact_directories_do_not_collide_for_same_basename` now exercises two valid PNG renders from distinct same-named PDFs and confirms distinct artifact directories and bytes. |
| B07 | [x] Verified on current branch | `_materialize_image_payloads()` requires positive, unique, ascending page numbers before publishing artifacts; it no longer silently sorts explicitly out-of-order input. `tests/test_workflow_ingest_ocr_pipeline.py::test_materialize_image_payloads_rejects_invalid_page_numbers_and_symlinks` covers non-positive, duplicate, and symlink cases, and `::test_materialize_image_payloads_rejects_out_of_order_page_numbers` covers ordering. |
| B08 | [~] Partial | `OCRQualityPolicy.reject_empty_page` now provides an explicit opt-in fail-closed gate for provider-reported empty OCR output, with `test_ocr_quality_policy_can_reject_empty_provider_output`. The default remains permissive for legitimate blank pages; independent image/text completeness checks are still needed. |
| B09 | [~] Fixed locally; CI pending | Summary target selection measures exact ordered `EvidenceSlice` payloads and now skips a node whose estimated evidence exceeds `maximum_input_tokens`, instead of clamping accounting while still sending the oversized payload. Regressions: `test_summary_target_planner_counts_exact_slices_not_whole_source_items` and `test_summary_target_planner_does_not_send_oversized_evidence`. Provider serialization overhead and full CI remain open. |
| B10 | [~] Partial | Durable artifact IDs and runtime integration exist, but accumulated `raw_pages` checkpoint serialization and linear-growth evidence remain open. |
| B11 | [~] Fixed locally; CI pending | Render manifests include page count and the built-in renderer identity now includes PDF raster defaults, installed `pdf2image` version, and `POPPLER_PATH`; reuse rejects missing, duplicate, or out-of-order pages. Regressions: `test_pdf_render_manifest_rejects_a_missing_page_from_a_valid_pdf` and `test_builtin_pdf_render_identity_includes_poppler_selection`. |
| B12 | [~] Fixed locally; CI pending | Reference extraction now maps matches across the deterministic ordered OCR-unit stream back to exact source slices. Regression: `test_reference_mentions_cross_ocr_units_preserve_exact_source_slices`; the filtered OCR CI module passed 60 tests locally. |
| B13 | [ ] Open | The default PDF path still rasterizes pages; hybrid native PDF text extraction is not implemented. |
| B14 | [~] Fixed locally; CI pending | `_render_pdf_to_images` rejects page symlinks, validates cached image decoding, and publishes regenerated PNGs through a temporary file and atomic rename. Render-manifest reuse verifies page digests, dimensions, and valid-PDF page count. Regressions: `test_pdf_rasterization_regenerates_corrupt_cached_page_atomically`, `test_pdf_render_manifest_rejects_a_missing_page_from_a_valid_pdf`, and `test_png_publication_failure_keeps_previous_artifact_and_cleans_temp`. |

This ledger is a release gate, not a claim that the branch is complete. The
local filtered OCR CI module passed `60 passed, 14 deselected` with
`-p no:cacheprovider`; remote CPython/PyPy matrix verification and the
remaining cross-store, provider-matrix, and crash-window gates are still open.
Kogwistar runtime fixes are on `fix/workflow-run-ownership`; downstream pins
must not advance until the required remote matrix is green and the corrected
release is available.
