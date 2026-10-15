# OCR Review Fix Checklist

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
but does not close the installed-release GitHub gate, dual-store crash ordering,
or the requirement for a published core with run ownership enforcement.

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

Parser CI run `38777972861` failed all four interpreter jobs at the installed
Kogwistar contract gate before parser tests ran.  Therefore this change is
locally verified but not release-verified.  Kogwistar `v0.6.10` is tagged but
not yet confirmed publicly installable, and the ownership lease implementation
remains on the unmerged `fix/workflow-run-ownership` branch.  S09, C03, C05,
and the installed-release portions of the recovery checklist remain open.

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

The Kogwistar `v0.6.10` tag is now on `main`. Its PyPI release workflow has
passed verification, wheel builds, sdist, and artifact audit, but the protected
`pypi` deployment is still waiting for approval. The parser must continue to
pin the published `0.6.9` package until a compatible core release is publicly
installable and a clean CPython/PyPy matrix verifies the released package.
The ownership-lease API exercised by the parser's fresh-process recovery tests
is currently only on the separately pushed, unmerged
`fix/workflow-run-ownership` core branch; it must not be attributed to
`v0.6.10` until that branch is merged and released.

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

The release decision therefore remains unchanged: C01 is the only reviewer
gate currently evidenced as PASS; C02-C06 remain PARTIAL or BLOCKED, and the
parser version must not be bumped until the open rows have implementation,
reusable regression tests, and runtime evidence.

### New recovery evidence (2026-10-15)

The real persisted 100-page regression
`tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_root_resumes_after_page_63_and_matches_uninterrupted_reference`
passed in 1030.06 seconds with merged Kogwistar source commit `c0eb92c` on
`PYTHONPATH`. The uninterrupted and interrupted/resumed canonical graph
payloads matched, and the injected page failure was retried without rebuilding
accepted page work. This improves C05 evidence but does not close it: the
test used a coordinated source checkout, while Kogwistar `v0.6.10` is only
tagged and its protected PyPI deployment is still pending. A true
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
still depends on publishing and verifying Kogwistar `0.6.10`. Linux `SIGKILL`,
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
| Runtime ownership | PARTIAL | Stable root/page/grouping/semantic child identities exist in the persisted design. Core branch `fix/workflow-run-ownership` commits `817b03a` and `bc336e4` add persisted CAS ownership leases, explicit suspended-run release, and durable checkpoint metadata fallback when the graph node is not present. The branch is not yet merged or released. | A fresh process must reopen the same parent and child lineage without a second accepted semantic generation using the published core package. |
| 100-page resume | PARTIAL | The persisted test passes with a coordinated Kogwistar source checkout, but that is not installed-release evidence. Core `v0.6.10` is tagged and its protected PyPI publish is still waiting; the ownership/recovery commits `817b03a` and `bc336e4` are only on an unmerged feature branch. Installed-release verification, SIGKILL/new-process, and duplicate-resume evidence are still required. |
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
  matrix must be rerun after Kogwistar `0.6.10` is published. Slow/manual
  provider experiments remain separate gates.

## Reviewer-required acceptance additions

These require an implementation location, an exact pytest node, before/after
behavior, and a runtime classification. C01 was previously evidenced, but the
current released-package check found a Kogwistar `0.6.9` public-version
mismatch. Core PR #61 is now merged and tagged as `v0.6.10`; C01 is awaiting
PyPI publication and the parser matrix rerun. C02-C06 have partial implementation
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
This is evidence for the release gate, not a reason to relax the version
contract.
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
- Kogwistar `0.6.10` is merged and tagged, but its protected PyPI deployment is
  still pending. The parser lockfile must not be updated to that release until
  a clean environment can install it and confirm the public version/API. The
  installed-package recovery gate still needs to be rerun before C05 can be
  promoted.

## Broad OCR Audit Ledger (2026-10-16)

Source: `kg_doc_parser_ocr_broad_audit_2026-10-15.md`. The source review was
static and did not prove runtime behavior. The statuses below are deliberately
conservative and apply to the current parser branch HEAD `d40531e`. A row is
not release-complete until its
focused test, full matrix, and exact HEAD are verified.

| ID | Status | Current evidence or remaining work |
| --- | --- | --- |
| B01 | [~] Fixed locally; CI pending | `_materialize_image_payloads` now derives page paths, rejects rendered-directory/page symlinks, and writes byte payloads atomically. Tests: `test_materialize_image_payloads_uses_safe_atomic_page_paths`, `test_materialize_image_payloads_rejects_invalid_page_numbers_and_symlinks`. |
| B02 | [ ] Open | Unrepresented non-text OCR units still need explicit retained placeholders or structured rejection and input-unit accounting. |
| B03 | [ ] Open | Cross-page continuation still needs item-level kind, geometry, and text-boundary compatibility rather than page-wide flags alone. |
| B04 | [ ] Open | Summary generation still needs exact owned evidence slices at the provider boundary, not only selected source-item IDs. |
| B05 | [~] Fixed locally; CI pending | `OCRClusterResponse.spatial_accuracy` and normalized `SourceUnit.metadata` now carry an explicit page/region/line/word/character declaration. Unknown or absent precision fails closed to page accuracy; direct OCR and item-level Docling paths declare region precision. Tests: `test_synthetic_page_box_does_not_claim_region_accuracy`, `test_explicit_region_precision_is_preserved`, and the full `tests/test_document_adapters.py` file (18 passed). |
| B06 | [ ] Open | Render and legacy artifact paths still need document/revision collision-resistant namespacing for same-named inputs. |
| B07 | [~] Fixed locally; CI pending | Page numbers are required to be positive and unique, and resolved paths are deterministic. Focused tests are listed under B01. Out-of-order inputs are normalized by sorted output; conflicting duplicates fail closed. |
| B08 | [ ] Open | Default quality policy remains permissive for provider-reported quality; independent completeness and non-text checks remain needed. |
| B09 | [ ] Open | Summary target selection does not yet prove that serialized provider evidence is bounded by the configured token budget. |
| B10 | [~] Partial | Durable artifact IDs and runtime integration exist, but accumulated `raw_pages` checkpoint serialization and linear-growth evidence remain open. |
| B11 | [ ] Open | Render identity does not yet include all rasterizer configuration and output properties such as DPI, rotation, and Poppler identity. |
| B12 | [ ] Open | Reference extraction remains item-local and does not yet map split mentions across ordered OCR units. |
| B13 | [ ] Open | The default PDF path still rasterizes pages; hybrid native PDF text extraction is not implemented. |
| B14 | [~] Fixed locally; CI pending | `_render_pdf_to_images` now rejects page symlinks, validates cached image decoding, and publishes regenerated PNGs through a temporary file and atomic rename. Test: `test_pdf_rasterization_regenerates_corrupt_cached_page_atomically`. Manifest-level page digests and dimensions remain open. |

This ledger is a release gate, not a claim that the branch is complete. The
previous parser matrix run `38809250015` failed its deterministic test step in
all four jobs, so B01/B05/B07 and the broader release remain unverified in CI.
The current parser commit is being checked by run `38811815564`. Kogwistar
typing fixes are on `fix/workflow-run-ownership` at `647d9d9`; its current
required matrix is still running and must complete before downstream pins are
advanced.
