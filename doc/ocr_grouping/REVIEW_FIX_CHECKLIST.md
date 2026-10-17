# OCR Review Fix Checklist

## Current status ledger

The live status is maintained in
[`OCR_STATUS_LEDGER.md`](OCR_STATUS_LEDGER.md). This document is the fixed
requirements and review record; do not append progress sections here.

This table is the only active checklist summary. Update these existing rows in
place; do not append a new requirement or a new progress row after a test run.
The detailed requirement rows later in this file remain the source of the
individual acceptance criteria.

| Track | Status | Release-blocking remainder |
| --- | --- | --- |
| Runtime ownership and persisted stage graph | PARTIAL | Published-core fresh-process lineage, duplicate-resume, and no-double-publication evidence are exercised locally, and parser CI run `38993433709` passed the supported matrix for commit `efccca0`. Linux SIGKILL, orphan-free graph, and no-double-publication evidence across the full release fixture remain open. |
| Artifact and checkpoint crash consistency | PARTIAL | Reconcile both artifact-first and checkpoint-first crash orders across SQLite and Kogwistar runtime state. |
| Grouping and summary durability | PARTIAL | Audit every writer, prove monotonic generation/frontier behavior, and prove bounded long-document work. |
| Semantic hierarchy publication | PARTIAL | Prove the accepted grouping hierarchy is preserved through the persisted semantic graph/export path. |
| OCR completeness and positional fidelity | PARTIAL | Provider precision and complete rejected-unit diagnostics are enforced; live non-text completeness and table/figure evidence remain open. |
| Typing and interpreter matrix | PARTIAL | Pyright is now an explicit CPython 3.12 CI gate; promote it only after the remote package check passes, then retain the CPython/PyPy matrix evidence. |
| Release | OPEN | Do not bump the parser version or downstream pins until the release-blocking rows are green with evidence. |

The status history is the Git history. For example:

```text
git log --oneline --follow -- doc/ocr_grouping/REVIEW_FIX_CHECKLIST.md
git show <commit> -- doc/ocr_grouping/REVIEW_FIX_CHECKLIST.md
```

## Checklist maintenance rule

This file is a fixed requirements ledger, not an append-only work journal.
Update the existing A/S/C/P status row and its evidence in place when a finding
changes. Do not add a new checklist item merely because another test was run;
the commit history and diff show historical progress. Keep a short dated note
only when it records a materially different verification environment or a
regression that cannot be represented by the existing row.

The dated sections below are frozen historical context. They are not additional
requirements and must not be extended. If new evidence changes a status, edit
`OCR_STATUS_LEDGER.md` and let the commit diff record the change.

## Historical evidence log (retained for context; do not append)

The focused fresh-process ownership regression
`test_ocr_root_concurrent_resume_has_one_owner_and_one_result` passed locally
in `91.18s` with two competing resume processes and exactly one accepted
result. The complete repository-local CI selection then passed with
`346 passed, 143 deselected, 48 warnings` in `12m47s`, including fresh-process
OCR recovery, grouping recovery, and concurrent resume. This promotes the
local recovery gate; remote CI and the remaining semantic-publication,
OCR-completeness, typing, and release gates remain open.

## Remote CI promotion (2026-10-17)

Parser CI run `38948398992` passed on commit `b93351d` for CPython 3.12,
CPython 3.13, CPython 3.14, and PyPy 3.11. The fresh-process recovery,
grouping recovery, and concurrent-resume evidence is therefore verified by
the repository matrix as well as locally. Semantic publication, dual-store
crash-order reconciliation, OCR completeness, package-wide typing, and
release gates remain open.

## Latest remote CI promotion (2026-10-17)

Parser CI run `38949951639` passed on commit `cbd9f15` for CPython 3.12,
CPython 3.13, CPython 3.14, and PyPy 3.11. The explicit ambiguous-provider-
ordering regression and the selected parser matrix are therefore green on all
supported interpreters. Semantic publication, dual-store crash-order
reconciliation, OCR completeness, package-wide typing, and release gates
remain open.

## Latest evidence update (2026-10-17, commit `a0332d1`, pending CI)

Proposal/review cache namespaces now accept only JSON-safe `JsonValue` values,
and the external provider boundary is explicitly fail-closed. Parser-core and
layerwise protocol tests pass locally (`29 passed, 6 skipped`); Ruff and
`git diff --check` pass. The branch was pushed, but remote CI could not be
queried because the local environment has no `gh` executable, so this remains
pending CI evidence.

## Latest evidence update (2026-10-17, commit `c36613e`, pending CI)

The OCR runtime state boundary now validates nested `grouping_passes` as
`list[list[JsonObject]]` instead of allowing malformed lists or scalar entries
to reach grouping conversion. The focused regression passes (`1 passed`), Ruff
and `git diff --check` pass, and branch CI run `38941361615` is in progress.
This is a verified local seam fix, not yet a remote-CI promotion.

## Latest evidence update (2026-10-17, commits `5e0b133`, `39ab248`, `dd371b3`, pending CI)

Workflow command `extra` results and cache namespace/request fingerprints now
use explicit JSON-safe contracts. Semantic graph export normalizes opaque
construction-time metadata through `json_safe` before emitting typed node and
edge records. CLI runner tests pass locally (`11 passed`), cache/fixture tests
pass (`56 passed, 4 skipped`), and pointer/legacy/fixture tests pass (`65
passed`); Ruff and `git diff --check` pass. Remote CI and the broader
recovery/release gates remain pending.

## CI promotion update (2026-10-17)

Parser CI passed all four supported interpreter jobs for `87c0074` in run
`38930968502`, `ebfb726` in `38931695531`, `d588b0d` in `38932121311`, and
`bfc4535` plus checklist commit `a59c252` in `38932502441`. Earlier pending
seams are also verified: `ed4ac40` passed `38930015792` and `9e0631a` passed
`38930317634`. These results promote only the changed typing/serialization
seams; recovery, semantic publication, OCR completeness, and release gates
remain open.

## Latest evidence update (2026-10-17, pending CI)

Commits `e9ca370` and `e62c68a` remove remaining PageIndex helper return
annotations widened to `dict[str, object]` and constrain node metadata to the
JSON contract. PageIndex passed locally (`71 passed, 8 skipped`) plus two
focused metadata tests; Ruff and `git diff --check` also pass. This remains
local seam evidence, not full Pyright or runtime-release evidence.

## Latest evidence update (2026-10-17, pending CI)

Commits `bff3f70`, `621acdf`, and `77913f1` remove additional internal
`dict[str, object]` widening from source-map excerpts, layer-child grouping,
and boundary assembly. Focused tests, Ruff, and `git diff --check` pass
locally; this is seam evidence only and does not close full Pyright, recovery,
or release verification.

## Latest evidence update (2026-10-17, pending CI)

Commit `37ed1e3` adds a regression-safe optional `skipped` field for normal
coverage segments, preventing a `KeyError` while keeping gap detection typed.
The named coverage regression and full layerwise suite pass locally (`45
passed`), with Ruff and `git diff --check` clean. Remote CI remains unverified
because the Actions API returns HTTP 403.

## Latest evidence update (2026-10-17, pending CI)

Commits `29a36f8` and `99d62d2` close two additional layerwise typing seams:
boundary refinement/review diagnostics now have dedicated `TypedDict` records,
and retry context is read directly from the typed `ParseSessionState` rather
than reconstructed through `dict[str, object]`. Ruff, `git diff --check`, and
the full layerwise suite pass locally (`45 passed`). GitHub Actions remains
pending verification because its API currently returns HTTP 403.

## Latest evidence update (2026-10-17, pending CI)

Layerwise boundary prompt and lookup payloads now use typed records for parent
IDs, source spans, excerpts, and legal cutpoints instead of
`dict[str, object]`. The existing layerwise suite, Ruff, and the diff check
pass locally; arbitrary provider responses remain isolated at the external
structured-output boundary.

## Latest evidence update (2026-10-17, pending CI)

PageIndex assignment and block-spec debug payload builders now preserve typed
`JsonObject` output instead of widening deterministic trace records to
`dict[str, object]`. Ruff, the PageIndex suite, and the diff check pass
locally; no parser behavior changed.

## Latest evidence update (2026-10-17, pending CI)

PageIndex coverage preserves its existing `dict[str, JsonValue]` contract from
calculation through the public parse result; an unnecessary widening cast was
removed. Ruff, the PageIndex suite, and the diff check pass locally. This does
not close the broader recovery or release gates.

## Latest evidence update (2026-10-17, pending CI)

OCR runtime page-source decoding now uses one typed `_PageSourceState`
decoder for both persisted-state validation and execution. Regression coverage
rejects malformed page numbers and paths (`4 passed`); this narrows a resume
boundary without changing accepted workflow semantics. CI and broader recovery
gates remain pending.

## Latest evidence update (2026-10-17, pending CI)

The canonical graph client keeps validated graph records typed as `JsonObject`
through ID extraction instead of widening them to `Mapping[str, object]`. The
client/demo persistence suite passes (`28 passed, 5 skipped`); this is a typing
only seam and does not change graph payload behavior.

## Latest evidence update (2026-10-17, pending CI)

PageIndex diagnostics and salvage results now use the typed JSON-value alias
`PageIndexDiagnostics` instead of `dict[str, object]` at the validated output
boundary. Ruff, the PageIndex test command, and the diff check pass locally;
CI evidence for this slice is pending and the broader runtime/release gates
remain open.

## Latest evidence update (2026-10-17, pending CI)

Provider-call diagnostics now have one shared scalar-value contract across OCR,
layerwise, PageIndex, and strategy boundaries. The regression
`test_provider_diagnostics_are_json_safe_scalar_records` verifies that nested
provider metadata cannot leak opaque objects into persisted/event diagnostics.
Provider-focused tests and Ruff pass locally; package-wide Pyright remains
blocked by the known Windows Node launcher `EPERM` issue, and CI for this slice
is pending. Recovery, semantic-publication, OCR-completeness, and release gates
remain open.

## Latest evidence update (2026-10-17, pending CI)

OCR page payloads reconstructed by the resume, duplicate-delivery, and
accepted-page paths now pass through `_validated_ocr_page_json` instead of
being asserted with an unchecked `cast`. This keeps the persisted page shape
validated at each reconstruction boundary. The focused OCR boundary and
resume tests pass (`7 passed`), Ruff and `git diff --check` pass; package-wide
Pyright remains subject to the known Windows Node launcher `EPERM` issue. The
full long-running OCR selection was not claimed as green because an existing
grouping process-termination test failed and the 100-page case was stopped
after reproducing its long runtime. Remote CI for the resulting commit is
still pending.

## Latest evidence update (2026-10-17, commit `bfc4535`, CI verified)

Probe event serialization now uses the shared `ModelDumpLike` protocol and
typed recursive JSON traversal rather than dynamic `getattr`. A regression
covers nested model-dump payloads and preserves the backend serialization
mode. Changed-module Pyright and Ruff pass, the focused test passes (`1
passed`), and the diff check passes. Parser CI `38932502441` passed all four
supported interpreters; the recovery, semantic-publication, OCR-completeness,
and release gates remain open.

## Latest evidence update (2026-10-17, commit `d588b0d`, CI verified)

Resume identity comparisons now pass through a typed key/value helper. A
malformed `ocr_configuration_fingerprints` list is rejected rather than
silently widening through `object`; the focused regression passes (`1 passed,
95 deselected`), changed-module Pyright and Ruff pass, and the diff check
passes. The package-wide Pyright attempt was blocked by the known Windows
Node launcher `EPERM` path-resolution failure, so no package-wide claim is
made. Parser CI `38932121311` passed all four interpreters; the
recovery/release gates remain open.

## Latest evidence update (2026-10-17, commit `ebfb726`, CI verified)

The recursive JSON serialization alias now uses named
`typing_extensions.TypeAliasType`. This avoids the Pydantic recursion caused
by a plain recursive `TypeAlias`, while remaining valid for PyPy 3.11 and
CPython 3.12-3.14. The dedicated serialization regression passes (`1 passed`),
full parser-package Pyright reports `0 errors`, changed-file Ruff passes, and
the diff check passes. Commit `87c0074` is pushed; Parser CI run
`38930968502` remains in progress. This is seam evidence only and does not
close the recovery, semantic-publication, OCR-completeness, or release gates.

## Latest evidence update (2026-10-16, commit `1908772`)

The OCR SDK content boundary now has an explicit runtime-checkable Protocol.
Regression coverage proves that SDK-like content is unwrapped while mappings
and `None` are not coerced or reinterpreted. Two focused tests and the
CI-relevant Ruff subset passed locally; Parser CI runs `#309` and `#310`
passed CPython 3.12, 3.13, 3.14, and PyPy 3.11. Full-module Ruff findings in
the legacy OCR module remain open and are not represented as fixed by this
seam change. Broader payload validation, recovery, and release gates remain
partial or open.

## Latest evidence update (2026-10-16, commit `2ec61bd`)

Expanded concurrent-resume regression coverage to distinguish ordinary nested
failures from ownership loss. Invalid grouping frontiers, provider request
failures, and malformed persisted checkpoints are now explicit negative cases,
while direct and wrapped ownership-loss forms remain positive cases. Focused
coverage passed 5 tests, and Parser CI run `#308` (`38906216334`) passed
CPython 3.12, 3.13, 3.14, and PyPy 3.11. This is narrow error-classification
evidence; true process recovery, crash-order reconciliation, OCR completeness,
long-document equivalence, and release readiness remain partial or open.

## Latest evidence update (2026-10-16, commit `aa661f2`)

Persisted OCR page JSON is now validated before it is narrowed to the typed
page contract. The loader preserves legacy artifacts without an embedded page
number by deriving it from the authoritative artifact key, while malformed
checkpoint page numbers fail closed. The OCR/grouping subset passed 15 tests,
and Parser CI run `#306` (`38903836991`) passed CPython 3.12, 3.13, 3.14, and
PyPy 3.11. This closes one persisted-state typing seam only; deeper payload
validation, crash-order reconciliation, OCR completeness, long-document
equivalence, and release readiness remain partial or open.

## Latest evidence update (2026-10-16, commit `520675c`)

Added a negative concurrent-resume regression: an unrelated nested workflow
failure is not accepted as evidence that a worker lost ownership. The direct
ownership and exact wrapped grouping-failure forms from the CI failure remain
covered. Parser CI run `#304` (`38902399870`) passed CPython 3.12, 3.13, 3.14,
and PyPy 3.11. This remains narrow evidence; crash-order reconciliation, true
process recovery, OCR completeness, long-document equivalence, and release
readiness remain partial or open.

## Latest evidence update (2026-10-16, commit `4009bc9`)

Added reusable parametrized coverage for the two valid concurrent-resume
loser errors: direct ownership loss and a nested grouping failure wrapped by
the root workflow. The long subprocess race test now uses that same assertion.
Parser CI run `#300` (`38898230216`) passed CPython 3.12, 3.13, 3.14, and
PyPy 3.11. This is narrow regression evidence only; crash-order reconciliation,
true process recovery, OCR completeness, long-document equivalence, and release
readiness remain partial or open.

## Latest evidence update (2026-10-16, commits `94bc900` and `22603c3`)

Added a reusable regression for failed nested workflow results: the grouping
wrapper preserves the failure diagnostic and never returns a tree. Strategy
predicate state decoding now validates known persisted field types rather than
using blind casts, including IDs, booleans, strategy lists, and child records.
Parser CI run `#297` (`38896087720`) passed CPython 3.12, 3.13, 3.14, and PyPy
3.11. This is seam evidence only; crash-order recovery, OCR completeness,
long-document equivalence, and release readiness remain partial or open.

## Latest evidence update (2026-10-16, commits `a438cdb` and `f56ddc5`)

Page-index refinement now fails closed when a proposed terminal leaf excerpt
is a strict substring of the authoritative leaf span. A regression test proves
that dropped punctuation does not reduce terminal coverage and that the exact
source text remains persisted. Callback accounting and model serialization
also use typed protocols instead of untyped attribute probes. The full
page-index suite passes (`71 passed, 8 skipped`) and provider/cache coverage
passes (`21 passed`).

Parser CI run `#295` (`38893492822`) passed CPython 3.12, 3.13, 3.14, and
PyPy 3.11. This is verified evidence for the changed seams only; crash-order
reconciliation, Linux SIGKILL recovery, OCR completeness, long-document
equivalence, and release readiness remain partial or open.

## Latest evidence update (2026-10-16, commit `35a0383`)

The concurrent fresh-process resume regression now recognizes both valid
fail-closed ownership-loss paths: direct `WorkflowRunOwnershipError` and a
nested grouping failure after lease loss. The exact test passed locally in
95.16 seconds. Parser CI run `#292` (`38890341276`) passed CPython 3.12,
3.13, 3.14, and PyPy 3.11. This is a test-diagnostic correction and does not
close the remaining crash-order, Linux SIGKILL, OCR completeness, or
long-document evidence gates.

## Latest evidence update (2026-10-16, commit `77e7fd5`)

The OCR runtime state-update seam now preserves the concrete `StateUpdate`
shape when applying a typed checkpoint repair. This prevents the operation tag
from widening to an arbitrary `str` at the runtime boundary. The focused
Pyright scan for `ocr_runtime.py`, `handlers.py`, `clients.py`, and
`demo_harness.py` reports zero errors; Ruff also passes for the changed
runtime/demo files.

The focused OCR recovery run passed six tests, including grouping frontier
mismatch rejection, grouping resume after process termination, incremental
resume equivalence, persisted grouping integration, OCR runtime state-boundary
validation, and cross-page checkpoint preparation. Parser CI run `#282`
(`38885261941`) passed. Run `#283` (`38885766112`), triggered by `77e7fd5`,
is still in progress, so the full branch verification remains pending.

Commit `e7139cf` also replaces the direct OCR polygon decoder's untyped
sequence traversal and `type: ignore` with explicit nested-sequence and
boolean-coordinate validation. The provider suite passes `19 passed, 1
skipped`, including `test_direct_rapidocr_adapter_rejects_boolean_polygon_coordinates`.
Parser CI run `#285` (`38886515971`) is pending for this change; do not promote
B05 or any release gate based on the local provider result alone.

This evidence does not close the release gates: the package-wide Pyright
command is not currently usable in this Windows checkout because Node fails
with `EPERM` while resolving the workspace path, and dual-store reconciliation,
Linux SIGKILL recovery, OCR completeness, and the remaining end-to-end gates
remain partial or open below.

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
  `test_ocr_root_fresh_process_resume_after_page_63` now passes in 471.12s on
  a 100-page document, with pages 1-63 not replayed and one accepted grouping
  result. Coordinated source-checkout tests cover process termination and
  concurrent ownership fencing; installed-package, Linux SIGKILL, and full
  graph publication verification remain open. A separate installed Kogwistar
  `0.6.12` reopen of the persisted 100-page state succeeds with zero provider
  calls and the same run ID; the initial run still used the coordinated core
  source checkout.
- [~] The installed released Kogwistar version supports retrying a durably
  failed child; the parser does not substitute a private child-ID workaround.
  Installed Kogwistar `0.6.12` passes
  `test_failed_ocr_child_retries_from_kogwistar_checkpoint_without_replaying_pages`:
  page one is not replayed, page two is retried from its checkpoint, and the
  parent retains one child lineage edge. Supported-interpreter remote evidence
  remains pending.

### B. Strict seam validation

- [~] S01-S02 source/render/provider/config/artifact identities are persisted,
  checked on reuse, and protected by atomic single-file publication.
- [~] S02 has tamper, stale-config, orphan-adoption, and late-writer tests;
  two-store crash-order reconciliation remains open.
- [~] S03-S04 preserve declared spatial precision, media provenance, and
  deterministic mixed-object ordering; mixed pages without declared ordering
  now record `ordering_status="ambiguous_provider_order"` instead of silently
  implying text-before-non-text order. Live multi-object provider evidence and
  caption permutation coverage are still required.
- [~] S05 proves accepted hierarchy reaches semantic parsing and graph export,
  rather than surviving only as ignored metadata. The export metamorphic test
  now proves parent-child structure is consumed; persisted semantic-parser
  publication evidence remains open.
- [~] S06-S08 protect grouping checkpoints, preserve atomic objects, and
  invalidate summaries; S08's final-group identity loss is fixed and covered,
  while monotonic generation and the complete writer audit remain open.
- [~] S09-S10 verify runtime checkpoint ownership and collision-safe identity
  separation for same-stem, changed-path, and changed-content inputs.
- [~] S11-S12 cover cross-page references, preserve references outside the
  primary coverage partition, and reject conflicting public versus
  internal policy values. Multiple reference slices now round-trip and validate
  at the document-model boundary, and contradictory public/runtime parse
  strategies now fail closed with focused contract tests. Custom-policy
  persistence is covered by
  `test_custom_grouping_policy_reaches_checkpoint_cache_and_export`; downstream
  export and cross-page consumer evidence remain open.
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
| Artifact and SQLite/Kogwistar checkpoint commits are crash-consistent | PARTIAL | `test_ocr_page_artifact_publish_crash_is_recoverable` injects a crash in the real `_process_ocr_page` path after atomic rename and before the SQLite completion update, then reopens SQLite and adopts only the source/config-matching artifact. `test_prepare_ocr_workflow_retries_checkpoint_without_artifact` proves the public preparation path reruns a database-first incomplete page. `test_ocr_root_resume_replays_checkpointed_page_when_artifact_is_missing` proves a fresh root resume also rejects a runtime checkpoint whose accepted page artifact is missing. Full root/runtime checkpoint reconciliation remains open. |
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
| S05 accepted hierarchy consumption | PARTIAL | The semantic callback receives and consumes the accepted `document_tree` to derive semantic output, `WorkflowExportBundle.grouping_tree` preserves it, and the OCR root rejects a missing or changed grouping artifact. Regression: `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_workflow_resumes_cached_pages_and_terminal_semantic_run`, which also verifies unique initial/recovery `wf_invoked` edges for page, grouping, and semantic child runs. | Explicit compatibility behavior for consumers that ignore hierarchy remains open. |
| S06 grouping checkpoint safety | PARTIAL | Expected-revision/absence fencing, frontier consistency checks, stable proposal identities, and `test_ocr_state_store_round_trips_grouped_tree_and_invalidates_changed_source` now reject unfenced updates to an existing checkpoint; the same transaction now persists source/policy identity, accepted page frontier, tree digest, and last runtime step in `grouping_checkpoint_metadata`; duplicate accepted proposals are idempotent. | Every writer must be audited; late-writer, monotonic generation, and crash/restart behavior remain release blockers. |
| Runtime ownership | PARTIAL | Stable root/page/grouping/semantic child identities and persisted CAS ownership leases are exercised by the installed Kogwistar `0.6.12` environment. `test_ocr_root_concurrent_resume_has_one_owner_and_one_result` and `test_grouping_checkpoint_reopens_after_process_termination` pass locally together, as do the broader fresh-process recovery checks. `test_ocr_workflow_resumes_cached_pages_and_terminal_semantic_run` additionally checks unique persisted `wf_invoked` edges across the initial and recovery root incarnations, and `test_direct_graph_publication_is_idempotent_for_replayed_bundle` verifies that replaying an accepted bundle writes no second canonical node or edge set. Parser CI run `38993433709` passed the supported interpreter matrix for `efccca0`. | Linux SIGKILL, orphan-free graph, and full release-fixture evidence remain open. |
| 100-page resume | PARTIAL | `test_ocr_root_fresh_process_resume_after_page_63` passes locally in 471.12s using the coordinated parser/core source checkout: it terminates after page 64 starts, resumes the same run ID, avoids replaying pages 1-63, and accepts one grouping result. A separate installed Kogwistar `0.6.12` reopen of that persisted state succeeds in 177.01s with zero provider calls, the same run ID, and a grouping tree. The persisted reference/resume comparison also passes. | A full 100-page initial run through the installed release, Linux SIGKILL, and complete dual-store reconciliation remain open. |
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
| P1 page artifact/database publication is crash-safe | PARTIAL | The current write path validates JSON, preserves the established `ValueError` contract for invalid page numbers, atomically replaces the artifact through unique same-directory temporary files, records an artifact SHA-256, rejects a well-formed tampered artifact on reuse, and adopts a valid rename-before-DB orphan. The legacy `record_page_completed()` path is also fenced against late writers by source/config identity and pending status, with `test_ocr_page_completion_rejects_a_late_writer_after_acceptance`. `test_ocr_two_store_recovery_predicates_hold_in_a_fresh_process` proves the direct restart predicates, `test_ocr_page_artifact_publish_crash_is_recoverable` injects the crash at the real publication boundary, and `test_duplicate_ocr_page_workers_accept_only_one_artifact` proves duplicate page delivery cannot replace the accepted artifact. These are still separate sub-properties, not crash-safe publication as a whole: checkpoint-before-artifact and full runtime/checkpoint reconciliation remain before PASS. |
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
- [x] Fence duplicate accepted summary delivery. The summary journal now uses
  an immediate transaction to return the existing accepted attempt for the same
  `(document_id, node_id, input_fingerprint)` instead of appending a second
  accepted row. Regression:
  `tests/test_workflow_ingest_ocr_pipeline.py::test_duplicate_accepted_summary_delivery_is_idempotent`.
  This proves accepted-result idempotence for the summary journal; it does not
  close the broader root publication or dual-store crash gates.
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
  `test_grounded_summary_generator_receives_only_owned_source_slices` now also
  runs two distinct grounded generator callables against the same persisted
  state and proves the second implementation cannot reuse the first accepted
  result. Callers must change the fingerprint whenever output-affecting
  generation configuration changes.
- [~] Run the full parser CI matrix after the P0 runtime work. Parser CI run
  `38948398992` passed CPython 3.12-3.14 and PyPy 3.11 against the current
  `kogwistar` `0.6.12` source profile. Slow/manual provider experiments,
  published-package verification, and the remaining release gates remain
  separate.

## Reviewer-required acceptance additions

These require an implementation location, an exact pytest node, before/after
behavior, and a runtime classification. C01 was previously evidenced, but the
current released-package check found a Kogwistar `0.6.9` public-version
mismatch. The current parser matrix run `38948398992` is green against the
`0.6.12` source profile; C02-C06 have partial implementation
evidence, but none is a release-gate pass yet:

- [~] Verify the installed Kogwistar package and public workflow persistence
  API, rather than relying on a source checkout or import-only probe. The
  earlier `0.6.9` public-version mismatch is covered by the current `0.6.12`
  source-profile import and the green CPython 3.12-3.14/PyPy 3.11 matrix in
  run `38948398992`; final published-package and release evidence remains
  required.

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

### Concurrent-resume CI follow-up (2026-10-16)

- [x] Reproduced the reported `test_ocr_root_concurrent_resume_has_one_owner_and_one_result`
  locally twice with the locked `kogwistar 0.6.12`; both runs passed in a fresh
  process race.
- [x] Confirmed the ordinary frontier-mismatch guard also passes locally.
- [x] CI once reported the losing worker as a generic nested grouping failure
  (`Nested workflow ... returned status 'failure'`) instead of exposing
  `WorkflowRunOwnershipError`. The current lockfile and published
  `kogwistar 0.6.12` were rerun in Parser CI `#286` and passed; the ownership
  assertion remains strict. If the older symptom recurs, capture the child
  runtime error and fix propagation at its source rather than weakening this
  test.

Source: `kg_doc_parser_ocr_broad_audit_2026-10-15.md`. The source review was
static and did not prove runtime behavior. The statuses below are deliberately
conservative and apply to the current parser branch HEAD `e7139cf`. A row is
not release-complete until its
focused test, full matrix, and exact HEAD are verified.

| ID | Status | Current evidence or remaining work |
| --- | --- | --- |
| B01 | [~] Fixed locally; CI pending | `_materialize_image_payloads` now derives page paths, rejects rendered-directory/page symlinks, and writes byte payloads atomically. Tests: `test_materialize_image_payloads_uses_safe_atomic_page_paths`, `test_materialize_image_payloads_rejects_invalid_page_numbers_and_symlinks`. |
| B02 | [~] Fixed locally; CI pending | `build_document_tree` now raises typed `UnrepresentedSourceUnitError` with a structured diagnostic instead of silently dropping a non-text unit that lacks a description and verified content-addressed media. Blank pages with no units remain empty. Regression: `test_build_document_tree_rejects_unrepresented_nontext_unit`; `tests/test_document_adapters.py` passes 19 tests locally. Input-unit accounting and full CI remain open. |
| B03 | [~] Fixed locally; CI pending | `suggest_cross_page_continuations` now requires both frontier nodes to be canonical text leaves, so page-level incomplete-text flags cannot join a paragraph to a figure/table/caption description or media item. Regressions: `test_cross_page_continuation_never_merges_text_into_nontext` and `test_cross_page_continuation_suggestion_requires_ocr_boundary_signal`. Geometry and stronger paragraph-boundary evidence remain open. |
| B04 | [~] Fixed locally; CI pending | Added immutable `EvidenceSlice` records and a typed `GroundedSummaryGenerator` boundary. The public preparation, grouping, and root workflows thread this boundary through; `test_grounded_summary_generator_receives_only_owned_source_slices` proves two summaries over one source item receive only their ordered, exact ranges. Legacy `SummaryGenerator` remains available for compatibility; provider adoption and full CI remain open. |
| B05 | [~] Fixed locally; CI pending | `OCRClusterResponse.spatial_accuracy` and normalized `SourceUnit.metadata` now carry an explicit page/region/line/word/character declaration. Unknown or absent precision fails closed to page accuracy; direct OCR and item-level Docling paths declare region precision. The direct polygon decoder now rejects malformed nested sequences and boolean coordinates before constructing region-grounded clusters. Tests: `test_synthetic_page_box_does_not_claim_region_accuracy`, `test_explicit_region_precision_is_preserved`, `test_direct_rapidocr_adapter_rejects_boolean_polygon_coordinates`, and the provider/document-adapter suites. Full CI and live-provider evidence remain open. |
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

### Typed provider metadata seam regression (2026-10-16)

- [~] Fixed locally; CI pending. `document_ingester_logger.py` now narrows
  optional provider `usage_metadata` through `_UsageMetadataCarrier` and
  `_message_usage_metadata()` instead of reading it with an untyped `getattr`
  boundary. Invalid metadata fails closed to an empty mapping.
- Regression:
  `tests/test_document_ingester_logger.py::test_message_usage_metadata_narrows_optional_provider_extension`
  verifies valid SDK-like metadata, ordinary messages, and malformed list
  payloads. The focused test, Ruff CI selection, and `git diff --check` pass
  locally; parser CI runs #311 and #312 remain pending.
  The demo HTTP response adapter now uses the same runtime-checkable protocol
  boundary rather than untyped attribute reads; its malformed-response
  regression is `test_demo_http_response_rejects_objects_without_typed_response_protocol`.

### Typed response and semantic-node seam evidence (2026-10-16)

- [x] Parser CI #316 verified `98d8153`, which narrows LangChain usage/model/generation/response metadata
  through runtime-checkable Protocols. The focused accounting regressions
  cover valid SDK-like payloads and malformed ordinary messages; local Ruff,
  tests, and `git diff --check` passed. Full CI evidence was pending when this
  ledger entry was written.
- [x] Parser CI #316 verified `86a7179`, which adds the shared
  `sdk_boundary.raw_content()` contract and
  routes OCR/pipeline content extraction through it. The regression confirms
  mappings are not mistaken for SDK message carriers and empty content is
  preserved. Local focused tests and lint checks passed; full CI remained
  passed in the full run.
- [x] Parser CI #316 verified `2780377`, which makes semantic tree/integrity node shapes explicit
  runtime-checkable Protocols and removes untyped dynamic attribute access from
  validator helpers. Pointer, layerwise, legacy-invariant, and fixture-payload
  tests passed locally, together with changed-file Pyright, Ruff, and diff
  checks. This is seam evidence, not closure of the repository-wide typing
  backlog or the runtime/recovery gates.
- [~] `91c20db` removes redundant `hasattr` checks from concrete
  `ChatGeneration` paths and fixes logger shutdown/Ruff issues. Focused tests,
  Ruff, and changed-file Pyright pass locally; Parser CI #318 is still running.
- [x] Parser CI #321 verified `c3f3f92`, which adds regression coverage for
  wrapped concurrent-resume failures. Direct ownership loss remains
  distinguishable from fail-closed nested grouping failures, and unrelated
  nested failures are not mislabeled as ownership loss.
- [x] Parser CI #322 (`38913709355`) verified `e130e26` across CPython
  3.12-3.14 and PyPy 3.11. It exposes the `WorkflowProviderSettings` contract
  to the OCR root workflow without adding a runtime dependency cycle.
- [x] Parser CI #329 (`38918021548`) verified `466d70f`, which closes the next
  full-package Pyright seam set: optional review
  sessions now narrow to an explicit empty attempt context, PDF render manifest
  fields are narrowed before use, the terminal-correction Protocol is read-only
  and compatible with the concrete NamedTuple result, and proposal reasoning
  entries use their declared Pydantic type. Full local Pyright is now
  `0 errors, 0 warnings, 0 informations`; focused regressions pass. This is
  seam evidence only and does not close the repository-wide typing backlog or
  runtime/recovery gates.

### Typed CLI namespace seam (2026-10-16)

- [x] Commit `d72682d` replaces six untyped optional CLI `getattr` reads with
  the generic, runtime-checked `_optional_namespace_value` accessor. The
  regression `tests/test_workflow_ingest_cli.py::test_cli_optional_namespace_value_validates_dynamic_namespace_fields`
  covers missing, valid, and wrong-typed values. Focused tests and Ruff pass;
  changed-module Pyright reports zero errors. The preceding parser CI #331
  (`38919154090`) passed all four supported interpreter jobs; CI #332
  (`38919826283`) also passed. This closes only the CLI namespace seam, not the
  repository-wide typing backlog.

### Recovery matrix revalidation (2026-10-16)

- [x] The isolated local recovery matrix passed **11 tests** in 145.04s with
  `-p no:cacheprovider`: concurrent root resume has one owner/result,
  grouping resumes after process termination, fresh-process two-store recovery
  predicates hold, and artifact publication crash recovery adopts only the
  valid artifact. This is real subprocess-backed evidence, but it does not
  close the separate Linux `SIGKILL`, installed-release, or full dual-store
  reconciliation release gates.

### Typed OCR smoke-report seam (2026-10-16)

- [x] Parser CI #335 (`38920780847`) verified commit `2f975b9`. Capability and
  live-result reports now use `JsonObject`/`JsonValue` rather than
  `dict[str, object]`; `test_ocr_provider_smoke_writes_typed_json_report`
  proves the non-live command emits a JSON object with a provider list. This
  closes the report serialization seam only, not provider-output validation or
  live OCR completeness.

### Typed persisted-state and optional-backend seam (2026-10-16)

- [x] Commit `cddb7de` narrows workflow-state object reads to
  `Mapping[str, JsonValue]` after the persisted JSON boundary, and isolates
  the probe-only CLI regression from optional Chroma engine construction.
  Ruff and targeted Pyright report no errors; the focused workflow/CLI set
  passes (`4 passed`). This closes the typed state and test-dependency seams
  only; it does not claim the full repository typing or runtime release gates.

### Typed JSON seam follow-up (2026-10-16)

- [x] Commits `b3a0a59`, `957363c`, `3ea269e`, and `5cffeff` close the
  verified service, OCR resume-identity, workflow-cache, and canonical-payload
  JSON seams. Each changed module passes Pyright and Ruff, with focused
  workflow, resume, cache, and canonical persistence tests passing. This is
  evidence for these boundaries only; the remaining broad typing and release
  gates stay open.

### Shared parser payload contract (2026-10-16)

- [x] Commits `b61f428` and `96fe1bc` align the parser adapter, parser core,
  and PageIndex result contracts on `JsonObject`/`ParserSourceMap`. Changed
  modules pass Pyright, the CI Ruff selection passes, and the focused suite
  passes (`17 passed`). CI for `b61f428` remains in progress; full release
  readiness is not claimed.

### Persisted-state and legacy runner contract (2026-10-16)

- [x] Commit `8b8d3fb` adds fail-closed persisted parser payload, source-map,
  and object-map decoding for invalid keys and entries, with a reusable
  regression test and zero changed-module Pyright errors. This closes the
  decoder boundary only; it does not close the repository-wide typing sweep.
- [x] Commit `9341bd4` aligns the legacy facade, runner result, and graph export
  source-map types using `ParserSourceMap` and read-only mappings. Changed
  modules pass Pyright and the focused legacy workflow selection passes
  (`2 passed`). Full Ruff/CI and release readiness remain open because the
  legacy monolith has unrelated existing Ruff findings.
- [x] Commit `33aa16a` removes the duplicated invariant source-map annotations
  from layerwise pointer and boundary helpers in favor of read-only mapping
  views. Ruff and the focused hydration/boundary suite pass (`34 passed`).
  Local Pyright is currently blocked by the Windows Node launcher path error;
  full CI and release readiness remain open.
- [x] Commit `4008804` makes the proposal retry payload cross the JSON boundary
  through `json_safe` rather than an unsafe `ParserPayload` cast. The changed
  module reports zero Pyright errors, Ruff passes, and the focused proposal /
  boundary suite passes (`23 passed`). CI is queued; release readiness remains
  open.
- [x] Parser CI run `38928577822` verified commit `f0bb1cc`, which makes the client workflow-state boundary normalize through
  `json_safe` and reject a non-object result instead of casting arbitrary mapping
  values to `dict[str, JsonValue]`. Pyright and the CI Ruff selection pass for
  `clients.py`; the focused client/demo contract selection passes (`11 passed,
  4 skipped`). This closes one client serialization seam only; the repository-wide
  typing and release gates remain open. The checklist push `89b8dc3` was also
  verified by Parser CI run `38928761211`.
- [x] Parser CI run `38929473712` verified commit `eb02d6c`, which replaces provider-response `model_dump` discovery with the
  shared runtime-checkable `ModelDumpLike` protocol and strengthens the provider
  regression reads with explicit JSON/diagnostic narrowing. Provider and test
  Pyright reports zero errors, Ruff passes, and the focused selection passes
  (`10 passed`).
- [x] Commit `ed4ac40` replaces the legacy semantic-tree `model_dump` lookup with
  the shared `ModelDumpLike` protocol and adds a dedicated coercion regression.
  Production and dedicated-test Pyright report zero errors, Ruff passes, and the
  regression passes (`1 passed`). Parser CI `38930015792` passed all four
  supported interpreter jobs; the pre-existing broad state-edge test file is
  not treated as clean evidence.
- [x] Commit `9e0631a` narrows document source and summary fingerprint payloads
  through the shared JSON-safe object helper instead of retaining
  `list[dict[str, object]]`. Document-model Pyright and Ruff pass, and the
  fingerprint regressions pass (`2 passed`). Parser CI `38930317634` passed all
  four supported interpreter jobs.
- [x] Commit `017bb13` validates unknown non-runtime OCR checkpoint keys
  recursively through the JSON contract, while preserving Kogwistar's reserved
  underscore-prefixed runtime plumbing values. The new opaque-future-state
  regression and the OCR runtime resume seam pass; changed-file Ruff and
  `git diff --check` pass. Remote push/CI is pending connectivity; this does
  not close the package-wide typing or release gates.
- [x] Commit `3566164` removes unchecked casts from persisted handler object
  maps, object lists, and mappings by using the shared recursive JSON sum type.
  Nested opaque-value regressions pass with the handler contract selection
  (`3 passed`); changed-file Ruff and `git diff --check` pass. Full Pyright and
  remote CI remain pending.
- [x] Commit `f2dc457` applies the same recursive JSON boundary to grouping
  metadata during graph export, replacing the unchecked `document_tree` cast.
  The export-boundary regression and adjacent persisted-payload tests pass
  (`3 passed`); changed-file Ruff and `git diff --check` pass. Full Pyright,
  remote CI, and release readiness remain open.
- [x] The legacy OCR page loader now uses the shared recursive JSON decoder
  instead of casting `json.load()` output, and `regen_page()` relies on the
  typed model return values. The malformed-root regression plus adjacent
  persisted/export boundary tests pass (`3 passed`); changed-file Ruff passes.
  Full Pyright, remote CI, and release readiness remain open.
- [x] The legacy structured OCR response envelope now performs runtime mapping
  and model checks before parsed values are consumed, replacing direct casts in
  the retry path. The malformed-envelope, malformed-page, and persisted-payload
  regressions pass (`3 passed`); changed-file Ruff and `git diff --check` pass.
  Full Pyright, remote CI, and release readiness remain open.
- [x] Legacy graph reconstruction helpers now validate mapping keys and
  materialize typed string-key mappings instead of returning unchecked casts.
  The malformed-key helper regressions pass (`3 passed`); Ruff passes on the
  changed test surface. The legacy monolith still has pre-existing `E402`
  import-order debt, so full module Ruff, Pyright, remote CI, and release
  readiness remain open.
- [x] The legacy semantic node-type decoder now narrows its finite `Literal`
  union through explicit branches instead of a cast. Accepted and unsupported
  value regressions pass (`2 passed`), and the changed module has no `F` lint
  findings. Pre-existing `E402`, full Pyright, remote CI, and release
  readiness remain open.
