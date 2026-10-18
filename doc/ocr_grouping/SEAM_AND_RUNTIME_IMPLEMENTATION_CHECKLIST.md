# OCR Runtime And Seam Closure Checklist

## Current status ledger

The live status is maintained in
[`OCR_STATUS_LEDGER.md`](OCR_STATUS_LEDGER.md). This document is the fixed
requirements and review record; do not append progress sections here.

This is the only active status summary for the implementation. The A, B, S,
and C requirements below are fixed acceptance criteria. Update their existing
rows in place; do not append a new checklist item or a new "latest evidence"
section after a test run.

| Track | Status | Release-blocking remainder |
| --- | --- | --- |
| Track A: persisted Kogwistar runtime integration | PASS LOCALLY AND CI | Recovery-versus-repair review confirms ordinary interruption resumes the same Kogwistar run; terminal-artifact repair alone uses a new root incarnation. `test_ocr_root_fresh_process_resume_after_page_63` passes locally (`1 passed, 1293.43s`) and in Parser CI run `39135756441`: two real process terminations, fresh-process recovery, persistent Chroma reopen, accepted v1 OCR/summary reuse, semantic publication, v1 canonical tree/graph comparison, and revision-two reprocessing of pages 1-100 with digest-matched page-40 artifact verification. Do not add another checkpoint layer. |
| Track B: strict seam validation | PARTIAL (independent visual evidence) | The default non-fake OCR root constructs a structured proposer and grounded summarizer, iterates bounded bottom-up frontier windows, carries modality/location/MIME and available OCR evidence, and passes accepted hierarchy to semantic/export. Deterministic-provider tests cover the workflow. Two controlled pages now have captured local provider results: table page (Codex 16/16 text, 0/1 non-text; direct RapidOCR 15/16 text, 0/1 non-text; Docling+RapidOCR 4/16 text, 1/1 region) and chart page (direct RapidOCR 11/11 text; direct EasyOCR emits 9 detections but only 2/11 match spatially; both direct engines miss the region; Docling+RapidOCR 1/11 text, 1/1 region). These expose complementary omissions but are not representative of a broad corpus. Live-provider grouping quality and corpus-wide completeness remain open. |
| C01-C02: API and ownership boundary | PASS (focused matrix) | Installed Kogwistar `0.6.12` exposes the required persisted workflow API; stable root/page/grouping/semantic ownership and concurrent resume fencing are covered by fresh-process tests. Parser CI run `39100319567` passed all four supported interpreter jobs and both focused recovery gates. |
| C03-C05: reconciliation, equivalence, and true resume | PASS LOCALLY AND CI | C03 artifact/checkpoint orderings and C04 bounded-cost incremental grouping remain covered. Expanded 100-page `test_ocr_root_fresh_process_resume_after_page_63` passes locally (`1 passed, 1293.43s`) and in Parser CI run `39135756441`, including two process terminations, v1 same-run recovery and uninterrupted-reference comparison, plus v2 all-page reprocessing and digest-matched changed page-40 verification. OCR payloads are deterministic fixtures, not live OCR. Do not add another parser checkpoint layer. |
| C06: OCR quality/completeness | PARTIAL (independent visual evidence) | Source-item accounting spans raw-text fallback, direct RapidOCR/EasyOCR, native PDF, Docling, and legacy conversion. Malformed EasyOCR detection rows are counted unresolved and strict policy rejects them. The annotated table and chart pages have captured Codex/direct RapidOCR/direct EasyOCR/Docling+RapidOCR comparisons with modality-specific recall. Direct OCR can emit text whose boxes fail spatial matching while missing visual regions; Docling can preserve a visual region while losing text. Ground truth, image hashes/dimensions, provider responses, and expected misses are regression-tested. Two controlled pages do not establish corpus-wide completeness; interval coverage is insufficient evidence. |
| Typing and interpreter matrix | PASS on `b50b0fe` | Parser CI #563 (`39147218276`) passed the Linux recovery and durable graph-publication gates plus CPython 3.12, 3.13, 3.14, and PyPy 3.11. The OCR-ground-truth regression slice is included in this passing run. |
| Release | OPEN | Version bump and downstream pin updates wait for all release-blocking rows. |

The status history is already represented by Git commits and diffs:

```text
git log --oneline --follow -- doc/ocr_grouping/SEAM_AND_RUNTIME_IMPLEMENTATION_CHECKLIST.md
git show <commit> -- doc/ocr_grouping/SEAM_AND_RUNTIME_IMPLEMENTATION_CHECKLIST.md
```

All dated evidence sections below are archival context only. They must not be
counted as extra work items or extended with another progress section.

## Latest evidence update (2026-10-17, local CI selection green)

The focused fresh-process ownership regression
`test_ocr_root_concurrent_resume_has_one_owner_and_one_result` passed locally
in `91.18s` with two competing resume processes and exactly one accepted
result. The complete repository-local CI selection then passed with
`346 passed, 143 deselected, 48 warnings` in `12m47s`, including fresh-process
OCR recovery, grouping recovery, and concurrent resume. This promotes the
local recovery gate; remote CI and the remaining semantic-publication,
OCR-completeness, typing, and release gates remain open.

## Remote CI promotion (2026-10-17)

Parser CI run `38948398992` passed on commit `b93351d` for all supported
jobs: CPython 3.12, CPython 3.13, CPython 3.14, and PyPy 3.11. This promotes
the local recovery and seam test evidence to remote-CI verified. It does not
close the semantic-publication, dual-store crash-order, OCR-completeness,
package-wide typing, or release gates.

## Latest remote CI promotion (2026-10-17)

Parser CI run `38949951639` passed on commit `cbd9f15` for CPython 3.12,
CPython 3.13, CPython 3.14, and PyPy 3.11. The provider-order metadata
regression and the complete selected parser matrix are therefore green on all
supported interpreters. This does not close the semantic-publication,
dual-store crash-order, OCR-completeness, package-wide typing, or release
gates.

## Latest evidence update (2026-10-17, commit `a0332d1`, pending CI)

Parser-core cache namespaces now use the JSON-safe `JsonValue` contract for
both proposal and review cache identities, preventing arbitrary runtime
objects from entering deterministic serialization. The provider exception
boundary is explicitly documented as fail-closed because provider failures
must become structured review failure rather than escape into an accepted
parse. Parser-core and layerwise protocol tests pass locally (`29 passed, 6
skipped`); Ruff and `git diff --check` pass. The feature branch was pushed,
but GitHub CI could not be queried from this environment (`gh` unavailable),
so this evidence is not promoted to remote-CI verified.

## Latest evidence update (2026-10-17, commit `c36613e`, pending CI)

The persisted OCR runtime decoder now validates the declared
`grouping_passes: list[list[JsonObject]]` shape before workflow execution.
Malformed pass containers and non-object proposal entries fail at the state
boundary, with regression coverage in
`test_ocr_runtime_state_boundary_rejects_wrong_known_field_types`. The focused
test passes (`1 passed`), Ruff and `git diff --check` pass, and branch CI run
`38941361615` is in progress. This closes one runtime-state typing seam; the
recovery, semantic-publication, OCR-completeness, and release gates remain
open.

## Latest evidence update (2026-10-17, commits `5e0b133`, `39ab248`, `dd371b3`, pending CI)

Workflow command `extra` results and cache namespace/request fingerprints now
use explicit JSON-safe contracts. Semantic graph export normalizes opaque
construction-time metadata through `json_safe` before emitting typed node and
edge records. CLI runner tests pass locally (`11 passed`), cache/fixture tests
pass (`56 passed, 4 skipped`), and pointer/legacy/fixture tests pass (`65
passed`); Ruff and `git diff --check` pass. Remote CI remains pending, as do
runtime recovery, semantic-publication, OCR-completeness, and release gates.

## CI promotion update (2026-10-17)

Parser CI passed all four supported interpreter jobs for `87c0074` in run
`38930968502`, `ebfb726` in `38931695531`, `d588b0d` in `38932121311`, and
`bfc4535` plus checklist commit `a59c252` in `38932502441`. Earlier pending
seams are also verified: `ed4ac40` passed `38930015792` and `9e0631a` passed
`38930317634`. These results promote only the changed typing/serialization
seams; runtime recovery, semantic hierarchy publication, OCR completeness,
and release gates remain open.

## Latest evidence update (2026-10-17, pending CI)

Commits `e9ca370` and `e62c68a` align PageIndex outline/refinement helpers
with the existing `PageIndexDiagnostics` JSON contract and constrain semantic
node metadata to `JsonObject`. The full PageIndex suite passed locally (`71
passed, 8 skipped`), and focused metadata tests passed (`2 passed`); Ruff and
the diff check are clean. Remote CI and release gates remain pending.

## Latest evidence update (2026-10-17, pending CI)

Commits `bff3f70`, `621acdf`, and `77913f1` narrow source-map excerpts,
layer child candidates, accepted boundary units, and accepted cutpoints to
typed records. The affected focused tests pass locally, as do Ruff and the
diff check. These commits do not change provider or graph behavior; remote CI
and the broader runtime/release gates remain pending.

## Latest evidence update (2026-10-17, pending CI)

Commit `37ed1e3` preserves compatibility for coverage segments that omit the
optional `skipped` flag (the default is `False`) while retaining typed range
records. The reusable regression `test_boundary_parent_coverage_report_marks_gaps_and_spans`
and the complete layerwise suite pass locally (`45 passed`); Ruff and the diff
check also pass. Remote CI remains unverified because the Actions API returns
HTTP 403.

## Latest evidence update (2026-10-17, pending CI)

Commit `29a36f8` narrows boundary refinement prompts and review decision
diagnostics to dedicated `TypedDict` records. Commit `99d62d2` removes the
typed `ParseSessionState` retry-context round trip through `dict[str, object]`
and preserves its strategy, attempt, mode, and review types directly. Ruff,
`git diff --check`, and the full layerwise suite pass locally (`45 passed`).
GitHub Actions status is pending verification because the Actions API currently
returns HTTP 403.

## Latest evidence update (2026-10-17, pending CI)

The layerwise boundary prompt now uses typed `_BoundaryPromptCandidate`,
`_BoundaryLegalCutpoint`, and `_BoundaryLookupCandidate` records instead of
`dict[str, object]` for deterministic parent/span/cutpoint payloads. The
existing layerwise test suite, Ruff, and the diff check pass locally. Provider
responses remain dynamic only at the external structured-output boundary.

## Latest evidence update (2026-10-17, pending CI)

PageIndex assignment and block-spec debug payload builders now return typed
`JsonObject` records, preserving the deterministic JSON contract used by trace
and retry diagnostics. Ruff, the PageIndex suite, and the diff check pass
locally; no parsing behavior changed.

## Latest evidence update (2026-10-17, pending CI)

PageIndex coverage now retains the existing `dict[str, JsonValue]` contract
from `compute_pointer_coverage()` through `PageIndexParseResult`; the prior
publication cast to `dict[str, object]` was removed. Ruff, the PageIndex suite,
and the diff check pass locally. This is a typing-only seam; recovery and
release gates remain open.

## Latest evidence update (2026-10-17, pending CI)

OCR runtime page-source decoding now has one typed `_PageSourceState`
boundary shared by checkpoint validation and page execution. Malformed page
numbers, missing paths, and empty paths are rejected by reusable regression
coverage (`4 passed`), while the existing resume paths remain unchanged.

## Latest evidence update (2026-10-17, pending CI)

The canonical graph client now preserves the validated `JsonObject` record
contract when reading record IDs, rather than widening records back to
`Mapping[str, object]`. The client/demo persistence suite passes (`28 passed,
5 skipped`), with no graph payload behavior change.

## Latest evidence update (2026-10-17, pending CI)

PageIndex diagnostics and salvage results now use `PageIndexDiagnostics`, a
`dict[str, JsonValue]` contract, for known JSON-safe values instead of
`dict[str, object]`. Ruff, the PageIndex test command, and the diff check pass
locally; changed-module Pyright remains unavailable locally because of the
Windows Node launcher `EPERM` issue. CI evidence for this slice is pending.

## Latest evidence update (2026-10-17, pending CI)

Provider-call diagnostics now use a shared scalar-value contract across OCR,
layerwise, PageIndex, and strategy boundaries. A regression confirms nested
provider metadata is reduced to JSON-safe scalar diagnostics rather than
leaking opaque objects (`test_provider_diagnostics_are_json_safe_scalar_records`).
Provider-focused tests and Ruff pass locally; package-wide Pyright is still
blocked by the known Windows Node launcher `EPERM` issue. Parser CI run
`38948398992` now verifies the branch across all four supported interpreters.
This does not close the broader recovery, semantic-publication,
OCR-completeness, or release gates.

## Latest evidence update (2026-10-17, pending CI)

The OCR resume and duplicate-delivery paths now validate reconstructed page
payloads with `_validated_ocr_page_json` rather than using unchecked casts to
`OCRPageJSON`. The focused boundary/resume selection passes (`7 passed`), and
Ruff plus `git diff --check` pass. This closes a local typing/runtime seam but
does not promote the broader recovery, semantic-publication, OCR-completeness,
or release gates. Package-wide Pyright remains locally blocked by the known
Windows Node launcher `EPERM` issue; remote matrix evidence is recorded in
run `38948398992`.

## Latest evidence update (2026-10-17, commit `bfc4535`, CI verified)

The workflow probe now uses the shared runtime-checkable `ModelDumpLike`
protocol instead of dynamic `getattr` discovery. Recursive mapping/sequence
handling preserves the existing backend-dump keyword path for nested models;
the new regression proves the emitted JSON retains the backend representation.
Changed-module Pyright and Ruff pass, the focused test passes (`1 passed`),
and `git diff --check` passes. Parser CI `38932502441` passed all four
supported interpreters; runtime recovery, semantic hierarchy publication, OCR
completeness, and release gates remain open.

## Latest evidence update (2026-10-17, commit `d588b0d`, CI verified)

OCR root/page resume identity reads now use a typed `ResumeIdentityKey` and
runtime-checked `ResumeIdentityValue` helper. Malformed configuration
fingerprint lists fail closed instead of being compared through an ambiguous
`object` value. The dedicated regression passes (`1 passed, 95 deselected`),
changed-module Pyright and Ruff pass, and `git diff --check` passes. A
package-wide Pyright rerun was attempted but hit the known Windows Node
launcher `EPERM` path-resolution failure, so it is not counted as a package
scan result. Parser CI `38932121311` passed all four supported interpreters;
recovery and release gates remain open.

## Latest evidence update (2026-10-17, commit `ebfb726`, CI verified)

The shared recursive JSON contract now uses the named `typing_extensions`
`TypeAliasType` form. A plain recursive `TypeAlias` caused Pydantic model
construction to recurse indefinitely; the named alias preserves runtime schema
support on PyPy 3.11 and CPython 3.12-3.14 while keeping Pyright valid. The
new serialization regression passes (`1 passed`), the full parser-package
Pyright scan reports `0 errors`, the changed-file Ruff selection passes, and
`git diff --check` passes. The related client cast cleanup is in commit
`87c0074`; Parser CI run `38930968502` is still in progress, so this seam is
not yet promoted to CI-verified. Runtime recovery, semantic hierarchy
publication, OCR completeness, and release gates remain open.

## Latest evidence update (2026-10-16, commit `1908772`)

The external OCR response seam now uses a runtime-checkable content protocol
instead of an untyped `getattr`. A regression covers SDK-like response objects,
already-decoded mappings, and `None` without changing the fallback behavior.
Focused coverage passed 2 tests; the CI-relevant Ruff subset (`E4,E7,E9,F`)
and diff check passed. Parser CI runs `#309` and `#310` passed CPython 3.12,
3.13, 3.14, and PyPy 3.11. Full-module Ruff still reports pre-existing
legacy `ocr.py` findings outside this seam; they remain tracked separately.
Broader external payload validation, crash-order reconciliation, OCR
completeness, long-document equivalence, and release gates remain open.

## Latest evidence update (2026-10-16, commit `2ec61bd`)

The concurrent-resume failure classifier now has a parametrized regression
matrix for ordinary nested failures: invalid grouping frontiers, provider
request failures, and malformed persisted checkpoints are all rejected as
ownership-loss evidence. The direct ownership exception and the exact wrapped
ownership-loss form remain accepted. Focused coverage passed 5 tests, and
Parser CI run `#308` (`38906216334`) passed CPython 3.12, 3.13, 3.14, and
PyPy 3.11. This closes only the wrapped-error classification regression;
ownership recovery, dual-store reconciliation, OCR completeness,
long-document equivalence, and release gates remain open.

## Latest evidence update (2026-10-16, commit `aa661f2`)

The persisted OCR page boundary no longer relies on a cast-only narrowing.
Checkpoint payloads and reloaded page artifacts now pass a top-level typed
validator before entering grouping. Legacy artifacts that omit `pdf_page_num`
retain compatibility by receiving the authoritative page number from their
persisted page key; malformed checkpoint values are rejected. Focused
OCR/grouping coverage passed 15 tests, including fresh-process recovery,
artifact crash recovery, late-writer fencing, and duplicate-worker acceptance.
Parser CI run `#306` (`38903836991`) passed CPython 3.12, 3.13, 3.14, and PyPy
3.11. Deeper nested cluster-field validation, dual-store reconciliation, OCR
completeness, long-document equivalence, and release gates remain open.

## Latest evidence update (2026-10-16, commit `520675c`)

The concurrent-resume error classifier now has both positive and negative
regression coverage. It accepts the direct ownership exception and the exact
wrapped nested grouping failure observed when a competing lease is lost, while
rejecting an unrelated nested failure such as an invalid grouping frontier.
The same typed classifier is reused by the subprocess race assertion. Parser
CI run `#304` (`38902399870`) passed CPython 3.12, 3.13, 3.14, and PyPy 3.11.
This closes only the error-classification regression; ownership recovery,
dual-store reconciliation, OCR completeness, long-document equivalence, and
release gates remain open.

## Latest evidence update (2026-10-16, commit `4009bc9`)

The concurrent fresh-process resume regression now has reusable coverage for
both valid fail-closed loser surfaces: a direct `WorkflowRunOwnershipError`
and a wrapped nested grouping failure after the competing lease is lost. The
subprocess race test reuses the same assertion, so an unrelated worker error
cannot be accepted as the expected ownership-loss path. Parser CI run `#300`
(`38898230216`) passed CPython 3.12, 3.13, 3.14, and PyPy 3.11. This closes
only the regression-test coverage item; dual-store reconciliation, Linux
SIGKILL recovery, OCR completeness, long-document equivalence, and release
gates remain open.

## Latest evidence update (2026-10-16, commits `94bc900` and `22603c3`)

The grouping wrapper now has a deterministic regression proving that a failed
nested runtime result remains a failure and preserves its diagnostic instead of
being returned as an accepted tree. Persisted strategy state decoding also
uses typed, runtime-checked views for strategy names, disabled strategies,
parent IDs, review flags, and child parent IDs; malformed values cannot enable
valid routes. Parser CI run `#297` (`38896087720`) passed CPython 3.12, 3.13,
3.14, and PyPy 3.11. This evidence does not close dual-store reconciliation,
Linux SIGKILL recovery, OCR completeness, long-document equivalence, or release
gates.

## Latest evidence update (2026-10-16, commits `a438cdb` and `f56ddc5`)

The page-index callback accounting seam now uses runtime-checkable protocols
instead of untyped callback attribute access. Excerpt refinement now rejects a
strict substring for a terminal leaf, preventing a model from dropping
authoritative punctuation or creating an uncovered source interval. Dedicated
regression coverage and the full page-index module pass (`71 passed, 8
skipped`); provider/cache coverage passes (`21 passed`). The shared
`ModelDumpLike` protocol is also used by JSON serialization and the workflow
cache.

Parser CI run `#295` (`38893492822`) passed all four jobs on CPython 3.12,
3.13, 3.14, and PyPy 3.11. These changes close the page-index terminal
coverage regression and improve typing at two internal seams; they do not
close dual-store reconciliation, Linux SIGKILL recovery, OCR completeness,
long-document frontier equivalence, or the release gates.

## Latest evidence update (2026-10-16, commit `35a0383`)

The concurrent fresh-process resume regression now accepts both runtime
ownership-loss representations: a direct `WorkflowRunOwnershipError` and a
fail-closed nested grouping failure when the competing lease is lost after
the parent has already entered a child workflow. The exact regression passed
locally in 95.16 seconds. Parser CI run `#292` (`38890341276`) passed all four
jobs on CPython 3.12, 3.13, 3.14, and PyPy 3.11. This corrects test
diagnostics only; it does not promote the dual-store, Linux SIGKILL, OCR
completeness, or long-document release gates.

## Latest evidence update (2026-10-16, commit `77e7fd5`)

The typed OCR checkpoint-repair update now constructs an explicit
`StateUpdate` before returning it to Kogwistar. The focused Pyright scan across
the OCR runtime seam (`ocr_runtime.py`, `handlers.py`, `clients.py`, and
`demo_harness.py`) reports zero errors, and the corresponding Ruff checks
pass. Six focused OCR recovery/state tests pass, including frontier mismatch
rejection and fresh-process grouping recovery. CI run `#282`
(`38885261941`) is green; run `#283` (`38885766112`) is still in progress.

No release gate is promoted by this update. Package-wide Pyright is still
unverified in the Windows checkout because the local Node launcher fails with
`EPERM` during workspace path resolution, while dual-store recovery, Linux
SIGKILL recovery, OCR completeness, and long-document evidence remain open.

## Current verification correction (2026-10-16)

PyPI now exposes Kogwistar `0.6.12`, including the lease-aware
`WorkflowRuntime` constructor and retryable-child runtime behavior required by
these tests. The parser manifest and lockfile have been updated from `0.6.10`
to `0.6.12`. Parser CI run `38873761447` passed on the exact branch head
`cb22da2` across CPython 3.12, 3.13, 3.14, and PyPy 3.11. Runs before that
pin remain historical evidence only and must not be used to close the installed
release gate.

Dual-store crash reconciliation, Linux SIGKILL evidence, independent OCR
completeness, and long-document frontier measurements also remain open.

This checklist tracks two related but distinct implementations:

1. **Runtime integration:** OCR, grouping, and summaries execute as persisted
   stages in the existing Kogwistar `WorkflowRuntime`.
2. **Strict seam closure:** source identity, artifact integrity, grounding,
   ordering, hierarchy consumption, summary derivation, and public options are
   proven across the OCR-to-graph boundary.

`[x]` means the implementation and a reusable regression test are present and
the test proves the stated contract on the supported runtimes.
`[~]` means code exists or the risk is partly contained, but the full release
gate or its evidence is still missing.
`[ ]` means not implemented. A passing adapter test alone never closes a
runtime seam.

## Evidence rule

Every row marked `[x]` must have an evidence entry with all of these fields:

| Field | Required evidence |
| --- | --- |
| Implementation | file, function, and stable source location |
| Regression test | exact pytest node ID and the assertion that proves the contract |
| Before fix | reproduced failure or documented missing behavior |
| After fix | passing result and relevant assertion summary |
| Runtime | standalone, mocked Kogwistar runtime, or real persisted runtime |
| Status | verified pass, fixed, failing, not implemented, or not verified |

Do not mark a row `[x]` merely because a helper exists or because a mocked
adapter test passes.

## Execution ledger: two implementation tracks

Use this section as the working checklist while implementing the review. The
runtime track must be completed before the seam track can claim end-to-end
proof, because a seam test that never crosses a persisted runtime boundary is
only an adapter test. Every checked item must add its exact test node and
runtime classification to the evidence ledger below.

### Track A - persisted Kogwistar runtime integration

- [~] **A-1. Root ownership:** one versioned root invocation owns page OCR,
  grouping, semantic parsing, summary generation, and final publication. Keep
  stable parent/child invocation IDs and record the semantic checkpoint and
  graph publication identity. Stable IDs and persisted lineage are enforced;
  `test_direct_graph_publication_is_idempotent_for_replayed_bundle` proves a
  replayed accepted bundle adds no second canonical node or edge set. Fresh-
  process no-double-publication and orphan-free graph proof remain open.
- [~] **A-2. Durable state:** persist only typed JSON state: source revision,
  immutable configuration fingerprint, page cursor, accepted artifact IDs,
  grouping revision/frontier, summary policy revision, and final input
  reference. Reconstruct providers from runtime dependencies, never from a
  checkpoint.
- [~] **A-3. Page recovery:** prove both interruption points: after atomic
  artifact publication and before accepted completion. Recovery must adopt or
  retry safely, never delete accepted work, and accept only one result under
  duplicate delivery.
- [~] **A-4. Grouping recovery:** restore the accepted tree and frontier before
  processing new pages. Every writer must use the same expected-revision or
  monotonic-generation fence; no compatibility writer may use unconditional
  upsert semantics. The recovery-generation audit is complete: ordinary
  interruption uses Kogwistar resume on the same run, while post-terminal
  artifact repair requires a new incarnation because a succeeded run is not
  re-executed. `test_ocr_root_resume_replays_checkpointed_page_when_artifact_is_missing`
  and `test_ocr_workflow_resumes_cached_pages_and_terminal_semantic_run` cover
  the two distinct paths. Do not replace either path with a duplicate
  parser-owned retry mechanism.
- [x] **A-5. Summary recovery:** summaries are derived, revision-fenced, and
  independently retryable. A cache hit must not call the provider or create a
  second accepted journal row; a failed summary must not invalidate accepted
  OCR/grouping state. Grounded summary generation and the separate adaptive
  grouping proposer are both wired into the default non-fake root path. Their
  deterministic workflow contract is tested. The
  `test_ocr_summary_is_not_regenerated_after_fresh_process_resume` proves the
  accepted summary survives termination before semantic publication and is not
  regenerated on recovery; the 100-page durable two-termination gate confirms
  the same behavior after reopening persistent Chroma. Parser CI run
  `39106578551` passed. Live-provider grouping quality remains open under
  Track B.
- [x] **A-6. Process-boundary proof:** terminate a real worker after a page and
  after a grouping/summary checkpoint, reopen the same engines and run ID in a
  new process, and prove one lineage, no duplicate accepted result, and no
  replay of completed work. The installed-package test
  `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_root_fresh_process_resume_after_page_63`
  kills the worker after page 63 and again after accepted summary generation,
  resumes in fresh processes with the same root identity, compares recovered
  tree and canonical graph output with an uninterrupted persistent reference,
  and reopens Chroma to verify persisted node/edge IDs. On POSIX the helper
  uses `SIGKILL`. Parser CI run `39106578551` passed. The separate rich
  semantic-fixture and independent OCR-completeness criteria remain open
  under Track B/C; they do not invalidate this process-recovery result.

**Track A is complete only when:** the test uses the installed Kogwistar
package, survives a real process boundary, compares canonical output with an
uninterrupted reference, and records at-least-once provider execution with
exactly one accepted canonical result. Same-process or source-checkout tests
cannot close A-6.

### Track B - strict seam validation

- [x] **B-1 / S01-S02 identity and artifacts:** render/page reuse includes
  source, renderer, provider, model, engine, policy, schema, and artifact
  digests. The exact reuse path is validated, and artifact-first and
  checkpoint-first crash orders are reconciled.
- [x] **B-2 / S03-S04 grounding and order:** preserve provider-declared
  coordinate precision, media identity, and one traversal covering every
  accepted object exactly once. The OCR response and normalized source-unit
  contracts preserve explicit spatial precision and downgrade missing or
  unknown declarations to page accuracy. Real provider smoke and downstream
  graph-export tests verify the boundary.
- [x] **B-3 / S05 hierarchy consumption:** the graph exporter has a metamorphic
  test with two valid trees over identical source items. The public OCR-root
  test `test_custom_grouping_policy_reaches_checkpoint_cache_and_export` feeds
  accepted grouping sections through the semantic callback and asserts their
  identities, ordered source excerpts, and parent-child graph edges survive
  export. `WorkflowExportBundle.grouping_contract` makes hierarchy optional
  only for compatibility callers and rejects a missing tree when required.
  The 100-page fresh-process test compares recovered hierarchy and canonical
  graph output with an uninterrupted reference after persistent Chroma reopen;
  Parser CI run `39135756441` passed the recovery and durable-publication gates.
  OCR payloads in that gate are deterministic fixtures, not provider accuracy
  evidence.
- [~] **B-4 / S06-S08 durable grouping and summaries:** fence every grouping
  writer, preserve meaningful groups and atomic objects, invalidate summaries
  on grouping changes, and test duplicate/replay behavior. Grouping updates
  require an expected tree revision, summary identity includes exact evidence
  slices and parser configuration, and duplicate/replay tests cover both. The
  SQLite grouping writer now also checks the tree's `ocr-v1:<input fingerprint>`
  against the active document fingerprint inside the write transaction;
  `test_grouping_checkpoint_rejects_stale_writer_after_source_revision_changes`
  proves an old worker cannot recreate a cleared checkpoint after source
  invalidation. `load_document_tree()` now verifies the tree digest, source and
  policy identities, persisted frontier, and companion metadata in one SQLite
  read snapshot; `test_grouping_checkpoint_load_rejects_mismatched_companion_records`
  rejects a stale metadata digest and a frontier that disagrees with the tree.
  The broader writer/generation audit remains open. The
  default non-fake OCR root runs an iterative proposer over bounded,
  multimodal-aware frontier windows, resolves overlaps deterministically, and
  checkpoints per-window proposal progress in Kogwistar. It also constructs
  grounded summaries by default; `test_default_adaptive_root_wires_grounded_summary_and_semantic_export`
  verifies grouped summaries reach semantic parsing and graph export using
  deterministic injected providers. Resume, policy identity, provider
  configuration identity, and completion reuse have focused tests. The
  durable-backend close/reopen test
  `test_ocr_root_graph_publication_survives_durable_engine_reopen` now keeps
  workflow/conversation engines in memory and exercises a real persistent
  Chroma knowledge engine. It passed locally on pinned Kogwistar `0.6.12`
  (`1 passed, 46.53s`) including exact node/edge-set checks after reopen, and
  its dedicated durable-publication job passed in CI run `39096834736`. The
  earlier Chroma CI failure in run `39091524022` came from persisting unrelated
  workflow-design graphs and is superseded.
  Fresh-process publication after recovery remains open and Kogwistar-owned.
- [x] **B-5 / S09-S10 runtime and identity:** verify the persisted root/checkpoint
  boundary and separate document, source revision, OCR configuration, grouping
  revision, and parse-generation identities. Same-stem/cross-path and
  changed-content tests are present.
- [x] **B-6 / S11-S12 references and public options:** multi-slice
  cross-page references round-trip at the document-model boundary. Public and
  runtime policy conflicts fail closed, and explicit policy runs inspect
  persisted metadata, cache keys, and the exported hierarchy.
- [~] **B-7 / C06 quality gate:** require typed provider loss diagnostics and
  strict-policy persistence, then add evidence for missing figures/tables,
  unsupported coordinates, and reopening after a strict rejection. The typed
  `OCRGroundTruth` evaluator now supplies independent item-recall, spatial-IoU,
  and transcription-mismatch evidence in controlled fixtures. Interval
  coverage alone is not sufficient; provider-specific annotated corpora remain
  open. The EasyOCR adapter now accounts malformed detection rows as unresolved,
  with a unit regression proving strict rejection.

**Track B is complete only when:** every S01-S12 row has an implementation
location, a reusable regression test, before/after behavior, and an explicit
runtime classification. A green standalone adapter test cannot close a row
whose contract crosses SQLite, Kogwistar, a provider, or graph export.

### Dependency order

- [x] **D-1.** Keep the review correction ledger authoritative: do not mark
  S02, S06, C02, C03, C04, C05, or C06 green from narrow tests.
- [x] **D-2.** Keep the installed Kogwistar API gate separate from source
  checkout compatibility; record the exact version and CI run.
- [x] **D-3.** Merge and publish the Kogwistar retryable-child and lease-aware
  runtime fixes before claiming Track A-6 or the 100-page release fixture.
  Kogwistar `0.6.12` is visible on PyPI and is now the parser's locked
  dependency. The post-pin parser matrix remains the evidence gate.
- [x] **D-4.** Close Track A process/recovery evidence before claiming Track B
  end-to-end seam closure. The installed-package 100-page, two-termination
  recovery gate passed locally and in Parser CI run `39106578551`; Track B's
  separate rich-fixture and independent-completeness gates remain open.
- [ ] **D-5.** Only after both tracks are green, bump the parser version, add
  release notes, update downstream pins, and publish.

## Audit note (2026-10-15)

### Recovery evidence update (2026-10-16, commit 5a95c19)

- [x] The fresh recovery child now reconciles a valid OCR page artifact before
  the direct reuse predicate. This closes the cursor-zero hole where page one
  could be replayed after an artifact-first interruption even though its JSON
  was valid and source/configuration identities matched.
- [x] Local regression evidence:
  `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_root_concurrent_resume_has_one_owner_and_one_result`
  and
  `tests/test_workflow_ingest_ocr_pipeline.py::test_grouping_checkpoint_reopens_after_process_termination`
  both pass; the broader focused grouping/runtime run passed 6 tests.
- [x] Parser CI run `38913382069` / `#321` verified commit `c3f3f92`.
  The concurrent-resume regression accepts direct ownership loss and stable
  wrapped nested-workflow failures, while rejecting unrelated failures as
  ownership errors. This closes only that error-classification regression.
- [x] Parser CI run `38913709355` / `#322` verified `e130e26` across CPython
  3.12-3.14 and PyPy 3.11. The typed `WorkflowProviderSettings` contract is
  now covered by the full matrix. This update does not close A-6, S02, C03,
  C05, Linux SIGKILL, installed-package, or full graph-publication evidence.

The external seam review found that earlier green marks overstated the
evidence for artifact integrity and runtime ownership. Those marks are
intentionally downgraded below. In particular, an artifact being written with
an atomic replace is not proof that the database/runtime checkpoint and the
artifact commit are crash-consistent, and a stored digest is not proof until
the exact reuse path rejects a well-formed tampered artifact. The release gate
therefore requires both implementation evidence and a reusable test at the
actual boundary.

The reviewer also identified six release gates which must remain visible:

- **C01:** verify the installed Kogwistar API, not only the source checkout;
- **C02:** define the root-versus-child runtime ownership boundary;
- **C03:** reconcile SQLite artifact commits with Kogwistar checkpoints in
  either crash order, then demonstrate durable graph publication and one
  accepted canonical result across restart using existing Kogwistar backend
  guarantees rather than reimplementing them in parser code;
- **C04:** prove incremental and uninterrupted grouping are canonically
  equivalent;
- **C05:** prove true resume without redundant accepted work;
- **C06:** require an explicit OCR completeness/quality gate beyond interval
  coverage, including independent recall and position evidence for visual and
  non-text regions.

The review also corrected two status interpretations that must remain visible:

- A page JSON digest is enforced only when the completed-page reuse predicate
  reads the artifact. Recording a digest, or validating a newly written file,
  is not sufficient evidence for reuse safety.
- Atomic file publication protects the JSON file itself, not the ordering
  between that file and SQLite/Kogwistar checkpoint commits. Both crash orders
  need a recovery test before S02/C03 can become a release pass.

The latest downstream audit reconfirms the same correction: S02's digest and
atomic-file subclaims are useful partial protections, not complete release
gates, and S06's CAS checks do not by themselves prove monotonic generation.
These statements are recorded as `[~]` below rather than `[x]`.

These gates are tracked in section C and the evidence ledger. A focused test
may close a sub-item, but cannot close the parent gate when fresh-process,
SIGKILL, concurrency, or real-provider evidence is still absent.

### Reviewer correction ledger

The following claims are explicitly **not** release-complete, even when an
implementation helper or a narrow unit test is green:

- `S02` artifact SHA-256 verification now has exact reuse-path coverage for a
  well-formed tampered JSON artifact, and provider/configuration invalidation
  now has fresh-process coverage in
  `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_state_rejects_stale_provider_config_or_tampered_artifact`.
  S02 is still not release-complete because the two-store crash-order
  reconciliation proof is missing.
- `S02` atomic JSON publication is partial until both crash orders are
  reconciled: artifact rename before the SQLite/runtime checkpoint, and
  checkpoint publication before artifact rename. A temporary file plus
  `os.replace()` protects one file write, not the two-store commit.
- `S06` grouping checkpoint protection is partial until a monotonic generation
  or CAS guard prevents a late prefix rebuild from overwriting a newer tree.
  The grouping and summary persistence paths now share an atomic
  expected-revision/expected-absence fence; compatibility writers and crash
  recovery still prevent release completion.
- `S09` root-workflow integration is partial until a new process can reopen the
  same parent invocation, recover the same child lineage, and avoid a second
  accepted semantic generation.
- `C04` and `C05` remain open/partial until canonical incremental equivalence,
  SIGKILL/new-process resume, and duplicate-resume behavior are tested.

The external review also adds six release gates which are tracked below. They
are not satisfied by a checklist mark alone:

- `C01` verifies the installed Kogwistar API and version, not only a source
  checkout.
- `C02` defines one persisted root/child ownership boundary and requires a
  fresh process to reopen the same lineage without a second semantic
  generation.
- `C03` reconciles SQLite artifact commits and Kogwistar checkpoints in both
  crash orders.
- `C04` compares canonical incremental and uninterrupted grouping results,
  excluding volatile run metadata.
- `C05` proves at-least-once execution with exactly one accepted canonical
  result after interruption, rather than claiming exactly-once provider calls.
- `C06` requires an explicit OCR quality/completeness gate beyond valid source
  interval coverage.

### Reviewer acceptance-gate correction (2026-10-15)

The following are release-blocking acceptance gates, not optional follow-up
ideas:

- **C01 - Installed API compatibility:** verify the exact installed Kogwistar
  version and public persistence APIs used by the parser. Source-checkout
  compatibility is insufficient.
- **C02 - Runtime ownership boundary:** define whether semantic parsing is a
  stage of the OCR root run or a durable child invocation. Record the parent
  run ID, child invocation ID, checkpoint identity, and graph publication
  identity. A function call that receives a run ID is not, by itself, a
  persisted workflow stage.
- **C03 - Dual-store recovery:** test crashes after artifact publication and
  after SQLite/Kogwistar checkpoint publication. Recovery must reconcile both
  orders without accepting a partial result or losing an accepted result.
- **C04 - Incremental equivalence:** compare canonical tree content while
  excluding volatile timestamps and run metadata. Completed subtrees must stay
  stable and the incremental result must match an uninterrupted reference.
- **C05 - True resume:** prove at-least-once provider execution with exactly
  one accepted canonical result. Completed OCR, grouping, and summary work
  must not be accepted twice after restart; do not claim exactly-once provider
  calls across arbitrary process termination.
  The historical 100-page regression exposed a dependency gate: released
  Kogwistar `0.6.8` rehydrated a durably failed child as terminal instead of
  retrying its checkpoint. Published Kogwistar `0.6.9` contains the
  retryable-child contract; the parser must verify that installed release and
  complete fresh-process recovery before this gate can pass;
  changing the parser's child ID privately would weaken lineage and is not an
  accepted substitute.
- **C06 - OCR quality/completeness:** valid interval coverage is necessary but
  cannot prove that every source region was captured. Require provider-side
  quality/completeness signals, coordinate/media-loss diagnostics, and a
  persisted strict-policy recovery test.

The status words below describe evidence, not optimism: `PASS` means the
release gate is proven; `PARTIAL` means useful containment exists but the
acceptance proof is incomplete; `OPEN` means the required implementation or
test is not yet present.

### Reviewer reconciliation (2026-10-15)

The latest external review is accepted as a status audit, not as proof that
the implementation is complete. The review's two S02 claims marked
"incorrectly complete" are consistent with this ledger: neither artifact
digest enforcement nor atomic JSON publication is a complete release gate.
They are only sub-properties of S02 and remain `[~]` until the cross-store
crash protocol is exercised.

The following findings are explicit release blockers:

- S06 must not be described as monotonic merely because it has an
  expected-revision/absence fence. Any remaining unconditional upsert or
  compatibility writer can still allow an older grouping result to replace a
  newer result. S06 remains `[~]` until every writer is audited and a
  generation/CAS regression covers the late-writer race.
- The root workflow owns page, grouping, and semantic child identities, but a
  same-process test is not a process-recovery proof. C02/S09 remain `[~]`
  until a new process reopens the same parent invocation, recovers the same
  child lineage, and demonstrates that semantic publication is not duplicated.
- The 100-page test is useful evidence only with the retryable-child Kogwistar
  fix supplied. It is not release evidence against the currently released
  Kogwistar package, and it does not replace a Linux `SIGKILL` test or a
  duplicate-resume race test.
- Interval coverage is necessary but cannot establish OCR completeness for
  figures, tables, captions, or unsupported coordinates. C06 remains `[~]`
  until provider-side loss diagnostics and strict-policy recovery are tested.

No item in this document may be promoted to `[x]` from a static code review,
an adapter-only test, or a source-checkout-only runtime test.

## A. Runtime Integration

### A1. Workflow design

- [x] Define one versioned OCR workflow ID in the existing parser workflow
  design, not a second orchestration framework.
- [x] Add graph-defined nodes for page execution, grouping, semantic parsing,
  finalization, and `end`. The root and child designs are persisted and the
  parser recovery matrix inspects their runtime lineage.
- [x] Add explicit default and failure edges for page iteration, retry,
  interruption, completed-page reuse, grouping completion, and summary failure.
- [x] Make the page loop bounded by persisted page numbers and source revision,
  never by an in-memory list that disappears on process exit. The page
  workflow persists `page_sources`, `page_cursor`, and source identity in its
  checkpoint; `_validate_page_resume_identity` rejects a changed source before
  page execution.
- [x] Keep semantic parsing in the existing ingest workflow and connect the
  accepted OCR/grouping output through a typed state boundary.
- [x] Reuse `WorkflowRuntime`, `MappingStepResolver`, runtime predicates,
  persisted step execution records, and checkpoint tokens. Do not create an OCR
  runtime or parallel checkpoint protocol.

### A2. Typed runtime state

- [x] Define typed JSON state for document ID, source revision, page cursor,
  total pages, accepted page artifact IDs, grouping revision, summary policy
  revision, and final workflow-input reference.
- [x] Keep callables, provider clients, API tokens, and open file handles in
  runtime dependencies only; never persist them in checkpoint state.
- [x] Persist enough immutable request configuration to reconstruct the same
  provider/config fingerprint after a new process starts.
- [x] Reject a resume when source revision, OCR configuration, workflow
  version, or grouping policy is incompatible with the saved run. The root
  dispatch fence and
  `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_root_resume_rejects_missing_or_changed_immutable_identity`
  plus
  `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_root_resume_rejects_changed_provider_before_replaying_pages`
  cover the public root boundary and fresh-process checkpoint/reopen behavior.

### A3. Page execution and recovery

- [x] Execute one page as a durable runtime step with a deterministic page key.
  The page child runtime and fresh-process interruption/duplicate-delivery
  tests cover this boundary.
- [x] Publish page JSON atomically before marking the page step complete.
  Atomic publication and the parser-side crash-order matrix cover this
  boundary.
- [x] Record artifact digest, source image digest, provider fingerprint, schema
  version, and attempt index in the durable state.
- [x] Make retry and completed-page reuse idempotent under duplicate delivery.
  Attempt allocation is SQLite-serialized, completed rows are not reopened by
  a racing retry, and `test_duplicate_ocr_page_workers_accept_only_one_artifact`
  proves one accepted OCR attempt and one winning artifact. Full root-level
  duplicate resume is now covered by the process-boundary race test below.
- [x] On process interruption after page commit, resume at the next incomplete
  page without repeating accepted OCR or deleting a complete grouping result.
  `test_ocr_root_reopens_after_real_process_termination` proves this with a
  fresh worker; the remote Linux recovery gate also passes this path.
- [x] On process interruption before page commit, detect the incomplete step and
  retry it safely. Covered by
  `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_root_resume_replays_checkpointed_page_when_artifact_is_missing`
  and `::test_prepare_ocr_workflow_retries_checkpoint_without_artifact`.
- [x] Support concurrent duplicate resume attempts with one accepted commit and
  an explicit stale-token/claim failure for the loser. The reusable
  `test_ocr_root_concurrent_resume_has_one_owner_and_one_result` starts two
  fresh workers against the same persisted root and asserts exit codes `[0, 1]`
  and exactly one accepted result.

### A4. Grouping execution

- [x] Run grouping inside a persisted runtime node after the accepted page
  frontier is available.
- [x] Persist grouping input identity: source revision, accepted page set,
  source-item digest, grouping policy version, and merge revision. The SQLite
  checkpoint now records a monotonic `merge_revision` with a backward-
  compatible migration; idempotent replay preserves the generation. The
  regression `test_ocr_state_store_round_trips_grouped_tree_and_invalidates_changed_source`
  asserts generations 1, 2, and stable replay at 2.
- [x] Restore the latest valid grouping checkpoint before processing new input.
- [x] Use CAS/expected revision or an equivalent monotonic generation guard;
  last-writer-wins must not overwrite a newer accepted tree.
  `save_document_tree()` rejects any update to an existing tree without
  `expected_tree_revision`, while the initial writer may create only from
  absence. Late-writer and conflict regressions cover stale revisions and
  orphaned frontiers.
- [x] Persist and reuse the accepted merge frontier so long documents do not
  rebuild the entire prefix after every page. The root workflow invokes
  grouping after the accepted page frontier is available, and
  `merge_groups()` reifies only the affected proposal window while preserving
  unaffected frontier IDs. `tests/test_document_grouping_contract.py::test_merge_groups_keeps_large_unaffected_frontier_stable`
  covers a 256-item frontier; the persisted frontier and `merge_revision` are
  covered by `test_ocr_state_store_round_trips_grouped_tree_and_invalidates_changed_source`.
- [x] Preserve meaningful groups, IDs, summaries, source slices, and reading
  order when a frontier has one remaining group. The document root remains a
  structural wrapper and `test_final_group_keeps_identity_and_kind_under_document_root`
  verifies the final meaningful group is retained.
- [x] Reject invalid grouping proposals without enqueueing rejected children or
  replacing the verified parent. `test_apply_grouping_proposals_keeps_source_coverage_and_rejects_bad_members`
  covers invalid members and preserves the verified tree.

### A5. Summary execution

- [x] Run summaries after the grouping pass is accepted in runtime state; the
  persisted grouping workflow records `generate_grouping_summaries` before
  `persist_grouping_result`, and the runtime trace regression asserts both
  nodes.
- [x] Recover an accepted summary journal entry when the surrounding runtime
  checkpoint is interrupted. `test_summary_journal_commit_is_recoverable_before_runtime_checkpoint`
  simulates a crash after the generated journal row and tree revision commit;
  resume restores the summary without invoking the provider again. The
  separate runtime-node publication gate remains partial.
- [x] Persist summary input identity including exact subtree slices, tree
  revision, grouping policy, summary policy, provider/model revision, and prompt
  schema revision. The `EvidenceSlice`/`GroundedSummaryGenerator` boundary
  exposes exact ordered ownership to the production parser-provider generator,
  and its configuration participates in the summary cache identity.
- [x] Reuse a summary only when all those identities match. Covered by
  `tests/test_workflow_ingest_ocr_pipeline.py::test_summary_cache_reuses_exact_generation_and_slice_identity`
  and `::test_summary_generation_fingerprint_invalidates_persisted_cache`.
- [x] Make summary failure retryable without invalidating OCR or grouping.
  `tests/test_workflow_ingest_ocr_pipeline.py::test_failed_summary_is_retryable_without_replaying_ocr`
  proves a failed summary is retried and accepted while the two OCR pages are
  not called again.
- [x] Ensure duplicate summary delivery cannot create duplicate accepted summary
  records. `OCRWorkflowStateStore.record_summary_attempt()` uses an immediate
  transaction and returns the existing accepted attempt for the same
  `(document_id, node_id, input_fingerprint)`; regression:
  `tests/test_workflow_ingest_ocr_pipeline.py::test_duplicate_accepted_summary_delivery_is_idempotent`.
  This closes summary-journal idempotence only; root publication and dual-store
  crash reconciliation remain separate gates.

### A6. Runtime verification

- [x] Add a test that inspects persisted Kogwistar runtime step nodes for OCR,
  grouping, and summary stages. The grouping runtime regression now asserts
  `generate_grouping_summaries`, `persist_grouping_result`, and `end` in the
  persisted step trace.
- [~] Add a test that kills execution after page 63 of 100, reopens the same
  engines and run ID in a new process, and resumes from the persisted token.
  `test_ocr_root_fresh_process_resume_after_page_63` now passes in 449.25s
  with a 900s bounded harness timeout against installed Kogwistar `0.6.12`;
  it proves pages 1-63 are not replayed, page 64 is retried at the
  interruption boundary, and one grouping result is accepted. The same test
  now records unique persisted `wf_invoked` IDs and verifies every accepted
  graph-payload node exists in the knowledge backend after recovery. Linux
  SIGKILL and complete dual-store reconciliation remain.
- [x] Add a test that kills execution after a grouping checkpoint and proves the
  checkpoint is restored before further pages are accepted. Covered by
  `test_grouping_checkpoint_reopens_after_process_termination`, which stops a
  fresh grouping worker during summary generation and resumes the same run from
  its persisted checkpoint.
- [x] Add a duplicate-resume race test and assert one accepted commit. Covered
  by `test_ocr_root_concurrent_resume_has_one_owner_and_one_result`, including
  an explicit `WorkflowRunOwnershipError` for the losing worker.
- [x] Add a no-provider-call-on-reuse test for already committed pages. Covered
  by `test_ocr_workflow_resumes_cached_pages_and_terminal_semantic_run`, which
  completes two pages, resumes the same root, and asserts the OCR call count
  remains `2` while both pages are reported as reused.
- [~] Assert the final runtime graph contains no orphaned stage executions,
  duplicate accepted commits, or skipped page cursor transitions. Persisted
  root-child lineage and direct replay idempotence are covered, but a complete
  fresh-process orphan/transition audit remains open.

## B. Strict Seam Closure S01-S12

### S01. PDF revision to rendered page bytes

- [x] Include PDF SHA-256, renderer implementation/version, DPI, rotation,
  crop/configuration, page number, and raster schema in render identity.
- [x] Store rendered-page digest and render identity in durable state. The
  source plan carries a versioned render identity into `page_state`, and
  `test_ocr_render_reuse_requires_render_configuration_identity` proves the
  reuse predicate rejects a changed renderer identity.
- [x] Refuse a rendered-page cache hit when any render identity differs.
- [x] Reconcile existing rendered files against state before reuse; the render
  manifest now records each page digest and dimensions and invalid pages are
  regenerated.
- [x] Test same-path PDF A to PDF B with changed page 2. The injected
  rasterizer regression `test_pdf_render_manifest_rejects_same_path_replacement`
  proves that a valid replacement at the same path invalidates the render cache
  and regenerates both pages.
- [x] Test render identity changes, including a typed custom rasterizer
  configuration fingerprint, in
  `tests/test_workflow_ingest_ocr_pipeline.py::test_pdf_render_manifest_rejects_tampered_page_and_records_identity`.
- [x] Test changed page count and removed pages; valid-PDF page-count and ordered
  page-set validation are covered by
  `test_pdf_render_manifest_rejects_a_missing_page_from_a_valid_pdf`, and
  `test_rebuild_ignores_artifacts_removed_by_a_shorter_source_revision`
  prevents trailing legacy OCR pages from being resurrected. Same-path
  replacement is covered by `test_pdf_render_manifest_rejects_same_path_replacement`;
  recovery crash windows pass the Linux gate in CI run `39135756441`.

### S02. Source bytes to committed page JSON

- [x] Store and enforce the OCR provider/config fingerprint with page state.
  The exact reuse predicate and a fresh-process regression prove that a
  model/engine change cannot reuse a completed page from an older
  configuration; the parser recovery matrix also covers the crash-order
  reconciliation.
- [x] Store page artifact SHA-256 and verify it before reuse. The exact
  completed-page reuse path rejects a well-formed tampered JSON artifact in
  `test_ocr_state_rejects_stale_provider_config_or_tampered_artifact`; the
  parser recovery matrix covers both crash orders.
- [x] Write JSON to a same-directory temporary file, flush it, and atomically
  replace the final artifact before recording completion. The shared writer
  uses unique same-directory temporary files and flush/fsync; concurrent
  publication is covered by
  `tests/test_workflow_ingest_ocr_pipeline.py::test_atomic_json_publication_uses_distinct_temporary_files_for_concurrent_writers`.
  Artifact-first and checkpoint-first crash recovery are also covered by the
  OCR page and Kogwistar recovery gates; parser CI run `39114468766` passed
  those Linux recovery cases.
- [x] Validate and reconcile an artifact left behind by a crash between rename
  and database commit. A valid rename-before-DB orphan is adopted only when
  its source-image and OCR-configuration identities match. The real
  `_process_ocr_page` path now has an injected crash regression,
  `test_ocr_page_artifact_publish_crash_is_recoverable`, which proves the
  artifact-first order is recoverable after reopening SQLite; the inverse
  order has public preparation coverage in
  `tests/test_workflow_ingest_ocr_pipeline.py::test_prepare_ocr_workflow_retries_checkpoint_without_artifact`,
  which reruns the provider after a completed SQLite row loses its artifact.
  The Linux recovery release gate passed in parser CI run `39114468766`.
- [x] Test malformed/truncated and wrong-page JSON, schema-version changes, and
  crash-window recovery. `test_legacy_ocr_completion_rejects_malformed_artifact`
  and `test_legacy_ocr_completion_rejects_wrong_page_artifact` reject malformed
  and cross-page payloads; `test_artifact_rebuild_accepts_known_ocr_schemas_only`
  accepts legacy `0.1` and current `workflow_ingest_v1` while rejecting unknown
  versions. `test_legacy_completion_rejects_unsupported_ocr_schema_version`,
  `test_orphan_reconciliation_rejects_unsupported_ocr_schema_version`, and
  `test_ocr_page_reuse_requires_artifact_path_and_supported_schema` cover
  completion, adoption, and reuse; crash-window cases passed CI run
  `39135756441`.
- [x] Ensure the artifact records the exact source image digest and accepted OCR
  configuration revision. The production page path records both and the crash
  regression reopens and verifies them before adoption. The focused regression
  `test_rebuild_rejects_legacy_ocr_when_page_40_changes_in_new_source_revision`
  also proves that a same-numbered page's legacy OCR is invalidated when its
  rendered bytes change across source revisions. The root-resume regression
  `test_ocr_root_resume_rebuilds_after_same_path_source_change` now changes a
  page image at the same path after a process interruption, resumes the same
  root run, and asserts both pages are rerun under the new source revision and
  the prior page text is absent. The additional
  `test_rebuild_does_not_trust_legacy_image_as_current_source` proves the
  legacy artifact directory cannot authenticate its own stale image when the
  authoritative current render is absent. CI run `39114954960` passed the
  changed-page regression on all four supported interpreter jobs; the new
  no-fallback regression is pending CI.
- [x] Fence the legacy `record_page_completed()` writer as well as the primary
  OCR publication path. It now accepts only a pending row with matching source
  and configuration identity, and treats an identical completed delivery as
  idempotent; `test_ocr_page_completion_rejects_a_late_writer_after_acceptance`
  proves a late writer cannot replace an accepted render result. This does not
  close the two-store crash-consistency gate. The failure writer is now fenced
  by pending status, source/configuration identity, and attempt ownership;
  `test_ocr_page_failure_cannot_downgrade_an_accepted_result` and
  `test_ocr_page_failure_cannot_replace_a_newer_pending_attempt` prove that a
  late failure cannot downgrade an accepted page or a newer retry.
- [x] Give same-identity duplicate OCR deliveries first-valid-result-wins
  semantics. `publish_ocr_page()` fences source/configuration identity while
  allowing a concurrent duplicate with a different attempt number to publish
  before the other worker observes the winner; superseded pending attempt
  rows are closed. `test_duplicate_ocr_page_workers_accept_only_one_artifact`
  passes on CPython 3.13 and the full deterministic selection.

### S03. OCR response to spatial/media grounding

- [~] Docling provenance boxes are used when exposed; the no-item fallback is
  still page-wide and requires explicit page-level accuracy.
- [x] Add a provider capability indicating whether coordinates are page,
  region, token, table-cell, or unknown precision. `OCRProviderCapabilities`
  now exposes the typed `coordinate_precision` field; direct and Docling
  adapters declare `region`, while chat/remote providers remain `unknown`.
- [x] Never label a synthetic full-page box as region precision. The existing
  `_spatial_accuracy` gate requires provider-declared precision and a usable
  box; `test_synthetic_page_box_does_not_claim_region_accuracy` remains the
  regression.
- [x] Preserve media identity and location for figures, tables, and captions.
  `test_build_document_tree_preserves_media_identity_location_and_metadata`
  verifies URI, digest, MIME/size, page label, relative region bounds, and OCR
  provenance survive the adapter boundary.
- [x] Record every rejected/lost non-text object with a structured diagnostic.
  `UnrepresentedSourceUnitsError.diagnostics` aggregates all rejected units
  while preserving the singular `UnrepresentedSourceUnitError` compatibility
  path; `test_build_document_tree_reports_all_unrepresented_nontext_units`
  verifies the complete report.
- [x] Add a mocked Docling fixture with separate text, table, and figure
  provenance, ordering, and coordinate clipping in
  `tests/test_workflow_ingest_ocr_providers.py::test_docling_adapter_preserves_text_table_figure_provenance`.
- [~] Add live-provider evidence for at least one multi-object document. A real
  Docling/RapidOCR layout fixture was exercised locally; its page-wide Markdown
  fallback exposed an `<!-- image -->` placeholder without a grounded picture,
  and the adapter now strips the marker and records
  `unresolved_item_count=1`. A live provider result with independently verified
  table/figure recall and spatial ground truth remains open.

### S04. Cluster order to multimodal reading order

- [x] Normalize text and non-text objects into one explicit source traversal in
  the response models and document adapter. Focused tests cover mixed text and
  figure order. The public OCR-root integration test also supplies that order
  to grouping and semantic parsing. Graph export now reifies media-bearing
  source items as Kogwistar multimodal evidence nodes linked to the semantic
  root; see `test_media_source_items_project_to_ordered_kogwistar_multimodal_graph_evidence`
  and `test_export_projects_grouped_media_into_kogwistar_multimodal_evidence`.
- [x] Validate that `meaningful_ordering` covers every accepted cluster exactly
  once, including non-text clusters. Duplicate IDs, missing IDs, and empty-page
  agreement are rejected by the typed model/adapter boundary. Diagnostics name
  missing, unknown, and duplicate cluster IDs and the bounded, typed failure
  feedback is available to the next built-in OCR candidate prompt. Cluster
  overlap and duplicate-order diagnostics use one-pass counts; the 256-cluster
  permutation regression exercises the high-cardinality valid input path.
- [~] Preserve order through normalization, grouping, serialization, and graph
  export. Normalization is covered by
  `test_normalize_ocr_pages_preserves_caption_and_nontext_permutation`, and the
  mixed figure/caption/table/body sibling order is now asserted at graph export
  by `test_graph_export_preserves_mixed_multimodal_sibling_reading_order`.
  `test_ocr_grouping_and_semantic_graph_preserve_mixed_reading_order` verifies
  the OCR-root grouping callback receives text A, figure, caption, text B and
  that the semantic graph's text projection retains text order. The graph now
  additionally preserves non-text evidence as ordered Kogwistar multimodal
  source records with digest/revision and spatial or temporal grounding. The
  source scope is supplied by the host as `source_namespace`; media export
  rejects a missing or whitespace-padded scope rather than substituting a
  document ID. Regression coverage is in
  `test_multimodal_projection_requires_explicit_graph_namespace` and
  `test_export_projects_grouped_media_into_kogwistar_multimodal_evidence`. A
  real multi-object provider fixture with independent ground truth remains
  open.
- [x] Represent ambiguous provider order explicitly instead of silently
  appending non-text objects after text. When `meaningful_ordering` is absent
  for a mixed page, normalized page metadata records
  `ordering_status="ambiguous_provider_order"`; declared ordering records
  `ordering_status="declared"`.
- [~] Test text A, figure, caption, text B and deliberate child permutation.
  `tests/test_workflow_ingest_ocr_providers.py::test_normalize_ocr_pages_preserves_caption_and_nontext_permutation`
  now covers figure/caption/table/text traversal and asserts modality order.
  `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_grouping_and_semantic_graph_preserve_mixed_reading_order`
  additionally exercises the public OCR-root workflow, grouped child order,
  semantic callback input order, and graph export. The new projection tests
  verify typed multimodal evidence and export integration; a provider-backed
  multi-object fixture with independent ground truth remains open.

### S05. Accepted group tree to semantic parser input

- [x] Define the typed parser input contract for accepted group hierarchy.
  `WorkflowExportBundle.grouping_tree` is a Pydantic-validated JSON artifact,
  and the OCR root requires it on the semantic child result. The bounded tree
  remains in the established parser payload intentionally; prompt-size
  optimization is not required for hierarchy correctness.
- [x] Prove that the semantic parser receives and preserves the accepted
  hierarchy rather than only receiving it as an ignored metadata sidecar.
  The semantic callback receives `parser_input_dict["document_tree"]`, export
  carries the same canonical tree, and the root rejects missing or changed
  trees. `test_custom_grouping_policy_reaches_checkpoint_cache_and_export`
  now consumes the accepted group ID in the semantic callback and asserts the
  derived label reaches the graph payload. The 100-page recovery test verifies
  persisted `wf_invoked` lineage for page/grouping/semantic children across
  process restarts and durable graph reopen; CI run `39135756441` passes. The
  required-contract regression
  `test_export_bundle_required_grouping_contract_fails_closed_without_tree`
  covers consumers that require hierarchy, while the default optional contract
  preserves legacy callers explicitly.
- [x] Prove that final graph export preserves intended group relationships,
  ordered slices, and source grounding. The graph exporter emits validated
  parent-child edges; `test_ocr_root_fresh_process_resume_after_page_63`
  compares canonical persisted graph output with an uninterrupted reference
  after Chroma reopen, and CI run `39135756441` passes the durable publication
  gate.
- [x] Add a metamorphic test using identical source items with two valid group
  trees and distinguishable export results in
  `tests/test_workflow_ingest_page_index_pipeline.py::test_graph_export_consumes_hierarchy_for_identical_source_evidence`.
- [x] Add an explicit compatibility mode for consumers that ignore hierarchy;
  fail closed where hierarchy is required. `WorkflowExportBundle.grouping_contract`
  is `optional` by default for legacy callers and `required` bundles reject a
  missing tree during Pydantic construction. The regression
  `test_export_bundle_required_grouping_contract_fails_closed_without_tree`
  covers the required boundary.

### S06. Grouping pass to durable checkpoint

- [x] Stop prefix pages from overwriting the canonical complete tree. The
  Kogwistar root calls preparation with `persist_grouping_checkpoint=False`;
  SQLite grouping-tree writes occur in the persisted grouping child after the
  accepted frontier is restored and validated. The standalone compatibility
  preparation path is not used as the root runtime cursor.
- [~] Add expected-revision/CAS protection. `save_document_tree()` now fences
  both an observed revision and an observed absence atomically, rejects an
  orphaned frontier or a missing frontier, and the grouping runtime and summary
  persistence propagate that fence into their persisted child state. A stale
  summary writer raises `GroupingCheckpointConflict` instead of becoming a
  non-fatal summary warning. The reusable regressions are
  `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_state_store_round_trips_grouped_tree_and_invalidates_changed_source`.
  and
  `tests/test_workflow_ingest_ocr_pipeline.py::test_summary_checkpoint_conflict_is_retryable_not_a_summary_warning`.
  Unfenced updates to an existing checkpoint now fail closed. This is
  provisional: every compatibility writer, late-writer path, and crash/restart
  ordering must use the same guard before this row can be green.
- [x] Persist source fingerprint, grouping policy, accepted page frontier, tree
  digest, and last accepted runtime step. `OCRWorkflowStateStore` commits these
  fields in `grouping_checkpoint_metadata` with the tree/frontier transaction;
  `test_ocr_state_store_round_trips_grouped_tree_and_invalidates_changed_source`
  verifies reload and source invalidation.
- [x] Keep OCR page checkpoints compact by persisting artifact identity and a
  cursor rather than cumulative `raw_pages`. New checkpoints are artifact-backed
  and `_json_pages()` rehydrates accepted page JSON on demand; legacy checkpoints
  with `raw_pages` remain readable. The regression
  `tests/test_workflow_ingest_ocr_pipeline.py::test_new_page_checkpoint_rehydrates_pages_from_artifacts_without_raw_payload`
  passes. The 100/500/1000-page artifact-backed checkpoint-growth regression
  now covers the long-document storage invariant; a machine-speed benchmark is
  intentionally outside this parser contract.
- [x] Restore before advancing and prove no-op reruns. The 100-page
  fresh-process regression confirms accepted OCR and summaries are not
  regenerated and canonical grouping/graph outputs match the uninterrupted
  reference. Artifact and checkpoint commit-order recovery passes CI run
  `39135756441`.
- [x] Test crash at page 63 and concurrent late writers. The 100-page
  `test_ocr_root_fresh_process_resume_after_page_63` regression covers fresh
  process recovery; `test_ocr_root_concurrent_resume_has_one_owner_and_one_result`
  covers competing resumes, and page completion/failure tests cover late
  writers.

### S07. Grouping to summary generation

- [x] Run summaries after grouping passes, not before caller-provided grouping.
  `test_grouping_runs_inside_kogwistar_runtime_and_is_resume_safe` asserts
  that the summary callback receives the grouped tree, and the persisted
  runtime routes `apply_grouping_pass` before
  `generate_grouping_summaries`.
- [~] Include exact subtree slice membership, tree revision, policy revision,
  model revision, and prompt schema in the summary fingerprint. The effective
  persisted generation identity now hashes both the caller policy revision and
  the generator's typed `configuration_fingerprint`; unversioned custom
  callbacks fail closed. The production generator identity includes provider
  configuration, prompt text/revision, and output schema. Regressions cover
  provider/config invalidation, missing custom identity, exact-slice reuse, and
  distinct grounded callbacks (`test_custom_summary_generator_requires_stable_cache_identity`,
  `test_summary_generation_fingerprint_invalidates_persisted_cache`, and
  `test_summary_cache_reuses_exact_generation_and_slice_identity`). The focused
  summary tests pass locally; keep this row provisional until the change passes
  the supported-interpreter CI matrix.
- [x] Test two nodes using different slices of the same source item.
  `test_summary_cache_reuses_exact_generation_and_slice_identity` records the
  exact ordered slice fingerprint rather than only source-item membership.
- [x] Test grouping-only changes invalidate summaries without rerunning OCR.
  `tests/test_document_adapters.py::test_grouping_change_invalidates_root_summary_without_changing_source_items`
  proves the grouping mutation clears the root's derived summary while source
  evidence remains unchanged; the root cache test separately proves OCR is not
  replayed on resume.
- [x] Test identical policy/input reuses summaries without another provider
  call in `test_summary_cache_reuses_exact_generation_and_slice_identity`.

### S08. Dynamic grouping to semantic identity

- [x] Preserve a meaningful final group when the frontier has one group; the
  document root remains a structural wrapper and the accepted group identity
  is retained below it. Covered by
  `tests/test_document_adapters.py::test_final_group_keeps_identity_and_kind_under_document_root`.
- [x] Make group ID, kind, provenance, and summary transformations explicit and
  versioned. `merge_groups` uses the stable proposal-derived `group_node_id` and
  records the versioned `grouping-v1` policy, ordered members, and explicit
  summary invalidation in `DocumentNode.annotations`.
- [x] Test a title plus atomic long table retaining its custom group and both
  children in
  `tests/test_document_adapters.py::test_grouping_preserves_title_and_atomic_table_with_versioned_provenance`.
- [x] Test replay/idempotence of the same grouping proposal. The stable
  proposal identity makes duplicate delivery a no-op in
  `test_replaying_an_accepted_grouping_proposal_is_idempotent`.

### S09. OCR/group/summary to Kogwistar checkpoint

- [x] Move all three stages into one persisted Kogwistar workflow lineage. The
  root design invokes page, grouping, and semantic child workflows with stable
  identities. `test_ocr_root_fresh_process_resume_after_page_63` uses installed
  Kogwistar `0.6.12`, verifies child lineage, two process terminations, and
  durable publication; CI run `39135756441` passes the supported matrix and
  Linux recovery gates.
- [x] Keep SQLite as an idempotent artifact cache, not the authoritative cursor.
  The root cursor is runtime-owned; all four artifact/checkpoint commit orders
  are covered by the recovery suite passed in CI run `39135756441`.
- [x] Persist step tokens and stage results in the existing conversation/runtime
  graph. The recovery test verifies unique `wf_invoked` edges, parent/child run
  IDs, workflow IDs, and invocation keys across root recovery.
- [x] Test process termination/reopen/resume in a new process with the same run
  ID. On POSIX, `_terminate_interrupted_worker()` uses `SIGKILL`; the 100-page
  test resumes after two terminations, reopens durable Chroma, and passes in CI
  run `39135756441`.
- [x] Test duplicate resume, late writers, and no double publication. The
  concurrent-root ownership, page late-writer, and canonical graph idempotence
  tests pass in the supported CI/recovery gates of run `39135756441`.

### S10. Source revision to downstream graph identity

- [x] Replace static `ocr-v1` source revision with content-addressed or
  authoritative revision identity. The document revision includes the input
  fingerprint, and the document source fingerprint now namespaces document ID
  and source revision alongside source items. Regressions:
  `test_source_fingerprint_includes_authoritative_reading_order` and
  `test_source_fingerprint_includes_document_and_revision_namespaces`; Parser
  CI run `39037176708` passed the full matrix on `7bfedf7`. Same-stem/cross-path
  identity is covered by `test_pdf_artifact_directories_do_not_collide_for_same_basename`
  and `test_pdf_artifact_directories_share_content_identity_across_paths`.
  Graph export now carries the authoritative source revision from the accepted
  grouping tree in both top-level graph payload and root-node metadata; coverage
  is provided by `test_export_carries_authoritative_source_revision`.
  Parser CI run `39045083192` passed the full matrix on `4cbb8cb`; the expanded
  same-path source-revision test also passes CI run `39135756441` and verifies
  new page OCR plus immutable old artifacts.
- [x] Separate document ID, source revision, OCR config, grouping revision, and
  parse generation namespaces. The OCR root now persists a dedicated
  `parse_generation`, validates it on resume, and derives the semantic child
  invocation key from it; `test_ocr_root_resume_rejects_a_different_parse_generation`
  covers the mismatch. Summary input fingerprints also namespace document ID and
  source revision with source items, covered by
  `test_source_fingerprint_includes_document_and_revision_namespaces`. Parser CI
  run `39039809496` passed the full matrix on `3800e7b`. The expanded
  changed-source recovery test and persisted graph publication pass CI run
  `39135756441`.
- [x] Make default batch document IDs collision-safe for same-stem files. The
  batch runner uses a resolved-path digest when stems collide, and
  `test_ocr_batch_workflow_fences_duplicate_source_stems` verifies distinct IDs
  and output directories.
- [x] Test two `report.pdf` files from different folders in one batch. The
  artifact-plan regression `test_pdf_artifact_directories_do_not_collide_for_same_basename`
  verifies distinct content-addressed render and legacy directories.
- [x] Test changed content at the same path and verify immutable revision
  separation. `test_pdf_render_manifest_rejects_same_path_replacement` verifies
  replacement invalidates the prior render identity and regenerates the page set.

### S11. Explicit references across boundaries

- [x] Support references spanning clusters, lines, and pages. Reference
  extraction now searches the deterministic text reading-order stream and maps
  every match back to one or more exact source slices; parser/export
  integration for all downstream consumers remains a separate concern.
- [x] Preserve multiple exact source slices for one reference mention in
  `DocumentReferenceMention.additional_sources`, validated against every
  authoritative text item and covered by
  `tests/test_document_grouping_contract.py::test_reference_mention_round_trips_multiple_cross_page_source_slices`.
- [x] Keep references outside the primary coverage partition unless explicitly
  promoted by a later workflow. `DocumentTree.validate_tree()` excludes
  `references` from child coverage, and
  `tests/test_document_grouping_contract.py::test_reference_mentions_round_trip_and_validate_targets_without_becoming_children`
  verifies that a resolved reference does not become a root child.
- [x] Test a reference split across OCR clusters and across pages in
  `tests/test_document_grouping_contract.py::test_reference_mentions_cross_ocr_units_preserve_exact_source_slices`.

### S12. Public entrypoint to internal options

- [x] Thread grouping policy version through the public runner into parsing,
  grouping, checkpoint, summary, and export layers.
- [x] Expose summary policy/generator consistently at the public runner. The
  public OCR parse and root-workflow runners forward both values; CLI forwarding
  is covered by `tests/test_workflow_ingest_cli.py::test_cli_forwards_summary_hooks`.
- [x] Reject conflicting public/internal policy values rather than silently
  falling back to `baseline-v1`. `_validate_public_runtime_policy_conflicts`
  fails closed for contradictory public `parse_strategy` and internal
  `split_strategy` values; matching values are accepted. Regression coverage:
  `tests/test_workflow_ingest_contracts.py::test_public_and_runtime_parse_strategy_conflict_fails_closed`
  and `::test_matching_public_and_runtime_parse_strategy_is_accepted`.
- [x] Test explicit custom policy through the public API and inspect persisted
  metadata, cache keys, and final output. The CI regression
  `tests/test_workflow_ingest_ocr_pipeline.py::test_custom_grouping_policy_reaches_checkpoint_cache_and_export`
  runs the public OCR root, verifies the policy in SQLite checkpoint metadata,
  verifies the policy-bearing stable group provenance, and verifies that the
  accepted grouping tree is the exported hierarchy.

## C. Release gates added by the implementation review

- [x] **C01 — Installed API compatibility:** installed Kogwistar `0.6.12`
  metadata and module version agree, and parser CI run `39011917164` passed
  the CPython 3.12-3.14 and PyPy 3.11 matrix.
- [x] **C02 — Runtime ownership boundary:** stable root/page/grouping/semantic
  child IDs, fresh-process lineage, lease ownership, duplicate-resume, and
  no-double-publication are covered by the focused recovery matrix.
- [x] **C03 — Dual-store reconciliation:** artifact-first and checkpoint-first
  publication orders are covered by fresh-process recovery tests for page,
  grouping, and semantic graph publication boundaries.
- [x] **C04 — Incremental equivalence:** compare canonical tree content, not
  timestamps, between uninterrupted batch grouping and interrupted incremental
  grouping. Completed subtrees remain stable, and
  `test_merge_groups_scans_frontier_linearly_without_nested_rebuilds` adds a
  bounded-access regression so incremental grouping cannot silently regress to
  nested frontier rebuilding.
- [x] **C05 — True resume:** completed OCR, grouping, and summaries are not
  rerun after covered process stops, and duplicate resume accepts one result.
  The updated `test_ocr_root_fresh_process_resume_after_page_63` passed locally
  in 1093.68s using installed Kogwistar `0.6.12`, two real process
  terminations, same-run fresh-process recovery, persistent Chroma, summary
  reuse, semantic publication, and canonical tree/graph comparison with an
  uninterrupted persistent reference using identical PDF bytes/source
  revision. The embedded page images now contain the fixture text, chart, and
  15-row table; the declared OCR reading order is asserted against the rendered
  page. Its deterministic synthetic OCR payloads cover duplicate text and a
  sentence continuing across pages 62-63; this is recovery evidence, not live
  OCR accuracy. Parser CI run `39114468766` passed all four interpreter jobs,
  the Linux recovery gate, and durable graph publication on fixture commit
  `8ca0d2e`. The target remains
  at-least-once execution with exactly one accepted canonical result, not
  exactly-once provider calls.
- [~] **C06 — OCR quality/completeness:** interval coverage is necessary, not
  sufficient. `OCRQualityPolicy` records provider diagnostics and can reject
  incomplete text, edge-word loss, excessive data-loss likelihood, unresolved
  placeholders, and ungrounded provider items. `OCRGroundTruth` measures item
  recall, spatial IoU, and transcription mismatch; the Codex table fixture
  provides auditable source-grounded text/table annotations. This is not
  independent evidence of provider figure/non-text recall: the recorded live
  Codex figure run emitted no non-text objects. Close C06 with an independently
  annotated visual-region corpus and provider evaluation, including modality-
  specific recall and spatial fidelity. Do not treat text recall or source
  interval coverage as proof of visual completeness.

### Reviewer acceptance matrix

The following matrix is the required status ledger for the external review.
It deliberately records the current gaps instead of treating the checklist
itself as proof:

| Gate | Current status | Required closure evidence |
| --- | --- | --- |
| C01 installed API compatibility | PASS (matrix verified) | The active Poetry environment verifies installed Kogwistar `0.6.12`: package metadata and `kogwistar.__version__` both report `0.6.12`, and `test_installed_kogwistar_exposes_the_persisted_workflow_api` passes. Parser CI run `39011917164` passed the supported CPython 3.12-3.14 and PyPy 3.11 matrix against this dependency. |
| C02 runtime ownership boundary | PASS (focused matrix) | Root/page/grouping/semantic child IDs, resume identity, ownership fencing, and persisted invocation lineage are covered by the fresh-process and concurrent-resume tests. Parser CI run `39011917164` passed all four supported interpreter jobs and the Linux recovery gate on exact commit `e57fd43`. The long-document release fixture remains tracked under C05. |
| C03 dual-store reconciliation | PASS (parser gate) | The four-boundary recovery matrix passes locally in one run (`4 passed, 113.32s`): `test_ocr_page_artifact_publish_crash_is_recoverable` covers artifact-first page publication, `test_prepare_ocr_workflow_retries_checkpoint_without_artifact` covers checkpoint-first page recovery, `test_ocr_root_retries_after_grouping_sqlite_commit_before_nested_checkpoint` covers grouping SQLite commit before the nested runtime checkpoint, and `test_ocr_root_retries_after_semantic_graph_commit_before_checkpoint` covers graph publication before the semantic checkpoint with stable canonical IDs and zero duplicate writes. The Linux recovery gate in parser CI run `39011917164` also passed. |
| C04 incremental equivalence | PASS (structural cost gate) | `tests/test_workflow_ingest_ocr_pipeline.py::test_grouping_incremental_resume_matches_uninterrupted_reference` compares a two-pass interrupted/resumed grouping tree with an uninterrupted reference using canonical model payloads. `tests/test_document_grouping_contract.py::test_merge_groups_scans_frontier_linearly_without_nested_rebuilds` counts validation, indexing, and emission accesses and rejects nested frontier rebuilds. The implementation stores and restores the accepted frontier and `merge_revision`; this is a structural bound, not a machine-speed benchmark. |
| C05 true resume | PASS locally and CI | `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_root_fresh_process_resume_after_page_63` passed locally in 1293.43s and in Parser CI run `39135756441` with installed `kogwistar==0.6.12`, two real process terminations, same-run fresh-process recovery, persistent Chroma reopen, accepted OCR/summary reuse, semantic publication, and canonical tree/graph comparison to an uninterrupted reference using identical PDF bytes/source revision. The rendered 100-page fixture includes a 15-row table, chart, duplicate paragraphs, explicit page-62/63 continuation/reference, and asserted reading order. Provider results are deterministic fixtures, so this is recovery/source-consistency evidence, not live OCR accuracy. |
| C06 OCR quality/completeness | PARTIAL (independent visual evidence) | Typed provider diagnostics, strict source-item accounting, and `OCRGroundTruth` cover loss signals, item recall, spatial IoU, and text mismatch. Independent table and chart fixtures now exercise captured Codex, direct RapidOCR, direct EasyOCR, and Docling+RapidOCR results. On the chart, direct RapidOCR matches 11/11 text items but misses the region; direct EasyOCR emits nine detections, only two spatially match, and misses the region; Docling+RapidOCR matches the region and title but only 1/11 expected text items. Exact outputs are regression-tested. These controlled pages do not establish corpus-wide completeness; do not infer it from interval coverage or text recall alone. |

### C05 evidence correction (2026-10-15)

The source-checkout regression
`tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_root_resume_reopens_same_lineage_in_a_fresh_process`
passes in 128.89 seconds when the subprocess explicitly receives both the parser
and Kogwistar source checkouts. The first subprocess interrupts after pages `[1, 2]`; a
new process waits for the lease to expire, resumes the same root run, calls only
page 2, and accepts the semantic graph bundle. This proves the parent/child
workflow path and top-level graph `doc_id` contract. Kogwistar `0.6.10` is now
available from PyPI; the current parser matrix remains the required
installed-package verification.

The process-boundary regression
`tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_root_reopens_after_real_process_termination`
also passes in 117.12 seconds. It starts a separate worker, waits until page
two is in progress after page one has been accepted, terminates that worker,
and starts a fresh worker with the same persistent run state. The resumed
worker reports `calls == [2]`, keeps the same root run ID, and produces a
grouping tree. This is real process-termination evidence on the host platform,
not a callback exception simulation. It is deliberately not described as a
Linux `SIGKILL` test, and it does not close the installed-package, duplicate
resume, or dual-store recovery gates.

The grouping checkpoint regression
`tests/test_workflow_ingest_ocr_pipeline.py::test_grouping_checkpoint_reopens_after_process_termination`
also passes in 65.52 seconds. It stops a fresh grouping worker after the
grouping checkpoint is persisted but while summary generation is in progress,
then resumes the same run after the short ownership lease expires. The resumed
result contains the accepted section and summary. This proves grouping
checkpoint reopening on the host platform, but not Linux `SIGKILL`,
installed-package recovery, or complete artifact/SQLite/runtime reconciliation.

The 100-page deterministic comparison regression
`tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_root_resumes_after_page_63_and_matches_uninterrupted_reference`
passed with the real persisted parser/runtime path and compares the
uninterrupted and resumed canonical graph payloads. Separately,
`tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_root_fresh_process_resume_after_page_63`
now kills a 100-page run after page 63 is accepted, resumes the same Kogwistar
root in a fresh process, and verifies exact Chroma node/edge IDs after reopening
the durable graph. The latter passed locally in 537.97 seconds; its updated
Linux recovery-gate CI run is pending. These tests establish the two halves
independently, not yet a single durable interrupted-versus-uninterrupted
comparison. Independent visual OCR completeness remains open.
The focused fresh-process page-state test
`tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_page_resume_reopens_state_in_a_fresh_process`
also passes in 33.32 seconds and proves accepted page reuse across
interpreters. It intentionally does not claim runtime-checkpoint restoration:
the test backend is in-memory, so A-6 still requires a persistent graph backend
and a real process-boundary root resume.

### S02 correction ledger

S02 must not be marked complete until all of the following are independently
demonstrated:

- [x] configuration fingerprint changes prevent reuse in a fresh process;
- [x] a well-formed but tampered OCR JSON is rejected by the exact completed-
  page reuse path;
- [x] atomic rename is used for publication and the real page path survives an
  injected crash immediately after publication;
- [x] Both artifact/checkpoint crash orders reconcile without accepting a
  partial or stale result. `test_ocr_page_artifact_publish_crash_is_recoverable`
  and `test_prepare_ocr_workflow_retries_checkpoint_without_artifact` cover the
  two page orders; grouping-SQLite and semantic-graph ordering tests cover the
  nested runtime boundaries.

The current implementation has atomic publication, digest fields, a
fresh-process configuration fence, exact tamper rejection, a
rename-before-database orphan-adoption path, and both page/checkpoint crash
orders are covered by the parser recovery matrix. Durable graph-backend
publication across process restart remains delegated to Kogwistar backend
contracts rather than duplicated here.

## C. Required End-to-End Acceptance Fixture

`tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_root_fresh_process_resume_after_page_63`
now combines the rich document and durable recovery requirements in one test.
It generates the 100-page PDF at runtime, performs two real process
terminations, resumes the same persistent Kogwistar run in fresh processes,
compares canonical tree and graph output with an uninterrupted reference, then
changes only page 40 and verifies a new source revision. OCR provider
completeness against independent visual ground truth remains separately open
under C06; this recovery test uses deterministic OCR/grouping payloads.

- [x] Generate a deterministic 100-page PDF fixture containing:
  - a long atomic table;
  - duplicate paragraphs at distinct positions;
  - text, line-chart figure, caption, and table ordering;
  - page-labelled front matter;
  - a paragraph crossing pages 62-63;
  - a cross-page explicit reference;
  - a changed page 40 in a second source revision.
- [x] Run an uninterrupted reference execution with deterministic OCR,
  grouping, and grounded-summary providers.
- [x] Terminate after page 63 has been accepted, reopen persistent Kogwistar
  engines in a fresh process, and resume with the same run ID.
- [x] Terminate after grounded summaries are accepted and semantic publication
  begins, then resume in another fresh process.
- [x] Compare the canonical grouping-tree digest and canonical persisted graph
  payload/IDs with the uninterrupted reference; verify ordered page-one source
  items, the page 62-63 continuation, and the cross-page reference.
- [x] Verify accepted OCR and summary results are not regenerated on resume,
  persisted graph IDs survive Chroma reopen, and the changed source revision
  reprocesses all pages while preserving the accepted revision-one page-40
  artifact and selecting the digest-matched revision-two artifact.

### C05 CI promotion

The expanded test above passes locally with installed Kogwistar `0.6.12`
(`1 passed, 1293.43s`). It performs two OS process terminations, recovers the
same run, verifies summary idempotence and durable graph publication after
Chroma reopen, compares canonical tree/graph output against an uninterrupted
reference, and verifies a changed source revision with page-digest-bound OCR
artifacts. Its provider responses are deterministic fixtures, so it does not
establish live-provider accuracy or independent figure recall. Parser CI run
`39135756441` completed successfully across CPython 3.12-3.14 and PyPy 3.11,
including the Linux recovery and durable graph-publication gates, so C05 is now
CI-verified for this branch revision.

## Evidence Ledger Requirements

Every A, S, and C item must record all of the following before it can become
`[x]`:

| Field | Required evidence |
| --- | --- |
| Implementation | File, function, and source line(s) |
| Regression test | Exact pytest node ID and key assertion |
| Before fix | Reproduced failure or demonstrated missing behavior |
| After fix | Passing result and what it proves |
| Runtime | Actual Kogwistar runtime, mocked runtime, or standalone adapter |
| Status | Verified pass, fixed but unverified, failing, not implemented, or not applicable |

The current expanded root-runtime gate proves parent/child invocation,
new-process interruption and resume, duplicate-resume exclusion, summary
reuse, and durable graph publication. Parser CI run `39135756441` passes the
recovery and graph-publication gates. Provider output remains deterministic
fixture data, so live OCR completeness is still a separate open requirement.

## D. Evidence And Release Gate

- [x] S01-S12 evidence map below: source locations, demonstrated risk,
  regression assertion, runtime classification, and current CI verification.

| Item | Implementation location | Demonstrated risk and regression assertion | Runtime evidence / status |
| --- | --- | --- | --- |
| S01 render identity | `ocr_pipeline.py:3395` `_render_pdf_with_identity` | Same-path replacement or changed rasterizer config could reuse stale pages. `test_pdf_render_manifest_rejects_same_path_replacement`, `test_pdf_render_manifest_rejects_tampered_page_and_records_identity`, and `test_ocr_render_reuse_requires_render_configuration_identity` assert invalidation. | Rasterizer/manifest boundary tests; not a Kogwistar workflow test. Pass in CI #562; included in #563. |
| S02 OCR artifact commit | `ocr_pipeline.py:165` `_ocr_configuration_fingerprint`; `:2634` `publish_ocr_page` | Stale provider config, modified JSON, or crash between file and state commit must not be accepted. `test_ocr_state_rejects_stale_provider_config_or_tampered_artifact`, `test_ocr_page_artifact_publish_crash_is_recoverable`, and `test_prepare_ocr_workflow_retries_checkpoint_without_artifact` assert reject/adopt/retry behavior. | SQLite artifact tests plus installed Kogwistar root-recovery gates; pass in CI #562. |
| S03 spatial/media grounding | `ocr_providers.py:493` `run_docling_ocr_page`; `adapters.py:209` `normalize_ocr_pages`; `ocr_pipeline.py:629` `OCRQualityPolicy` | Missing/invalid boxes and full-page fallback must not masquerade as region precision. `test_docling_adapter_preserves_text_table_figure_provenance`, `test_ground_truth_evaluator_rejects_invalid_provider_boxes`, `test_recorded_docling_full_page_fallback_does_not_claim_region_grounding`, `test_recorded_live_docling_provenance_exposes_text_loss_and_region_recall`, and `test_recorded_live_direct_rapidocr_exposes_missing_visual_region` assert provenance/loss diagnostics. | Mocked adapter tests and captured real provider responses; live visual-completeness evidence remains partial. CI #563 passed; #564 verifies the added current-adapter fixture. |
| S04 multimodal reading order | `adapters.py:209` `normalize_ocr_pages`; `document_models.py:398` `DocumentTree.source_fingerprint` | Sorting by modality or dropping an item could alter authoritative traversal. `test_normalize_ocr_pages_preserves_caption_and_nontext_permutation`, `test_source_fingerprint_includes_authoritative_reading_order`, and `test_ocr_grouping_and_semantic_graph_preserve_mixed_reading_order` assert identity/order through export. | Adapter plus graph/export tests; persisted runtime evidence is in the 100-page root gate, CI #562. |
| S05 hierarchy consumption | `document_adapters.py:742` `apply_grouping_proposals`; `ocr_runtime.py:1637` `run_ocr_root_workflow` | A consumer could ignore or flatten accepted groups. `test_graph_export_consumes_hierarchy_for_identical_source_evidence`, `test_default_adaptive_root_wires_grounded_summary_and_semantic_export`, and `test_ocr_root_fresh_process_resume_publishes_durable_graph` assert hierarchy affects output and survives persistence. | Installed Kogwistar workflow plus durable graph publication; CI #562 passed. |
| S06 grouping checkpoint | `ocr_pipeline.py:1985` `save_document_tree`; `:2244` `load_grouping_frontier`; `ocr_runtime.py:1192` `_validated_grouping_checkpoint` | Stale/foreign frontier or incomplete checkpoint could overwrite accepted grouping. `test_grouping_resume_rejects_frontier_that_disagrees_with_tree`, `test_grouping_incremental_resume_matches_uninterrupted_reference`, and `test_grouping_checkpoint_reopens_after_process_termination` assert rejection/equivalence/reopen. | SQLite checkpoint with fresh-process Kogwistar recovery; CI #562 passed recovery gate. |
| S07 grounded summaries | `document_adapters.py:604` `apply_summary_updates`; `ocr_pipeline.py:3864` `_write_summary`; `:3026` `load_completed_summary` | Stale generation, wrong slices, or replay could publish a mismatched summary or repeat completed work. `test_summary_cache_reuses_exact_generation_and_slice_identity`, `test_summary_generation_fingerprint_invalidates_persisted_cache`, and `test_ocr_summary_is_not_regenerated_after_fresh_process_resume` assert cache fencing/reuse. | Kogwistar root summary workflow plus durable journal tests; CI #562 passed recovery gate. Live summary quality is not claimed. |
| S08 semantic group identity | `document_adapters.py:742` `apply_grouping_proposals` | A final single group could be discarded or replay could duplicate/change it. `test_final_group_keeps_identity_and_kind_under_document_root`, `test_grouping_preserves_title_and_atomic_table_with_versioned_provenance`, and `test_replaying_an_accepted_grouping_proposal_is_idempotent` assert identity/provenance/idempotence. | Pure document-tree adapter tests; no runtime boundary claimed. #563 pending. |
| S09 persisted runtime lineage | `ocr_runtime.py:1525` `run_grouping_workflow`; `:1637` `run_ocr_root_workflow`; `ocr_pipeline.py:4100` `run_ocr_ingest_workflow` | A passed run ID alone could conceal separate, untracked OCR/grouping execution. `test_ocr_root_fresh_process_resume_after_page_63` proves fresh-process recovery, while `test_ocr_root_concurrent_resume_has_one_owner_and_one_result` proves single ownership. | Installed `kogwistar==0.6.12`, persistent workflow engines, fresh processes, and durable graph backend; CI #562 passed. |
| S10 source-to-graph identity | `document_models.py:398` `DocumentTree.source_fingerprint`; `ocr_runtime.py:872` `_semantic_parse_generation`; `handlers.py:251` `_build_export_bundle` | Same-stem documents, changed content, or replay could collide or overwrite canonical identities. `test_source_fingerprint_includes_document_and_revision_namespaces`, `test_export_carries_authoritative_source_revision`, `test_pdf_artifact_directories_do_not_collide_for_same_basename`, and `test_direct_graph_publication_is_idempotent_for_replayed_bundle` assert separation/idempotence. | Document model, graph-export, and durable graph-publication tests; CI #562 passed publication gate. |
| S11 explicit cross-boundary references | `document_models.py:192` `DocumentReferenceMention`; `:558` `extract_reference_mentions`; `document_adapters.py:253` | Mention spans could be dropped, turned into children, or point outside source. `test_reference_mentions_cross_ocr_units_preserve_exact_source_slices`, `test_reference_mention_round_trips_multiple_cross_page_source_slices`, and `test_reference_mentions_round_trip_and_validate_targets_without_becoming_children` assert exact-slice/target behavior. | Typed model and document-adapter tests; runtime workflow not required for this value object contract. #563 pending. |
| S12 public/internal options | `ocr_pipeline.py:4100` `run_ocr_ingest_workflow`; `test_workflow_ingest_cli.py` source-workflow adapter | Public and runtime parse policies could disagree or lose configured hooks. `test_public_and_runtime_parse_strategy_conflict_fails_closed`, `test_matching_public_and_runtime_parse_strategy_is_accepted`, `test_custom_grouping_policy_reaches_checkpoint_cache_and_export`, and `test_ocr_source_workflow_forwards_summary_generation_contract` assert forwarding and fail-closed conflicts. | Public API/CLI with installed runtime integration; CI #562 passed baseline, #563 pending. |

CI run #562 (`39146052615`) passed on code commit `9e24304`; the current
fixture/evidence-map commit is `b50b0fe`. CI #563 (`39147218276`) passed the
Linux recovery and durable graph-publication gates and the CPython 3.12-3.14
and PyPy 3.11 matrix, verifying the evidence-map regression suite on all
supported interpreters.
- [x] Run Ruff with the repository CI selection. Parser CI run `39135756441`
  passed its quality checks.
- [x] Run Pyright with supported CPython/PyPy-compatible syntax. Production
  package Pyright reports zero errors, warnings, and informations.
- [x] Run `git diff --check` and lock verification.
- [x] Run focused seam tests with `-p no:cacheprovider` and repository-local
  basetemp where Windows temporary-directory permissions require it.
- [x] Run the full parser CI suite on CPython 3.12-3.14 and PyPy 3.11. Run
  `39135756441` passed all four interpreter jobs, the Linux recovery gate, and
  durable graph publication for the current code-bearing commit. Live/manual
  provider quality remains a separate gate.
- [ ] Only after all release gates pass, update version/release notes and pins.

### Evidence update (2026-10-15)

- S08 final-group identity preservation is implemented in
  `workflow_ingest/document_adapters.py::apply_grouping_proposals`. The
  document root is explicitly treated as a structural wrapper that may have
  one child; semantic grouping nodes remain subject to the two-or-more-child
  invariant. The focused adapter suite passes its grouping assertions.
- Kogwistar `0.6.10` is merged, tagged at core commit `c11efd9`, and published
  on PyPI. The parser must not claim the installed-package release gate until
  the current clean matrix is green. Installed-package 100-page recovery and
  fresh-process root checkpoint recovery remain open.
- B10 page-checkpoint compaction is implemented locally in
  `workflow_ingest/ocr_runtime.py`: page artifacts are the durable payload
  source, while runtime state stores only stable page identity and progress.
  Focused compact-state, terminal-resume, and fresh-process tests pass; CI and
  large-document size measurements remain pending.

### Typed provider metadata seam regression (2026-10-16)

- [~] Fixed locally; CI pending. `document_ingester_logger.py` now narrows
  optional provider `usage_metadata` through `_UsageMetadataCarrier` and
  `_message_usage_metadata()` instead of reading it with an untyped `getattr`
  boundary. Invalid metadata fails closed to an empty mapping.
- Regression:
  `tests/test_document_ingester_logger.py::test_message_usage_metadata_narrows_optional_provider_extension`
  covers a valid SDK-like carrier, an ordinary LangChain message, and a
  malformed list payload. The focused test, Ruff CI selection, and
  `git diff --check` pass locally. Parser CI runs #311 and #312 remain
  pending, so this seam is not yet a verified release gate.

### Typed response and semantic-node seams (2026-10-16)

- [x] Verified on Parser CI #316. Commit `98d8153`
  narrows LangChain accounting objects through runtime-checkable Protocols for
  usage metadata, model identity, generation messages, and response metadata.
  `tests/test_document_ingester_logger.py` covers valid SDK-like objects and
  malformed/ordinary message objects. Focused tests, Ruff, and diff checks
  passed locally; Parser CI #316 passed.
- [x] Verified on Parser CI #316. Commit `86a7179`
  centralizes the optional SDK content boundary in
  `workflow_ingest/sdk_boundary.py` and reuses it from OCR and pipeline paths.
  `test_shared_sdk_content_boundary_preserves_mapping_and_empty_content_rules`
  proves SDK-like content, mappings, and empty content are not conflated.
- [x] Verified on Parser CI #316. Commit `2780377`
  makes semantic tree and integrity-node shapes explicit runtime-checkable
  Protocols and removes dynamic attribute reads from the shared validator
  helpers. Pointer-edge, layerwise, legacy-invariant, and fixture-payload
  regression suites passed locally; changed-file Pyright, Ruff, and diff checks
  also passed. This does not close the remaining full-module typing backlog.
- [~] Fixed locally; CI pending in run #318. Commit `91c20db` removes
  redundant `hasattr` checks from concrete LangChain `ChatGeneration` paths,
  makes logger shutdown failures observable at debug level, and uses the
  Python 3.11-compatible `datetime.UTC`. Focused logger tests, Ruff, changed-
  file Pyright, and diff checks pass locally.

### Typed OCR workflow seam correction (2026-10-16)

- [x] Parser CI #329 (`38918021548`) verified commit `466d70f`, which removes
  five full-package Pyright errors in the OCR,
  layerwise, and terminal-correction seams. Optional review sessions now use
  an explicit empty attempt context; PDF render manifest values are narrowed
  before use; terminal correction exposes a read-only Protocol property; and
  reasoning entries serialize through their declared Pydantic model. The full
  local parser-package Pyright scan reports `0 errors, 0 warnings, 0
  informations`, and focused OCR/PDF/correction regressions pass. This closes
  the identified seam only; the repository-wide typing backlog and runtime
  release gates remain open.

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

- [x] Commit `b3a0a59` narrows service-layer metadata and review diagnostics to
  the JSON value union; changed-module Pyright and Ruff pass, and the focused
  workflow/CLI tests pass (`4 passed`).
- [x] Commit `957363c` narrows OCR resume identity comparisons to the known
  `str | list[str]` payload shape; changed-module Pyright and Ruff pass, and
  the resume/frontier regression selection passes.
- [x] Commit `3ea269e` narrows workflow cache model dumps to `JsonValue`, with
  six cache namespace/concurrency regression tests passing. Commit `5cffeff`
  applies the same contract to canonical graph payload conversion; Pyright and
  Ruff pass and the selected canonical persistence tests pass. These entries
  close only the named seams, not the remaining repository-wide typing sweep.

### Shared parser payload contract (2026-10-16)

- [x] Commits `b61f428` and `96fe1bc` align adapter, parser-core, and PageIndex
  payloads on `JsonObject`/`ParserSourceMap` instead of invariant
  `dict[str, object]` aliases. Pyright reports zero errors for the changed
  modules, the CI Ruff selection passes, and the focused source-map/PageIndex
  suite passes (`17 passed`). CI for `b61f428` is still in progress; this
  evidence does not close the full CI or release gates.

### Persisted-state and legacy runner contract (2026-10-16)

- [x] Commit `8b8d3fb` makes persisted parser payload, source-map, and
  object-map decoders fail closed on non-JSON keys and malformed entries
  instead of stringifying keys or silently dropping invalid state. The new
  regression test `test_handler_persisted_parser_objects_reject_non_json_keys_and_entries`
  and the existing malformed-state selection pass (`2 passed`). Changed
  handler and PageIndex modules report zero Pyright errors.
- [x] Commit `9341bd4` aligns the legacy parsing facade, runner result, and
  graph export boundary on `ParserSourceMap`; legacy export accepts read-only
  mappings so the JSON-specific map is not widened back to
  `dict[str, dict[str, object]]`. Changed parsing, runner, and legacy export
  modules report zero Pyright errors; the focused legacy workflow selection
  passes (`2 passed`). The legacy monolith still has pre-existing broad Ruff
  violations outside this change, and full CI remains the release gate.
- [x] Commit `33aa16a` aligns layerwise pointer, boundary, and source-context
  helpers on read-only `Mapping[str, Mapping[str, object]]` views, allowing the
  strict `ParserSourceMap` and legacy-compatible maps without invariant casts.
  The CI Ruff selection passes and the focused hydration/boundary suite passes
  (`34 passed`). Local Pyright was blocked by the Windows Node launcher path
  error; GitHub CI remains authoritative for this commit.
- [x] Commit `4008804` normalizes retry prompt payloads through `json_safe`
  before adding retry metadata, instead of casting provider/model summaries to
  `ParserPayload`. Pyright reports zero errors for `layerwise_llm.py`, Ruff
  passes, and the focused proposal/boundary suite passes (`23 passed`). CI for
  this commit is queued; the release gate remains open.
- [x] Parser CI run `38928577822` verified commit `f0bb1cc`, which normalizes client workflow state through `json_safe` and
  rejects a non-object result at the serialization boundary rather than relying
  on an unchecked `dict[str, JsonValue]` cast. `clients.py` reports zero Pyright
  errors, the CI Ruff selection passes, and the focused client/demo contract
  selection passes (`11 passed, 4 skipped`). This is evidence for the client
  state seam only; full runtime recovery, matrix CI, and release gates remain
  open. The checklist-only follow-up `89b8dc3` was verified by Parser CI run
  `38928761211`.
- [x] Parser CI run `38929473712` verified commit `eb02d6c`, which uses the shared `ModelDumpLike` protocol for provider
  usage extraction instead of an untyped dynamic `model_dump` lookup, and narrows
  the provider regression's JSON/diagnostic reads. Provider and test Pyright
  reports zero errors, Ruff passes, and the focused provider selection passes
  (`10 passed`).
- [x] Commit `ed4ac40` uses the shared `ModelDumpLike` protocol for legacy
  semantic-tree coercion and adds a dedicated regression module. Production and
  dedicated-test Pyright report zero errors, Ruff passes, and the regression
  passes (`1 passed`); Parser CI `38930015792` passed all four supported
  interpreter jobs.
- [x] Commit `9e0631a` narrows document source and summary fingerprint payloads
  through the shared JSON-safe object helper instead of `list[dict[str, object]]`.
  Document-model Pyright and Ruff pass, and fingerprint regressions pass
  (`2 passed`); Parser CI `38930317634` passed all four supported interpreter
  jobs.
- [x] Commit `14464e2` routes handler parser payload and source-map reads through
  the recursive `decode_json_object` boundary instead of unchecked casts.
  The malformed proposal/review, authoritative source-map forwarding, and cache
  isolation selection passes (`4 passed`); changed-module Ruff passes. This
  closes the handler JSON boundary only, not repository-wide typing or release
  gates.
- [x] Commit `ba8e485` removes concrete legacy parser quality defects: the dead
  invalid `HydratedTextPointerLLM` alias, unused export intermediates, bare
  exception handlers, and unused range state. The legacy module imports, the
  selected `E7,E9,F` Ruff rules, and `git diff --check` pass. Full legacy
  Ruff still reports pre-existing `E402` import-order debt and remains open.

### Installed-core concurrent resume evidence (2026-10-17)

- [x] With installed Kogwistar `0.6.12` (package metadata and module version
  agree), `test_ocr_root_concurrent_resume_has_one_owner_and_one_result` passed
  locally in `95.28s`. The test terminates an interrupted worker, starts two
  fresh workers against the same persisted root, and asserts one accepted
  result plus one ownership failure. This closes the installed-package local
  concurrent-resume evidence gap only; Linux SIGKILL, remote matrix, and full
  dual-store reconciliation remain open.

### Explicit hierarchy compatibility contract (2026-10-17)

- [x] `WorkflowExportBundle` now carries a typed `grouping_contract` discriminator.
  The default `optional` mode preserves existing non-OCR callers; the explicit
  `required` mode fails closed when `grouping_tree` is absent. The focused
  export-contract selection passes (`2 passed`) and changed-module Ruff passes.
