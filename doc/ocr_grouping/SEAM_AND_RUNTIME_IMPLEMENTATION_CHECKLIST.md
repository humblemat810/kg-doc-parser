# OCR Runtime And Seam Closure Checklist

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

## Audit note (2026-10-15)

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
  either crash order;
- **C04:** prove incremental and uninterrupted grouping are canonically
  equivalent;
- **C05:** prove true resume without redundant accepted work;
- **C06:** require an explicit OCR completeness/quality gate beyond interval
  coverage.

These gates are tracked in section C and the evidence ledger. A focused test
may close a sub-item, but cannot close the parent gate when fresh-process,
SIGKILL, concurrency, or real-provider evidence is still absent.

## A. Runtime Integration

### A1. Workflow design

- [x] Define one versioned OCR workflow ID in the existing parser workflow
  design, not a second orchestration framework.
- [~] Add graph-defined nodes for page execution, grouping, semantic parsing,
  finalization, and `end`. The root and child designs exist; the named
  prepare/commit/validation sub-stages still need explicit release evidence.
- [~] Add explicit default and failure edges for page iteration, retry,
  interruption, completed-page reuse, grouping completion, and summary failure.
- [ ] Make the page loop bounded by persisted page numbers and source revision,
  never by an in-memory list that disappears on process exit.
- [x] Keep semantic parsing in the existing ingest workflow and connect the
  accepted OCR/grouping output through a typed state boundary.
- [x] Reuse `WorkflowRuntime`, `MappingStepResolver`, runtime predicates,
  persisted step execution records, and checkpoint tokens. Do not create an OCR
  runtime or parallel checkpoint protocol.

### A2. Typed runtime state

- [~] Define typed JSON state for document ID, source revision, page cursor,
  total pages, accepted page artifact IDs, grouping revision, summary policy
  revision, and final workflow-input reference.
- [x] Keep callables, provider clients, API tokens, and open file handles in
  runtime dependencies only; never persist them in checkpoint state.
- [~] Persist enough immutable request configuration to reconstruct the same
  provider/config fingerprint after a new process starts.
- [ ] Reject a resume when source revision, OCR configuration, workflow
  version, or grouping policy is incompatible with the saved run.

### A3. Page execution and recovery

- [~] Execute one page as a durable runtime step with a deterministic page key.
  The page child runtime exists; interruption and duplicate-delivery evidence
  are still required.
- [~] Publish page JSON atomically before marking the page step complete.
  Atomic publication exists; crash-window reconciliation is open.
- [~] Record artifact digest, source image digest, provider fingerprint, schema
  version, and attempt index in the durable state.
- [ ] Make retry and completed-page reuse idempotent under duplicate delivery.
- [ ] On process interruption after page commit, resume at the next incomplete
  page without repeating accepted OCR or deleting a complete grouping result.
- [ ] On process interruption before page commit, detect the incomplete step and
  retry it safely.
- [ ] Support concurrent duplicate resume attempts with one accepted commit and
  an explicit stale-token/claim failure for the loser.

### A4. Grouping execution

- [x] Run grouping inside a persisted runtime node after the accepted page
  frontier is available.
- [~] Persist grouping input identity: source revision, accepted page set,
  source-item digest, grouping policy version, and merge revision.
- [x] Restore the latest valid grouping checkpoint before processing new input.
- [~] Use CAS/expected revision or an equivalent monotonic generation guard;
  last-writer-wins must not overwrite a newer accepted tree.
- [ ] Persist an incremental merge frontier so long documents do not rebuild the
  entire prefix after every page.
- [ ] Preserve meaningful groups, IDs, summaries, source slices, and reading
  order when a frontier has one remaining group.
- [ ] Reject invalid grouping proposals without enqueueing rejected children or
  replacing the verified parent.

### A5. Summary execution

- [~] Run summaries after the grouping pass is accepted in runtime state; the
  final SQLite tree publication currently occurs in the following persistence
  node, so this is not yet a separate committed-revision gate.
- [ ] Persist summary input identity including exact subtree slices, tree
  revision, grouping policy, summary policy, provider/model revision, and prompt
  schema revision.
- [ ] Reuse a summary only when all those identities match.
- [ ] Make summary failure retryable without invalidating OCR or grouping.
- [ ] Ensure duplicate summary delivery cannot create duplicate accepted summary
  records.

### A6. Runtime verification

- [~] Add a test that inspects persisted Kogwistar runtime step nodes for OCR,
  grouping, and summary stages.
- [ ] Add a test that kills execution after page 63 of 100, reopens the same
  engines and run ID in a new process, and resumes from the persisted token.
- [ ] Add a test that kills execution after a grouping checkpoint and proves the
  checkpoint is restored before further pages are accepted.
- [ ] Add a duplicate-resume race test and assert one accepted commit.
- [ ] Add a no-provider-call-on-reuse test for already committed pages.
- [ ] Assert the final runtime graph contains no orphaned stage executions,
  duplicate accepted commits, or skipped page cursor transitions.

## B. Strict Seam Closure S01-S12

### S01. PDF revision to rendered page bytes

- [x] Include PDF SHA-256, renderer implementation/version, DPI, rotation,
  crop/configuration, page number, and raster schema in render identity.
- [ ] Store rendered-page digest and render identity in durable state.
- [x] Refuse a rendered-page cache hit when any render identity differs.
- [ ] Reconcile existing rendered files against state before reuse; stale files
  must be regenerated or quarantined.
- [ ] Test same-path PDF A to PDF B with changed page 2.
- [ ] Test DPI and rotation changes with otherwise identical PDF bytes.
- [ ] Test changed page count and removed pages.

### S02. Source bytes to committed page JSON

- [~] Store and enforce the OCR provider/config fingerprint with page state.
  The reuse predicate compares the current fingerprint set, but the release
  gate still needs a fresh-process regression proving that a model/engine
  change cannot reuse a completed page from an older configuration.
- [~] Store page artifact SHA-256 and verify it before reuse. The state store
  rejects missing or mismatched digests, but the reviewer-required release
  evidence for a well-formed tampered JSON artifact must still be recorded
  against the exact reuse path.
- [~] Write JSON to a same-directory temporary file, flush it, and atomically
  replace the final artifact before recording completion. The write path is
  atomic, but rename-before-database and database-before-rename crash-window
  reconciliation are not yet proven.
- [~] Validate and reconcile an artifact left behind by a crash between rename
  and database commit. A valid rename-before-DB orphan is adopted only when
  its source-image and OCR-configuration identities match; DB-before-rename
  and fresh-process crash-order tests remain open.
- [~] Test truncated JSON, valid JSON from another page, schema-version change,
  and crash-window recovery. Rebuild now validates `SplitPage` and leaves
  malformed/ambiguous artifacts pending; crash-window reconciliation remains.
- [ ] Ensure the artifact records the exact source image digest and accepted OCR
  configuration revision.

### S03. OCR response to spatial/media grounding

- [~] Docling provenance boxes are used when exposed; the no-item fallback is
  still page-wide and requires explicit page-level accuracy.
- [ ] Add a provider capability indicating whether coordinates are page,
  region, token, table-cell, or unknown precision.
- [ ] Never label a synthetic full-page box as region precision.
- [ ] Preserve media identity and location for figures, tables, and captions.
- [ ] Record every rejected/lost non-text object with a structured diagnostic.
- [ ] Add a mocked Docling fixture with two text regions, a table, and a figure.
- [ ] Add live-provider evidence for at least one multi-object document.

### S04. Cluster order to multimodal reading order

- [x] Normalize text and non-text objects into one explicit source traversal in
  the response models and document adapter. Focused tests cover mixed text and
  figure order; downstream grouping/graph preservation remains open.
- [x] Validate that `meaningful_ordering` covers every accepted cluster exactly
  once, including non-text clusters. Duplicate IDs, missing IDs, and empty-page
  agreement are rejected by the typed model/adapter boundary.
- [~] Preserve order through normalization, grouping, serialization, and graph
  export. Normalization is tested, but grouping/export and a real multi-object
  provider fixture are not yet evidenced.
- [ ] Represent ambiguous provider order explicitly instead of silently
  appending non-text objects after text.
- [~] Test text A, figure, caption, text B and deliberate child permutation.
  The current test covers text/figure/text; caption permutation and rejection
  diagnostics remain open.

### S05. Accepted group tree to semantic parser input

- [ ] Define the typed parser input contract for accepted group hierarchy,
  avoiding an unbounded duplicate tree inside model prompt metadata.
- [ ] Prove that the semantic parser consumes the accepted hierarchy rather than
  only receiving it as an ignored metadata sidecar.
- [ ] Prove that final graph export preserves intended group relationships,
  ordered slices, and source grounding.
- [ ] Add a metamorphic test using identical source items with two valid group
  trees and distinguishable parser/export results.
- [ ] Add an explicit compatibility mode for consumers that ignore hierarchy;
  fail closed where hierarchy is required.

### S06. Grouping pass to durable checkpoint

- [~] Stop prefix pages from overwriting the canonical complete tree. The
  preparation path still writes a baseline SQLite tree before the grouping
  runtime; this is not yet a monotonic runtime-owned checkpoint. The reviewer
  finding remains open until generation/CAS ownership is exercised by the
  persisted grouping child.
- [ ] Add monotonic checkpoint generation and expected-revision/CAS protection.
- [ ] Persist source fingerprint, grouping policy, accepted page frontier, tree
  digest, and last accepted runtime step.
- [~] Restore before advancing and prove byte-equivalent no-op reruns. The
  grouping runtime has a passing resume test; OCR/page and dual-store resume
  remain unproven.
- [ ] Test crash at page 63 and concurrent late writers.

### S07. Grouping to summary generation

- [ ] Run summaries after grouping passes, not before caller-provided grouping.
- [ ] Include exact subtree slice membership, tree revision, policy revision,
  model revision, and prompt schema in the summary fingerprint.
- [ ] Test two nodes using different slices of the same source item.
- [ ] Test grouping-only changes invalidate summaries without rerunning OCR.
- [ ] Test identical policy/input reuses summaries without another provider call.

### S08. Dynamic grouping to semantic identity

- [ ] Preserve a meaningful final group when the frontier has one group; only a
  synthetic root may be collapsed under an explicit rule.
- [ ] Make group ID, kind, provenance, and summary transformations explicit and
  versioned.
- [ ] Test a title plus atomic long table retaining its custom group and both
  children.
- [ ] Test replay/idempotence of the same grouping proposal.

### S09. OCR/group/summary to Kogwistar checkpoint

- [~] Move all three stages into one persisted Kogwistar workflow run. The root
  design now invokes page, grouping, and semantic child runs, but SIGKILL and
  duplicate-resume evidence is still open.
- [~] Keep SQLite as an idempotent artifact cache, not the authoritative cursor.
  The root cursor is runtime-owned; dual-store reconciliation is still open.
- [~] Persist step tokens and stage results in the existing conversation/runtime
  graph.
- [ ] Test SIGKILL/reopen/resume in a new process with the same run ID.
- [ ] Test duplicate resume, late writers, and no double publication.

### S10. Source revision to downstream graph identity

- [~] Replace static `ocr-v1` source revision with content-addressed or
  authoritative revision identity. The current document revision includes the
  input fingerprint, but same-stem/cross-path identity and immutable revision
  separation still lack acceptance tests.
- [ ] Separate document ID, source revision, OCR config, grouping revision, and
  parse generation namespaces.
- [ ] Make default document IDs collision-safe for same-stem files.
- [ ] Test two `report.pdf` files from different folders in one batch.
- [ ] Test changed content at the same path and verify immutable revision
  separation.

### S11. Explicit references across boundaries

- [ ] Support references spanning clusters, lines, and pages.
- [ ] Preserve multiple exact source slices for one reference mention.
- [ ] Keep references outside the primary coverage partition unless explicitly
  promoted by a later workflow.
- [ ] Test a reference split across OCR clusters and across pages.

### S12. Public entrypoint to internal options

- [x] Thread grouping policy version through the public runner into parsing,
  grouping, checkpoint, summary, and export layers.
- [ ] Expose summary policy/generator consistently at the public runner.
- [ ] Reject conflicting public/internal policy values rather than silently
  falling back to `baseline-v1`.
- [ ] Test explicit custom policy through the public API and inspect persisted
  metadata, cache keys, and final output.

## C. Release gates added by the implementation review

- [x] **C01 — Installed API compatibility:** record the installed Kogwistar
  version and verify the exact persisted invocation/checkpoint APIs used by the
  parser on CPython 3.12-3.14 and PyPy 3.11. Parser CI run `38655395569`
  passed this matrix.
- [~] **C02 — Runtime ownership boundary:** the root run owns page, grouping,
  and semantic child invocations with stable child IDs. Prove recovery of the
  complete parent/child lineage before release.
- [ ] **C03 — Dual-store reconciliation:** test crashes after artifact rename
  and after SQLite/runtime checkpoint writes in both orders; recovery must
  accept one canonical result or safely redo an unaccepted attempt.
- [ ] **C04 — Incremental equivalence:** compare canonical tree content, not
  timestamps, between uninterrupted batch grouping and interrupted incremental
  grouping. Completed subtrees must remain stable.
- [~] **C05 — True resume:** prove completed OCR, grouping, and summaries are
  not rerun after a failed child checkpoint. The same-process 100-page
  interruption/resume test is evidence for retryable child recovery; a new
  process/SIGKILL proof, duplicate-resume race, and artifact/database
  reconciliation remain open. The acceptance target is at-least-once execution
  with exactly one accepted canonical result, not exactly-once provider calls.
- [~] **C06 — OCR quality/completeness:** interval coverage is necessary but not
  sufficient. `OCRQualityPolicy` records typed provider diagnostics and can
  reject incomplete text, edge-word loss, or excessive data-loss likelihood;
  provider-side missing-region detection, unsupported-coordinate handling, and
  persisted strict-policy evidence remain open.

### Reviewer acceptance matrix

The following matrix is the required status ledger for the external review.
It deliberately records the current gaps instead of treating the checklist
itself as proof:

| Gate | Current status | Required closure evidence |
| --- | --- | --- |
| C01 installed API compatibility | PASS | Installed-package contract test and parser CI matrix `38655395569`; source-checkout-only tests do not count. |
| C02 runtime ownership boundary | PARTIAL | Root/page/grouping/semantic child IDs exist, but a fresh-process recovery test must prove one parent lineage and no second semantic generation. |
| C03 dual-store reconciliation | OPEN | Crash tests for rename-before-checkpoint and checkpoint-before-rename, plus tampered JSON rejection and deterministic recovery. |
| C04 incremental equivalence | OPEN | Canonical tree comparison between uninterrupted batch and interrupted incremental grouping; exclude timestamps/run metadata. |
| C05 true resume | PARTIAL | Same-process 100-page interruption evidence exists; SIGKILL/new-process resume, duplicate-resume race, and no redundant accepted work remain. |
| C06 OCR quality/completeness | PARTIAL | Typed provider diagnostics and strict policy exist; provider-side missing-region/coordinate evidence and reopen-after-failure proof remain. |

### S02 correction ledger

S02 must not be marked complete until all of the following are independently
demonstrated:

- [~] configuration fingerprint changes prevent reuse in a fresh process;
- [~] a well-formed but tampered OCR JSON is rejected by the reuse path;
- [~] atomic rename is used for publication;
- [ ] both artifact/checkpoint crash orders reconcile without accepting a
  partial or stale result.

The current implementation has atomic publication, digest fields, and a
rename-before-database orphan-adoption path. These are implementation facts,
not yet a complete S02 release proof.

## C. Required End-to-End Acceptance Fixture

- [ ] Generate a deterministic 100-page PDF fixture containing:
  - a long atomic table;
  - duplicate paragraphs at distinct positions;
  - text, figure, caption, and table ordering;
  - page-labelled front matter;
  - a paragraph crossing pages 62-63;
  - a cross-page explicit reference;
  - a changed page 40 in a second source revision.
- [ ] Run an uninterrupted reference execution with a deterministic fake OCR
  provider.
- [ ] Kill after page 62 commit, reopen the persistent Kogwistar engines, and
  resume with the same run ID.
- [ ] Kill after a summary commit and resume again.
- [ ] Compare source interval identities, ordered traversal, grouping revision,
  summaries, runtime trace, and final graph with the uninterrupted reference.
- [ ] Require zero duplicate accepted commits, no stale artifacts, no fabricated
  spatial precision, and no unsupported cross-revision reuse.

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

The current focused root-runtime test proves parent/child invocation and
terminal semantic reuse with a mocked provider. It does not yet prove
new-process interruption, duplicate-resume exclusion, or the dual-store commit
protocol.

## D. Evidence And Release Gate

- [ ] For every S01-S12 item, record source lines, pre-fix failing test or
  demonstrated risk, post-fix passing test, and whether it exercised the real
  Kogwistar runtime or only a standalone adapter.
- [ ] Run Ruff with the repository CI selection.
- [ ] Run Pyright with supported CPython/PyPy-compatible syntax.
- [ ] Run `git diff --check` and lock verification.
- [ ] Run focused seam tests with `-p no:cacheprovider` and repository-local
  basetemp where Windows temporary-directory permissions require it.
- [x] Run the full parser CI suite on CPython 3.12-3.14 and PyPy 3.11.
  Parser CI run `38655395569` passed; slow/manual provider experiments remain
  separate gates.
- [ ] Only after all release gates pass, update version/release notes and pins.
