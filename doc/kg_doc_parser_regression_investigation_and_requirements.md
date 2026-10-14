# KG Doc Parser — Upstream Correctness, Recovery, and Regression-Prevention Handoff

**Status:** executable engineering specification + investigation checklist
**Prepared:** 2026-10-14 (Australia/Brisbane)
**Target repository:** https://github.com/humblemat810/kg-doc-parser
**Primary audience:** coding agent working **in that repository** with Git history, source, test runner, and CI files available.
**Priority:** P0 correctness/data-loss and invariant preservation; P1 triage/cache/checkpoint integration and observability.
**Important:** This document is a **requirement**, not proof that each suspected bug persists at current `HEAD`. Pin `HEAD` and reproduce every claim before changing code. Do not rely on `main` URLs or line numbers as immutable evidence.

---

## A. Execution prompt — paste into the coding agent

> You are responsible for investigating and repairing **kg-doc-parser** without weakening any previously established parser requirements. Work in the local Git checkout, fetch latest branch history if permitted, **record the exact commit SHA**, run tests, and produce a reproducible root-cause analysis (RCA) **before and after** making changes.
>
> The user has repeatedly required: **no semantic tree node may have exactly one child**, universally at **all depths and paths**. The handling depends on the parsing strategy:
>
> - **PageIndex-style** (heuristic, provider-backed/LLM, one-layer PageIndex fallback): deterministic, **lossless structural morphism** for any `Parent → Child` unary relation. If the parent owns distinct grounded source content, preserve that content as an explicit sibling leaf. If the parent is a redundant content-free wrapper, safely promote/collapse it with identity/provenance remapping. Otherwise fail; **never silently discard children/grandchildren or invent source text**. An atomic table may remain one TABLE item. `Title → Table` is merely one example, **not** the scope of the invariant.
> - **Boundary/excerpt LLM proposal modes:** unary split is an invalid proposal. Run **CUD** review/correction/update (user also called it CRD; verify the project's actual naming), **revalidate**, retry up to the configured finite budget, then **fail that strategy/attempt explicitly** if still invalid. A different strategy may be tried only as a **logged, intentional fallback**; never silently convert a failed unary split into a successful atomic result.
> - **All paths:** tree shape, source ownership/coverage, char-span grounding, identity/reparenting, and cycle/duplicate checks must hold at commit, materialization, final export, and checkpoint recovery.
>
> Also investigate the linked deficiencies: a downstream-reported 70-deep unary chain; table subtree lost by PageIndex `atomic_retained`; missed/unused LLM triage; PageIndex LLM response cache not wired; standalone checkpoint/resume not enabled; possible contradictory, weakened, deleted, or `manual`-excluded tests. **Trace when and why these escaped**, including old commits and CI history. Do **not** claim a test was disabled deliberately or blame an agent without a verifying diff.
>
> Deliver implementation, mandatory offline CI regressions, a dated commit-by-commit RCA, an entry-point coverage matrix, command outputs, and a fail/pass proof against the baseline and fixed revision. Do not silently change the specification to make tests pass. If any item is already fixed in today's code, **mark it verified fixed** and add/retain protection tests rather than duplicate the implementation. If a required invariant conflicts with architecture, explicitly report the conflict, preserve data, and fail safely.

## B. User-established architectural contract (normative, non-negotiable)

### B1. Universal minimum branching — **P0**

For **every** semantic tree node `n`, including document root, pages, headings, synthetic containers, and deeper descendants:

```
number_of_children(n) == 0 OR number_of_children(n) >= 2
```

A tree with `Document → Section → …` containing 70 successive unary nodes **must never pass** final success/export, even if max-depth permits it or pointers happen to validate.

- [ ] Apply to **all source formats, node types, and depths**; no quiet root/page/heading exemptions.
- [ ] `0 children` is a legitimate **leaf**, but only with valid source grounding/explicit justified atomic state according to existing contracts.
- [ ] `1 child` is invalid in final semantic trees **regardless of whether the child's span differs from the parent's span**.
- [ ] A single-child proposal that repeats the parent's full span is independently invalid as a **non-progressing refinement**.
- [ ] Check the invariant **after** deduplication, pointer repair, node filtering, heading conversion, and any structural normalization—never only before transformations.
- [ ] Reject cyclic graphs, repeated IDs, invalid parent references, or stale depth/level metadata; do not hide issues via a shallow traversal.
- [ ] A success status implies recursive validation **on the persisted/exported tree**, not merely a proposal-level check. Note: current source already has `validate_semantic_tree_shape()` / `validate_semantic_tree_integrity()`; audit **all call sites** and what transformations happen *before* those gates.

### B2. PageIndex: universal deterministic unary-node **morphism** — **P0**

This applies to **any** `Parent → Child`, not only `Heading → Table`.

**Case 1: parent owns distinct, independently grounded content.**

```
Before:
  Parent {own source P}
    └── Child {source C, possibly descendants}

After:
  Parent [structural container]
    ├── ParentContent [grounded leaf for exactly P]
    └── Child {source C, descendants all preserved}
```

Examples: title+table, title+paragraph, section+figure, paragraph+subcontent, heading+nested section. The *display title or model-generated label is not evidence*: the new leaf must use exact **independent source text and offsets** actually owned by the parent, not duplicated inherited/aggregate pointers.

**Case 2: parent has no independently owned content and is only a wrapper.**

```
Before: Parent [no exclusive source] → Child → [A, B]
After:  Child → [A, B]  (or equivalent safe promotion)
```

Promotion is allowed **only if** the semantic contract permits removal of that wrapper and all parent references, IDs, levels, metadata, descendants, and source provenance are remapped consistently. A mandatory root wrapper may not be removed without an explicit representation-safe root transformation; if no safe solution exists, fail rather than allow a unary root.

**Case 3: neither transform can safely preserve identity/semantics/grounding.**

Reject the structural decomposition, retaining the previous **valid committed state** (or emit explicit failure). **Do not** delete the child/grandchildren, copy only aggregate pointers onto the parent, invent dummy siblings, duplicate evidence, or force an atomic source object into rows.

- [ ] Put the shared PageIndex morphism at an appropriate deterministic semantic boundary, reused by direct LLM, direct heuristic, and one-layer fallback/materialization paths; no divergent ad hoc normalizers.
- [ ] Preserve exact source identity `(source_cluster_id, start_char, end_char, verbatim_text)` with correct **inclusive/exclusive convention** and line mapping.
- [ ] Preserve **all** descendants and structural/semantic relationships; only change structure when provably safe.
- [ ] Preserve title/heading **as an actual grounded leaf** if independently source-bearing, not simply as a label.
- [ ] Keep an indivisible `TABLE` atomic **when the table itself is one legitimate source object**. Row extraction is optional separate semantic refinement, not required to satisfy branching.
- [ ] Do not indiscriminately reparent all content under headings: legitimate `TABLE → TABLE_ROW`, list→item, and other typed containment may be meaningful. If an invalid content-under-content relationship is proposed, inspect its **source ownership and semantic role** before repair.
- [ ] Normalization is **deterministic** and **idempotent**: `N(N(T)) == N(T)` (use structural+pointer canonicalization, not incidental runtime IDs).
- [ ] Normalization is **lossless**, preserving source-owner coverage and all supported descendant evidence. Aggregate pointers may summarize descendant coverage but **cannot substitute for missing leaves or hide discarded descendants**.
- [ ] A heading-only document/node with no legitimate content children becomes a grounded leaf; don't synthesize an empty sibling.
- [ ] PageIndex fallback obeys the **one-layer-at-a-time** contract. After a one-layer fallback, ordinary strategy selection may resume at the child frontier; PageIndex must not silently parse to the bottom in one operation.

### B3. Boundary and excerpt: **CUD corrective retry**, not PageIndex morphism — **P0**

```
propose (boundary or excerpt)
   → deterministic validation + review
       → valid: commit
       → invalid (including one child): CUD review/correction/update
           → revalidate → if valid, commit
           → if invalid, retry within finite budget
           → budget exhausted: explicit FAILED attempt/strategy
               → optionally record transition to another enabled strategy
               → if all authorized strategies exhausted: FAILED parse
```

- [ ] Reject unary proposals at source adapter **and** shared layer/commit boundary.
- [ ] Route invalid proposal into actual **review/update correction** where configured (verify `review_cud_proposal`, `apply_cud_update`, correction round configuration); no mere immediate silent fallback that makes the original strategy appear successful.
- [ ] Count proposal retries, correction rounds, review failures, timeouts, and fallback transitions explicitly and independently where relevant. All budgets finite, testable, and visible in diagnostics.
- [ ] Enforce the retry limit; no unbounded loops, repeated identical-content same-span re-enqueue, or infinite correction-at-same-frontier.
- [ ] After exhausted retries, mark the **specific method FAILED** with reason and full trace; a different enabled method can be attempted only via explicit strategy transition, independently validated.
- [ ] Do **not** reinterpret an invalid attempted unary split as `atomic_retained` or successful parsing to bypass corrections. Truly indivisible content may be a **legitimate unsplit leaf** only through an explicit, separately validated decision, not as error laundering.
- [ ] Never commit a failed correction; transactionally preserve last known valid state.
- [ ] A provider timeout/schema failure cannot be conflated with valid LLM semantic review. If deterministic checks authorize a result under documented policy, the trace must truthfully distinguish that from completed LLM review.

### B4. Universal source and identity invariants — **P0**

- [ ] Every source-bearing leaf has validated grounding at **character-span** level, not just a matching excerpt string. For multimodal inputs, use the authoritative modality-specific anchor and contract; don't fabricate text spans.
- [ ] Coverage is measured against the authoritative expected source intervals (and explicitly documented exclusions), independently from the current tree's self-reported aggregate pointers.
- [ ] No gap, unexplained overlap, duplicate logical source ownership, misplaced span, silent truncation, or lost grandchild after morphism/correction/fallback.
- [ ] Repeated **identical text at different source positions** is legal and must remain distinguishable; never reject simply because normalized sibling text matches.
- [ ] Multi-page sources and table-heavy documents retain page boundaries and source identity; whitespace/separators follow a documented ownership policy.
- [ ] Parent exclusive content and descendant content do not double-count ownership; inherited/aggregate pointers are tracked separately.
- [ ] Final tree succeeds **only after** structural, source-pointer, coverage, and reference integrity validation. The current `compute_terminal_content_coverage()` reports `missing_nonws_ranges` and `multiply_owned_nonws_ranges`: verify this exact leaf-ownership metric is checked as a **hard success gate**, not only reported with `partial_degraded` status. A structurally valid, content-losing tree is still invalid.
- [ ] Use `max_depth` only as a safety bound, **never** as a substitute for minimum branching/progress.

---

## C. Required fault-origin investigation (historical RCA) — **P0**

**Do this before coding.** The exact timeline and the fate of tests are **not established** by the old chat or version `0.2.15` downstream reports; investigate, do not assume.

### C1. Pin the state and reproduce

- [ ] In working checkout: `git status --short`; `git fetch --all --tags --prune` if allowed; `git rev-parse HEAD`; `git log -1 --format='%H %cI %s'`.
- [ ] Record repository origin, current branch, latest release/tag, Python/Kogwistar versions, current date, and whether working tree is dirty; don't overwrite local work.
- [ ] Read existing `AGENTS.md`, `agent.md`, authoritative design/ADR and historical minimum-branching requirement; extract **verbatim requirement wording and commit/file evidence**.
- [ ] Construct **minimal reproducible fixtures**: (i) `Title → atomic Table`; (ii) arbitrary content parent→single content child with grandchild; (iii) **70-deep unary chain**; (iv) a valid branching control; (v) distinct source ranges with identical text.
- [ ] Reproduce on current pinned `HEAD` via public Python APIs and CLI, not solely private functions. Capture output tree, coverage, events, exact entry point, provider mode, and whether failure is correctly explicit.
- [ ] Classify each suspected defect **reproduced / already fixed / partially fixed / not reproducible / uncertain**.

### C2. Find when the protection was introduced, bypassed, or contradicted

- [ ] Examine `git log --all --follow`/`git blame` and diffs of `page_index.py`, `parser_core.py`, `layerwise_llm.py`, `semantics.py`, `handlers.py`, `design.py`, `tests/`, `pytest.ini`, and `.github/workflows/ci.yml` (paths may move).
- [ ] Search old and new code for `single child`, `unary`, `min_children`, `heading_text`, `atomic_retained`, `child_nodes=[]`, `total_content_pointers`, `structural_issues`, `coverage`, `retry`, `fallback`.
- [ ] Find the **last good commit** for each failing *execution path*, the **first bad commit** if provable, and the exact code diff that changed behaviour. Use `git bisect` with a deterministic test where feasible.
- [ ] Investigate the previously discussed candidate refactor **`ca90f2a` (2026-04-08)** and strengthened LLM proposal validation **`c8d3369` (2026-06-04)**; treat these as **leads to verify**, not proven sole causes.
- [ ] Determine whether safeguards were **removed, never implemented for the new path, or retained but bypassed** after a refactor; distinguish the three clearly.
- [ ] Inspect history of `test_propose_layer_fn_rejects_single_child_fake_split`, `test_layerwise_workflow_retries_when_cud_coverage_check_fails`, and relevant title/table fixtures. Confirm **which exact assertions** passed and what they actually proved.
- [ ] For each missing regression: check **was test deleted? weakened? renamed? moved to `manual`? excluded by marker/CI selection? mocked away? expecting success from an invalid fixture?** Cite `git diff` commits, not conjecture about intentions.
- [ ] Check if CI always ran the relevant files and markers on the affected commits; test presence ≠ test execution. A green workflow is not proof of semantic invariants.
- [ ] Review any commit/test code that treats `coverage == 1` and `children nonempty` as sufficient: these are **not** substitutes for recursive shape validation.
- [ ] Check changes in the **last 24–48 hours** specifically, then re-run against the current SHA. Do not use an old release's behaviour to assert current defect.
- [ ] Report **evidence**, **mechanism**, **contributing testing/process gap**, **scope**, and **confidence**. Do not assign personal blame or claim tests were deliberately disabled absent evidence.

### C3. Required RCA deliverable per distinct defect

For each issue: `ID`, severity, current SHA, affected public entry points, minimal fixture, expected/observed, earliest verified good/bad commits + timestamps, relevant diff/lines, existing guard(s), bypass route, test-history evidence, CI selection evidence, root cause, preventive guard, regression test name, and before/after command+result. If no first bad commit is established, state **UNKNOWN**.

---

## D. Entry-point inventory and propagation matrix — **P0/P1**

- [ ] Enumerate every supported public path (actual names from pinned source): standalone `workflow-ingest page-index`, `workflow-ingest layerwise`, `workflow-ingest ocr`, direct `parse_page_index_document`, `parse_page_index_layer`, `parse_tree_document`, `run_ingest_workflow`, OCR-to-semantic runner, legacy API, demo harness, adapter/server path, and LLM-Wiki usage if available.
- [ ] For each path, record: parser mode; whether LLM is called; triage invocation and reason; CUD/retry/fallback; morphism; cache; checkpoint save/resume; final validation; persistence/export.
- [ ] Build a test per **supported** combination with a correct expected contract; deprecated/unsupported paths must be explicitly documented as such instead of silently diverging.
- [ ] Do not treat a unit test of an internal helper as proof that the corresponding real public caller reaches the helper.
- [ ] Treat partial-commit, provider timeout, review-skipped, deterministic fallback, and resumed-from-checkpoint paths as independent possible invariant bypasses.

Suggested table to fill:

| Public path | LLM/triage | Unary handling | Cache | Resume | Final guard | CI integration test |
|---|---|---|---|---|---|---|
| Direct heuristic PageIndex | ... | deterministic morphism | N/A | ... | ... | ... |
| Direct provider PageIndex | ... | deterministic morphism | ... | ... | ... | ... |
| PageIndex layer fallback | ... | deterministic one-layer morphism | N/A or mode-dependent | ... | ... | ... |
| Excerpt | ... | CUD+bounded retry/fail | ... | ... | ... | ... |
| Boundary | ... | CUD+bounded retry/fail | ... | ... | ... | ... |
| CLI, OCR, workflow, adapters | ... | shared final guard | ... | ... | ... | ... |

---

## E. Triage/strategy-selection correctness — **P1, integration contract**

The repo has discussed/implemented `ParseStrategyTriage`, `select_parse_strategy`, and a workflow triage step; **presence of an implementation does not establish execution from each public entry point**.

- [ ] Verify whether automatic per-frontier strategy selection runs an **LLM triage** when configured, or just deterministic preference order, and why.
- [ ] Preserve the independent per-layer/frontier decision: excerpt, boundary, or PageIndex based on **that unit** (not one global choice for the whole document).
- [ ] Record `selected_strategy`, source (`llm_triage`/configured/deterministic/fallback), available alternatives, reason, confidence (where applicable), provider call status, and policy/budget.
- [ ] After a failed strategy, record disable/failure and **re-triage/next authorized method**; never mask a failure as method success.
- [ ] Verify automatic provider settings actually reach the triage handler in standalone APIs and CLI; if triage is intentionally off, expose the reason clearly.
- [ ] Test first-time triage, low-confidence triage, provider error, explicit override, partial layer failure, and method exhaustion **through public entry points**.
- [ ] Do not silently change heuristic PageIndex fallback to LLM PageIndex merely to manufacture a triage event; the mode/runtime must be explicit.

---

## F. Table-heavy PageIndex and subtree preservation — **P0**

Downstream v0.2.15 report: `_split_page_blocks` treated a large Markdown table as one `TABLE` candidate and a unary normalizer dropped a content subtree. Reported input had `coverage≈0.04895`, 3 nodes, no table rows before local patch; `coverage=1.0`, 55 nodes after local patches. **These numbers are downstream observations; independently reproduce against current pinned HEAD.**

- [ ] First repair **lossless ownership/morphism and coverage**; do not assume forced row segmentation is the primary solution.
- [ ] Keep a full large table as a single grounded atomic table if that is what the source and requested parsing granularity represent. `Title → Table` becomes a two-child structure of grounded parent text + intact table.
- [ ] If table rows are later requested as **optional** subdivision, implement source-position-exact `TABLE_ROW`/`TABLE_ROW_GROUP` segmentation with header/separator mapping, and test that change independently.
- [ ] Never drop grandchild content in an `atomic_retained` branch. Inspect use of `model_copy(update={"child_nodes": []})` and whether it can delete a descendant tree after only copying the child's aggregate/total pointer.
- [ ] Never convert coverage from <1 to 1 by redefining expected source to the surviving pointers; check against original source interval manifest. Verify an exporter never treats `partial_degraded` or a missing-terminal-ownership result as **complete success**.
- [ ] Test content→content relationships that are **valid semantics** separately from invalid LLM nesting. Do not globally force content blocks to become unrelated siblings under the nearest heading.
- [ ] Preserve exact paragraph/table page ordering and relationships; handle headings that are real evidence versus LLM-only display titles.

---

## G. LLM response caching: standalone and PageIndex — **P1**

Downstream v0.2.15 report: `WorkflowLLMCallCache` exists, demo harness constructs it, other production entry points may not; provider-backed PageIndex `structured.invoke` historically bypasses the parser cache. This is separate from server-side prefix/KV caching.

- [ ] Verify all actual LLM operations in PageIndex (block assignments, excerpt refine, summary) and excerpt/boundary propose/review/triage; enumerate cache coverage **per call**, not per document only.
- [ ] Wire the shared existing response cache into applicable public production entry points, not only demo/test harness; provide explicit documented configuration **and usable default where persistent state exists**.
- [ ] For deployments without writable persistent storage, provide explicit ephemeral/disabled status, never pretend reuse works; honor privacy/permission boundaries.
- [ ] Fingerprint exact relevant input: source revision + slice/span, operation, prompts/prompt version, response schema version, provider/model/decoding settings, semantic mode, retries/context, namespace/tenant isolation.
- [ ] Revalidate cached payload through **current schema, source grounding, structural invariants** before use; malformed or old-schema cached responses must never bypass current checks.
- [ ] Atomic writes; no partial/failed/time-out results cached as success. Ensure cache not treated as authoritative parser state.
- [ ] Test repeated identical call (one provider execution), new-process reuse, changed source/provider/prompt/schema invalidation, isolation, invalid cached record, concurrent calls, per-operation cache telemetry.

---

## H. Checkpoint + resume without LLM-Wiki — **P1**

Downstream v0.2.15 report and prior API inspection: semantic workflow may checkpoint via underlying Kogwistar runtime but `resume_from_checkpoint=False` and randomly new run identity typically prevent automatic recovery; direct PageIndex and legacy parser historically have no integrated semantic resume. OCR's page-level resume is distinct.

- [ ] Map checkpoint **write** versus **read/resume** for all public paths. Do not claim `checkpoint_every_n_steps=1` means effective recovery.
- [ ] Standalone parsing **must not require LLM-Wiki** to persist/recover supported long-running semantic work.
- [ ] Provide a **stable explicit/recoverable run identity** and compatible source/config identity; avoid `new uuid` defeating resume. Handle concurrent runs and intentional fresh runs distinctly.
- [ ] Choose/document safe default resume when persistent state is configured; expose flags to start fresh, disable cache, or opt out. No hidden reuse after source/config/provider/schema change.
- [ ] Persist at meaningful semantic boundaries (accepted frontier/layer/page/unit), never at inconsistent half-commit state. Restore the last valid state; revalidate shape, pointers, and source fingerprint when resuming.
- [ ] Distinguish **checkpoint resume** from reuse of completed LLM responses within a partially failed page (cache solves latter). On recovery, incomplete LLM calls may be executed again.
- [ ] Verify OCR→semantic transition, direct PageIndex, workflow, CLI, Python APIs, and server-backed adapter; report unsupported paths explicitly and avoid promising unsupported recovery.
- [ ] Test deterministic crash/injected failure after a committed stage, restart in a fresh process with the same run identity, exact reuse of completed steps, no stale state, and final valid output.
- [ ] If external side effects can run twice around checkpoint boundaries, use idempotency/transactional keys and test replay safety.

---

## I. Required mandatory offline CI regression matrix — **P0**

**All correctness tests below must be deterministic, fast enough for ordinary CI, and avoid a live LLM requirement using scripted fake providers.** Add a separate optional real-model smoke suite if useful, never as the only protection.

### I1. General tree shape and morphism

- [ ] **Property**: every exported node has 0 or ≥2 children, recursively, including root; report the **offending path/ID** on failure.
- [ ] General `Parent → single Child` when `Parent` has exclusive grounded text → 2 children (`ParentContent`, original `Child` with descendants preserved). Vary parent and child node types; not title/table only.
- [ ] Contentless wrapper + child with ≥2 descendants: safe collapse/rebase, preserve exact pointers and semantic identity; fail when wrapper cannot be safely removed.
- [ ] Parent with overlapping/ambiguous exclusive content: fail; never duplicate spans or invent a synthetic leaf.
- [ ] 70-deep unary chain: rejected or **safely normalized where possible**; not accepted unchanged; run on both direct and final export paths.
- [ ] Title+atomic table, title+paragraph, paragraph+table, heading+figure, nested heading, content-only document, heading-only document, root→one page; each must preserve evidence or fail safely.
- [ ] Repeated normalizer `N(N(T)) == N(T)` for shape + pointers + stable identity contract.
- [ ] Source coverage/ownership before and after normalization; no lost/grandchildren; no collision in node IDs or wrong `parent_id/level_from_root`.

### I2. Per-mode behavioural tests

- [ ] Heuristic PageIndex direct API: required morphism + final validation.
- [ ] Provider-backed/LLM PageIndex with mocked outputs: required morphism + final validation + invalid nesting correction/rejection.
- [ ] PageIndex one-layer fallback: same morphism; no unexpected parse-to-bottom.
- [ ] Boundary proposal unary→CUD corrects to two grounded children within budget→commit.
- [ ] Excerpt proposal unary→CUD corrects within budget→commit.
- [ ] Boundary and excerpt: repeated invalid corrections **exhaust budget → method fails**, independently; authorized fallback logged, or parse fails if no alternatives.
- [ ] Timeout, malformed structured response, review disabled, provider failure, legacy compatibility, partial commit, cache hit, and checkpoint replay **cannot skip final invariant**.
- [ ] Same-parent interval/no-progress child never repeatedly re-enqueued.

### I3. Adversarial grounding/coverage

- [ ] Large atomic tables spanning most page; heading + table; one table with >50 rows; table as one source object (not forced split).
- [ ] If row mode offered, exact header+separator+row offsets, complete source coverage and preserved table-to-row semantics.
- [ ] Identical repeated paragraphs/rows at **different offsets** remain distinct, not false duplicates.
- [ ] Non-overlap/contiguity, leading/trailing whitespace, Windows CRLF, multi-page text, Unicode offsets, empty headings, source revision mismatch, an LLM title with no source span.
- [ ] A parse with missing/lost content must **fail coverage**, regardless of any aggregate pointer copied to the parent.

### I4. Triage/cache/resume and public API

- [ ] Automatic per-frontier triage performed or **explicitly documented deterministic bypass**; verify trace and provider call count.
- [ ] Cached PageIndex LLM operation runs once for identical inputs, survives process restart, invalidates correctly, revalidates cache hit.
- [ ] Interrupted long semantic parse resumes without LLM-Wiki from stable saved identity; no duplicate committed nodes; source-change invalidation.
- [ ] CLI and Python routes produce the same validated contract; direct integrations are not weaker than demo workflow.

### I5. Test-history/CI enforcement (anti-regression policy)

- [ ] Promote any currently `@pytest.mark.manual` **deterministic structural/grounding regression** to mandatory `ci`, or extract a small CI-safe fixture retaining the *same meaningful assertions*. Large real-provider tests may remain manual **in addition**.
- [ ] Confirm `pytest -m 'ci ... and not manual ...'` actually collects **all** mandatory tests. Save collection output; no missing test through file selector, marker, accidental deselection, xfail, skip, or conditional logic.
- [ ] Reconcile contradictory tests that previously accepted invalid one-child success: change invalid fixture or expected result **without reducing scenario coverage**.
- [ ] Prevent deletion, skipping, weakening assertions, or reclassifying invariant regressions as `manual` to make the build green. Add a CI **contract canary**/required test module with strong assertion names and a check verifying selected node IDs/tests are collected.
- [ ] Run the mandatory suite locally and in CI; include coverage showing execution of actual final guard, PageIndex morphism, CUD rejection/retry, and exported tree checks (line coverage alone insufficient).
- [ ] Mutation/fault injection proof: temporarily inject a unary final tree, delete a grandchild, or bypass the validator; **CI must fail**. Restore code afterward.
- [ ] Keep the user-approved contract in a durable design/contract document and in code-level invariant functions; **tests derived directly from that text**. Document intentional specification changes through explicit review, not incidental refactoring.

---

## J. Definition of done / release gates

No success claim unless **all** are checked with evidence:

- [ ] Source SHA and baseline/test environment recorded; current vs historical behaviours distinguished.
- [ ] Existing design contract quoted/linked; no unsupported rewriting of the requirement.
- [ ] Each defect has reproducible pre-fix case or marked not-reproducible with reasons.
- [ ] Root-cause diff/commit timeline for tests, code, and CI; unknown causes clearly marked unknown.
- [ ] PageIndex universal **deterministic, idempotent, lossless** morphism enforced on every applicable path; atomic content allowed.
- [ ] Boundary/excerpt use **bounded CUD correction and explicit failure**, not PageIndex normalization or silent atomic pass.
- [ ] All final exported/persisted trees pass minimum branching + identity + char-span grounding + source coverage (including 70-node chain fixture).
- [ ] No destructive `atomic_retained` data loss; table-rich fixtures preserve source regardless of whether rows are split.
- [ ] Public-entry-point matrix demonstrates triage, strategy handoff, cache, and resume semantics; gaps documented or fixed by scope.
- [ ] Standalone cache/checkpoint options and defaults explicitly implemented and documented, no LLM-Wiki dependency.
- [ ] Mandatory deterministic CI includes fixed regressions and proves failures when guards are intentionally sabotaged.
- [ ] No existing regression tests secretly removed, weakened, skipped or shifted to `manual`; any consolidation documented in test mapping.
- [ ] Before/after commands and captured passing results provided; `git diff` reviewed to ensure fixes do not alter unrelated behaviour.
- [ ] Issue cannot be marked resolved solely because a specific fixture passes: **whole API/path and adversarial matrix** must pass.

### Final agent report format (fill with actual evidence)

```text
BASELINE
  repository / branch / SHA / commit date:
  install and runtime versions:
  exact user-visible entry points:

HISTORICAL RCA
  old requirement source + commit:
  last verified good commit per affected path:
  first verified bad commit per affected path:
  code diff / test diff / CI selector diff:
  cause (vs symptom) + evidence + unknowns:

CURRENT-HEAD FINDINGS
  [ID] severity: reproduced/already-fixed/partially-fixed/not-reproducible
  fixture + exact CLI/API invocation:
  expected vs observed + source coverage + tree representation:
  failure mechanism and bypassed validator:

IMPLEMENTATION
  files/functions changed:
  strategy-specific handling:
  grounding/coverage/identity impact:
  backward compatibility / migration / checkpoint/cache invalidation:

REGRESSION PREVENTION
  CI test names and assertions:
  collected by CI? exact command/proof:
  mutation/fault-injection caught?:
  trace IDs / run stats:

RESULTS
  baseline test red / fixed test green:
  entire deterministic CI summary:
  real-model smoke (optional):
  remaining known limitations / follow-ups:
```

---

## K. Source leads and caveats for the coding agent

**Direct repository source inspected 2026-10-14 (URLs point to mutable `main`; replace with the pinned SHA in final RCA):**

- [`page_index.py`](https://github.com/humblemat810/kg-doc-parser/blob/main/kg_doc_parser/workflow_ingest/page_index.py): PageIndex materialization, `_normalize_page_index_unary_nodes`, `atomic_retained`, `parse_page_index_layer`, provider calls.
- [`parser_core.py`](https://github.com/humblemat810/kg-doc-parser/blob/main/kg_doc_parser/workflow_ingest/parser_core.py): **current code includes** `_layer_structural_issues` which identifies single-child refinements; review/commit must be traced to determine whether every path enforces it.
- [`semantics.py`](https://github.com/humblemat810/kg-doc-parser/blob/main/kg_doc_parser/workflow_ingest/semantics.py): **current code includes** `validate_semantic_tree_shape()`, `validate_semantic_tree_integrity()`, and `compute_terminal_content_coverage()` (including missing/multiply-owned terminal positions). Trace whether **every** export and success status enforces these; code may reject unary nodes yet still lose content by normalizing to an atomic leaf.
- [`layerwise_llm.py`](https://github.com/humblemat810/kg-doc-parser/blob/main/kg_doc_parser/workflow_ingest/layerwise_llm.py): `_proposal_validation_reason` rejects single-child LLM proposals; trace actual CUD correction and retry path.
- [`service.py`](https://github.com/humblemat810/kg-doc-parser/blob/main/kg_doc_parser/workflow_ingest/service.py): `run_ingest_workflow(..., resume_from_checkpoint=False)` in inspected source; `build_runtime()` does not construct a cache there.
- [`strategy.py`](https://github.com/humblemat810/kg-doc-parser/blob/main/kg_doc_parser/workflow_ingest/strategy.py): strategy-triage implementation; determine whether invoked by concrete public entry points.
- [`tests/test_workflow_ingest_page_index_pipeline.py`](https://github.com/humblemat810/kg-doc-parser/blob/main/tests/test_workflow_ingest_page_index_pipeline.py): the title/full-table fixture remains `@pytest.mark.manual` in inspected source; the universal requirement needs mandatory tests beyond this fixture.
- [`.github/workflows/ci.yml`](https://github.com/humblemat810/kg-doc-parser/blob/main/.github/workflows/ci.yml): inspected deterministic CI excludes `manual` tests. Verify actual CI test collection on pin.
- [`doc/cache_reuse_and_isolation.md`](https://github.com/humblemat810/kg-doc-parser/blob/main/doc/cache_reuse_and_isolation.md) and [`doc/provider_timeout_safety.md`](https://github.com/humblemat810/kg-doc-parser/blob/main/doc/provider_timeout_safety.md): preserve existing cache/timeout policies.
- Earlier repo/history *leads* (not independently proven by this handoff): April 8 refactor `ca90f2a`, June 4 unary-guard update `c8d3369`; investigate actual diffs and branch ancestry.

**Downstream uploaded reports (contextual evidence, version `graph-knowledge-doc-parser 0.2.15`, not current HEAD proof):**

1. `table_row_pre_segmentation_and_content_reparenting.md`: reports table-heavy low coverage and content/grandchild loss; proposes automatic row segmentation and reparenting. **Do not automatically adopt its remedies** because they may violate the user's atomic-table and semantic-containment requirements. Reproduce and repair the general cause.
2. `workflow_cache_and_checkpoint_not_enabled_by_default.md`: reports demo-only cache wiring, PageIndex cache omission, and caller-driven checkpoint resume; verify the currently installed runtime, source, and public API.

### Final guardrails

- **No deleting child/grandchild evidence as a shortcut to remove unary nodes.**
- **No compulsory table-row split as a shortcut to remove unary nodes.**
- **No silently successful atomic retention for failed excerpt/boundary CUD corrections.**
- **No skipping or downgrading required CI tests to obtain green.**
- **No assuming that an old bug persists today without pinning and reproducing.**
- **No assigning blame for test omissions without historical evidence.**
- **No 'complete' status if any supported exported tree can still contain a unary node or lose source content.**


---

## L. Executable regression payload pack (new companion files)

The requirements above are complemented by two tangible deliverables:

- **`kg_doc_parser_fixture_acceptance_checklist.md`** — copy/paste in-repository coding-agent prompt, per-case matrix, CI collection and mutation-testing requirements, and a minimal actual-model smoke-check example.
- **`kg_doc_parser_test_payloads.json`** — `tree_cases` contain `SemanticNode`-shaped `input_tree` objects, authoritative `source_map` texts, optional illustrative normalized trees, and explicit expected outcomes; `workflow_cases` contain adapter-neutral deterministic scripted mock workflows for CUD retries, fallback triage, PageIndex caching, and standalone checkpoint recovery.

Read both **before modifying tests**. JSON workflow scripts are **not** claims about current provider JSON schema or immediately runnable production APIs; the agent must adapt them to actual public entry points and current HEAD. Some inputs are intentionally structurally invalid. `end_char` is inclusive. `expected.illustrative_normalized_tree` expresses one lossless, valid tree shape but does not mandate synthetic node IDs or exact wrapper arrangement for every public parser mode.

**Mandatory fixture IDs:** `P001`–`P018` and `W001`–`W010` (with boundary/excerpt variants). A passing shape validator alone is insufficient: verify all source content against original authoritative input through terminal ownership, then demonstrate source-loss and unary-acceptance fault mutations cause CI failure.
