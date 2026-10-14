# KG Doc Parser — Executable Fixture Checklist & Coding-Agent Runbook

**Companion requirements:** `kg_doc_parser_regression_investigation_and_requirements.md`
**Machine-readable payloads:** `kg_doc_parser_test_payloads.json`
**Date:** 2026-10-14 · **Repository:** https://github.com/humblemat810/kg-doc-parser

> **Normative rule:** Every non-leaf node has **at least two children**, at **every depth**, for **all parsing paths**. Recovery differs by strategy: **PageIndex = deterministic, lossless structural morphism**; **boundary/excerpt = CUD correction/revalidation with finite retries and explicit failure**, with separately recorded strategy fallback where authorized. `Title → Table` is merely one example of the **universal `Parent → Child` invariant**.
>
> **Evidence policy:** The payloads are proposed regression fixtures, **not claims that today's code fails them**. Before changing code, pin the local `HEAD`, run an initial baseline, then establish first/last failing revisions where possible. Do not attribute a deletion/disabled test or cause to a person without Git evidence.

## A. Copy-paste prompt for the in-repository coding agent

```text
Implement and demonstrate the KG Doc Parser universal branching, grounding,
coverage, correction, and regression-prevention contract using the attached:
- kg_doc_parser_regression_investigation_and_requirements.md
- kg_doc_parser_fixture_acceptance_checklist.md
- kg_doc_parser_test_payloads.json

Step 1. Record git rev-parse HEAD, git status --short, latest applicable tag,
Python and dependency versions, and the CI test selection command.
Do not infer HEAD from a mutable GitHub /main/ link. Read repository AGENTS.md.

Step 2. Load JSON fixtures. Adapt tree_cases input_tree to current
SemanticNode / HydratedTextPointer schemas. Treat workflow_cases scripts as
adapter-neutral mock instructions, not actual provider requests. Verify all
supplied source offsets against authoritative source_map text (end_char is
inclusive), and run shape + integrity + terminal-source-ownership checks.

Step 3. For each fixture, first run the current public entry point and record:
source mode, strategy, expected/actual tree, success/failure status, grounded
terminal spans, non-whitespace coverage, missing/multiply-owned positions,
provider/mock call count, CUD retries, cache hits, checkpoint status, trace.
Produce baseline red or verified-already-fixed evidence.

Step 4. Trace historical cause with git log --follow, git blame, git diff,
unit-test history and CI marker/selection history. Identify last good/first
bad commit per execution path if provable; otherwise state unknown.
Investigate removed vs never-added vs bypassed vs weakened regression tests.

Step 5. Fix with ONE shared, authoritative structural and coverage gate.
PageIndex must morph any unary structure deterministically without dropping
content or inventing siblings. An independently grounded parent content span
becomes a real sibling leaf; purely structural wrappers are safely collapsed,
if representation permits; otherwise explicit failure. Atomic tables need not
be split into rows. All grandchildren and typed containment must be preserved.
Boundary/excerpt must instead invoke CUD review/correction/update, validate,
retry a finite number of times, then fail that attempt. A fallback requires
explicit transition events and an independently valid result.

Step 6. Install EVERY mandatory fixture in standard, offline CI, covering
heuristic/LLM/layerwise PageIndex, boundary/excerpt, fallback, public CLI and
Python APIs, tree export, and resumed runs. Do not mark tests manual/slow,
skip/xfail them, or change acceptance criteria to make CI green.

Step 7. Perform mutation proof: reintroduce (in a disposable worktree) one
unary node and one subtree-drop bug. Confirm mandatory CI fails both. Restore
the worktree; rerun CI green. Report exact commands, test names, SHAs and logs.

If a fixture cannot be wired to a public path, report an explicit gap with a
tracked follow-up; do not mark it verified. If a contract cannot be honored
without data loss, fail explicitly and keep the last valid state.
```

## B. Fixture format & source-ownership rules

- [ ] Validate JSON syntax and unique fixture IDs; use the top-level `tree_cases` and `workflow_cases` arrays.
- [ ] `input_tree` and `expected.illustrative_normalized_tree` use current `SemanticNode` field names (`child_nodes`, `total_content_pointers`, `aggregate_content_pointers`, `node_id`, `parent_id`, `level_from_root`, etc.). The *illustrative* expected tree shows one valid normalization, **not** necessarily mandatory node IDs/types for every public parser entry point.
- [ ] `source_map.<cluster>.text` is the authoritative source. `start_char` and `end_char` are **inclusive** character offsets; `verbatim_text` must exactly match `source[start:end+1]` except where a fixture explicitly tests invalid grounding.
- [ ] `expected.outcome` uses `normalized_success`, `explicit_failure`, or `normalized_or_explicit_failure`. The last allows successful **provably lossless** normalization or an explicit safe rejection—not silent success or content drop.
- [ ] Use **terminal-source ownership** rather than ancestor aggregate pointers: every non-whitespace character has **exactly one** leaf owner. A graph can have acceptable shape and still be incorrect for missing or multiply owned source data.
- [ ] `workflow_cases.script` is **test-double orchestration notation**, *not* a native provider or runner JSON schema. Convert steps to deterministic mock provider/review/triage/cache/checkpoint behaviour within the repository.
- [ ] Do not rely only on private helper tests. Execute the corresponding public entry point for every supported strategy and assert the returned, persisted and exported objects satisfy the same release gate.

## C. Mandatory PageIndex / tree-case matrix

| Fixture | Purpose | Required observable result |
|---|---|---|
| `P001` | Title → atomic table | Parent content leaf + original atomic table leaf; no table-row coercion |
| `P002` | Paragraph → atomic table | Same general morphism, independent of heading type |
| `P003` | Section → figure | Same rule across content modalities/types |
| `P004` | Contentless structural wrappers | Safe flatten/rebase preserving descendants and evidence |
| `P005` | Content parent → content child with grandchildren | **No dropped rows**, including through `atomic_retained` path |
| `P006` | Root → page → one content leaf | Safe representation-aware transform or **explicit failure**; no phantom sibling |
| `P007` | LLM-only title without source span | No fabricated heading evidence |
| `P008` | 70-deep unary chain | Rejected or safely normalized; cannot succeed unchanged or loop |
| `P009` | Repeated identical source strings | Distinct offsets accepted; no textual-dedup false positive |
| `P010` | Duplicate ownership at one offset | Coverage gate fails |
| `P011` | Missing content with duplicate surviving owners | Coverage gate fails despite branching |
| `P012` | Incorrect pointer verbatim text | Grounding gate fails |
| `P013` | Unicode + CRLF | Inclusive character offsets verified |
| `P014` | Case-varied repeated strings | Preserve distinct source positions |
| `P015` | 51-row atomic table | Full ownership with one TABLE leaf; no forced row explosion |
| `P016` | Already valid two-child branch | Unmodified, deterministic, idempotent |
| `P017` | Parent pointer overlaps child's source interval | Exclusive ownership or explicit failure; never duplicate |
| `P018` | Mismatched parent ID | Integrity validation fails |

- [ ] For **each** normalized-success tree, traverse iteratively and assert `len(child_nodes) != 1` for **every** node.
- [ ] For **each** normalized-success tree, assert exact char-span grounding, no lost/duplicate non-whitespace terminal characters, no missing descendants, no duplicate IDs, and valid `parent_id`/depth.
- [ ] Run the normalizer twice and compare **semantic + pointer canonical forms**: `N(N(tree)) == N(tree)`.
- [ ] Apply a *table stays atomic* assertion to `P001`, `P002`, and `P015`; splitting the source into rows is **not** an acceptable workaround.
- [ ] Ensure `P005` detects the case where normalizer changes `child_nodes` to `[]` and copies a child pointer while **losing grandchildren**.
- [ ] Ensure `P010` and `P011` fail even when legacy `compute_pointer_coverage()` might look complete; assert `compute_terminal_content_coverage()` and enforce it as a success gate.
- [ ] Test `P008` without recursive DFS risk and without relying only on a `max_depth` cutoff.
- [ ] Run `P001`/`P002`/`P005`/`P008`/`P015` through all three PageIndex-related public paths (heuristic, mocked LLM, one-layer fallback) or record an explicit unsupported-mode reason.

## D. Mandatory CUD / retry / strategy / reliability matrix

| Fixture | What it verifies |
|---|---|
| `W001_layer_boundary_cud_repairs` | Unary proposal → CUD → valid correction → commit |
| `W001_layer_excerpt_cud_repairs` | Same for excerpt |
| `W002_layer_boundary_budget_exhaustion` | Invalid CUD repeatedly → finite budget → FAILED strategy |
| `W002_layer_excerpt_budget_exhaustion` | Same for excerpt |
| `W003_explicit_fallback_after_exhaustion` | Failure recorded before independently validated PageIndex fallback |
| `W004_one_layer_page_index_no_recursive_bottom` | PageIndex fallback is one layer; ordinary workflow decides next frontier |
| `W005_triage_active` | LLM triage invoked through public auto entry point |
| `W006_triage_provider_failure` | Fallback is visible and not mislabeled as LLM success |
| `W007_page_index_call_cache_reuse` | Per-operation response cache hit, with revalidation |
| `W008_page_index_cache_invalidation` | Source/schema/provider/namespace changes force safe misses |
| `W009_standalone_checkpoint_resume` | Crash + new process resumes stable run without LLM-Wiki |
| `W010_resume_stale_document_reject` | Source revision mismatch prevents stale checkpoint reuse |

- [ ] Boundary and excerpt retries go through **actual CUD** (`review_cud_proposal` / `apply_cud_update` or equivalent), rather than a PageIndex unary morphism or silent `atomic_retained` fallback.
- [ ] Verify failure reason, retry budget, attempts, and method status in the output trace; strategy fallback cannot retroactively mark the failed strategy as successful.
- [ ] Mock timeout, malformed response, disabled review, cache hit, retry, and resumed checkpoint to prove none bypasses shape/grounding/coverage release gates.
- [ ] Verify normal standalone parsing **actually injects cache**, including PageIndex assignment/refine/summary; never assume a demo harness configuration applies to normal callers.
- [ ] Verify semantic resume uses a stable run ID and persistent store, and is not confused with completed-call response cache or OCR-only checkpoints.

## E. Fault origin, test history & accountability

- [ ] Pin commit and baseline before edits; keep the original baseline test results.
- [ ] Identify the first path where a unary or source-loss tree could pass **despite existing validators**; trace public entry points and all transformations.
- [ ] Inspect history of old unary tests, contradictory integration tests, title/table tests, markers, and CI collection; distinguish **removed**, **weakened**, **excluded**, **mocked**, **never added**, **bypassed**.
- [ ] Use Git evidence for any claim that a test was skipped to make CI pass; do **not** infer intent from `@pytest.mark.manual` alone.
- [ ] Confirm the expected contract existed before the change; link dated source/ADR and specific code diffs, not retrospective assumptions.
- [ ] Write an RCA table: defect, triggering path, last-good/first-bad SHAs (or unknown), bypass mechanism, historical test gap, corrected code, mandatory test, exact CI collection proof.
- [ ] Demonstrate **two independent mutations** (unary acceptance and subtree deletion) produce red mandatory CI tests, then green after restore/fix.
- [ ] Assert tests are neither `manual`, skipped, xfailed, nor excluded by the actual CI invocation; save `pytest --collect-only` and CI logs.
- [ ] Report blockers or partial implementation openly rather than marking checklist satisfied.

## F. Suggested in-repository test harness (illustrative)

```python
import json
from pathlib import Path
from kg_doc_parser.workflow_ingest.semantics import (
    SemanticNode,
    validate_semantic_tree_integrity,
    compute_terminal_content_coverage,
)

fixtures = json.loads(Path("tests/fixtures/kg_doc_parser_test_payloads.json").read_text())

for case in fixtures["tree_cases"]:
    source_map = case["source_map"]
    candidate = SemanticNode.model_validate(case["input_tree"])
    # The candidate may intentionally be invalid. Do not mistake model
    # constructibility for final integrity validation.
    # Supply candidate to the PageIndex repair/materialization boundary under
    # test; verify the resulting tree OR explicit failure against case.expected.
    # Do not merely call validate_semantic_tree_integrity() on invalid inputs
    # and claim that the normalization pipeline has been tested.

    expected_tree = case["expected"].get("illustrative_normalized_tree")
    if expected_tree is not None:
        target = SemanticNode.model_validate(expected_tree)
        validate_semantic_tree_integrity(target, parser_source_map=source_map)
        coverage = compute_terminal_content_coverage(target, source_map)
        assert coverage["valid"], (case["id"], coverage)
```

The snippet is a **fixture smoke check, not the full parser test**: route candidate input through actual repair logic, then assert real output and CI entry-point coverage. `P012` deliberately supplies a wrong pointer transcription, and `P018` deliberately supplies an incorrect parent ID.

### Completion / deliverable checklist

- [ ] A pinned-head RCA document with historical commit and test evidence.
- [ ] Test adapters plus **all machine-readable fixtures installed into mandatory offline CI**.
- [ ] Parser fixes preserving the distinct PageIndex vs CUD recovery contracts.
- [ ] No data loss, no ungrounded fabricated leaves, no unary success, no retry-limit bypass.
- [ ] Baseline/fixed test outputs, public-entry-point coverage matrix, and mutation-test proof.
- [ ] Explicitly report anything already fixed, any missing integration, and remaining unsupported public paths.
