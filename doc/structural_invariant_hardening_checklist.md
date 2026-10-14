# Structural-Invariant Hardening Checklist

This checklist is the implementation record for the parser structural-
invariant release. Check an item only after code and a reusable test provide
evidence for it.

## Core Boundaries

- [x] Shared iterative tree-shape validator
- [x] Shared iterative finalized-tree validator checks IDs, ancestry, pointers, cycles, and depth
- [x] Layer invariant detection reports unary and unchanged refinements
- [x] Review and repair revalidate the final candidate set
- [x] Commit gate rejects unary refinements and invalid atomic retention
- [x] Post-mutation final validation runs before returning the tree
- [x] Handler commit and finalization pass the configured session maximum depth
- [x] Frontier scheduling rejects unary, unchanged, and rejected refinements
- [x] Layered DTO expansion uses the shared commit gate
- [x] Layered DTO request/core expansion validates source grounding; seed/result models validate identity, shape, and depth at construction
- [x] Layered DTO partial commits requeue failed parents
- [x] Parent-local diagnostics do not poison independently valid partial commits
- [x] Child IDs shared across parent groups are rejected before commit
- [x] PageIndex assignment and materialization use the invariant
- [x] PageIndex final output is validated
- [x] PageIndex wrapper promotion rebases all descendant ancestry and levels iteratively
- [x] Legacy parser conversion and export fail closed for unary trees
- [x] Legacy and workflow exports reject duplicate IDs and parent mismatches
- [x] Both graph export APIs require and validate the authoritative source map for source-bearing trees
- [x] Legacy mixed batches retain terminal parents per parent while committing valid branching siblings
- [x] Legacy compatibility adapter reports a missing frontier parent as a failure instead of retaining it atomically
- [x] Legacy `parse_doc` forwards its configured depth limit to final validation
- [x] Atomic retention fields validate at Pydantic construction and reject truthy non-boolean legacy metadata
- [x] Lookup, commit rebuilding, coverage, printing, and export traversals support valid trees beyond Python's recursion limit
- [x] Pydantic model-validation failures become typed retry diagnostics
- [x] Pydantic rejects intrinsically unary materialized child trees; contextual unary layers remain gate-validated

## Regression Coverage

- [x] Typed proposal payloads
- [x] Mapping proposal payloads
- [x] Malformed payloads and provider fallbacks
- [x] Review dropping a sibling
- [x] Repair collapsing two children into one
- [x] Full-span unchanged child
- [x] Narrowed one-child result
- [x] Explicit valid and invalid atomic retention
- [x] Duplicate, overlap, foreign-parent, and missing-pointer cases
- [x] Nested unary candidates
- [x] Generated deep unary chain
- [x] Valid deep branching tree
- [x] Configured maximum-depth rejection
- [x] Invalid atomic source pointer
- [x] Invalid explicit atomic pointer override is rejected before commit
- [x] Atomic fallback preserves the verified parent span instead of a narrowed child span
- [x] Empty-frontier unary DTO input
- [x] Public partial-commit failed-parent requeue
- [x] PageIndex descendant-level rebase and atomic child provenance
- [x] PageIndex heading-only and one-page normalization
- [x] Legacy import/export rejection
- [x] Legacy mixed terminal/branching frontier batch
- [x] Missing legacy frontier parent fails closed
- [x] Direct modern and legacy export reject bad source excerpts and offsets
- [x] Deep valid tree passes export and coverage paths
- [x] Typed atomic metadata rejects malformed and conflicting dispositions
- [x] Legacy parse entry point rejects a mocked result beyond `max_depth`
- [x] All excerpt, boundary, and PageIndex pathways
- [x] Malformed model construction cannot bypass retry routing

## Quality Gates

- [x] Ruff configured CI selection (changed hardening modules)
- [x] Pyright without new errors (changed production modules; repository-wide test scan remains legacy-noisy)
- [x] `git diff --check`
- [x] Lock verification and package/import checks
- [x] Full structural hardening suite with cache provider disabled
- [x] CPython 3.12, 3.13, and 3.14 (GitHub Parser CI run 38461434518)
- [x] PyPy 3.11 (GitHub Parser CI run 38461434518)
- [x] Feature branch pushed
- [x] GitHub CI monitored to completion (run 38461434518 succeeded)

## Review Follow-Up Verification

- [x] Focused structural regression matrix: 161 passed, 8 skipped (Ollama unavailable)
- [x] Pyright on changed production modules: 0 errors
- [x] Configured Ruff selection on changed workflow modules and tests
- [x] `git diff --check`
- [x] Exact deterministic CI test selection on local CPython 3.14.6: 258 passed, 118 deselected
- [x] Full GitHub CI and supported-interpreter matrix rerun after review follow-up changes
- [x] Push and monitor the updated feature-branch CI

## 2026-10-14 Follow-Up Fixture Audit

The earlier checked items describe the structural-hardening release, not full
acceptance of the later P001-P018/W001-W010 fixture matrix. Do not infer that
the matrix is complete from the historical checkboxes above.

- [x] P005: reproduce the PageIndex descendant-loss bug with a failing test; preserve both nested rows and exact source ownership.
- [x] P005: exercise the same nested-row shape through the public PageIndex parser API.
- [x] P001-P003/P015: exercise analogous content-parent/atomic-child shapes through the public heuristic PageIndex API; keep the table as one leaf.
- [x] P004: verify contentless-wrapper collapse, descendant IDs, and source ownership at the normalizer boundary.
- [x] P010/P011-style corruption: reject missing and multiply owned terminal source positions at the public PageIndex return boundary.
- [x] Atomic fallback cannot use a broad parent aggregate to conceal a narrower child span and missing source content.
- [x] P012-style corruption: reject mismatched pointer text through the public PageIndex parser API instead of rehydrating it after parsing.
- [x] Audit all 18 supplied tree shapes against the local PageIndex normalizer, integrity check, and terminal-ownership calculation; the Unicode/CRLF case requires UTF-8 file input on Windows.
- [ ] Run the exact P001-P018 JSON fixtures through each specified public profile; the local normalizer-only sweep is not sufficient for full acceptance.
- [ ] Run the W001-W010 workflow cases with deterministic provider, retry, cache, and resume doubles.
- [x] Run the CI-marked PageIndex selection locally on CPython 3.13 (23 passed, 54 deselected).
- [x] Run parser production-module Pyright (0 errors), Ruff CI selection, lock verification, and `git diff --check` locally.
- [ ] Complete the exact deterministic parser CI selection. The local Windows run was stopped after slow workflow tests; it did not return a final result.
- [ ] Run the CPython 3.12-3.14 and PyPy 3.11 GitHub CI matrix for this follow-up branch. Local Docker's Linux engine was unavailable.
- [ ] Merge this parser branch before updating the LLM-Wiki parser pin or releasing either package.
