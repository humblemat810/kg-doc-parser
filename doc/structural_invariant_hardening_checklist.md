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
- [x] CPython 3.12, 3.13, and 3.14 (GitHub Parser CI run 38421568523)
- [x] PyPy 3.11 (GitHub Parser CI run 38421568523)
- [x] Feature branch pushed
- [x] GitHub CI monitored to completion (run 38421568523 succeeded)

## Review Follow-Up Verification

- [x] Focused structural regression matrix: 161 passed, 8 skipped (Ollama unavailable)
- [x] Pyright on changed production modules: 0 errors
- [x] Configured Ruff selection on changed workflow modules and tests
- [x] `git diff --check`
- [x] Exact deterministic CI test selection on local CPython 3.14.6: 258 passed, 118 deselected
- [ ] Full GitHub CI and supported-interpreter matrix rerun after review follow-up changes
- [ ] Push and monitor the updated feature-branch CI
