# OCR Grouping Implementation Status

## Latest release-gate update (2026-10-16)

Kogwistar `0.6.10` is published on PyPI. A clean parser environment now
verifies `metadata.version("kogwistar") == "0.6.10"` and
`kogwistar.__version__ == "0.6.10"`; the previous `0.6.9` module-version
mismatch is closed. Installed-release fresh-process root resume, concurrent
resume ownership, and real process termination all pass. Remaining partial
gates are the dual-store crash orderings, Linux SIGKILL coverage, independent
OCR completeness evidence, and bounded-frontier performance measurement.

The terminal-root recovery gap is now covered by an additional local change
under verification. When a completed root run has a missing or invalid page
artifact, the parser detects the broken SQLite/artifact seam before asking
Kogwistar to resume. Because Kogwistar correctly treats a successful run as
terminal, it starts a deterministic `|recovery-N` root incarnation while
preserving the original logical lineage. Valid pages are reused and only the
invalid page is retried; the public `RunResult.run_id` remains the stable
logical root ID while the recovery suffix remains an internal persisted
incarnation identity.

The focused fresh-process matrix passes locally: the stable-lineage recovery,
concurrent resume, real process termination, and grouping process termination
tests pass. The deleted-artifact terminal-resume regression also passes: three
OCR calls become two accepted page artifacts on the recovery run, the valid
second page is reused, and semantic parsing runs once for the recovery
incarnation. Ruff and `git diff --check` pass for the changed files. This is
The clean parser environment uses published Kogwistar `0.6.10`, so the local
deterministic suite provides installed-package evidence. Linux SIGKILL,
complete dual-store reconciliation, and independent OCR completeness remain
open release gates.

Commit `c173057` adds a root-finalization preflight that revalidates the
terminal page result against the authoritative SQLite/artifact state before
publishing the root result.  The real regression
`tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_workflow_resumes_cached_pages_and_terminal_semantic_run`
passes locally and proves that deleting an accepted page artifact is rejected
instead of silently reusing stale checkpoint data.  Ruff and `git diff --check`
also pass, and the commit is pushed on `feat/ocr-provider-abstraction`.

Parser CI run `38777972861` still failed all CPython 3.12-3.14 and PyPy 3.11
jobs at the installed-Kogwistar contract step, before the parser test suite.
This is a release dependency gate, not evidence that `c173057` regressed the
parser tests.  Kogwistar `v0.6.10` is tagged at `c11efd9` but its protected
PyPI deployment remains pending; the ownership/recovery API used by the fresh
process tests is still only on the separately pushed, unmerged
`fix/workflow-run-ownership` branch.  Do not bump the parser version or pin to
that API until a compatible Kogwistar release is published and the clean
CPython/PyPy matrix passes.

This is the implementation checklist for the contract in `reference/START_HERE_AGENT_PROMPT.md`.
`PASS` means current code and a focused test prove the requirement. `PARTIAL` means
the contract exists but the real OCR workflow or all required modalities are not yet
covered. `DEFERRED` means intentionally outside this implementation slice.

Audit rule: a focused adapter test does not close a persistence/runtime seam.
The status below distinguishes implemented code from release evidence. In
particular, artifact digests, atomic replacement, and same-process resume are
not claims of complete crash consistency or fresh-process recovery.

> **Identifier note:** The C01-C21 table below is the historical OCR contract
> ledger from the grouping reference pack. It is not the same numbering as the
> reviewer release gates C01-C06 in
> `SEAM_AND_RUNTIME_IMPLEMENTATION_CHECKLIST.md`. The reviewer gate matrix is
> authoritative for release readiness; a `PASS` in this historical table means
> only that the contract-level test passed.

| ID | Status | Current evidence |
| --- | --- | --- |
| C01 | PASS (contract only) | Frozen `SourceItem`/`DocumentTree` models; `test_document_tree_requires_exact_source_coverage` |
| C02 | PASS (contract only) | Half-open interval validator; gap/overlap and duplicate coverage tests |
| C03 | PASS (contract only) | `SourceLocation` preserves page/region accuracy and declared pixels/relative/points frames; unknown frames downgrade to page accuracy; `test_unknown_ocr_coordinate_system_downgrades_to_page_accuracy` |
| C04 | PASS (contract only) | Cycle, orphan, duplicate-parent, and unary-node validation |
| C05 | PASS (contract only) | `merge_groups()` keeps unmerged frontier members |
| C06 | PASS (contract only) | Adjacent, disjoint, known-member proposal validation |
| C07 | PASS (contract-level only) | Stable merge IDs, ordered source fingerprints, SQLite replay, and grouping-policy matching; `test_ocr_state_store_round_trips_grouped_tree_and_invalidates_changed_source`. This historical contract row does not close the reviewer S06 release gate; late-writer, monotonic-generation, and crash/restart evidence remain tracked there. |
| C08 | PASS | Content-addressed `MediaRef` records, captions/non-text descriptions, and page/time locations survive normalized tree and JSON round-trips without fabricated geometry; `test_multimodal_media_and_time_ranges_round_trip_without_fabricated_geometry` |
| C09 | PARTIAL | OCR pages are normalized into one validated complete tree, apply OCR-signaled adjacent continuations, and persist the canonical tree only after complete input; prefix trees are deliberately not persisted because they could overwrite a complete resume checkpoint. Incremental frontier execution remains a follow-up. Mixed text/non-text traversal is now validated at the model and adapter boundary, but downstream grouping/export preservation remains open. |
| C10 | PASS | Bounded planner now invokes an injected summary generator, validates evidence/fingerprint provenance, persists derived summaries independently, and records failures without failing OCR; `test_prepare_ocr_workflow_input_generates_grounded_summary_with_journal` and `test_summary_failure_is_journaled_without_failing_ocr_ingestion` |
| C11 | PASS | Empty OCR observations return no tree and normalized pages carry explicit `is_blank_page` metadata; `test_normalize_ocr_pages_marks_blank_pages_without_fabricating_source_items` |
| C12 | PASS | Existing OCR models and workflow input remain unchanged; adapter is additive |
| C13 | PASS | Grouping adapter performs no semantic inference |
| C14 | PASS | `DocumentTree` remains standalone; OCR metadata and lineage enter the existing `WorkflowIngestInput`, semantic parser, `WorkflowExportBundle`, and `CanonicalGraphPersistenceClient` boundary. No persistent Kogwistar model is subclassed; a separate direct `DocumentTree` serializer is intentionally not duplicated |
| C15 | PASS | Six lazy Docling engines, direct RapidOCR/EasyOCR, platform/CLI gating, KServe v2 transport, and process-local user-provider registration; `tests/test_workflow_ingest_ocr_providers.py` passed 16 tests (one optional-runtime skip); real-provider observations are recorded in `doc/ocr_grouping/PROVIDER_SMOKE_RESULTS.md` |
| C16 | PASS | `SummaryPolicy` and `select_summary_targets()` bound eligible kinds, evidence size, information gain, estimated input tokens, and target count without invoking a provider; `test_summary_target_planner_is_bounded_and_skips_low_information_nodes` |
| C17 | PARTIAL / RELEASE EVIDENCE OPEN | Atomic tree replacement, migrated SQLite page/config/artifact checkpoints, root resume identity fencing, idempotent summary-attempt journal, and a versioned root Kogwistar workflow with page/grouping/semantic child invocations are present. Grouping and summary checkpoint writes share atomic expected-revision/expected-absence fencing, reject orphaned or missing frontier rows, reject a restored tree whose persisted frontier disagrees, and prove multi-pass interrupted/resumed grouping equivalence. OCR/tree publication uses unique temporary files with a concurrent-writer regression; `test_ocr_page_artifact_publish_crash_is_recoverable` injects a crash after real page artifact publication and verifies fresh-process adoption; `test_duplicate_ocr_page_workers_accept_only_one_artifact` proves one accepted result under duplicate page delivery. The focused recovery matrix passed locally in 356.33 seconds, and the full deterministic local suite passed with `335 passed, 119 deselected, 50 warnings` in 1645.93 seconds against the coordinated source checkout. Parser CI run `38774809344` for `16f2f00` failed all CPython 3.12-3.14 and PyPy 3.11 jobs at the installed-package contract, before the parser suite ran. Kogwistar `v0.6.10` is tagged on `main`; its PyPI workflow has passed verification/build/audit and is waiting for protected `pypi` deployment approval. This is not installed-release evidence until that deployment completes and the parser matrix passes from a clean environment. Linux SIGKILL, full root/checkpoint reconciliation, and dual-store recovery after both commit orderings remain release gates; see `REVIEW_FIX_CHECKLIST.md`. |
| C18 | PASS | `DocumentReferenceMention` keeps explicit page/section/figure/table phrases source-grounded and outside the primary coverage partition; unresolved targets remain pending; `test_explicit_reference_mentions_are_source_grounded_and_outside_partition`, `test_reference_mentions_round_trip_and_validate_targets_without_becoming_children`, and `test_reference_mentions_reject_unknown_targets_and_non_text_sources` |
| C19 | PASS | `render_preview()` reads only accepted source intervals and display separators |
| C20 | PASS | Open node kinds and heterogeneous adjacent merges are supported |
| C21 | PASS | Public vocabulary and legacy exports are preserved |

## Current Call Chain

`run_ocr_source_workflow()` -> `run_ocr_ingest_workflow()` ->
`_build_ocr_page_preparation()` -> `run_ocr_root_workflow()` -> persisted
`ocr_pages` child -> persisted `grouping` child -> persisted semantic ingest
child -> root finalization.

The root and child stages now use the existing Kogwistar runtime and stable
parent/child invocation IDs. If a successful root is already terminal when a
later artifact audit finds damage, the parser creates a deterministic recovery
incarnation and records the original logical lineage instead of bypassing
Kogwistar's terminal-run rule. SQLite remains an artifact/cache index, not the
authoritative workflow cursor. Crash reconciliation and release-grade
interruption evidence remain open; passing a runtime `run_id` alone is not
treated as proof of recovery.

The tree is written atomically to `legacy_dir/document-tree.json` and is also
available as `OCRWorkflowArtifacts.document_tree`. The semantic parser still
receives its existing `WorkflowIngestInput` and authoritative source map, so this
slice does not silently replace the established parser contract.

OCR page recovery and semantic workflow recovery are separate, deliberate
boundaries. A process interruption during a multi-page OCR pass reuses accepted
page artifacts and retries only the incomplete page(s) from `ocr-state.sqlite`.
After OCR preparation succeeds, callers may pass the returned runtime `run_id`
back to `run_ocr_source_workflow(..., resume_from_checkpoint=True)`; the existing
Kogwistar node/edge runtime then resumes the semantic stage from its persisted
checkpoint instead of replaying completed parser steps. No second OCR runtime or
parallel checkpoint protocol is introduced.

## Verified Commands

```text
python -m pytest tests/test_document_grouping_contract.py -q -p no:cacheprovider
9 passed
python -m pytest tests/test_document_adapters.py -q -p no:cacheprovider
focused adapter tests passed
python -m compileall -q kg_doc_parser/workflow_ingest/...
git diff --check
poetry run ruff check kg_doc_parser/workflow_ingest/... tests/test_document_adapters.py
poetry run pyright kg_doc_parser/workflow_ingest/...  # changed production modules: 0 errors
poetry run pytest tests/test_document_grouping_contract.py -q -p no:cacheprovider --basetemp .tmp_summary_policy_test
10 passed
poetry run pytest tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_state_store_round_trips_grouped_tree_and_invalidates_changed_source -q -p no:cacheprovider --basetemp .tmp_frontier_test
1 passed
poetry run pytest tests/test_workflow_ingest_ocr_pipeline.py::test_prepare_ocr_workflow_input_checkpoints_cross_page_grouping -q -p no:cacheprovider --basetemp .tmp_live_grouping_test
1 passed
poetry run pytest tests/test_workflow_ingest_ocr_pipeline.py::test_prepare_ocr_workflow_input_generates_grounded_summary_with_journal tests/test_workflow_ingest_ocr_pipeline.py::test_summary_failure_is_journaled_without_failing_ocr_ingestion -q -p no:cacheprovider
2 passed
poetry run pytest tests/test_workflow_ingest_ocr_providers.py -q -p no:cacheprovider --basetemp .tmp_ocr_provider_test3
16 passed, 1 skipped
```

The legacy summary below is retained for historical context; the current
release status is tracked above and is not all green. A direct export format
for `DocumentTree` independent of the established semantic parser is not part
of this contract; adding one would duplicate the existing Kogwistar projection
boundary and is intentionally out of scope.

The status above is not a release claim. C09, C17 and several seam rows remain
partial or unchecked pending runtime lineage, crash-window, incremental-
frontier, and multimodal evidence. A row may move to `PASS` only when an
evidence ledger records its implementation location, exact pytest node ID,
before/after behavior, and whether the proof used the persisted Kogwistar
runtime or only a standalone/mock path. The additional review gates C01-C06
are defined in `SEAM_AND_RUNTIME_IMPLEMENTATION_CHECKLIST.md`.

## Current evidence ledger

| Gate | Implementation | Regression test | Runtime | Status |
| --- | --- | --- | --- | --- |
| C01 | `pyproject.toml` Kogwistar `0.6.10`; `kogwistar.runtime.WorkflowRuntime` public resume API | `tests/test_pypy_ci_contract.py::test_installed_kogwistar_exposes_the_persisted_workflow_api` | Installed package contract on CPython 3.12-3.14 and PyPy 3.11 | PASS for the installed-package dependency gate: metadata and `kogwistar.__version__` both report `0.6.10`, and the persisted workflow API imports. |
| C02 | `workflow_ingest/design.py:build_ocr_workflow_design`; `workflow_ingest/ocr_runtime.py:run_ocr_root_workflow`, `_validate_root_resume_identity`, `_validate_page_resume_identity` | `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_root_resume_rejects_missing_or_changed_immutable_identity`; `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_root_resume_rejects_changed_provider_before_replaying_pages`; `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_pages_run_as_persisted_steps_and_terminal_resume_does_not_recall_provider`; existing terminal-resume tests | Persisted Kogwistar runtime with page, grouping, and semantic child invocations; published parser environment | PARTIAL: root and page identity fences plus fresh-process lineage, termination, and concurrent ownership tests pass locally. Linux SIGKILL and release-CI evidence remain open |
| C03 | `workflow_ingest/ocr_runtime.py:_reconcile_page_checkpoint`; `OCRWorkflowStateStore.reconcile_orphan_artifact`; `OCRWorkflowStateStore.should_skip_page` | `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_state_adopts_valid_artifact_left_before_completion_commit`; `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_state_does_not_reuse_completion_without_published_artifact`; `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_state_rejects_stale_provider_config_or_tampered_artifact`; `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_two_store_recovery_predicates_hold_in_a_fresh_process` | Persisted runtime plus SQLite/artifact cache | PARTIAL: direct order predicates, exact reuse-path tamper rejection, fresh-process config invalidation, and fresh-process two-store predicates are covered; inverse checkpoint-first crash, full Kogwistar checkpoint reconciliation, and dual-store acceptance are open |
| C05 | `workflow_ingest/ocr_runtime.py:run_ocr_root_workflow` stable child IDs and terminal resume | `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_workflow_resumes_cached_pages_and_terminal_semantic_run`; `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_root_resumes_after_page_63_and_matches_uninterrupted_reference`; `tests/test_workflow_ingest_ocr_pipeline.py::test_grouping_incremental_resume_matches_uninterrupted_reference`; `tests/runtime/test_workflow_invocation_and_route_next.py::test_nested_workflow_failure_short_circuits_parent_routing` | Persisted Kogwistar runtime with published `0.6.10`; local fresh-process, termination, and concurrent-resume evidence | PARTIAL: the published dependency gate is closed and the selected parser CI suite passes. Linux SIGKILL, complete dual-store crash recovery, and full 100-page installed-release evidence remain open |
| C06 | `workflow_ingest/ocr_pipeline.py:OCRQualityPolicy`, `_evaluate_ocr_quality`, `_process_ocr_page`; policy is included in the OCR configuration fingerprint and persisted page metadata | `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_quality_policy_flags_and_can_reject_provider_loss_signals`; `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_quality_diagnostics_are_persisted_and_strict_policy_fails_closed` | OCR preparation boundary with typed provider payload; warning metadata is persisted and strict rejection records a failed page without an accepted artifact | PARTIAL: diagnostics, cache identity, persistence, and configurable rejection are implemented; completeness beyond provider signals, coordinate support, and fresh-process strict-policy recovery remain open |
