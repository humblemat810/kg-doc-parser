# OCR Grouping Implementation Status

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
| C17 | PARTIAL | Atomic tree replacement, migrated SQLite page/config/artifact checkpoints, root resume identity fencing, idempotent summary-attempt journal, and a versioned root Kogwistar workflow with page/grouping/semantic child invocations are present. Grouping and summary checkpoint writes now share atomic expected-revision/expected-absence fencing, reject orphaned or missing frontier rows, reject a restored tree whose persisted frontier disagrees, and prove multi-pass interrupted/resumed grouping equivalence; `test_summary_checkpoint_conflict_is_retryable_not_a_summary_warning`, `test_grouping_resume_rejects_frontier_that_disagrees_with_tree`, and `test_grouping_incremental_resume_matches_uninterrupted_reference` cover those boundaries. OCR/tree publication uses unique temporary files with a concurrent-writer regression, `test_ocr_page_artifact_publish_crash_is_recoverable` injects a crash after real page artifact publication and verifies fresh-process adoption, `test_duplicate_ocr_page_workers_accept_only_one_artifact` proves one accepted result under duplicate page delivery, and page failure writers are fenced by status, source/configuration identity, and attempt ownership in `test_ocr_page_failure_cannot_downgrade_an_accepted_result` and `test_ocr_page_failure_cannot_replace_a_newer_pending_attempt`. The 100-page persisted-root regression passes in 742.02 seconds with core commit `8f09fd7` supplied on `PYTHONPATH`, including matching canonical graph payloads and exact retry counts. The same `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_root_resumes_after_page_63_and_matches_uninterrupted_reference` test was then run against the installed released Kogwistar `0.6.8` package and failed after 530.99 seconds, so that result is explicitly not release evidence: the released runtime rehydrates the durably failed page child as terminal instead of retrying it. The retryable-child core fix must be released and consumed before this becomes release evidence; crash-window inverse ordering, SIGKILL resume, and duplicate root-resume evidence remain open; see `REVIEW_FIX_CHECKLIST.md`. |
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
parent/child invocation IDs. SQLite remains an artifact/cache index, not the
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
| C01 | `pyproject.toml` Kogwistar `0.6.8`; `kogwistar.runtime.WorkflowRuntime` public resume API | `tests/test_pypy_ci_contract.py::test_installed_kogwistar_exposes_the_persisted_workflow_api` | Installed package contract on CPython 3.12-3.14 and PyPy 3.11; latest Parser CI run `38658488434` at `a1bc69a` | PASS: all four required matrix jobs succeeded; this proves API compatibility, not runtime recovery |
| C02 | `workflow_ingest/design.py:build_ocr_workflow_design`; `workflow_ingest/ocr_runtime.py:run_ocr_root_workflow`, `_validate_root_resume_identity`, `_validate_page_resume_identity` | `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_root_resume_rejects_missing_or_changed_immutable_identity`; `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_root_resume_rejects_changed_provider_before_replaying_pages`; `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_pages_run_as_persisted_steps_and_terminal_resume_does_not_recall_provider`; existing terminal-resume tests | Persisted Kogwistar runtime with page, grouping, and semantic child invocations; coordinated parser/core checkout | PARTIAL: root and page identity fences plus same-process child resume are covered; fresh-process lineage and no-second-semantic-generation evidence remain open |
| C03 | `workflow_ingest/ocr_runtime.py:_reconcile_page_checkpoint`; `OCRWorkflowStateStore.reconcile_orphan_artifact`; `OCRWorkflowStateStore.should_skip_page` | `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_state_adopts_valid_artifact_left_before_completion_commit`; `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_state_does_not_reuse_completion_without_published_artifact`; `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_state_rejects_stale_provider_config_or_tampered_artifact`; `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_two_store_recovery_predicates_hold_in_a_fresh_process` | Persisted runtime plus SQLite/artifact cache | PARTIAL: direct order predicates, exact reuse-path tamper rejection, fresh-process config invalidation, and fresh-process two-store predicates are covered; inverse checkpoint-first crash, full Kogwistar checkpoint reconciliation, and dual-store acceptance are open |
| C05 | `workflow_ingest/ocr_runtime.py:run_ocr_root_workflow` stable child IDs and terminal resume; installed Kogwistar `0.6.8` persisted checkpoint API | `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_workflow_resumes_cached_pages_and_terminal_semantic_run`; `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_root_resumes_after_page_63_and_matches_uninterrupted_reference`; `tests/test_workflow_ingest_ocr_pipeline.py::test_grouping_incremental_resume_matches_uninterrupted_reference`; `tests/runtime/test_workflow_invocation_and_route_next.py::test_nested_workflow_failure_short_circuits_parent_routing` | Persisted Kogwistar runtime; the 100-page test was run against the installed released package and against the coordinated source checkout | PARTIAL / BLOCKED BY CORE RELEASE: source checkout `8f09fd7` proves canonical accepted-result equivalence and exact retry counts, but the installed `0.6.8` run failed after 530.99 seconds because a durably failed child is treated as terminal. SIGKILL/new-process, duplicate-resume race, and dual-store crash recovery remain open. |
| C06 | `workflow_ingest/ocr_pipeline.py:OCRQualityPolicy`, `_evaluate_ocr_quality`, `_process_ocr_page`; policy is included in the OCR configuration fingerprint and persisted page metadata | `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_quality_policy_flags_and_can_reject_provider_loss_signals`; `tests/test_workflow_ingest_ocr_pipeline.py::test_ocr_quality_diagnostics_are_persisted_and_strict_policy_fails_closed` | OCR preparation boundary with typed provider payload; warning metadata is persisted and strict rejection records a failed page without an accepted artifact | PARTIAL: diagnostics, cache identity, persistence, and configurable rejection are implemented; completeness beyond provider signals, coordinate support, and fresh-process strict-policy recovery remain open |
