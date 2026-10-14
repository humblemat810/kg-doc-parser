# OCR Grouping Implementation Status

This is the implementation checklist for the contract in `reference/START_HERE_AGENT_PROMPT.md`.
`PASS` means current code and a focused test prove the requirement. `PARTIAL` means
the contract exists but the real OCR workflow or all required modalities are not yet
covered. `DEFERRED` means intentionally outside this implementation slice.

| ID | Status | Current evidence |
| --- | --- | --- |
| C01 | PASS | Frozen `SourceItem`/`DocumentTree` models; `test_document_tree_requires_exact_source_coverage` |
| C02 | PASS | Half-open interval validator; gap/overlap and duplicate coverage tests |
| C03 | PASS | `SourceLocation` preserves page/region accuracy and declared pixels/relative/points frames; unknown frames downgrade to page accuracy; `test_unknown_ocr_coordinate_system_downgrades_to_page_accuracy` |
| C04 | PASS | Cycle, orphan, duplicate-parent, and unary-node validation |
| C05 | PASS | `merge_groups()` keeps unmerged frontier members |
| C06 | PASS | Adjacent, disjoint, known-member proposal validation |
| C07 | PASS | Stable merge IDs, ordered source fingerprints, SQLite replay, and grouping-policy matching; `test_ocr_state_store_round_trips_grouped_tree_and_invalidates_changed_source` |
| C08 | PASS | Content-addressed `MediaRef` records, captions/non-text descriptions, and page/time locations survive normalized tree and JSON round-trips without fabricated geometry; `test_multimodal_media_and_time_ranges_round_trip_without_fabricated_geometry` |
| C09 | PASS | OCR pages rebuild the validated tree after every accepted/reused page, apply OCR-signaled adjacent continuations, and persist the open frontier/tree in SQLite; `test_prepare_ocr_workflow_input_checkpoints_cross_page_grouping` |
| C10 | PASS | Bounded planner now invokes an injected summary generator, validates evidence/fingerprint provenance, persists derived summaries independently, and records failures without failing OCR; `test_prepare_ocr_workflow_input_generates_grounded_summary_with_journal` and `test_summary_failure_is_journaled_without_failing_ocr_ingestion` |
| C11 | PASS | Empty OCR observations return no tree and normalized pages carry explicit `is_blank_page` metadata; `test_normalize_ocr_pages_marks_blank_pages_without_fabricating_source_items` |
| C12 | PASS | Existing OCR models and workflow input remain unchanged; adapter is additive |
| C13 | PASS | Grouping adapter performs no semantic inference |
| C14 | PASS | `DocumentTree` remains standalone; OCR metadata and lineage enter the existing `WorkflowIngestInput`, semantic parser, `WorkflowExportBundle`, and `CanonicalGraphPersistenceClient` boundary. No persistent Kogwistar model is subclassed; a separate direct `DocumentTree` serializer is intentionally not duplicated |
| C15 | PASS | Six lazy Docling engines, direct RapidOCR/EasyOCR, platform/CLI gating, KServe v2 transport, and process-local user-provider registration; `tests/test_workflow_ingest_ocr_providers.py` passed 16 tests (one optional-runtime skip); real-provider observations are recorded in `doc/ocr_grouping/PROVIDER_SMOKE_RESULTS.md` |
| C16 | PASS | `SummaryPolicy` and `select_summary_targets()` bound eligible kinds, evidence size, information gain, estimated input tokens, and target count without invoking a provider; `test_summary_target_planner_is_bounded_and_skips_low_information_nodes` |
| C17 | PASS | Atomic tree replacement, versioned SQLite page/tree/frontier checkpoints after each page/pass, idempotent summary-attempt journal, and forwarding of the existing Kogwistar `run_id`/`resume_from_checkpoint` contract through `run_ocr_source_workflow()`; `test_prepare_ocr_workflow_input_checkpoints_cross_page_grouping`, `test_prepare_ocr_workflow_input_resumes_after_mid_document_interruption`, and runtime resume tests |
| C18 | PASS | `DocumentReferenceMention` keeps explicit page/section/figure/table phrases source-grounded and outside the primary coverage partition; unresolved targets remain pending; `test_explicit_reference_mentions_are_source_grounded_and_outside_partition`, `test_reference_mentions_round_trip_and_validate_targets_without_becoming_children`, and `test_reference_mentions_reject_unknown_targets_and_non_text_sources` |
| C19 | PASS | `render_preview()` reads only accepted source intervals and display separators |
| C20 | PASS | Open node kinds and heterogeneous adjacent merges are supported |
| C21 | PASS | Public vocabulary and legacy exports are preserved |

## Current Call Chain

`run_ocr_source_workflow()` -> `run_ocr_ingest_workflow()` ->
`parse_ocr_document()` -> `prepare_ocr_workflow_input()` ->
`normalize_ocr_pages()` -> `build_document_tree()` -> optional
`apply_grouping_proposals()` -> existing `run_ingest_workflow()` semantic path.

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

All required C01–C21 rows are currently marked `PASS`. A direct export format
for `DocumentTree` independent of the established semantic parser is not part
of this contract; adding one would duplicate the existing Kogwistar projection
boundary and is intentionally out of scope.
