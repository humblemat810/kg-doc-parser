# OCR Implementation Status Ledger

This is the only live status checklist for the OCR runtime and seam work.
Update existing rows in place. Do not append dated progress sections or create
new checklist items for individual test runs.

Historical implementation evidence is represented by Git:

```text
git log --oneline --follow -- doc/ocr_grouping/OCR_STATUS_LEDGER.md
git show <commit> -- doc/ocr_grouping/OCR_STATUS_LEDGER.md
```

## Current Status

| Area | Status | Release-blocking remainder |
| --- | --- | --- |
| Persisted Kogwistar runtime integration and page scaling | PASS for fixed runtime/recovery gates; current typing work in progress | Previously verified at parser CI `39188130268` and `39135756441`; exact tests and operation counts are retained in Git history. No extra 1000-page graph benchmark is required. Parser now targets published Kogwistar `0.6.15`; verify the updated pin in parser CI. |
| Artifact/checkpoint crash consistency | PASS for existing recovery gates; page-outcome refactor locally verified | Prior parser CI `39135756441` passed artifact-first, checkpoint-first, grouping-commit-before-child-checkpoint, and semantic-graph-before-checkpoint cases. Commit `1bdbc11` includes the page-outcome slice; `test_ocr_page_artifact_publish_crash_is_recoverable` and `test_duplicate_ocr_page_workers_accept_only_one_artifact` passed locally. The full interpreter matrix remains outstanding. |
| Adaptive grouping and summary durability | IMPLEMENTED; live-provider quality PARTIAL | Iterative grouping, grounded summaries, resume, and tree/frontier CAS have deterministic coverage. Live-provider grouping quality remains unverified. |
| Semantic hierarchy publication | PASS in prior CI; root artifact deduplication locally verified | Prior parser CI `39188130268` passed durable graph publication. The current root path now writes the accepted grouped tree and summary once; `test_default_adaptive_root_wires_grounded_summary_and_semantic_export` verifies one tree and summary publication. |
| OCR completeness and positional fidelity | PARTIAL; see B08/B13 below | Cross-page automatic merges require an explicit matching item identity. Hybrid PDF mode now OCRs reliable direct-image regions and falls back to full-page OCR when PDF visual coverage is uncertain. Structural and visual screening do not prove provider-wide semantic completeness. |
| Typing and supported interpreters | Standard Pyright baseline previously passed; current changed-file Pyright unavailable locally | Parser Pyright on the code reviewed for commit `1bdbc11`: 0 errors/warnings (`--pythonversion 3.11`). The current Windows invocation fails before analysis because Node cannot `lstat C:\\Users\\chanh` (`EPERM`). Prior strict scans reported 2,168 parser and 4,421 core unknown-type findings. Changed-file Ruff passes; current changes still require CI Pyright and the CPython/PyPy matrix. |
| Release | Next candidate `0.2.17`; not yet tagged or published | `v0.2.16` is the latest released parser tag. The `0.2.17` candidate consolidates unreleased changes; Kogwistar `0.6.15` is the dependency pin. Do not tag or publish until parser CI passes. |

## B01-B14 Audit Reconciliation

Statuses below describe the current parser feature-branch changes. Test names
are reusable regression evidence; local passes are not CI evidence, and a
quality gate does not mean that a provider is universally accurate.

| ID | Status | Current implementation and regression evidence | Remaining work |
| --- | --- | --- | --- |
| B01 | PASS | `_materialize_image_payloads`; `test_materialize_image_payloads_uses_safe_atomic_page_paths`, `test_materialize_image_payloads_rejects_invalid_page_numbers_and_symlinks` | None identified in this audit. |
| B02 | PASS | `build_document_tree`; `test_build_document_tree_rejects_unrepresented_nontext_unit`, `test_build_document_tree_reports_all_unrepresented_nontext_units` | None identified in this audit. |
| B03 | IMPLEMENTED (LOCAL TESTS PASS; CI PENDING) | OCR `TextCluster.continuation_id` is validated and carried into source annotations; automatic proposals require the same nonblank item-level ID on adjacent text items in addition to edge evidence, geometry, and text checks. Tests: `test_cross_page_continuation_suggestion_requires_ocr_boundary_signal`, `test_cross_page_continuation_requires_matching_item_identity`, `test_page_boundary_candidate_without_item_identity_is_reviewable`, `test_cross_page_continuation_does_not_join_a_new_heading`, `test_cross_page_continuation_never_merges_text_into_nontext`, and persisted handoff `test_prepare_ocr_workflow_input_checkpoints_cross_page_grouping`. | Providers that do not supply continuation IDs will not auto-merge; they retain a review diagnostic. This intentionally prefers false negatives over heuristic false joins. |
| B04 | IMPLEMENTED (LOCAL TESTS PASS; CI PENDING) | `_generate_tree_summaries` passes ordered exact slices to grounded generators. Before a legacy callback or summary-attempt journal write, it rejects partial/discontiguous text ownership with a migration message. Tests: `test_grounded_summary_generator_receives_only_owned_source_slices`, `test_legacy_summary_generator_rejects_partial_source_ownership`. | Legacy callbacks remain supported for complete source-item ownership only. Partial ownership requires `GroundedSummaryGenerator`. |
| B05 | PASS | `_spatial_accuracy`; `test_synthetic_page_box_does_not_claim_region_accuracy`, `test_explicit_region_precision_is_preserved` | None identified in this audit. |
| B06 | PASS | Source-revision-scoped artifact directories; `test_pdf_artifact_directories_do_not_collide_for_same_basename` | None identified in this audit. |
| B07 | PASS | Safe generated names and positive/ordered unique page numbers; `test_materialize_image_payloads_rejects_invalid_page_numbers_and_symlinks`, `test_materialize_image_payloads_rejects_out_of_order_page_numbers` | None identified in this audit. |
| B08 | IMPLEMENTED (LOCAL TESTS PASS; CI PENDING) | `OCRQualityPolicy` now has strict-default and permissive modes. Strict rejects visible ink with no evidence and substantial ink-bearing grid tiles outside reported OCR boxes; errors enter the existing candidate fallback path. Diagnostics include measured counts/ratios. Tests: `test_ocr_quality_independently_flags_empty_output_on_visible_source_ink`, `test_ocr_quality_strict_mode_flags_unexplained_image_regions`, `test_strict_empty_ocr_result_uses_next_candidate_model`, `test_ocr_quality_diagnostics_are_persisted_and_strict_policy_fails_closed`. | This is an independent visual screening heuristic, not proof that all text, table cells, or figures were recognized. Low-contrast or coarse/page-wide boxes can limit its sensitivity; provider-wide completeness is not guaranteed. |
| B09 | PASS | Evidence is bounded before provider invocation; `test_summary_target_planner_does_not_send_oversized_evidence` | None identified in this audit. |
| B10 | PASS | Artifact-backed cursor checkpoints; `test_ocr_scaling_artifact_reads_and_checkpoint_state_bytes_are_linear` passed locally on CPython 3.14.6 with 100/500/1000 deterministic pages. At 1,000 pages: accepted reads `1,000` vs legacy prefix decodes `500,500`; accepted JSON bytes `538,893` vs legacy prefix bytes `269,666,388`; cumulative checkpoint-state bytes `353,893` vs legacy `190,069,781` (about 500.5x fewer reads, 500.4x fewer source bytes, and 537.0x fewer checkpoint bytes). At 100 and 500 pages the same test verifies linear compact-state growth against quadratic legacy growth. | Preserve the test and these invariants during refactor. These are deterministic accounting measurements, not wall-clock performance claims. |
| B11 | IMPLEMENTED (LOCAL TESTS PASS; CI PENDING) | Render identity includes DPI, format, rotation, `pdf2image` version, Poppler selection, resolved `pdfinfo` path/version, source/page digests, and page dimensions. Built-in cache reuse is disabled if Poppler or `pdf2image` identity is unknown. Tests: `test_pdf_render_manifest_rejects_tampered_page_and_records_identity`, `test_builtin_pdf_render_identity_includes_poppler_selection`, `test_builtin_pdf_render_identity_includes_poppler_binary_version`, `test_builtin_pdf_render_does_not_reuse_cache_when_poppler_version_is_unknown`. | CI/interpreter verification remains pending. |
| B12 | PASS | Reading-order reference extraction maps mentions to exact slices; `test_reference_mentions_cross_ocr_units_preserve_exact_source_slices` | None identified in this audit. |
| B13 | IMPLEMENTED (LOCAL TESTS PASS; CI PENDING) | Opt-in `prefer_native` retains positioned selectable text and OCRs direct embedded-image XObjects as page-coordinate crops only when the content stream contains supported text/image operations. Text-only pages use native text; forms, vector drawing, unsupported transforms, or uncertain placement use full-page OCR. Native/OCR deduplication still requires matching normalized text and overlapping geometry. Tests: `test_hybrid_pdf_ocr_crops_a_scanned_figure_and_table_region`, `test_pdf_text_only_page_uses_native_text_without_page_ocr`, `test_pdf_image_region_detection_fails_closed_for_vector_graphics`, `test_hybrid_pdf_merge_deduplicates_matching_text_and_keeps_ocr_regions`, `test_pdf_preparation_prefers_native_text_at_the_production_runner_boundary`, `test_native_pdf_text_reads_a_real_selectable_pdf`. | Only direct image XObjects in recognized page content are cropped. Complex/vector/Form-XObject pages intentionally take the full-page fallback; real-provider recognition quality remains outside this deterministic contract. |
| B14 | PASS | Decode-verified atomic PNG publication and render-manifest validation; `test_png_publication_failure_keeps_previous_artifact_and_cleans_temp`, `test_pdf_render_manifest_rejects_tampered_page_and_records_identity` | None identified in this audit. |

## Evidence Standard

For every row promoted to `PASS`, record the implementation path, exact test
node, before/after result, runtime type, and CI evidence in the commit that
changes this ledger. A local adapter test cannot close a persisted-runtime
release gate.

## Source Documents

- [Runtime and seam requirements](SEAM_AND_RUNTIME_IMPLEMENTATION_CHECKLIST.md)
- [Review findings and acceptance criteria](REVIEW_FIX_CHECKLIST.md)
- [Broad OCR audit](kg_doc_parser_ocr_broad_audit_2026-10-15.md)

The source documents contain the fixed requirements and historical context;
their status prose is not a second live checklist. Use this file for the
current answer to "what remains?".
