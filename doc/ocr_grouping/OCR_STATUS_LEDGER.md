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
| Persisted Kogwistar runtime integration and page scaling | PASS for fixed runtime/recovery gates; current typing work in progress | Previously verified at parser CI `39188130268` and `39135756441`; exact tests and operation counts are retained in Git history. No extra 1000-page graph benchmark is required. Current core typing branch is not yet merged/released; parser remains pinned to released Kogwistar `0.6.13` until core CI/release gates pass. |
| Artifact/checkpoint crash consistency | PASS for existing recovery gates; current outcome refactor locally verified | Prior parser CI `39135756441` passed artifact-first, checkpoint-first, grouping-commit-before-child-checkpoint, and semantic-graph-before-checkpoint cases. The current working-tree page-outcome slice passes `test_ocr_page_artifact_publish_crash_is_recoverable` and `test_duplicate_ocr_page_workers_accept_only_one_artifact`; it still needs the full interpreter matrix. |
| Adaptive grouping and summary durability | IMPLEMENTED; live-provider quality PARTIAL | Iterative grouping, grounded summaries, resume, and tree/frontier CAS have deterministic coverage. Live-provider grouping quality remains unverified. |
| Semantic hierarchy publication | PASS in prior CI; root artifact deduplication locally verified | Prior parser CI `39188130268` passed durable graph publication. The current root path now writes the accepted grouped tree and summary once; `test_default_adaptive_root_wires_grounded_summary_and_semantic_export` verifies one tree and summary publication. |
| OCR completeness and positional fidelity | PARTIAL; see B08 below | Cross-page continuation now requires item-specific edge geometry; hybrid PDF mode now merges native text with raster OCR. Structural accounting and the independent blank-output screen do not prove completeness of partially omitted content. Do not claim provider-wide completeness. |
| Typing and supported interpreters | Standard Pyright baseline passes; strict unknown-type goal remains open | Parser Pyright on the current working tree: 0 errors/warnings (`--pythonversion 3.11`). A diagnostic strict scan reports 2,168 errors in parser production modules; the matching core scan reports 4,421. These scans expose existing unknown/dynamic typing, not introduced regressions. Changed-file Ruff passes. A full typing migration and CPython/PyPy matrix are still required; do not represent the broad typing goal as complete. |
| Release | Parser candidate `0.2.19`; no release action in this work | Kogwistar `0.6.13` remains the released parser dependency. Do not change the pin or publish until the core typing branch is merged/released and parser CI passes. |

## B01-B14 Audit Reconciliation

Statuses below are checked against parser branch HEAD `0200534` plus the
current uncommitted working-tree changes. Test names are reusable regression
evidence; PASS means the existing implementation and test agree, not that a
provider is universally accurate.

| ID | Status | Current implementation and regression evidence | Remaining work |
| --- | --- | --- | --- |
| B01 | PASS | `_materialize_image_payloads`; `test_materialize_image_payloads_uses_safe_atomic_page_paths`, `test_materialize_image_payloads_rejects_invalid_page_numbers_and_symlinks` | None identified in this audit. |
| B02 | PASS | `build_document_tree`; `test_build_document_tree_rejects_unrepresented_nontext_unit`, `test_build_document_tree_reports_all_unrepresented_nontext_units` | None identified in this audit. |
| B03 | PARTIAL | `build_document_tree` attaches tail/head evidence only to the last/first text unit when page-edge geometry and incomplete-text signals agree; suggestions additionally require adjacent pages, compatible text continuation, and aligned pixel left edges. Tests: `test_cross_page_continuation_suggestion_requires_ocr_boundary_signal`, `test_cross_page_continuation_requires_item_specific_edge_geometry`, `test_cross_page_continuation_does_not_join_a_new_heading`, `test_cross_page_continuation_never_merges_text_into_nontext`. | This remains a conservative heuristic, not proof that two blocks share paragraph identity. False negatives and some false positives remain possible; stronger identity requires explicit provider/user continuation evidence. |
| B04 | PARTIAL | The grounded generator receives exact owned `EvidenceSlice` values; `test_grounded_summary_generator_receives_only_owned_source_slices`. | The legacy `SummaryGenerator` callback receives whole `SourceItem` values for compatibility and cannot enforce node-owned sub-slices. Production defaults use the grounded callback; compatibility callers retain this weaker contract. |
| B05 | PASS | `_spatial_accuracy`; `test_synthetic_page_box_does_not_claim_region_accuracy`, `test_explicit_region_precision_is_preserved` | None identified in this audit. |
| B06 | PASS | Source-revision-scoped artifact directories; `test_pdf_artifact_directories_do_not_collide_for_same_basename` | None identified in this audit. |
| B07 | PASS | Safe generated names and positive/ordered unique page numbers; `test_materialize_image_payloads_rejects_invalid_page_numbers_and_symlinks`, `test_materialize_image_payloads_rejects_out_of_order_page_numbers` | None identified in this audit. |
| B08 | PARTIAL | `_evaluate_ocr_quality` independently samples the source image for visible ink when OCR returns no text or non-text records; it warns by default and can reject by policy. `test_ocr_quality_independently_flags_empty_output_on_visible_source_ink`. Existing provider-loss and item-accounting tests remain. | The image check detects total empty-output loss only. It does not prove partial text/table/figure completeness; that remains an explicit provider/quality limitation, not an accepted completeness guarantee. |
| B09 | PASS | Evidence is bounded before provider invocation; `test_summary_target_planner_does_not_send_oversized_evidence` | None identified in this audit. |
| B10 | PASS | Artifact-backed cursor checkpoints; `test_ocr_scaling_artifact_reads_and_checkpoint_state_bytes_are_linear` passed locally on CPython 3.14.6 with 100/500/1000 deterministic pages. At 1,000 pages: accepted reads `1,000` vs legacy prefix decodes `500,500`; accepted JSON bytes `538,893` vs legacy prefix bytes `269,666,388`; cumulative checkpoint-state bytes `353,893` vs legacy `190,069,781` (about 500.5x fewer reads, 500.4x fewer source bytes, and 537.0x fewer checkpoint bytes). At 100 and 500 pages the same test verifies linear compact-state growth against quadratic legacy growth. | Preserve the test and these invariants during refactor. These are deterministic accounting measurements, not wall-clock performance claims. |
| B11 | PASS | Render identity includes DPI, format, rotation, `pdf2image` version, Poppler selection, resolved `pdfinfo` executable path, and its reported version. Tests: `test_pdf_render_manifest_rejects_tampered_page_and_records_identity`, `test_builtin_pdf_render_identity_includes_poppler_selection`, `test_builtin_pdf_render_identity_includes_poppler_binary_version`. | If the executable cannot be probed, the fingerprint records that uncertainty as `unavailable`; deployment should ensure the renderer is installed and probeable when reproducible cache reuse is required. |
| B12 | PASS | Reading-order reference extraction maps mentions to exact slices; `test_reference_mentions_cross_ocr_units_preserve_exact_source_slices` | None identified in this audit. |
| B13 | PASS | Opt-in `prefer_native` always runs raster OCR, projects native line boxes into raster coordinates, deduplicates only text-and-geometry matches, and preserves unmatched OCR text and non-text/table evidence. `native_only` rejects unknown/image-bearing pages. Rotated/cropped PDF pages defer to raster OCR. Tests: `test_hybrid_pdf_merge_deduplicates_matching_text_and_keeps_ocr_regions`, `test_pdf_preparation_prefers_native_text_at_the_production_runner_boundary`, `test_native_pdf_text_reads_a_real_selectable_pdf`, `test_native_pdf_text_is_retained_for_hybrid_pages_with_embedded_images`. Default raster-first mode and cache fingerprinting are unchanged. | No remaining implementation gap identified for the audit seam. Real-provider recognition quality remains outside this deterministic contract. |
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
