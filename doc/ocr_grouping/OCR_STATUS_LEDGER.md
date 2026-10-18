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
| Persisted Kogwistar runtime integration | PARTIAL | Parser CI run `39074207944` completed successfully on commit `f12c823`. The installed Kogwistar `0.6.12` local acceptance run passed `test_ocr_root_resumes_after_page_63_and_matches_uninterrupted_reference` in 850.22s, comparing uninterrupted and interrupted/resumed canonical graph and grouping payloads with accepted-page reuse. The current worktree recovery compatibility matrix also passes 4/4 locally against lazy-replay commit `2c1bff3`: fresh-process lineage, real process termination, concurrent resume ownership, and grouping checkpoint recovery (`354.00s`). This remains local evidence pending CI promotion. Persistent graph-backend publication across restart remains a Kogwistar backend-specific gate, so it is not duplicated here; this parser test uses the in-memory knowledge backend and injects a provider interruption rather than an OS kill. |
| Artifact/checkpoint crash consistency | PASS (parser gate) | Parser CI run `39011917164` passes artifact-first, checkpoint-first, grouping-commit-before-child-checkpoint, and semantic-graph-before-checkpoint recovery tests. |
| Grouping and summary durability | PARTIAL | Monotonic frontier fencing, large-frontier identity stability, bounded linear frontier access, summary journaling, all parser writer call sites, document/source-revision-namespaced summary fingerprints, exact-evidence parser-provider summaries with configuration-sensitive identity, and the 100-page interrupted/resumed comparison are covered. Durable graph-backend publication remains a separate Kogwistar gate. |
| Semantic hierarchy publication | PARTIAL | `test_custom_grouping_policy_reaches_checkpoint_cache_and_export` passes locally on the public OCR-root workflow: two accepted groups are consumed by the semantic callback and exported as grounded graph nodes with ordered source excerpts and parent-child edges. The test uses the in-memory knowledge backend; fresh-process publication against a durable graph backend remains open and is Kogwistar-owned. |
| OCR completeness and positional fidelity | PARTIAL | Coordinate precision and provider loss diagnostics are enforced; `unresolved_item_count` records Docling and direct-provider items without usable provenance, `OCRQualityPolicy(reject_unresolved_items=True)` fails closed, and provider `source_item_count` must reconcile at both the quality and normalized-page boundaries. Invalid `meaningful_ordering` now reports missing, unknown, and duplicate cluster IDs; only bounded schema/quality diagnostics are persisted separately from raw provider errors and supplied to the next built-in OCR candidate prompt. Typed internal quality rejections provide allow-listed diagnostic codes; a forged provider `ValueError` cannot inject retry instructions. Six focused retry/migration tests pass locally, covering retry flow, provider-error exclusion, quality-rejection feedback, and legacy attempt-table migration (`test_failed_ocr_candidate_passes_sanitized_validation_into_next_prompt`, `test_provider_exception_is_not_reused_as_model_retry_feedback`, `test_internal_quality_rejection_provides_only_diagnostic_codes_to_retry`, `test_ocr_attempts_migrate_legacy_schema_with_retry_feedback_column`). Multimodal order is tested at normalization, grouping input, and graph text projection (`test_normalize_ocr_pages_preserves_caption_and_nontext_permutation`, `test_graph_export_preserves_mixed_multimodal_sibling_reading_order`, `test_ocr_grouping_and_semantic_graph_preserve_mixed_reading_order`); non-text graph projection and provider-backed multimodal fixtures remain open. Render-manifest identity and duplicate input-unit rejection are CI-verified by Parser run `39074207944`. The typed `OCRGroundTruth` evaluator uses deterministic maximum-cardinality one-to-one matching to independently measure item recall, spatial IoU, and text mismatch; the recorded real Codex table-page fixture covers all 16 title/list/table-cell regions with 16/16 recall, transcription, and spatial checks. An audit of all available live provider response artifacts found no provider-emitted figure/non-text objects, so provider-specific annotated completeness corpora for those objects remain open. |
| Typing and supported interpreters | PASS (source matrix) | Production-package Pyright reports 0 errors, 0 warnings. Parser CI run `39074207944` completed successfully on commit `f12c823`; the supported CPython 3.12-3.14 and PyPy 3.11 matrix passed. |
| Release | OPEN | Do not bump or publish until the release-blocking rows have evidence. |

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
