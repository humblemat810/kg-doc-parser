# OCR Review Finding Index

This file is an index of recurring external-review concerns, not a checklist
or progress log. Current status is recorded only in
[`OCR_STATUS_LEDGER.md`](OCR_STATUS_LEDGER.md). The finite acceptance criteria
are in [`SEAM_AND_RUNTIME_IMPLEMENTATION_CHECKLIST.md`](SEAM_AND_RUNTIME_IMPLEMENTATION_CHECKLIST.md).

Do not append review dates, CI transcripts, duplicate status tables, or new
acceptance items here. For a new finding, map it to an existing A/S/C
requirement, reproduce it with a minimal regression test, and update that
requirement's implementation/evidence in code and the canonical ledger. Git
history retains previous review interpretations and implementation progress.

## Finding-to-Scope Map

| Recurring review concern | Scope | Best code/test entry point |
| --- | --- | --- |
| Completed OCR or grouping state overwritten during resume | A3-A4, S02, S06, S09 | `ocr_runtime.py`; `ocr_pipeline.py`; `test_grouping_checkpoint_reopens_after_process_termination`; `test_grouping_checkpoint_rejects_stale_writer_after_source_revision_changes` |
| OCR/model/config changes reuse stale artifacts | A2-A3, S02, S10 | `ocr_pipeline.py`; `test_ocr_root_resume_rejects_changed_provider_before_replaying_pages`; `test_ocr_state_rejects_stale_provider_config_or_tampered_artifact` |
| Artifact and SQLite/runtime state diverge after a crash | A3, A6, S02, S06, S09, C03 | `ocr_pipeline.py`; `test_ocr_page_artifact_publish_crash_is_recoverable`; `test_prepare_ocr_workflow_retries_checkpoint_without_artifact`; grouping/semantic commit-before-checkpoint recovery tests |
| Prefix regrouping causes quadratic work or loses accepted frontier | A4, S06, C04 | `document_grouping.py`; `test_ocr_scaling_artifact_reads_and_checkpoint_state_bytes_are_linear`; `test_merge_groups_scans_frontier_linearly_without_nested_rebuilds`; `test_grouping_incremental_resume_matches_uninterrupted_reference` |
| Final group identity or hierarchy is dropped before semantic parsing/export | A4, S05, S08 | `document_adapters.py`; `test_final_group_keeps_identity_and_kind_under_document_root`; `test_graph_export_consumes_hierarchy_for_identical_source_evidence` |
| OCR reading order omits, duplicates, or moves multimodal items | S03-S05 | OCR response models/adapters; `test_ocr_cluster_response_rejects_inconsistent_empty_or_ordering_state`; `test_ocr_grouping_and_semantic_graph_preserve_mixed_reading_order` |
| OCR appears complete despite missing tables, figures, or coordinates | S03-S04, C06 | OCR provider adapters and quality policy; `test_ocr_quality_policy_can_reject_unaccounted_source_items`; `test_build_document_tree_reports_all_unrepresented_nontext_units`; OCR ground-truth provider fixtures |
| Summary is ungrounded, stale, or regenerated after interruption | A5, S07 | `ocr_pipeline.py`; `test_summary_cache_reuses_exact_generation_and_slice_identity`; `test_summary_journal_commit_is_recoverable_before_runtime_checkpoint`; `test_failed_summary_is_retryable_without_replaying_ocr` |
| Resume silently changes source/parse identity or collides across files | A2, S10 | `ocr_runtime.py`; source-plan identity tests; `test_ocr_root_resume_rebuilds_after_same_path_source_change`; PDF same-basename artifact isolation tests |
| Recovery claims rely only on mocks or same-process tests | A6, S09, C01-C05 | `_ocr_root_fresh_process_worker.py`; `test_ocr_root_fresh_process_resume_after_page_63`; `test_ocr_root_concurrent_resume_has_one_owner_and_one_result`; durable graph reopen tests |
| Public options are ignored or contradictory options silently win | S12 | Public OCR runner and CLI tests; `test_custom_grouping_policy_reaches_checkpoint_cache_and_export`; parse-strategy conflict contract tests |

## Historical Lookup

The current source and tests are the implementation truth. To inspect a prior
review response or the change that addressed it, use the parser repository's
history rather than copying the old status narrative back into this index:

```text
git log --oneline --follow -- doc/ocr_grouping/REVIEW_FIX_CHECKLIST.md
git log --oneline -- kg_doc_parser/workflow_ingest/ocr_pipeline.py
git show <commit> -- <path>
```

Useful implementation commits include `51a5ba6` (quadratic OCR progress
snapshot removal), `43b3553` (fresh-process configuration identity),
`d50fe3f` (quality enforcement at artifact acceptance seams), and
`4512001` (accepted summary journal authority). The acceptance-scope document
contains additional source/test locators; this index intentionally avoids
duplicating those requirements or maintaining a second completion percentage.
