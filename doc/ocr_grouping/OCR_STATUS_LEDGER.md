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
| Persisted Kogwistar runtime integration and page scaling | PASS for fixed checklist scope | Current HEAD `9151654`: exact CPython CI test selection passes locally (`435 passed, 177 deselected`, 29m23s) with repo-local `--basetemp`; includes fresh-process recovery. `test_ocr_root_fresh_process_resume_after_page_63` passed Parser CI run `39135756441` with two process terminations and durable graph comparison. `test_ocr_scaling_artifact_reads_and_checkpoint_state_bytes_are_linear` measures 100/500/1000 state/artifact operations (old prefix decodes 5,050/125,250/500,500 vs 100/500/1,000 after). A full graph-backed 1000-page benchmark is not required by A1-A6/S01-S12/C01-C06 and is not a release blocker. No duplicate parser recovery framework is needed. |
| Artifact/checkpoint crash consistency | PASS locally; prior Parser CI gate PASS | The current CPython CI selection passed artifact/page recovery cases. Parser CI run `39135756441` passed artifact-first, checkpoint-first, grouping-commit-before-child-checkpoint, and semantic-graph-before-checkpoint recovery cases. |
| Adaptive grouping and summary durability | IMPLEMENTED; live-provider quality PARTIAL | The exact CI selection passed the grouping/retry tests at current HEAD. Iterative grouping, grounded summaries, resume, and tree/frontier CAS have deterministic coverage, including `test_grouping_checkpoint_concurrent_updates_accept_one_revision`, `test_grouping_checkpoint_load_rejects_mismatched_companion_records`, and `test_grouping_checkpoint_rejects_stale_writer_after_source_revision_changes`. Live-provider grouping quality remains unverified; do not duplicate Kogwistar recovery. |
| Semantic hierarchy publication | PASS locally and prior CI | Current CPython CI selection passed; Parser CI run `39135756441` passed durable graph publication and 100-page process-recovery comparison. OCR responses in these gates are deterministic fixtures, not live accuracy evidence. |
| OCR completeness and positional fidelity | PASS (evidence discipline; completeness not claimed) | Provider-loss diagnostics, source-item reconciliation, coordinate precision checks, and strict rejection are persisted and revalidated at artifact acceptance. The independent evaluator measures modality-specific recall, transcription, and spatial overlap against image-anchored annotations; checked-in responses make those evaluations reproducible. The table and chart results expose real misses across providers. They support fixture-specific comparisons only: no provider-wide or corpus-wide OCR-completeness claim is made. The focused ground-truth and artifact-quality tests pass; quality metadata also passed Parser CI run `39160013729` across CPython 3.12-3.14 and PyPy 3.11. |
| Typing and supported interpreters | Current local Pyright PASS; interpreter matrix pending | Pyright on current parser HEAD: 0 errors, 0 warnings. Configured Ruff CI target and `git diff --check` pass. Prior Parser CI #563 (`39147218276`) passed CPython 3.12-3.14 and PyPy 3.11; rerun that matrix after the final code change. |
| Release | Parser candidate `0.2.19`; remote matrix pending | Kogwistar `0.6.13` is published and verified from PyPI; parser pin, lock, exported requirements, CI source ref, and pin-contract tests now target `0.6.13`. Local packaging/pin tests (`16 passed`), persisted-runtime recovery tests against the installed package (`8 passed`), Pyright (0 errors), Ruff, and lock verification pass. Push the parser feature branch and monitor CPython 3.12-3.14 plus PyPy 3.11 CI before merge/tag/publication. |

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
