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
| C08 | PARTIAL | Valid content-addressed media references and non-text descriptions round-trip; provider-specific media acquisition remains outside the adapter |
| C09 | PARTIAL | Multi-page source locations and OCR-signaled continuation proposals are supported; streaming open-frontier persistence remains |
| C10 | PARTIAL | Versioned grounded `Summary` model, stale-fingerprint validation, and validated summary updates are present; provider-backed generation policy remains |
| C11 | PASS | Empty OCR observations return no tree and normalized pages carry explicit `is_blank_page` metadata; `test_normalize_ocr_pages_marks_blank_pages_without_fabricating_source_items` |
| C12 | PASS | Existing OCR models and workflow input remain unchanged; adapter is additive |
| C13 | PASS | Grouping adapter performs no semantic inference |
| C14 | DEFERRED | Kogwistar projection remains a separate optional adapter |
| C15 | PASS | Six lazy Docling engines, direct RapidOCR/EasyOCR, platform/CLI gating, KServe v2 transport, and process-local user-provider registration; `tests/test_workflow_ingest_ocr_providers.py` passed 16 tests (one optional-runtime skip) |
| C16 | DEFERRED | Summary scheduling policy is not enabled by default |
| C17 | PARTIAL | Atomic tree artifact replacement plus versioned SQLite `document_tree_state` checkpoint; page/merge persistence is covered, while a separate summary/frontier journal remains |
| C18 | DEFERRED | Explicit reference mentions are outside this slice |
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
poetry run pytest tests/test_workflow_ingest_ocr_providers.py -q -p no:cacheprovider --basetemp .tmp_ocr_provider_test3
16 passed, 1 skipped
```

The full C01–C21 goal is not complete while any row above is `PARTIAL` or
`DEFERRED`; this file must be updated with stronger evidence as implementation
continues.
