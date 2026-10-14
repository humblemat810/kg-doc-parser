# OCR Grouping Implementation Status

This is the implementation checklist for the contract in `reference/START_HERE_AGENT_PROMPT.md`.
`PASS` means current code and a focused test prove the requirement. `PARTIAL` means
the contract exists but the real OCR workflow or all required modalities are not yet
covered. `DEFERRED` means intentionally outside this implementation slice.

| ID | Status | Current evidence |
| --- | --- | --- |
| C01 | PASS | Frozen `SourceItem`/`DocumentTree` models; `test_document_tree_requires_exact_source_coverage` |
| C02 | PASS | Half-open interval validator; gap/overlap and duplicate coverage tests |
| C03 | PARTIAL | `SourceLocation` preserves page-level versus region-level accuracy; media geometry adapter still pending |
| C04 | PASS | Cycle, orphan, duplicate-parent, and unary-node validation |
| C05 | PASS | `merge_groups()` keeps unmerged frontier members |
| C06 | PASS | Adjacent, disjoint, known-member proposal validation |
| C07 | PARTIAL | Stable merge IDs and source fingerprints; grouped trees now replay from the SQLite checkpoint, but full policy/model invalidation coverage remains |
| C08 | PARTIAL | Media contract exists; current OCR adapter preserves non-text descriptions and needs content-addressed media input |
| C09 | PARTIAL | Multi-page source locations are supported; automatic continuation grouping is pending |
| C10 | PARTIAL | Versioned grounded `Summary` model exists; generation/invalidation workflow is pending |
| C11 | PASS | Empty OCR observations return no tree and normalized pages carry explicit `is_blank_page` metadata; `test_normalize_ocr_pages_marks_blank_pages_without_fabricating_source_items` |
| C12 | PASS | Existing OCR models and workflow input remain unchanged; adapter is additive |
| C13 | PASS | Grouping adapter performs no semantic inference |
| C14 | DEFERRED | Kogwistar projection remains a separate optional adapter |
| C15 | PARTIAL | Existing direct, Docling, and chat provider registry remains; all provider-specific grouping coverage is pending |
| C16 | DEFERRED | Summary scheduling policy is not enabled by default |
| C17 | PARTIAL | Atomic tree artifact replacement plus versioned SQLite `document_tree_state` checkpoint; page/merge persistence is covered, while summary/frontier journal coverage remains |
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
5 passed
python -m pytest tests/test_document_adapters.py::test_apply_grouping_proposals_keeps_source_coverage_and_rejects_bad_members -q -p no:cacheprovider
1 passed
python -m compileall -q kg_doc_parser/workflow_ingest/...
git diff --check
```

The full C01–C21 goal is not complete while any row above is `PARTIAL` or
`DEFERRED`; this file must be updated with stronger evidence as implementation
continues.
