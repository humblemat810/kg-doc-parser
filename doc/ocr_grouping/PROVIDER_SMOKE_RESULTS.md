# OCR Provider Smoke Results

Date: 2026-10-15

These results are from the real OCR provider boundary, not mocked response
payloads. The primary fixture was
`experiments/ocr-provider-smoke/assets/ocr_layout_page.png`, which contains a
title, three ordered list rows, and a three-column table. A two-page PDF fixture
was also used for the Codex bridge run.

## Results

| Provider | Execution | Result | Evidence |
| --- | --- | --- | --- |
| RapidOCR direct | Local cached ONNX models | Passed, with omissions | 16 clusters; current ground-truth evaluation matches 15/16 text items, misses `8`, and emits one extra cluster; no table-region detection |
| EasyOCR direct | Local model after first-run download | Passed | 13 text clusters; boxes returned; list numbering and numeric table values were missed |
| Codex bridge | Real bridge request | Text passed; visual region missed | 16/16 expected text items and 100% text-character coverage, but no non-text table-region output |
| Codex bridge | Real two-page PDF run | Passed | 2/2 pages; page order preserved; one terminal punctuation mark omitted on page 2 |
| Bonsai loopback | Real local loopback endpoint | Passed | 4/4 returned regions; 159/160 expected characters; 3 exact regions and one minor spacing difference |
| Bonsai external endpoint | Real configured endpoint | Failed | Request timeout; not counted as an OCR success |
| Docling + EasyOCR | Real optional Docling runtime | Passed in prior artifact | Five positioned items: four text items and one positioned picture covering the table region; no semantic table object was emitted |
| Docling + RapidOCR | Live adapter run, 2026-10-18 | Partial | Four text items and one positioned picture box covering the table region; no semantic table object or table-cell text was emitted |

The direct RapidOCR and EasyOCR smoke was rerun locally on this date. The
EasyOCR model download was initialization, not test data generation. The
Docling run on 2026-10-18 completed through the current adapter and independent
ground-truth evaluator. Its table box is spatially accurate, but Docling emits a
picture placeholder rather than a semantic table and omits the cell text. The
earlier provenance report remains useful as raw vendor-output evidence; it is
not used as a substitute for the current adapter run.

## Independent Ground-Truth Evaluation (2026-10-18)

All three evaluations use the checked-in 1600x1200 page
`tests/fixtures/ocr_ground_truth/codex_ocr_layout_page.png` and its independent
17-item annotation: 16 text items plus the visible table-region box. The
evaluation is reproducible with `tests/test_ocr_ground_truth.py` and
`kg_doc_parser.workflow_ingest.ocr_quality.evaluate_ocr_ground_truth`.

| Provider output | Total recall | Text recall | Non-text recall | Mean spatial IoU | Detected limitation |
| --- | ---: | ---: | ---: | ---: | --- |
| Recorded Codex bridge | 16/17 (94.12%) | 16/16 (100%) | 0/1 (0%) | 0.6152 | Text is complete, but the table region is absent. |
| Direct RapidOCR | 15/17 (88.24%) | 15/16 (93.75%) | 0/1 (0%) | 0.7147 | Misses text item `8` and the table region; one extra cluster is reported. |
| Docling + RapidOCR | 5/17 (29.41%) | 4/16 (25%) | 1/1 (100%) | 0.8244 | Finds the table-region geometry but misses 12 table-cell text items and mismatches list item `list-1`. |

The Docling result reports `source_item_count=5` and
`unresolved_item_count=0`; the independent evaluator still detects the missing
table-cell content. Provider self-accounting and valid source intervals cannot
prove recall. The evaluator therefore reports modality-specific failures
instead of allowing strong text recall to conceal missing visual regions. These
measurements describe this fixture and provider configuration only; they do
not certify universal provider completeness.

## Interpretation

- No single provider is complete across the text and visual-region modalities
  on this fixture. Direct RapidOCR has the strongest text recall of the local
  direct engines, while Docling + RapidOCR is the only measured path that
  returns the table-region box.
- EasyOCR returns useful boxes but loses small numeric/list details on this
  fixture, so it should not be treated as table-complete.
- Codex produced the best measured fixture result, but its geometry check is
  approximate and does not include word-level IoU ground truth.
- Docling's adapter reports item-level geometry when provenance is exposed and
  falls back to page-level geometry only when it is not. The live artifacts
  expose separate text and picture boxes; the picture classification of the
  rendered table is not evidence of semantic table extraction.
- A provider marked available by the capability inventory is not thereby a
  successful deployment. Missing optional packages, model initialization
  failures, endpoint timeouts, and platform restrictions remain explicit
  failure states.

The generated JSON outputs remain local experiment artifacts and are not part
of the parser package or its required test fixtures.
