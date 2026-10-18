# OCR Ground-Truth Fixture

`codex_ocr_layout_page.png` is the 1600 x 1200 controlled layout page used by
the real Codex bridge OCR smoke. It contains a title, three ordered-list rows,
and a three-column, three-row table. The image is checked in so the response
and the annotations can be reviewed against the same source in a clean clone.

`codex_ocr_layout_page_ground_truth.json` records the expected text regions and
their source-image boxes. The annotations are static and are not derived from
the OCR response. `codex_ocr_layout_page_response.json` is the sanitized OCR
response captured from the real Codex bridge run; the test does not call a live
provider. CI verifies the source image digest and dimensions, then checks
one-to-one item recall, spatial IoU, transcription, and extra detections.

This is a controlled fixture, not a representative OCR benchmark. Its table
region is annotated as one non-text item, but it has no separate figure or
illustration; it does not close the broader visual-completeness gate.

The real Docling+EasyOCR and Docling+RapidOCR runs on this exact image are
preserved as `docling_easyocr_layout_page_response.json` and
`docling_rapidocr_layout_page_response.json`. The normalized responses were
captured on 2026-10-14; their raw provider outputs remain in the local smoke
artifacts. Both returned one full-page text box and no non-text objects. Against
the independent 17-item annotations, neither has a spatially matched item
(0/17 recall at IoU 0.5); the page-sized box is deliberately not accepted as
region-level grounding. EasyOCR's text contains "First ordered item" while
RapidOCR's contains "Firstordered item". These captured-response tests are
reproducibility checks, not fresh live provider calls.

`docling_rapidocr_provenance_response.json` is a separate fresh run of the
current `run_docling_ocr_page` adapter on this fixture using local Docling
2.139.0 and RapidOCR ONNX models. It preserves four text regions and the
table/picture region with source coordinates. Against the independent
annotations it matches 5/17 items: 4/16 text regions (including one
transcription mismatch) and 1/1 non-text region. This is a captured provider
result, not a CI provider call. It demonstrates successful spatial provenance
for the table/picture region while the selected OCR engine still misses most
table-cell text; it does not close the figure or corpus-wide completeness gate.

SHA-256: `5f55174cf5cfddd227c5833c07de0c56563fd5850817f7ae518f25cbceecfc3c`
