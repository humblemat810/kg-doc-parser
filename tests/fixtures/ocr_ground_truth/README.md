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

This is a controlled fixture, not a representative OCR benchmark. It contains
no figure or other non-text object, so it does not close the independent
completeness gate for visual regions.

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

SHA-256: `5f55174cf5cfddd227c5833c07de0c56563fd5850817f7ae518f25cbceecfc3c`
