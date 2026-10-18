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

SHA-256: `5f55174cf5cfddd227c5833c07de0c56563fd5850817f7ae518f25cbceecfc3c`
