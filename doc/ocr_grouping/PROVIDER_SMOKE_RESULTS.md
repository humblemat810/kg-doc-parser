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
| RapidOCR direct | Local cached ONNX models | Passed | 16 text clusters; expected title, list, and table text detected; pixel boxes returned |
| EasyOCR direct | Local model after first-run download | Passed | 13 text clusters; boxes returned; list numbering and numeric table values were missed |
| Codex bridge | Real bridge request | Passed | 16/16 expected regions; 100% expected text-character coverage; table and list regions preserved |
| Codex bridge | Real two-page PDF run | Passed | 2/2 pages; page order preserved; one terminal punctuation mark omitted on page 2 |
| Bonsai loopback | Real local loopback endpoint | Passed | 4/4 returned regions; 159/160 expected characters; 3 exact regions and one minor spacing difference |
| Bonsai external endpoint | Real configured endpoint | Failed | Request timeout; not counted as an OCR success |
| Docling + EasyOCR | Real optional Docling runtime | Passed in prior artifact | One full-page text cluster; no word/region geometry is claimed |
| Docling + RapidOCR | Prior v2 artifact | Passed in prior artifact | One full-page text cluster; no detailed region geometry is claimed |

The direct RapidOCR and EasyOCR smoke was rerun locally on this date. The
EasyOCR model download was initialization, not test data generation. The
Docling rerun was stopped after it produced no output during optional model
initialization; its prior successful artifacts remain labeled as prior runs.

## Interpretation

- Direct RapidOCR currently gives the strongest local structured-region result
  on this fixture.
- EasyOCR returns useful boxes but loses small numeric/list details on this
  fixture, so it should not be treated as table-complete.
- Codex produced the best measured fixture result, but its geometry check is
  approximate and does not include word-level IoU ground truth.
- Docling's adapter intentionally reports page-level geometry when its export
  does not expose finer reliable coordinates. A full-page box is not evidence
  of region accuracy.
- A provider marked available by the capability inventory is not thereby a
  successful deployment. Missing optional packages, model initialization
  failures, endpoint timeouts, and platform restrictions remain explicit
  failure states.

The generated JSON outputs remain local experiment artifacts and are not part
of the parser package or its required test fixtures.
