# OCR Provider Abstraction Checklist

This is the implementation ledger for the OCR provider slice. An item is
checked only when code, focused tests, or retained smoke evidence supports it.

## Provider Contract

- [x] Keep one typed `OCRClusterResponse` contract for all providers.
- [x] Keep optional provider imports lazy.
- [x] Preserve the durable `run_ocr_source_workflow` pathway and resume state.
- [x] Keep Docling engines selectable as `docling:<engine>`.
- [x] Add direct `rapidocr` provider selection without Docling.
- [x] Add direct `easyocr` provider selection without Docling.
- [x] Forward OCR images through the authenticated Codex bridge using temporary validated data-image files.
- [x] Normalize direct quadrilateral detections to pixel bounding boxes.
- [x] Produce deterministic reading order and cluster numbering.
- [x] Reject unsupported provider and Docling-engine names before runtime import.

## Coverage

- [x] Capability inventory covers both direct engines and all six Docling engines.
- [x] Provider normalization tests cover direct and Docling names.
- [x] Direct RapidOCR adapter test uses a mocked vendor module and verifies geometry.
- [x] Direct EasyOCR adapter test uses a mocked vendor module and verifies geometry.
- [x] Missing optional runtime errors are actionable.
- [x] Direct RapidOCR real fixture smoke is retained under `experiments/`.
- [x] Direct EasyOCR real fixture smoke is retained with the cached Latin model (`fr`).
- [ ] Run the default English EasyOCR model after `english_g2.pth` is cached.
- [x] Docling RapidOCR and EasyOCR comparison artifacts are retained.
- [x] Bonsai real vision smoke is retained.
- [x] Real Codex bridge structured-output smoke is retained (transport only).
- [x] Real Codex OCR page smoke is retained through the bridge with text and spatial results.

## Explicit Limitations

- [x] Direct OCR does not claim table structure or merged-cell semantics.
- [x] Docling page-level adapter does not claim word-level geometry.
- [ ] Add a separate table/layout contract only if a later slice requires it.
- [ ] Run CPython 3.12-3.14 and PyPy 3.11 CI for the completed branch.

## Verification Commands

```text
python -m pytest tests/test_workflow_ingest_ocr_providers.py -q -p no:cacheprovider
ruff check kg_doc_parser/workflow_ingest tests/test_workflow_ingest_ocr_providers.py
git diff --check
```
