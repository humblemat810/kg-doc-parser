# OCR Provider Abstraction

`run_ocr_source_workflow` already provides the durable OCR pathway: render or
materialize pages, run one page provider, persist page JSON and SQLite resume
state, normalize the result, and continue through the existing parser workflow.
This design adds provider selection without introducing OCR v2.

## Provider matrix

| Provider | Role | Optional dependency | Vision | Geometry |
| --- | --- | --- | --- | --- |
| `docling:rapidocr` | local Docling engine | `docling` and RapidOCR runtime | yes | page-level |
| `docling:easyocr` | local Docling engine | `docling` and EasyOCR runtime | yes | page-level |
| `docling:tesseract` | local Docling engine | Docling and Tesseract CLI | yes | page-level |
| `docling:tesserocr` | local Docling engine | Docling and TesserOCR runtime | yes | page-level |
| `docling:ocrmac` | macOS Docling engine | Docling and macOS OCR | yes | page-level |
| `docling:nemotron-ocr` | local Docling engine | Docling/Nemotron runtime | yes | page-level |
| `bonsai` | OpenAI-compatible local vision model | `langchain-openai` | configured model | page-level |
| `codex` | bridge chat provider | configured bridge | no, currently | not applicable |

The six Docling engines are lazy imports. Selecting one without its runtime
produces an actionable error and does not affect users of other providers.
The current OCR model stores a full-page bounding box when an adapter cannot
prove more precise geometry; it does not claim word-level coordinates.

## Configuration

```text
KG_DOC_OCR_PROVIDER=bonsai
KG_DOC_OCR_MODEL=Ternary-Bonsai-2-27B-PTQ1_0
KG_DOC_OCR_BASE_URL=http://127.0.0.1:8001/v1
KG_DOC_OCR_API_KEY_ENV=LLM_WIKI_LOCAL_MODEL_API_KEY
```

For Docling:

```text
KG_DOC_OCR_PROVIDER=docling
KG_DOC_OCR_ENGINE=rapidocr
```

## Smoke evidence

The capability command writes a sanitized JSON report under the chosen
experiment directory. A real one-page provider smoke is reproducible with
`ocr-provider-smoke --live --image <page.png> --output-dir <dir>` after loading
the provider environment. It writes `live-result.json` without credentials or
raw responses. Keep rendered pages, `ocr-state.sqlite`, page JSON, diagnostics,
and provider caches together for a full workflow run; do not commit them.

This slice intentionally does not change OCR source semantics, add a second
workflow, or claim that Codex supports image input.
