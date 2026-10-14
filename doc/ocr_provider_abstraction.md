# OCR Provider Abstraction

`run_ocr_source_workflow` already provides the durable OCR pathway: render or
materialize pages, run one page provider, persist page JSON and SQLite resume
state, normalize the result, and continue through the existing parser workflow.
This design adds provider selection without introducing OCR v2.

## Provider matrix

| Provider | Role | Optional dependency | Vision | Geometry |
| --- | --- | --- | --- | --- |
| `rapidocr` | direct local OCR engine | `rapidocr-onnxruntime` | yes | detection boxes |
| `easyocr` | direct local OCR engine | `easyocr` | yes | detection boxes |
| `docling:rapidocr` | local Docling engine | `docling` and RapidOCR runtime | yes | page-level |
| `docling:easyocr` | local Docling engine | `docling` and EasyOCR runtime | yes | page-level |
| `docling:tesseract` | local Docling engine | Docling and Tesseract CLI | yes | page-level |
| `docling:tesserocr` | local Docling engine | Docling and TesserOCR runtime | yes | page-level |
| `docling:ocrmac` | macOS Docling engine | Docling and macOS OCR | yes | page-level |
| `docling:nemotron-ocr` | local Docling engine | Docling/Nemotron runtime | yes | page-level |
| `bonsai` | OpenAI-compatible local vision model | `langchain-openai` | configured model | page-level |
| `codex` | authenticated Codex bridge with CLI image forwarding | configured bridge and vision-capable Codex model | yes | page-level |

The six Docling engines are lazy imports. Selecting one without its runtime
produces an actionable error and does not affect users of other providers.
The direct `rapidocr` and `easyocr` providers are also lazy imports and do not
import Docling. They normalize each engine's quadrilateral detections into the
same typed `OCRClusterResponse`, preserving one cluster per detected text
region and deterministic top-to-bottom/left-to-right ordering. Direct OCR does
not infer table rows, columns, merged cells, or non-text layout objects; use a
Docling layout pipeline when those semantics are required.
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

For direct local OCR without Docling:

```text
KG_DOC_OCR_PROVIDER=rapidocr
# or: KG_DOC_OCR_PROVIDER=easyocr
```

## Smoke evidence

The capability command writes a sanitized JSON report under the chosen
experiment directory. A real one-page provider smoke is reproducible with
`ocr-provider-smoke --live --image <page.png> --output-dir <dir>` after loading
the provider environment. It writes `live-result.json` without credentials or
raw responses. Keep rendered pages, `ocr-state.sqlite`, page JSON, diagnostics,
and provider caches together for a full workflow run; do not commit them.

The Codex bridge accepts only inline base64 image data URLs, materializes them
in a temporary directory, and passes them to `codex exec --image`. Remote URLs,
arbitrary file paths, and App Server image forwarding are rejected. The bridge
therefore requires a vision-capable Codex model and an available Codex CLI.

See [`ocr_provider_smoke_report.md`](ocr_provider_smoke_report.md) for the
recorded real Bonsai and local-engine vision smoke results and retained
experiment artifacts.
