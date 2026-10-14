# OCR Provider Smoke Report

## Scope

This report records the provider-abstraction smoke run. It does not introduce
an OCR v2 workflow: every request used the existing
`run_ocr_source_workflow` provider boundary and page-artifact pathway.

## Capability Coverage

The adapter inventory exposes all six Docling engines without importing
optional runtimes at package-import time:

- RapidOCR
- EasyOCR
- Tesseract CLI
- TesserOCR
- ocrmac
- Nemotron-OCR

It also exposes the existing OpenAI-compatible vision path as `bonsai` and
keeps `codex` available for structured parser/chat work. Codex is explicitly
not advertised as an OCR vision provider because image support was not
verified for that bridge.

## Real Bonsai Run

The local Bonsai 2 model was loaded with its multimodal projector through
`llama-server` and called through the parser's `bonsai` adapter.

- Model: `Ternary-Bonsai-2-27B-PTQ1_0.gguf`
- Vision projector: `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`
- Server: `http://127.0.0.1:8181/v1`
- Reasoning effort: `low`
- Reasoning budget: `256`
- Input: `ocr_smoke_page_1.png`
- Result: succeeded
- OCR text clusters: `4`
- Non-text objects: `0`
- Empty page: `false`
- Credentials recorded in artifacts: `false`

The saved response was compared with the exact strings and drawing positions
used by the deterministic smoke-image generator:

- Regions detected: `4/4` (`100%` region recall).
- Exact region strings: `3/4`.
- Characters returned: `159/160` (`99.375%` character coverage); the only
  mismatch was the missing space in `OCRWorkflow Smoke Page 1`.
- Spatial result: approximate pass. Returned starts were approximately
  `x=62` in the provider's 1000-wide coordinate system and `y=72, 162, 252,
  342`; the source starts map to approximately `x=57.1` and
  `y=70, 160, 250, 340`.
- Spatial limitation: this smoke does not establish IoU or word-level
  grounding because the provider returns coarse line boxes and the adapter
  does not invent finer geometry.

The detailed response and measurements are retained in
`experiments/ocr-provider-smoke/bonsai-live-loopback/ocr-response.json` and
`quality-report.json`.

The first attempt used `reasoning_effort=none`, which the Bonsai chat template
rejects. A second attempt bound the server to a non-responsive virtual
interface. The successful run used the supported `low` setting and loopback
binding; these were local server-launch issues, not parser-adapter failures.

## Reproduction

Start the existing local Bonsai launcher with a supported reasoning mode:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_local_bonsai.ps1 `
  -HostAddress 0.0.0.0 -Context 4096 `
  -ReasoningEffort low -ReasoningBudget 256
```

Then configure the parser process with `KG_DOC_OCR_BASE_URL` set to
`http://127.0.0.1:8181/v1` and run:

```powershell
python -m kg_doc_parser.workflow_ingest.ocr_provider_smoke `
  --live `
  --image experiments/ocr-provider-smoke/assets/ocr_smoke_page_1.png `
  --output-dir experiments/ocr-provider-smoke/bonsai-live-loopback
```

## Retained Evidence

Generated evidence is intentionally untracked and remains under:

`experiments/ocr-provider-smoke/`

The successful run is in `bonsai-live-loopback/` and includes:

- `live-result.json`, the sanitized outcome summary;
- `capability-report.json`, the provider inventory and live status;
- the existing smoke image assets;
- prior fake-provider and failed Bonsai attempt artifacts for comparison.

Credentials are never stored. The normalized OCR responses used for this
quality comparison are intentionally retained as experiment artifacts; they
contain no credentials or transport headers.

## Docling Engine Comparison

For this comparison, a deterministic 1600x1200 page contained a title, three
numbered list items, and a bordered three-column table. Docling 2.139.0 was
run with both RapidOCR and EasyOCR.

| Engine | List exactness | Table structure | Spatial provenance |
| --- | --- | --- | --- |
| RapidOCR | 2/3 exact; `Firstordered` lost one space | 0 structured tables; region classified as `picture` | Text starts within about 1-4 px of source; table region within about 1 px |
| EasyOCR | 2/3 exact; item 2 lost its period | 0 structured tables; region classified as `picture` | Text starts within about 1-2 px of source; table region within about 1 px |

Both engines recovered the title and all three list regions. Neither recovered
the table cells in this raster fixture. The table region itself was localized
well, but its content was not structurally extracted.

The normalized parser adapter intentionally converts Docling output to one
full-page `TextCluster` and sets `contains_table` to false because its current
contract promises page-level geometry only. Consequently, the native Docling
provenance is more spatially informative than the adapter response, and table
structure is not currently preserved by this adapter. The native Markdown and
provenance reports are retained in:

- `experiments/ocr-provider-smoke/docling-rapidocr-v2/`
- `experiments/ocr-provider-smoke/docling-easyocr/`

These results do not justify claiming table extraction correctness. A future
contract extension would be needed to preserve Docling table cells and
fine-grained provenance without changing the current OCR workflow entry point.

## Verification Boundary

The real run proves that the Bonsai image request crosses the parser adapter
and produces the existing normalized OCR response. It does not claim that all
six optional Docling runtimes are installed on this host. Their selection and
lazy-import behavior are covered by unit tests and the capability inventory;
each engine still needs its own environment-specific smoke when that optional
runtime is available.
