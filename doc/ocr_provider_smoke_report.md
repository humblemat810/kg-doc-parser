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
the Codex bridge OCR path. Codex image forwarding is now implemented through
validated temporary files and `codex exec --image`; a real Codex smoke remains
environment-dependent.

The provider inventory now also includes direct `rapidocr` and `easyocr`
adapters. These bypass Docling and return the existing typed OCR response with
one pixel bounding box per detected text region.

## Real Codex Bridge Smoke

On 2026-10-14, a real one-request smoke was run through the host bridge using
the signed-in Codex CLI session. This was not a mocked provider response:

- Bridge health: `200`, provider `codex`
- Structured request: succeeded
- Duration: `7.43s`
- Output: schema-valid JSON with the expected `confidence: "inferred"`
- Graph writes: none
- Maintenance jobs: none
- MCP/tool calls: none

This verifies bridge authentication, the host-to-Codex CLI transport, bounded
structured output, and response decoding. It is separate from OCR accuracy:
the request contained text only. A real Codex image-page OCR smoke remains
unchecked until a vision-capable Codex model is exercised with an image and
its text, coverage, and spatial results are measured. The implementation now
supports that path through validated inline base64 image parts and CLI
`--image` files; App Server transport currently rejects image requests
explicitly rather than silently dropping them.

## Direct RapidOCR Run

The direct RapidOCR adapter was run against the retained 1600x1200 layout
fixture without importing Docling:

- Provider: `rapidocr`
- Result: succeeded
- Text clusters: `16`
- Non-text objects: `0`
- Coordinates: preserved as pixel bounding boxes
- Artifacts: `experiments/ocr-provider-smoke/direct-rapidocr/`

This confirms the direct dispatch and normalization path, not table semantics.
RapidOCR recognized the table text as individual detections; it did not produce
rows, columns, or merged-cell records.

## Direct EasyOCR Run

The direct EasyOCR adapter was run through the parser against the same fixture
using the cached Latin recognition model (`KG_DOC_OCR_LANGUAGE=fr`). This is a
real EasyOCR run, not a mocked response:

- Provider: `easyocr`
- Result: succeeded
- Text clusters: `13`
- Non-text objects: `0`
- Coordinates: preserved as pixel bounding boxes
- Artifacts: `experiments/ocr-provider-smoke/direct-easyocr-fr/`

The title and all three list regions were localized. Nine of the twelve table
cell regions were localized; the small numeric cells `42`, `17`, and `8` were
missed. OCR text also had minor recognition errors such as missing spaces in
the title/list text and `Ready` becoming `Peady`. The table was detected as
independent text regions; no row/column structure was inferred.

The default English model (`english_g2.pth`) was not cached on this host. A
separate default-English run attempted to download it and failed with a DNS
error; that failed artifact remains in
`experiments/ocr-provider-smoke/direct-easyocr-cached/`. The explicit language
setting prevents that environmental limitation from being hidden.

## Codex Bridge OCR

The parser now forwards image content parts to the authenticated Codex bridge.
The bridge accepts only inline base64 image data URLs, writes them to a
temporary directory, and passes those files to the Codex CLI with
`--image`. It rejects remote URLs, unsupported image formats, oversized
payloads, and does not forward images through the App Server transport yet.

The real structured bridge smoke above confirms the authenticated transport.
The real page-OCR smoke below now measures the image path separately.

### Real Codex Page OCR Result

The retained `ocr_layout_page.png` fixture was sent through the parser's
`codex` OCR provider, the authenticated bridge, and the real Codex CLI session.
The sanitized response is retained under
`experiments/ocr-provider-smoke/direct-codex-bridge/`.

- Provider: `codex` through the host bridge
- Model: `gpt-5.6-luna`
- Result: succeeded
- Text clusters: `16`
- Exact expected regions recovered: `16/16`
- Expected text character coverage: `100%` for the fixture's title, three list
  items, and nine table cells
- Table signal: `contains_table=true`
- Non-text objects: `0`
- Empty page: `false`
- Spatial result: approximate pass; returned boxes align with the fixture's
  title, list rows, and three table columns/row bands on the 1600x1200 page
- Credentials recorded: `false`

This is a real one-page OCR smoke, not a mocked response. It demonstrates that
the Codex bridge can carry image input through the parser and return typed OCR
geometry and table classification. It is still a fixture smoke rather than a
general accuracy benchmark: the fixture has clean typography, and the current
measurement records approximate box agreement rather than IoU or word-level
ground-truth polygons.

### Real Codex Multi-Page Result

Both rendered pages from `ocr_smoke_document.pdf` were then sent through the
same real Codex OCR provider, one page at a time:

- Pages attempted: `2`
- Pages succeeded: `2`
- Page 1: `4/4` regions and exact fixture text
- Page 2: `4/4` regions; one terminal punctuation mark was omitted from the
  final sentence
- Page ordering: preserved (`1`, then `2`)
- Non-text objects: `0` on both pages
- Table detection: not applicable to this document (`false` on both pages)

The full parser workflow wrapper could not be used for this particular run
because local Chroma engine initialization failed first with its installed
embedding-function signature mismatch. The direct page calls use the same
`_run_live_ocr_page` provider boundary and isolate OCR behavior from that
unrelated local graph-engine dependency issue. The sanitized summary is in
`experiments/ocr-provider-smoke/direct-codex-bridge-multipage/`.

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

### Live multimodal figure comparison (2026-10-18)

The same manually selected text-plus-figure fixture was exercised through the
three available paths. The figure was a real 2421x1188 UML workflow image; the
expected annotations were one manually specified title text region and one
real figure region covering the diagram. The personal source image is not
committed as a parser fixture.

| Path | Transport result | Accepted OCR result | Independent quality result |
| --- | --- | --- | --- |
| Fake typed payload | N/A | Yes | 2/2 items, recall `1.0`, mean IoU `1.0`, exact text |
| Codex bridge (`gpt-5.6-luna`) | Bridge returned HTTP 200 | No | Rejected by `OCRClusterResponse`: `meaningful_ordering` did not cover all returned clusters |
| Bonsai 2 (`medium`, budget 2048) | Model loaded and generated | No | Rejected by `OCRClusterResponse`: source-item accounting reported `0 observed` versus `17 represented/unresolved` |

The fake result is contract coverage only and is not provider evidence. The
two live failures are important: neither malformed response was accepted or
measured as accurate. They expose provider-output conformance gaps at the
typed parser boundary, while the existing recorded Codex table-page result
above remains the valid accepted real multimodal smoke.

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

## Coverage Audit

This branch does not claim a real multi-page, multimodal smoke for every
provider. Real evidence currently covers direct RapidOCR, direct EasyOCR,
Docling RapidOCR/EasyOCR, Bonsai, and Codex. The Codex path now has both a
single-page table fixture and a two-page document fixture. The other real
provider runs are single-page. Cloud providers, Ollama, and the remaining
Docling engines have capability and mocked-contract coverage only.

The current OCR contract accepts rendered image pages (including pages
rendered from PDF). It does not accept audio or video inputs, so audio/video
coverage would require a separate multimodal evidence contract rather than a
larger OCR smoke matrix. No audio/video result is being claimed here.

The legacy `kg_doc_parser/ocr.py` implementation is pre-existing and was not
duplicated by this branch. The new direct adapters intentionally remain
separate from Docling adapters, while the Codex parser alias reuses
Kogwistar's shared `StructuredBridgeChatModel` instead of creating a second
bridge client.

## Real Two-Page Provider Matrix

The two generated pages from `ocr_smoke_document.pdf` were also run through
the locally installed providers in the parser environment:

| Provider | Page 1 | Page 2 | Result | Interpretation |
| --- | ---: | ---: | --- | --- |
| Direct RapidOCR | 4 clusters | 4 clusters | succeeded | Direct pixel-box path; no table in this fixture |
| Direct EasyOCR (`fr`) | 7 clusters | 11 clusters | succeeded | Real recognition, but detector fragmentation means counts are not text accuracy |
| Docling RapidOCR | 1 page cluster | 1 page cluster | succeeded | Adapter intentionally collapses native Docling layout to one page cluster |
| Docling EasyOCR | 1 page cluster | 1 page cluster | succeeded | Same page-level adapter contract |
| Bonsai | 4 clusters | 4 clusters | succeeded | Real local multimodal model path |
| Codex bridge | 4 clusters | 4 clusters | succeeded | Page 1 exact; page 2 omitted one terminal punctuation mark |

This is real provider execution, not mocked payload coverage. The matrix is a
two-page execution smoke, not a claim that all providers have equal semantic
or table accuracy. Direct EasyOCR and the two Docling cases require richer
text/layout scoring before comparing quality fairly. Cloud providers,
Ollama, and optional Docling engines not installed in this environment were
not silently substituted with fake results.
