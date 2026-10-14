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

Raw credentials and raw model responses are not stored.

## Verification Boundary

The real run proves that the Bonsai image request crosses the parser adapter
and produces the existing normalized OCR response. It does not claim that all
six optional Docling runtimes are installed on this host. Their selection and
lazy-import behavior are covered by unit tests and the capability inventory;
each engine still needs its own environment-specific smoke when that optional
runtime is available.
