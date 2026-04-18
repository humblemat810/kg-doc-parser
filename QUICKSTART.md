# Quickstart

This repo now exposes the workflow-ingest pipelines as reusable Python APIs,
thin CLI commands, and composable subworkflows.

## 1. Use The Workspace Venv

Make sure you are using:

```powershell
.venv\Scripts\python.exe
```

If you installed the project with Poetry, you can also use:

```powershell
poetry run workflow-ingest --help
```

If you want the local `./kogwistar` subtree to be used for development instead
of the GitHub dependency, run:

```powershell
bash ./scripts/bootstrap-dev.sh
```

If the helper is not run, the install will keep using the GitHub dependency
declared in `pyproject.toml`.

## 2. Learn The Command Surface

The CLI family is:

```powershell
workflow-ingest ocr --help
workflow-ingest page-index --help
workflow-ingest layerwise --help
workflow-ingest demo --help
workflow-ingest ocr-smoke-assets --help
```

CLI cheatsheet:

| Command | Use when you want | Main tests |
|---|---|---|
| `workflow-ingest ocr <source> --output-dir <dir>` | OCR a file or folder and write inspectable artifacts | `tests/test_workflow_ingest_parsing_api.py`, `tests/test_workflow_ingest_cli.py` |
| `workflow-ingest page-index <source> --output-dir <dir>` | Parse page-index text into structured output | `tests/test_workflow_ingest_parsing_api.py`, `tests/test_workflow_ingest_cli.py` |
| `workflow-ingest layerwise <source> --output-dir <dir>` | Run recursive layerwise parsing | `tests/test_workflow_ingest_layerwise_parser.py`, `tests/test_workflow_ingest_cli.py` |
| `workflow-ingest demo --output-dir <dir>` | Run the end-to-end demo harness | `tests/test_workflow_ingest_demo_harness.py` |
| `workflow-ingest ocr-smoke-assets --output-dir <dir>` | Generate local smoke assets for OCR testing | `tests/test_workflow_ingest_cli.py` |

`workflow-ingest demo --help` only prints the demo harness options. It does not
create artifacts.

To run the demo harness and produce files you can inspect, use:

```powershell
workflow-ingest demo --output-dir logs\workflow_ingest_demo
```

Then open:

```powershell
explorer logs\workflow_ingest_demo
```

That directory contains the run summary, probe trail, cache directory, and the
local engine/server data for the demo.

Common commands:

```powershell
workflow-ingest ocr-smoke-assets --output-dir tests\.tmp_workflow_ingest_ocr\generated_smoke_assets
workflow-ingest ocr tests\.tmp_workflow_ingest_ocr\generated_smoke_assets\ocr_smoke_document.pdf --output-dir logs\ocr_run
workflow-ingest ocr tests\.tmp_workflow_ingest_ocr\generated_smoke_assets --output-dir logs\ocr_batch
workflow-ingest page-index tests\fixtures\page_index\sample_page_index.txt --output-dir logs\page_index
workflow-ingest layerwise tests\.tmp_workflow_ingest_ocr\manual_cases\ollama\glm-ocr_latest\image\artifacts\legacy_split_pages\ocr-manual-ollama-image --output-dir logs\layerwise
```

## 3. Inspect The Outputs

Each workflow keeps inspectable artifacts on disk:

- `workflow-events.jsonl`
  - human-readable orchestration step trail
- `ocr-state.sqlite`
  - authoritative OCR/render resume state
- `ocr-progress.json`
  - readable mirror of the OCR state store
- `ocr-summary.json`
  - final OCR run summary
- `legacy_split_pages/<document>/page_N.json`
  - legacy-compatible OCR page artifacts
- `rendered_pages/<document>/page_N.png`
  - rasterized pages for OCR reruns and inspection
- `page-index-summary.json`
  - page-index parser summary
- `layerwise-summary.json`
  - legacy recursive parser summary
- `layerwise-graph.json`
  - legacy recursive graph payload

## 4. Run The Fast Tests

```powershell
.venv\Scripts\python.exe -m pytest tests/test_workflow_ingest_contracts.py tests/test_workflow_ingest_resolver_invariants.py tests/test_workflow_ingest_layerwise_parser.py tests/test_workflow_ingest_cli.py -q
```

This covers:

- workflow contracts
- resolver retries
- strategy switching
- reusable CLI wiring
- reusable runner wiring

## 5. Run The Workflow Pipelines Directly From Python

The main reusable APIs live in `src.workflow_ingest`:

- `run_ocr_source_workflow(...)`
- `run_ocr_batch_workflow(...)`
- `parse_page_index_document(...)`
- `run_page_index_source_workflow(...)`
- `run_layerwise_source_workflow(...)`
- `run_demo_harness_workflow(...)`

Those are the same helpers the CLI calls under the hood.

## 6. Recommended Test Split

Fast local loop:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_workflow_ingest_contracts.py tests/test_workflow_ingest_resolver_invariants.py tests/test_workflow_ingest_layerwise_parser.py tests/test_workflow_ingest_conversation_graph.py tests/test_workflow_ingest_provider_adapters.py tests/test_workflow_ingest_cli.py -q
```

Normal workflow/demo CI:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_workflow_ingest_contracts.py tests/test_workflow_ingest_resolver_invariants.py tests/test_workflow_ingest_layerwise_parser.py tests/test_workflow_ingest_conversation_graph.py tests/test_workflow_ingest_provider_adapters.py tests/test_workflow_ingest_demo_harness.py tests/test_workflow_ingest_cli.py -q
```

Full backend / server CI:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_workflow_ingest_backends_ci_full.py tests/test_workflow_ingest_server_e2e.py tests/test_workflow_ingest_demo_harness.py -m ci_full -q
```

Note: `tests/test_semantic_layerwise_doc_parsing.py::test_semantic_document_splitting[ollama]`
is a server-backed end-to-end smoke test. It expects a Kogwistar server to be
running already at `http://127.0.0.1:28110` and will skip if the server cannot
bind or does not become healthy.

## 7. Read Next

- [README.md](README.md)
- [workflow_ingest_resolver_orchestration.md](workflow_ingest_resolver_orchestration.md)
- [workflow_ingest_layerwise_proposal.md](workflow_ingest_layerwise_proposal.md)

## Mental Model

- The API is the primary contract.
- The CLI is a thin adapter over the API.
- OCR prep resumes from `ocr-state.sqlite`.
- Page-index parsing has heuristic and Ollama-backed modes.
- Recursive layerwise parsing stays reusable as a subworkflow boundary.
- Canonical KG persistence is still server-canonical.
