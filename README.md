# Kogwistar-docparser 

If you want the shortest path into the project, start with [QUICKSTART.md](QUICKSTART.md).

## Graph Knowledge Doc Parser

Utilities and experiments for document ingestion, PDF splitting, OCR, and page-level parsing. Refactored from the kogwistar project as a stand alone ingestor.

## Status

This repository is still being refactored and should be treated as work in progress.

The document ingestion pipeline is currently being extracted and consolidated into the main `kogwistar` repository. Until that refactor is complete, this repo should be considered an active staging area for parser and ingestion-related work.

## What Is Here

- PDF splitting and image generation helpers in `src/pdf2png.py`
- Gemini-based OCR and page parsing flows in `src/ocr.py`
- SQLite-based ingestion telemetry in `src/document_ingester_logger.py`
- File discovery and filtering helpers in `src/utils/file_loaders.py`
- Experimental and regression-style tests under `tests/`

## Workflow Surface

The reusable workflow-ingest code now has three layers:

- Python APIs, which are the primary contract for tests and orchestration
- CLI entrypoints, which are thin wrappers around those APIs
- composable subworkflows for OCR, page-index parsing, and recursive layerwise parsing

The reusable helpers live under `src/workflow_ingest/` and are designed so the
same core logic can be called from tests, scripts, and higher-level workflow
code without duplicating orchestration.

### CLI Commands

After `poetry install`, the repo exposes a `workflow-ingest` command family:

```powershell
workflow-ingest --help
workflow-ingest ocr --help
workflow-ingest page-index --help
workflow-ingest layerwise --help
workflow-ingest demo --help
workflow-ingest ocr-smoke-assets --help
```

Typical examples:

```powershell
workflow-ingest ocr tests\.tmp_workflow_ingest_ocr\generated_smoke_assets\ocr_smoke_document.pdf --output-dir logs\ocr_run
workflow-ingest ocr tests\.tmp_workflow_ingest_ocr\generated_smoke_assets --output-dir logs\ocr_batch
workflow-ingest page-index tests\fixtures\page_index\sample_page_index.txt --output-dir logs\page_index
workflow-ingest layerwise tests\.tmp_workflow_ingest_ocr\manual_cases\ollama\glm-ocr_latest\image\artifacts\legacy_split_pages\ocr-manual-ollama-image --output-dir logs\layerwise
workflow-ingest ocr-smoke-assets --output-dir tests\.tmp_workflow_ingest_ocr\generated_smoke_assets
```

### Reusable Artifacts

The workflow outputs are intentionally inspectable on disk:

- `workflow-events.jsonl`: readable step trail for the outer orchestration layer
- `ocr-state.sqlite`: authoritative OCR/render resume state
- `ocr-progress.json`: human-readable mirror of the current OCR state
- `ocr-summary.json`: final OCR run summary
- `legacy_split_pages/<document>/page_N.json`: legacy-compatible OCR page artifacts
- `rendered_pages/<document>/page_N.png`: rasterized page images
- `page-index-summary.json`: page-index run summary
- `layerwise-summary.json`: recursive layerwise parser summary
- `layerwise-graph.json`: legacy recursive layerwise graph payload

### Python APIs

If you want to embed the pipelines directly, the main helpers are:

- `src.workflow_ingest.run_ocr_source_workflow(...)`
- `src.workflow_ingest.run_ocr_batch_workflow(...)`
- `src.workflow_ingest.parse_page_index_document(...)`
- `src.workflow_ingest.run_page_index_source_workflow(...)`
- `src.workflow_ingest.run_layerwise_source_workflow(...)`
- `src.workflow_ingest.run_demo_harness_workflow(...)`

Those helpers are meant to stay stable and are what the CLI layer calls under
the hood.

## Setup

1. Create and activate a Python 3.13 environment.
2. Install dependencies with Poetry:

```powershell
poetry install
```

3. Create a local env file from the example:

```powershell
Copy-Item .env.example .env
```

4. Add your local `GOOGLE_API_KEY` and any optional file-list paths needed for your workflow.

## Environment Variables

The project currently expects or optionally uses:

- `GOOGLE_API_KEY`: required for Gemini OCR and LLM-backed parsing flows
- `LANGSMITH_TRACING`: optional LangSmith tracing toggle
- `ocr_file_list`: optional allow-list file for OCR runs
- `split_raw_file_list`: optional allow-list file for PDF splitting runs
- `answer_export_list`: optional export list path used by local workflows

An example template is provided in [`.env.example`](/c:/Users/chanh/Documents/kg_doc_parser/.env.example).

## Provider Guide

The workflow layer is vendor-neutral, but the concrete OCR, parser, and embedding
backends are selected by config.

### OCR Provider Examples

- Google GenAI OCR:
  - `KG_DOC_OCR_PROVIDER=gemini`
  - `KG_DOC_OCR_MODEL=gemini-2.5-flash`
- Ollama OCR or vision-capable local model:
  - `KG_DOC_OCR_PROVIDER=ollama`
  - `KG_DOC_OCR_MODEL=llava:latest`
  - `KG_DOC_OCR_BASE_URL=http://127.0.0.1:11434`
- Vertex AI OCR:
  - `KG_DOC_OCR_PROVIDER=vertex`
  - `KG_DOC_OCR_MODEL=gemini-2.5-pro`
  - `KG_DOC_OCR_PROJECT=my-project`
  - `KG_DOC_OCR_LOCATION=us-central1`

### Parser / LLM Provider Examples

The parser provider is the chat model used for semantic parsing, layer review,
and structured extraction.

- LangChain Google GenAI:
  - `KG_DOC_PARSER_PROVIDER=gemini`
  - `KG_DOC_PARSER_MODEL=gemini-2.5-flash`
- ChatGPT / OpenAI REST:
  - `KG_DOC_PARSER_PROVIDER=openai`
  - `KG_DOC_PARSER_MODEL=gpt-4.1-mini`
  - `KG_DOC_PARSER_API_KEY_ENV=OPENAI_API_KEY`
- LangChain Ollama:
  - `KG_DOC_PARSER_PROVIDER=ollama`
  - `KG_DOC_PARSER_MODEL=llama3.1`
  - `KG_DOC_PARSER_BASE_URL=http://127.0.0.1:11434`
- LangChain Vertex AI:
  - `KG_DOC_PARSER_PROVIDER=vertex`
  - `KG_DOC_PARSER_MODEL=gemini-2.5-pro`
  - `KG_DOC_PARSER_PROJECT=my-project`
  - `KG_DOC_PARSER_LOCATION=us-central1`

### Recipe Parsing Example

If you are parsing a cooking recipe, one practical split is:

- OCR on Gemini or another vision model
- parser on OpenAI, Ollama, or Vertex AI

For example:

```powershell
KG_DOC_OCR_PROVIDER=gemini
KG_DOC_OCR_MODEL=gemini-2.5-flash
KG_DOC_PARSER_PROVIDER=openai
KG_DOC_PARSER_MODEL=gpt-4.1-mini
KG_DOC_PARSER_API_KEY_ENV=OPENAI_API_KEY
KG_DOC_EMBED_PROVIDER=fake
```

That setup can extract a recipe into structured graph data such as:
- ingredients
- steps
- tools
- timers
- inferred sections like `prep`, `cook`, and `serve`

### Embedding Examples

- Fake deterministic CI embedding:
  - `KG_DOC_EMBED_PROVIDER=fake`
- OpenAI embeddings:
  - `KG_DOC_EMBED_PROVIDER=openai`
  - `KG_DOC_EMBED_MODEL=text-embedding-3-small`
- Vertex AI embeddings:
  - `KG_DOC_EMBED_PROVIDER=vertex`
  - `KG_DOC_EMBED_MODEL=text-embedding-004`
- Ollama embeddings:
  - `KG_DOC_EMBED_PROVIDER=ollama`
  - `KG_DOC_EMBED_MODEL=nomic-embed-text`

Note:

- `embedding_space` in the workflow ingest models is currently a metadata and
  routing-intent label.
- It does not yet imply that the engine is using a separate embedder per space.
- The current engine bootstrap still wires one embedding function per engine
  instance, while the multi-space routing proposal remains a future Kogwistar
  core concern.

## Running Tests

Some tests are integration-style and expect local document folders and API credentials to exist. That means not every test is portable in a clean checkout.

To run the test suite:

```powershell
pytest -q
```

If you only want to work on isolated units, review the test files first and run a narrower subset.

## Demo Harness

There is now a manual workflow-ingest demo harness that can run the end-to-end flow against:

- an in-process isolated FastAPI server
- a subprocess-hosted local server
- an already running external Kogwistar server

Example:

```powershell
.venv\Scripts\python.exe scripts\run_workflow_ingest_demo.py --output-dir logs\workflow_ingest_demo
```

External live server example:

```powershell
.venv\Scripts\python.exe scripts\run_workflow_ingest_demo.py --server-mode external_http --external-base-url http://127.0.0.1:28110
```

Demo artifacts are written into the chosen output directory:

- `probe-events.jsonl`: demo-friendly step and lifecycle probe events
- `demo-summary.json`: run summary, persistence result, and artifact pointers
- `llm-cache/`: workflow-native cached proposal/review call results
- `engines/`: local workflow and conversation graph storage for the run
- `server-data/`: isolated server-side persistence directory when the harness boots its own server

Notes:

- The workflow-native layer proposal/review path uses deterministic file-backed caching to reduce repeated token cost and compute time.
- The legacy parser path also supports a redirected `joblib` cache via `KG_DOC_PARSER_JOBLIB_CACHE_DIR`.
- Probe logging is separate from CDC and conversation graph traces, so demos can show a short readable event trail without digging into runtime internals.

## OCR And Parsing Workflows

The newer workflow-first paths are designed as reusable subworkflows:

- OCR image/PDF ingest
  - resumable via `ocr-state.sqlite`
  - emits `workflow-events.jsonl`
  - keeps legacy OCR page artifacts on disk
- page-index parsing
  - heuristic mode for deterministic structure extraction
  - Ollama mode for local parser-backed parsing
- recursive layerwise parsing
  - wraps the legacy recursive parser in a reusable workflow runner

These can be invoked from Python directly or through the `workflow-ingest`
CLI family, depending on whether you want reusable orchestration or a quick
shell command.

## Notes

- `README.md`, env handling, and ingestion boundaries are still being cleaned up as part of the ongoing refactor.
- Runtime outputs such as `logs/`, local `.env`, caches, and generated artifacts should remain uncommitted.
- If behavior diverges between this repo and `kogwistar`, prefer the direction of the ongoing migration and refactor work.
