# Kogwistar-docparser 

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

## Notes

- `README.md`, env handling, and ingestion boundaries are still being cleaned up as part of the ongoing refactor.
- Runtime outputs such as `logs/`, local `.env`, caches, and generated artifacts should remain uncommitted.
- If behavior diverges between this repo and `kogwistar`, prefer the direction of the ongoing migration and refactor work.
