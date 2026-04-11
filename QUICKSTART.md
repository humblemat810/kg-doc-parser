# Quickstart

This project has a workflow-native ingest demo path with fake-first tests and optional real-server modes.

## 1. Use The Workspace Venv

Make sure you are using:

```powershell
.venv\Scripts\python.exe
```

## 2. Run The Fast Tests

```powershell
.venv\Scripts\python.exe -m pytest tests/test_workflow_ingest_contracts.py tests/test_workflow_ingest_resolver_invariants.py tests/test_workflow_ingest_layerwise_parser.py -q
```

These cover:
- workflow contracts
- resolver retries
- overlap and coverage conflicts
- strategy switching
- fake parser and review behavior

## 3. Recommended Test Split

Use these tiers depending on how much you want to exercise:

Fast local loop:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_workflow_ingest_contracts.py tests/test_workflow_ingest_resolver_invariants.py tests/test_workflow_ingest_layerwise_parser.py tests/test_workflow_ingest_conversation_graph.py tests/test_workflow_ingest_provider_adapters.py -q
```

This tier now uses Kogwistar's in-memory backend for the workflow-facing engine tests.

Normal workflow/demo CI:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_workflow_ingest_contracts.py tests/test_workflow_ingest_resolver_invariants.py tests/test_workflow_ingest_layerwise_parser.py tests/test_workflow_ingest_conversation_graph.py tests/test_workflow_ingest_provider_adapters.py tests/test_workflow_ingest_demo_harness.py -q
```

Full backend / server CI:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_workflow_ingest_backends_ci_full.py tests/test_workflow_ingest_server_e2e.py tests/test_workflow_ingest_demo_harness.py -m ci_full -q
```

This tier keeps the persisted backend matrix and server-backed coverage.

## 4. Run The Demo Harness

```powershell
.venv\Scripts\python.exe scripts\run_workflow_ingest_demo.py --output-dir logs\workflow_ingest_demo
```

This writes:
- `probe-events.jsonl`
- `demo-summary.json`
- `llm-cache/`
- `engines/`
- `server-data/` when the harness starts its own server

## 5. Run Against A Live Server

```powershell
.venv\Scripts\python.exe scripts\run_workflow_ingest_demo.py --server-mode external_http --external-base-url http://127.0.0.1:28110
```

## 6. Read Next

- [README.md](README.md)
- [workflow_ingest_resolver_orchestration.md](workflow_ingest_resolver_orchestration.md)
- [workflow_ingest_layerwise_proposal.md](workflow_ingest_layerwise_proposal.md)

## Mental Model

- Parsing is owned locally in this repo.
- Canonical KG persistence is server-canonical.
- The workflow can run directly in-process for development and tests.
- `embedding_space` is a metadata label for now, not proof of a separate embedding pipeline.
