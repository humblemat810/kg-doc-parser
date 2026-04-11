# Demo Harness Failure Follow-Up

## Summary

`tests/test_workflow_ingest_demo_harness.py::test_demo_harness_writes_probe_summary_and_cache`
is failing in the demo harness path with `artifacts.status == "failure"`.

The current failure is not in the probe summary or cache plumbing itself. The run gets
all the way to canonical persistence, and then the server-side tree upsert path rejects
an endpoint token that is already present in the exported graph payload.

## Reproduction

Run:

```bat
.venv\Scripts\python.exe -m pytest tests/test_workflow_ingest_demo_harness.py::test_demo_harness_writes_probe_summary_and_cache -q
```

Observed outcome:

- `artifacts.status == "failure"`
- `artifacts.canonical_write_confirmed == False`
- `artifacts.persistence_mode == "server_canonical"`
- `artifacts.kg_authority == "server"`

## What the probe shows

The probe progresses normally through:

- `workflow.run_started`
- the layered workflow steps
- `export_graph`
- `persist_canonical_graph`

The failure happens inside `persist_canonical_graph` when the demo harness calls the
server tree-upsert endpoint.

The final probe tail contains:

```text
workflow.step_exception
step: persist_canonical_graph
error: RuntimeError('canonical server persistence failed: status=500 body=...')
```

And the server-side root error is:

```text
ValueError: Unresolvable node endpoint token: demo-harness-doc|root
```

## Where it fails

The failure path is:

1. `src/workflow_ingest/demo_harness.py`
2. `src/workflow_ingest/clients.py`
3. `kogwistar/kogwistar/server_mcp_with_admin.py`
4. `kogwistar/kogwistar/engine_core/subsystems/persist.py`

The exact server route involved is:

- `POST /api/document.upsert_tree`

That route currently forwards the payload into:

- `PersistSubsystem.persist_document_graph_extraction(...)`

which then calls:

- `resolve_llm_ids(...)`

The resolver rejects the endpoint token:

- `demo-harness-doc|root`

even though that node is part of the same payload.

## Why this looks like a contract mismatch

The demo export uses stable, human-readable node IDs for the graph payload.
The persistence resolver currently accepts:

- temp ids like `nn:*` and `ne:*`
- aliases
- UUIDs
- label-based fallback

But it does not appear to accept an exact in-batch node id as a valid endpoint token.

That means a graph can be structurally valid and still fail at persistence time if its
edge endpoints are written using the stable ids produced by the workflow export.

## Relevant payload shape

The workflow export bundle is built from:

- `src/workflow_ingest/semantics.py::semantic_tree_to_kge_payload(...)`

That payload includes nodes and edges with ids such as:

- `demo-harness-doc|root`
- `demo-harness-doc|section|overview`

The exported edges point at those ids directly.

## Why the test is meaningful

This is not just a demo-only issue. It exercises the real end-to-end contract:

- fake layered parse input
- workflow execution
- export bundle creation
- canonical persistence through the server

So the failure is a useful signal that the server persistence contract does not yet
accept the graph shape produced by the demo harness.

## What already works

The earlier demo-harness issues are not the current blocker anymore:

- document seeding succeeds
- workflow execution succeeds through `export_graph`
- the failure is specifically at canonical graph persistence

## Suggested maintainer follow-up

The likely next steps are:

1. Decide whether `resolve_llm_ids(...)` should accept exact in-batch node ids as valid
   edge endpoint tokens.
2. Or, route the demo export through the persistence API that is intended for already
   resolved canonical graph ids.
3. Add a regression test covering `POST /api/document.upsert_tree` with stable ids like
   `demo-harness-doc|root`.

## Notes

- I am not proposing code changes in this note.
- This file is intended as a maintainer handoff for the current failure only.
