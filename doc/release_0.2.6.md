# kg-doc-parser 0.2.6

This patch release publishes the parser audit corrections merged after the
`0.2.5` release. It does not change the parser's public package structure or
add a new provider requirement.

## Included

- Adds the post-transform commit gate and parent-local requeue behavior for
  layerwise parsing.
- Enforces terminal content ownership and reports complete, atomic-valid,
  partial/degraded, and failed outcomes explicitly.
- Runs deterministic structural, source-bound, and exact-excerpt validation
  even when semantic review is disabled.
- Keeps stable PageIndex heading identities and parent links while preserving
  title text as content rather than using it only as a structural label.
- Uses source occurrence identity and authoritative revision excerpts for
  duplicate and pointer validation.
- Covers malformed, oversized, nested, table-heavy, and adversarial Markdown
  through the committed corrective test matrix and fixture documentation.
- Keeps paid or hosted provider calls optional and reports unexecuted live
  providers as `NOT VERIFIED`; local and offline evidence is not relabeled as
  cloud-provider coverage.

## Validation

The release candidate is based on merged `main` and is intended for the normal
CI matrix: CPython 3.12, 3.13, 3.14, and PyPy 3.11. Focused local checks
include the workflow-ingest contract and corrective matrix, packaging import
tests, Ruff, and whitespace validation. No provider credentials are required
for the release gate.

## Release Gate

Run the full parser CI suite and publish only from the matching `v0.2.6` tag.
The focused local command is:

```powershell
python -m pytest tests/test_packaging_imports.py tests/test_workflow_ingest_provider_contract_matrix.py -q -p no:cacheprovider
```

The audit checklist remains the source of truth for evidence that requires a
clean interpreter, backend-specific execution, or live provider credentials:
[`v0.2.4 corrective task checklist`](v0.2.4_corrective_task_checklist.md).
