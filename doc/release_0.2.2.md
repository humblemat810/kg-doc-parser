# kg-doc-parser 0.2.2

## Layerwise Strategy Arbitration

This release records the per-layer parser routing contract introduced in the
workflow-ingest implementation.

- The safe deterministic default is `layer_excerpt`, then `layer_boundary`,
  then `page_index`.
- A caller may provide a complete permutation such as
  `layer_boundary,layer_excerpt,page_index`.
- An explicit strategy selects the first operator for the current layer while
  retaining the configured remaining operators as fallbacks.
- With triage enabled, the provider may select any currently enabled operator
  for the current layer. A failed operator is disabled for that layer and
  routing returns to triage or the deterministic cascade.
- With triage disabled, the configured order is deterministic and exhaustion
  reaches the explicit `parse_failure` route.
- Invalid strategy configuration is rejected before parsing and is represented
  by the workflow's `strategy_selection_failed` failure edge.

## PageIndex Semantics

`page_index` is a one-layer refinement operator, not a recursive ownership
mode. It may produce a structural heading container, its grounded title-text
leaf, and immediate content children. The parent aggregate span remains the
source-owning span; title and content leaves retain exact source pointers.

Expandable descendants return to normal frontier processing and receive the
configured default strategy or a fresh triage decision. PageIndex is therefore
the final deterministic fallback by default, but it is not permanently forced
on descendants and may be selected earlier when triage is enabled.

PageIndex summaries are advisory metadata. Source text, source-map identity,
and hydrated pointers remain authoritative. The workflow does not accept
model-authored source pointers or recursively materialize an unbounded tree.

## State And Diagnostics

Each selected operator records its attempt count and a bounded execution event
in workflow state. The state records layer and parent identity, selected
strategy, attempt number, success/failure, and bounded failure reasons. This
is workflow audit state, not hidden model reasoning and not canonical document
content.

## Compatibility

- Existing grounded source maps and graph payloads remain authoritative.
- Existing direct child proposal mode remains supported.
- Existing legacy input and omitted strategy arguments retain their previous
  compatibility behavior.
- PageIndex summary text can be disabled without changing source grounding.
- The release CI matrix covers CPython 3.12-3.14 and a PyPy 3.11
  compatibility profile; CPython 3.11 is not a supported runtime. The
  package metadata remains `python = \"^3.12\"`, so PyPy 3.11 is validated as
  a source/CI profile rather than advertised as a Poetry-installable runtime
  until its packaging constraint is intentionally widened.

## Release Gate

Before tagging or publishing, run the full parser CI suite, including the
strategy, PageIndex, layerwise, workflow contract, packaging, and PyPy 3.11
jobs. The focused regression command is:

```powershell
python -m pytest tests/test_workflow_ingest_strategy.py tests/test_workflow_ingest_page_index_pipeline.py -q -p no:cacheprovider
```

The full release is not considered ready from the focused suite alone.
