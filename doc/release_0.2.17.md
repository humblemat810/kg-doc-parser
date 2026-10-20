# KG Doc Parser 0.2.17

Release tag: `v0.2.17`

## Release status

This is the next parser release after `v0.2.16`. It consolidates the changes
merged since that release; earlier untagged `0.2.18`-`0.2.20` candidates are
superseded and are not separate releases. Do not tag or publish until the
required release checks below pass.

The parser is pinned to published Kogwistar `0.6.15`.

## Included changes

- Harden OCR source-item accounting, quality diagnostics, hybrid PDF evidence,
  and image/render artifact validation.
- Keep OCR pages, grouping, summaries, and semantic graph publication within
  persisted Kogwistar workflow execution, with SQLite retaining OCR artifacts
  and reconciliation state.
- Strengthen checkpoint decoding, recovery identity, and artifact/checkpoint
  crash-boundary handling while preserving resumable execution.
- Decompose OCR, layerwise, PageIndex, and parser-core internals into cohesive
  modules while preserving supported public imports and callback protocols.
- Retain support for PyPy 3.11 and CPython 3.12-3.14.
- Upgrade `pydantic-extension` to `0.0.8` for its intended performance
  improvement.

## Verification required before tagging

- Full parser CI passes on CPython 3.12-3.14 and PyPy 3.11.
- Pyright and the configured Ruff checks pass.
- Packaging, lock, pin-contract, recovery, and durable graph-publication tests
  pass against the published Kogwistar `0.6.15` package.
- The release workflow validates version `0.2.17` and tag `v0.2.17`.

This release does not claim corpus-wide OCR completeness. Remaining OCR quality
and provider-specific fidelity limitations stay documented in the OCR status
ledger.
