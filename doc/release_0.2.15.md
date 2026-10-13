# KG Doc Parser 0.2.15

## Release Status

- Version: `0.2.15`
- Required Kogwistar release: `0.6.8`
- Required Kogwistar package: `kogwistar==0.6.8`
- Release tag: `v0.2.15`

## Changes

- Enforce the zero-or-two-plus semantic branching invariant across layered,
  PageIndex, legacy, repair, frontier, and export boundaries.
- Preserve verified parents through failed refinements and support explicit
  atomic retention when no meaningful split is available.
- Rehydrate pointer text from the authoritative source map before validation,
  including review and repair paths.
- Add reusable structural diagnostics and regression coverage for malformed,
  unary, cyclic, deep, and stale parser results.

## Compatibility

This is a backward-compatible parser patch release. Existing serialized
payloads and public parsing modes remain supported. The release supports
CPython 3.12-3.14 and PyPy 3.11.

## Release Gate

Publish only after the merged release commit passes the complete parser CI
matrix. Tag the merged commit as `v0.2.15` to trigger the PyPI release
workflow. Downstream repositories must pin the published `0.2.15` release only
after publication succeeds.
