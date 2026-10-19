# KG Doc Parser 0.2.20

Release tag: `v0.2.20`

## Summary

This release candidate completes the typing-first decomposition of the OCR,
layerwise, PageIndex, and parser-core workflows while preserving their public
entry points. It also aligns the runtime dependency with published Kogwistar
`0.6.14`.

## Included changes

- Split large workflow modules into cohesive state, provider, validation,
  persistence, and orchestration modules without changing supported public
  imports.
- Keep checkpoint decoding and provider boundaries typed, with explicit
  validation for serialized state.
- Update CPython dependency, PyPy source checkout, exported requirements, and
  lock metadata to Kogwistar `0.6.14`.
- Update regression tests to patch implementation owners after module moves.

## Verification Status

- Kogwistar `0.6.14` pin-contract and packaging tests: `16 passed`.
- Pointer-edge and focused corrective-matrix tests: `64 passed`.
- Full CI-marked parser suite: still in progress; no release tag or publication
  is authorized until the GitHub CI matrix passes.
- Pyright, Ruff, full CPython 3.12-3.14, and PyPy 3.11 results remain release
  gates for this candidate.
