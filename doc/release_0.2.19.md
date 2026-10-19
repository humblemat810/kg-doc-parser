# KG Doc Parser 0.2.19

Release tag: `v0.2.19`

## Summary

This release pins the persisted OCR workflow integration to published
Kogwistar `0.6.13`, including its fresh-process recovery improvements. It also
includes the OCR runtime/seam acceptance work from this feature branch and
keeps the release evidence explicit about measured OCR limitations.

## Included changes

- Use the published Kogwistar `0.6.13` package for CPython and the matching
  `v0.6.13` source checkout for the PyPy 3.11 Python-authority CI profile.
- Keep OCR pages, grouping, summaries, and semantic graph publication under the
  persisted Kogwistar workflow runtime, with SQLite used for OCR artifacts and
  page-state reconciliation.
- Preserve linear page-artifact reads and checkpoint-state growth in the
  deterministic 100/500/1000-page scaling regression.
- Preserve typed OCR loss diagnostics, strict artifact acceptance, and
  independent modality-specific evaluation against the checked-in table and
  chart ground truth. These measurements are fixture-specific and do not claim
  corpus-wide OCR completeness.

## Verification

- PyPI package and module report Kogwistar `0.6.13` (`metadata.version()` and
  `kogwistar.__version__` agree).
- Packaging and pin-contract tests: `16 passed`.
- Focused persisted-runtime recovery against installed Kogwistar 0.6.13:
  `8 passed` in 9 minutes, covering fresh-process resume, real process
  termination, missing-artifact replay, grouping recovery, and both
  SQLite/artifact-first and checkpoint-first commit orderings.
- Pyright: `0 errors, 0 warnings`; configured Ruff target passes; Poetry lock
  verification passes.
- Supported CPython 3.12-3.14 and PyPy 3.11 GitHub CI matrix: pending for this
  release candidate.
