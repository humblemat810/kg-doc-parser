# KG Doc Parser 0.2.18

Release tag: `v0.2.18`

## Release status

This release aligns the parser with published Kogwistar `0.6.10` and records
the OCR runtime and seam-hardening work from the OCR provider abstraction
branch. The dependency is verified from a clean environment: distribution
metadata and `kogwistar.__version__` both report `0.6.10`.

## Included fixes

- Keep OCR, grouping, summaries, and semantic publication under persisted
  Kogwistar workflow stages with stable recovery identities.
- Preserve and validate typed OCR workflow state across fresh-process resume,
  termination recovery, duplicate delivery, and artifact/database seams.
- Reject oversized summary evidence before it crosses the provider boundary.
- Include built-in PDF rasterizer defaults, installed `pdf2image` version, and
  `POPPLER_PATH` in renderer cache identity.
- Publish regenerated PNG artifacts through validated temporary files and
  atomic replacement while preserving the previous artifact on failure.
- Strengthen OCR runtime state typing with explicit `TypedDict` boundaries and
  keep parser CI compatible with PyPy 3.11 and CPython 3.12-3.14.

## Verification

- Selected deterministic parser CI: `429 passed, 1 skipped, 327 deselected`.
- Full parser Pyright: zero errors, warnings, and informational diagnostics.
- Focused OCR recovery, packaging, pin-contract, Ruff, and diff checks pass.
- The selected suite includes fresh-process root resume, real process
  termination, concurrent resume ownership, grouping recovery, and atomic
  artifact publication tests.

## Remaining tracked gates

The release does not claim that all operational recovery evidence is complete.
Linux SIGKILL coverage, both dual-store crash orderings with complete
Kogwistar checkpoint reconciliation, independent OCR completeness evidence,
and bounded-frontier performance evidence remain marked `PARTIAL` in the OCR
checklists. These are follow-up release gates, not silently waived behavior.
