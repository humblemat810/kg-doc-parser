# KG Doc Parser 0.2.8

## Release Status

- Version: `0.2.8`
- Required Kogwistar release: `0.6.6`
- Required Kogwistar package: `kogwistar==0.6.6`
- Release tag: `v0.2.8`

## Changes

- Rehydrate provider-facing layer pointers from authoritative source offsets.
- Keep review payloads offset-only at the typed model boundary, while accepting
  legacy extra fields from older providers.
- Make pointer repair source-authoritative: conflicting valid offsets are
  rejected, and relocation requires one unique exact text match.
- Keep successful repairs source-derived so offsets and `verbatim_text` agree.
- Align the runtime dependency and lockfile with published Kogwistar `0.6.6`.

## Release Gate

Publish only after the full parser CI matrix passes on CPython 3.12-3.14 and
PyPy 3.11, including lint and lock verification. The focused pointer/page-index
validation currently passes with `92 passed, 8 skipped`; this is supporting
evidence, not a substitute for the full release matrix.

Downstream repositories must pin the merged `v0.2.8` commit only after this
release is published successfully.
