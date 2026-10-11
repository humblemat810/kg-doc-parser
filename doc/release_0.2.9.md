# KG Doc Parser 0.2.9

## Release Status

- Version: `0.2.9`
- Required Kogwistar release: `0.6.7`
- Required Kogwistar package: `kogwistar==0.6.7`
- Release tag: `v0.2.9`

## Changes

- Align the parser runtime dependency and lockfile with published Kogwistar
  `0.6.7`.
- Run the PyPy 3.11 source-profile lane against Kogwistar `v0.6.7`.
- Include the parser-state and document-structure adversarial coverage from the
  boundary-pointer hardening work.

## Release Gate

Publish only after the full parser CI matrix passes on CPython 3.12-3.14 and
PyPy 3.11, including lint, type checking, and lock verification.

Downstream repositories must pin the merged `v0.2.9` commit only after this
release is published successfully.
