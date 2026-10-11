# KG Doc Parser 0.2.10

## Release Status

- Version: `0.2.10`
- Required Kogwistar release: `0.6.7`
- Required Kogwistar package: `kogwistar==0.6.7`
- Release tag: `v0.2.10`

## Changes

- Preserve the authoritative parser source map when invoking layer review
  callbacks, so reviewed pointers are rehydrated from their offsets instead of
  being validated against an empty source map.
- Treat a valid source span as authoritative when the model's returned
  transcription differs in punctuation, escaping, or typography; persisted
  `verbatim_text` is reconstructed from the source slice.
- Include the parser source revision in review-cache identity so cached review
  results cannot cross source-map revisions.
- Add regression coverage for review callback forwarding, source-derived
  pointer rehydration, malformed and boundary pointer cases, and the CI test
  contract.

## Compatibility

This is a backward-compatible parser patch release. Existing serialized
payloads and public parsing modes remain supported. The release supports
CPython 3.12-3.14 and PyPy 3.11; no PyPy-specific syntax or runtime-only
dependency is introduced.

## Verification

The release branch must pass the full parser CI matrix, including lint, type
checking, lock verification, CPython 3.12-3.14, and PyPy 3.11. Focused local
verification for this change covers the review/source-map regression, pointer
edge cases, parser-state edge cases, and CI contract tests.

## Release Gate

Publish only after the merged release commit passes the complete parser CI
matrix. Tag the merged commit as `v0.2.10` to trigger the PyPI release
workflow. Downstream repositories must pin the published `0.2.10` release
only after publication succeeds.
