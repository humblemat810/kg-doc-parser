# KG Doc Parser 0.2.14

## Release Status

- Version: `0.2.14`
- Required Kogwistar release: `0.6.8`
- Required Kogwistar package: `kogwistar==0.6.8`
- Release tag: `v0.2.14`

## Changes

- Align the parser release with the merged Kogwistar `0.6.8` dependency pin.
- Keep the published parser behavior and structured-output safeguards from
  `0.2.13` unchanged.

## Compatibility

This is a backward-compatible parser patch release. Existing serialized
payloads and public parsing modes remain supported. The release supports
CPython 3.12-3.14 and PyPy 3.11.

## Release Gate

Publish only after the merged release commit passes the complete parser CI
matrix. Tag the merged commit as `v0.2.14` to trigger the PyPI release
workflow. Downstream repositories must pin the published `0.2.14` release only
after publication succeeds.
