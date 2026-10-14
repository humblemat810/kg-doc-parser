# KG Doc Parser 0.2.16

## Release Status

- Version: `0.2.16`
- Required Kogwistar release: `0.6.8`
- Required Kogwistar package: `kogwistar==0.6.8`
- Release tag: `v0.2.16`

## Changes

- Preserve authoritative source pointers through review, repair, and PageIndex
  normalization.
- Reject malformed ancestry before wrapper normalization can hide it.
- Enforce the zero-or-two-plus semantic branching invariant across the public
  parser paths while retaining explicit atomic results.
- Add the machine-readable P001-P018 and W001-W010 regression fixture pack and
  executable coverage through the public parser APIs.

## Compatibility

This is a backward-compatible parser patch release. Existing serialized
payloads and public parsing modes remain supported. The release supports
CPython 3.12-3.14 and PyPy 3.11.

## Release Gate

Publish only after the merged release commit passes the complete parser CI
matrix. Tag the merged commit as `v0.2.16` to trigger the PyPI release
workflow. Downstream repositories must pin the published `0.2.16` release
only after publication succeeds.
