# KG Doc Parser 0.2.13

## Release Status

- Version: `0.2.13`
- Required Kogwistar release: `0.6.7`
- Required Kogwistar package: `kogwistar==0.6.7`
- Release tag: `v0.2.13`

## Changes

- Reject router/provider payloads with an unrelated free-form JSON shape at the
  structured-output validation boundary.
- Add regression coverage for missing required fields and unexpected top-level
  fields in boundary proposals.
- Document that OpenAI-compatible gateways may accept but drop or translate
  nested `json_schema` response-format fields.

## Compatibility

This is a backward-compatible parser patch release. Existing serialized
payloads and public parsing modes remain supported. The release supports
CPython 3.12-3.14 and PyPy 3.11.

## Verification

The focused structured-output and provider-schema suite passed with `50
passed`, and Ruff passed for the changed test file.

## Release Gate

Publish only after the merged release commit passes the complete parser CI
matrix. Tag the merged commit as `v0.2.13` to trigger the PyPI release
workflow. Downstream repositories must pin the published `0.2.13` release
only after publication succeeds.
