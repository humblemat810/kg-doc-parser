# KG Doc Parser 0.2.12

## Release Status

- Version: `0.2.12`
- Required Kogwistar release: `0.6.7`
- Required Kogwistar package: `kogwistar==0.6.7`
- Release tag: `v0.2.12`

## Changes

- Forward `reasoning_effort` as the first-class `ChatOpenAI` and
  `AzureChatOpenAI` constructor argument instead of placing it in
  `model_kwargs`.
- Preserve strict structured-output validation. A provider response that omits
  required fields or returns an unrelated shape remains a provider/schema
  failure; relaxing `extra` validation cannot repair missing required data.
- Add regression coverage for explicit reasoning forwarding on OpenAI and
  Azure-compatible providers.

## Compatibility

This is a backward-compatible parser patch release. Existing serialized
payloads and public parsing modes remain supported. The release supports
CPython 3.12-3.14 and PyPy 3.11; no PyPy-specific syntax or runtime-only
dependency is introduced.

## Verification

Focused provider contract and timeout verification passed with `33 passed`,
and Ruff passed for the changed Python files. The full release must still pass
the parser CI matrix before publication.

## Release Gate

Publish only after the merged release commit passes the complete parser CI
matrix. Tag the merged commit as `v0.2.12` to trigger the PyPI release
workflow. Downstream repositories must pin the published `0.2.12` release
only after publication succeeds.
