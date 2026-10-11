# KG Doc Parser 0.2.11

## Release Status

- Version: `0.2.11`
- Required Kogwistar release: `0.6.7`
- Required Kogwistar package: `kogwistar==0.6.7`
- Release tag: `v0.2.11`

## Changes

- Bound provider transport timeouts to the configured parser or OCR timeout
  for OpenAI, Azure, Anthropic, Google GenAI, Vertex, Ollama, and Codex
  adapters.
- Prevented the Codex timeout environment override from exceeding the parser
  timeout, avoiding unnecessarily long-lived in-flight calls after wrapper
  timeout.
- Added provider-constructor and timeout regression coverage without making
  network requests.
- Exposed `KG_DOC_PARSER_REASONING_EFFORT` and
  `KG_DOC_OCR_REASONING_EFFORT` in `.env.example` and documented their use
  with OpenAI-compatible endpoints.
- Documented in-flight slot retention, orphaned-call metrics, and the limits
  of thread-based cancellation.

## Compatibility

This is a backward-compatible parser patch release. Existing serialized
payloads and public parsing modes remain supported. The release supports
CPython 3.12-3.14 and PyPy 3.11; no PyPy-specific syntax or runtime-only
dependency is introduced.

## Verification

Focused provider contract and timeout verification passed with `32 passed`,
and Ruff passed for the changed Python files. The full release must still pass
the parser CI matrix before publication.

## Release Gate

Publish only after the merged release commit passes the complete parser CI
matrix. Tag the merged commit as `v0.2.11` to trigger the PyPI release
workflow. Downstream repositories must pin the published `0.2.11` release
only after publication succeeds.
