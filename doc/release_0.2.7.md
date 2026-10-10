# KG Doc Parser 0.2.7

## Release status

- Version: `0.2.7`
- Required Kogwistar release: `0.6.5`
- Release tag: `v0.2.7`

## Changes

- Align the parser runtime dependency with the merged Kogwistar `0.6.5`
  release.
- Run the PyPy 3.11 source-profile lane against Kogwistar `v0.6.5`.
- Keep the parser's public parsing and provider contracts unchanged.

## Release gate

Publish this release only after the dependency lock resolves Kogwistar `0.6.5`
and the full parser CI matrix passes on CPython 3.12-3.14 and PyPy 3.11.
Downstream repositories must pin the merged `v0.2.7` commit only after this
release is published successfully.
