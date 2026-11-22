# KG Doc Parser 0.2.21

Release tag: `v0.2.21`

## Included changes

- Align the released Kogwistar dependency with `0.7.0` for CPython installs.
- Align the PyPy 3.11 source-profile checkout and import contract with Kogwistar `v0.7.0`.
- No parser behavior or serialized document contract changes.

## Verification required before tagging

- Parser CI passes on CPython 3.12-3.14 and PyPy 3.11.
- Packaging, lock, and dependency-pin contract tests pass.
- The release workflow validates version `0.2.21` and tag `v0.2.21`.
