# KG Doc Parser 0.2.17

Release tag: `v0.2.17`

## Release status

This release contains the OCR/grouping recovery hardening from the current
feature branch. It rejects orphaned grouping frontiers instead of allowing a
new document tree to overwrite recoverable checkpoint state.

The release remains compatible with the currently published Kogwistar
dependency. The parser pin is intentionally unchanged until the Kogwistar
`0.6.11` ownership-lease fix is merged and published; downstream release
verification must then confirm the installed package before changing that pin.

## Included fixes

- Reject grouping frontiers that survive without their document-tree revision.
- Preserve fail-closed recovery behavior after interrupted OCR/grouping runs.
- Keep materialized OCR page paths inside the rendered directory, reject
  duplicate/non-positive page numbers, and publish byte payloads atomically.
- Record the broad OCR audit as an explicit B01-B14 release ledger; unresolved
  fidelity, quality, performance, and rasterization gates remain open.
- Keep the existing PyPy 3.11 and CPython 3.12-3.14 compatibility contract.

## Verification required before tagging

- Full parser CI on CPython 3.12-3.14 and PyPy 3.11.
- Kogwistar `0.6.11` published and verified before any dependency-pin update.
- Packaging, lock, Ruff, and parser recovery contract checks.
