# KG Doc Parser Fixture Execution Status

This checklist tracks executable evidence for
`kg_doc_parser_fixture_acceptance_checklist.md` and
`kg_doc_parser_regression_investigation_and_requirements.md`. It is separate
from those requirement documents so an unchecked requirement is never mistaken
for a passing test.

## Completed In This Branch

- [x] Checked in `tests/fixtures/kg_doc_parser_test_payloads.json` with all
  P001-P018 and W001-W010 cases.
- [x] Load the fixture pack and assert the complete mandatory case inventory.
- [x] Run P001-P018 through the public `parse_page_index_document` path with
  deterministic mocked materialization.
- [x] Assert exact source-pointer rehydration for every accepted public result.
- [x] Assert zero-or-two-plus branching for every accepted public result.
- [x] Reject malformed parent ancestry before PageIndex wrapper normalization.
- [x] Exercise W001 boundary and excerpt CUD repair through
  `commit_layer_children`.
- [x] Exercise W002 retry-budget exhaustion and W003 explicit strategy
  transition without treating fallback as success.
- [x] Exercise W004 one-layer PageIndex output through
  `parse_page_index_layer`.
- [x] Exercise W005 triage selection and W006 provider-failure fallback.
- [x] Exercise W007 cache reuse and W008 source/provider cache invalidation.
- [x] Exercise W010 stale-document rejection through the source-revision
  fingerprint gate.
- [x] Link W009 standalone checkpoint resume to the existing
  `test_conversation_graph_resume_from_suspended_checkpoint` coverage.
- [x] Preserve the existing nested-descendant, atomic-retention, pointer, and
  terminal-ownership regressions.
- [x] `poetry run ruff check` for the changed parser/test files.
- [x] `poetry run pyright kg_doc_parser/workflow_ingest/page_index.py
  kg_doc_parser/workflow_ingest/semantics.py
  tests/test_fixture_payload_regressions.py` (`0 errors`).
- [x] `poetry check --lock` (existing Poetry metadata deprecation warnings
  only).
- [x] `git diff --check`.

## Pending Before Push

- [x] Run the parser's full focused workflow suite with
  `-p no:cacheprovider` and `-o log_cli=false`.
- [x] Run parser lock verification, Pyright, and the complete CI command.
- [ ] Run supported CPython 3.12-3.14 and PyPy 3.11 checks.
- [x] Record exact local results and implementation commit `9e34a31`.
- [x] Push the feature branch at `aef4051`.
- [ ] Open the GitHub PR and monitor its CPython 3.12-3.14 and PyPy 3.11
  checks to completion.

## Current Focused Evidence

The fixture regression module currently passes 28 tests, including all 18 tree
cases and the W001-W010 workflow contract checks. The first full run reached
`292 passed, 1 failed`; that one failure was the older cache test invocation
using the host's unwritable temporary directory. After the workspace-local
cache fix, the complete focused command passes:

```text
297 passed, 118 deselected, 32 warnings in 1002.96s (0:16:42)
```

The warnings are existing Kogwistar runtime underscore-state and legacy-update
warnings, not new failures from this branch.

Static evidence so far:

- `poetry run ruff check tests/test_fixture_payload_regressions.py`: passed.
- `poetry run pyright kg_doc_parser/workflow_ingest/page_index.py
  kg_doc_parser/workflow_ingest/semantics.py
  tests/test_fixture_payload_regressions.py`: `0 errors, 0 warnings, 0 informations`.
- `poetry check --lock`: passed with existing Poetry metadata deprecation
  warnings only.
- `git diff --check`: passed.
