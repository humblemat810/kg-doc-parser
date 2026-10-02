# graph-knowledge-doc-parser 0.2.1

Released as tag `v0.2.1`.

## Highlights

- Aligns the parser with the released Kogwistar `0.6.2` package.
- Keeps the parser MCP dependency on the MCP 2.x contract.
- Adds bounded provider generation and context handling for structured parser
  workflows, including provider-specific limits and Claude context overflow
  handling.
- Consolidates OCR, page-index, layered parsing, conversation-graph, and
  resumability contracts with durable test coverage.
- Uses Joblib on CPython and DiskCache on PyPy for the shared parser cache.
- Adds normal PyPy 3.11 CI coverage plus the local Linux/PyPy test runner.
- Publishes the parser through the gated PyPI release workflow.

## Compatibility

- Python: CPython 3.12-3.14 and PyPy 3.11.
- Kogwistar: `0.6.2`.
- MCP: `>=2.2.0,<3`.
- PyPy uses the DiskCache provider; CPython uses Joblib.

## Validation

The release was validated by the parser CI suite, packaging/import contract
tests, PyPy 3.11 coverage, dependency lock verification, and the release
workflow. Downstream applications should pin `graph-knowledge-doc-parser==0.2.1`
when reproducible parser behavior is required.

