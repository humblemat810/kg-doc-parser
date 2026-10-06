# kg-doc-parser 0.2.4

## PageIndex Evaluation Corpus

This patch release adds a non-sensitive adversarial Markdown fixture corpus
covering:

- escaped-pipe and separator-row tables;
- multiline and heading-adjacent tables;
- fenced code, blockquotes, and nested lists;
- inline markup, footnotes, and HTML tables;
- Unicode and malformed Markdown;
- a larger table and long-document boundary case.

The fixtures assert that heuristic PageIndex parsing does not crash, preserves
complete source coverage, and keeps every emitted pointer aligned with the
authoritative source text. The focused regression suite passes with
`50 passed, 8 skipped`.

The release also records a real text-only local Bonsai run using the
OpenAI-compatible provider boundary. The Bonsai vision projector was disabled
and the model used a 16,384-token context. All 13 adversarial documents
completed with full reported coverage. This demonstrates provider execution
and grounding compatibility, not complete semantic-quality validation; the
current CLI summary does not yet expose node-level metrics.

Known limitations remain explicit: fenced-code heading detection, nested-list
semantics, footnote relationships, and structural table row/cell parsing need
dedicated parser work rather than being inferred from source coverage alone.

See [`page_index_adversarial_fixture_report.md`](page_index_adversarial_fixture_report.md)
for the full deterministic and Bonsai results.

## Release Gate

Run the full parser CI suite, including CPython 3.12-3.14 and PyPy 3.11.
The focused local regression command is:

```powershell
python -m pytest tests/test_page_index_adversarial_fixtures.py tests/test_workflow_ingest_page_index_pipeline.py -q -p no:cacheprovider
```
