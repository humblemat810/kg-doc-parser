# Page-Index Adversarial Fixture Report

## Scope

This report records a real heuristic PageIndex run against the existing
fixtures and thirteen representative Markdown documents. The run used the
current mergeable parser commit and **did not use Codex, the Codex bridge, a
local deployed model, Ollama, or any other LLM provider**. It exercised only
the deterministic `mode="heuristic"` PageIndex implementation. The acceptance target
for this baseline is lossless source grounding, not full Markdown semantic
interpretation.

Command used:

```text
python -m pytest tests/test_page_index_adversarial_fixtures.py -q -p no:cacheprovider -o log_cli=false
```

Result: `13 passed in 14.56s`.

The existing PageIndex regression suite also passed: `37 passed, 8 skipped`.

This is therefore a source-grounding and deterministic-structure smoke test,
not an LLM quality or provider-throughput benchmark. A separate provider
evaluation is still needed for `layer_excerpt`, `layer_boundary`, and any
future table-structure strategy.

## Text-Only Bonsai Run

A second run used the local Bonsai model through the parser's OpenAI-compatible
provider boundary. It was a real model run, not a fake provider and not
Codex.

Configuration:

- Model: `Ternary-Bonsai-2-27B-PTQ1_0.gguf`
- Runtime: local `llama-server`
- Vision projector: **disabled**; no `--mmproj` argument was supplied
- Endpoint: `http://127.0.0.1:8182/v1`
- Context slot: `16,384` tokens
- Parallel slots: `1`
- GPU offload: `-ngl 999`
- Reasoning effort: `medium`
- Reasoning budget: `512` tokens
- Parser mode: `openai`, structured PageIndex output
- Strategy triage: disabled
- Input: all 13 adversarial Markdown fixtures

Result: `13/13` files succeeded, each reported `overall_coverage=1.0`, and
the run took approximately 9 minutes 9 seconds. The workflow event trace
recorded file start/finish events for every input and ended with
`cli.command_finished status=ok`.

The llama-server log also confirmed:

- model loaded successfully without the vision tower;
- `n_ctx_slot=16384` for requests;
- prompt-cache reuse between requests;
- generation throughput varying roughly from 16 to 34 tokens/second as the
  run progressed.

The provider summaries reported `max_depth=1` for 12 fixtures and `max_depth=5`
for the malformed-Markdown fixture. This means the provider path validated and
grounded the documents, but it usually produced a shallow one-layer result.
The malformed case producing a deeper tree is an inconsistency worth reviewing;
it must not be interpreted as evidence that malformed Markdown was parsed more
correctly.

The current CLI summary is insufficient for detailed semantic evaluation: it
does not persist node counts, node-type distributions, provider validation
diagnostics, or the structured provider response. The workflow event trace and
llama-server log provide operational observability, but a future debug mode
should persist bounded structural metrics and rejection/fallback reasons.

## Observed Results

| Fixture | Pages | Result | Structural behavior | Finding |
| --- | ---: | --- | --- | --- |
| `sample_page_index.md` | 2 | 100% coverage | Nested headings, heading leaves, paragraphs, terms | Expected baseline |
| `sample_page_index.txt` | 2 | 100% coverage | Same semantic shape as Markdown fixture | Expected baseline |
| `manual_short_title_full_table.txt` | 1 | 100% coverage | Table-like content remains one paragraph leaf | Safe, not row-structured |
| `manual_irregular_table_pages.txt` | 2 | 100% coverage | Page boundaries preserved; table-like blocks remain grounded | Safe, not row-structured |
| `01_pipe_table_escaped_pipes.md` | 1 | 100% coverage | One paragraph table block | Escaped pipes are preserved, not decoded into cells |
| `02_header_separator.md` | 1 | 100% coverage | One paragraph table block | Separator semantics are not modeled |
| `03_multiline_table_cells.md` | 1 | 100% coverage | One paragraph table block | Multiline cell boundaries are not modeled |
| `04_heading_adjacent_table.md` | 1 | 100% coverage | Heading container plus paragraph table block | Good containment, no cells |
| `05_fenced_heading_text.md` | 1 | 100% coverage | Fence contents include heading nodes | **Incorrect Markdown interpretation** |
| `06_blockquotes.md` | 1 | 100% coverage | Quote content remains paragraph text | Quote role is not modeled |
| `07_nested_lists.md` | 1 | 100% coverage | List-like lines become term/heading-like nodes | Nested list structure is not faithfully modeled |
| `08_inline_markup.md` | 1 | 100% coverage | Inline markup remains paragraph source text | Safe lossless behavior |
| `09_footnotes.md` | 1 | 100% coverage | Footnote definition can become a heading-like node | Footnote semantics are not modeled |
| `10_html_table.md` | 1 | 100% coverage | HTML table remains one paragraph block | No HTML-table operator |
| `11_unicode.md` | 1 | 100% coverage | Unicode source remains grounded | Passed exact source checks |
| `12_malformed_markdown.md` | 1 | 100% coverage | Malformed content is retained without crashing | Safe fallback, weak structure expected |
| `13_large_table_long_document.md` | 1 | 100% coverage | Large table remains bounded paragraph leaves | No row expansion or scale failure |

## What Passed

- Every fixture parsed without an exception.
- Every fixture reported complete source coverage.
- Every emitted pointer matched the authoritative source text exactly.
- Unicode, malformed syntax, escaped pipes, and long table content did not
  corrupt source offsets.
- Heading-adjacent tables stayed under the heading container.
- The large fixture did not expand into one node per row.

## Issues Found

### 1. Fenced code is not Markdown-aware

`05_fenced_heading_text.md` produced heading nodes for `# This is code` and
`## Nor is this one`. Those lines are code content and should not become
document headings. This is a semantic parsing defect even though grounding is
correct.

### 2. Nested list structure is only approximate

`07_nested_lists.md` produced term/heading-like nodes rather than a faithful
ordered-list and nested-list tree. The current heuristic is safe for source
preservation but should not be advertised as Markdown list parsing.

### 3. Footnote semantics are not represented

`09_footnotes.md` preserves the text but does not connect the reference to its
definition. A later footnote operator would be needed for semantic linking.

### 4. Tables are intentionally coarse

All table variants remain grounded paragraph/table-like leaves. This is safe as
the default, but row-by-row or cell-level parsing requires an explicit table
strategy with its own pointer validation and expansion budget.

### 5. HTML tables are not structurally parsed

`10_html_table.md` is preserved as source text. This is acceptable fallback
behavior, but HTML table semantics are not currently available.

## Coverage Assessment

The fixture set is now broad enough to expose the main safety boundaries, but
the parser’s document-type support should be described as:

```text
source-grounded outline parser with conservative table fallback
```

It should not yet be described as a general Markdown parser or structural
table parser.

The next correctness fixes should be:

1. Make heading detection fence-aware.
2. Decide whether list structure belongs in the deterministic PageIndex
   operator or a separate Markdown operator.
3. Add an optional table operator for row/cell decomposition.
4. Add explicit footnote and HTML-table operators only if downstream retrieval
   needs those relationships.

Those changes should preserve the current invariant: when structure is
uncertain, retain one exact grounded block rather than inventing child spans.
