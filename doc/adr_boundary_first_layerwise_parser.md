# ADR: Boundary-First Workflow-Layered Parsing

## Status

Proposed

## Context

The current workflow-layered parser asks the LLM-backed `propose_layer_fn` to
return child nodes with source pointers. This is expressive, but it puts several
jobs into one model call:

- choose semantic child units
- choose exact source spans
- avoid duplicate or overlapping excerpts
- keep the result valid for the current layer
- preserve parent meaning without inventing text

That creates a fragile validation loop. A response can be semantically close
while still failing because a pointer is not exact, a span cuts through a word,
or an excerpt is recombined from non-contiguous text.

For parsing, exact provenance is a contract. The parser should prefer a narrow
LLM task whose output can be validated and assembled deterministically.

## Decision

Introduce a boundary-first proposal mode behind the existing
`propose_layer_fn` callback contract.

The workflow runtime will keep consuming `CurrentLayerResult`. Boundary-first
mode will be an internal proposal strategy:

1. The model proposes ordered cutpoints for each current parent source span.
2. A cutpoint reviewer validates and, when safe, snaps proposed cutpoints to
   legal textual boundaries.
3. A deterministic assembler converts accepted adjacent cutpoints into child
   spans and then into `CurrentLayerResult`.
4. Ambiguous cutpoints are isolated for refinement rather than forcing the
   whole layer to fail.

The public workflow contract remains stable. The difference is that child nodes
are derived from accepted boundaries instead of directly authored by the model.

## Design Principles

- Boundaries must be unique within a parent source span.
- Boundary identity should be canonical, such as
  `(parent_node_id, source_cluster_id, cut_offset)`.
- Internally, cutpoints should use half-open offsets `[start, end)` and convert
  to inclusive `HydratedTextPointer` spans only at assembly time.
- The LLM-facing schema should use provider-controlled `json_schema` first, as
  described in the root `doc/llm_structured_output_policy.md`.
- The model should propose semantic breakpoints, not final child text.
- The deterministic assembler owns exact excerpts and final child pointers.
- Summaries generated after assembly are hints for later recursive layers, not
  authoritative source.

## Cutpoint Reviewer

Boundary-first mode needs a reviewer because a semantically plausible cut can
still be textually invalid.

The reviewer should reject or repair cutpoints that:

- split a word
- split a number, citation, URL, code token, or identifier
- cut through a sentence in an unreasonable place
- split list markers or heading markers
- fall outside the parent source span
- duplicate another accepted cutpoint

The reviewer should prefer cutpoints at:

- section or heading boundaries
- paragraph boundaries
- list item boundaries
- sentence boundaries
- word boundaries, only when finer cuts are intentional

Reviewer outcomes should be discrete:

- `accept`
- `shift_left`
- `shift_right`
- `reject`
- `needs_refinement`

## Partial Acceptance

Boundary-first parsing should allow monotonic progress.

If a proposal contains 20 cutpoints and 18 are valid, the parser should accept
the 18 valid cutpoints, isolate the ambiguous regions, and refine only those
regions. If refinement still cannot produce a valid cut, the region should stay
as a coarse child for a later recursive layer.

This avoids rejecting an entire layer because a small number of boundaries are
uncertain.

## Recursive Unit Model

After accepted boundaries are assembled into child spans, the parser may
summarize each span into a recursive unit:

- source span and exact pointers
- generated summary
- structural hints
- confidence and unresolved flags
- validation notes

Later layers may parse over these units and summaries, but final graph
provenance must continue to point back to exact source spans.

## Consequences

Benefits:

- smaller LLM task
- stricter source grounding
- easier validation
- better partial-progress behavior
- cleaner fit for strict structured output
- less risk of invented or recombined excerpts

Costs:

- new boundary proposal schema
- new cutpoint reviewer
- deterministic assembly logic
- additional diagnostics for ambiguous boundaries
- a refinement path for difficult regions

## Alternatives Considered

Continue direct child proposal:

This preserves the current implementation shape, but leaves the model
responsible for both semantic grouping and exact source grounding.

Use only deterministic boundaries:

This is reliable for well-structured markdown or page-index sources, but too
weak for documents whose semantic boundaries require interpretation.

Ask the LLM to output final recursive trees:

This was rejected for the same reason the page-index parser moved to flat block
assignment: recursive tree generation is too large and too easy to validate
poorly.

## Acceptance Criteria

- Boundary-first mode plugs into the existing `propose_layer_fn` callback.
- Runtime handlers and workflow state transitions do not need a new state
  machine.
- Boundary proposal uses strict LLM-facing schemas.
- Accepted boundaries are unique and ordered within each parent span.
- Deterministic assembly creates exact child pointers.
- Ambiguous boundaries can be refined without rejecting the whole layer.
- The existing child-proposal mode remains available during migration.

