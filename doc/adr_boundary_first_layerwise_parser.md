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

Boundary-first is a one-layer refinement operator. It does not own recursive
descent below a successfully refined parent; the progressive refinement engine
selects a strategy again for each later child layer. The complete arbitration
and PageIndex-fallback rule is defined in
[ADR: Progressive Per-Layer Refinement And Strategy Arbitration](adr_progressive_refinement_strategy_arbitration.md).

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

## Source-Authoritative Pointer Invariant

The parser has an implicit source-authority invariant that all accepted
grounding pointers must satisfy:

```text
source_map[source_cluster_id].text[start_char:end_char + 1]
    == pointer.verbatim_text
```

The source slice, not provider-generated transcription, is authoritative.

OCR normalization has the same identity boundary: each input page must carry a
positive integer `pdf_page_num`, and a document snapshot cannot contain the
same page number twice. Serialized digit strings may be accepted and normalized
to integers, but booleans, fractional values, non-numeric values, and duplicate
page identities are rejected rather than coerced. This prevents two physical
pages from silently sharing source-map identity or changing identity through
Python's permissive numeric conversions.

OCR response validation is also explicit rather than assertion-based. The
empty-page flag must agree with whether text clusters exist, text/non-text
cluster identities must be globally distinct within a page, and
`meaningful_ordering` must contain every text-cluster identity exactly once.
These checks remain active when Python is optimized with `-O`.

Legacy OCR export and refinement paths follow the same rule: input preconditions,
cluster identity checks, and text-preservation gates use explicit exceptions,
never `assert`. Optimized execution must not disable a source-preservation or
malformed-input check.

At the normalized document boundary, collection identifiers are non-empty and
unique within one request, and embedding-space labels are non-empty and unique
within one collection. Dimension or label collisions must not silently merge
independent projection spaces.

The same fail-closed rule applies at every downstream boundary. A source record
is authoritative only when it is a mapping with a string `text` value. Missing,
malformed, or non-string source records are treated as unavailable; they must
not be converted with `str(...)` into prompt context, pointer coverage, repair
input, or persisted evidence. This prevents diagnostic or provider-shaped data
from becoming a false source of truth.
`start_char` and `end_char` are inclusive pointer offsets at the persisted
model boundary. The full-source sentinel `end_char == -1` is resolved to the
actual final character before persistence or export.

This invariant has two stages:

1. **Repair and hydration.** A valid source cluster and valid offsets identify
   the intended span even if an LLM has retyped the excerpt with changed
   whitespace, Markdown escapes, Unicode punctuation, or entities. The parser
   rehydrates `verbatim_text` from the source slice. Fuzzy or exact matching
   may locate a replacement only when the original offsets are unusable.
2. **Acceptance and persistence.** The repaired pointer is accepted only after
   its offsets and source-derived text agree exactly. An unrehydrated pointer
   with independently supplied text is invalid evidence, even if the text is
   approximately similar.

Fuzzy repair is therefore a **locator**, not an authority. It must never make
   fuzzy text itself the persisted evidence, move a pointer to another source
   cluster, or silently cross a source revision. Ambiguous, missing, malformed,
   or cross-cluster matches fail closed and remain eligible for a later retry.

### Disagreement Decision Table

| Condition | Interpretation | Required action |
| --- | --- | --- |
| Valid offsets, provider text differs | Lossy provider transcription | Keep offsets and rehydrate exact source text |
| Invalid offsets, one exact/fuzzy location in the same cluster | Repairable pointer location | Relocate, rehydrate, and record the repair method |
| Multiple plausible locations | Evidence identity is uncertain | Reject the pointer; retry or request review |
| Source cluster is missing or malformed | Authoritative evidence unavailable | Reject; do not stringify or invent source text |
| Source revision/content hash differs | Persisted pointer is stale | Reject as stale and reparse against the new revision |
| Pointer is valid but the node claim is semantically wrong | Node-level interpretation error | Preserve source evidence and review/rebuild the node |

The current implementation enforces the first four checks relative to the
source map supplied to validation. Source-revision/content-hash binding is not
yet implemented. It is required before this invariant can be considered
globally stable across updates: a cluster ID alone must not allow an old node
to be reinterpreted against replacement text.

This invariant does not require both the original provider text and the source
slice to remain authoritative. Provider text is advisory input; the immutable
source revision and its exact slice are the sole authority for persisted
grounding.

### Complete Pointer And Repair Invariant

The source-authority equation is the visible postcondition. The parser also
relies on the following implicit invariants, which apply before a pointer is
committed:

- The source map used for validation is the authoritative snapshot for the
  current parse attempt. Validation does not silently substitute text from a
  different source cluster or document.
- Pointer offsets are Unicode code-point offsets into the source-cluster text,
  and persisted `start_char`/`end_char` values are inclusive. A full-source
  sentinel is normalized to concrete bounds before persistence.
- A valid offset range identifies the evidence even when the provider's
  `verbatim_text` is lossy, reformatted, escaped differently, or otherwise
  different. Valid offsets therefore take precedence over provider text.
- Repair is allowed to change the locator only when it can identify one
  unambiguous location in the same source cluster. Repair must not cross
  clusters, guess between duplicate occurrences, or use approximate text as
  persisted evidence.
- A repair is successful only when the repaired locator and the rehydrated
  source slice agree exactly. The provider's returned transcription is not a
  second authority that can override the source slice.
- Any pointer that cannot satisfy the equation after hydration or repair is
  rejected, and the containing child remains eligible for retry or review
  rather than being committed with invented or stale evidence.
- Structural review, summaries, and semantic claims may accept or reject a
  proposed node, but they cannot alter the source slice represented by an
  accepted pointer. A semantically wrong node is a review error, not a reason
  to weaken pointer provenance.

In short:

```text
accepted pointer
  = one source-cluster snapshot
  + one valid/unambiguous locator
  + exact source-derived text
  + no unresolved disagreement
```

This is intentionally asymmetric. A provider/source disagreement is evidence
that the provider transcription is unreliable, unless the locator itself is
invalid or the source snapshot has changed. It is not evidence that two text
values should be retained as co-authoritative.

### Implicit Repair Invariant

The following invariant is required even when it is not represented as a
separate field in the persisted model:

```text
repair_success
  -> exactly one authoritative source snapshot
  -> exactly one valid or unambiguous locator in that snapshot
  -> persisted text == source_snapshot[locator]
  -> provider transcription is never persisted as a competing authority
```

Transport adapters apply the same fail-closed rule to external JSON: malformed
edge identifiers, references, and counters are rejected or defaulted safely;
they are never stringified or numerically truncated into apparently valid
parser state.

Fresh layered parse sessions also bind their source-map identity to a
deterministic SHA-256 fingerprint of source-cluster IDs and authoritative text.
Resuming such a session with replacement text is rejected before proposal or
review. Older sessions without this metadata remain readable for compatibility,
but are not protected by this cross-run guard.

Repair has two distinct outcomes that must not be conflated:

- **Locator repair:** the provider supplied a usable hint but its offsets are
  invalid or stale; the parser may relocate the pointer only when one
  same-cluster occurrence is unambiguous, then it must rehydrate the exact
  source slice.
- **Evidence disagreement:** the source slice and provider transcription differ
  while the source offsets are valid; this is not a request to merge or choose
  between two authorities. The source snapshot wins, and the transcription is
  discarded as advisory input.

If the source snapshot itself has changed, the parser must not treat a
successful text match against the replacement as proof that the old pointer is
still valid. Source revision or content-hash binding is therefore a required
future guard for cross-run repair. Until that binding exists, validation is
authoritative only within the source snapshot passed to the current parse
attempt.

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

