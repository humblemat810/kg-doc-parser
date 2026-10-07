# ADR: Progressive Per-Layer Refinement And Strategy Arbitration

## Status

Accepted; locally verified on the feature branch; remote CI pending

## Context

The ingestion parser already has several useful structural techniques:

- direct LLM child/excerpt proposals;
- boundary-first LLM proposals with deterministic cutpoint assembly;
- deterministic PageIndex-style structural extraction;
- deterministic helpers for tables and other future special structures.

Treating any of these as a recursive document parser creates an accidental
ownership boundary. In particular, selecting PageIndex after one difficult
layer must not make PageIndex the parser for every descendant in that subtree.
That would prevent better-suited strategies from being considered at later
layers and would make a transient fallback become a permanent policy choice.

## Decision

The parser is a **progressive refinement engine**. It owns recursion and calls
strategy operators only to refine one grounded parent interval by one level.

```text
for each expandable parent in the current frontier:
    choose a strategy for this parent layer
    attempt one-layer refinement
    validate source coverage and child invariants
    commit the valid children, or retain the unchanged parent

children enter a later frontier:
    select a strategy independently again
```

The scoped operation is:

```text
refine(parent, strategy) -> RefinementResult | no useful refinement
```

It is not:

```text
choose(strategy) -> strategy recursively owns parent subtree
```

### Strategy Operators

The first operators are:

- `layer_excerpt`: direct children with exact, verbatim source evidence;
- `layer_boundary`: model-suggested cutpoints, locally reviewed and assembled;
- `page_index`: deterministic PageIndex-style structural decomposition;
- future operators such as `table_structural` or format-specific refiners.

Every operator receives one parent interval (or a bounded same-depth batch of
independent parent intervals) and returns only its direct child candidates.
The normal workflow performs review, coverage validation, commit, and frontier
enqueueing after every operator. No operator can bypass those steps.

### PageIndex Scope

`page_index` is a one-layer refinement operator. Given one source-owning
parent span, it may derive useful immediate children such as a heading/body
split, paragraph units, table units, or a next-level section decomposition.

It must not recursively materialize the remaining subtree. If PageIndex is
used for parent `P` at depth `d`, its children return to the normal frontier.
At depth `d + 1`, each child starts selection from the configured default or
receives a fresh triage decision. A PageIndex fallback therefore never locks a
subtree into PageIndex.

When a PageIndex structural container is used as a parent, its aggregate source
span is the parent interval for the next refinement. Its title-text leaf and
ordinary content leaves retain their exact ownership spans.

### Arbitration Modes

Strategy selection is configurable per parse invocation and is evaluated for
every frontier layer.

1. `auto` with configured provider triage: request a bounded structured
   assessment for the current parent context. The provider may select an
   allowed operator only.
2. `auto` without an available triage callable: use the deterministic cascade.
3. Explicit `layer_excerpt`, `layer_boundary`, or `page_index`: begin that
   current layer with the requested operator and retain the remaining methods
   as fallbacks. An explicit `parse_strategy_order` may instead provide a
   complete permutation such as `layer_boundary,layer_excerpt,page_index`.
   With triage disabled, the final entry in the configured permutation is the
   deterministic last fallback and exhaustion fails the layer. With triage
   enabled, triage may select any currently enabled operator, including
   `page_index`; an unusable result disables that operator and returns to
   routing through the remaining enabled methods.

The default deterministic order is:

```text
layer_excerpt -> layer_boundary -> page_index
```

The order is stored in the layer decision and is validated as a permutation;
duplicates or omissions are rejected before execution. After a method fails
the satisfaction checks, the runtime records that method in the layer's
`disabled_strategies` set. Routing returns to `triage_parse_strategy`, which
cannot select a disabled method. When all methods are disabled, the workflow
takes the explicit `all_strategies_exhausted` edge to `parse_failure`.

An LLM triage choice affects only the current layer. A low-confidence,
malformed, timed-out, or unavailable triage response falls back to the same
deterministic policy; it cannot stop parsing or add an unapproved strategy.

The triage contract requires one bounded advantage and one bounded limitation
for each of `layer_excerpt`, `layer_boundary`, and `page_index`. The accepted
assessment is retained in the workflow's backend diagnostics alongside the
selected strategy and fallback reason. It is not copied into the document
content or sent back to a later model call as an authority signal.

Each provider-backed triage call uses the parser provider timeout. Timeout or
structured-output failure is recorded as a bounded fallback diagnostic rather
than retried indefinitely.

### Configuration Surface

Provider settings establish the process or service defaults through
`KG_DOC_PARSER_PARSE_STRATEGY` and `KG_DOC_PARSER_TRIAGE_ENABLED`. A single
parse request may override either default on `WorkflowIngestInput`:

```python
WorkflowIngestInput(
    ...,
    parse_strategy="page_index",  # or "auto", "layer_excerpt", "layer_boundary"
    triage_enabled=False,
)
```

`None` means "use the provider default". These controls are workflow metadata,
not document content, and are excluded from LLM-facing model payloads. An
explicit request override is applied at every frontier layer of that parse;
children still remain independently attributable to their parent layer.

PageIndex summaries are enabled by default. The process default can be changed
with `KG_DOC_PARSER_PAGE_INDEX_SUMMARY_ENABLED`, and one request can override it
with `WorkflowIngestInput.page_index_summary_enabled`. Disabling summaries
changes only the advisory summary field; source spans, leaf ownership, and
structural validation remain unchanged.

An optional parent-summary context pass is disabled by default with
`KG_DOC_PARSER_PAGE_INDEX_HIERARCHICAL_SUMMARY_ENABLED`. When enabled for a
provider-backed PageIndex parse, summaries are generated in tree-depth order.
Each call receives bounded block excerpts and only the already accepted direct
parent summary. A failed summary call retains the initial assignment summary,
records provider diagnostics, and never changes source grounding or tree
structure. The request-level override is
`WorkflowIngestInput.page_index_hierarchical_summary_enabled`.

The CLI exposes the same per-call controls for `ocr`, `page-index`, and `demo`:

```text
--parse-strategy auto|layer_excerpt|layer_boundary|page_index
--triage-enabled
--no-triage-enabled
--page-index-summary-enabled
--no-page-index-summary-enabled
--page-index-hierarchical-summary-enabled
--no-page-index-hierarchical-summary-enabled
```

The flags override environment/provider defaults only for that invocation.

### Layer Semantics

The workflow may process a bounded batch of parents at the same depth for
efficiency, but each candidate remains attributable to exactly one parent.
Coverage and non-overlap are validated against that parent interval. A failed
refinement leaves the parent unchanged and permits the next configured fallback
for that same layer.

A successful refinement must satisfy these invariants for each parent:

```text
children belong to that parent
children have valid, exact source pointers
children do not duplicate or ambiguously overlap
union(child ownership spans) covers the meaningful parent interval
```

Structural containers may use aggregate spans for refinement context; leaf
ownership spans remain the authoritative source evidence. Generated summaries
are advisory and never replace source text or pointers.

## Workflow Design

```text
prepare_layer_frontier
        |
        v
triage_parse_strategy  (per layer, bounded context and validated order)
        |
        +----------------------+----------------------+-------------------+
        |                      |                      |
        v                      v                      v
 layer_excerpt          layer_boundary          page_index
        |                      |                      |
        +----------------------+----------------------+
                               v
                     review and validate
                       |              |
                 success       failure: disable method
                       |              |
                       v              v
                 commit children   triage again
                                      |
                                      v
                           all disabled -> parse_failure
                     |
                     v
          enqueue expandable children
                     |
                     v
           prepare next frontier layer
                     |
                     v
          triage again from normal policy
```

For a preferred LLM operator that fails validation after bounded retries:

```text
preferred strategy -> alternate LLM strategy -> page-index -> retain/fail by policy
```

The fallback route is local to the current parent layer. An explicit
`page_index` request starts with PageIndex while retaining the configured
remaining methods as fallbacks. When triage is disabled, the final entry in
the configured permutation is the last deterministic fallback and exhaustion
fails the layer. When triage is enabled, triage may choose any still-enabled
operator after a failure. The selected route never changes the default
preference for descendants.

The transitions above are ordinary Kogwistar workflow nodes and edges. The
strategy branches use named predicates (`parse_strategy_layer_excerpt`,
`parse_strategy_layer_boundary`, `parse_strategy_page_index`), while
`strategy_failed_with_remaining`, `all_strategies_exhausted`, and
`layer_satisfied` guard the review result. Handler return values do not bypass
these guards.

The runtime state keeps two separate audit surfaces. `strategy_attempt_counts`
is a replace-style (`u`) map keyed by strategy, while
`strategy_execution_history` is an append-style (`a`) list of typed events.
Each selected strategy records its layer, parent IDs, and attempt number;
completion records add `succeeded` or `failed` with bounded reasons. This is
workflow execution state, not document content and not an LLM reasoning trace.

## Consequences

Benefits:

- difficult layers can use a deterministic structural fallback without losing
  later semantic precision;
- boundary, excerpt, PageIndex, and future table operators share one durable
  validation and commit path;
- provider triage improves selection but never becomes the authority for
  source grounding or workflow control;
- timeouts cannot create an unbounded local-model retry loop;
- diagnostics can explain the selected strategy, fallback reason, and result
  for each parent layer.

Costs:

- operators must implement a common one-layer refinement contract;
- workflow state must retain per-layer rather than subtree-wide strategy
  diagnostics;
- PageIndex integration needs a layer adapter instead of replacing the whole
  semantic tree.

## Non-Goals

- Reproduce PageIndex as a recursive end-to-end parsing mode in the layered
  workflow.
- Let an LLM create source pointers, coverage exceptions, or new strategy
  identifiers without host validation.
- Persist generated summaries as canonical source evidence.
- Require every document or every layer to be recursively expanded to a fixed
  depth.

## Acceptance Criteria

- Strategy selection occurs after a frontier layer is prepared, not once per
  document.
- The CLI and API expose the same strategy and triage overrides for one parse
  invocation.
- PageIndex returns only direct children for the selected parent layer.
- A PageIndex-successful parent has children that re-enter normal strategy
  selection independently.
- A preferred strategy can fall through deterministically to the other LLM
  strategy and then PageIndex for the same parent layer.
- A fake layer payload can force each operator to pass or fail, proving that a
  failed operator is disabled, fallback returns to routing, and exhaustion
  reaches `parse_failure` rather than retrying forever.
- Explicit strategy configuration and provider triage are bounded by the
  allowed strategy set and provider timeout.
- Triage diagnostics contain bounded pros/cons for all three allowed
  strategies without exposing workflow controls as model-facing content.
- Every operator result follows existing review, pointer repair, coverage,
  commit, and frontier-enqueue steps.
- Tests cover parent-local fallback, child-layer priority reset, triage failure
  fallback, exact span coverage, and no recursive PageIndex subtree ownership.

## Related ADRs

- [Boundary-first workflow-layered parsing](adr_boundary_first_layerwise_parser.md)
- [Bounded layered parse contract](adr_bounded_layered_parse_contract.md)
