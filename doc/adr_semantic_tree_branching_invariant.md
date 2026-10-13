# ADR: Semantic Tree Branching Invariant

## Status

Accepted for the structural-invariant hardening release.

## Decision

Every semantic node has either zero children or at least two children. A
single-child refinement is not a meaningful decomposition and must never enter
the committed tree or the expansion frontier.

An unsplittable parent is retained as an explicit atomic node. The parser may
not invent a sibling merely to satisfy the shape rule. Atomic retention is
valid only when the parent has verified authoritative source pointers.

The source span remains authoritative. Provider text is a candidate
transcription and is rehydrated from the source map before pointer validation.

## Why

The workflow-core refactor allowed a one-child result to pass coverage checks,
be attached to the tree, and be scheduled again. Repeating that path produced
deep unary trees and could report structurally complete output without a
meaningful split.

## Validation Boundary Matrix

| Boundary | Required action |
| --- | --- |
| Provider proposal | Decode typed or mapping payload, then detect unary, unchanged, duplicate, overlap, and coverage issues. |
| Review and repair | Rehydrate pointers from the authoritative source map and run deterministic checks again after repair. |
| Commit | Validate immediately before mutation and again after mutation. Reject unary children and invalid atomic retention. |
| Frontier | Never enqueue a unary, unchanged, or rejected refinement. |
| Layered DTO adapter | Validate request/source grounding at the core expansion boundary; validate result DTO identity, shape, and depth at construction, use the same commit gate, and requeue failed parents instead of dropping them. |
| PageIndex | Normalize heading/page wrappers deterministically, rebase descendant parent IDs and levels when promoting a wrapper, retain unresolved branches atomically, then run final validation. |
| Legacy import/export | Validate before conversion and serialization, including IDs and parent ancestry. Persisted unary trees fail closed and are not silently rewritten. |
| Final result | Run the iterative identity, parent, pointer, cycle, duplicate-ID, depth, and branching checks. |

## Wrapper Normalization

Known structural wrappers may be removed when doing so preserves source
ownership and produces a meaningful branch. Heading-only blocks become grounded
heading leaves. A page or root wrapper with no meaningful decomposition becomes
an atomic evidence-bearing node. Wrapper promotion rebases every promoted
descendant, not just its direct children. No arbitrary semantic node is
flattened just because it has one child.

## Diagnostics

- `unary_refinement`: one parent received exactly one child.
- `unchanged_refinement`: the only child has the parent's source span.
- `nested_unary_refinement`: a candidate already contains a one-child nested
  refinement.
- `atomic_retained`: no valid decomposition was committed and the verified
  parent was retained as a leaf.

Diagnostics are carried in review metadata and parser reports. They do not
change Kogwistar graph invariants or rewrite existing persisted graph data.

## Model-validation and retry handoff

Provider-facing result models remain strict Pydantic DTOs. Required fields,
strict scalar types, forbidden extra fields, and interval validators run when
the DTO is constructed. Those checks are intentionally before structural tree
mutation. A malformed proposal or review payload is converted into a typed
`LayerValidationDiagnostic` attached to the layer result and review packet,
with its phase, bounded error type/message, and field location. The workflow
then marks the persisted result unsatisfied and routes it through the existing
retry policy rather than treating a construction exception as a successful
or unstructured runtime failure.

The contextual layer shape is not enforced by a Pydantic model validator on an
individual result: whether one child is unary depends on the parent context,
and atomic retention requires verified source pointers, including any explicit
atomic pointer override. The intrinsic materialized-child shape is enforced
immediately by `LayerChildCandidate`, which rejects exactly one nested child.
The shared structural validator therefore remains the authority at proposal,
review/repair, commit, frontier, and finalization boundaries. Model diagnostics
complement that validator; they do not replace it.

Layer batches are evaluated per parent for partial commit. A diagnostic with a
parent or child location blocks only the affected parent, so an independently
verified sibling can be committed while the failed parent is requeued. A
location-free diagnostic is global and blocks the complete batch. Child IDs
shared across parent groups are converted into parent-scoped commit diagnostics
before mutation; they cannot be allowed to overwrite or alias one another.

The finalized-tree validator is iterative and checks identity, ancestry,
cycles, duplicate IDs, source-pointer ownership when a source map is
available, configured depth, and the zero-or-two-plus branching rule. Layered
request models and core expansion validate source grounding at the boundary;
seed and result DTOs validate identity, shape, and depth at construction
because those DTOs do not carry the source map. The mutation primitive runs
the full validator both before and after attachment. This keeps retryable
model errors separate from post-mutation integrity failures.

## Consequences

- A valid flat document may be represented by one atomic root rather than a
  root/page/content wrapper chain.
- Existing historical unary graph payloads are rejected at legacy conversion
  and export boundaries. An explicit audit/repair operation is required before
  they can be exported again.
- Valid deep branching trees remain supported; validation is iterative so deep
  input cannot exhaust Python recursion merely while checking shape.
