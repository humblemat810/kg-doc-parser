# Boundary-First Layerwise Parser Checklist

## Summary

Implement a boundary-first proposal mode for workflow-layered parsing.

The workflow should continue to consume `CurrentLayerResult`, but the proposal
backend should be able to derive that result from validated cutpoints:

1. propose semantic cutpoints
2. review and snap cutpoints to legal text boundaries
3. deterministically assemble child spans
4. summarize accepted spans for the next recursive layer
5. refine only ambiguous regions when needed

## Current Status

- Implemented: closed boundary models, strict-schema-first boundary proposal/review flow, deterministic cutpoint review/assembly, and boundary-mode event logging.
- Implemented: boundary-mode callback coverage, a local ambiguous-cutpoint refinement loop, recursive boundary summaries stored on semantic nodes, stricter legality checks, proposal-mode reporting, and a widened workflow regression that runs boundary-first end to end on the in_memory backend.
- Still open: broader operator-facing A/B reporting beyond the focused fixture harness and the deliberate default-promotion gate.

## Goals

- [x] Keep `propose_layer_fn` as the workflow extension point.
- [x] Add boundary-first mode without changing runtime handler wiring.
- [x] Preserve existing direct child-proposal mode during migration.
- [x] Make boundary proposals strict-schema-first.
- [x] Ensure cutpoints are unique, sorted, and parent-scoped.
- [x] Allow partial acceptance when some proposed cutpoints are ambiguous.
- [x] Keep source spans authoritative and summaries advisory.
- [x] Treat malformed source records and non-string source text as unavailable;
  never stringify them into prompts, pointer validation, coverage, or persisted
  evidence.
- [x] Require OCR page identities to be positive, unambiguous integers and
  reject duplicate page numbers during normalization.
- [x] Enforce OCR empty-page, cross-modality identity, and complete ordering
  invariants with explicit validation rather than `assert`.
- [x] Keep legacy OCR export and source-preservation gates active under
  optimized Python; reject malformed input with explicit exceptions.
- [x] Reject duplicate collection identities within one request and duplicate
  or blank embedding-space labels within one collection.
- [x] Reject non-finite, negative, or inverted normalized bounding-box
  coordinates at the model boundary, not only in the OCR adapter.
- [x] Reject blank source-unit embedding-space labels and whitespace-only
  non-text source URIs.
- [x] Validate diagnostic intervals and review counters with strict integer
  bounds before they enter workflow state or exported reports.
- [x] Treat human-readable OCR progress metadata as advisory; malformed JSON
  or non-SHA-256 page hashes must not override SQLite/artifact evidence.
- [x] Mark OCR documents complete only when the completed page-number set is
  exactly the expected `1..total_pages` set, not merely when row counts match.
- [x] Use the same reserved-cluster allocator when producing parser input and
  authoritative source-map identities; explicit IDs must not shift implicit
  IDs during round-trip conversion.
- [x] Persist the allocated cluster identity in each authoritative source
  record, including when the input omitted a cluster number.
- [x] Bind fresh layered parse sessions to a deterministic source-map
  fingerprint and reject replacement source text on resume; keep legacy
  sessions readable without claiming the new guard.

## Non-Goals

- [ ] Do not replace the workflow runtime state machine.
- [ ] Do not make summaries authoritative source text.
- [ ] Do not ask the model to emit final recursive trees.
- [ ] Do not require provider-specific cache behavior for correctness.

## Slice 1: Boundary Data Models

**Goal:** introduce strict LLM-facing and runtime-facing boundary models.

- [x] Add `LLMBoundaryProposal`.
- [x] Add `LLMBoundaryProposalBatch`.
- [x] Add `BoundaryCutpoint`.
- [x] Add `BoundaryReviewDecision`.
- [x] Add `BoundaryReviewBatch`.
- [x] Use `ConfigDict(extra="forbid")` on LLM-facing models.
- [x] Avoid `Any` in LLM-facing response models.
- [x] Use half-open offsets internally: `[start, end)`.
- [x] Add model round-trip tests.

Suggested minimal fields:

```python
parent_node_id: str
source_cluster_id: str
cut_offset: int
boundary_kind: Literal["section", "paragraph", "list_item", "sentence", "word", "semantic"]
confidence: float
reason: str
```

## Slice 2: Deterministic Boundary Primitives

**Goal:** build local helpers before adding LLM behavior.

- [x] Extract candidate legal cutpoints from source text.
- [x] Classify cutpoints as section, paragraph, list item, sentence, or word.
- [x] Implement uniqueness by `(parent_node_id, source_cluster_id, cut_offset)`.
- [x] Reject offsets outside the parent source span.
- [x] Reject duplicate cutpoints.
- [x] Reject unsorted proposals.
- [x] Add deterministic tests for word, sentence, paragraph, list, and heading boundaries.

## Slice 3: Cutpoint Reviewer

**Goal:** validate and repair proposed cutpoints before assembly.

- [x] Implement deterministic review first.
- [x] Reject cuts inside words.
- [x] Reject cuts inside numbers, citations, URLs, code tokens, or identifiers.
- [x] Reject cuts through list or heading markers.
- [x] Snap near-boundary cuts left or right when the adjustment is small.
- [x] Mark uncertain cuts as `needs_refinement`.
- [x] Emit review diagnostics with before and after offsets.
- [x] Add tests for `accept`, `shift_left`, `shift_right`, `reject`, and `needs_refinement`.

## Slice 4: Boundary Proposal Strategy

**Goal:** add a boundary proposal backend behind `propose_layer_fn`.

- [x] Add `proposal_mode="children" | "boundaries"` to `build_layerwise_llm_callbacks(...)`.
- [x] Keep `children` as the compatibility default until boundary mode is proven.
- [x] Add `_propose_boundaries_fn(...)`.
- [x] Prompt the model to propose only cutpoints, not child text.
- [x] Use `build_structured_output_runnable(...)` with strict `json_schema` first.
- [x] Record `proposal_mode="boundaries"` in event logs.
- [x] Fall back to existing deterministic layer result if the boundary path cannot produce a valid layer.

## Slice 5: Deterministic Assembly

**Goal:** convert accepted cutpoints into `CurrentLayerResult`.

- [x] Add `_assemble_layer_result_from_boundaries(...)`.
- [x] Use adjacent accepted cutpoints to derive child spans.
- [x] Convert half-open spans to inclusive `HydratedTextPointer` only at the final pointer step.
- [x] Derive child titles from headings or first sentence snippets.
- [x] Preserve reading order.
- [x] Avoid empty child spans.
- [x] Preserve parent meaning collectively by covering the parent span or explicitly marking gaps.
- [x] Add tests proving assembled children have exact source pointers.
- [x] Document the implicit repair invariant: successful repair must resolve
  one same-cluster location and persist only the exact source-derived slice.
- [x] Treat provider/source disagreement as advisory transcription loss, not
  as two co-authoritative evidence values.
- [x] Keep locator repair and evidence disagreement as separate outcomes.
- [x] Reject malformed transport identifiers and counters instead of coercing
  them into valid-looking parser state.
- [x] Bind persisted pointers to a source revision or content hash before
  allowing cross-run repair against replacement source text for fresh layered
  sessions; legacy sessions without a fingerprint remain compatibility-only.

## Slice 6: Partial Acceptance And Refinement

**Goal:** avoid rejecting a whole layer when only a few boundaries are weak.

- [x] Accept valid cutpoints.
- [x] Isolate ambiguous intervals.
- [x] Add a local refinement prompt for ambiguous intervals only.
- [x] Let refinement choose from nearby legal cutpoints or return no split.
- [x] Keep unresolved regions as coarse child spans after retries.
- [x] Record unresolved intervals in diagnostics.
- [x] Add tests where 18 of 20 cutpoints pass and 2 are refined or retained as coarse spans.

## Slice 7: Recursive Unit Summaries

**Goal:** create summary hints for later recursive layers without losing source truth.

- [x] Add a summary payload for accepted boundary units.
- [x] Store exact source pointers alongside summaries.
- [x] Mark summary text as advisory metadata.
- [x] Feed boundary unit summaries into later proposal prompts.
- [x] Ensure final graph provenance still points to source spans.
- [x] Add tests proving summaries do not replace source pointers.

## Slice 8: Diagnostics And Evaluation

**Goal:** make boundary-first behavior inspectable in normal artifacts.

- [x] Log proposed cutpoint count.
- [x] Log accepted, shifted, rejected, and refined counts.
- [x] Log unresolved interval count.
- [x] Add layer-log entries for boundary review decisions.
- [x] Add cost/token summaries by proposal mode.
- [x] Add A/B reporting against direct child proposal:
  accepted proposal rate, fallback rate, review satisfaction, overlap rate,
  coverage gaps, and average children per parent.

## Slice 9: Migration

**Goal:** introduce boundary-first mode without disrupting current parser runs.

- [x] Keep direct child proposal as the default.
- [x] Add an env or config switch for boundary mode.
- [x] Add a VS Code/debug launch option only after tests are green.
- [x] Run both modes against the same fixture set.
- [x] Compare final semantic trees and validation diagnostics.
- [ ] Promote boundary mode to default only after it beats child proposal on fallback rate and manual inspection.

## Focused Test Commands

```powershell
.\.venv\Scripts\python.exe -m pytest kg-doc-parser\tests\test_workflow_ingest_layerwise_llm.py -q -p no:cacheprovider
```

```powershell
.\.venv\Scripts\python.exe -m pytest kg-doc-parser\tests\test_workflow_ingest_layerwise_parser.py -q -p no:cacheprovider
```

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_longrun_parser_worker_callbacks.py -q -p no:cacheprovider
```

## Implementation Notes

- Boundary-first should be a proposal strategy, not a new runtime path.
- The model proposes where to split; local code owns exact excerpts.
- The reviewer is part of the parser contract, not just a debugging tool.
- Partial acceptance is preferred over whole-layer rejection.
- Summaries support recursion, but source spans remain authoritative.
