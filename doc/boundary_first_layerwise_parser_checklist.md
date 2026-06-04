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

## Goals

- [ ] Keep `propose_layer_fn` as the workflow extension point.
- [ ] Add boundary-first mode without changing runtime handler wiring.
- [ ] Preserve existing direct child-proposal mode during migration.
- [ ] Make boundary proposals strict-schema-first.
- [ ] Ensure cutpoints are unique, sorted, and parent-scoped.
- [ ] Allow partial acceptance when some proposed cutpoints are ambiguous.
- [ ] Keep source spans authoritative and summaries advisory.

## Non-Goals

- [ ] Do not replace the workflow runtime state machine.
- [ ] Do not make summaries authoritative source text.
- [ ] Do not ask the model to emit final recursive trees.
- [ ] Do not require provider-specific cache behavior for correctness.

## Slice 1: Boundary Data Models

**Goal:** introduce strict LLM-facing and runtime-facing boundary models.

- [ ] Add `LLMBoundaryProposal`.
- [ ] Add `LLMBoundaryProposalBatch`.
- [ ] Add `BoundaryCutpoint`.
- [ ] Add `BoundaryReviewDecision`.
- [ ] Add `BoundaryReviewBatch`.
- [ ] Use `ConfigDict(extra="forbid")` on LLM-facing models.
- [ ] Avoid `Any` in LLM-facing response models.
- [ ] Use half-open offsets internally: `[start, end)`.
- [ ] Add model round-trip tests.

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

- [ ] Extract candidate legal cutpoints from source text.
- [ ] Classify cutpoints as section, paragraph, list item, sentence, or word.
- [ ] Implement uniqueness by `(parent_node_id, source_cluster_id, cut_offset)`.
- [ ] Reject offsets outside the parent source span.
- [ ] Reject duplicate cutpoints.
- [ ] Reject unsorted proposals.
- [ ] Add deterministic tests for word, sentence, paragraph, list, and heading boundaries.

## Slice 3: Cutpoint Reviewer

**Goal:** validate and repair proposed cutpoints before assembly.

- [ ] Implement deterministic review first.
- [ ] Reject cuts inside words.
- [ ] Reject cuts inside numbers, citations, URLs, code tokens, or identifiers.
- [ ] Reject cuts through list or heading markers.
- [ ] Snap near-boundary cuts left or right when the adjustment is small.
- [ ] Mark uncertain cuts as `needs_refinement`.
- [ ] Emit review diagnostics with before and after offsets.
- [ ] Add tests for `accept`, `shift_left`, `shift_right`, `reject`, and `needs_refinement`.

## Slice 4: Boundary Proposal Strategy

**Goal:** add a boundary proposal backend behind `propose_layer_fn`.

- [ ] Add `proposal_mode="children" | "boundaries"` to `build_layerwise_llm_callbacks(...)`.
- [ ] Keep `children` as the compatibility default until boundary mode is proven.
- [ ] Add `_propose_boundaries_fn(...)`.
- [ ] Prompt the model to propose only cutpoints, not child text.
- [ ] Use `build_structured_output_runnable(...)` with strict `json_schema` first.
- [ ] Record `proposal_mode="boundaries"` in event logs.
- [ ] Fall back to existing deterministic layer result if the boundary path cannot produce a valid layer.

## Slice 5: Deterministic Assembly

**Goal:** convert accepted cutpoints into `CurrentLayerResult`.

- [ ] Add `_assemble_layer_result_from_boundaries(...)`.
- [ ] Use adjacent accepted cutpoints to derive child spans.
- [ ] Convert half-open spans to inclusive `HydratedTextPointer` only at the final pointer step.
- [ ] Derive child titles from headings or first sentence snippets.
- [ ] Preserve reading order.
- [ ] Avoid empty child spans.
- [ ] Preserve parent meaning collectively by covering the parent span or explicitly marking gaps.
- [ ] Add tests proving assembled children have exact source pointers.

## Slice 6: Partial Acceptance And Refinement

**Goal:** avoid rejecting a whole layer when only a few boundaries are weak.

- [ ] Accept valid cutpoints.
- [ ] Isolate ambiguous intervals.
- [ ] Add a local refinement prompt for ambiguous intervals only.
- [ ] Let refinement choose from nearby legal cutpoints or return no split.
- [ ] Keep unresolved regions as coarse child spans after retries.
- [ ] Record unresolved intervals in diagnostics.
- [ ] Add tests where 18 of 20 cutpoints pass and 2 are refined or retained as coarse spans.

## Slice 7: Recursive Unit Summaries

**Goal:** create summary hints for later recursive layers without losing source truth.

- [ ] Add a summary payload for accepted boundary units.
- [ ] Store exact source pointers alongside summaries.
- [ ] Mark summary text as advisory metadata.
- [ ] Feed boundary unit summaries into later proposal prompts.
- [ ] Ensure final graph provenance still points to source spans.
- [ ] Add tests proving summaries do not replace source pointers.

## Slice 8: Diagnostics And Evaluation

**Goal:** make boundary-first behavior inspectable in normal artifacts.

- [ ] Log proposed cutpoint count.
- [ ] Log accepted, shifted, rejected, and refined counts.
- [ ] Log unresolved interval count.
- [ ] Add layer-log entries for boundary review decisions.
- [ ] Add cost/token summaries by proposal mode.
- [ ] Add A/B reporting against direct child proposal:
  accepted proposal rate, fallback rate, review satisfaction, overlap rate,
  coverage gaps, and average children per parent.

## Slice 9: Migration

**Goal:** introduce boundary-first mode without disrupting current parser runs.

- [ ] Keep direct child proposal as the default.
- [ ] Add an env or config switch for boundary mode.
- [ ] Add a VS Code/debug launch option only after tests are green.
- [ ] Run both modes against the same fixture set.
- [ ] Compare final semantic trees and validation diagnostics.
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

