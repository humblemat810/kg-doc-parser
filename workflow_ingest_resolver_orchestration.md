# Workflow Ingest Resolver Orchestration

This document shows the implemented workflow resolver orchestration for
`src/workflow_ingest`, then compares it with the earlier design proposal in
[workflow_ingest_layerwise_proposal.md](/c:/Users/chanh/Documents/kg_doc_parser/workflow_ingest_layerwise_proposal.md#L1).

The implementation source of truth is:

- [design.py](/c:/Users/chanh/Documents/kg_doc_parser/src/workflow_ingest/design.py#L1)
- [handlers.py](/c:/Users/chanh/Documents/kg_doc_parser/src/workflow_ingest/handlers.py#L1)

## Final Implemented Workflow Graph

```mermaid
flowchart TD
    A[start]
    B[normalize_input]
    C[build_source_map]
    D[init_parse_session]
    E{check_frontier_remaining}
    F[prepare_layer_frontier]
    G[propose_layer_breakdown]
    H[review_cud_proposal]
    I[apply_cud_update]
    J{check_layer_coverage}
    K{check_layer_satisfaction}
    L[repair_layer_pointers]
    M[dedupe_and_filter_layer]
    N[commit_layer_children]
    O{check_children_expandable}
    P[enqueue_next_layer_frontier]
    Q[finalize_semantic_tree]
    R[validate_tree]
    S[export_graph]
    T[persist_canonical_graph]
    U[end]

    A --> B --> C --> D --> E
    E -- frontier queued --> F
    E -- frontier empty --> Q

    F --> G --> H --> I --> J --> K
    K -- uncovered or unsatisfied and retry budget remains --> G
    K -- covered and satisfied --> L --> M --> N --> O
    K -- retry budget exhausted --> X[layer failure]

    O -- expandable children exist --> P --> E
    O -- no expandable children --> E

    Q --> R
    R -- coverage ok --> S --> T --> U
    R -- coverage below threshold --> Y[validation failure]
    T -- persistence failure --> Z[persist failure]
```

## Resolver-Orchestration View

This version focuses on what each registered resolver step consumes, mutates,
and routes to.

```mermaid
flowchart TD
    A[start]
    B[normalize_input<br/>writes normalized_input]
    C[build_source_map<br/>writes authoritative_source_map<br/>parser_input_dict<br/>parser_source_map]
    D[init_parse_session<br/>writes parse_session<br/>layer_frontier_queue<br/>semantic_tree]
    E[check_frontier_remaining<br/>reads layer_frontier_queue]
    F[prepare_layer_frontier<br/>writes current_layer_context<br/>trimmed layer_frontier_queue<br/>updated parse_session]
    G[propose_layer_breakdown<br/>writes current_layer_result]
    H[review_cud_proposal<br/>writes current_layer_review<br/>current_layer_context.retry_count<br/>parse_session.layer_attempts]
    I[apply_cud_update<br/>writes current_layer_result]
    J[check_layer_coverage<br/>writes current_layer_review.coverage_ok]
    K[check_layer_satisfaction<br/>routes retry or continue]
    L[repair_layer_pointers<br/>writes current_layer_result<br/>corrected_pointer_count]
    M[dedupe_and_filter_layer<br/>writes filtered current_layer_result]
    N[commit_layer_children<br/>writes semantic_tree]
    O[check_children_expandable<br/>routes enqueue or frontier-check]
    P[enqueue_next_layer_frontier<br/>writes layer_frontier_queue<br/>clears current_layer_context/current_layer_result/current_layer_review]
    Q[finalize_semantic_tree<br/>writes semantic_tree]
    R[validate_tree<br/>writes validation_report or workflow_errors]
    S[export_graph<br/>writes export_bundle]
    T[persist_canonical_graph<br/>writes canonical_write_result<br/>updates export_bundle]
    U[end]

    A --> B --> C --> D --> E
    E -->|queue non-empty| F --> G --> H --> I --> J --> K
    K -->|retry same layer| G
    K -->|accept layer| L --> M --> N --> O
    O -->|expandable children| P --> E
    O -->|leaf-only layer| E
    E -->|queue empty| Q --> R --> S --> T --> U
```

## State-Orchestration View

```mermaid
flowchart TD
    A[input]
    B[normalized_input]
    C[authoritative_source_map]
    D[parser_input_dict]
    E[parser_source_map]
    F[parse_session]
    G[layer_frontier_queue]
    H[current_layer_context]
    I[current_layer_result]
    J[current_layer_review]
    K[semantic_tree]
    L[corrected_pointer_count]
    M[validation_report]
    N[export_bundle]
    O[canonical_write_result]
    P[workflow_errors]

    A --> B --> C
    C --> D
    C --> E
    D --> F
    E --> F
    F --> G --> H --> I --> J --> K
    I --> L
    K --> M --> N --> O
    I --> P
    J --> P
    M --> P
```

## Comparison With The Designed Diagram

## What stayed consistent

- The top-level node sequence is still aligned with the proposed resolver plan.
- The two intended loops are both implemented:
  - same-layer retry:
    - `propose_layer_breakdown -> review_cud_proposal -> apply_cud_update -> check_layer_coverage -> check_layer_satisfaction -> propose_layer_breakdown`
  - next-layer loop:
    - `enqueue_next_layer_frontier -> check_frontier_remaining -> prepare_layer_frontier -> ...`
- Post-parse stages remain outside the parser loop:
  - `validate_tree`
  - `export_graph`
  - `persist_canonical_graph`

## What became more explicit in implementation

- The proposal's `frontier remaining?` is implemented as a resolver node:
  - `check_frontier_remaining`
- The proposal's `children expandable?` is implemented as a resolver node:
  - `check_children_expandable`
- The CUD phase is now broken into explicit resolver-visible nodes:
  - `review_cud_proposal`
  - `apply_cud_update`
  - `check_layer_coverage`
- The proposal's repair/finalize layer stage is split into:
  - `repair_layer_pointers`
  - `dedupe_and_filter_layer`
  - `commit_layer_children`

That means the final implementation is slightly more granular than the original
moderate-granularity proposal, while still not as exploded as a full prompt,
normalize, review, edit, and audit subgraph.

## What is still different from the ideal future parser workflow

- `propose_layer_breakdown` is workflow-visible, but the internal parsing engine
  is still dual-path:
  - workflow-layered path when `propose_layer_fn` is supplied
  - legacy-compat path when only `parse_semantic_fn` is supplied
- The workflow now exposes the CUD control loop, but deeper parser internals are
  not yet fully extracted into smaller parser-core components.

## Important implementation detail not obvious in the old proposal

- Kogwistar routing in this workflow depends on outgoing edge labels matching
  the `_route_next` value.
- In practice that means workflow edge labels are the target step names, not a
  generic relationship label.
- That detail is encoded in [design.py](/c:/Users/chanh/Documents/kg_doc_parser/src/workflow_ingest/design.py#L38).

## Practical reading of the current system

The workflow now does this at orchestration level:

- queue frontier
- select current layer
- propose layer children
- review the proposal
- apply CUD updates
- check layer coverage
- retry if still uncovered or unsatisfied
- repair, dedupe, and commit accepted layer output
- enqueue the next layer
- finalize the tree

The next architectural step, if we continue, is not more workflow wiring first.
It is extracting more of the old parser internals so `propose_layer_breakdown`
and `review_cud_proposal` become thinner wrappers over a true parser-core
implementation.
