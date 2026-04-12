
Mermaid overview: text ingress
------------------------------

```mermaid
flowchart TD
    A["WorkflowIngestInput.from_text(...)"] --> B["run_ingest_workflow(...)"]
    B --> N1["start"]
    N1 --> N2["normalize_input"]
    N2 --> N3["build_source_map"]
    N3 --> N4["init_parse_session"]
    N4 --> N5["check_frontier_remaining"]

    N5 -->|frontier remains| N6["prepare_layer_frontier"]
    N5 -->|frontier empty| N18["finalize_semantic_tree"]

    N6 --> N7["propose_layer_breakdown"]
    N7 --> N8["review_cud_proposal"]
    N8 --> N9["apply_cud_update"]
    N9 --> N10["check_layer_coverage"]
    N10 --> N11["check_layer_satisfaction"]

    N11 -->|retry same strategy| N7
    N11 -->|switch strategy| N12["switch_split_strategy"]
    N12 --> N7
    N11 -->|accept layer| N13["repair_layer_pointers"]

    N13 --> N14["dedupe_and_filter_layer"]
    N14 --> N15["commit_layer_children"]
    N15 --> N16["check_children_expandable"]

    N16 -->|enqueue next frontier| N17["enqueue_next_layer_frontier"]
    N17 --> N5
    N16 -->|check frontier again| N5

    N18 --> N19["validate_tree"]
    N19 --> N20["export_graph"]
    N20 --> N21["persist_canonical_graph"]
    N21 --> N22["end"]
```

Mermaid overview: OCR ingress
-----------------------------

```mermaid
flowchart TD
    A["image_payloads or pdf_path"] --> B["prepare_ocr_workflow_input(...)"]
    B --> C["ocr-state.sqlite"]
    B --> D["rendered_pages/"]
    C --> E["resolve next page"]
    E --> F["resolve next candidate model"]
    F --> G["OCR model call per page"]
    G --> H["legacy_split_pages/page_N.json"]
    H --> I["ocr-progress.json"]
    I --> J["normalize_ocr_pages(...)"]
    J --> K["run_ingest_workflow(...)"]

    K --> N1["start"]
    N1 --> N2["normalize_input"]
    N2 --> N3["build_source_map"]
    N3 --> N4["init_parse_session"]
    N4 --> N5["check_frontier_remaining"]

    N5 -->|frontier remains| N6["prepare_layer_frontier"]
    N5 -->|frontier empty| N18["finalize_semantic_tree"]

    N6 --> N7["propose_layer_breakdown"]
    N7 --> N8["review_cud_proposal"]
    N8 --> N9["apply_cud_update"]
    N9 --> N10["check_layer_coverage"]
    N10 --> N11["check_layer_satisfaction"]

    N11 -->|retry same strategy| N7
    N11 -->|switch strategy| N12["switch_split_strategy"]
    N12 --> N7
    N11 -->|accept layer| N13["repair_layer_pointers"]

    N13 --> N14["dedupe_and_filter_layer"]
    N14 --> N15["commit_layer_children"]
    N15 --> N16["check_children_expandable"]

    N16 -->|enqueue next frontier| N17["enqueue_next_layer_frontier"]
    N17 --> N5
    N16 -->|check frontier again| N5

    N18 --> N19["validate_tree"]
    N19 --> N20["export_graph"]
    N20 --> N21["persist_canonical_graph"]
    N21 --> N22["end"]
```

Mermaid overview: OCR pages mirroring legacy
--------------------------------------------

```mermaid
flowchart TD
    A["image_payloads or pdf_path"] --> B["prepare_ocr_workflow_input(...)"]
    B --> C["ocr-state.sqlite"]
    C --> D["resolve completed render + ocr pages"]
    D --> E["OCR model call per page"]
    E --> F["SplitPage(...)"]
    F --> G["legacy_split_pages/page_N.json"]
    G --> H["SplitPage.dump_supercede_parse()"]
    H --> I["artifacts.ocr_pages"]
    I --> J["normalize_ocr_pages(...)"]
    J --> K["artifacts.workflow_input"]
    K --> L["run_ingest_workflow(...)"]
```

Mermaid overview: reusable CLI and subworkflow surface
------------------------------------------------------

```mermaid
flowchart TD
    A["workflow-ingest CLI"] --> B["cli.py"]
    B --> C["runner helpers"]

    C --> D["run_ocr_source_workflow(...)"]
    C --> E["run_ocr_batch_workflow(...)"]
    C --> F["run_page_index_source_workflow(...)"]
    C --> G["run_page_index_batch_workflow(...)"]
    C --> H["run_layerwise_source_workflow(...)"]
    C --> I["run_layerwise_batch_workflow(...)"]
    C --> J["run_demo_harness_workflow(...)"]

    D --> K["prepare_ocr_workflow_input(...)"]
    E --> K
    K --> L["ocr-state.sqlite"]
    K --> M["workflow-events.jsonl"]
    K --> N["legacy_split_pages/page_N.json"]
    K --> O["normalize_ocr_pages(...)"]
    O --> P["run_ingest_workflow(...)"]

    F --> Q["parse_page_index_document(...)"]
    G --> Q
    Q --> R["page-index-summary.json"]
    Q --> S["workflow-events.jsonl"]

    H --> T["legacy parse_doc(...)"]
    I --> T
    T --> U["layerwise-graph.json"]
    T --> V["layerwise-summary.json"]
    T --> W["workflow-events.jsonl"]

    J --> X["demo harness artifacts"]
    X --> Y["probe-events.jsonl"]
    X --> Z["demo-summary.json"]
```

This diagram focuses on the bridge layer:

- `page_N.json` is the legacy-style OCR artifact written to disk
- `ocr-state.sqlite` is the authoritative rerun state store for render + OCR page stages
- `workflow-events.jsonl` is the readable outer step trail for file/page orchestration
- `SplitPage.dump_supercede_parse()` converts that artifact into the page dict
  shape used by workflow ingest
- `artifacts.ocr_pages` is the in-memory mirror of the legacy OCR pages just
  before normalization into `WorkflowIngestInput`

Retry and resume notes
----------------------

- The OCR prep layer now resumes from `ocr-state.sqlite`, not from JSON alone.
- If `ocr-state.sqlite` is missing, the resolver best-effort rebuilds it from:
  - `rendered_pages/`
  - `legacy_split_pages/page_N.json`
  - `ocr-progress.json`
- OCR retry is page-scoped:
  - resolve next page
  - resolve next candidate model
  - attempt OCR
  - mark success or failure in SQLite
  - move to the next candidate until exhausted
- The downstream `run_ingest_workflow(...)` run still starts fresh; only OCR prep resumes.
