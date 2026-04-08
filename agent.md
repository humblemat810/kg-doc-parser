This repo is entering a major architectural update.

Previous direction:
- keep `kg_doc_parser` independent from Kogwistar
- avoid importing Kogwistar directly
- let clients know the local JSON DTO shape
- treat the parser mostly as a staged parsing utility that can later be integrated elsewhere

New direction:
- keep `kg_doc_parser` decoupled from direct Kogwistar model imports unless there is a very strong reason
- but actively reuse Kogwistar runtime semantics, workflow design ideas, and ingestion orchestration patterns
- stop assuming callers already know ad hoc local JSON DTOs
- move toward a workflow-native ingestion system with stable contracts, explicit step boundaries, and replayable execution semantics

Kogwistar should now be treated as:
- the reference peer library for workflow runtime semantics
- the reference for workflow design metadata and execution patterns
- the reference for graph-native ingestion, rebuildable projections, and consolidation ideas
- not an editable surface inside this repo

Import rule:
- do not import strict Kogwistar models into parser code just to satisfy shape compatibility
- if this repo needs compatible request/response shapes, define local protocol or mirror DTO layers first
- those local shapes should be designed so they can later align with or be generated from workflow-facing contracts
- importing Kogwistar workflow design/runtime artifacts is expected if this repo adopts workflow-native orchestration
- in particular, workflow design nodes, workflow design edges, workflow runtime contracts, and resolver-facing workflow semantics are acceptable integration imports
- the thing to avoid is accidental parser-domain coupling to Kogwistar ingestion DTOs or storage models when a local protocol shape would be cleaner
- direct Kogwistar imports are acceptable when the import is clearly part of workflow/runtime integration and the coupling is deliberate

Core architectural change:
- the parser should no longer be thought of as "a client posts JSON in the expected DTO shape and gets parsed output"
- the parser should be thought of as "a workflow-driven ingestion pipeline that happens to expose DTO boundaries"
- the important abstraction is the workflow step contract, not the incidental JSON dump shape

What this means in practice:
- parsing, OCR, correction, validation, export, consolidation, and indexing should become explicit workflow stages
- LLM calls should be isolated into named workflow operations with typed outputs and retry policy
- deterministic repair and validation should stay deterministic and cacheable
- intermediate artifacts should be treated as workflow state, not loose undocumented JSON conventions
- replay, resume, retry, fanout, and join semantics should be first-class design concerns

Importance of `pydantic-extension`:
- `pydantic-extension` is an important part of this repo's contract design, not just a convenience library
- it is the main tool for expressing different views of the same model across LLM-facing, backend-facing, frontend-facing, and DTO-facing boundaries
- it should remain central when refactoring parser DTOs into workflow-oriented state and protocol shapes
- use it to keep one authoritative model with controlled projections, rather than duplicating near-identical models with drifting fields

Proper use of `pydantic-extension`:
- use field-mode slicing deliberately to define which fields are exposed to LLMs, backend persistence, frontend views, and DTO transport
- prefer one well-annotated model with explicit mode rules over many ad hoc dict transforms
- keep authoritative/internal fields excluded from LLM mode unless they are truly needed
- do not leak backend-only or runtime-only metadata into LLM-facing payloads by accident
- when changing a shared model, check every intended field mode, not just default `model_dump()`
- when export/import behavior depends on sliced views, treat those mode contracts as part of the public architecture
- do not replace `pydantic-extension` usage with loose raw-dict manipulation unless there is a strong reason
- if a model becomes too overloaded, split it intentionally into protocol layers instead of quietly bypassing slicing discipline

Recommended workflow decomposition:
1. Discover input documents
2. Split PDF or normalize source assets
3. Convert pages or images into OCR-ready artifacts
4. OCR page or image region
5. Normalize OCR result into local protocol DTOs
6. Build authoritative source map / provenance store
7. Parse semantic structure layer by layer
8. Run deterministic pointer correction
9. Run coverage and structural invariant validation
10. Export graph payload
11. Run consolidation against previously ingested graph content
12. Materialize rebuildable indexes, insights, and projections

Core invariants to preserve:
- source OCR/page data is authoritative
- ingested source data must not be rewritten in place
- all semantic output must remain traceable to source spans or source objects
- deterministic verification must guard every LLM-produced grounding
- insights, aliases, equivalence links, consolidations, and summaries are derived graph projections, not source replacements
- indexes are rebuildable projections, not source of truth

Important current repo capability:
- raw PDF splitting and page image generation already exist
- Gemini OCR into `SplitPage` / `OCRClusterResponse` already exists
- stored OCR JSON can already be regenerated into parser input
- semantic parsing already exists as a layer-wise breadth-first tree builder
- deterministic pointer validation and correction is already one of the strongest parts of the repo
- KGE-style payload export and roundtrip reconstruction already exist

Important current repo limitations:
- current design still assumes local parser-owned DTO conventions more than workflow contracts
- `prepare_document_for_llm` can collide ids between OCR text clusters and `non_text_objects`
- non-text/image objects exist in OCR schema but are not yet first-class citizens in semantic parsing
- current KGE export duplicates excerpt/source text and roundtrip import still depends on duplicated excerpt payloads
- most tests are integration-heavy and do not yet express workflow-level invariants cleanly

Expected big-update direction:
1. Replace parser-first orchestration with workflow-first orchestration.
2. Keep local DTO/protocol definitions, but make them subordinate to workflow step contracts.
3. Introduce workflow-native states for OCR results, source maps, semantic levels, correction results, validation results, and export payloads.
4. Design consolidation as a separate workflow or sub-workflow, not as a side effect buried inside parsing.
5. Make image ingestion and OCR-document ingestion first-class workflow branches, not edge cases inside text-only assumptions.
6. Move from "cached parsing script" style to resumable, inspectable workflow execution.

Guidance for future implementation:
- if a feature is easier to express as a workflow step boundary, prefer that over adding more hidden parser-side state
- if a shape is shared across parser, storage, and workflow edges, define a local protocol type rather than depending on raw dict conventions
- if a workflow integration needs Kogwistar runtime concepts, reuse the semantics directly
- if the repo is importing workflow design nodes and edges, that is part of the intended architecture rather than a violation
- if a workflow integration needs non-workflow Kogwistar strict model classes, pause and verify the coupling is worth it before adopting it
- when changing mention/reference formats, update export and import together
- when adding consolidation, prefer explicit graph relationships, adjudication, and tombstoning over overwrite
- when adding image support, define provenance for non-text objects before teaching semantic parsing to consume them

Working assumption for this repo now:
- `kg_doc_parser` remains its own codebase
- but its future architecture should align with Kogwistar runtime and workflow design, not just with ad hoc parser DTO compatibility
