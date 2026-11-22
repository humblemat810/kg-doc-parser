# Rust Port Fine Tracklist

| Task | Implementation Detail | Status | Gate |
| --- | --- | --- | --- |
| Semantic node model | Mirror `workflow_ingest/semantics.py::SemanticNode`; node type, title, summary, total/aggregate pointers, children and level | DONE | Model unit test |
| Stable node ID | Mirror Kogwistar stable-id semantics; pointer fingerprint plus parent, type, title and level | DONE | Stable ID unit test |
| Text pointer model | Mirror `HydratedTextPointer`; source cluster ID and exact character spans | DONE | Pointer model unit test |
| Parsing modes | Mirror `text_processing_utils.parse_doc_text`; snippet and delimiter modes with bounded depth | DONE | Mode unit tests |
| Markdown section parser | Heading and paragraph segmentation, source pointers, atomic leaf fallback | DONE | Text/Markdown unit tests |
| Source text extraction | Mirror `semantic_source_text.py`; direct text fields, OCR clusters, non-text objects and nested non-string evidence | DONE | Source text unit tests |
| Pointer matching (exact and whitespace) | Mirror `semantic_pointer_matching.py`; inclusive offsets, non-overlapping exact matches, whitespace-normalized spans and diagnostics | DONE | Pointer matching unit tests |
| Pointer matching (fuzzy) | Mirror explicit fuzzy threshold path; LCSseq, difflib fallback and threshold diagnostics | TODO | Fuzzy matching fixtures |
| Semantic indexing batching | Mirror `semantic_indexing_contracts.py`; short IDs, token estimates, compact JSON and deterministic batching | DONE | Batching unit tests |
| Raw OCR input adapter | Mirror `text_processing_utils.text_to_ocr_format`; deterministic single-cluster input envelope | TODO | Adapter contract test |
| Document models | Mirror `workflow_ingest/document_models.py`; nodes, references, evidence slices, media refs, summaries and policies | TODO | Model contract tests |
| Document grouping and tree | Mirror `document_grouping.py` and `page_index_tree.py`; grouping, merge rules and tree materialization | TODO | Tree golden fixtures |
| Page-index parser | Mirror `page_index*`; block specs, candidates, classification, refinement and validation | TODO | Page-index golden fixtures |
| Layerwise parser | Mirror `semantic_document_splitting_layerwise_edits.parse_doc`; source map, semantic tree and layer edits | TODO | Layerwise golden fixtures |
| Parser core session | Mirror `parser_core*`; session initialization, frontier preparation, review, commit and validation | TODO | Parser core contract tests |
| OCR normalization | Mirror `models.py` plus `document_adapters.py`; page clusters, coordinates, identity, source maps and continuation rules | TODO | OCR normalization fixtures |
| OCR runtime | Mirror `ocr_runtime*`; grouping, continuation, lineage, page state and finalization | TODO | OCR runtime fixtures |
| OCR providers | Mirror `ocr_providers.py`; provider registry, capabilities and runtime adapters | TODO | Provider contract suite |
| OCR quality | Mirror `ocr_quality*`; ground truth, reports and policy thresholds | TODO | Quality tests |
| Workflow artifacts | Mirror `ocr_artifacts.py` and `serialization.py`; typed JSON schemas and artifact I/O | TODO | Artifact schema suite |
| Provider and LLM settings | Mirror `providers.py` and `llm_structured_output.py`; role-based model selection, retries, metrics and callbacks | TODO | Provider contract tests |
| Strategy selection | Mirror `strategy.py`; priority, triage, assessment and decision contracts | TODO | Strategy fixtures |
| Service and runners | Mirror `service.py`, `runners.py` and `workflow_ingest/cli.py`; OCR, page-index, layerwise, demo and smoke commands | TODO | CLI parity suite |
| Demo harness | Mirror `demo_harness.py`; isolated server launch, artifacts and external connection | TODO | Demo smoke test |
| Semantic graph import | Mirror `semantic_graph_import.py` and `semantic_pointer_matching.py`; graph publication and pointer grounding | TODO | Graph fixture suite |
| Semantic source/index contracts | Mirror `semantic_source_text.py` and `semantic_indexing_contracts.py`; source text and indexing boundaries | TODO | Contract tests |
| Cross-language parity | Golden fixtures across text, page-index, layerwise and OCR lanes | TODO | Python/Rust fixture suite |
| CI | Add fmt, clippy, test and cross-language parity workflow | TODO | Workflow green |
