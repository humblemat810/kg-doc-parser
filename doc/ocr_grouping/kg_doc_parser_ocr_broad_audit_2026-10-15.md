# kg-doc-parser PR #46 — Broader OCR Workflow Audit

**Review date:** 2026-10-15. **Scope:** publicly retrieved `feat/ocr-provider-abstraction` source (`ocr_pipeline.py`, `ocr_runtime.py`, `document_adapters.py`, `document_models.py`) and PR metadata. Static review only; not a local checkout or executed production test suite. Confirm each issue against the exact HEAD before changing code.

**Rating convention:** Confirmed code pattern means the described path exists in the reviewed source; runtime impact may still require reproducing. P0 = release blocker if exposed to untrusted content / silent data loss; P1 = high priority; P2 = moderate.

## Findings and mandatory reproduction gates

### B01 — P0 conditional: Image filename path escape
Source: `ocr_pipeline.py`, `_materialize_image_payloads`, around lines 2243–2265.

`rendered_dir / payload.filename` is written directly, including paths containing `..` or absolute paths. `OCRImagePayload` does not constrain the name. If external callers control `image_bytes_b64` and `filename`, this can write outside the OCR folder, subject to process permissions. **Fix:** derive canonical `page_{validated_number}.{detected_extension}` within the intended directory; store original filename as metadata only; reject symlink/path escape. **Test:** `../somefile`, absolute path, symlink out of folder, duplicate page IDs; nothing outside the page directory is modified.

### B02 — P0 fidelity: Non-text OCR units silently excluded from source coverage
Source: `document_adapters.py`, `build_document_tree`, around lines 39–60, then `document_models.py`, `validate_tree`.

If a unit has neither nonempty `text`/`description` nor a complete verified media reference, the builder `continue`s. The subsequent tree coverage validation accounts for constructed `SourceItem`s, *not* the discarded original OCR units. **Fix:** retain a grounded non-text/unknown placeholder or reject with a structured unrepresented-source error; create explicit rejected-unit diagnostics and compare input-unit IDs to represented IDs. **Test:** table/image with bbox but no caption or media digest; must never silently disappear.

**Current branch status (`e1fb47a`): fixed locally, CI pending.** `build_document_tree` now raises `UnrepresentedSourceUnitError` carrying `UnrepresentedSourceDiagnostic` for this case. `test_build_document_tree_rejects_unrepresented_nontext_unit` proves the unit identity, page, modality, and reason are retained. The focused document-adapter suite passes 19 tests; full input-unit accounting across all adapters remains open.

### B03 — P1: False cross-page merges based on page-wide flags
Source: `document_adapters.py`, `_page_annotations` and `suggest_cross_page_continuations`, around lines 185–235.

Page-level `incomplete_text` / `incomplete_words_on_edge` annotations are copied to each source item. The continuation algorithm joins the adjacent last item on one page and first item on the next solely using a page boundary and previous item flag; it does not establish that both are text fragments from the same paragraph. A figure, caption, heading or table can be treated as a continuation. **Fix:** enforce compatible source kinds, text-specific boundary evidence, reading-order and geometry constraints, and an explicit merge-confidence/proposal stage; never use a page-wide flag as item-specific proof. **Test:** page ends with truncated paragraph then page begins with figure/caption/header before continuation; no wrong merge.

**Current branch status (`e1fb47a`): partially fixed locally, CI pending.** Continuation proposals now require both frontier nodes to have the canonical `text` kind, and `test_cross_page_continuation_never_merges_text_into_nontext` passes alongside the valid text-to-text regression. Geometry and stronger item-level paragraph-boundary evidence remain open.

### B04 — P1: Group summaries receive text outside owned source slices
Source: `ocr_pipeline.py`, `_generate_tree_summaries`, whole-`SourceItem` collection and `SummaryGenerator` contract.

The newer slice fingerprint detects changed slice ownership but the generator still receives whole SourceItems selected by item ID. **Fix:** introduce `GroundedTextSlice`/`EvidenceSlice` in generator input with exact ranges, separators, order and source location. **Test:** one item with two statements partitioned into two nodes; each summary call sees only its own part.

**Current branch status (2026-10-16): fixed locally, CI pending.** The parser now
provides immutable `EvidenceSlice` values and a typed
`GroundedSummaryGenerator` protocol. The standalone preparation API, persisted
grouping workflow, and root OCR workflow all thread the new boundary. The
regression `test_grounded_summary_generator_receives_only_owned_source_slices`
partitions one source item across two grouped nodes and verifies exact ordered
ranges, text, separators, and source locations. The old whole-`SourceItem`
callback remains a compatibility path; production provider migration and full
matrix verification remain open.

### B05 — P1: Spatial precision can be overstated
Source: `document_adapters.py`, `build_document_tree`, around lines 61–93.

Any positive-area bbox yields `SourceLocation.accuracy='region'`, regardless of whether provider metadata says the bounding box is a synthetic full-page approximation. **Fix:** map provider-declared precision explicitly (`page`, `region`, `line`, `word`, etc.); keep synthetic full-page boxes as `page` accuracy. **Test:** full-page fallback text box must not claim region-level grounding; genuine region boxes retain true precision.

### B06 — P1: Two documents sharing a filename collide in artifact paths
Source: `ocr_pipeline.py`, `_resolve_ocr_source_plan` and `_legacy_folder_name`, around lines 670–700 and 2579–2582.

PDFs are rendered under `rendered_root / pdf_source.stem` and legacy OCR artifacts use `legacy_root / pdf_path.name`, without including logical document identity or content revision. Two different `report.pdf` paths targeting the same `output_dir` can share files and fight over the render manifest; concurrent runs risk mixed or invalid artifacts. **Fix:** namespace by collision-resistant document key and source revision, while keeping a readable alias. **Test:** simultaneous processing of two distinct `report.pdf` files with different document IDs in the same output root.

### B07 — P1: Page-number/filename validation too permissive
Source: `ocr_pipeline.py`, `OCRImagePayload` and `_materialize_image_payloads`.

Page index has no positive bound, uniqueness check or ordering check. Duplicate page numbers can overwrite files; non-positive numbers are accepted (`0` is replaced by an index because of `or`). **Fix:** validate unique positive page numbers and deterministic ordering; fail on conflicting inputs. **Test:** duplicate page 2, negative page, page 0, out-of-order pages and mixed extensions.

### B08 — P1: OCR quality is largely self-reported and permissive by default
Source: `ocr_pipeline.py`, `OCRQualityPolicy` and `_evaluate_ocr_quality`, around lines 230–277.

Default `max_data_loss_likelihood=1.0`; incomplete text/edge flags are warnings. Only provider-declared scores/flags are checked: a structurally valid but nearly empty OCR output can be accepted even if the image contains substantial text. **Fix:** make explicit quality modes; add independent empty-page, coverage and geometry checks; escalate suspicious results to fallback or review. **Test:** fake OCR returns empty regions for a text-heavy synthetic page, and returns a false high-quality score.

**Current branch status (`e1fb47a`): partially fixed locally, CI pending.** `OCRQualityPolicy.reject_empty_page` adds a typed `empty_ocr_output` diagnostic and optional error severity; `test_ocr_quality_policy_can_reject_empty_provider_output` passes. Independent image/text completeness and geometry checks remain open.

### B09 — P1 performance: 'Bounded' summary budget does not bound provider payload
Source: `document_adapters.py`, `select_summary_targets`, around lines 297–325, and `ocr_pipeline.py`, summary call.

`estimated_tokens` is clamped with `min(... maximum_input_tokens)`, but the entire evidence items are still sent to the generator. A single huge table can exceed the model context or the intended budget. **Fix:** enforce budget at the evidence-materialization boundary; summarize large items in grounded chunks or reject/skip with diagnostics. **Test:** 500k-character table, small configured token budget, verify bytes/tokens handed to provider are actually bounded.

### B10 — P1 performance: Quadratic checkpoint serialization growth
Source: `ocr_runtime.py`, `_build_page_resolver` state contains accumulated `raw_pages`; each page step saves it again (around lines 250–332).

If one page result averages S bytes, writing a cumulative list for N page checkpoints results in roughly S×N(N+1)/2 bytes of checkpoint payload, before runtime overhead. **Fix:** checkpoint immutable artifact IDs/digests and a cursor; read exact accepted page payloads on demand. **Test:** 100/500/1000 pages, record checkpoint bytes and peak memory; expect near-linear cumulative growth.

**Current branch status (2026-10-16): fixed locally, CI pending.** New page checkpoints persist the artifact directory, page-source identity, cursor, and completion metadata without `raw_pages`; `_json_pages()` rehydrates accepted page JSON from the durable artifact set. Legacy checkpoints that already contain `raw_pages` remain readable, and repair preserves that field only for the legacy format. `test_new_page_checkpoint_rehydrates_pages_from_artifacts_without_raw_payload`, the terminal-resume test, and the fresh-process resume test pass. A 100/500/1000-page checkpoint-size benchmark remains open.

### B11 — P2: Render cache identity incomplete for variable rasterizer configurations
Source: `ocr_pipeline.py`, `_pdf_render_manifest_payload`, around lines 2279–2295.

**Current branch status (2026-10-16): partially fixed locally, CI pending.** The render manifest now includes an explicit rasterizer configuration fingerprint, plus every accepted page's SHA-256 and dimensions; cache reuse validates those values. A dedicated tamper/configuration regression passes. Built-in DPI/rotation options and a multi-page page-set regression remain open.

### B12 — P2: Reference extraction misses mentions split across OCR units
Source: `document_models.py`, `extract_reference_mentions`, around lines 383–416.

**Current branch status (2026-10-16): fixed locally, CI pending.** Detection now runs over the deterministic text reading-order stream and maps each match back to exact source slices, including multiple OCR units and pages. References remain metadata outside the primary coverage partition. `tests/test_document_grouping_contract.py::test_reference_mentions_cross_ocr_units_preserve_exact_source_slices` proves a cross-unit, cross-page mention retains both item IDs and exact half-open offsets.

### B13 — P2/architectural gap: No hybrid native-PDF-text path
Source: `ocr_pipeline.py`, `_render_pdf_to_images`; uses `PdfReader` to count pages, then rasterizes/OCRs each page. No native text extraction is visible in that pathway. This is a completeness/quality limitation rather than a regression proven by a failing test. **Fix:** optionally ingest embedded PDF glyph text and precise native coordinates first, then OCR only missing/scanned regions; reconcile duplicate coverage. **Test:** text-selectable PDF with equations, scanned figure, embedded caption and searchable table.

### B14 — P1: Interrupted rasterization can be reused as a complete PNG
Source: `ocr_pipeline.py`, `_render_pdf_with_identity` and `_render_pdf_to_images`, around lines 2297–2363.

**Current branch status (2026-10-16): partially fixed locally, CI pending.** Built-in page writes already use decode-verified atomic PNG publication. Render-manifest reuse now validates every recorded page's decode, digest, and dimensions, and a tampered-page restart regression passes. An injected mid-save failure and full page-set recovery test remain open.

## Previously tracked release gates (retain, do not silently mark closed)

- **S05:** Prove semantic parser consumes and persists the accepted grouping hierarchy; comparing intermediate JSON isn't enough.
- **S09:** Real new-process `SIGKILL` recovery and root/child resume with released Kogwistar, idempotent publication, not just injected exceptions.
- **Dual-store commits:** crash before/after SQLite and Kogwistar step completion, repeat with duplicate concurrent resume attempts.
- **Document correctness:** exact input-unit accounting and source-interval coverage, ordered multimodal traversal, semantic tree/graph source-grounding round-trip.

## Agent instructions

For each B01–B14: pin exact PR HEAD SHA; mark **confirmed in current HEAD / already fixed / not reproduced / design gap**; write a failing regression test before the fix whenever feasible; make the smallest correct patch; run focused and full parser tests. Record source line, exact test ID, expected behaviour, observed result, and Kogwistar version. Do not claim a fix merely because the checklist changed. Prioritise B01/B02/B03/B04, then B05/B06/B07 and the performance issues. Maintain backward compatibility with existing `SplitPage`, `SourceUnit` and parser APIs unless a documented versioned migration is required.

**Source entrypoints:**
- https://github.com/humblemat810/kg-doc-parser/pull/46
- https://raw.githubusercontent.com/humblemat810/kg-doc-parser/feat/ocr-provider-abstraction/kg_doc_parser/workflow_ingest/ocr_pipeline.py
- https://raw.githubusercontent.com/humblemat810/kg-doc-parser/feat/ocr-provider-abstraction/kg_doc_parser/workflow_ingest/ocr_runtime.py
- https://raw.githubusercontent.com/humblemat810/kg-doc-parser/feat/ocr-provider-abstraction/kg_doc_parser/workflow_ingest/document_adapters.py
- https://raw.githubusercontent.com/humblemat810/kg-doc-parser/feat/ocr-provider-abstraction/kg_doc_parser/workflow_ingest/document_models.py
