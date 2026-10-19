# OCR Runtime and Seam Acceptance Scope

This document is the finite acceptance-scope reference, not a progress log.
[`OCR_STATUS_LEDGER.md`](OCR_STATUS_LEDGER.md) is the only live status tracker.
Completed work and its evolution belong in Git history; update the existing
scope here only when the original requirement is clarified, not to append run
results or new gates.

The scope is fixed to A1-A6, S01-S12, and C01-C06. A newly discovered defect
should first be mapped to one of these requirements and fixed with a focused
regression test. Add scope only when it is a genuinely new product requirement
approved by the user, not merely another implementation detail or test case.

## A. Persisted Runtime

| ID | Acceptance requirement |
| --- | --- |
| A1 | OCR pages, grouping, summaries, and semantic parsing use the existing versioned Kogwistar workflow runtime and persisted lineage; no parallel orchestration framework. |
| A2 | Checkpoint state is typed JSON; dependencies remain runtime-only; resume rejects incompatible source, provider/configuration, workflow, grouping, or summary identities. |
| A3 | Page work is deterministic and idempotent: validated artifacts are published atomically, accepted pages are not repeated, and incomplete work can retry safely. |
| A4 | Grouping is a persisted, revision-fenced operation that restores accepted tree/frontier state before progressing and preserves stable groups and reading order. |
| A5 | Summaries run only on accepted grouping output, are grounded in exact source slices, have stable generation identity, and can retry without invalidating OCR/grouping. |
| A6 | Recovery is demonstrated with persisted runtime state and fresh processes; duplicate resume has one accepted owner/result and terminal output has no orphaned or duplicate accepted work. |

## B. Strict Seams

| ID | Boundary | Acceptance requirement |
| --- | --- | --- |
| S01 | PDF to rendered pages | Cache identity binds source bytes, renderer/configuration, page set and raster schema; changed or removed pages cannot reuse stale renders. |
| S02 | Source image to OCR artifact | Validate schema, source/config identity, digest and quality before reuse; publish atomically and reconcile both artifact-first and checkpoint-first crash windows. |
| S03 | OCR response to grounding | Preserve text and media identity/location; accurately label coordinate precision; report every unresolved or dropped source unit. |
| S04 | Clusters to reading order | One explicit traversal covers every accepted modality exactly once and remains stable through normalization, grouping, serialization and graph export; ambiguity is explicit. |
| S05 | Group tree to semantic input | The semantic parser consumes the accepted typed hierarchy; required hierarchy fails closed if absent, and graph output preserves group identity, order and grounding. |
| S06 | Grouping to durable checkpoint | Persist source/policy identity, tree and frontier atomically; use revision/absence fencing and reject stale, inconsistent or incomplete checkpoint state. |
| S07 | Grouping to summaries | Summary cache identity includes exact ordered slices and all output-affecting policy/provider/schema revisions; retries do not redo accepted OCR. |
| S08 | Dynamic groups to semantic identity | Group IDs, kinds, provenance and summaries are deterministic/versioned; final meaningful groups survive, and replay is idempotent. |
| S09 | OCR/group/summary to runtime recovery | Stages share Kogwistar lineage and run identity; process termination/reopen resumes accepted work without duplicate publication. |
| S10 | Source revision to graph identity | Distinguish document, source revision, OCR config, grouping revision and parse generation; changed source cannot alias prior artifacts or graph identity. |
| S11 | Cross-boundary references | Preserve exact multi-slice references across clusters/pages without silently treating references as coverage children or graph edges. |
| S12 | Public API to internal options | Public runner options reach parsing, grouping, summaries, checkpoints and export; conflicting public/internal policies fail closed. |

## C. Review Acceptance Gates

| ID | Acceptance requirement |
| --- | --- |
| C01 | Verify compatibility against the installed, pinned Kogwistar package and the actual persisted-workflow API, not only a source checkout. |
| C02 | Define one root ownership boundary, including semantic parsing as a stage or an explicitly linked durable child invocation. |
| C03 | Reconcile crashes between SQLite/artifact commits and Kogwistar checkpoints in both commit orders. |
| C04 | Prove incremental grouping produces the canonical-equivalent tree of uninterrupted grouping while recomputing only the affected frontier. |
| C05 | Demonstrate real process interruption, fresh-process resume of the same persisted run, no unnecessary accepted work replay, and one canonical publication. Guarantee at-least-once execution with idempotent acceptance, not exactly-once provider calls. |
| C06 | Treat interval coverage as insufficient evidence of OCR completeness; retain typed loss diagnostics and require modality-specific, independently grounded quality evidence for any completeness claim. |

## Implementation Locator

This is a navigation aid, not a second status ledger. The canonical state and
latest evidence live in `OCR_STATUS_LEDGER.md`; commit history is authoritative
for when and how implementation changed.

| Area | Primary implementation and regression locations | Useful history entry points |
| --- | --- | --- |
| Runtime, recovery, scaling | `kg_doc_parser/workflow_ingest/ocr_runtime.py`; `ocr_pipeline.py`; `tests/test_workflow_ingest_ocr_pipeline.py` (fresh-process, 100-page, scaling and crash-window tests) | `51a5ba6` quadratic progress snapshot removal; `43b3553` fresh-process config identity; `4512001` summary journal acceptance |
| Grouping and tree contracts | `kg_doc_parser/workflow_ingest/document_adapters.py`; `document_grouping.py`; `tests/test_document_grouping_contract.py`; `tests/test_document_adapters.py` | `aee5969` concurrent grouping CAS; `ba903eb` checkpoint companion validation |
| OCR artifacts and quality seams | `kg_doc_parser/workflow_ingest/ocr_pipeline.py`; `ocr_models.py`; `tests/test_workflow_ingest_ocr_pipeline.py`; `tests/test_workflow_ingest_ocr_providers.py` | `d50fe3f` quality validation at acceptance seams; `6ee4f8b` strict-quality retry coverage |
| Hierarchy, references and graph export | `kg_doc_parser/workflow_ingest/document_adapters.py`; `serialization.py`; pipeline and page-index export tests | `e757a75` cross-page reference export coverage |

To inspect complete historical edits without preserving a prose timeline:

```text
git log --oneline --follow -- <path>
git show <commit> -- <path>
```

## Non-Goals And Evidence Rules

- Do not add another parser-owned checkpoint/recovery framework or projection
  subsystem when Kogwistar or the existing SQLite artifact store owns that
  responsibility.
- A mocked adapter test does not prove persisted runtime recovery, live OCR
  accuracy, independent visual completeness, or durable graph publication.
- Atomic file replacement alone is not a transaction across SQLite and
  Kogwistar state.
- Never silently rewrite historical graph/source data to make a validation
  pass; reject or report it through the established audit path.
- Update the matching existing row in `OCR_STATUS_LEDGER.md` when evidence
  changes. Do not append a dated progress section here.
