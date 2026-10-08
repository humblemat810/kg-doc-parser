# kg-doc-parser 0.2.5

This corrective patch release makes the parser's provider and packaging
contracts explicit and keeps the layerwise workflow behavior observable.

## Included

- Keeps per-layer parsing strategy order, triage behavior, strategy disabling,
  and explicit terminal parse failure consistent with the progressive
  refinement contract.
- Reports provider-backed semantic assignment accurately instead of labeling it
  as deterministic fallback.
- Declares OpenAI/Azure, Gemini, and Vertex adapters as optional extras so the
  base installation remains provider-neutral:

  ```powershell
  poetry install -E cloud
  ```

- Adds CI coverage that installs and imports the optional cloud adapter set
  without requiring provider credentials.
- Retains the portable local Bonsai evidence and adversarial Markdown fixture
  documentation from the 0.2.4 corrective work.
- Adds the v0.2.5 audit corrections for terminal-content ownership, exact
  source excerpt validation, stable source-occurrence identity, PAGE parent
  links, title-safe deduplication, and the post-transform layer commit gate.
- Persists an accepted atomic/no-op disposition on retained parents so final
  validation can report `atomic_valid` rather than silently treating an
  intentional zero-child refinement as ordinary completion.
- Keeps final tree validation layered: structural, source-bound, and exact
  excerpt checks run during finalization; terminal ownership is evaluated by
  the post-parse `validate_tree` gate so failed runs still return a structured
  export bundle for diagnosis.
- Verifies the corrective matrix locally on CPython 3.13: the durable
  mixed-parent batch cases pass for `excerpt_first`, `boundary_first`, and
  `page_index` (the PageIndex rerun completed in `250.78s`). Each case proves
  that a valid sibling is committed while a failed parent is requeued and
  completed independently. Durable empty-provider atomic fallback,
  invalid-pointer fail-closed, and duplicate-occurrence regressions also pass.
  The remaining audit checklist is authoritative for items that still require
  a clean interpreter, backend-specific evidence, or live provider execution.
- Rebuilds and verifies the `0.2.5` wheel and sdist. The installed wheel
  passed three focused parser regressions from its isolated target path; the
  artifact hashes are recorded in the corrective checklist.

## Validation

The release branch is validated by the normal matrix for CPython 3.12, 3.13,
3.14, and PyPy 3.11, plus focused packaging and workflow-ingest regressions.
Cloud provider live calls are not part of the release gate: no paid requests
are made, and provider credentials are not included in the repository.

OpenRouter free-route probing was stopped at credential preflight with an
authentication failure before inference; it is not presented as parser
provider coverage.

The paid and hosted provider adapters are not release-blocking evidence in
this environment. They remain explicitly `NOT VERIFIED` unless a user-owned,
authorized configuration is actually executed. Offline adapter-construction
tests and local Ollama tests must not be read as live OpenAI, Azure, Gemini,
Vertex, or Codex coverage.

## Release Gate

Run the full parser CI suite and publish only from the matching `v0.2.5` tag.
The focused local command is:

```powershell
python -m pytest tests/test_packaging_imports.py tests/test_workflow_ingest_provider_contract_matrix.py -q -p no:cacheprovider
```
