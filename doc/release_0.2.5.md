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

## Validation

The release branch is validated by the normal matrix for CPython 3.12, 3.13,
3.14, and PyPy 3.11, plus focused packaging and workflow-ingest regressions.
Cloud provider live calls are not part of the release gate: no paid requests
are made, and provider credentials are not included in the repository.

OpenRouter free-route probing was stopped at credential preflight with an
authentication failure before inference; it is not presented as parser
provider coverage.

## Release Gate

Run the full parser CI suite and publish only from the matching `v0.2.5` tag.
The focused local command is:

```powershell
python -m pytest tests/test_packaging_imports.py tests/test_workflow_ingest_provider_contract_matrix.py -q -p no:cacheprovider
```
