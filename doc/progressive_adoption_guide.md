# Progressive Adoption: Parse First, Add Graph Maintenance Later

This guide describes what `kg-doc-parser` can do by itself, how to use its
grounded output in a lightweight retrieval application, and what changes when
you later add LLM-Wiki for cross-document links and maintenance.

## What The Parser Does (And Does Not Do)

`kg-doc-parser` is suitable as the first stage of a system. It can:

- normalize text/OCR input into source units;
- produce an authoritative source map;
- parse page-index or layered semantic structure with source-grounded spans;
- validate pointer coverage and produce graph payloads/export bundles;
- provide parsers and serializable bounded frontier contracts for a caller to
  orchestrate.

It is **not by itself a complete production RAG server**. Its `page-index`
command produces parser outputs and summaries. It does not provide a turnkey
vector-database service, query API, production multi-tenant authorization, or
the durable application maintenance lifecycle. The parser may run through a
Kogwistar-backed workflow client, but production graph storage, embedding
profiles/index lifecycle, and retrieval service need a Kogwistar/application
owner. The parser's deterministic fake embedding provider is for tests and
development, not semantic retrieval.

The key useful handoff is not a vector. It is source-grounded structured data:

```text
immutable original source
  + authoritative source map (unit IDs and text/locations)
  + parsed tree with exact source pointers
  + graph payload with mentions/spans
```

### Layerwise strategy selection

The recursive workflow chooses an operator for each frontier layer rather than
locking an entire document subtree to one parser. Without triage, the default
cascade is:

```text
layer_excerpt -> layer_boundary -> page_index -> parse_failure
```

The caller may provide a complete permutation. With triage enabled, triage can
choose any currently enabled operator for the current layer; a failed operator
is disabled and the workflow routes back to the remaining choices. An explicit
PageIndex selection therefore starts with PageIndex but does not force
PageIndex on descendants.

PageIndex performs one layer of structural refinement. A heading can be stored
as a source-grounded container with a title-text leaf and immediate content
children. Expandable descendants return to the normal frontier and receive a
fresh strategy decision. Summaries are advisory; the immutable source map and
exact hydrated spans remain authoritative.

An application can build a small vector index over the source-map units and
store each result's `unit_id`, document ID, source URI, page/cluster, and
revision/digest alongside it. That vector index is application-owned; the
parser does not silently create or manage it.

## Step 1: Parse A Document And Export Grounded Evidence

Install the package in a Python 3.12+ environment (or the supported PyPy 3.11
profile). `parse_page_index_document` accepts both plain `.txt` and `.md`
Markdown files. The following quick example uses the **heuristic, non-LLM**
mode so users can check source mapping and serialization without credentials:

```python
from hashlib import sha256
import json
from pathlib import Path

from kg_doc_parser.workflow_ingest import parse_page_index_document
from kg_doc_parser.workflow_ingest.semantics import semantic_tree_to_kge_payload

source_path = Path("notes/chip-supply-chain.md")
raw_text = source_path.read_text(encoding="utf-8")
source_format = "markdown" if source_path.suffix.lower() == ".md" else "text"
document_id = "chip-supply-chain-v1"
source_digest = sha256(raw_text.encode("utf-8")).hexdigest()

parsed = parse_page_index_document(
    document_id=document_id,
    title=source_path.stem,
    raw_text=raw_text,
    source_format=source_format,
    mode="heuristic",
)

export = {
    "schema_version": 1,
    "document_id": document_id,
    "source_uri": source_path.resolve().as_uri(),
    "source_sha256": source_digest,
    "semantic_tree": parsed.semantic_tree.model_dump(mode="json"),
    "source_map": {
        unit_id: record.model_dump(field_mode="backend", dump_format="json")
        for unit_id, record in parsed.authoritative_source_map.items()
    },
    "graph_payload": semantic_tree_to_kge_payload(
        parsed.semantic_tree,
        doc_id=document_id,
    ),
    "coverage": parsed.coverage,
    "diagnostics": parsed.diagnostics,
}

Path("build/chip-supply-chain.parse.json").parent.mkdir(
    parents=True,
    exist_ok=True,
)
Path("build/chip-supply-chain.parse.json").write_text(
    json.dumps(export, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(f"source units: {len(export['source_map'])}")
print(f"parsed nodes: {len(export['graph_payload']['nodes'])}")
print(f"source SHA-256: {source_digest}")
```

### Use LLM-Backed Semantic Parsing

The previous example does **not** make an LLM call. For LLM-backed semantic
outline and block assignment, configure a real parser provider and use its name
as the parsing mode. The same call works for either plain text or Markdown; the
format controls input interpretation, while `mode` selects heuristic versus
provider-backed parsing.

For example, configure a local Ollama model in the environment:

```powershell
$env:KG_DOC_PARSER_PROVIDER = "ollama"
$env:KG_DOC_PARSER_MODEL = "qwen3:4b-instruct-2507-q8_0"
$env:KG_DOC_PARSER_BASE_URL = "http://127.0.0.1:11434"
```

Or configure OpenAI, with the key kept in the environment:

```powershell
$env:KG_DOC_PARSER_PROVIDER = "openai"
$env:KG_DOC_PARSER_MODEL = "gpt-4.1-mini"
$env:KG_DOC_PARSER_API_KEY_ENV = "OPENAI_API_KEY"
$env:OPENAI_API_KEY = "<set-this-outside-source-control>"
```

Then run the provider-backed parser:

```python
from pathlib import Path

from kg_doc_parser.workflow_ingest import (
    WorkflowProviderSettings,
    parse_page_index_document,
)

source_path = Path("notes/chip-supply-chain.txt")  # .txt or .md are supported
settings = WorkflowProviderSettings.from_env()
provider = settings.parser.provider
if provider in {"fake"}:
    raise RuntimeError("configure a real KG_DOC_PARSER_PROVIDER for LLM parsing")

parsed = parse_page_index_document(
    document_id="chip-supply-chain-v1",
    title=source_path.stem,
    raw_text=source_path.read_text(encoding="utf-8"),
    source_format="markdown" if source_path.suffix.lower() == ".md" else "text",
    mode=provider,
    provider_settings=settings,
)
print("provider:", provider)
print("coverage:", parsed.coverage)
print("diagnostics:", parsed.diagnostics)
print("grounded source units:", len(parsed.authoritative_source_map))
```

Provider-backed mode asks the configured model to propose semantic structure,
then validates and hydrates its pointers against the parser's source map. It is
not equivalent to arbitrary model-generated chunks: retain the original text,
source IDs, and validation diagnostics. For provider options, authentication,
and all supported environment settings, see the [Provider Guide](../README.md#provider-guide).

You can also use the CLI for local `.txt` and `.md` files:

```powershell
workflow-ingest page-index .\notes\chip-supply-chain.md --output-dir .\build\chip-md
workflow-ingest page-index .\notes\chip-supply-chain.txt --output-dir .\build\chip-text
```

The CLI defaults to heuristic mode. Select an LLM provider using the CLI's
provider and mode options and configure the matching provider first. For
example, for the Ollama environment above:

```powershell
workflow-ingest page-index .\notes\chip-supply-chain.md `
  --output-dir .\build\chip-llm `
  --source-format markdown `
  --mode ollama `
  --parser-provider ollama `
  --parser-model qwen3:4b-instruct-2507-q8_0 `
  --parser-base-url http://127.0.0.1:11434
```

Use `--source-format text` for `.txt` inputs. Provider mode and provider
configuration must agree; `--mode heuristic` is the no-LLM structural mode.

The `source_map` is the grounding catalog. A retrieval adapter should embed
eligible text records and retain their stable IDs and source metadata. For
example, define your vector adapter around this contract (the method name is
application-owned, not a `kg-doc-parser` API):

```python
records = list(parsed.authoritative_source_map.values())
texts = [record.text for record in records if record.participates_in_semantic_text]
metadatas = [
    {
        "document_id": document_id,
        "source_sha256": source_digest,
        "unit_id": record.unit_id,
        "page_number": record.page_number,
        "cluster_number": record.cluster_number,
    }
    for record in records
    if record.participates_in_semantic_text
]

# Adapt this call to your selected vector store and embedding provider.
vector_index.upsert_texts(texts=texts, metadatas=metadatas)
```

The application must preserve the original source and verify the digest when
resolving citations. Do not cite only a model-generated excerpt, discard the
unit IDs, or treat nearest-neighbor similarity as a verified graph relation.

For provider-backed parsing, configure `WorkflowProviderSettings` or the
documented `KG_DOC_PARSER_*` variables and select the matching page-index
provider mode. For OCR/PDF input, use the OCR workflow to generate normalized
pages first; preserve its page and bounding-box locations in the exported
source map. See [README](../README.md), [Quickstart](../QUICKSTART.md), and the
[layered parse contract](adr_bounded_layered_parse_contract.md).

## Step 2: Keep Retrieval Lightweight

At this stage the application can stay small:

1. Store the original file or immutable object and its digest.
2. Parse it with `kg-doc-parser`.
3. Embed source-map text units in one chosen embedding profile.
4. Store vector rows with the `unit_id`, source identity/revision, and locator.
5. On retrieval, return the source-map record and original-source citation.

This gives source-grounded vector RAG without requiring graph cross-linking.
The vector backend and query endpoint are your application's responsibility.
If you need a graph database, durable indexing queues, profile isolation, or
multi-user ACLs, use the corresponding Kogwistar/host-application facilities
rather than treating the parser's demo engines as a production service.

## Step 3: Add LLM-Wiki For Cross-Document Maintenance

LLM-Wiki's maintenance jobs require source identities registered in the
LLM-Wiki workspace. The currently supported simple route is:

1. Ingest the original sources into the LLM-Wiki workspace using its `ingest`
   operation. The normal app ingestion path invokes `kg-doc-parser`, registers
   an immutable source revision, and writes its own source-map/graph evidence.
2. Keep each returned `source_document_id`.
3. Queue a bounded maintenance request for those exact IDs with
   `maintenance_kind="document_propose_crosslinks"`.
4. Run the maintenance worker and inspect the resulting job/source status.

Example maintenance request:

```json
{
  "workspace_id": "research",
  "source_document_ids": ["<source-document-id-1>", "<source-document-id-2>"],
  "maintenance_kind": "document_propose_crosslinks",
  "objective": "Propose only cross-document links supported by source spans.",
  "max_rounds": 1,
  "max_llm_calls": 2,
  "max_tokens": 8000,
  "max_steps": 4
}
```

This keeps LLM-Wiki's role focused on cross-link proposals after the sources
are registered. It does **not** import the external parser export or vector
index: LLM-Wiki ingests/registers its own immutable source evidence and runs
its parser integration as part of that flow. The external vector RAG can remain
in place and be queried in parallel; the systems should share stable source
identity/digest metadata, not assume their graph or vector IDs are identical.

### Can LLM-Wiki Skip Parsing Entirely?

There is no general public “import parser bundle, then cross-link only” API
today. The pipeline has an injectable parser in Python composition, but
building a safe adapter/cache that pins outputs to the exact raw-source digest
and LLM-Wiki source revision is deployment integration work, not a documented
drop-in CLI workflow. Do not import graph nodes directly to bypass source
revision, readiness, namespace, ACL, and provenance checks.

If avoiding duplicate model calls is important, a future supported handoff
should accept a versioned parser export plus the exact source digest, validate
all spans against the registered immutable revision, and persist it through
LLM-Wiki's canonical ingestion path. Until such a contract exists, the safest
option is to let LLM-Wiki parse on ingest (or use its `maintenance_first`
operation mode when you want graph structure to emerge in bounded worker
phases), then run only cross-link maintenance afterward.

## Migration Checklist

- [ ] Keep original source bytes and record SHA-256 and stable URI.
- [ ] Export and retain the parser source map with every parse result.
- [ ] Store vector metadata that resolves to source-map unit IDs and a pinned
  source revision/digest.
- [ ] Keep one embedding profile per isolated vector space; do not combine
  scores from different models/preprocessing profiles.
- [ ] Preserve the existing vector index while evaluating a new graph-backed
  or multimodal projection.
- [ ] Register the same original sources in the LLM-Wiki workspace before
  requesting cross-link maintenance.
- [ ] Use exact `source_document_ids` and explicit budgets for maintenance.
- [ ] Compare citations, source coverage, and link provenance before switching
  retrieval consumers.
- [ ] Do not promote nearest-neighbor results to canonical semantic edges
  without evidence and the normal review/patch path.

## Ownership At A Glance

| Concern | Owner |
|---|---|
| OCR normalization, source units, parse trees, pointer validation, bounded parser expansion | `kg-doc-parser` |
| Graph entities, vector backend contracts, embeddings/profile isolation, graph/runtime primitives | Kogwistar core |
| Source revisions, workspace mapping, application ACLs, durable workflow jobs, cross-link policy and review | Kogwistar LLM-Wiki |
| External vector index, application-specific query API, object storage and citation UX in parser-only mode | The integrating application |
