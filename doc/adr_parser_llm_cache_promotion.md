# ADR: Parser LLM Cache Promotion

Status: accepted

## Context

Structured LLM output can be valid enough to return from a provider call while
still failing pointer validation, tree validation, or canonical graph
persistence. Persisting that response immediately makes a later retry replay a
result that was never accepted by the knowledge system.

## Decision

Parser LLM caching is transaction-scoped and has two states:

1. Committed entries were produced by a parse whose validation and canonical
   graph persistence succeeded.
2. Staged entries belong to the current parse/persist attempt and remain
   memory-only until the caller explicitly promotes them.

`parser_llm_cache_transaction()` establishes the scope. The application and
maintenance worker call `promote()` only after `ingest_parse_result()` returns
successfully and the current maintenance lease is still owned. Exceptions,
failed persistence, cancellation, and lost ownership discard staged entries.
The cache transaction is not a database transaction and does not keep a graph
connection open during an LLM call.

Outside an explicit transaction, decorated parser operations bypass durable
caching. This is intentional: a parser function returning a value alone does
not prove that the value became authoritative graph state.

## Correction semantics

`correct_level_children_with_iterative_pipeline` is an atomic cache unit only
when it returns no `pending_fix_children`. A partial reduction, such as ten
unresolved children becoming three, is meaningful progress but is not cached as
a completed result because the current API does not persist and resume a
correction frontier. The three children remain retryable.

Successful provider calls inside a transaction may be staged, but they are
discarded if the enclosing parse does not reach canonical persistence. This
prevents a successful inner call from masking a failed outer parse.

## Cache identity and operations

Cache identity includes the parser configuration fingerprint, provider context,
requested model names, operation name, and semantic inputs. Injected
`call_llm_structured` callbacks and `max_rounds` are excluded because they
control invocation rather than the successful semantic result. Exceptions are
never cached. Corrupt entries are treated as misses. Promotion writes through a
temporary file and atomically replaces the committed entry.

The cache root is selected by `KG_DOC_PARSER_JOBLIB_CACHE_DIR`; the semantic
revision is controlled by `KG_DOC_PARSER_LLM_CACHE_REVISION`. Bump the revision
when prompts, schemas, or result semantics change.

## Maintenance-first relationship

`maintenance_first` does not parse during initial source seeding. Its later
maintenance worker does invoke the parser, and therefore uses the same staged
promotion boundary. A lost lease or failed canonical write cannot publish a
parser cache entry. Durable partial correction checkpoints remain a separate
future feature and must include the source fingerprint, fixed frontier, pending
frontier, and retry history.

## Consequences

Successful runs avoid repeating expensive, irregular provider calls. Failed or
incomplete runs remain retryable. The tradeoff is that standalone parser calls
without an explicit persistence owner receive no durable cache benefit, and
partial correction progress is not retained until a formal checkpoint contract
is implemented.
