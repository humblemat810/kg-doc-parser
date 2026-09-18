# ADR: Bounded Layered Parse Contract

The parser exposes `LayeredParseSeedRequest`/`LayeredParseSeedResult` and
`LayeredParseExpandRequest`/`LayeredParseExpandResult` in
`workflow_ingest.layered_contracts`.

The contract is deliberately stateless: the caller supplies the serializable
session, frontier, semantic tree, source map, and limits on every invocation.
The parser returns at most the requested frontier batch, child candidates,
remaining frontier, diagnostics, and usage. The caller owns leases, durable
commit IDs, retries, and recovery.

`workflow_layered` expansion requires an injected proposal function. The
legacy compatibility mode can expand deterministically from its stored tree.
No parser temporary directory is a recovery source.
