# ADR: Parser Compatibility With Two-Stage Conversation Materialization

## Status

Accepted. No parser-local embedding queue is introduced.

## Decision

The workflow-ingest parser produces a grounded `WorkflowExportBundle` and
writes parser artifacts through the conversation engine supplied by its
caller. The caller may construct that engine with Kogwistar's
`persistence_mode="two_stage"`.

In that configuration, canonical conversation artifacts are written in stage
one and remain addressable by ID. Semantic embeddings and readiness promotion
are handled by the engine's existing stage-two index worker. The parser must
not call a second embedding provider, create a second queue, or claim semantic
readiness before the engine reports it.

Workflow checkpoints, parser output, source spans, and grounding remain
available at stage one. Consumers requiring semantic retrieval must honor the
engine readiness gate and may use ID/reference reads while promotion is
pending.

## Ownership

- kg-doc-parser owns parse payload shape, span validation, and grounded graph extraction.
- llm-wiki owns which application namespace opts into two-stage mode.
- Kogwistar owns stage ordering, indexing, leases, retries, and readiness.

This ADR does not change Kogwistar core or make knowledge, workflow, wisdom, or
Obsidian namespaces two-stage by default.
