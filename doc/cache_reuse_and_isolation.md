# Workflow Cache Reuse And Isolation

The parser cache is an optimization for provider responses. It is not parser
state and it must never be treated as authoritative source data.

## What May Be Reused

An entry may be reused when the request fingerprint is unchanged. The parser
fingerprint includes the operation, source-map content, parse session, layer
context, current result, parser input, and strategy-specific inputs. Because
the source-map content is included, changing an authoritative document
revision produces a different entry.

The normal workflow also adds a cache namespace containing:

- cache contract version;
- installed parser package version;
- Python implementation and major/minor version;
- effective provider settings, including provider, model, endpoint, and output
  configuration.

This permits reuse across repeated runs using the same parser/provider/runtime
combination while preventing accidental reuse between, for example, two
models that happen to receive the same document.

## What Must Not Be Shared

Do not point unrelated providers, models, parser releases, or Python runtimes
at a common cache directory unless they use distinct namespaces. Never put API
keys or access tokens into cache fingerprints or namespace configuration.
Provider settings include identifiers such as an API-key environment variable,
not its secret value.

The cache does not contain vectors, credentials, or canonical graph state.
Deleting it is safe; the next run recomputes provider results.

## Concurrency And Corruption

Entries are written to a temporary file in the cache directory, flushed, and
atomically replaced into place. A partially written JSON file is therefore
not published as a valid hit. If an older or externally damaged entry cannot
be read, it is discarded and recomputed.

Concurrent callers may still duplicate a provider call when they miss the
same key. That is a cost/performance issue, not a correctness shortcut: every
result is validated by the normal parser path before it becomes parser state.

## Operator Guidance

Use a stable cache directory for repeated runs of one parser/provider setup.
When changing a prompt/schema contract intentionally, bump the parser release
or cache contract version rather than manually copying entries between
providers. The cache namespace is not a substitute for source revisioning or
for deterministic pointer validation.
