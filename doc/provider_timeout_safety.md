# Provider Timeout Safety

`workflow_ingest` uses two related protections for provider calls:

1. The workflow wrapper gives each call a wall-clock budget and records a
   timeout if that budget expires.
2. The selected provider adapter receives a finite transport timeout so the
   underlying request normally exits and releases its in-flight slot.

The adapter timeout is derived from `ProviderEndpointConfig.timeout_seconds`.
For the Codex bridge, `KOGWISTAR_MAINTENANCE_CODEX_TIMEOUT_SECONDS` may lower
that value, but it cannot increase it. The default parser and OCR endpoint
timeout is `120` seconds.

The configured transport settings are passed as follows:

- OpenAI, Azure, and Anthropic: `timeout`.
- Google GenAI and Vertex: `request_timeout`.
- Ollama: client `timeout` through `client_kwargs`.
- Codex bridge: `timeout_seconds`.

## Why a timed-out slot is not released immediately

The wrapper runs vendor calls in a daemon worker because Python cannot safely
terminate an arbitrary provider thread. If the wrapper deadline expires while
the worker is still running, the slot remains reserved until that worker exits.
Releasing it immediately would allow new requests to start while the old
request is still consuming provider capacity, defeating `max_in_flight` and
potentially multiplying load during an outage.

The process-local `provider_call_metrics_snapshot()` result exposes:

- `active_calls`: workers that still hold slots;
- `orphaned_calls`: calls whose wrapper timed out while the worker remained
  alive; and
- `in_flight_rejections`: calls refused by the admission limit.

## Operational guidance

- Keep provider transport timeouts finite and no greater than the parser
  endpoint budget.
- Treat a growing `active_calls` or `orphaned_calls` count as provider health
  pressure, not as permission to raise `max_in_flight` blindly.
- Use a provider/client implementation with cooperative cancellation when
  strict interruption is required. A Python thread timeout alone cannot kill a
  request that ignores cancellation.
- The transport timeout is normally an inactivity/request timeout supplied by
  the SDK. It is not a guarantee that a peer which continually sends bytes will
  be terminated at exactly that wall-clock duration.
