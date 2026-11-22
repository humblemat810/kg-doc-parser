# ADR 0001: Native Rust Parser Port

## Status

Accepted.

## Context

The Python parser has mature OCR, page-index, layerwise, provider, and artifact surfaces. A native Rust port must start at the smallest reusable seam without rewriting `kogwistar-core` or duplicating `kogwistar-agent`.

## Decision

- Use an in-repo Cargo workspace under `rust/`.
- Keep Python packaging and entry points as the compatibility layer.
- Start with a native semantic-node model, stable node IDs, deterministic text/Markdown section parsing, and a JSON CLI seam.
- Treat OCR providers, LLM layering, and artifact contracts as later parity milestones, not silently approximated behavior.

## Consequences

Rust and Python can be compared with text/Markdown fixtures first. The parser parity gate stays incomplete until provider-independent and provider-backed surfaces have golden coverage.
