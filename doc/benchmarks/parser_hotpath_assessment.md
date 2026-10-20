# Parser hot-path assessment (2026-10-20)

## Scope and method

This is a CPU-oriented synthetic benchmark, not an end-to-end OCR or LLM run.
The harness is `scripts/benchmark_parser_hotpaths.py`; its raw Windows result
is `doc/benchmarks/parser_hotpaths_windows_2026-10-20.json`. It constructs
grounded layer models before timing, then takes the median of three warmed
samples. The OCR case reads 16 KiB cached files per page and uses an accepted
artifact stub, so it does not measure database commits or cold storage. It runs
on CPython 3.14.6, Windows, an i7-11800H, parser base revision
`76cdc4f2595a4425523825d05abb0a0fe67cdd85`. PyPy and Linux have not
been measured. Reproduce from the parser repository with:

```text
python scripts/benchmark_parser_hotpaths.py --samples 3 --profile --output doc/benchmarks/parser_hotpaths_windows_2026-10-20.json
```

## Measurements

| Case | Size | Median wall time |
| --- | ---: | ---: |
| Disjoint sibling invariant check | 20 / 80 / 320 | 0.70 / 8.38 / 147.27 ms |
| Every sibling overlaps every other | 20 / 40 | 713.11 / 2559.79 ms |
| Two children per parent | 20 / 80 / 160 parents | 2.10 / 12.92 / 53.69 ms |
| `CurrentLayerResult.model_validate` from backend JSON | 320 children | 2018.91 ms |
| Same validation with explicit backend mode context | 320 children | 7.08 ms |
| `CurrentLayerResult.model_dump(field_mode="backend", dump_format="json")` | 320 children | 1128.15 ms |
| Source-pointer validation | 10,000 checks | 34.00 ms |
| OCR checkpoint reconciliation | 80 / 320 cached pages | 29.19 / 97.08 ms |

The benchmark asserts that implicit and explicit backend-mode validation
produce equal model values for its payload. This is not a complete semantic
equivalence proof for all parser models or LLM-facing validation paths.

`cProfile` attributed the implicit validation time mainly to 641 calls to
`pydantic_extension`'s `_infer_mode_from_stack` and Python `inspect.stack`.
The backend dump spent substantial time in the extension's wrap serializer,
including repeated `typing.get_type_hints` and forward-reference evaluation.
The invariant check used about 102,000 `_pointer_end` calls for 320 disjoint
siblings, consistent with the pairwise sibling loop. The all-overlap case
materializes every conflicting pair; its output itself is quadratic.

## Decision

1. **Do not port parser model validation to Rust yet.** In trusted backend
   state-load call sites, an explicit backend validation context is the first
   candidate. Validate nested models, mode slicing, and serialized output
   parity before adopting it. LLM output must retain its distinct LLM-mode
   semantics. The model-slicing extension may also benefit from caching type
   hints; do not bypass its field exclusion policy to gain speed.
2. **Fix the Python invariant algorithm before considering Rust.** Group
   children by parent once and compare interval-sorted pointers by source
   cluster. Preserve exact conflict/coverage diagnostics and deterministic
   order. Re-benchmark ordinary and adversarial cases after that change.
   Where every pair truly conflicts, no implementation can emit all pairs in
   subquadratic output time; a bounded diagnostic policy would be a separate
   behavioral decision.
3. **Do not port source-pointer checks or OCR checkpoint reconciliation on
   this evidence.** They are much smaller and OCR reconciliation includes
   file reads/hashing. Profile real 1,000-page runs before treating either as
   a CPU hotspot.
4. **Keep the Kogwistar core comparison separate.** A fresh current core
   SQLite facade benchmark (revision `cbe4ea9edc0877cb0a419c7ba17302a33203e7f6`)
   found the Rust mode slower for append, replay, projection list, and replace
   at 30 items. The current Python-to-Rust JSON boundary and per-call SQLite
   connection are included. Move an entire batched transaction or query
   capability across the boundary and pass its existing parity and speed
   gates before proposing another Rust cutover. The core's fast-inline
   checkpoint path also outperformed eager reconciliation in a three-step
   smoke; these nested method timings are diagnostic, not additive.

The core result was measured on CPython 3.13.3 in the same Windows machine.
Median elapsed times for Python/Rust respectively were: event append
504/1398 ms, event replay 94/183 ms, projection list 49/162 ms, and projection
replace 486/1290 ms. The benchmark's speed gate failed. Its command from the
core repository is:

```text
python scripts/rust_port_persistent_benchmark.py --output-root .tmp_perf_profiles/rust-port-current --items 30 --read-cycles 5 --samples 3 --report .tmp_perf_profiles/rust-port-current.json
```

The separate core checkpoint smoke used
`python scripts/profile_in_memory_checkpoint_write.py --iterations 3`.
Eager reconciliation totaled 2151 ms and fast-inline totaled 1041 ms for
that small workload. Its `persist_checkpoint` and `persist_step_exec` method
timings were each hundreds of milliseconds, but they include nested graph
writes, and three iterations do not identify a Rust-sized CPU kernel.

The next performance change should therefore be a measured parser
validation-mode fix, followed by an algorithmic interval-index improvement.
Rust remains a candidate only for a bounded, high-frequency operation after
those changes, with cross-runtime parity and a net shipped-facade win.
