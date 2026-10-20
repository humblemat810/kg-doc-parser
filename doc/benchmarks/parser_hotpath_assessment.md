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

## Initial measurements

These were exploratory measurements from the first run. Absolute timings
varied on later runs, so use the controlled before/after comparison below for
the implementation decision.

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

## Decision and implementation status

1. **Backend model validation: implemented.** Internal persisted
   `WorkflowIngestInput`, `CurrentLayerContext`, `ParseSessionState`,
   `LayerFrontierItem`, `CurrentLayerResult`, `CurrentLayerReview`, and
   `WorkflowExportBundle` payload reloads now pass explicit backend context
   across parser handlers, clients, and OCR orchestration. Provider outputs
   retain their existing validation path. Do not port this path to Rust yet.
   The model-slicing extension is the correct owner for the remaining cached
   type-hint optimization; parser code must not bypass its field exclusion
   policy.
2. **Invariant algorithm: implemented in Python.** Children are grouped by
   parent once, pointer ends are cached per check, and large sibling sets use
   an interval sweep. Tests preserve diagnostics and ordering. Dense conflict
   output remains quadratic because every pair is an observable diagnostic;
   changing that would require a separate bounded-diagnostics decision.
3. **Source-pointer checks and OCR checkpoint reconciliation remain Python.**
   They are much smaller and OCR reconciliation includes file reads/hashing.
   Profile real 1,000-page runs before treating either as a CPU hotspot.
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

## Implemented parser speed-ups and controlled comparison

To avoid comparing runs affected by machine load, the pre-change commit
`fa61fa5` and the modified working tree were benchmarked back-to-back with the
same CPython 3.14.6 environment, hardware, payloads, and three-sample median.
The original raw outputs were `parser_hotpaths_windows_2026-10-20_baseline_control.json`
and `parser_hotpaths_windows_2026-10-20_after.json`. The latter was later
refreshed with additional cases; its current contents are a follow-up profile,
not the original paired run. Historical paired figures are retained in the
table below.

| Case | Before | After | Result |
| --- | ---: | ---: | ---: |
| Disjoint sibling invariant check, 320 children | 140.08 ms | 6.01 ms | 23.3x faster |
| Two children per parent, 160 parents | 43.52 ms | 6.54 ms | 6.7x faster |
| Dense overlaps, 20 children | 1328.48 ms | 1399.25 ms | within run-to-run noise |
| Dense overlaps, 40 children | 5309.72 ms | 5332.43 ms | effectively unchanged |
| `CurrentLayerResult` validation, 320 children, implicit mode | 4290.42 ms | 4270.19 ms | unchanged |
| Same payload with explicit backend context | 6.52 ms | 6.47 ms | unchanged fast path |
| `CurrentLayerResult.model_dump`, 320 children | 1167.68 ms | 1185.74 ms | unchanged |

The application changes use explicit backend-mode validation only when
reconstructing internally persisted `CurrentLayerResult` and
`CurrentLayerReview` payloads. Provider output still follows its existing
validation path. A focused test verifies persisted-state validation does not
invoke stack inference. Parser invariant checks now group children by parent,
cache resolved pointer ends within one check, and use an interval sweep for
large layers; tests compare the sweep output against the legacy pairwise
ordering and check the end-to-end invariant result.

Dense overlap diagnostics still require materializing every conflicting pair,
so their output cost remains quadratic by design. The controlled comparison
does not show a meaningful regression there. The `model_dump` time remains a
separate dependency-level hotspot: the `pydantic-extension 0.0.7` wrap
serializer repeatedly resolves type hints. The parser must not bypass that
serializer because it enforces field-mode exclusions. A candidate cache and
its parity tests are described below; it must be released by the dependency
before the parser adopts it.

### Follow-up mode-validation profile

The refreshed `parser_hotpaths_windows_2026-10-20_after.json` adds a
`WorkflowIngestInput` case and measures current behavior on CPython 3.14.6.
This is a three-sample follow-up, not a paired baseline comparison:

| Case | Implicit stack inference | Explicit backend context | Approximate reduction |
| --- | ---: | ---: | ---: |
| `CurrentLayerResult`, 320 children | 2964.27 ms | 4.40 ms | 674x |
| `WorkflowIngestInput`, one normalized source | 19.87 ms | 0.036 ms | 557x |

These cases isolate validation overhead. They do not claim the entire workflow
is hundreds of times faster; model construction, parsing/provider latency,
serialization, graph writes, and OCR I/O remain separate costs.

The targeted parser validation/resolver tests passed (58 tests), including
explicit-context checks for both layer results and normalized workflow input.
Three in-memory workflow integration cases also passed after broadening the
internal validation call sites. The broader
parser integration run reported 149 passed, 14 failed, and one error. The 14
failures were Chroma-parameterized cases. A representative failure was traced
to Chroma import failing before parser test logic with
`ImportError: cannot import name '_clean_attribute_value' from
'opentelemetry.attributes'`. The additional fixture error was pytest failing to
create its default temporary directory under a sandbox-protected Windows temp
path. The fixture passes when pytest uses a temporary root inside the writable
checkout. This environment cannot currently validate the Chroma paths; rerun
them after the OpenTelemetry dependency set is repaired.

### Mode-aware serializer type-hint cache candidate

The remaining serializer hotspot belongs to `pydantic-extension`, not the
parser. An isolated candidate branch, `perf/modeslice-type-hint-cache`, adds a
bounded cache for resolved type hints and extracted `Annotated` metadata. The
type-hint cache fingerprints each class in the MRO and the names/identities of
its annotation entries, retaining the annotation values to prevent identity
reuse from serving stale entries. Metadata extraction is cached only for
hashable hints, with an uncached fallback for custom unhashable annotations.
Mode-registry checks and include/exclude policy still run on every call, so
the caches do not cache or bypass field-slicing decisions; mutable marker
objects remain live.

On CPython 3.14.6, a focused 320-child `CurrentLayerResult` JSON dump produced
identical payloads in all configurations. A same-process three-sample
comparison measured median dump times of 1141 ms without either cache, 272 ms
with the type-hint cache only, and 173 ms with both caches. That is 4.2x for
type-hint caching, a further 1.6x from metadata caching, and 6.6x overall.

After tightening the annotation fingerprint to detect in-place mapping
changes, the same comparison was rerun on CPython 3.13.3. Each configuration
was warmed once and then measured for four samples; all three produced the
same JSON payload (96,038 characters):

| Cache configuration | Median dump time | Relative speed |
| --- | ---: | ---: |
| None | 815.59 ms | 1.0x |
| Resolved type hints | 210.10 ms | 3.9x |
| Resolved type hints + `Annotated` metadata | 119.48 ms | 6.8x |

This is a same-process microbenchmark, not a full ingestion throughput claim.
The initial dependency candidate passed all 27 tests on CPython 3.12.13 and
3.14.6, and a pure-Python wheel built successfully with its installed artifact
passing those 27 tests on CPython 3.12.13. A cache-invalidation audit then
found that replacing `__annotations__` invalidated the first implementation,
but mutating the existing annotations dictionary in place did not. The
candidate now fingerprints annotation entries and retains their values, and a
regression test covers the in-place mutation case. The final isolated source
passes all 28 dependency tests on CPython 3.12 and 3.13; the parser's
backend-validation and resolver regression tests were rerun and pass (58
tests) with that source on `PYTHONPATH`. The wheel verification predates this
final invalidation adjustment; the rebuilt pure-Python wheel was installed
into an isolated target and passed all 28 dependency tests on CPython 3.12.
PyPy 3.11 was unavailable locally. This remains an unpublished local
candidate; the parser must not pin the branch.

Rust remains deferred. The measured parser gains come from removing repeated
Python work and an algorithmic change, while the current core Rust facade
benchmark still fails its speed gate. Reconsider a Rust cutover only after the
serializer candidate is reviewed and released, then profile remaining costs
and pass cross-runtime parity and net-facade speed gates.
