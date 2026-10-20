"""Measure parser-owned CPU paths without invoking OCR or LLM providers.

Run with ``python scripts/benchmark_parser_hotpaths.py --output report.json``.
Inputs are constructed before timing. The OCR case reads cached local files
and uses a fake accepted-artifact store, so it excludes SQLite and cold disk I/O.
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import hashlib
import json
import platform
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

# A direct script invocation must prefer this checkout to an installed parser.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg_doc_parser.workflow_ingest.models import (
    CurrentLayerContext,
    CurrentLayerResult,
    LayerChildCandidate,
)
from kg_doc_parser.workflow_ingest.ocr_pipeline import (
    OCRPageProcessingContext,
    OCRSourcePlan,
)
from kg_doc_parser.workflow_ingest.ocr_runtime_page_state import (
    reconcile_page_checkpoint,
)
from kg_doc_parser.workflow_ingest.parser_core_validation import (
    detect_layer_invariants,
)
from kg_doc_parser.workflow_ingest.semantics import (
    HydratedTextPointer,
    pointer_source_validation_error,
)


def _pointer(source_id: str, text: str, start: int, end: int) -> HydratedTextPointer:
    return HydratedTextPointer(
        source_cluster_id=source_id,
        start_char=start,
        end_char=end,
        verbatim_text=text[start : end + 1],
    )


def _layer(
    children_per_parent: int, parent_count: int, *, overlap: bool
) -> tuple[CurrentLayerContext, CurrentLayerResult, dict[str, dict[str, str]]]:
    parent_ids: list[str] = []
    parent_pointers: dict[str, list[HydratedTextPointer]] = {}
    source_map: dict[str, dict[str, str]] = {}
    children: list[LayerChildCandidate] = []
    for parent_index in range(parent_count):
        parent_id = f"parent-{parent_index}"
        source_id = f"source-{parent_index}"
        text = "".join(f"{index:08d}" for index in range(children_per_parent))
        parent_ids.append(parent_id)
        parent_pointers[parent_id] = [_pointer(source_id, text, 0, len(text) - 1)]
        source_map[source_id] = {"text": text}
        for index in range(children_per_parent):
            start = 0 if overlap else index * 8
            children.append(
                LayerChildCandidate(
                    node_id=f"child-{parent_index}-{index}",
                    parent_node_id=parent_id,
                    title=f"Section {parent_index}-{index}",
                    node_type="TEXT_FLOW",
                    total_content_pointers=[
                        _pointer(source_id, text, start, start + 7)
                    ],
                    expandable=False,
                )
            )
    return (
        CurrentLayerContext(
            depth=0,
            parent_node_ids=parent_ids,
            parent_content_pointers_by_id=parent_pointers,
        ),
        CurrentLayerResult(children=children),
        source_map,
    )


def _measure(work: Callable[[], object], *, samples: int) -> dict[str, object]:
    work()
    pilot_start = time.perf_counter_ns()
    work()
    pilot_ns = max(1, time.perf_counter_ns() - pilot_start)
    iterations = min(200, max(1, int(30_000_000 / pilot_ns)))
    wall_samples: list[float] = []
    cpu_samples: list[float] = []
    for _ in range(samples):
        gc.collect()
        cpu_start = time.process_time_ns()
        wall_start = time.perf_counter_ns()
        for _ in range(iterations):
            work()
        wall_samples.append((time.perf_counter_ns() - wall_start) / iterations / 1e6)
        cpu_samples.append((time.process_time_ns() - cpu_start) / iterations / 1e6)
    return {
        "iterations_per_sample": iterations,
        "wall_ms_samples": [round(value, 4) for value in wall_samples],
        "wall_ms_median": round(statistics.median(wall_samples), 4),
        "cpu_ms_median": round(statistics.median(cpu_samples), 4),
    }


class _AcceptedArtifactStore:
    def reconcile_orphan_artifact(self, **_kwargs: object) -> bool:
        return True


def _ocr_case(root: Path, page_count: int) -> Callable[[], object]:
    image = b"x" * 16_384
    page_sources: list[tuple[int, Path]] = []
    for page_number in range(1, page_count + 1):
        path = root / f"page-{page_number}.png"
        path.write_bytes(image)
        page_sources.append((page_number, path))
    source_plan = OCRSourcePlan(
        pdf_source=None,
        rendered_dir=root,
        legacy_dir=root,
        page_sources=page_sources,
        source_kind="image",
        input_fingerprint=hashlib.sha256(image).hexdigest(),
    )
    context = cast(
        OCRPageProcessingContext,
        SimpleNamespace(
            document_id="benchmark-document",
            configuration_fingerprints=("benchmark-config",),
            state_store=_AcceptedArtifactStore(),
        ),
    )

    def run() -> object:
        return reconcile_page_checkpoint(
            state={"page_cursor": page_count},
            context=context,
            source_plan=source_plan,
        )

    return run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")

    measurements: list[dict[str, object]] = []
    profile_work: dict[str, Callable[[], object]] = {}
    for child_count in (20, 80, 320):
        context, result, source_map = _layer(child_count, 1, overlap=False)

        def validate(
            context: CurrentLayerContext = context,
            result: CurrentLayerResult = result,
            source_map: dict[str, dict[str, str]] = source_map,
        ) -> object:
            return detect_layer_invariants(
                current_layer_context=context,
                current_layer_result=result,
                parser_source_map=source_map,
            )

        outcome = validate()
        assert outcome[0] and outcome[1]
        print(f"disjoint_siblings={child_count}", file=sys.stderr, flush=True)
        measurements.append(
            {
                "case": "disjoint_siblings",
                "items": child_count,
                **_measure(validate, samples=args.samples),
            }
        )
        if child_count == 320:
            profile_work["disjoint_siblings_320"] = validate

    for child_count in (20, 40):
        context, result, source_map = _layer(child_count, 1, overlap=True)

        def validate_overlaps(
            context: CurrentLayerContext = context,
            result: CurrentLayerResult = result,
            source_map: dict[str, dict[str, str]] = source_map,
        ) -> object:
            return detect_layer_invariants(
                current_layer_context=context,
                current_layer_result=result,
                parser_source_map=source_map,
            )

        outcome = validate_overlaps()
        assert len(outcome[2]) == child_count * (child_count - 1) // 2
        print(f"overlapping_siblings={child_count}", file=sys.stderr, flush=True)
        measurements.append(
            {
                "case": "overlapping_siblings",
                "items": child_count,
                **_measure(validate_overlaps, samples=args.samples),
            }
        )

    for parent_count in (20, 80, 160):
        context, result, source_map = _layer(2, parent_count, overlap=False)

        def validate_parents(
            context: CurrentLayerContext = context,
            result: CurrentLayerResult = result,
            source_map: dict[str, dict[str, str]] = source_map,
        ) -> object:
            return detect_layer_invariants(
                current_layer_context=context,
                current_layer_result=result,
                parser_source_map=source_map,
            )

        outcome = validate_parents()
        assert outcome[0] and outcome[1]
        print(f"two_children_per_parent={parent_count}", file=sys.stderr, flush=True)
        measurements.append(
            {
                "case": "two_children_per_parent",
                "items": parent_count,
                **_measure(validate_parents, samples=args.samples),
            }
        )

    context, result, source_map = _layer(320, 1, overlap=False)
    payload = result.model_dump(field_mode="backend", dump_format="json")

    def validate_payload() -> object:
        return CurrentLayerResult.model_validate(payload)

    def validate_payload_with_mode() -> object:
        return CurrentLayerResult.model_validate(
            payload, context={"modeslice:mode": "backend"}
        )

    def dump_payload() -> object:
        return result.model_dump(field_mode="backend", dump_format="json")

    assert validate_payload() == validate_payload_with_mode()
    print("pydantic_layer_validate=320", file=sys.stderr, flush=True)
    measurements.append(
        {
            "case": "pydantic_layer_validate",
            "items": 320,
            **_measure(validate_payload, samples=args.samples),
        }
    )
    print("pydantic_layer_validate_explicit_mode=320", file=sys.stderr, flush=True)
    measurements.append(
        {
            "case": "pydantic_layer_validate_explicit_mode",
            "items": 320,
            **_measure(validate_payload_with_mode, samples=args.samples),
        }
    )
    print("pydantic_layer_dump=320", file=sys.stderr, flush=True)
    measurements.append(
        {
            "case": "pydantic_layer_dump",
            "items": 320,
            **_measure(dump_payload, samples=args.samples),
        }
    )
    profile_work["pydantic_layer_validate_320"] = validate_payload
    profile_work["pydantic_layer_validate_explicit_mode_320"] = (
        validate_payload_with_mode
    )
    profile_work["pydantic_layer_dump_320"] = dump_payload

    pointer = result.children[0].total_content_pointers[0]

    def check_pointer_batch() -> object:
        for _ in range(10_000):
            if pointer_source_validation_error(pointer, source_map) is not None:
                raise AssertionError("benchmark pointer lost its source grounding")
        return None

    print("source_pointer_validation=10000", file=sys.stderr, flush=True)
    measurements.append(
        {
            "case": "source_pointer_validation",
            "items": 10_000,
            **_measure(check_pointer_batch, samples=args.samples),
        }
    )

    with tempfile.TemporaryDirectory(prefix="parser-hotpaths-") as directory:
        root = Path(directory)
        for page_count in (80, 320):
            work = _ocr_case(root, page_count)
            assert work() is None
            print(
                f"ocr_cached_artifact_reconcile={page_count}",
                file=sys.stderr,
                flush=True,
            )
            measurements.append(
                {
                    "case": "ocr_cached_artifact_reconcile",
                    "items": page_count,
                    **_measure(work, samples=args.samples),
                }
            )

    report = {
        "benchmark_version": 1,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.system(),
        "samples": args.samples,
        "measurements": measurements,
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if args.profile:
        import pstats

        for name, work in profile_work.items():
            print(f"\n[profile] {name}")
            profiler = cProfile.Profile()
            profiler.runcall(work)
            stats = pstats.Stats(profiler)
            stats.sort_stats("cumtime")
            stats.print_stats(16)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
