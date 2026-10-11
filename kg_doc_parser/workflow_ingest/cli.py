"""Command line entrypoints for reusable workflow ingest runners.

The CLI is intentionally thin: it only parses arguments, builds provider
settings, and calls the reusable runner helpers in `kg_doc_parser.workflow_ingest.runners`.
That keeps the API-first surface available to tests and orchestration code
while still giving humans a simple command family to run locally.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import get_args

from .demo_harness import DemoHarnessConfig
from .page_index import PageIndexMode
from .probe import WorkflowProbe
from .providers import WorkflowProviderSettings
from .runners import (
    WorkflowCommandResult,
    _fallback_parse_semantic_fn,
    build_legacy_parse_semantic_fn,
    run_demo_harness_workflow,
    run_layerwise_batch_workflow,
    run_layerwise_source_workflow,
    run_ocr_batch_workflow,
    run_ocr_source_workflow,
    run_page_index_batch_workflow,
    run_page_index_source_workflow,
)
from .smoke_assets import generate_ocr_smoke_assets


def _provider_settings_from_args(args: argparse.Namespace) -> WorkflowProviderSettings | None:
    ocr_values = {
        "provider": args.ocr_provider,
        "model": args.ocr_model,
        "base_url": args.ocr_base_url,
        "api_key_env": args.ocr_api_key_env,
    }
    parser_values = {
        "provider": args.parser_provider,
        "model": args.parser_model,
        "base_url": args.parser_base_url,
        "api_key_env": args.parser_api_key_env,
    }
    proposal_mode = getattr(args, "proposal_mode", None)
    parse_strategy = getattr(args, "parse_strategy", None)
    parse_strategy_order = getattr(args, "parse_strategy_order", None)
    triage_enabled = getattr(args, "triage_enabled", None)
    page_index_summary_enabled = getattr(args, "page_index_summary_enabled", None)
    page_index_hierarchical_summary_enabled = getattr(
        args, "page_index_hierarchical_summary_enabled", None
    )
    ocr_override = any(value is not None for value in ocr_values.values())
    parser_override = any(value is not None for value in parser_values.values())
    proposal_override = proposal_mode is not None
    strategy_override = parse_strategy is not None
    strategy_order_override = parse_strategy_order is not None
    triage_override = triage_enabled is not None
    summary_override = page_index_summary_enabled is not None
    hierarchical_summary_override = page_index_hierarchical_summary_enabled is not None
    if (
        not ocr_override
        and not parser_override
        and not proposal_override
        and not strategy_override
        and not strategy_order_override
        and not triage_override
        and not summary_override
        and not hierarchical_summary_override
    ):
        return None
    settings = WorkflowProviderSettings.from_env()
    if proposal_override:
        settings = settings.model_copy(update={"proposal_mode": proposal_mode})
    if strategy_override:
        settings = settings.model_copy(update={"parse_strategy": parse_strategy})
    if strategy_order_override:
        settings = settings.model_copy(
            update={"parse_strategy_order": tuple(parse_strategy_order.split(","))}
        )
    if triage_override:
        settings = settings.model_copy(update={"triage_enabled": triage_enabled})
    if summary_override:
        settings = settings.model_copy(update={"page_index_summary_enabled": page_index_summary_enabled})
    if hierarchical_summary_override:
        settings = settings.model_copy(
            update={
                "page_index_hierarchical_summary_enabled": page_index_hierarchical_summary_enabled
            }
        )
    if ocr_override:
        settings = settings.model_copy(
            update={
                "ocr": settings.ocr.model_copy(
                    update={key: value for key, value in ocr_values.items() if value is not None}
                )
            }
        )
    if parser_override:
        settings = settings.model_copy(
            update={
                "parser": settings.parser.model_copy(
                    update={key: value for key, value in parser_values.items() if value is not None}
                )
            }
        )
    return settings


def _add_provider_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("provider overrides")
    group.add_argument("--proposal-mode", choices=["children", "boundaries"], default=None)
    group.add_argument(
        "--parse-strategy",
        choices=["auto", "layer_excerpt", "layer_boundary", "page_index"],
        default=None,
        help="Select the parser strategy for every frontier layer in this parse call",
    )
    group.add_argument(
        "--parse-strategy-order",
        default=None,
        help="comma-separated parser order, e.g. layer_boundary,layer_excerpt,page_index",
    )
    group.add_argument(
        "--triage-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable provider-backed per-layer strategy triage",
    )
    group.add_argument(
        "--page-index-summary-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable summaries on PageIndex nodes for this parse call",
    )
    group.add_argument(
        "--page-index-hierarchical-summary-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable the bounded parent-summary context pass",
    )
    group.add_argument("--ocr-provider", default=None)
    group.add_argument("--ocr-model", default=None)
    group.add_argument("--ocr-base-url", default=None)
    group.add_argument("--ocr-api-key-env", default=None)
    group.add_argument("--parser-provider", default=None)
    group.add_argument("--parser-model", default=None)
    group.add_argument("--parser-base-url", default=None)
    group.add_argument("--parser-api-key-env", default=None)


def _emit_result(result: WorkflowCommandResult) -> None:
    payload = {
        "kind": result.kind,
        "input_path": str(result.input_path),
        "output_dir": str(result.output_dir),
        "status": result.status,
        "probe_path": str(result.probe_path) if result.probe_path else None,
        "summary_path": str(result.summary_path) if result.summary_path else None,
        "extra": result.extra,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def _emit_result_list(results: Iterable[WorkflowCommandResult]) -> None:
    print(
        json.dumps(
            [
                {
                    "kind": result.kind,
                    "input_path": str(result.input_path),
                    "output_dir": str(result.output_dir),
                    "status": result.status,
                    "probe_path": str(result.probe_path) if result.probe_path else None,
                    "summary_path": str(result.summary_path) if result.summary_path else None,
                    "extra": result.extra,
                }
                for result in results
            ],
            indent=2,
            sort_keys=True,
        )
    )


def _build_probe(output_dir: Path, *, command: str) -> WorkflowProbe:
    probe = WorkflowProbe(output_dir / "workflow-events.jsonl")
    probe.emit("cli.command_started", command=command, output_dir=str(output_dir))
    return probe


def _add_common_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", required=True)


def _ocr_command(args: argparse.Namespace) -> int:
    provider_settings = _provider_settings_from_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = _build_probe(output_dir, command="ocr")
    deps: dict[str, object] = {"parse_semantic_fn": _fallback_parse_semantic_fn}
    if any(
        value is not None
        for value in (
            args.parser_provider,
            args.parser_model,
            args.parser_base_url,
            args.parser_api_key_env,
        )
    ):
        effective_settings = provider_settings or WorkflowProviderSettings.from_env()
        deps = {
            "parse_semantic_fn": build_legacy_parse_semantic_fn(
                provider_settings=effective_settings,
                model_names=[args.parser_model] if args.parser_model else None,
            )
        }
    try:
        inputs = [Path(item) for item in args.inputs]
        if len(inputs) == 1 and inputs[0].is_file():
            result = run_ocr_source_workflow(
                inputs[0],
                output_dir=output_dir,
                provider_settings=provider_settings,
                ocr_candidate_models=args.ocr_candidate_model or None,
                probe=probe,
                deps=deps,
            )
            _emit_result(result)
        else:
            result = run_ocr_batch_workflow(
                inputs,
                output_dir=output_dir,
                provider_settings=provider_settings,
                ocr_candidate_models=args.ocr_candidate_model or None,
                probe=probe,
                deps=deps,
            )
            _emit_result_list(result)
        probe.emit("cli.command_finished", status="ok")
        return 0
    finally:
        probe.close()


def _page_index_command(args: argparse.Namespace) -> int:
    provider_settings = _provider_settings_from_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = _build_probe(output_dir, command="page-index")
    try:
        inputs = [Path(item) for item in args.inputs]
        if len(inputs) == 1 and inputs[0].is_file():
            result = run_page_index_source_workflow(
                inputs[0],
                output_dir=output_dir,
                mode=args.mode,
                source_format=args.source_format,
                provider_settings=provider_settings,
                probe=probe,
            )
            _emit_result(result)
        else:
            results = run_page_index_batch_workflow(
                inputs,
                output_dir=output_dir,
                mode=args.mode,
                source_format=args.source_format,
                provider_settings=provider_settings,
                probe=probe,
            )
            _emit_result_list(results)
        probe.emit("cli.command_finished", status="ok")
        return 0
    finally:
        probe.close()


def _layerwise_command(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = _build_probe(output_dir, command="layerwise")
    try:
        inputs = [Path(item) for item in args.inputs]
        if len(inputs) == 1 and inputs[0].is_dir():
            result = run_layerwise_source_workflow(
                inputs[0],
                output_dir=output_dir,
                parsing_mode=args.parsing_mode,
                max_depth=args.max_depth,
                probe=probe,
            )
            _emit_result(result)
        else:
            results = run_layerwise_batch_workflow(
                inputs,
                output_dir=output_dir,
                parsing_mode=args.parsing_mode,
                max_depth=args.max_depth,
                probe=probe,
            )
            _emit_result_list(results)
        probe.emit("cli.command_finished", status="ok")
        return 0
    finally:
        probe.close()


def _demo_command(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = _build_probe(output_dir, command="demo")
    try:
        provider_settings = _provider_settings_from_args(args)
        config = DemoHarnessConfig(
            output_dir=output_dir,
            document_id=args.document_id,
            title=args.title,
            text=args.text,
            parser_mode=args.parser_mode,
            server_mode=args.server_mode,
            external_base_url=args.external_base_url,
            provider_settings=provider_settings,
            enable_sys_monitoring=not args.disable_sys_monitoring,
        )
        artifacts = run_demo_harness_workflow(config)
        print(json.dumps(artifacts.__dict__, indent=2, sort_keys=True, default=str))
        probe.emit("cli.command_finished", status="ok")
        return 0
    finally:
        probe.close()


def _ocr_smoke_assets_command(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = generate_ocr_smoke_assets(output_dir)
    print(json.dumps({"output_dir": str(output_dir), **outputs}, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="workflow-ingest", description="Reusable workflow ingest commands")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ocr = subparsers.add_parser("ocr", help="Run OCR workflow for a file or folder")
    ocr.add_argument("inputs", nargs="+", help="One image/PDF file or a folder of files")
    _add_common_output_args(ocr)
    _add_provider_args(ocr)
    ocr.add_argument("--ocr-candidate-model", action="append", default=[], help="Candidate OCR model names")
    ocr.set_defaults(func=_ocr_command)

    page_index = subparsers.add_parser("page-index", help="Parse text/markdown into a page-index tree")
    page_index.add_argument("inputs", nargs="+", help="One file or a folder of text/markdown files")
    _add_common_output_args(page_index)
    _add_provider_args(page_index)
    page_index.add_argument("--mode", choices=get_args(PageIndexMode), default="heuristic")
    page_index.add_argument("--source-format", choices=["auto", "text", "markdown"], default="auto")
    page_index.set_defaults(func=_page_index_command)

    layerwise = subparsers.add_parser("layerwise", help="Run the legacy recursive layerwise parser")
    layerwise.add_argument("inputs", nargs="+", help="One legacy OCR folder or a folder batch")
    _add_common_output_args(layerwise)
    layerwise.add_argument("--parsing-mode", default="snippet")
    layerwise.add_argument("--max-depth", type=int, default=10)
    layerwise.set_defaults(func=_layerwise_command)

    demo = subparsers.add_parser("demo", help="Run the workflow ingest demo harness")
    _add_common_output_args(demo)
    _add_provider_args(demo)
    demo.add_argument("--document-id", default="demo-doc")
    demo.add_argument("--title", default="Workflow Ingest Demo")
    demo.add_argument("--text", default="Alpha clause\nBeta clause\nGamma clause")
    demo.add_argument("--parser-mode", choices=["fake_layered", "legacy_cached"], default="fake_layered")
    demo.add_argument(
        "--server-mode",
        choices=["testclient", "subprocess_http", "external_http"],
        default="testclient",
    )
    demo.add_argument("--external-base-url", default=None)
    demo.add_argument("--disable-sys-monitoring", action="store_true")
    demo.set_defaults(func=_demo_command)

    smoke = subparsers.add_parser("ocr-smoke-assets", help="Generate OCR smoke assets")
    smoke.add_argument("--output-dir", required=True)
    smoke.set_defaults(func=_ocr_smoke_assets_command)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
