from __future__ import annotations

"""Reusable workflow runners for CLI and higher-level orchestration.

These helpers compose the existing workflow ingest primitives without changing
their core behavior. The CLI calls into this module, but test code and other
workflow code can also reuse the same wrappers directly.
"""

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from kogwistar.engine_core.models import Edge, Node

from .demo_harness import DemoHarnessConfig, run_demo_harness
from .ocr_pipeline import OCRImagePayload, OCRWorkflowArtifacts, prepare_ocr_workflow_input, run_ocr_ingest_workflow
from .page_index import PageIndexParseResult, PageIndexSourceFormat
from .parsing import parse_ocr_document, parse_page_index_document, parse_tree_document
from .probe import WorkflowProbe, emit_probe_event
from .providers import WorkflowProviderSettings
from .parser_core import default_parse_semantic_fn
from .service import build_default_engines, run_ingest_workflow
from .semantics import HydratedTextPointer, SemanticNode

SupportedOCRInput = Literal["image", "pdf"]
SupportedPageIndexInput = Literal["text", "markdown"]

OCR_FILE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".pdf"}
PAGE_INDEX_SUFFIXES = {".txt", ".md"}


@dataclass(slots=True)
class WorkflowCommandResult:
    kind: str
    input_path: Path
    output_dir: Path
    status: str | None = None
    probe_path: Path | None = None
    summary_path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OcrWorkflowCommandResult(WorkflowCommandResult):
    artifacts: OCRWorkflowArtifacts | None = None


@dataclass(slots=True)
class PageIndexWorkflowCommandResult(WorkflowCommandResult):
    result: PageIndexParseResult | None = None


@dataclass(slots=True)
class LayerwiseWorkflowCommandResult(WorkflowCommandResult):
    tree: Any | None = None
    source_map: dict[str, Any] | None = None
    graph_payload: dict[str, Any] | None = None


def _fallback_parse_semantic_fn(*, collection, parser_input_dict: dict[str, Any], parser_source_map: dict[str, dict[str, Any]]):
    root = SemanticNode(
        title=collection.title,
        node_type="DOCUMENT_ROOT",
        total_content_pointers=[],
        child_nodes=[],
        level_from_root=0,
    )
    for page in collection.pages:
        page_text_parts: list[str] = []
        page_node = SemanticNode(
            title=f"Page {page.page_number}",
            node_type="PAGE",
            total_content_pointers=[],
            child_nodes=[],
            level_from_root=1,
            parent_id=root.node_id,
        )
        for unit in page.units:
            if not getattr(unit, "text", None):
                continue
            text = str(unit.text)
            page_text_parts.append(text)
            page_node.child_nodes.append(
                SemanticNode(
                    title=(text.splitlines()[0].strip()[:80] or f"Unit {unit.cluster_number or 0}"),
                    node_type="TEXT_FLOW",
                    total_content_pointers=[
                        HydratedTextPointer(
                            source_cluster_id=unit.unit_id or f"{collection.collection_id}|p{page.page_number}",
                            start_char=0,
                            end_char=max(0, len(text) - 1),
                            verbatim_text=text,
                        )
                    ],
                    child_nodes=[],
                    level_from_root=2,
                    parent_id=page_node.node_id,
                )
            )
        if not page_node.child_nodes:
            page_node.child_nodes.append(
                SemanticNode(
                    title=f"Page {page.page_number} Empty",
                    node_type="TEXT_FLOW",
                    total_content_pointers=[],
                    child_nodes=[],
                    level_from_root=2,
                    parent_id=page_node.node_id,
                )
            )
        if page_text_parts:
            page_node.total_content_pointers = [
                HydratedTextPointer(
                    source_cluster_id=page.units[0].unit_id or f"{collection.collection_id}|p{page.page_number}",
                    start_char=0,
                    end_char=max(0, len("\n".join(page_text_parts)) - 1),
                    verbatim_text="\n".join(page_text_parts),
                )
            ]
        root.child_nodes.append(page_node)
    return root


@contextmanager
def _temporary_env(overrides: dict[str, str | None]):
    previous: dict[str, str | None] = {}
    try:
        for key, value in overrides.items():
            previous[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def build_legacy_parse_semantic_fn(
    *,
    provider_settings: WorkflowProviderSettings,
    model_names: Sequence[str] | None = None,
):
    parser_spec = provider_settings.parser
    parser_model_names = list(model_names) if model_names else [parser_spec.model]

    def _parse_semantic_fn(*, collection, parser_input_dict: dict[str, Any], parser_source_map: dict[str, dict[str, Any]]):
        env_overrides = {
            "KG_DOC_PARSER_PROVIDER": parser_spec.provider,
            "KG_DOC_PARSER_MODEL": parser_spec.model,
            "KG_DOC_PARSER_TEMPERATURE": str(parser_spec.temperature),
            "KG_DOC_PARSER_BASE_URL": parser_spec.base_url,
            "KG_DOC_PARSER_API_KEY_ENV": parser_spec.api_key_env,
            "KG_DOC_PARSER_PROJECT": parser_spec.project,
            "KG_DOC_PARSER_LOCATION": parser_spec.location,
            "KG_DOC_PARSER_MAX_RETRIES": str(parser_spec.max_retries),
        }
        with _temporary_env(env_overrides):
            return default_parse_semantic_fn(
                collection=collection,
                parser_input_dict=parser_input_dict,
                parser_source_map=parser_source_map,
                model_names=parser_model_names,
            )

    return _parse_semantic_fn


def _ensure_probe(output_dir: Path, probe: WorkflowProbe | None = None) -> WorkflowProbe:
    if probe is not None:
        return probe
    return WorkflowProbe(output_dir / "workflow-events.jsonl")


def _emit(probe: WorkflowProbe | None, kind: str, /, **payload: Any) -> None:
    emit_probe_event(probe, kind, **payload)


def discover_input_files(
    paths: Sequence[str | Path],
    *,
    allowed_suffixes: set[str],
    recursive: bool = True,
) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            iterator = path.rglob("*") if recursive else path.iterdir()
            for candidate in iterator:
                if candidate.is_file() and candidate.suffix.lower() in allowed_suffixes:
                    files.append(candidate)
            continue
        if path.suffix.lower() in allowed_suffixes:
            files.append(path)
    return sorted({p.resolve(): p for p in files}.values(), key=lambda p: str(p))


def run_ocr_source_workflow(
    source_path: str | Path,
    *,
    output_dir: str | Path,
    provider_settings: WorkflowProviderSettings | None = None,
    ocr_runner=None,
    pdf_rasterizer=None,
    ocr_candidate_models: Sequence[str] | None = None,
    workflow_engine=None,
    conversation_engine=None,
    knowledge_engine=None,
    probe: WorkflowProbe | None = None,
    deps: dict[str, Any] | None = None,
    document_id: str | None = None,
    title: str | None = None,
) -> OcrWorkflowCommandResult:
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = _ensure_probe(output_dir, probe)
    provider_settings = provider_settings or WorkflowProviderSettings.from_env()
    document_id = document_id or source_path.stem
    title = title or source_path.stem

    _emit(
        probe,
        "workflow.file_started",
        workflow_kind="ocr",
        source_path=str(source_path),
        output_dir=str(output_dir),
        document_id=document_id,
    )
    if source_path.suffix.lower() == ".pdf":
        image_payloads = None
        pdf_path = source_path
    else:
        pdf_path = None
        image_payloads = [
            OCRImagePayload(page_number=1, image_path=str(source_path)),
        ]
    if workflow_engine is None or conversation_engine is None:
        workflow_engine, conversation_engine, default_knowledge_engine = build_default_engines(
            output_dir / "engines",
            provider_settings=provider_settings,
        )
        if knowledge_engine is None:
            knowledge_engine = default_knowledge_engine
    run, bundle, artifacts = run_ocr_ingest_workflow(
        document_id=document_id,
        title=title,
        output_dir=output_dir,
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        image_payloads=image_payloads,
        pdf_path=pdf_path,
        provider_settings=provider_settings,
        ocr_runner=ocr_runner,
        pdf_rasterizer=pdf_rasterizer,
        ocr_candidate_models=ocr_candidate_models,
        deps={"probe": probe, **(deps or {})},
        probe=probe,
    )
    _emit(
        probe,
        "workflow.file_finished",
        workflow_kind="ocr",
        source_path=str(source_path),
        output_dir=str(output_dir),
        document_id=document_id,
        status=run.status,
        summary_path=str(artifacts.summary_path),
    )
    return OcrWorkflowCommandResult(
        kind="ocr",
        input_path=source_path,
        output_dir=output_dir,
        status=run.status,
        probe_path=probe.path,
        summary_path=artifacts.summary_path,
        extra={
            "run_id": run.run_id,
            "bundle": bundle.model_dump(field_mode="backend", dump_format="json") if bundle is not None else None,
            "state_db_path": str(artifacts.state_db_path),
            "legacy_dir": str(artifacts.legacy_dir),
            "rendered_dir": str(artifacts.rendered_dir),
        },
        artifacts=artifacts,
    )


def run_ocr_batch_workflow(
    source_paths: Sequence[str | Path],
    *,
    output_dir: str | Path,
    provider_settings: WorkflowProviderSettings | None = None,
    ocr_runner=None,
    pdf_rasterizer=None,
    ocr_candidate_models: Sequence[str] | None = None,
    workflow_engine=None,
    conversation_engine=None,
    knowledge_engine=None,
    probe: WorkflowProbe | None = None,
    deps: dict[str, Any] | None = None,
) -> list[OcrWorkflowCommandResult]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = _ensure_probe(output_dir, probe)
    files = discover_input_files(source_paths, allowed_suffixes=OCR_FILE_SUFFIXES, recursive=True)
    _emit(probe, "workflow.batch_started", workflow_kind="ocr", file_count=len(files), output_dir=str(output_dir))
    results: list[OcrWorkflowCommandResult] = []
    for source_path in files:
        relative_output = output_dir / source_path.stem
        result = run_ocr_source_workflow(
            source_path,
            output_dir=relative_output,
            provider_settings=provider_settings,
            ocr_runner=ocr_runner,
            pdf_rasterizer=pdf_rasterizer,
            ocr_candidate_models=ocr_candidate_models,
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            knowledge_engine=knowledge_engine,
            probe=probe,
            deps=deps,
            document_id=source_path.stem,
            title=source_path.stem,
        )
        results.append(result)
    _emit(probe, "workflow.batch_finished", workflow_kind="ocr", file_count=len(files), output_dir=str(output_dir))
    return results


def run_page_index_source_workflow(
    source_path: str | Path,
    *,
    output_dir: str | Path,
    mode: str = "heuristic",
    source_format: str = "auto",
    provider_settings: WorkflowProviderSettings | None = None,
    probe: WorkflowProbe | None = None,
) -> PageIndexWorkflowCommandResult:
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = _ensure_probe(output_dir, probe)
    raw_text = source_path.read_text(encoding="utf-8")
    inferred_format: PageIndexSourceFormat = (
        "markdown" if source_format == "auto" and source_path.suffix.lower() == ".md" else "text"
    )
    if source_format in {"text", "markdown"}:
        inferred_format = source_format  # type: ignore[assignment]

    _emit(
        probe,
        "workflow.file_started",
        workflow_kind="page_index",
        source_path=str(source_path),
        output_dir=str(output_dir),
        mode=mode,
    )
    result = parse_page_index_document(
        document_id=source_path.stem,
        title=source_path.stem,
        raw_text=raw_text,
        source_format=inferred_format,
        mode=mode,  # type: ignore[arg-type]
        provider_settings=provider_settings,
    )
    summary = {
        "kind": "page_index",
        "source_path": str(source_path),
        "mode": mode,
        "source_format": inferred_format,
        "overall_coverage": result.coverage.get("overall"),
        "page_count": len(result.workflow_input.collections[0].pages),
        "max_depth": max((len(page.child_nodes) for page in result.semantic_tree.child_nodes), default=0),
    }
    summary_path = output_dir / "page-index-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _emit(
        probe,
        "workflow.file_finished",
        workflow_kind="page_index",
        source_path=str(source_path),
        output_dir=str(output_dir),
        mode=mode,
        summary_path=str(summary_path),
    )
    return PageIndexWorkflowCommandResult(
        kind="page_index",
        input_path=source_path,
        output_dir=output_dir,
        status="succeeded",
        probe_path=probe.path,
        summary_path=summary_path,
        extra=summary,
        result=result,
    )


def run_page_index_batch_workflow(
    source_paths: Sequence[str | Path],
    *,
    output_dir: str | Path,
    mode: str = "heuristic",
    source_format: str = "auto",
    provider_settings: WorkflowProviderSettings | None = None,
    probe: WorkflowProbe | None = None,
) -> list[PageIndexWorkflowCommandResult]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = _ensure_probe(output_dir, probe)
    files = discover_input_files(source_paths, allowed_suffixes=PAGE_INDEX_SUFFIXES, recursive=True)
    _emit(
        probe,
        "workflow.batch_started",
        workflow_kind="page_index",
        file_count=len(files),
        output_dir=str(output_dir),
    )
    results: list[PageIndexWorkflowCommandResult] = []
    for source_path in files:
        result = run_page_index_source_workflow(
            source_path,
            output_dir=output_dir / source_path.stem,
            mode=mode,
            source_format=source_format,
            provider_settings=provider_settings,
            probe=probe,
        )
        results.append(result)
    _emit(probe, "workflow.batch_finished", workflow_kind="page_index", file_count=len(files), output_dir=str(output_dir))
    return results


def run_layerwise_source_workflow(
    source_path: str | Path,
    *,
    output_dir: str | Path,
    parsing_mode: str = "snippet",
    max_depth: int = 10,
    probe: WorkflowProbe | None = None,
) -> LayerwiseWorkflowCommandResult:
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = _ensure_probe(output_dir, probe)
    _emit(
        probe,
        "workflow.file_started",
        workflow_kind="layerwise",
        source_path=str(source_path),
        output_dir=str(output_dir),
        parsing_mode=parsing_mode,
    )
    if not source_path.is_dir():
        raise ValueError("layerwise workflow expects a directory of legacy OCR page artifacts")
    from kg_doc_parser.ocr import regen_doc
    from kg_doc_parser.semantic_document_splitting_layerwise_edits import (
        semantic_tree_to_kge_payload as legacy_semantic_tree_to_kge_payload,
    )

    raw_doc = {source_path.name: regen_doc(str(source_path), use_raw=True)}
    tree, source_map = parse_tree_document(
        doc_id=source_path.name,
        raw_doc_dict=raw_doc,
        parsing_mode=parsing_mode,  # type: ignore[arg-type]
        max_depth=max_depth,
    )
    graph_payload = legacy_semantic_tree_to_kge_payload(tree, doc_id=source_path.name)
    graph_path = output_dir / "layerwise-graph.json"
    graph_path.write_text(json.dumps(graph_payload, indent=2), encoding="utf-8")
    summary = {
        "kind": "layerwise",
        "source_path": str(source_path),
        "node_count": len(graph_payload.get("nodes", [])),
        "edge_count": len(graph_payload.get("edges", [])),
        "source_count": len(source_map),
        "graph_path": str(graph_path),
    }
    summary_path = output_dir / "layerwise-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _emit(
        probe,
        "workflow.file_finished",
        workflow_kind="layerwise",
        source_path=str(source_path),
        output_dir=str(output_dir),
        summary_path=str(summary_path),
    )
    return LayerwiseWorkflowCommandResult(
        kind="layerwise",
        input_path=source_path,
        output_dir=output_dir,
        status="succeeded",
        probe_path=probe.path,
        summary_path=summary_path,
        extra=summary,
        tree=tree,
        source_map=source_map,
        graph_payload=graph_payload,
    )


def run_layerwise_batch_workflow(
    source_paths: Sequence[str | Path],
    *,
    output_dir: str | Path,
    parsing_mode: str = "snippet",
    max_depth: int = 10,
    probe: WorkflowProbe | None = None,
) -> list[LayerwiseWorkflowCommandResult]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = _ensure_probe(output_dir, probe)
    dirs = [Path(path) for path in source_paths if Path(path).is_dir()]
    _emit(
        probe,
        "workflow.batch_started",
        workflow_kind="layerwise",
        file_count=len(dirs),
        output_dir=str(output_dir),
    )
    results: list[LayerwiseWorkflowCommandResult] = []
    for source_path in dirs:
        result = run_layerwise_source_workflow(
            source_path,
            output_dir=output_dir / source_path.name,
            parsing_mode=parsing_mode,
            max_depth=max_depth,
            probe=probe,
        )
        results.append(result)
    _emit(probe, "workflow.batch_finished", workflow_kind="layerwise", file_count=len(dirs), output_dir=str(output_dir))
    return results


def run_demo_harness_workflow(config: DemoHarnessConfig):
    return run_demo_harness(config)
