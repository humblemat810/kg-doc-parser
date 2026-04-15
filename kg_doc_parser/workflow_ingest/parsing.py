from __future__ import annotations

"""Unified public parsing facade for workflow-ingest.

This module exposes one parse-first surface for the three parsing lanes that
already exist in the repo:

- ``ocr`` for OCR preparation and normalization
- ``page_index`` for text / Markdown page-index parsing
- ``tree`` for the semantic layerwise legacy parser

The facade keeps provider/model selection explicit while still falling back to
``WorkflowProviderSettings.from_env()`` when callers omit overrides.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence
from contextlib import contextmanager
import os

from .ocr_pipeline import OCRImagePayload, OCRWorkflowArtifacts, prepare_ocr_workflow_input
from .page_index import PageIndexParseResult, PageIndexSourceFormat, parse_page_index_document as _parse_page_index_document
from .providers import WorkflowProviderSettings
from .semantics import SemanticNode

ParseMode = Literal["ocr", "page_index", "tree"]
OCRParseResult = OCRWorkflowArtifacts
PageIndexParseResultType = PageIndexParseResult
TreeParseResult = tuple[SemanticNode, dict[str, Any]]
ParseDocumentResult = OCRParseResult | PageIndexParseResultType | TreeParseResult


@dataclass(slots=True)
class OCRParseRequest:
    document_id: str
    title: str
    output_dir: str | Path
    image_payloads: Sequence[OCRImagePayload] | None = None
    pdf_path: str | Path | None = None
    provider_settings: WorkflowProviderSettings | None = None
    provider: str | None = None
    model: str | None = None
    ocr_runner: Any = None
    pdf_rasterizer: Any = None
    ocr_candidate_models: Sequence[str] | None = None
    probe: Any = None


@dataclass(slots=True)
class PageIndexParseRequest:
    document_id: str
    title: str
    raw_text: str
    source_format: PageIndexSourceFormat = "text"
    mode: Literal["heuristic", "ollama"] = "heuristic"
    provider_settings: WorkflowProviderSettings | None = None
    provider: str | None = None
    model: str | None = None


@dataclass(slots=True)
class TreeParseRequest:
    doc_id: str
    raw_doc_dict: dict[str, Any]
    parsing_mode: Literal["snippet", "delimiter"] = "snippet"
    max_depth: int = 10
    model_names: Sequence[str] | None = None
    provider_settings: WorkflowProviderSettings | None = None
    provider: str | None = None
    model: str | None = None


def _resolve_provider_settings(
    *,
    provider_settings: WorkflowProviderSettings | None,
    role: Literal["ocr", "parser"],
    provider: str | None = None,
    model: str | None = None,
) -> WorkflowProviderSettings:
    base = provider_settings or WorkflowProviderSettings.from_env()
    endpoint = base.ocr if role == "ocr" else base.parser
    updates: dict[str, Any] = {}
    if provider is not None:
        updates["provider"] = provider
    if model is not None:
        updates["model"] = model
    if not updates:
        return base
    return base.model_copy(
        update={
            role: endpoint.model_copy(update=updates),
        }
    )


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


def parse_ocr_document(
    *,
    document_id: str,
    title: str,
    output_dir: str | Path,
    image_payloads: Sequence[OCRImagePayload] | None = None,
    pdf_path: str | Path | None = None,
    provider_settings: WorkflowProviderSettings | None = None,
    provider: str | None = None,
    model: str | None = None,
    ocr_runner=None,
    pdf_rasterizer=None,
    ocr_candidate_models: Sequence[str] | None = None,
    probe=None,
) -> OCRWorkflowArtifacts:
    """Parse OCR inputs into normalized workflow-ingest artifacts."""

    settings = _resolve_provider_settings(
        provider_settings=provider_settings,
        role="ocr",
        provider=provider,
        model=model,
    )
    return prepare_ocr_workflow_input(
        document_id=document_id,
        title=title,
        output_dir=output_dir,
        image_payloads=image_payloads,
        pdf_path=pdf_path,
        provider_settings=settings,
        ocr_runner=ocr_runner,
        pdf_rasterizer=pdf_rasterizer,
        ocr_candidate_models=ocr_candidate_models,
        probe=probe,
    )


def parse_page_index_document(
    *,
    document_id: str,
    title: str,
    raw_text: str,
    source_format: PageIndexSourceFormat = "text",
    mode: Literal["heuristic", "ollama"] = "heuristic",
    provider_settings: WorkflowProviderSettings | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> PageIndexParseResult:
    """Parse a text / Markdown page-index document into a semantic tree."""

    settings = _resolve_provider_settings(
        provider_settings=provider_settings,
        role="parser",
        provider=provider,
        model=model,
    )
    return _parse_page_index_document(
        document_id=document_id,
        title=title,
        raw_text=raw_text,
        source_format=source_format,
        mode=mode,
        provider_settings=settings,
    )


def parse_tree_document(
    *,
    doc_id: str,
    raw_doc_dict: dict[str, Any],
    parsing_mode: Literal["snippet", "delimiter"] = "snippet",
    max_depth: int = 10,
    model_names: Sequence[str] | None = None,
    provider_settings: WorkflowProviderSettings | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> tuple[SemanticNode, dict[str, Any]]:
    """Parse a legacy split-page OCR document with the layerwise tree parser."""

    settings = _resolve_provider_settings(
        provider_settings=provider_settings,
        role="parser",
        provider=provider,
        model=model,
    )
    parser_model_names = list(model_names) if model_names else [settings.parser.model]
    env_overrides = {
        "KG_DOC_PARSER_PROVIDER": settings.parser.provider,
        "KG_DOC_PARSER_MODEL": settings.parser.model,
        "KG_DOC_PARSER_TEMPERATURE": str(settings.parser.temperature),
        "KG_DOC_PARSER_BASE_URL": settings.parser.base_url,
        "KG_DOC_PARSER_API_KEY_ENV": settings.parser.api_key_env,
        "KG_DOC_PARSER_PROJECT": settings.parser.project,
        "KG_DOC_PARSER_LOCATION": settings.parser.location,
        "KG_DOC_PARSER_MAX_RETRIES": str(settings.parser.max_retries),
    }
    from kg_doc_parser.semantic_document_splitting_layerwise_edits import parse_doc as legacy_parse_doc

    with _temporary_env(env_overrides):
        return legacy_parse_doc(
            doc_id=doc_id,
            raw_doc_dict=raw_doc_dict,
            parsing_mode=parsing_mode,
            max_depth=max_depth,
            model_names=parser_model_names,
        )


def parse_document(*, mode: ParseMode, **kwargs):
    """Dispatch to the requested parse mode and return the mode-specific result."""

    if mode == "ocr":
        return parse_ocr_document(**kwargs)
    if mode == "page_index":
        return parse_page_index_document(**kwargs)
    if mode == "tree":
        return parse_tree_document(**kwargs)
    raise ValueError(f"unsupported parse mode: {mode}")


__all__ = [
    "OCRParseRequest",
    "OCRParseResult",
    "PageIndexParseRequest",
    "PageIndexParseResultType",
    "ParseMode",
    "ParseDocumentResult",
    "TreeParseRequest",
    "TreeParseResult",
    "parse_document",
    "parse_ocr_document",
    "parse_page_index_document",
    "parse_tree_document",
]
