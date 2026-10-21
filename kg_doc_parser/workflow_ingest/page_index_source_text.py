"""Convert plain text or Markdown input into normalized PageIndex pages."""

from __future__ import annotations

import re

from .models import (
    NormalizedPage,
    NormalizedSourceCollection,
    SourceUnit,
    WorkflowIngestInput,
)
from .page_index_contracts import PageIndexSourceFormat


def _split_pages(raw_text: str) -> list[str]:
    """Split a logical document into page-sized chunks."""

    pages = re.split(r"\f|^\s*--- PAGE BREAK ---\s*$", raw_text, flags=re.MULTILINE)
    return [page.strip("\n") for page in pages if page.strip()]


def build_page_index_workflow_input(
    *,
    document_id: str,
    title: str,
    raw_text: str,
    source_format: PageIndexSourceFormat,
) -> WorkflowIngestInput:
    pages = _split_pages(raw_text)
    normalized_pages: list[NormalizedPage] = []
    for page_number, page_text in enumerate(pages, start=1):
        normalized_pages.append(
            NormalizedPage(
                page_number=page_number,
                units=[
                    SourceUnit(
                        modality="text",
                        page_number=page_number,
                        cluster_number=0,
                        text=page_text,
                        embedding_space="default_text",
                        metadata={"source_format": source_format},
                    )
                ],
                metadata={"source_format": source_format},
            )
        )
    return WorkflowIngestInput(
        request_id=document_id,
        collections=[
            NormalizedSourceCollection(
                collection_id=document_id,
                title=title,
                modality="text",
                pages=normalized_pages,
                embedding_spaces=["default_text"],
                metadata={"source_format": source_format},
            )
        ],
    )
