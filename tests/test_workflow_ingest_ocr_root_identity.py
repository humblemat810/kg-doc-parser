"""Root-run identity and source-revision validation tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from _kogwistar_test_helpers import build_workflow_engine_triplet
from ocr_workflow_support import draw_test_image as _draw_test_image
from ocr_workflow_support import fake_ocr_response as _fake_ocr_response
from ocr_workflow_support import scratch_path as _scratch

from kg_doc_parser.models import OCRClusterResponse
from kg_doc_parser.workflow_ingest import (
    EmbeddingProviderConfig,
    OCRImagePayload,
    ProviderEndpointConfig,
    WorkflowProviderSettings,
    run_ocr_ingest_workflow,
)
from kg_doc_parser.workflow_ingest.ocr_pipeline import OCRQualityPolicy
from kg_doc_parser.workflow_ingest.semantics import HydratedTextPointer, SemanticNode

pytestmark = [pytest.mark.workflow]


def _grounded_semantic_tree(*, collection, parser_input_dict, parser_source_map):
    del parser_input_dict
    root = SemanticNode(
        title=collection.title,
        node_type="DOCUMENT_ROOT",
        total_content_pointers=[],
        child_nodes=[],
        level_from_root=0,
    )
    for unit_id, record in parser_source_map.items():
        if not record.get("participates_in_semantic_text", True):
            continue
        text = record["text"]
        root.child_nodes.append(
            SemanticNode(
                title=f"section:{unit_id}",
                node_type="TEXT_FLOW",
                total_content_pointers=[
                    HydratedTextPointer(
                        source_cluster_id=unit_id,
                        start_char=0,
                        end_char=max(0, len(text) - 1),
                        verbatim_text=text,
                    )
                ],
                child_nodes=[],
                level_from_root=1,
                parent_id=root.node_id,
            )
        )
    return root


@pytest.mark.ci
def test_ocr_root_resume_rejects_changed_provider_before_replaying_pages() -> None:
    scratch = _scratch("ocr_root_resume_identity")
    payloads: list[OCRImagePayload] = []
    for page_number in (1, 2):
        image_path = scratch / "source" / f"page-{page_number}.png"
        _draw_test_image(image_path, lines=[f"Identity page {page_number}"])
        payloads.append(
            OCRImagePayload(page_number=page_number, image_path=str(image_path))
        )
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="identity-ocr-v1"),
        parser=ProviderEndpointConfig(provider="fake", model="identity-parser"),
        embedding=EmbeddingProviderConfig(
            provider="fake", model="identity-embed", dimension=1
        ),
        triage_enabled=False,
    )
    changed_settings = settings.model_copy(
        update={"ocr": settings.ocr.model_copy(update={"model": "identity-ocr-v2"})}
    )
    engines = build_workflow_engine_triplet(scratch / "engines", "in_memory")
    calls: list[str] = []

    def _interrupting_runner(
        image_path: Path,
        page_number: int,
        provider_settings: WorkflowProviderSettings,
    ) -> OCRClusterResponse:
        del image_path, provider_settings
        calls.append(f"v1:{page_number}")
        if page_number == 2:
            raise RuntimeError("simulated interruption")
        return _fake_ocr_response(page_number, f"Identity page {page_number}")

    with pytest.raises(RuntimeError, match="OCR root workflow"):
        run_ocr_ingest_workflow(
            document_id="ocr-root-identity-doc",
            title="OCR Root Identity",
            output_dir=scratch / "artifacts",
            image_payloads=payloads,
            provider_settings=settings,
            workflow_engine=engines[0],
            conversation_engine=engines[1],
            knowledge_engine=engines[2],
            ocr_runner=_interrupting_runner,
            source_namespace="workspace-a",
            run_id="ocr|ocr-root-identity",
        )
    assert calls == ["v1:1", "v1:2"]

    def _unexpected_runner(
        image_path: Path,
        page_number: int,
        provider_settings: WorkflowProviderSettings,
    ) -> OCRClusterResponse:
        raise AssertionError(
            f"provider replayed before identity validation: {image_path}:{page_number}"
        )

    with pytest.raises(RuntimeError, match="identity mismatch"):
        run_ocr_ingest_workflow(
            document_id="ocr-root-identity-doc",
            title="OCR Root Identity",
            output_dir=scratch / "artifacts",
            image_payloads=payloads,
            provider_settings=changed_settings,
            workflow_engine=engines[0],
            conversation_engine=engines[1],
            knowledge_engine=engines[2],
            ocr_runner=_unexpected_runner,
            source_namespace="workspace-a",
            run_id="ocr|ocr-root-identity",
            resume_from_checkpoint=True,
        )


@pytest.mark.ci
def test_fresh_process_root_resume_rejects_each_output_affecting_ocr_setting() -> None:
    """Changed OCR identity is rejected before fresh-process page replay."""
    scratch = _scratch("ocr_root_config_fingerprint_fresh_process")
    for page_number in (1, 2):
        _draw_test_image(
            scratch / "source" / f"page-{page_number}.png",
            lines=[f"provider identity page {page_number}"],
        )
    worker = Path(__file__).with_name("_ocr_root_fresh_process_worker.py")
    repo_root = Path(__file__).resolve().parents[1]
    child_env = dict(os.environ)
    child_env["OCR_TEST_LEASE_MS"] = "50"
    child_env["PYTHONPATH"] = os.pathsep.join(
        path
        for path in (
            str(repo_root),
            str(repo_root.parent / "kogwistar"),
            child_env.get("PYTHONPATH", ""),
        )
        if path
    )
    interrupted = subprocess.run(
        [sys.executable, str(worker), str(scratch), "durable-config-interrupt"],
        capture_output=True,
        text=True,
        check=False,
        env=child_env,
        timeout=120,
    )
    assert interrupted.returncode == 17, interrupted.stderr
    interruption = json.loads(
        (scratch / "interrupt-result.json").read_text(encoding="utf-8")
    )
    assert interruption["calls"] == [1, 2]

    for setting_name in (
        "temperature",
        "reasoning-effort",
        "location",
        "output-tokens",
        "engine",
        "language",
        "retries",
        "timeout",
        "fallback",
    ):
        resumed = subprocess.run(
            [
                sys.executable,
                str(worker),
                str(scratch),
                f"durable-config-resume-{setting_name}",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=child_env,
            timeout=120,
        )
        assert resumed.returncode != 0, resumed.stdout
        error_path = scratch / f"durable-config-resume-{setting_name}-error.txt"
        assert error_path.is_file(), f"worker did not write {error_path.name}"
        error = error_path.read_text(encoding="utf-8")
        assert "identity mismatch" in error.lower(), error
        assert "provider was called before root identity validation" not in error


@pytest.mark.ci
def test_ocr_root_resume_rebuilds_after_same_path_source_change() -> None:
    """A changed source revision rebuilds instead of reusing accepted pages."""
    scratch = _scratch("ocr_root_resume_source_revision")
    payloads: list[OCRImagePayload] = []
    for page_number in (1, 2):
        image_path = scratch / "source" / f"page-{page_number}.png"
        _draw_test_image(image_path, lines=[f"Source revision page {page_number}"])
        payloads.append(
            OCRImagePayload(page_number=page_number, image_path=str(image_path))
        )
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="source-revision-ocr"),
        parser=ProviderEndpointConfig(provider="fake", model="source-revision-parser"),
        embedding=EmbeddingProviderConfig(
            provider="fake", model="source-revision-embed", dimension=1
        ),
        triage_enabled=False,
    )
    engines = build_workflow_engine_triplet(scratch / "engines", "in_memory")
    calls: list[int] = []

    def _interrupting_runner(
        image_path: Path,
        page_number: int,
        provider_settings: WorkflowProviderSettings,
    ) -> OCRClusterResponse:
        del image_path, provider_settings
        calls.append(page_number)
        if page_number == 2:
            raise RuntimeError("simulated interruption")
        return _fake_ocr_response(page_number, f"Source revision page {page_number}")

    with pytest.raises(RuntimeError, match="OCR root workflow"):
        run_ocr_ingest_workflow(
            document_id="ocr-root-source-revision-doc",
            title="OCR Root Source Revision",
            output_dir=scratch / "artifacts",
            image_payloads=payloads,
            provider_settings=settings,
            workflow_engine=engines[0],
            conversation_engine=engines[1],
            knowledge_engine=engines[2],
            ocr_runner=_interrupting_runner,
            quality_policy=OCRQualityPolicy(mode="permissive"),
            deps={"parse_semantic_fn": _grounded_semantic_tree},
            run_id="ocr|ocr-root-source-revision",
        )
    assert calls == [1, 2]

    _draw_test_image(
        Path(payloads[0].image_path), lines=["Changed page 1 in source revision two"]
    )

    def _revision_two_runner(
        image_path: Path,
        page_number: int,
        provider_settings: WorkflowProviderSettings,
    ) -> OCRClusterResponse:
        del provider_settings
        calls.append(page_number)
        page_text = (
            "Changed page 1 in source revision two"
            if page_number == 1
            else "Source revision page 2"
        )
        assert Path(image_path).read_bytes()
        return _fake_ocr_response(page_number, page_text)

    result, bundle, _ = run_ocr_ingest_workflow(
        document_id="ocr-root-source-revision-doc",
        title="OCR Root Source Revision",
        output_dir=scratch / "artifacts",
        image_payloads=payloads,
        provider_settings=settings,
        workflow_engine=engines[0],
        conversation_engine=engines[1],
        knowledge_engine=engines[2],
        ocr_runner=_revision_two_runner,
        quality_policy=OCRQualityPolicy(mode="permissive"),
        deps={"parse_semantic_fn": _grounded_semantic_tree},
        run_id="ocr|ocr-root-source-revision",
        resume_from_checkpoint=True,
    )
    assert result.status == "succeeded"
    assert calls == [1, 2, 1, 2], "the new source revision reused a prior OCR page"
    assert bundle is not None and bundle.grouping_tree is not None
    source_items = bundle.grouping_tree["source_items"]
    page_one_text = [
        item.get("text")
        for item in source_items
        if isinstance(item, dict)
        and isinstance(item.get("location"), dict)
        and item["location"].get("page_index") == 0
    ]
    assert "Changed page 1 in source revision two" in page_one_text
    assert "Source revision page 1" not in page_one_text
