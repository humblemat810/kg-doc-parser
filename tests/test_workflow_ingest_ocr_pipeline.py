"""Workflow-first OCR ingest tests.

This file is intentionally split into two layers:

- small CI-safe tests that validate resume, legacy artifact compatibility, and
  PDF raster handoff without needing a live model
- manual OCR matrix cases that are meant to be clicked from VS Code and then
  inspected on disk

For the manual cases, inspect these artifacts under the stable case directory:

- `source/`
- `artifacts/rendered_pages/`
- `artifacts/legacy_split_pages/<document>/page_N.json`
- `artifacts/ocr-state.sqlite`
- `artifacts/ocr-progress.json`
- `artifacts/ocr-summary.json`
- `workflow-events.jsonl`

If a local OCR model returns stale or low-quality output, delete that specific
manual case directory and rerun the clicked case so the workflow regenerates
the per-page OCR artifacts from scratch.

Manual OCR matrix flow
----------------------

Image case:

    source/page_1.png + source/page_2.png
        -> OCR model call per page
        -> artifacts/rendered_pages/<document>/
        -> artifacts/legacy_split_pages/<document>/page_N.json
        -> artifacts/ocr-state.sqlite
        -> artifacts/ocr-progress.json
        -> normalize_ocr_pages(...)
        -> run_ingest_workflow(...)
        -> artifacts/ocr-summary.json

PDF case:

    source/manual_source.pdf
        -> render PDF page-by-page into artifacts/rendered_pages/<document>-<pdf-stem>-<digest>/
        -> OCR model call per rendered page
        -> artifacts/legacy_split_pages/<document>-<pdf-stem>-<digest>/page_N.json
        -> artifacts/ocr-state.sqlite
        -> artifacts/ocr-progress.json
        -> normalize_ocr_pages(...)
        -> run_ingest_workflow(...)
        -> artifacts/ocr-summary.json

Both manual matrix variants end in the same normalized workflow-ingest path.

"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from _kogwistar_test_helpers import (
    build_workflow_engine_triplet,
    drain_phase1_indexes_until_idle,
)
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.runtime.runtime import apply_state_update_inplace
from ocr_workflow_support import (
    draw_test_image as _draw_test_image,
)
from ocr_workflow_support import (
    fake_ocr_response as _fake_ocr_response,
)
from ocr_workflow_support import (
    manual_case_dir as _manual_case_dir,
)
from ocr_workflow_support import (
    scratch_path as _scratch,
)
from ocr_workflow_support import (
    write_two_page_pdf as _write_two_page_pdf,
)
from PIL import Image, ImageDraw
from pypdf import PdfReader, PdfWriter

import kg_doc_parser.workflow_ingest.ocr_finalization as ocr_finalization_module
import kg_doc_parser.workflow_ingest.ocr_page_ingestion as ocr_page_ingestion_module
import kg_doc_parser.workflow_ingest.ocr_pipeline as ocr_pipeline_module
import kg_doc_parser.workflow_ingest.ocr_runtime as ocr_runtime_module
import kg_doc_parser.workflow_ingest.ocr_runtime_finalization as ocr_runtime_finalization_module
import kg_doc_parser.workflow_ingest.ocr_runtime_grouping as ocr_runtime_grouping_module
import kg_doc_parser.workflow_ingest.ocr_runtime_grouping_steps as ocr_runtime_grouping_steps_module
import kg_doc_parser.workflow_ingest.ocr_state_store as ocr_state_store_module
import kg_doc_parser.workflow_ingest.strategy as strategy_module
from kg_doc_parser.models import (
    NonTextCluster,
    OCRClusterResponse,
    SplitPage,
    SplitPageMeta,
    TextCluster,
)
from kg_doc_parser.ocr import RawOCRResponse
from kg_doc_parser.workflow_ingest import (
    EmbeddingProviderConfig,
    OCRImagePayload,
    ProviderEndpointConfig,
    WorkflowProviderSettings,
    prepare_ocr_workflow_input,
    run_ocr_ingest_workflow,
)
from kg_doc_parser.workflow_ingest.document_adapters import apply_grouping_proposals
from kg_doc_parser.workflow_ingest.document_grouping import (
    GroupingPassContext,
    GroupProposal,
    GroupProposalBatch,
    GroupProposalGenerator,
    build_grouping_pass_context,
    grouping_policy_identity,
)
from kg_doc_parser.workflow_ingest.document_models import (
    AdaptiveGroupingPolicy,
    BoundingBox,
    ChildRef,
    DocumentNode,
    DocumentTree,
    EvidenceSlice,
    MediaRef,
    SourceItem,
    SourceLocation,
    SourceSlice,
    Summary,
    SummaryPolicy,
)
from kg_doc_parser.workflow_ingest.ocr_pipeline import (
    GroundedChatGroupProposalGenerator,
    OCRPageOutcome,
    OCRPageProcessingContext,
    OCRQualityPolicy,
    OCRSourcePlan,
    OCRWorkflowArtifacts,
    PDFRasterizer,
    _compute_input_fingerprint,
    _generate_tree_summaries,
    _ocr_configuration_fingerprint,
    _ocr_page_metadata,
    _process_ocr_page,
    _provider_endpoint_configuration_identity,
    _write_json_atomic,
)
from kg_doc_parser.workflow_ingest.ocr_runtime import (
    run_grouping_workflow,
    run_ocr_page_workflow,
)
from kg_doc_parser.workflow_ingest.ocr_runtime_page_state import (
    page_workflow_initial_state as _page_workflow_initial_state,
)
from kg_doc_parser.workflow_ingest.ocr_runtime_page_state import (
    reconcile_page_checkpoint as _reconcile_page_checkpoint,
)
from kg_doc_parser.workflow_ingest.ocr_runtime_page_state import (
    reused_pages_from_trace as _reused_pages_from_trace,
)
from kg_doc_parser.workflow_ingest.ocr_runtime_state import json_pages as _json_pages
from kg_doc_parser.workflow_ingest.ocr_state_store import (
    GroupingCheckpointConflict,
    OCRWorkflowStateStore,
)
from kg_doc_parser.workflow_ingest.providers import SupportsStructuredOutput
from kg_doc_parser.workflow_ingest.semantics import HydratedTextPointer, SemanticNode
from kg_doc_parser.workflow_ingest.service import (
    _TinyEmbeddingFunction,
)

pytestmark = [pytest.mark.workflow]


_LOGGER = logging.getLogger(__name__)


def _terminate_interrupted_worker(process: subprocess.Popen[str]) -> None:
    """Use the strongest available process-stop primitive for recovery tests."""
    if os.name == "posix":
        os.kill(process.pid, signal.SIGKILL)
    else:
        process.terminate()


@pytest.mark.ci
def test_grouping_runs_inside_kogwistar_runtime_and_is_resume_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = _scratch("ocr_grouping_runtime")
    workflow_engine, conversation_engine, _knowledge_engine = (
        build_workflow_engine_triplet(
            scratch / "engines",
            "in_memory",
        )
    )
    tree = DocumentTree(
        document_id="grouping-runtime-doc",
        source_revision="ocr-v1:source-digest",
        tree_revision=1,
        root_id="root",
        source_items=(
            SourceItem(
                item_id="item-a",
                kind="text",
                text="alpha",
                location=SourceLocation(page_index=0, accuracy="page"),
            ),
            SourceItem(
                item_id="item-b",
                kind="text",
                text="beta",
                location=SourceLocation(page_index=0, accuracy="page"),
            ),
            SourceItem(
                item_id="item-c",
                kind="text",
                text="gamma",
                location=SourceLocation(page_index=0, accuracy="page"),
            ),
        ),
        nodes=(
            DocumentNode(
                node_id="leaf:item-a",
                kind="text",
                parts=(SourceSlice(item_id="item-a", start=0, end=5),),
            ),
            DocumentNode(
                node_id="leaf:item-b",
                kind="text",
                parts=(SourceSlice(item_id="item-b", start=0, end=4),),
            ),
            DocumentNode(
                node_id="leaf:item-c",
                kind="text",
                parts=(SourceSlice(item_id="item-c", start=0, end=5),),
            ),
            DocumentNode(
                node_id="root",
                kind="document",
                children=(
                    ChildRef(node_id="leaf:item-a"),
                    ChildRef(node_id="leaf:item-b"),
                    ChildRef(node_id="leaf:item-c"),
                ),
            ),
        ),
    )

    def _summary_generator(
        *,
        tree: DocumentTree,
        node_id: str,
        source_items: tuple[SourceItem, ...],
        input_fingerprint: str,
    ) -> Summary:
        assert node_id == tree.root_id
        assert any(node.kind == "section" for node in tree.nodes)
        return Summary(
            status="generated",
            text=f"summary:{node_id}",
            evidence_item_ids=tuple(item.item_id for item in source_items),
            provider_model="fake-summary",
            prompt_revision="summary-v1",
            input_fingerprint=input_fingerprint,
        )

    summary_policy = SummaryPolicy(
        eligible_kinds=("document",),
        minimum_source_items=1,
        minimum_source_chars=1,
        generation_fingerprint="fake-summary-v1",
    )
    grouped, run_id = run_grouping_workflow(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        state_store=OCRWorkflowStateStore(scratch / "ocr-state.sqlite"),
        tree=tree,
        document_id=tree.document_id,
        grouping_policy_version="test-v1",
        grouping_proposals=(
            GroupProposal(kind="section", members=("leaf:item-a", "leaf:item-b")),
        ),
        summary_policy=summary_policy,
        summary_generator=_summary_generator,
        run_id="ocr-grouping-test-run",
    )
    assert run_id == "ocr-grouping-test-run"
    assert any(node.kind == "section" for node in grouped.nodes)

    def _unexpected_grouping_replay(*args: object, **kwargs: object) -> DocumentTree:
        del args, kwargs
        raise AssertionError("terminal grouping checkpoint was replayed")

    with monkeypatch.context() as resumed_patch:
        resumed_patch.setattr(
            ocr_runtime_grouping_steps_module,
            "apply_grouping_proposals",
            _unexpected_grouping_replay,
        )
        resumed, resumed_run_id = run_grouping_workflow(
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            state_store=OCRWorkflowStateStore(scratch / "ocr-state.sqlite"),
            tree=tree,
            document_id=tree.document_id,
            grouping_policy_version="test-v1",
            grouping_proposals=(
                GroupProposal(kind="section", members=("leaf:item-a", "leaf:item-b")),
            ),
            summary_policy=summary_policy,
            summary_generator=_summary_generator,
            run_id=run_id,
            resume_from_checkpoint=True,
        )
    assert resumed_run_id == run_id
    assert resumed == grouped
    trace = list(
        conversation_engine.read.get_nodes(
            where={"$and": [{"entity_type": "workflow_step_exec"}, {"run_id": run_id}]}
        )
    )
    assert {str(node.metadata.get("op")) for node in trace} >= {
        "start",
        "restore_grouping_checkpoint",
        "apply_grouping_pass",
        "generate_grouping_summaries",
        "persist_grouping_result",
        "end",
    }

    # Exercise the two-store crash seam directly: the SQLite grouping commit
    # succeeds, but the process dies before Kogwistar checkpoints the step.
    crash_scratch = _scratch("ocr_grouping_sqlite_commit_before_runtime_checkpoint")
    crash_workflow, crash_conversation, _ = build_workflow_engine_triplet(
        crash_scratch / "engines",
        "in_memory",
    )
    crash_store = OCRWorkflowStateStore(crash_scratch / "ocr-state.sqlite")
    original_save = crash_store.save_document_tree
    crashed = False

    def _save_then_crash(*args: object, **kwargs: object) -> str:
        nonlocal crashed
        revision = original_save(*args, **kwargs)  # type: ignore[arg-type]
        if not crashed:
            crashed = True
            raise RuntimeError("simulated crash after SQLite grouping commit")
        return revision

    monkeypatch.setattr(crash_store, "save_document_tree", _save_then_crash)
    with pytest.raises(RuntimeError, match="OCR grouping workflow"):
        run_grouping_workflow(
            workflow_engine=crash_workflow,
            conversation_engine=crash_conversation,
            state_store=crash_store,
            tree=tree,
            document_id=tree.document_id,
            grouping_policy_version="test-v1",
            grouping_proposals=(
                GroupProposal(kind="section", members=("leaf:item-a", "leaf:item-b")),
            ),
            run_id="ocr-grouping-sqlite-commit-crash",
        )
    resumed_after_crash, _ = run_grouping_workflow(
        workflow_engine=crash_workflow,
        conversation_engine=crash_conversation,
        state_store=OCRWorkflowStateStore(crash_scratch / "ocr-state.sqlite"),
        tree=tree,
        document_id=tree.document_id,
        grouping_policy_version="test-v1",
        grouping_proposals=(
            GroupProposal(kind="section", members=("leaf:item-a", "leaf:item-b")),
        ),
        run_id="ocr-grouping-sqlite-commit-crash",
        resume_from_checkpoint=True,
    )
    expected_after_crash = apply_grouping_proposals(
        tree,
        (GroupProposal(kind="section", members=("leaf:item-a", "leaf:item-b")),),
        grouping_policy_version="test-v1:1",
    )
    assert resumed_after_crash == expected_after_crash


@pytest.mark.ci
def test_adaptive_grouping_iterates_in_kogwistar_and_reuses_completed_tree() -> None:
    scratch = _scratch("adaptive_grouping_runtime")
    workflow_engine, conversation_engine, _knowledge_engine = (
        build_workflow_engine_triplet(
            scratch / "engines",
            "in_memory",
        )
    )
    tree = DocumentTree(
        document_id="adaptive-runtime-doc",
        source_revision="ocr-v1:source-digest",
        tree_revision=1,
        root_id="root",
        source_items=tuple(
            SourceItem(
                item_id=f"item-{suffix}",
                kind="text",
                text=suffix,
                location=SourceLocation(page_index=0, accuracy="page"),
            )
            for suffix in ("alpha", "beta", "gamma", "delta", "epsilon")
        ),
        nodes=(
            DocumentNode(
                node_id="leaf:alpha",
                kind="paragraph",
                parts=(SourceSlice(item_id="item-alpha", start=0, end=5),),
            ),
            DocumentNode(
                node_id="leaf:beta",
                kind="paragraph",
                parts=(SourceSlice(item_id="item-beta", start=0, end=4),),
            ),
            DocumentNode(
                node_id="leaf:gamma",
                kind="paragraph",
                parts=(SourceSlice(item_id="item-gamma", start=0, end=5),),
            ),
            DocumentNode(
                node_id="leaf:delta",
                kind="paragraph",
                parts=(SourceSlice(item_id="item-delta", start=0, end=5),),
            ),
            DocumentNode(
                node_id="leaf:epsilon",
                kind="paragraph",
                parts=(SourceSlice(item_id="item-epsilon", start=0, end=7),),
            ),
            DocumentNode(
                node_id="root",
                kind="document",
                children=tuple(
                    ChildRef(node_id=f"leaf:{suffix}")
                    for suffix in ("alpha", "beta", "gamma", "delta", "epsilon")
                ),
            ),
        ),
    )
    policy = AdaptiveGroupingPolicy(
        maximum_rounds=3,
        window_size=4,
        window_overlap=2,
    )
    seen_contexts: list[GroupingPassContext] = []

    class _Proposer:
        configuration_fingerprint = "adaptive-runtime-proposer-v1"

        def __call__(self, context: GroupingPassContext) -> GroupProposalBatch:
            seen_contexts.append(context)
            if context.pass_index == 0 and context.frontier_ids[0] == "leaf:alpha":
                return GroupProposalBatch(
                    proposals=(
                        GroupProposal(
                            kind="section",
                            members=("leaf:alpha", "leaf:beta"),
                        ),
                    )
                )
            return GroupProposalBatch()

    proposer: GroupProposalGenerator = _Proposer()
    state_store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    grouped, run_id = run_grouping_workflow(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        state_store=state_store,
        tree=tree,
        document_id=tree.document_id,
        grouping_policy_version="adaptive-runtime-v1",
        adaptive_grouping_policy=policy,
        grouping_proposer=proposer,
        run_id="adaptive-grouping-run",
    )

    assert run_id == "adaptive-grouping-run"
    assert len(seen_contexts) == 3
    assert seen_contexts[0].frontier_ids == (
        "leaf:alpha",
        "leaf:beta",
        "leaf:gamma",
        "leaf:delta",
    )
    assert seen_contexts[1].frontier_ids == (
        "leaf:gamma",
        "leaf:delta",
        "leaf:epsilon",
    )
    assert seen_contexts[0].pass_index == seen_contexts[1].pass_index == 0
    assert seen_contexts[2].pass_index == 1
    assert seen_contexts[2].frontier_ids[0].startswith("merge:")
    assert len(seen_contexts[2].frontier_ids) == 4
    expected_identity = grouping_policy_identity(
        policy_version="adaptive-runtime-v1",
        policy=policy,
        proposer_fingerprint=proposer.configuration_fingerprint,
    )
    assert (
        grouping_policy_identity(
            policy_version="adaptive-runtime-v1",
            policy=policy,
            proposer_fingerprint=proposer.configuration_fingerprint,
            summary_fingerprint="grounded-summary-config-v1",
        )
        != expected_identity
    )
    root = next(node for node in grouped.nodes if node.node_id == grouped.root_id)
    assert root.annotations["grouping:adaptive_complete"] == expected_identity
    step_events = list(
        conversation_engine.read.get_nodes(
            where={
                "$and": [
                    {"entity_type": "workflow_step_exec"},
                    {"run_id": run_id},
                    {"op": "apply_grouping_pass"},
                ]
            }
        )
    )
    assert len(step_events) >= 6

    class _UnexpectedProviderCall:
        configuration_fingerprint = "adaptive-runtime-proposer-v1"

        def __call__(self, context: GroupingPassContext) -> GroupProposalBatch:
            del context
            raise AssertionError(
                "completed adaptive grouping called its provider again"
            )

    restored, _restored_run_id = run_grouping_workflow(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        state_store=OCRWorkflowStateStore(scratch / "ocr-state.sqlite"),
        tree=tree,
        document_id=tree.document_id,
        grouping_policy_version="adaptive-runtime-v1",
        adaptive_grouping_policy=policy,
        grouping_proposer=_UnexpectedProviderCall(),
        run_id="adaptive-grouping-repair-incarnation",
    )
    assert restored == grouped


@pytest.mark.ci
def test_group_proposer_fingerprint_uses_provider_config_without_mode_dump() -> None:
    base = ProviderEndpointConfig(
        provider="openai",
        model="group-model",
        base_url="https://router.example/v1",
        reasoning_effort="none",
        api_key_env="PARSER_API_KEY",
    )
    settings = WorkflowProviderSettings(parser=base)
    generator = GroundedChatGroupProposalGenerator(
        provider_settings=settings,
        chat_model=cast(SupportsStructuredOutput, object()),
    )

    assert (
        generator.configuration_fingerprint
        == GroundedChatGroupProposalGenerator(
            provider_settings=WorkflowProviderSettings(
                parser=base.model_copy(deep=True)
            ),
            chat_model=cast(SupportsStructuredOutput, object()),
        ).configuration_fingerprint
    )
    assert (
        generator.configuration_fingerprint
        != GroundedChatGroupProposalGenerator(
            provider_settings=WorkflowProviderSettings(
                parser=base.model_copy(update={"reasoning_effort": "high"})
            ),
            chat_model=cast(SupportsStructuredOutput, object()),
        ).configuration_fingerprint
    )
    identity = _provider_endpoint_configuration_identity(base)
    assert identity["api_key_env"] == "PARSER_API_KEY"
    assert "api_key" not in identity


def test_production_group_proposer_uses_typed_structured_output_payload() -> None:
    tree = DocumentTree(
        document_id="adaptive-provider-doc",
        source_revision="ocr-v1:provider-source",
        tree_revision=1,
        root_id="root",
        source_items=(
            SourceItem(
                item_id="item-a",
                kind="text",
                text="alpha",
                location=SourceLocation(page_index=0, accuracy="page"),
            ),
            SourceItem(
                item_id="item-b",
                kind="figure",
                media=MediaRef(
                    uri="memory://adaptive-provider-figure",
                    mime_type="image/png",
                    sha256_hex="a" * 64,
                    byte_length=12,
                ),
                location=SourceLocation(
                    page_index=1,
                    printed_page_label="2",
                    bbox=BoundingBox(
                        x0=0.1,
                        y0=0.2,
                        x1=0.8,
                        y1=0.9,
                        coordinate_system="relative",
                    ),
                    accuracy="region",
                ),
            ),
        ),
        nodes=(
            DocumentNode(
                node_id="leaf-a",
                kind="paragraph",
                parts=(SourceSlice(item_id="item-a", start=0, end=5),),
            ),
            DocumentNode(
                node_id="leaf-b",
                kind="paragraph",
                parts=(SourceSlice(item_id="item-b"),),
            ),
            DocumentNode(
                node_id="root",
                kind="document",
                children=(ChildRef(node_id="leaf-a"), ChildRef(node_id="leaf-b")),
            ),
        ),
    )
    context = build_grouping_pass_context(tree, pass_index=0)
    captured: dict[str, object] = {}

    class _Runnable:
        def __init__(self) -> None:
            self.steps: list[object] = []

        def invoke(
            self,
            messages: object,
            config: object | None = None,
            **kwargs: object,
        ) -> dict[str, object]:
            del config, kwargs
            captured["messages"] = messages
            return {
                "parsed": {
                    "proposals": [{"kind": "section", "members": ["leaf-a", "leaf-b"]}]
                }
            }

    class _Chat:
        def with_structured_output(self, schema: object, **kwargs: object) -> _Runnable:
            captured["schema"] = schema
            captured["structured_options"] = kwargs
            return _Runnable()

    settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(provider="fake", model="fixture-parser")
    )
    generator = GroundedChatGroupProposalGenerator(
        provider_settings=settings,
        chat_model=_Chat(),
    )

    batch = generator(context)

    assert batch.proposals == (
        GroupProposal(kind="section", members=("leaf-a", "leaf-b")),
    )
    assert captured["schema"] is GroupProposalBatch
    assert captured["structured_options"] == {
        "include_raw": True,
        "method": "json_schema",
    }
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert "untrusted data" in messages[0].content
    assert "do not claim unseen visual or audio content" in messages[0].content
    assert "leaf-a" in messages[1].content
    assert '"modality":"image"' in messages[1].content
    assert '"media_mime_type":"image/png"' in messages[1].content
    assert '"printed_page_label":"2"' in messages[1].content
    assert '"coordinate_system":"relative"' in messages[1].content


@pytest.mark.ci
def test_adaptive_grouping_resume_continues_at_uncheckpointed_window() -> None:
    scratch = _scratch("adaptive_grouping_window_resume")
    workflow_engine, conversation_engine, _knowledge_engine = (
        build_workflow_engine_triplet(
            scratch / "engines",
            "in_memory",
        )
    )
    suffixes = ("alpha", "beta", "gamma", "delta", "epsilon")
    tree = DocumentTree(
        document_id="adaptive-window-resume-doc",
        source_revision="ocr-v1:source-digest",
        tree_revision=1,
        root_id="root",
        source_items=tuple(
            SourceItem(
                item_id=f"item-{suffix}",
                kind="text",
                text=suffix,
                location=SourceLocation(page_index=0, accuracy="page"),
            )
            for suffix in suffixes
        ),
        nodes=(
            *tuple(
                DocumentNode(
                    node_id=f"leaf:{suffix}",
                    kind="paragraph",
                    parts=(
                        SourceSlice(
                            item_id=f"item-{suffix}",
                            start=0,
                            end=len(suffix),
                        ),
                    ),
                )
                for suffix in suffixes
            ),
            DocumentNode(
                node_id="root",
                kind="document",
                children=tuple(ChildRef(node_id=f"leaf:{name}") for name in suffixes),
            ),
        ),
    )
    policy = AdaptiveGroupingPolicy(maximum_rounds=3, window_size=4, window_overlap=2)
    first_window = tuple(f"leaf:{name}" for name in suffixes[:4])
    second_window = tuple(f"leaf:{name}" for name in suffixes[2:])
    seen: list[GroupingPassContext] = []

    class _FailOnceOnSecondWindow:
        configuration_fingerprint = "adaptive-window-resume-v1"

        def __init__(self) -> None:
            self.failed = False

        def __call__(self, context: GroupingPassContext) -> GroupProposalBatch:
            seen.append(context)
            if context.pass_index == 0 and context.frontier_ids == first_window:
                return GroupProposalBatch(
                    proposals=(GroupProposal(kind="section", members=first_window[:2]),)
                )
            if (
                context.pass_index == 0
                and context.frontier_ids == second_window
                and not self.failed
            ):
                self.failed = True
                raise RuntimeError("simulated provider interruption on second window")
            return GroupProposalBatch()

    proposer = _FailOnceOnSecondWindow()
    state_store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    run_id = "adaptive-window-resume-run"
    with pytest.raises(RuntimeError, match="OCR grouping workflow failure"):
        run_grouping_workflow(
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            state_store=state_store,
            tree=tree,
            document_id=tree.document_id,
            grouping_policy_version="adaptive-window-resume-v1",
            adaptive_grouping_policy=policy,
            grouping_proposer=proposer,
            run_id=run_id,
        )

    resumed, resumed_run_id = run_grouping_workflow(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        state_store=OCRWorkflowStateStore(scratch / "ocr-state.sqlite"),
        tree=tree,
        document_id=tree.document_id,
        grouping_policy_version="adaptive-window-resume-v1",
        adaptive_grouping_policy=policy,
        grouping_proposer=proposer,
        run_id=run_id,
        resume_from_checkpoint=True,
    )

    assert resumed_run_id == run_id
    assert [context.frontier_ids for context in seen[:3]] == [
        first_window,
        second_window,
        second_window,
    ]
    assert seen[0].pass_index == seen[1].pass_index == seen[2].pass_index == 0
    assert seen[3].pass_index == 1
    assert any(node.kind == "section" for node in resumed.nodes)


def test_workflow_runtime_compatibility_rejects_unsupported_lease_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovery must not silently downgrade on a legacy Kogwistar runtime."""

    class LegacyRuntime:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(ocr_runtime_module, "WorkflowRuntime", LegacyRuntime)
    with pytest.raises(
        ocr_runtime_module.KogwistarRuntimeCompatibilityError,
        match="does not support run_ownership_lease_ms",
    ):
        ocr_runtime_module._workflow_runtime(
            workflow_engine=object(),
            conversation_engine=object(),
            step_resolver=object(),
            predicate_registry={},
            trace=True,
            run_ownership_lease_ms=30_000,
        )


def test_grouping_wrapper_preserves_failed_nested_workflow_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed runtime result cannot be mistaken for an accepted tree."""
    workflow_engine, conversation_engine, _knowledge_engine = (
        build_workflow_engine_triplet(
            _scratch("ocr_grouping_failed_nested_workflow") / "engines",
            "in_memory",
        )
    )
    tree = DocumentTree(
        document_id="grouping-failed-nested-workflow-doc",
        source_revision="ocr-v1:source-digest",
        tree_revision=1,
        root_id="root",
        source_items=(
            SourceItem(
                item_id="item-a",
                kind="text",
                text="alpha",
                location=SourceLocation(page_index=0, accuracy="page"),
            ),
        ),
        nodes=(
            DocumentNode(
                node_id="root",
                kind="document",
                parts=(SourceSlice(item_id="item-a", start=0, end=5),),
            ),
        ),
    )

    class _FailedRuntime:
        def run(self, **kwargs: object) -> SimpleNamespace:
            del kwargs
            return SimpleNamespace(
                status="failure",
                errors=["nested workflow lost ownership"],
                final_state={},
                run_id="ocr-grouping-failed-nested-workflow",
            )

    monkeypatch.setattr(
        ocr_runtime_grouping_module,
        "ensure_ocr_grouping_workflow_design",
        lambda _engine: None,
    )
    monkeypatch.setattr(
        ocr_runtime_grouping_module,
        "_workflow_runtime",
        lambda **_kwargs: _FailedRuntime(),
    )

    with pytest.raises(
        RuntimeError,
        match=r"OCR grouping workflow failure: \['nested workflow lost ownership'\]",
    ):
        run_grouping_workflow(
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            state_store=OCRWorkflowStateStore(
                _scratch("ocr_grouping_failed_nested_workflow") / "ocr-state.sqlite"
            ),
            tree=tree,
            document_id=tree.document_id,
            grouping_policy_version="test-v1",
            run_id="ocr-grouping-failed-nested-workflow",
        )


def _is_ownership_loss_error(error_text: str) -> bool:
    return "WorkflowRunOwnershipError" in error_text or (
        "Nested workflow 'kg_doc_parser.ocr_grouping.v1' returned status" in error_text
        and "ownership" in error_text.lower()
    )


def _is_fail_closed_resume_error(error_text: str) -> bool:
    """Recognize the stable outer contract for a rejected resume.

    A competing process may lose at the root lease or while its already-started
    nested workflow is being reconciled.  The latter is intentionally surfaced
    as a wrapped root failure rather than always preserving the concrete
    ownership exception.
    """
    return (
        "OCR root workflow failure:" in error_text
        and "Nested workflow 'kg_doc_parser.ocr_grouping.v1' returned status"
        in error_text
    )


def _assert_concurrent_resume_loser_error(error_text: str) -> None:
    assert _is_ownership_loss_error(error_text) or _is_fail_closed_resume_error(
        error_text
    )


@pytest.mark.parametrize(
    "error_text",
    [
        "WorkflowRunOwnershipError: run is owned by another worker",
        (
            "RuntimeError: OCR root workflow failure: "
            "[\"Nested workflow 'kg_doc_parser.ocr_grouping.v1' returned status 'failure'\"]"
        ),
    ],
)
def test_concurrent_resume_loser_errors_are_fail_closed(error_text: str) -> None:
    """Direct and wrapped ownership loss remain rejected, never accepted."""
    _assert_concurrent_resume_loser_error(error_text)


@pytest.mark.parametrize(
    "nested_failure",
    [
        "invalid grouping frontier",
        "provider request failed",
        "persisted checkpoint is malformed",
    ],
)
def test_concurrent_resume_unrelated_nested_failure_is_not_misclassified(
    nested_failure: str,
) -> None:
    """Ordinary nested failures must not look like ownership loss."""
    error_text = (
        f'RuntimeError: OCR root workflow failure: ["Nested workflow '
        f"'kg_doc_parser.ocr_grouping.v1' returned status "
        f"'failure: {nested_failure}'\"]"
    )
    assert not _is_ownership_loss_error(error_text)
    assert _is_fail_closed_resume_error(error_text)


@pytest.mark.parametrize(
    "error_text",
    [
        (
            'RuntimeError: OCR root workflow failure: ["Nested workflow '
            "'kg_doc_parser.ocr_grouping.v1' returned status 'failure'\"]"
        ),
        (
            'RuntimeError: OCR root workflow failure: ["Nested workflow '
            "'kg_doc_parser.ocr_grouping.v1' returned status 'failure', "
            "'grouping checkpoint is already owned'\"]"
        ),
    ],
)
def test_concurrent_resume_wrapped_failures_remain_rejected(
    error_text: str,
) -> None:
    """A wrapped child failure is still a fail-closed resume outcome."""
    assert _is_fail_closed_resume_error(error_text)


@pytest.mark.ci
def test_grouping_resume_rejects_frontier_that_disagrees_with_tree() -> None:
    scratch = _scratch("ocr_grouping_frontier_mismatch")
    workflow_engine, conversation_engine, _knowledge_engine = (
        build_workflow_engine_triplet(
            scratch / "engines",
            "in_memory",
        )
    )
    tree = DocumentTree(
        document_id="grouping-frontier-mismatch-doc",
        source_revision="ocr-v1:source-digest",
        tree_revision=1,
        root_id="root",
        source_items=(
            SourceItem(
                item_id="item-a",
                kind="text",
                text="alpha",
                location=SourceLocation(page_index=0, accuracy="page"),
            ),
            SourceItem(
                item_id="item-b",
                kind="text",
                text="beta",
                location=SourceLocation(page_index=0, accuracy="page"),
            ),
        ),
        nodes=(
            DocumentNode(
                node_id="leaf:item-a",
                kind="text",
                parts=(SourceSlice(item_id="item-a", start=0, end=5),),
            ),
            DocumentNode(
                node_id="leaf:item-b",
                kind="text",
                parts=(SourceSlice(item_id="item-b", start=0, end=4),),
            ),
            DocumentNode(
                node_id="root",
                kind="document",
                children=(
                    ChildRef(node_id="leaf:item-a"),
                    ChildRef(node_id="leaf:item-b"),
                ),
            ),
        ),
    )
    store_path = scratch / "ocr-state.sqlite"
    store = OCRWorkflowStateStore(store_path)
    run_grouping_workflow(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        state_store=store,
        tree=tree,
        document_id=tree.document_id,
        grouping_policy_version="test-v1",
        run_id="ocr-grouping-frontier-mismatch",
    )
    with sqlite3.connect(store_path) as conn:
        conn.execute(
            "UPDATE grouping_frontier_state SET frontier_json = ? WHERE document_id = ?",
            (json.dumps(["not-in-tree"]), tree.document_id),
        )

    with pytest.raises(RuntimeError, match="frontier does not match the restored tree"):
        run_grouping_workflow(
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            state_store=OCRWorkflowStateStore(store_path),
            tree=tree,
            document_id=tree.document_id,
            grouping_policy_version="test-v1",
            run_id="ocr-grouping-frontier-mismatch",
            resume_from_checkpoint=True,
        )


@pytest.mark.ci
def test_grouping_incremental_resume_matches_uninterrupted_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _tree(document_id: str) -> DocumentTree:
        source_items = tuple(
            SourceItem(
                item_id=f"item-{name}",
                kind="text",
                text=name,
                location=SourceLocation(page_index=0, accuracy="page"),
            )
            for name in ("alpha", "beta", "gamma")
        )
        return DocumentTree(
            document_id=document_id,
            source_revision="ocr-v1:incremental-source",
            tree_revision=1,
            root_id="root",
            source_items=source_items,
            nodes=tuple(
                [
                    DocumentNode(
                        node_id=f"leaf:item-{name}",
                        kind="text",
                        parts=(
                            SourceSlice(item_id=f"item-{name}", start=0, end=len(name)),
                        ),
                    )
                    for name in ("alpha", "beta", "gamma")
                ]
                + [
                    DocumentNode(
                        node_id="root",
                        kind="document",
                        children=tuple(
                            ChildRef(node_id=f"leaf:item-{name}")
                            for name in ("alpha", "beta", "gamma")
                        ),
                    )
                ]
            ),
        )

    def _summary_generator(
        *,
        tree: DocumentTree,
        node_id: str,
        source_items: tuple[SourceItem, ...],
        input_fingerprint: str,
    ) -> Summary:
        del tree
        return Summary(
            status="generated",
            text=f"summary:{node_id}",
            evidence_item_ids=tuple(item.item_id for item in source_items),
            provider_model="fake-summary",
            prompt_revision="summary-v1",
            input_fingerprint=input_fingerprint,
        )

    reference_scratch = _scratch("ocr_grouping_incremental_reference")
    interrupted_scratch = _scratch("ocr_grouping_incremental_interrupted")
    # The stores and runtime engines are separate, but equivalence must use the
    # same logical document identity because grouping node IDs are document-scoped.
    reference_tree = _tree("grouping-incremental-document")
    interrupted_tree = _tree("grouping-incremental-document")

    # This test intentionally uses two passes; the first pass's stable output
    # is the frontier input for the second pass in the production workflow.
    first = apply_grouping_proposals(
        reference_tree,
        (GroupProposal(kind="section", members=("leaf:item-alpha", "leaf:item-beta")),),
        grouping_policy_version="incremental-v1:1",
    )
    first_root = next(node for node in first.nodes if node.node_id == first.root_id)
    first_frontier = tuple(child.node_id for child in first_root.children)
    assert len(first_frontier) == 2
    passes = (
        (GroupProposal(kind="section", members=("leaf:item-alpha", "leaf:item-beta")),),
        (GroupProposal(kind="topic", members=(first_frontier[0], "leaf:item-gamma")),),
    )
    policy = SummaryPolicy(
        eligible_kinds=("document",),
        minimum_source_items=1,
        minimum_source_chars=1,
        generation_fingerprint="fake-summary-v1",
    )

    def _reference_summary(
        *,
        tree: DocumentTree,
        node_id: str,
        source_items: tuple[SourceItem, ...],
        input_fingerprint: str,
    ) -> Summary:
        return _summary_generator(
            tree=tree,
            node_id=node_id,
            source_items=source_items,
            input_fingerprint=input_fingerprint,
        )

    ref_engines = build_workflow_engine_triplet(
        reference_scratch / "engines", "in_memory"
    )
    reference, _ = run_grouping_workflow(
        workflow_engine=ref_engines[0],
        conversation_engine=ref_engines[1],
        state_store=OCRWorkflowStateStore(reference_scratch / "ocr-state.sqlite"),
        tree=reference_tree,
        document_id=reference_tree.document_id,
        grouping_policy_version="incremental-v1",
        grouping_passes=passes,
        summary_policy=policy,
        summary_generator=_reference_summary,
        run_id="ocr-grouping-incremental-reference",
    )

    original_generate_tree_summaries = (
        ocr_runtime_grouping_steps_module._generate_tree_summaries
    )
    interrupted = True

    def _interrupt_after_grouping(*args: object, **kwargs: object) -> DocumentTree:
        nonlocal interrupted
        if interrupted:
            interrupted = False
            raise RuntimeError("simulated interruption after grouping checkpoint")
        return original_generate_tree_summaries(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        ocr_runtime_grouping_steps_module,
        "_generate_tree_summaries",
        _interrupt_after_grouping,
    )

    int_engines = build_workflow_engine_triplet(
        interrupted_scratch / "engines", "in_memory"
    )
    with pytest.raises(RuntimeError, match="OCR grouping workflow"):
        run_grouping_workflow(
            workflow_engine=int_engines[0],
            conversation_engine=int_engines[1],
            state_store=OCRWorkflowStateStore(interrupted_scratch / "ocr-state.sqlite"),
            tree=interrupted_tree,
            document_id=interrupted_tree.document_id,
            grouping_policy_version="incremental-v1",
            grouping_passes=passes,
            summary_policy=policy,
            summary_generator=_summary_generator,
            run_id="ocr-grouping-incremental-interrupted",
        )
    resumed, _ = run_grouping_workflow(
        workflow_engine=int_engines[0],
        conversation_engine=int_engines[1],
        state_store=OCRWorkflowStateStore(interrupted_scratch / "ocr-state.sqlite"),
        tree=interrupted_tree,
        document_id=interrupted_tree.document_id,
        grouping_policy_version="incremental-v1",
        grouping_passes=passes,
        summary_policy=policy,
        summary_generator=_summary_generator,
        run_id="ocr-grouping-incremental-interrupted",
        resume_from_checkpoint=True,
    )
    assert resumed.model_dump(mode="json") == reference.model_dump(mode="json")


@pytest.mark.ci
def test_summary_checkpoint_conflict_is_retryable_not_a_summary_warning() -> None:
    scratch = _scratch("ocr_summary_checkpoint_conflict")
    tree = DocumentTree(
        document_id="summary-conflict-doc",
        source_revision="ocr-v1:summary-conflict",
        tree_revision=1,
        root_id="root",
        source_items=(
            SourceItem(
                item_id="item-a",
                kind="text",
                text="summary source",
                location=SourceLocation(page_index=0, accuracy="page"),
            ),
            SourceItem(
                item_id="item-b",
                kind="text",
                text="second source",
                location=SourceLocation(page_index=0, accuracy="page"),
            ),
        ),
        nodes=(
            DocumentNode(
                node_id="leaf-a",
                kind="text",
                parts=(SourceSlice(item_id="item-a", start=0, end=14),),
            ),
            DocumentNode(
                node_id="leaf-b",
                kind="text",
                parts=(SourceSlice(item_id="item-b", start=0, end=13),),
            ),
            DocumentNode(
                node_id="root",
                kind="document",
                children=(ChildRef(node_id="leaf-a"), ChildRef(node_id="leaf-b")),
            ),
        ),
    )
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    initial_revision = store.save_document_tree(
        document_id=tree.document_id,
        tree=tree,
        grouping_policy_version="summary-v1",
    )
    store.save_document_tree(
        document_id=tree.document_id,
        tree=tree.model_copy(update={"tree_revision": 2}),
        grouping_policy_version="summary-v1",
        expected_tree_revision=initial_revision,
    )

    def _summary_generator(
        *,
        tree: DocumentTree,
        node_id: str,
        source_items: tuple[SourceItem, ...],
        input_fingerprint: str,
    ) -> Summary:
        del tree, source_items
        return Summary(
            status="generated",
            text=f"summary:{node_id}",
            evidence_item_ids=("item-a", "item-b"),
            provider_model="fake-summary",
            prompt_revision="summary-v1",
            input_fingerprint=input_fingerprint,
        )

    with pytest.raises(GroupingCheckpointConflict, match="revision conflict"):
        _generate_tree_summaries(
            tree,
            document_id=tree.document_id,
            state_store=store,
            policy=SummaryPolicy(
                eligible_kinds=("document",),
                minimum_source_items=1,
                minimum_source_chars=1,
                generation_fingerprint="summary-checkpoint-conflict-v1",
            ),
            generator=_summary_generator,
            grouping_policy_version="summary-v1",
            expected_tree_revision=initial_revision,
        )


@pytest.mark.ci
def test_ocr_pages_run_as_persisted_steps_and_terminal_resume_does_not_recall_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = _scratch("ocr_page_runtime")
    image_paths = []
    for page_number, text in ((1, "page one"), (2, "page two")):
        image_path = scratch / "source" / f"page-{page_number}.png"
        _draw_test_image(image_path, lines=[text])
        image_paths.append((page_number, image_path))
    workflow_engine, conversation_engine, _knowledge_engine = (
        build_workflow_engine_triplet(
            scratch / "engines",
            "in_memory",
        )
    )
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        parser=ProviderEndpointConfig(provider="fake", model="fake-parser"),
        embedding=EmbeddingProviderConfig(
            provider="fake", model="fake-embed", dimension=1
        ),
    )
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    reconcile_calls = 0
    original_reconcile = store.reconcile_orphan_artifact

    def _count_reconcile_orphan_artifact(*args, **kwargs):
        nonlocal reconcile_calls
        reconcile_calls += 1
        return original_reconcile(*args, **kwargs)

    monkeypatch.setattr(
        store,
        "reconcile_orphan_artifact",
        _count_reconcile_orphan_artifact,
    )
    calls = {"count": 0}

    def _runner(
        image_path: Path, page_number: int, provider_settings: WorkflowProviderSettings
    ):
        del image_path, provider_settings
        calls["count"] += 1
        return _fake_ocr_response(page_number, f"page {page_number}")

    source_plan = OCRSourcePlan(
        pdf_source=None,
        rendered_dir=scratch / "rendered",
        legacy_dir=scratch / "legacy",
        page_sources=image_paths,
        source_kind="image",
        input_fingerprint="source-fingerprint",
    )
    source_plan.legacy_dir.mkdir(parents=True)
    context = OCRPageProcessingContext(
        document_id="ocr-page-runtime-doc",
        title="OCR Page Runtime",
        source_kind="image",
        total_pages=2,
        input_fingerprint=source_plan.input_fingerprint,
        candidate_models=["fake-ocr"],
        provider_settings=settings,
        ocr_page_runner=_runner,
        state_store=store,
        probe=None,
        configuration_fingerprints=(
            ocr_pipeline_module._ocr_configuration_fingerprint(
                settings,
                candidate_models=["fake-ocr"],
            ),
        ),
    )
    raw_pages, completed, reused, run_id = run_ocr_page_workflow(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        context=context,
        source_plan=source_plan,
        progress_path=scratch / "ocr-progress.json",
        run_id="ocr-page-runtime-run",
    )
    assert len(raw_pages) == 2
    assert completed == [1, 2]
    assert reused == []
    assert calls["count"] == 2

    def _unexpected_runner(
        image_path: Path, page_number: int, provider_settings: WorkflowProviderSettings
    ):
        raise AssertionError(
            f"provider called during terminal resume for page {page_number}: {image_path}"
        )

    resumed_context = replace(context, ocr_page_runner=_unexpected_runner)
    resumed_pages, resumed_completed, resumed_reused, resumed_run_id = (
        run_ocr_page_workflow(
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            context=resumed_context,
            source_plan=source_plan,
            progress_path=scratch / "ocr-progress.json",
            run_id=run_id,
            resume_from_checkpoint=True,
        )
    )
    assert resumed_pages == raw_pages
    assert resumed_completed == completed
    assert resumed_reused == reused
    assert resumed_run_id == run_id
    trace = list(
        conversation_engine.read.get_nodes(
            where={"$and": [{"entity_type": "workflow_step_exec"}, {"run_id": run_id}]}
        )
    )
    assert {str(node.metadata.get("op")) for node in trace} >= {
        "start",
        "process_ocr_page",
        "finalize_ocr_pages",
        "end",
    }

    # Per-page durable progress is an append-only scalar delta. Repeating a
    # cumulative page-number list in every step would grow history quadratically.
    for step in trace:
        result = json.loads(step.metadata["result_json"])
        for operation, payload in result.get("state_update", []):
            if operation == "u":
                assert not {"completed_pages", "reused_pages"}.intersection(payload)
            if operation == "a":
                for key in {"completed_pages", "reused_pages"}.intersection(payload):
                    assert type(payload[key]) is int

    # Prove the parser's persisted updates are a complete state history, not
    # merely useful when replayed on top of a Kogwistar checkpoint. Start from
    # the exact initial state and fold every recorded delta without reading any
    # checkpoint, then compare with the workflow's materialized final snapshot.
    initial_state = _page_workflow_initial_state(source_plan)
    initial_state.update(
        {
            "document_id": context.document_id,
            "ocr_configuration_fingerprints": list(context.configuration_fingerprints),
        }
    )
    for step in sorted(trace, key=lambda node: int(node.metadata["step_seq"])):
        result = json.loads(step.metadata["result_json"])
        apply_state_update_inplace(
            initial_state,
            result.get("state_update", []),
        )

    checkpoints = list(
        conversation_engine.read.get_nodes(
            where={
                "$and": [
                    {"entity_type": "workflow_checkpoint"},
                    {"run_id": run_id},
                ]
            }
        )
    )
    assert checkpoints
    final_checkpoint = max(
        checkpoints,
        key=lambda node: int(node.metadata["step_seq"]),
    )
    snapshot_state = {
        key: value
        for key, value in json.loads(final_checkpoint.metadata["state_json"]).items()
        if not key.startswith("_")
    }
    assert initial_state == snapshot_state
    assert _json_pages(initial_state, source_plan=source_plan) == raw_pages

    # The reconciliation helper can resume verification at the already-checked
    # prefix boundary instead of re-reading every earlier source artifact.
    reconcile_calls = 0
    assert (
        _reconcile_page_checkpoint(
            state={
                "page_sources": [
                    {"page_number": number, "image_path": str(path)}
                    for number, path in image_paths
                ],
                "page_cursor": 2,
                "raw_pages": None,
                "completed_pages": completed,
                "reused_pages": [],
            },
            context=context,
            source_plan=source_plan,
            start_index=1,
        )
        is None
    )
    assert reconcile_calls == 1

    page_two = source_plan.legacy_dir / "page_2.json"
    page_two.write_text(page_two.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    repair = _reconcile_page_checkpoint(
        state={
            "page_sources": [
                {"page_number": number, "image_path": str(path)}
                for number, path in image_paths
            ],
            "page_cursor": 2,
            "raw_pages": raw_pages,
            "completed_pages": completed,
            "reused_pages": [],
        },
        context=context,
        source_plan=source_plan,
    )
    assert repair is not None
    assert repair["page_cursor"] == 1
    assert repair["raw_pages"] == raw_pages[:1]


def test_new_page_checkpoint_rehydrates_pages_from_artifacts_without_raw_payload() -> (
    None
):
    scratch = _scratch("ocr_page_compact_checkpoint")
    legacy_dir = scratch / "legacy"
    legacy_dir.mkdir(parents=True)
    image_paths = []
    for page_number in (1, 2):
        image_path = scratch / "source" / f"page-{page_number}.png"
        _draw_test_image(image_path, lines=[f"page {page_number}"])
        image_paths.append((page_number, image_path))
        (legacy_dir / f"page_{page_number}.json").write_text(
            json.dumps(
                _fake_ocr_response(page_number, f"page {page_number}").model_dump(
                    dump_format="json"
                )
            ),
            encoding="utf-8",
        )

    source_plan = OCRSourcePlan(
        pdf_source=None,
        rendered_dir=scratch / "rendered",
        legacy_dir=legacy_dir,
        page_sources=image_paths,
        source_kind="image",
        input_fingerprint="compact-checkpoint-source",
    )
    state = _page_workflow_initial_state(source_plan)
    state.update({"page_cursor": 2})

    assert "raw_pages" not in state
    pages = _json_pages(state, source_plan=source_plan)
    assert [page["printed_page_number"] for page in pages] == ["1", "2"]


@pytest.mark.ci
def test_standalone_ocr_publishes_progress_mirror_once_per_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_payloads: list[OCRImagePayload] = []
    for page_number in (1, 2):
        image_path = tmp_path / "source" / f"page-{page_number}.png"
        _draw_test_image(image_path, lines=[f"progress page {page_number}"])
        image_payloads.append(
            OCRImagePayload(page_number=page_number, image_path=str(image_path))
        )
    progress_sync_calls = 0
    original_sync = ocr_page_ingestion_module.sync_progress_from_state

    def _count_progress_sync(**kwargs):
        nonlocal progress_sync_calls
        progress_sync_calls += 1
        return original_sync(**kwargs)

    monkeypatch.setattr(
        ocr_page_ingestion_module,
        "sync_progress_from_state",
        _count_progress_sync,
    )
    artifacts = prepare_ocr_workflow_input(
        document_id="standalone-progress-once",
        title="Standalone progress",
        output_dir=tmp_path / "artifacts",
        image_payloads=image_payloads,
        provider_settings=WorkflowProviderSettings(
            ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
            embedding=EmbeddingProviderConfig(
                provider="fake", model="fake-embedding", dimension=1
            ),
        ),
        ocr_runner=lambda image_path, page_number, provider_settings: (
            _fake_ocr_response(page_number, f"progress page {page_number}")
        ),
    )

    assert artifacts.completed_pages == [1, 2]
    assert progress_sync_calls == 1
    assert (
        len(json.loads(artifacts.progress_path.read_text(encoding="utf-8"))["pages"])
        == 2
    )


@pytest.mark.ci
def test_reused_page_trace_query_is_not_truncated_at_default_read_limit() -> None:
    class _Read:
        def get_nodes(
            self, *, where: object, limit: int | None
        ) -> list[SimpleNamespace]:
            del where
            assert limit is None
            nodes = [
                SimpleNamespace(
                    metadata={
                        "step_seq": page_number,
                        "result_json": json.dumps(
                            {
                                "state_update": [
                                    [
                                        "u",
                                        {
                                            "page_cursor": page_number,
                                            "last_reused_page": page_number,
                                        },
                                    ]
                                ]
                            }
                        ),
                    }
                )
                for page_number in range(1, 251)
            ]
            nodes.append(
                SimpleNamespace(
                    metadata={
                        "step_seq": 251,
                        "result_json": json.dumps(
                            {
                                "state_update": [
                                    [
                                        "u",
                                        {
                                            "page_cursor": 250,
                                            "last_reused_page": None,
                                        },
                                    ]
                                ]
                            }
                        ),
                    }
                )
            )
            return nodes

    conversation_engine = cast(GraphKnowledgeEngine, SimpleNamespace(read=_Read()))
    assert _reused_pages_from_trace(conversation_engine, "large-resume") == [
        *range(1, 250),
    ]


def test_json_pages_rejects_malformed_persisted_page_shape() -> None:
    state = _page_workflow_initial_state(
        OCRSourcePlan(
            pdf_source=None,
            rendered_dir=Path("rendered"),
            legacy_dir=Path("legacy"),
            page_sources=[(1, Path("page-1.png"))],
            source_kind="image",
            input_fingerprint="malformed-page-source",
        )
    )
    state["raw_pages"] = [{"pdf_page_num": "one"}]

    with pytest.raises(TypeError, match="pdf_page_num"):
        _json_pages(state)


@pytest.mark.ci
def test_ocr_root_reopens_after_real_process_termination() -> None:
    """A terminated worker resumes the same root without replaying page one."""
    scratch = _scratch("ocr_root_process_termination")
    for page_number in (1, 2):
        _draw_test_image(
            scratch / "source" / f"page-{page_number}.png",
            lines=[f"terminated page {page_number}"],
        )
    worker = Path(__file__).with_name("_ocr_root_fresh_process_worker.py")
    repo_root = Path(__file__).resolve().parents[1]
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = os.pathsep.join(
        path
        for path in (
            str(repo_root),
            child_env.get("PYTHONPATH", ""),
        )
        if path
    )

    interrupted = subprocess.Popen(
        [sys.executable, str(worker), str(scratch), "hold"],
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    marker = scratch / "page-2-started"
    deadline = time.monotonic() + 90.0
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert marker.exists(), "worker did not reach page two before termination"
    _terminate_interrupted_worker(interrupted)
    interrupted.wait(timeout=30)
    assert interrupted.returncode != 0

    resumed = subprocess.run(
        [sys.executable, str(worker), str(scratch), "resume"],
        capture_output=True,
        text=True,
        check=False,
        env=child_env,
        timeout=120,
    )
    assert resumed.returncode == 0, resumed.stderr
    result = json.loads((scratch / "resume-result.json").read_text(encoding="utf-8"))
    assert result == {
        "status": "succeeded",
        "run_id": "ocr|fresh-root-document",
        "calls": [2],
        "has_grouping_tree": True,
    }


@pytest.mark.ci
@pytest.mark.slow
def test_ocr_root_fresh_process_resume_after_page_63() -> None:
    """A rich 100-page PDF survives two crashes and matches durable output."""
    pytest.importorskip("chromadb")
    scratch = _scratch("ocr_root_100_page_process_termination")
    page_count = 100
    child_env = dict(os.environ)
    child_env["OCR_TEST_LEASE_MS"] = "1000"

    def _write_pdf_case(directory: Path, *, change_page_40: bool = False) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "page-count.txt").write_text(str(page_count), encoding="utf-8")
        source = directory / "source"
        source.mkdir(parents=True, exist_ok=True)
        pages: list[Image.Image] = []
        rich_pages = {1, 40, 62, 63, 100}
        for page_number in range(1, page_count + 1):
            image = Image.new("L", (1200, 800), 255)
            draw = ImageDraw.Draw(image)
            if page_number not in rich_pages:
                draw.text((10, 20), f"root page {page_number}", fill=0)
                pages.append(image)
                continue

            draw.text((10, 10), f"Front matter page {page_number}", fill=0)
            draw.text(
                (10, 60),
                "duplicate paragraph retained at a distinct source position",
                fill=0,
            )
            cross_page_text = {
                62: "This paragraph starts on page 62 and continues",
                63: "on page 63 with a cross-page reference to page 40.",
            }.get(
                page_number, "This page contains the table and figure described below."
            )
            draw.text((10, 120), cross_page_text, fill=0)
            if change_page_40 and page_number == 40:
                draw.text((10, 150), "SOURCE REVISION TWO: page 40 changed.", fill=0)

            # This plotted region is deliberately non-text and has no OCR labels.
            draw.rectangle((20, 180, 300, 360), outline=0, width=3)
            draw.line(
                [(35, 330), (90, 290), (145, 310), (200, 230), (280, 205)],
                fill=0,
                width=4,
            )
            draw.text(
                (20, 380),
                "Figure 1. Deterministic multimodal acceptance fixture.",
                fill=0,
            )

            table_left, table_top = 320, 430
            column_edges = (table_left, 720, 1100)
            row_height = 20
            for row in range(17):
                y = table_top + row * row_height
                draw.line((table_left, y, 1100, y), fill=0, width=2)
            draw.line(
                (table_left, table_top, table_left, table_top + 16 * row_height),
                fill=0,
                width=2,
            )
            draw.line(
                (
                    column_edges[1],
                    table_top,
                    column_edges[1],
                    table_top + 16 * row_height,
                ),
                fill=0,
                width=2,
            )
            draw.line(
                (1100, table_top, 1100, table_top + 16 * row_height), fill=0, width=2
            )
            draw.text((330, table_top + 4), "Period", fill=0)
            draw.text((730, table_top + 4), "Value", fill=0)
            for row in range(1, 16):
                y = table_top + row * row_height + 4
                draw.text((330, y), f"row-{row}", fill=0)
                draw.text((730, y), str(row * 10), fill=0)
            draw.line(
                (
                    table_left,
                    table_top + 16 * row_height,
                    1100,
                    table_top + 16 * row_height,
                ),
                fill=0,
                width=2,
            )
            pages.append(image)

        pages[0].save(
            source / "acceptance.pdf",
            format="PDF",
            save_all=True,
            append_images=pages[1:],
            resolution=72,
        )
        pdf_pages = PdfReader(str(source / "acceptance.pdf")).pages
        assert len(pdf_pages) == page_count
        assert all(len(pdf_page.images) == 1 for pdf_page in pdf_pages)
        with Image.open(BytesIO(pdf_pages[0].images[0].data)) as embedded_page:
            assert embedded_page.convert("L").getpixel((20, 200)) < 128
        with Image.open(BytesIO(pdf_pages[61].images[0].data)) as embedded_page:
            assert embedded_page.convert("L").getpixel((20, 190)) < 128

    _write_pdf_case(scratch)
    worker = Path(__file__).with_name("_ocr_root_fresh_process_worker.py")
    repo_root = Path(__file__).resolve().parents[1]
    child_env["PYTHONPATH"] = os.pathsep.join(
        path
        for path in (
            str(repo_root),
            child_env.get("PYTHONPATH", ""),
        )
        if path
    )

    interrupted = subprocess.Popen(
        [sys.executable, str(worker), str(scratch), "durable-rich-summary-long-hold"],
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    marker = scratch / "page-64-started"
    deadline = time.monotonic() + 180.0
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert marker.exists(), "worker did not reach page 64 before termination"
    _terminate_interrupted_worker(interrupted)
    interrupted.wait(timeout=30)
    assert interrupted.returncode != 0

    grouping_resumed = subprocess.Popen(
        [
            sys.executable,
            str(worker),
            str(scratch),
            "durable-rich-summary-long-publish-hold",
        ],
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    summary_marker = scratch / "grounded-summary-ready"
    deadline = time.monotonic() + 900.0
    while not summary_marker.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    if not summary_marker.exists():
        _terminate_interrupted_worker(grouping_resumed)
        stdout, stderr = grouping_resumed.communicate(timeout=30)
        pytest.fail(
            "resumed 100-page run did not reach semantic publication after summary\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    _terminate_interrupted_worker(grouping_resumed)
    grouping_resumed.wait(timeout=30)
    assert grouping_resumed.returncode != 0
    summary_call_identities = [
        json.loads(line)
        for line in (scratch / "summary-call-identities.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    summary_call_keys = [
        (entry["node_id"], entry["input_fingerprint"])
        for entry in summary_call_identities
    ]
    assert summary_call_keys
    assert len(summary_call_keys) == len(set(summary_call_keys)), (
        "the same grounded summary identity was generated more than once before resume"
    )
    accepted_grouping_digest = (
        scratch / "semantic-input-grouping-tree-sha256.txt"
    ).read_text(encoding="utf-8")

    resumed = subprocess.run(
        [sys.executable, str(worker), str(scratch), "durable-rich-summary-long-resume"],
        capture_output=True,
        text=True,
        check=False,
        env=child_env,
        timeout=900,
    )
    assert resumed.returncode == 0, resumed.stderr
    result = json.loads((scratch / "resume-result.json").read_text(encoding="utf-8"))
    assert result["status"] == "succeeded", result
    assert result["run_id"] == "ocr|fresh-root-document", result
    assert result["calls"] == [], f"accepted OCR pages were replayed: {result}"
    assert result["has_grouping_tree"] is True, result
    assert result["grouping_summary_count"] == len(summary_call_keys), result
    resumed_summary_call_identities = [
        json.loads(line)
        for line in (scratch / "summary-call-identities.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert resumed_summary_call_identities == summary_call_identities, (
        "resuming semantic publication regenerated an already accepted summary"
    )
    page_call_records = [
        json.loads(line)
        for line in (scratch / "ocr-page-call-ledger.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert Counter(record["page_number"] for record in page_call_records) == (
        Counter({page_number: 1 for page_number in range(1, page_count + 1)})
        + Counter({64: 1})
    ), "recovery skipped a page or replayed an accepted OCR page"
    continued_page_calls = [
        record["page_number"]
        for record in page_call_records
        if record["mode"] == "durable-rich-summary-long-publish-hold"
    ]
    assert continued_page_calls == list(range(64, page_count + 1)), (
        "the fresh root resume did not continue contiguously from page 64"
    )
    assert not any(
        record["mode"] == "durable-rich-summary-long-resume"
        for record in page_call_records
    ), "semantic-publication recovery replayed OCR pages"
    lineage = json.loads(
        (scratch / "durable-rich-summary-long-resume-lineage.json").read_text(
            encoding="utf-8"
        )
    )
    lineage_rows = lineage["lineage"]
    assert len({row["id"] for row in lineage_rows}) == len(lineage_rows)
    assert {
        "kg_doc_parser.ocr_pages.v1",
        "kg_doc_parser.ocr_grouping.v1",
        "kg_doc_parser.ingest.v1",
    }.issubset({row["workflow_id"] for row in lineage_rows})
    assert lineage["graph_node_ids"]
    assert set(lineage["graph_node_ids"]) == set(lineage["persisted_graph_node_ids"])
    assert lineage["graph_edge_ids"]
    assert set(lineage["graph_edge_ids"]) == set(lineage["persisted_graph_edge_ids"])

    reopened_knowledge = GraphKnowledgeEngine(
        persist_directory=str(scratch / "engines" / "durable-knowledge"),
        kg_graph_type="knowledge",
        embedding_function=_TinyEmbeddingFunction(),
    )
    try:
        assert {
            node.id for node in reopened_knowledge.read.get_nodes(limit=10_000)
        } == set(lineage["graph_node_ids"])
        assert {
            edge.id for edge in reopened_knowledge.read.get_edges(limit=10_000)
        } == set(lineage["graph_edge_ids"])
    finally:
        reopened_knowledge.close()

    reference = scratch / "uninterrupted-reference"
    reference.mkdir(parents=True, exist_ok=True)
    (reference / "page-count.txt").write_text(str(page_count), encoding="utf-8")
    reference_source = reference / "source"
    reference_source.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        scratch / "source" / "acceptance.pdf",
        reference_source / "acceptance.pdf",
    )
    assert _compute_input_fingerprint(
        pdf_path=scratch / "source" / "acceptance.pdf"
    ) == _compute_input_fingerprint(pdf_path=reference_source / "acceptance.pdf")
    baseline = subprocess.run(
        [sys.executable, str(worker), str(reference), "durable-rich-summary-reference"],
        capture_output=True,
        text=True,
        check=False,
        env=child_env,
        timeout=900,
    )
    assert baseline.returncode == 0, baseline.stderr
    reference_publication = json.loads(
        (reference / "durable-rich-summary-reference-lineage.json").read_text(
            encoding="utf-8"
        )
    )
    reference_result = json.loads(
        (reference / "resume-result.json").read_text(encoding="utf-8")
    )
    assert (
        reference_result["grouping_summary_count"] == result["grouping_summary_count"]
    )
    assert set(reference_publication["graph_node_ids"]) == set(
        lineage["graph_node_ids"]
    )
    assert set(reference_publication["graph_edge_ids"]) == set(
        lineage["graph_edge_ids"]
    )
    assert (
        reference_publication["graph_payload_sha256"] == lineage["graph_payload_sha256"]
    )
    assert (
        reference_publication["grouping_tree_sha256"]
        == lineage["grouping_tree_sha256"]
        == accepted_grouping_digest
    )
    source_items = lineage["source_items"]
    assert isinstance(source_items, list) and source_items
    text_items = [
        item.get("text", "")
        for item in source_items
        if isinstance(item, dict) and item.get("kind") == "text"
    ]
    assert (
        sum(
            text == "duplicate paragraph retained at a distinct source position"
            for text in text_items
        )
        >= 2
    )
    assert any(
        "Figure 1. Deterministic multimodal acceptance fixture." in text
        for text in text_items
    )
    assert any("| row-15 | 150 |" in text for text in text_items)
    assert any("starts on page 62 and continues" in text for text in text_items)
    assert any(
        "on page 63 with a cross-page reference to page 40." in text
        for text in text_items
    )
    page_one_items = [
        item
        for item in source_items
        if isinstance(item, dict)
        and isinstance(item.get("location"), dict)
        and item["location"].get("page_index") == 0
    ]
    assert [str(item.get("text", "")) for item in page_one_items] == [
        "Front matter page 1",
        "duplicate paragraph retained at a distinct source position",
        "This page contains the table and figure described below.",
        "figure: line chart",
        "Figure 1. Deterministic multimodal acceptance fixture.",
        "| Period | Value |\n| --- | --- |\n"
        + "".join(f"| row-{row} | {row * 10} |\n" for row in range(1, 16)),
    ]
    assert any(
        isinstance(item, dict)
        and item.get("kind") == "non_text:description"
        and "line chart" in str(item.get("text", ""))
        for item in source_items
    )

    revision_one_pdf = scratch / "source" / "acceptance.pdf"
    revision_one_bytes = revision_one_pdf.read_bytes()
    revision_two_dir = scratch / "source-revision-two"
    _write_pdf_case(revision_two_dir, change_page_40=True)
    revision_two_pdf = revision_two_dir / "source" / "acceptance.pdf"
    revision_one_pages = PdfReader(BytesIO(revision_one_bytes)).pages
    revision_two_pages = PdfReader(str(revision_two_pdf)).pages
    assert len(revision_one_pages) == len(revision_two_pages) == page_count
    changed_pages = tuple(
        page_number
        for page_number, (old_page, new_page) in enumerate(
            zip(revision_one_pages, revision_two_pages, strict=True), start=1
        )
        if hashlib.sha256(old_page.images[0].data).digest()
        != hashlib.sha256(new_page.images[0].data).digest()
    )
    assert changed_pages == (40,)

    page_40_artifact = next((scratch / "artifacts").rglob("page_40.json"))
    accepted_page_40_digest = hashlib.sha256(page_40_artifact.read_bytes()).digest()
    revision_one_pdf.write_bytes(revision_two_pdf.read_bytes())
    changed_revision_resume = subprocess.run(
        [sys.executable, str(worker), str(scratch), "durable-rich-revision-two"],
        capture_output=True,
        text=True,
        check=False,
        env=child_env,
        timeout=900,
    )
    assert changed_revision_resume.returncode == 0, changed_revision_resume.stderr
    revision_two_result = json.loads(
        (scratch / "durable-rich-revision-two-result.json").read_text(encoding="utf-8")
    )
    assert revision_two_result["run_id"] == (
        "ocr|fresh-root-document|source-revision-two"
    )
    assert revision_two_result["calls"] == list(range(1, page_count + 1))
    revision_two_rendered_page_40 = max(
        (scratch / "artifacts" / "rendered_pages").rglob("page_40.png"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    revision_two_image_digest = hashlib.sha256(
        revision_two_rendered_page_40.read_bytes()
    ).hexdigest()
    revision_two_page_40_artifacts: list[Path] = []
    for candidate in (scratch / "artifacts").rglob("page_40.json"):
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        metadata = payload.get("metadata")
        if (
            isinstance(metadata, dict)
            and metadata.get("source_image_sha256") == revision_two_image_digest
        ):
            revision_two_page_40_artifacts.append(candidate)
    assert len(revision_two_page_40_artifacts) == 1
    revision_two_page_40_artifact = revision_two_page_40_artifacts[0]
    assert revision_two_page_40_artifact != page_40_artifact
    assert (
        hashlib.sha256(page_40_artifact.read_bytes()).digest()
        == accepted_page_40_digest
    )
    assert (
        hashlib.sha256(revision_two_page_40_artifact.read_bytes()).digest()
        != accepted_page_40_digest
    )
    revision_two_lineage = json.loads(
        (scratch / "durable-rich-revision-two-lineage.json").read_text(encoding="utf-8")
    )
    revision_two_page_40_items = [
        item
        for item in revision_two_lineage["source_items"]
        if isinstance(item, dict)
        and isinstance(item.get("location"), dict)
        and item["location"].get("page_index") == 39
    ]
    assert any(
        item.get("text") == "SOURCE REVISION TWO: page 40 changed."
        for item in revision_two_page_40_items
    )
    assert all(
        item.get("text") != "This page contains the table and figure described below."
        for item in revision_two_page_40_items
    )


@pytest.mark.ci
def test_ocr_root_concurrent_resume_has_one_owner_and_one_result() -> None:
    """Concurrent fresh resumes cannot both accept the same root run."""
    scratch = _scratch("ocr_root_concurrent_resume")
    for page_number in (1, 2):
        _draw_test_image(
            scratch / "source" / f"page-{page_number}.png",
            lines=[f"concurrent page {page_number}"],
        )
    worker = Path(__file__).with_name("_ocr_root_fresh_process_worker.py")
    repo_root = Path(__file__).resolve().parents[1]
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = os.pathsep.join(
        path
        for path in (
            str(repo_root),
            str(repo_root.parent / "kogwistar"),
            child_env.get("PYTHONPATH", ""),
        )
        if path
    )

    interrupted = subprocess.Popen(
        [sys.executable, str(worker), str(scratch), "hold"],
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    marker = scratch / "page-2-started"
    deadline = time.monotonic() + 90.0
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert marker.exists(), "worker did not reach page two before termination"
    _terminate_interrupted_worker(interrupted)
    interrupted.wait(timeout=30)
    assert interrupted.returncode != 0

    workers = [
        subprocess.Popen(
            [sys.executable, str(worker), str(scratch), mode],
            env=child_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for mode in ("race-a", "race-b")
    ]
    completed = [process.wait(timeout=120) for process in workers]
    assert sorted(completed) == [0, 1]

    accepted = []
    for mode in ("race-a", "race-b"):
        result_path = scratch / f"{mode}-result.json"
        if result_path.exists():
            accepted.append(json.loads(result_path.read_text(encoding="utf-8")))
        else:
            error_path = scratch / f"{mode}-error.txt"
            assert error_path.exists()
            error_text = error_path.read_text(encoding="utf-8")
            # Depending on where the competing process loses its lease, the
            # runtime may expose the typed ownership exception directly or
            # return it through the already-started nested grouping child as
            # a failed parent invocation.  Both paths are fail-closed; the
            # contract under test is that only one process accepts the root.
            _assert_concurrent_resume_loser_error(error_text)
    assert len(accepted) == 1
    assert accepted[0] == {
        "status": "succeeded",
        "run_id": "ocr|fresh-root-document",
        "calls": [2],
        "has_grouping_tree": True,
    }


@pytest.mark.ci
def test_grouping_checkpoint_reopens_after_process_termination() -> None:
    """A stopped grouping worker resumes from its persisted runtime checkpoint."""
    scratch = _scratch("ocr_grouping_process_termination")
    worker = Path(__file__).with_name("_ocr_grouping_fresh_process_worker.py")
    repo_root = Path(__file__).resolve().parents[1]
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = os.pathsep.join(
        path
        for path in (
            str(repo_root),
            str(repo_root.parent / "kogwistar"),
            child_env.get("PYTHONPATH", ""),
        )
        if path
    )

    interrupted = subprocess.Popen(
        [sys.executable, str(worker), str(scratch), "hold"],
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    marker = scratch / "grouping-summary-started"
    deadline = time.monotonic() + 90.0
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert marker.exists(), "grouping worker did not reach the post-checkpoint stage"
    _terminate_interrupted_worker(interrupted)
    interrupted.wait(timeout=30)
    assert interrupted.returncode != 0

    resumed = subprocess.run(
        [sys.executable, str(worker), str(scratch), "resume"],
        capture_output=True,
        text=True,
        check=False,
        env=child_env,
        timeout=120,
    )
    assert resumed.returncode == 0, resumed.stderr
    result = json.loads((scratch / "resume-result.json").read_text(encoding="utf-8"))
    assert result["run_id"] == "ocr-grouping|fresh-grouping-document"
    assert result["section_count"] == 1
    assert result["summary_count"] == 1


@pytest.mark.ci
@pytest.mark.slow
def test_ocr_root_resumes_after_page_63_and_matches_uninterrupted_reference() -> None:
    def _make_case(name: str) -> tuple[Path, Path, PDFRasterizer]:
        scratch = _scratch(name)
        pdf_path = scratch / "source" / "acceptance.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        writer = PdfWriter()
        for _ in range(100):
            writer.add_blank_page(width=1200, height=800)
        with pdf_path.open("wb") as stream:
            writer.write(stream)

        def _rasterizer(source: Path, rendered_dir: Path) -> list[Path]:
            page_count = len(PdfReader(str(source)).pages)
            paths: list[Path] = []
            for page_number in range(1, page_count + 1):
                image_path = rendered_dir / f"page_{page_number}.png"
                _draw_test_image(image_path, lines=[f"Acceptance page {page_number}"])
                paths.append(image_path)
            return paths

        return scratch, pdf_path, _rasterizer

    def _acceptance_response(page_number: int) -> OCRClusterResponse:
        """Return deterministic mixed-modality evidence for the release fixture."""
        if page_number not in {1, 40, 62, 63, 100}:
            return _fake_ocr_response(page_number, f"Acceptance page {page_number}")
        return OCRClusterResponse(
            OCR_text_clusters=[
                TextCluster(
                    text=f"Front matter page {page_number}",
                    bb_x_min=10.0,
                    bb_x_max=500.0,
                    bb_y_min=10.0,
                    bb_y_max=40.0,
                    cluster_number=0,
                ),
                TextCluster(
                    text="duplicate paragraph retained at a distinct source position",
                    bb_x_min=10.0,
                    bb_x_max=700.0,
                    bb_y_min=60.0,
                    bb_y_max=100.0,
                    cluster_number=1,
                ),
                TextCluster(
                    text=(
                        "cross-page reference continues from page 62 to page 63"
                        if page_number in {62, 63}
                        else "caption follows the figure and table"
                    ),
                    bb_x_min=10.0,
                    bb_x_max=700.0,
                    bb_y_min=120.0,
                    bb_y_max=160.0,
                    cluster_number=2,
                ),
            ],
            non_text_objects=[
                NonTextCluster(
                    description="figure: deterministic acceptance fixture",
                    bb_x_min=20.0,
                    bb_x_max=300.0,
                    bb_y_min=180.0,
                    bb_y_max=360.0,
                    cluster_number=3,
                ),
                NonTextCluster(
                    description="table: long atomic table with repeated rows",
                    bb_x_min=320.0,
                    bb_x_max=1100.0,
                    bb_y_min=180.0,
                    bb_y_max=500.0,
                    cluster_number=4,
                ),
            ],
            is_empty_page=False,
            printed_page_number=str(page_number),
            meaningful_ordering=[0, 3, 2, 4, 1],
            page_x_min=0.0,
            page_x_max=1200.0,
            page_y_min=0.0,
            page_y_max=800.0,
            estimated_rotation_degrees=0.0,
            incomplete_words_on_edge=False,
            incomplete_text=False,
            data_loss_likelihood=0.0,
            scan_quality="high",
            contains_table=True,
            spatial_accuracy="region",
        )

    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        parser=ProviderEndpointConfig(provider="fake", model="fake-parser"),
        embedding=EmbeddingProviderConfig(
            provider="fake", model="fake-embed", dimension=1
        ),
        triage_enabled=False,
    )
    interrupted_scratch, interrupted_pdf, interrupted_rasterizer = _make_case(
        "ocr_root_100_interrupted"
    )
    reference_scratch, reference_pdf, reference_rasterizer = _make_case(
        "ocr_root_100_reference"
    )
    interrupted_engines = build_workflow_engine_triplet(
        interrupted_scratch / "engines", "in_memory"
    )
    reference_engines = build_workflow_engine_triplet(
        reference_scratch / "engines", "in_memory"
    )
    interrupted_calls: list[int] = []
    reference_calls: list[int] = []

    def _interrupted_runner(
        image_path: Path,
        page_number: int,
        provider_settings: WorkflowProviderSettings,
    ) -> OCRClusterResponse:
        del image_path, provider_settings
        interrupted_calls.append(page_number)
        if page_number == 63 and interrupted_calls.count(63) == 1:
            raise RuntimeError("simulated process interruption after page 62")
        return _acceptance_response(page_number)

    def _reference_runner(
        image_path: Path,
        page_number: int,
        provider_settings: WorkflowProviderSettings,
    ) -> OCRClusterResponse:
        del image_path, provider_settings
        reference_calls.append(page_number)
        return _acceptance_response(page_number)

    def _parse_semantic_fn(*, collection, parser_input_dict, parser_source_map):
        return _grounded_semantic_tree(
            collection=collection,
            parser_input_dict=parser_input_dict,
            parser_source_map=parser_source_map,
        )

    reference_run, reference_bundle, _ = run_ocr_ingest_workflow(
        document_id="ocr-root-100-doc",
        title="OCR Root 100 Reference",
        output_dir=reference_scratch / "artifacts",
        pdf_path=reference_pdf,
        pdf_rasterizer=reference_rasterizer,
        provider_settings=settings,
        workflow_engine=reference_engines[0],
        conversation_engine=reference_engines[1],
        knowledge_engine=reference_engines[2],
        ocr_runner=_reference_runner,
        deps={"parse_semantic_fn": _parse_semantic_fn},
        run_id="ocr|ocr-root-100-reference",
    )
    with pytest.raises(RuntimeError, match="OCR root workflow"):
        run_ocr_ingest_workflow(
            document_id="ocr-root-100-doc",
            title="OCR Root 100 Reference",
            output_dir=interrupted_scratch / "artifacts",
            pdf_path=interrupted_pdf,
            pdf_rasterizer=interrupted_rasterizer,
            provider_settings=settings,
            workflow_engine=interrupted_engines[0],
            conversation_engine=interrupted_engines[1],
            knowledge_engine=interrupted_engines[2],
            ocr_runner=_interrupted_runner,
            deps={"parse_semantic_fn": _parse_semantic_fn},
            run_id="ocr|ocr-root-100-interrupted",
        )

    resumed_run, resumed_bundle, _resumed_artifacts = run_ocr_ingest_workflow(
        document_id="ocr-root-100-doc",
        title="OCR Root 100 Reference",
        output_dir=interrupted_scratch / "artifacts",
        pdf_path=interrupted_pdf,
        pdf_rasterizer=interrupted_rasterizer,
        provider_settings=settings,
        workflow_engine=interrupted_engines[0],
        conversation_engine=interrupted_engines[1],
        knowledge_engine=interrupted_engines[2],
        ocr_runner=_interrupted_runner,
        deps={"parse_semantic_fn": _parse_semantic_fn},
        run_id="ocr|ocr-root-100-interrupted",
        resume_from_checkpoint=True,
    )

    assert reference_run.status == "succeeded"
    assert resumed_run.status == "succeeded"
    assert reference_bundle is not None
    assert resumed_bundle is not None
    assert resumed_bundle.graph_payload == reference_bundle.graph_payload
    assert resumed_bundle.grouping_tree == reference_bundle.grouping_tree
    source_items = resumed_bundle.grouping_tree["source_items"]
    assert isinstance(source_items, list)
    assert {item["kind"] for item in source_items if isinstance(item, dict)} >= {
        "text",
        "non_text:description",
    }
    assert any(
        item.get("text") == "duplicate paragraph retained at a distinct source position"
        for item in source_items
        if isinstance(item, dict)
    )
    assert any(
        item.get("location", {}).get("page_index") == 61
        and item.get("text") == "cross-page reference continues from page 62 to page 63"
        for item in source_items
        if isinstance(item, dict)
    )
    assert interrupted_calls.count(63) == 2
    assert all(
        interrupted_calls.count(page_number) == 1 for page_number in range(1, 63)
    )
    assert len(reference_calls) == 100


@pytest.mark.ci
def test_cross_page_reference_survives_public_workflow_export() -> None:
    """Export keeps exact cross-page reference evidence as grouping metadata."""
    scratch = _scratch("ocr_cross_page_reference_export")
    payloads: list[OCRImagePayload] = []
    for page_number, text in ((1, "See table"), (2, " 7 for details.")):
        image_path = scratch / "images" / f"page-{page_number}.png"
        _draw_test_image(image_path, lines=[text])
        payloads.append(
            OCRImagePayload(page_number=page_number, image_path=str(image_path))
        )
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="reference-ocr"),
        parser=ProviderEndpointConfig(provider="fake", model="reference-parser"),
        embedding=EmbeddingProviderConfig(
            provider="fake", model="reference-embedding", dimension=1
        ),
        triage_enabled=False,
    )
    workflow_engine, conversation_engine, knowledge_engine = (
        build_workflow_engine_triplet(scratch / "engines", "in_memory")
    )

    def _runner(
        image_path: Path,
        page_number: int,
        provider_settings: WorkflowProviderSettings,
    ) -> OCRClusterResponse:
        del image_path, provider_settings
        text = "See table" if page_number == 1 else " 7 for details."
        return _fake_ocr_response(page_number, text)

    run, bundle, artifacts = run_ocr_ingest_workflow(
        document_id="ocr-cross-page-reference-export",
        title="OCR Cross-Page Reference Export",
        output_dir=scratch / "artifacts",
        image_payloads=payloads,
        provider_settings=settings,
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        ocr_runner=_runner,
        deps={"parse_semantic_fn": _grounded_semantic_tree},
        run_id="ocr|cross-page-reference-export",
    )

    assert run.status == "succeeded"
    assert bundle is not None and bundle.grouping_tree is not None
    assert artifacts.document_tree is not None
    mentions = artifacts.document_tree.references
    assert len(mentions) == 1
    mention = mentions[0]
    assert (mention.target_kind, mention.target_label, mention.status) == (
        "table",
        "7",
        "pending",
    )
    items = {item.item_id: item for item in artifacts.document_tree.source_items}
    source_fragments = [
        items[source.item_id].text[source.start : source.end]
        for source in mention.sources
    ]
    assert source_fragments == ["See table", " 7"]
    exported_mentions = bundle.grouping_tree.get("references")
    assert isinstance(exported_mentions, list)
    assert exported_mentions == [mention.model_dump(mode="json")]
    assert not any(
        isinstance(edge, dict) and edge.get("relation") == "REFERENCES"
        for edge in bundle.graph_payload["edges"]
    )


@pytest.mark.ci
def test_custom_grouping_policy_reaches_checkpoint_cache_and_export() -> None:
    """The public policy override must remain one identity at every seam."""
    scratch = _scratch("ocr_custom_grouping_policy")
    image_path = scratch / "images" / "page_1.png"
    _draw_test_image(
        image_path,
        lines=[
            "custom policy alpha",
            "custom policy beta",
            "custom policy gamma",
            "custom policy delta",
        ],
    )
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        parser=ProviderEndpointConfig(provider="fake", model="fake-parser"),
        embedding=EmbeddingProviderConfig(
            provider="fake", model="fake-embed", dimension=1
        ),
        triage_enabled=False,
    )

    def _four_cluster_runner(
        image_path: Path,
        page_number: int,
        provider_settings: WorkflowProviderSettings,
    ) -> OCRClusterResponse:
        del image_path, provider_settings
        return _fake_ocr_response(page_number, "custom policy alpha").model_copy(
            update={
                "OCR_text_clusters": [
                    TextCluster(
                        text="custom policy alpha",
                        bb_x_min=10.0,
                        bb_x_max=500.0,
                        bb_y_min=10.0,
                        bb_y_max=50.0,
                        cluster_number=0,
                    ),
                    TextCluster(
                        text="custom policy beta",
                        bb_x_min=10.0,
                        bb_x_max=500.0,
                        bb_y_min=60.0,
                        bb_y_max=100.0,
                        cluster_number=1,
                    ),
                    TextCluster(
                        text="custom policy gamma",
                        bb_x_min=10.0,
                        bb_x_max=500.0,
                        bb_y_min=110.0,
                        bb_y_max=150.0,
                        cluster_number=2,
                    ),
                    TextCluster(
                        text="custom policy delta",
                        bb_x_min=10.0,
                        bb_x_max=500.0,
                        bb_y_min=160.0,
                        bb_y_max=200.0,
                        cluster_number=3,
                    ),
                ],
                "meaningful_ordering": [0, 1, 2, 3],
            }
        )

    workflow_engine, conversation_engine, knowledge_engine = (
        build_workflow_engine_triplet(scratch / "engines", "in_memory")
    )
    policy_version = "custom-grouping-policy-v42"
    consumed_group_ids: list[str] = []
    semantic_roots: list[SemanticNode] = []

    def _group_aware_semantic_fn(*, collection, parser_input_dict, parser_source_map):
        grouping_tree = parser_input_dict.get("document_tree")
        assert isinstance(grouping_tree, dict)
        grouping_nodes = grouping_tree.get("nodes")
        assert isinstance(grouping_nodes, list)
        groups = [
            node
            for node in grouping_nodes
            if isinstance(node, dict) and node.get("kind") == "section"
        ]
        group_ids = [group.get("node_id") for group in groups]
        assert all(isinstance(group_id, str) for group_id in group_ids)
        consumed_group_ids.extend(
            group_id for group_id in group_ids if isinstance(group_id, str)
        )
        semantic = _grounded_semantic_tree(
            collection=collection,
            parser_input_dict=parser_input_dict,
            parser_source_map=parser_source_map,
        )
        leaf_nodes = semantic.child_nodes
        assert len(leaf_nodes) == 4
        semantic_groups: list[SemanticNode] = []
        for group_index, current_group_id in enumerate(consumed_group_ids):
            group_node_id = f"semantic-group:{current_group_id}"
            grouped_leaves = [
                leaf.model_copy(
                    update={
                        "parent_id": group_node_id,
                        "level_from_root": 2,
                    }
                )
                for leaf in leaf_nodes[group_index * 2 : group_index * 2 + 2]
            ]
            semantic_groups.append(
                SemanticNode(
                    node_id=group_node_id,
                    parent_id=semantic.node_id,
                    node_type="TEXT_FLOW",
                    title=f"grouped:{current_group_id}",
                    total_content_pointers=[
                        pointer
                        for leaf in grouped_leaves
                        for pointer in leaf.total_content_pointers
                    ],
                    child_nodes=[],
                    level_from_root=1,
                )
            )
        semantic.child_nodes = semantic_groups
        semantic_roots.append(semantic)
        return semantic

    run, bundle, artifacts = run_ocr_ingest_workflow(
        document_id="ocr-custom-policy-doc",
        title="OCR Custom Policy",
        output_dir=scratch / "artifacts",
        image_payloads=[OCRImagePayload(page_number=1, image_path=str(image_path))],
        provider_settings=settings,
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        ocr_runner=_four_cluster_runner,
        quality_policy=OCRQualityPolicy(mode="permissive"),
        grouping_policy_version=policy_version,
        grouping_proposals=(
            GroupProposal(
                kind="section",
                members=(
                    "leaf:ocr-custom-policy-doc|p1_t0",
                    "leaf:ocr-custom-policy-doc|p1_t1",
                ),
            ),
            GroupProposal(
                kind="section",
                members=(
                    "leaf:ocr-custom-policy-doc|p1_t2",
                    "leaf:ocr-custom-policy-doc|p1_t3",
                ),
            ),
        ),
        deps={
            "parse_semantic_fn": _group_aware_semantic_fn,
            "propose_layer_fn": None,
            "max_depth": 1,
        },
    )

    assert run.status == "succeeded"
    assert bundle is not None
    assert artifacts.document_tree is not None
    grouped_nodes = [
        node for node in artifacts.document_tree.nodes if node.kind == "section"
    ]
    assert len(grouped_nodes) == 2
    assert all(
        node.annotations["grouping:policy_version"] == f"{policy_version}:1"
        for node in grouped_nodes
    )
    checkpoint = OCRWorkflowStateStore(
        artifacts.state_db_path
    ).load_grouping_checkpoint_metadata(
        document_id="ocr-custom-policy-doc",
        grouping_policy_version=policy_version,
    )
    assert checkpoint is not None
    assert checkpoint.accepted_page_frontier == (1,)
    assert bundle.grouping_tree == artifacts.document_tree.model_dump(mode="json")
    assert bundle.grouping_tree["nodes"]
    assert consumed_group_ids == [node.node_id for node in grouped_nodes]
    assert len(semantic_roots) == 1
    semantic_root = semantic_roots[0]

    graph_nodes = [
        record for record in bundle.graph_payload["nodes"] if isinstance(record, dict)
    ]
    graph_edges = [
        record for record in bundle.graph_payload["edges"] if isinstance(record, dict)
    ]
    graph_node_ids = {record.get("id") for record in graph_nodes}
    assert {
        f"semantic-group:{group_id}" for group_id in consumed_group_ids
    } <= graph_node_ids
    assert all(
        any(record.get("label") == f"grouped:{group_id}" for record in graph_nodes)
        for group_id in consumed_group_ids
    )
    hierarchy_edges = [
        record for record in graph_edges if record.get("relation") == "HAS_CHILD"
    ]
    for group_index, group_id in enumerate(consumed_group_ids):
        graph_group_id = f"semantic-group:{group_id}"
        assert any(
            record.get("source_ids") == [semantic_root.node_id]
            and record.get("target_ids") == [graph_group_id]
            for record in hierarchy_edges
        )
        group_node = semantic_root.child_nodes[group_index]
        expected_sources = {
            pointer.source_cluster_id for pointer in group_node.total_content_pointers
        }
        graph_group = next(
            record for record in graph_nodes if record.get("id") == graph_group_id
        )
        mentions = graph_group.get("mentions")
        assert isinstance(mentions, list) and len(mentions) == 1
        mention = mentions[0]
        assert isinstance(mention, dict)
        spans = mention.get("spans")
        assert isinstance(spans, list)
        assert {
            span.get("source_cluster_id") for span in spans if isinstance(span, dict)
        } == expected_sources
        assert [span.get("excerpt") for span in spans if isinstance(span, dict)] == [
            pointer.verbatim_text for pointer in group_node.total_content_pointers
        ]


@pytest.mark.ci
def test_default_adaptive_root_wires_grounded_summary_and_semantic_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = _scratch("ocr_default_adaptive_root")
    image_path = scratch / "images" / "page_1.png"
    _draw_test_image(
        image_path,
        lines=[
            "adaptive root alpha",
            "adaptive root beta",
            "adaptive root gamma",
            "adaptive root delta",
        ],
    )
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="root-adaptive-ocr"),
        parser=ProviderEndpointConfig(provider="openai", model="root-adaptive-parser"),
        embedding=EmbeddingProviderConfig(
            provider="fake", model="root-adaptive-embed", dimension=1
        ),
        triage_enabled=False,
    )
    workflow_engine, conversation_engine, knowledge_engine = (
        build_workflow_engine_triplet(scratch / "engines", "in_memory")
    )
    proposal_calls: list[GroupingPassContext] = []
    summary_calls: list[tuple[str, tuple[str, ...]]] = []
    semantic_group_summaries: list[str] = []
    tree_publications: list[Path] = []
    summary_publications: list[str] = []
    original_write_tree = ocr_finalization_module._write_document_tree
    original_write_summary = ocr_finalization_module._write_summary

    def _write_tree_once(path: Path, tree: DocumentTree) -> None:
        tree_publications.append(path)
        original_write_tree(path, tree)

    def _write_summary_once(
        artifacts: OCRWorkflowArtifacts,
        *,
        document_id: str,
        provider_settings: WorkflowProviderSettings,
        state_store: OCRWorkflowStateStore,
    ) -> None:
        summary_publications.append("summary")
        original_write_summary(
            artifacts,
            document_id=document_id,
            provider_settings=provider_settings,
            state_store=state_store,
        )

    monkeypatch.setattr(
        ocr_finalization_module, "_write_document_tree", _write_tree_once
    )
    monkeypatch.setattr(ocr_finalization_module, "_write_summary", _write_summary_once)
    monkeypatch.setattr(
        ocr_runtime_finalization_module, "_write_document_tree", _write_tree_once
    )
    monkeypatch.setattr(
        ocr_runtime_finalization_module, "_write_summary", _write_summary_once
    )

    def _unexpected_chat_model(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("provider-free OCR test attempted to build a chat model")

    monkeypatch.setattr(
        strategy_module, "build_chat_model_for_role", _unexpected_chat_model
    )

    class _Proposer:
        configuration_fingerprint = "root-adaptive-proposer-v1"

        def __call__(self, context: GroupingPassContext) -> GroupProposalBatch:
            proposal_calls.append(context)
            if context.pass_index == 0 and len(context.frontier_ids) >= 4:
                return GroupProposalBatch(
                    proposals=(
                        GroupProposal(kind="section", members=context.frontier_ids[:2]),
                        GroupProposal(
                            kind="section", members=context.frontier_ids[2:4]
                        ),
                    )
                )
            return GroupProposalBatch()

    class _Summary:
        configuration_fingerprint = "root-adaptive-summary-v1"

        def __call__(
            self,
            *,
            tree: DocumentTree,
            node_id: str,
            evidence_slices: tuple[EvidenceSlice, ...],
            input_fingerprint: str,
        ) -> Summary:
            del tree
            item_ids = tuple(sorted({item.item_id for item in evidence_slices}))
            summary_calls.append((node_id, item_ids))
            return Summary(
                status="generated",
                text="Grounded section summary",
                evidence_item_ids=item_ids,
                provider_model="test:grounded-summary",
                prompt_revision="test-summary-v1",
                input_fingerprint=input_fingerprint,
            )

    proposer = _Proposer()
    summarizer = _Summary()
    monkeypatch.setattr(
        ocr_runtime_module,
        "build_production_group_proposal_generator",
        lambda _settings: proposer,
    )
    monkeypatch.setattr(
        ocr_runtime_module,
        "build_production_grounded_summary_generator",
        lambda _settings: summarizer,
    )

    def _runner(
        image_path: Path,
        page_number: int,
        provider_settings: WorkflowProviderSettings,
    ) -> OCRClusterResponse:
        del image_path, provider_settings
        return _fake_ocr_response(page_number, "adaptive root alpha").model_copy(
            update={
                "OCR_text_clusters": [
                    TextCluster(
                        text="adaptive root alpha",
                        bb_x_min=10.0,
                        bb_x_max=500.0,
                        bb_y_min=10.0,
                        bb_y_max=50.0,
                        cluster_number=0,
                    ),
                    TextCluster(
                        text="adaptive root beta",
                        bb_x_min=10.0,
                        bb_x_max=500.0,
                        bb_y_min=60.0,
                        bb_y_max=100.0,
                        cluster_number=1,
                    ),
                    TextCluster(
                        text="adaptive root gamma",
                        bb_x_min=10.0,
                        bb_x_max=500.0,
                        bb_y_min=110.0,
                        bb_y_max=150.0,
                        cluster_number=2,
                    ),
                    TextCluster(
                        text="adaptive root delta",
                        bb_x_min=10.0,
                        bb_x_max=500.0,
                        bb_y_min=160.0,
                        bb_y_max=200.0,
                        cluster_number=3,
                    ),
                ],
                "meaningful_ordering": [0, 1, 2, 3],
            }
        )

    def _parse(*, collection, parser_input_dict, parser_source_map):
        grouping_tree = parser_input_dict.get("document_tree")
        assert isinstance(grouping_tree, dict)
        nodes = grouping_tree.get("nodes")
        assert isinstance(nodes, list)
        sections = [
            node
            for node in nodes
            if isinstance(node, dict) and node.get("kind") == "section"
        ]
        assert len(sections) == 2
        semantic = _grounded_semantic_tree(
            collection=collection,
            parser_input_dict=parser_input_dict,
            parser_source_map=parser_source_map,
        )
        semantic_groups: list[SemanticNode] = []
        for group_index, section_record in enumerate(sections):
            section_id = section_record.get("node_id")
            assert isinstance(section_id, str)
            summary = section_record.get("summary")
            assert isinstance(summary, dict), f"summary calls so far: {summary_calls!r}"
            summary_text = summary.get("text")
            assert isinstance(summary_text, str)
            semantic_group_summaries.append(summary_text)
            group_leaves = semantic.child_nodes[group_index * 2 : group_index * 2 + 2]
            semantic_groups.append(
                SemanticNode(
                    node_id=f"semantic:{section_id}",
                    parent_id=semantic.node_id,
                    node_type="TEXT_FLOW",
                    title=summary_text,
                    total_content_pointers=[
                        pointer
                        for leaf in group_leaves
                        for pointer in leaf.total_content_pointers
                    ],
                    child_nodes=[],
                    level_from_root=1,
                )
            )
        semantic.child_nodes = semantic_groups
        return semantic

    run, bundle, artifacts = run_ocr_ingest_workflow(
        document_id="ocr-default-adaptive-root",
        title="Default Adaptive OCR Root",
        output_dir=scratch / "artifacts",
        image_payloads=[OCRImagePayload(page_number=1, image_path=str(image_path))],
        provider_settings=settings,
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        ocr_runner=_runner,
        quality_policy=OCRQualityPolicy(mode="permissive"),
        deps={"parse_semantic_fn": _parse, "propose_layer_fn": None, "max_depth": 1},
        run_id="ocr-default-adaptive-root-run",
    )

    assert run.status == "succeeded"
    assert len(tree_publications) == 1
    assert summary_publications == ["summary"]
    assert bundle is not None
    assert artifacts.document_tree is not None
    assert proposal_calls
    assert len(summary_calls) == 3
    assert sorted(len(item_ids) for _, item_ids in summary_calls) == [2, 2, 4]
    assert semantic_group_summaries == ["Grounded section summary"] * 2
    section_ids = {
        node.node_id for node in artifacts.document_tree.nodes if node.kind == "section"
    }
    sections = [
        node for node in artifacts.document_tree.nodes if node.kind == "section"
    ]
    assert len(sections) == 2
    assert all(node.summary is not None for node in sections)
    graph_node_ids = {
        record.get("id")
        for record in bundle.graph_payload["nodes"]
        if isinstance(record, dict)
    }
    assert section_ids
    assert {f"semantic:{node_id}" for node_id in section_ids} & graph_node_ids


@pytest.mark.ci
def test_raw_ocr_response_rejects_inconsistent_empty_or_ordering_state() -> None:
    payload = {
        "boxes_2d": [{"box_2d": [0, 0, 10, 10], "label": "Alpha", "id": 0}],
        "non_text_objects": [],
        "is_empty_page": False,
        "printed_page_number": None,
        "meaningful_ordering": [],
        "page_x_min": 0.0,
        "page_x_max": 10.0,
        "page_y_min": 0.0,
        "page_y_max": 10.0,
        "estimated_rotation_degrees": 0.0,
        "incomplete_words_on_edge": False,
        "incomplete_text": False,
        "data_loss_likelihood": 0.0,
        "scan_quality": "high",
        "contains_table": False,
    }

    with pytest.raises(ValueError, match="meaningful_ordering"):
        RawOCRResponse.model_validate(payload)


def _grounded_semantic_tree(*, collection, parser_input_dict, parser_source_map):
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


def _skip_if_live_ocr_unavailable(exc: Exception) -> None:
    message = str(exc).lower()
    if any(
        token in message
        for token in (
            "connect",
            "connection",
            "refused",
            "model",
            "ollama",
            "api key",
            "google_api_key",
            "no module named",
            "langchain_ollama",
            "poppler",
            "pdfinfo",
        )
    ):
        pytest.skip(f"live ocr unavailable: {exc}")


@pytest.mark.ci
def test_ocr_state_adopts_valid_artifact_left_before_completion_commit() -> None:
    scratch = _scratch("ocr_orphan_artifact")
    image = scratch / "page-1.png"
    image.write_bytes(b"stable-source-image")
    artifact = scratch / "page-1.json"
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="model-a"),
        parser=ProviderEndpointConfig(provider="fake", model="parser"),
        embedding=EmbeddingProviderConfig(provider="fake", model="embed", dimension=1),
    )
    image_digest = hashlib.sha256(image.read_bytes()).hexdigest()
    config_fingerprint = _ocr_configuration_fingerprint(settings)
    response = _fake_ocr_response(1, "orphan artifact")
    split_page = SplitPage(
        pdf_page_num=1,
        metadata=SplitPageMeta(
            ocr_model_name="model-a",
            ocr_datetime=0.0,
            ocr_json_version="workflow_ingest_v1",
        ),
        **response.model_dump(dump_format="python"),
    )
    payload = split_page.dump_raw(dump_format="json")
    payload["metadata"] = {
        **split_page.metadata.model_dump(mode="json"),
        **_ocr_page_metadata(
            settings,
            source_image_sha256=image_digest,
        ),
    }
    _write_json_atomic(artifact, payload)

    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="orphan-artifact-doc",
        title="Orphan Artifact",
        source_kind="image",
        total_pages=1,
        input_fingerprint="input-fingerprint",
    )
    store.record_attempt(
        document_id="orphan-artifact-doc",
        page_number=1,
        stage="ocr",
        content_hash=image_digest,
        model_name="model-a",
        artifact_path=artifact,
        config_fingerprint=config_fingerprint,
    )

    assert store.reconcile_orphan_artifact(
        document_id="orphan-artifact-doc",
        page_number=1,
        content_hash=image_digest,
        artifact_path=artifact,
        config_fingerprints=(config_fingerprint,),
    )
    assert store.should_skip_page(
        document_id="orphan-artifact-doc",
        page_number=1,
        stage="ocr",
        content_hash=image_digest,
        artifact_path=artifact,
        config_fingerprint=config_fingerprint,
    )

    artifact.write_text(json.dumps({"pdf_page_num": 1}), encoding="utf-8")
    assert not store.reconcile_orphan_artifact(
        document_id="orphan-artifact-doc",
        page_number=1,
        content_hash=image_digest,
        artifact_path=artifact,
        config_fingerprints=(config_fingerprint,),
    )


@pytest.mark.ci
def test_ocr_state_does_not_reuse_completion_without_published_artifact() -> None:
    """A DB completion without its artifact must be treated as an unaccepted attempt."""
    scratch = _scratch("ocr_db_before_artifact")
    artifact = scratch / "page-1.json"
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="db-before-artifact-doc",
        title="DB Before Artifact",
        source_kind="image",
        total_pages=1,
        input_fingerprint="input-fingerprint",
    )
    attempt = store.record_attempt(
        document_id="db-before-artifact-doc",
        page_number=1,
        stage="ocr",
        content_hash="source-image-digest",
        model_name="model-a",
        artifact_path=artifact,
        config_fingerprint="provider-a/model-a",
    )
    assert not store.record_page_completed(
        document_id="db-before-artifact-doc",
        page_number=1,
        stage="ocr",
        content_hash="source-image-digest",
        artifact_path=artifact,
        model_name="model-a",
        attempt_index=attempt,
        config_fingerprint="provider-a/model-a",
    )

    assert not artifact.exists()
    assert not store.should_skip_page(
        document_id="db-before-artifact-doc",
        page_number=1,
        stage="ocr",
        content_hash="source-image-digest",
        artifact_path=artifact,
        config_fingerprint="provider-a/model-a",
    )
    state = store.get_page_state(
        document_id="db-before-artifact-doc", page_number=1, stage="ocr"
    )
    assert state is not None and state.status == "pending"


@pytest.mark.ci
def test_ocr_two_store_recovery_predicates_hold_in_a_fresh_process() -> None:
    """Reopen both crash orders and apply the same acceptance predicates.

    This is deliberately a recovery-boundary test, not a claim that a process
    can atomically commit SQLite and a filesystem rename. A restart must reject
    the database-first order and adopt the validated artifact-first order.
    """
    scratch = _scratch("ocr_two_store_fresh_process")
    db_path = scratch / "ocr-state.sqlite"
    missing_artifact = scratch / "db-first.json"
    adopted_artifact = scratch / "artifact-first.json"
    source_digest = "source-image-digest"
    config_fingerprint = "provider-a/model-a"

    store = OCRWorkflowStateStore(db_path)
    store.ensure_document(
        document_id="db-first-doc",
        title="DB First",
        source_kind="image",
        total_pages=1,
        input_fingerprint="db-first-input",
    )
    attempt = store.record_attempt(
        document_id="db-first-doc",
        page_number=1,
        stage="ocr",
        content_hash=source_digest,
        model_name="model-a",
        artifact_path=missing_artifact,
        config_fingerprint=config_fingerprint,
    )
    store.record_page_completed(
        document_id="db-first-doc",
        page_number=1,
        stage="ocr",
        content_hash=source_digest,
        artifact_path=missing_artifact,
        model_name="model-a",
        attempt_index=attempt,
        config_fingerprint=config_fingerprint,
    )

    response = _fake_ocr_response(1, "artifact-first")
    split_page = SplitPage(
        pdf_page_num=1,
        metadata=SplitPageMeta(
            ocr_model_name="model-a",
            ocr_datetime=0.0,
            ocr_json_version="workflow_ingest_v1",
        ),
        **response.model_dump(dump_format="python"),
    )
    artifact_payload = split_page.dump_raw(dump_format="json")
    artifact_payload["metadata"] = {
        **split_page.metadata.model_dump(mode="json"),
        "source_image_sha256": source_digest,
        "ocr_configuration_fingerprint": config_fingerprint,
        "ocr_model_name": "model-a",
    }
    _write_json_atomic(adopted_artifact, artifact_payload)
    store.ensure_document(
        document_id="artifact-first-doc",
        title="Artifact First",
        source_kind="image",
        total_pages=1,
        input_fingerprint="artifact-first-input",
    )
    store.record_attempt(
        document_id="artifact-first-doc",
        page_number=1,
        stage="ocr",
        content_hash=source_digest,
        model_name="model-a",
        artifact_path=adopted_artifact,
        config_fingerprint=config_fingerprint,
    )

    child_script = f"""
from pathlib import Path
from kg_doc_parser.workflow_ingest.ocr_pipeline import OCRWorkflowStateStore

store = OCRWorkflowStateStore(Path({str(db_path)!r}))
assert not store.should_skip_page(
    document_id='db-first-doc', page_number=1, stage='ocr',
    content_hash={source_digest!r}, artifact_path=Path({str(missing_artifact)!r}),
    config_fingerprint={config_fingerprint!r},
)
assert store.reconcile_orphan_artifact(
    document_id='artifact-first-doc', page_number=1,
    content_hash={source_digest!r}, artifact_path=Path({str(adopted_artifact)!r}),
    config_fingerprints=({config_fingerprint!r},),
)
assert store.should_skip_page(
    document_id='artifact-first-doc', page_number=1, stage='ocr',
    content_hash={source_digest!r}, artifact_path=Path({str(adopted_artifact)!r}),
    config_fingerprint={config_fingerprint!r},
)
"""
    child = subprocess.run(
        [sys.executable, "-c", child_script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0


@pytest.mark.ci
def test_ocr_page_artifact_publish_crash_is_recoverable() -> None:
    """A crash after rename and before SQLite completion is safely recoverable."""
    scratch = _scratch("ocr_artifact_publish_crash")
    image_path = scratch / "source" / "page-1.png"
    _draw_test_image(image_path, lines=["crash boundary"])
    page_json_path = scratch / "legacy" / "page-1.json"
    page_image_path = scratch / "legacy" / "page-1.png"
    page_json_path.parent.mkdir(parents=True, exist_ok=True)
    page_image_path.parent.mkdir(parents=True, exist_ok=True)
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        embedding=EmbeddingProviderConfig(
            provider="fake", model="fake-embed", dimension=1
        ),
    )
    store_path = scratch / "ocr-state.sqlite"
    store = OCRWorkflowStateStore(store_path)
    config_fingerprint = ocr_pipeline_module._ocr_configuration_fingerprint(
        settings,
        candidate_models=["fake-ocr"],
        quality_policy=OCRQualityPolicy(mode="permissive"),
    )
    context = OCRPageProcessingContext(
        document_id="ocr-artifact-crash-doc",
        title="OCR Artifact Crash",
        source_kind="image",
        total_pages=1,
        input_fingerprint="source-fingerprint",
        candidate_models=["fake-ocr"],
        provider_settings=settings,
        ocr_page_runner=lambda _path, page_number, _settings: _fake_ocr_response(
            page_number, "crash boundary"
        ),
        state_store=store,
        probe=None,
        configuration_fingerprints=(config_fingerprint,),
        quality_policy=OCRQualityPolicy(mode="permissive"),
    )
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    original_write = ocr_state_store_module._write_bytes_atomic
    interrupted = True

    def _publish_then_interrupt(path: Path, artifact_bytes: bytes) -> None:
        nonlocal interrupted
        original_write(path, artifact_bytes)
        if interrupted and path == page_json_path:
            interrupted = False
            raise KeyboardInterrupt(
                "simulated process termination after artifact publish"
            )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        ocr_state_store_module, "_write_bytes_atomic", _publish_then_interrupt
    )
    try:
        with pytest.raises(KeyboardInterrupt, match="after artifact publish"):
            _process_ocr_page(
                context=context,
                done_count=0,
                page_number=1,
                image_path=image_path,
                image_sha256=image_sha256,
                page_json_path=page_json_path,
                page_image_path=page_image_path,
            )
    finally:
        monkeypatch.undo()

    pending = store.get_page_state(
        document_id=context.document_id,
        page_number=1,
        stage="ocr",
    )
    assert pending is not None and pending.status == "pending"
    assert page_json_path.is_file()

    reopened = OCRWorkflowStateStore(store_path)
    assert reopened.reconcile_orphan_artifact(
        document_id=context.document_id,
        page_number=1,
        content_hash=image_sha256,
        artifact_path=page_json_path,
        config_fingerprints=(config_fingerprint,),
    )
    assert reopened.should_skip_page(
        document_id=context.document_id,
        page_number=1,
        stage="ocr",
        content_hash=image_sha256,
        artifact_path=page_json_path,
        config_fingerprint=config_fingerprint,
    )


@pytest.mark.ci
def test_duplicate_ocr_page_workers_accept_only_one_artifact() -> None:
    """Concurrent duplicate delivery cannot replace the accepted page."""
    scratch = _scratch("ocr_duplicate_workers")
    image_path = scratch / "source" / "page-1.png"
    _draw_test_image(image_path, lines=["duplicate boundary"])
    page_json_path = scratch / "legacy" / "page-1.json"
    page_image_path = scratch / "legacy" / "page-1.png"
    page_json_path.parent.mkdir(parents=True, exist_ok=True)
    page_image_path.parent.mkdir(parents=True, exist_ok=True)
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        embedding=EmbeddingProviderConfig(
            provider="fake", model="fake-embed", dimension=1
        ),
    )
    store_path = scratch / "ocr-state.sqlite"
    config_fingerprint = ocr_pipeline_module._ocr_configuration_fingerprint(
        settings,
        candidate_models=["fake-ocr"],
        quality_policy=OCRQualityPolicy(mode="permissive"),
    )
    barrier = threading.Barrier(2)
    contexts: list[OCRPageProcessingContext] = []
    for worker_index in (1, 2):
        worker_store = OCRWorkflowStateStore(store_path)

        def _runner(
            _path: Path,
            page_number: int,
            _settings: WorkflowProviderSettings,
            *,
            worker: int = worker_index,
        ) -> OCRClusterResponse:
            barrier.wait(timeout=10)
            return _fake_ocr_response(page_number, f"worker-{worker}")

        contexts.append(
            OCRPageProcessingContext(
                document_id="ocr-duplicate-workers-doc",
                title="OCR Duplicate Workers",
                source_kind="image",
                total_pages=1,
                input_fingerprint="source-fingerprint",
                candidate_models=["fake-ocr"],
                provider_settings=settings,
                ocr_page_runner=_runner,
                state_store=worker_store,
                probe=None,
                configuration_fingerprints=(config_fingerprint,),
                quality_policy=OCRQualityPolicy(mode="permissive"),
            )
        )
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()

    def _process(context: OCRPageProcessingContext) -> OCRPageOutcome:
        return _process_ocr_page(
            context=context,
            done_count=0,
            page_number=1,
            image_path=image_path,
            image_sha256=image_sha256,
            page_json_path=page_json_path,
            page_image_path=page_image_path,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(_process, contexts))

    assert all(
        result.page_number == 1 and result.page is not None for result in results
    )
    assert sorted(result.status for result in results) == ["completed", "reused"]
    persisted = json.loads(page_json_path.read_text(encoding="utf-8"))
    assert persisted["OCR_text_clusters"][0]["text"] in {"worker-1", "worker-2"}
    with sqlite3.connect(store_path) as connection:
        completed_attempts = connection.execute(
            "SELECT COUNT(*) FROM model_attempts WHERE stage = 'ocr' AND status = 'completed'"
        ).fetchone()[0]
    assert completed_attempts == 1


@pytest.mark.ci
def test_ocr_state_does_not_reuse_legacy_completion_without_artifact_digest() -> None:
    scratch = _scratch("ocr_missing_artifact_digest")
    artifact = scratch / "page-1.json"
    artifact.write_text("{}", encoding="utf-8")
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="legacy-digest-doc",
        title="Legacy Digest",
        source_kind="image",
        total_pages=1,
        input_fingerprint="fingerprint",
    )
    attempt = store.record_attempt(
        document_id="legacy-digest-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-hash",
        model_name="model",
        artifact_path=artifact,
        config_fingerprint="provider/model",
    )
    store.record_page_completed(
        document_id="legacy-digest-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-hash",
        artifact_path=artifact,
        model_name="model",
        attempt_index=attempt,
        config_fingerprint="provider/model",
    )
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE page_state SET artifact_sha256 = NULL WHERE document_id = ?",
            ("legacy-digest-doc",),
        )
    assert not store.should_skip_page(
        document_id="legacy-digest-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-hash",
        artifact_path=artifact,
        config_fingerprint="provider/model",
    )


# Manual examples with explicit parametrized node ids:
#
# Image payload + glm-ocr:
#   .venv\Scripts\python.exe -m pytest tests/test_workflow_ingest_ocr_pipeline.py::test_workflow_first_ocr_manual_matrix[glm-ocr-image] -q
#
# PDF input + glm-ocr:
#   .venv\Scripts\python.exe -m pytest tests/test_workflow_ingest_ocr_pipeline.py::test_workflow_first_ocr_manual_matrix[glm-ocr-pdf] -q
#
# Image payload + gemma4:
#   .venv\Scripts\python.exe -m pytest tests/test_workflow_ingest_ocr_pipeline.py::test_workflow_first_ocr_manual_matrix[gemma4-image] -q
#
# Image payload + gemini:
#   .venv\Scripts\python.exe -m pytest tests/test_workflow_ingest_ocr_pipeline.py::test_workflow_first_ocr_manual_matrix[gemini-image] -q
#
# The output directory for each manual case is stable and inspectable under:
#   tests/.tmp_workflow_ingest_ocr/manual_cases/<provider>/<model>/<input_kind>
#
# Suggested manual inspection flow:
# 1. click a single case in VS Code
# 2. watch page-level OCR progress in the test output
# 3. inspect `artifacts/ocr-summary.json` for the run summary
# 4. inspect `artifacts/ocr-state.sqlite` for authoritative page state
# 5. inspect `artifacts/ocr-progress.json` for readable mirrored resume state
# 6. inspect `artifacts/legacy_split_pages/.../page_N.json` for OCR payloads
# 7. inspect `source/` and `artifacts/rendered_pages/` if the OCR looks wrong
#
# If a local model is weak or occasionally unstable:
# - delete only that case directory
# - rerun the same clicked case
# - compare the regenerated `page_N.json` files with the previous run


@pytest.mark.manual
@pytest.mark.llm_real
@pytest.mark.parametrize(
    "provider, model",
    [
        pytest.param(
            "ollama",
            "glm-ocr:latest",
            id="glm-ocr",
            marks=pytest.mark.requires_ollama,
        ),
        pytest.param(
            "ollama",
            "gemma4:e2b",
            id="gemma4",
            marks=pytest.mark.requires_ollama,
        ),
        pytest.param("gemini", "gemini-2.5-flash", id="gemini"),
    ],
)
@pytest.mark.parametrize(
    "input_kind",
    [
        pytest.param("image", id="image"),
        pytest.param("pdf", id="pdf"),
    ],
)
def test_workflow_first_ocr_manual_matrix(
    provider: str, model: str, input_kind: str
) -> None:
    """Manual OCR workflow case meant for inspection in VS Code.

    This case is intentionally artifact-heavy. Each run writes:

    - rendered page images
    - legacy-compatible page JSON files
    - a SQLite state store
    - a progress manifest
    - a short summary file
    - the normalized workflow-ingest handoff state that downstream parsing sees

    The directory is stable per case so reruns are easy to inspect. If the OCR
    model output becomes stale or corrupted, delete the case directory and rerun.

    This is a full workflow-first OCR path:

    1. source image or PDF pages
    2. per-page OCR model invocation
    3. legacy-compatible page artifact write
    4. workflow-ingest normalization
    5. workflow ingest execution with deterministic grounded semantic parsing

    The grounded semantic parse is intentional here: it keeps the downstream
    ingest deterministic while still asserting real span grounding against the
    OCR source map, so manual inspection can focus on the OCR and span quality
    rather than mixed OCR + parser variability.
    """

    if provider == "gemini" and not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY is required for the manual Gemini OCR case")
    if provider == "ollama" and importlib.util.find_spec("langchain_ollama") is None:
        pytest.skip("langchain_ollama is required for manual Ollama OCR cases")

    case_dir = _manual_case_dir(input_kind=input_kind, provider=provider, model=model)
    source_dir = case_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    image_paths = [source_dir / "page_1.png", source_dir / "page_2.png"]
    _draw_test_image(image_paths[0], lines=["OCR workflow page one", "Alpha clause"])
    _draw_test_image(image_paths[1], lines=["OCR workflow page two", "Beta clause"])

    image_payloads = [
        OCRImagePayload(page_number=1, image_path=str(image_paths[0])),
        OCRImagePayload(page_number=2, image_path=str(image_paths[1])),
    ]
    pdf_path = source_dir / "manual_source.pdf"
    _write_two_page_pdf(image_paths, pdf_path)

    workflow_engine, conversation_engine, knowledge_engine = (
        build_workflow_engine_triplet(
            case_dir / "engines",
            "in_memory",
        )
    )
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(
            provider=provider,
            model=model,
            base_url=os.getenv("KG_DOC_OCR_BASE_URL", "http://127.0.0.1:11434")
            if provider == "ollama"
            else None,
            api_key_env="GOOGLE_API_KEY" if provider == "gemini" else None,
        ),
        embedding=EmbeddingProviderConfig(
            provider="fake", model="ocr-manual-embed", dimension=1
        ),
    )

    try:
        run, bundle, artifacts = run_ocr_ingest_workflow(
            document_id=f"ocr-manual-{provider}-{input_kind}",
            title="Workflow OCR Manual Case",
            output_dir=case_dir / "artifacts",
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            knowledge_engine=knowledge_engine,
            image_payloads=image_payloads if input_kind == "image" else None,
            pdf_path=pdf_path if input_kind == "pdf" else None,
            provider_settings=settings,
            deps={"parse_semantic_fn": _grounded_semantic_tree},
        )
        drain_phase1_indexes_until_idle(
            workflow_engine, conversation_engine, knowledge_engine
        )
    except Exception as exc:
        _skip_if_live_ocr_unavailable(exc)
        raise

    assert run.status == "succeeded"
    assert bundle is not None
    semantic_tree = SemanticNode.model_validate(run.final_state["semantic_tree"])
    assert artifacts.summary_path.exists()
    assert artifacts.progress_path.exists()
    assert artifacts.state_db_path.exists()
    assert artifacts.legacy_dir.exists()
    assert len(list(artifacts.legacy_dir.glob("page_*.json"))) >= 2
    assert semantic_tree.child_nodes
    assert all(node.total_content_pointers for node in semantic_tree.child_nodes)

    _LOGGER.info("OCR manual case completed")
    _LOGGER.info("  input_kind=%s provider=%s model=%s", input_kind, provider, model)
    _LOGGER.info("  source_dir=%s", source_dir)
    _LOGGER.info("  rendered_dir=%s", artifacts.rendered_dir)
    _LOGGER.info("  legacy_dir=%s", artifacts.legacy_dir)
    _LOGGER.info("  state_db_path=%s", artifacts.state_db_path)
    _LOGGER.info("  progress_path=%s", artifacts.progress_path)
    _LOGGER.info("  summary_path=%s", artifacts.summary_path)
    _LOGGER.info(
        "  completed_pages=%s reused_pages=%s",
        artifacts.completed_pages,
        artifacts.reused_pages,
    )


@pytest.mark.ci
@pytest.mark.slow
def test_ocr_workflow_resumes_cached_pages_and_terminal_semantic_run() -> None:
    scratch = _scratch("ocr_grounded_cache")
    images_dir = scratch / "images"
    image_paths = [images_dir / "page_1.png", images_dir / "page_2.png"]
    _draw_test_image(image_paths[0], lines=["Grounded cache page one", "Alpha clause"])
    _draw_test_image(image_paths[1], lines=["Grounded cache page two", "Beta clause"])
    payloads = [
        OCRImagePayload(page_number=1, image_path=str(image_paths[0])),
        OCRImagePayload(page_number=2, image_path=str(image_paths[1])),
    ]
    calls = {"ocr": 0, "parse": 0}
    captured_grouping_trees: list[object] = []

    def _runner(
        image_path: Path, page_number: int, provider_settings: WorkflowProviderSettings
    ):
        calls["ocr"] += 1
        return _fake_ocr_response(page_number, f"Grounded page {page_number}")

    def _parse_semantic_fn(*, collection, parser_input_dict, parser_source_map):
        calls["parse"] += 1
        grouping_tree = parser_input_dict.get("document_tree")
        captured_grouping_trees.append(grouping_tree)
        assert isinstance(grouping_tree, dict)
        grouping_root_id = grouping_tree.get("root_id")
        assert isinstance(grouping_root_id, str)
        semantic_tree = _grounded_semantic_tree(
            collection=collection,
            parser_input_dict=parser_input_dict,
            parser_source_map=parser_source_map,
        )
        semantic_tree.child_nodes[0] = semantic_tree.child_nodes[0].model_copy(
            update={"title": f"grouped:{grouping_root_id}"}
        )
        return semantic_tree

    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        parser=ProviderEndpointConfig(provider="fake", model="fake-parser"),
        embedding=EmbeddingProviderConfig(
            provider="fake", model="fake-embed", dimension=1
        ),
    )
    workflow_engine, conversation_engine, knowledge_engine = (
        build_workflow_engine_triplet(
            scratch / "engines",
            "in_memory",
        )
    )

    first_run, first_bundle, _first_artifacts = run_ocr_ingest_workflow(
        document_id="ocr-grounded-cache-doc",
        title="OCR Grounded Cache Doc",
        output_dir=scratch / "artifacts",
        image_payloads=payloads,
        provider_settings=settings,
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        ocr_runner=_runner,
        deps={"parse_semantic_fn": _parse_semantic_fn},
    )
    second_run, second_bundle, second_artifacts = run_ocr_ingest_workflow(
        document_id="ocr-grounded-cache-doc",
        title="OCR Grounded Cache Doc",
        output_dir=scratch / "artifacts",
        image_payloads=payloads,
        provider_settings=settings,
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        ocr_runner=_runner,
        deps={"parse_semantic_fn": _parse_semantic_fn},
        resume_from_checkpoint=True,
    )

    assert first_run.status == "succeeded"
    assert second_run.status == "succeeded"
    assert first_bundle is not None
    assert second_bundle is not None
    assert first_bundle.graph_payload["nodes"]
    assert all(
        knowledge_engine.persist.exists_node(str(node["id"]))
        for node in first_bundle.graph_payload["nodes"]
    )
    assert calls["ocr"] == 2
    assert calls["parse"] == 1
    assert _first_artifacts.document_tree is not None
    assert captured_grouping_trees == [
        _first_artifacts.document_tree.model_dump(mode="json")
    ]
    assert second_artifacts.reused_pages == [1, 2]
    assert second_artifacts.state_db_path.exists()
    assert SemanticNode.model_validate(
        second_run.final_state["semantic_tree"]
    ).child_nodes
    assert second_run.final_state["semantic_tree"]["child_nodes"][0][
        "total_content_pointers"
    ]
    assert (
        second_run.final_state["semantic_tree"]["child_nodes"][0]["title"]
        == "grouped:root"
    )
    assert first_bundle.grouping_tree == _first_artifacts.document_tree.model_dump(
        mode="json"
    )
    assert second_bundle.grouping_tree == second_artifacts.document_tree.model_dump(
        mode="json"
    )
    # The semantic child must publish the exact accepted grouping snapshot,
    # rather than merely parsing the same source pages independently.
    assert (
        first_bundle.grouping_tree
        == first_run.final_state["export_bundle"]["grouping_tree"]
    )
    assert (
        second_bundle.grouping_tree
        == second_run.final_state["export_bundle"]["grouping_tree"]
    )
    (second_artifacts.legacy_dir / "page_1.json").unlink()
    third_run, third_bundle, third_artifacts = run_ocr_ingest_workflow(
        document_id="ocr-grounded-cache-doc",
        title="OCR Grounded Cache Doc",
        output_dir=scratch / "artifacts",
        image_payloads=payloads,
        provider_settings=settings,
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        ocr_runner=_runner,
        deps={"parse_semantic_fn": _parse_semantic_fn},
        resume_from_checkpoint=True,
    )
    assert third_run.status == "succeeded"
    assert third_bundle is not None
    assert calls["ocr"] == 3
    assert calls["parse"] == 2
    assert third_artifacts.reused_pages == [2]
    # Recovery uses a new persisted runtime incarnation internally, but the
    # public result keeps the stable logical root identity.  The trace below
    # proves that the artifact-loss recovery was actually executed.
    assert third_run.run_id == first_run.run_id
    root_trace = list(
        conversation_engine.read.get_nodes(
            where={
                "$and": [
                    {"entity_type": "workflow_step_exec"},
                    {"run_id": first_run.run_id},
                ]
            }
        )
    )
    assert {str(node.metadata.get("op")) for node in root_trace} >= {
        "ocr_root_invoke_pages",
        "ocr_root_invoke_grouping",
        "ocr_root_invoke_semantic",
    }
    recovery_trace = list(
        conversation_engine.read.get_nodes(
            where={
                "$and": [
                    {"entity_type": "workflow_step_exec"},
                    {"run_id": f"{first_run.run_id}|recovery-1"},
                ]
            }
        )
    )
    assert recovery_trace

    # The final artifact is not sufficient evidence that the stages belonged
    # to the same persisted execution.  Kogwistar's invocation edges are the
    # authoritative parent/child lineage record, so assert the complete
    # initial and recovery stage chains and reject duplicate accepted edges.
    lineage_edges = [
        edge
        for edge in conversation_engine.read.get_edges(limit=1000)
        if (edge.metadata or {}).get("relation") == "wf_invoked"
        and (edge.metadata or {}).get("parent_run_id", "").startswith(first_run.run_id)
    ]
    assert len({edge.id for edge in lineage_edges}) == len(lineage_edges)
    lineage = {
        (
            str((edge.metadata or {}).get("parent_run_id")),
            str((edge.metadata or {}).get("workflow_id")),
            str((edge.metadata or {}).get("invocation_key")),
            str((edge.metadata or {}).get("child_run_id")),
        )
        for edge in lineage_edges
    }
    assert {
        (
            first_run.run_id,
            "kg_doc_parser.ocr_pages.v1",
            "ocr-pages-v1",
            f"{first_run.run_id}|ocr-pages",
        ),
        (
            first_run.run_id,
            "kg_doc_parser.ocr_grouping.v1",
            "grouping-v1",
            f"{first_run.run_id}|grouping",
        ),
        (
            first_run.run_id,
            "kg_doc_parser.ingest.v1",
            "semantic-v1",
            f"{first_run.run_id}|semantic",
        ),
    }.issubset(lineage)
    assert {
        (
            f"{first_run.run_id}|recovery-1",
            "kg_doc_parser.ocr_pages.v1",
            "ocr-pages-recovery-v1",
            f"{first_run.run_id}|recovery-1|ocr-pages|recovery-1",
        ),
    }.issubset(lineage)
