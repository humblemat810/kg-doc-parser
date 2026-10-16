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

import base64
import hashlib
import importlib.util
import json
import logging
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from _kogwistar_test_helpers import (
    build_workflow_engine_triplet,
    drain_phase1_indexes_until_idle,
)
from PIL import Image, ImageDraw

import kg_doc_parser.workflow_ingest.ocr_pipeline as ocr_pipeline_module
import kg_doc_parser.workflow_ingest.ocr_runtime as ocr_runtime_module
from kg_doc_parser.models import (
    OCRClusterResponse,
    SplitPage,
    SplitPageMeta,
    TextCluster,
)
from kg_doc_parser.ocr import RawOCRResponse, regen_doc
from kg_doc_parser.workflow_ingest import (
    EmbeddingProviderConfig,
    OCRImagePayload,
    ProviderEndpointConfig,
    WorkflowIngestInput,
    WorkflowProviderSettings,
    prepare_ocr_workflow_input,
    run_ocr_ingest_workflow,
)
from kg_doc_parser.workflow_ingest.adapters import OCRPageJSON
from kg_doc_parser.workflow_ingest.document_adapters import apply_grouping_proposals
from kg_doc_parser.workflow_ingest.document_grouping import GroupProposal
from kg_doc_parser.workflow_ingest.document_models import (
    ChildRef,
    DocumentNode,
    DocumentTree,
    EvidenceSlice,
    SourceItem,
    SourceLocation,
    SourceSlice,
    Summary,
    SummaryPolicy,
)
from kg_doc_parser.workflow_ingest.ocr_pipeline import (
    GroupingCheckpointConflict,
    OCRPageProcessingContext,
    OCRQualityPolicy,
    OCRSourcePlan,
    OCRWorkflowStateStore,
    _compute_input_fingerprint,
    _evaluate_ocr_quality,
    _generate_tree_summaries,
    _materialize_image_payloads,
    _ocr_configuration_fingerprint,
    _ocr_page_metadata,
    _pdf_render_manifest_payload,
    _process_ocr_page,
    _render_pdf_to_images,
    _resolve_ocr_source_plan,
    _write_json_atomic,
    _write_png_atomic,
)
from kg_doc_parser.workflow_ingest.ocr_runtime import (
    _json_pages,
    _page_workflow_initial_state,
    _reconcile_page_checkpoint,
    _validate_root_resume_identity,
    run_grouping_workflow,
    run_ocr_page_workflow,
)
from kg_doc_parser.workflow_ingest.semantics import HydratedTextPointer, SemanticNode

pytestmark = [pytest.mark.workflow]


def test_materialize_image_payloads_uses_safe_atomic_page_paths(tmp_path: Path) -> None:
    rendered_dir = tmp_path / "rendered"
    payload = OCRImagePayload(
        page_number=2,
        image_bytes_b64=base64.b64encode(b"not-a-real-image").decode("ascii"),
        filename="../outside/escape.jpg",
    )

    resolved = _materialize_image_payloads([payload], rendered_dir)

    assert resolved == [(2, rendered_dir / "page_2.jpg")]
    assert resolved[0][1].read_bytes() == b"not-a-real-image"
    assert not (tmp_path / "outside" / "escape.jpg").exists()


def test_materialize_image_payloads_rejects_invalid_page_numbers_and_symlinks(
    tmp_path: Path,
) -> None:
    encoded = base64.b64encode(b"image").decode("ascii")
    with pytest.raises(ValueError, match="positive"):
        _materialize_image_payloads(
            [OCRImagePayload(page_number=0, image_bytes_b64=encoded)],
            tmp_path / "invalid",
        )
    with pytest.raises(ValueError, match="duplicate"):
        _materialize_image_payloads(
            [
                OCRImagePayload(page_number=1, image_bytes_b64=encoded),
                OCRImagePayload(page_number=1, image_bytes_b64=encoded),
            ],
            tmp_path / "duplicate",
        )

    rendered_dir = tmp_path / "symlinked"
    rendered_dir.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"original")
    page_link = rendered_dir / "page_1.png"
    try:
        page_link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is not available in this environment")
    with pytest.raises(ValueError, match="symlink"):
        _materialize_image_payloads(
            [OCRImagePayload(page_number=1, image_bytes_b64=encoded)],
            rendered_dir,
        )
    assert outside.read_bytes() == b"original"

    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    copied_dir = tmp_path / "copied"
    copied_dir.mkdir()
    copied_link = copied_dir / "page_1.png"
    try:
        copied_link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is not available in this environment")
    with pytest.raises(ValueError, match="symlink"):
        _materialize_image_payloads(
            [OCRImagePayload(page_number=1, image_path=str(source))],
            copied_dir,
        )
    assert outside.read_bytes() == b"original"


def test_materialize_image_payloads_rejects_out_of_order_page_numbers(tmp_path: Path) -> None:
    encoded = base64.b64encode(b"image").decode("ascii")

    with pytest.raises(ValueError, match="ascending order"):
        _materialize_image_payloads(
            [
                OCRImagePayload(page_number=2, image_bytes_b64=encoded),
                OCRImagePayload(page_number=1, image_bytes_b64=encoded),
            ],
            tmp_path / "out-of-order",
        )


def test_ocr_runtime_state_boundary_rejects_wrong_known_field_types() -> None:
    with pytest.raises(TypeError, match="page_cursor.*integer"):
        ocr_runtime_module._typed_state({"page_cursor": "1"})

    with pytest.raises(TypeError, match="completed_pages.*list of integers"):
        ocr_runtime_module._typed_state({"completed_pages": [1, "2"]})

    with pytest.raises(TypeError, match="page_sources contains a malformed entry"):
        ocr_runtime_module._typed_state(
            {"page_sources": [{"page_number": 1, "image_path": 42}]}
        )

    with pytest.raises(TypeError, match="source entry is malformed"):
        ocr_runtime_module._page_sources(
            {"page_sources": [{"page_number": 1}]}
        )


def test_ocr_runtime_child_state_uses_the_same_typed_boundary() -> None:
    with pytest.raises(TypeError, match="grouping child returned invalid typed state"):
        ocr_runtime_module._typed_child_state(
            {"page_cursor": "not-an-integer"},
            label="grouping child",
        )


def test_pdf_rasterization_regenerates_corrupt_cached_page_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rendered_dir = tmp_path / "rendered"
    rendered_dir.mkdir()
    corrupt_page = rendered_dir / "page_1.png"
    corrupt_page.write_bytes(b"partial-png")

    class _Reader:
        def __init__(self) -> None:
            self.pages = [object()]

    monkeypatch.setattr(ocr_pipeline_module, "PdfReader", lambda _: _Reader())
    import pdf2image

    calls: list[int] = []

    def _convert(*args: object, **kwargs: object) -> list[Image.Image]:
        calls.append(int(kwargs["first_page"]))
        return [Image.new("RGB", (8, 8), "white")]

    monkeypatch.setattr(pdf2image, "convert_from_path", _convert)

    paths = _render_pdf_to_images(tmp_path / "source.pdf", rendered_dir)

    assert paths == [corrupt_page]
    assert calls == [1]
    with Image.open(corrupt_page) as image:
        assert image.size == (8, 8)

_LOGGER = logging.getLogger(__name__)


def _terminate_interrupted_worker(process: subprocess.Popen[str]) -> None:
    """Use the strongest available process-stop primitive for recovery tests."""
    if os.name == "posix":
        os.kill(process.pid, signal.SIGKILL)
    else:
        process.terminate()


def _scratch(name: str) -> Path:
    root = Path("tests") / ".tmp_workflow_ingest_ocr"
    path = root / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _manual_case_dir(*, input_kind: str, provider: str, model: str) -> Path:
    safe_model = model.replace(":", "_").replace("/", "_")
    path = Path("tests") / ".tmp_workflow_ingest_ocr" / "manual_cases" / provider / safe_model / input_kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def _draw_test_image(path: Path, *, lines: list[str]) -> None:
    image = Image.new("RGB", (1200, 800), "white")
    draw = ImageDraw.Draw(image)
    y = 60
    for line in lines:
        draw.text((60, y), line, fill="black")
        y += 80
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG")


def _write_two_page_pdf(image_paths: list[Path], pdf_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    first, rest = images[0], images[1:]
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    first.save(pdf_path, "PDF", save_all=True, append_images=rest)
    for image in images:
        image.close()


@pytest.mark.ci
def test_ocr_root_resume_rejects_missing_or_changed_immutable_identity() -> None:
    expected = {
        "ocr_root_state_version": "ocr-root-v2",
        "document_id": "doc",
        "source_fingerprint": "source-a",
        "ocr_configuration_fingerprints": ["ocr-a"],
        "grouping_policy_version": "group-a",
    }

    _validate_root_resume_identity(
        expected,
        document_id="doc",
        source_fingerprint="source-a",
        configuration_fingerprints=("ocr-a",),
        grouping_policy_version="group-a",
        require_match=True,
    )
    with pytest.raises(RuntimeError, match="immutable identity fields"):
        _validate_root_resume_identity(
            {"document_id": "doc"},
            document_id="doc",
            source_fingerprint="source-a",
            configuration_fingerprints=("ocr-a",),
            grouping_policy_version="group-a",
            require_match=True,
        )
    changed = dict(expected, source_fingerprint="source-b")
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _validate_root_resume_identity(
            changed,
            document_id="doc",
            source_fingerprint="source-a",
            configuration_fingerprints=("ocr-a",),
            grouping_policy_version="group-a",
            require_match=True,
        )


@pytest.mark.ci
def test_grouping_runs_inside_kogwistar_runtime_and_is_resume_safe() -> None:
    scratch = _scratch("ocr_grouping_runtime")
    workflow_engine, conversation_engine, _knowledge_engine = build_workflow_engine_triplet(
        scratch / "engines",
        "in_memory",
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
            DocumentNode(node_id="leaf:item-a", kind="text", parts=(SourceSlice(item_id="item-a", start=0, end=5),)),
            DocumentNode(node_id="leaf:item-b", kind="text", parts=(SourceSlice(item_id="item-b", start=0, end=4),)),
            DocumentNode(node_id="leaf:item-c", kind="text", parts=(SourceSlice(item_id="item-c", start=0, end=5),)),
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
    )
    grouped, run_id = run_grouping_workflow(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        state_store=OCRWorkflowStateStore(scratch / "ocr-state.sqlite"),
        tree=tree,
        document_id=tree.document_id,
        grouping_policy_version="test-v1",
        grouping_proposals=(GroupProposal(kind="section", members=("leaf:item-a", "leaf:item-b")),),
        summary_policy=summary_policy,
        summary_generator=_summary_generator,
        run_id="ocr-grouping-test-run",
    )
    assert run_id == "ocr-grouping-test-run"
    assert any(node.kind == "section" for node in grouped.nodes)
    resumed, resumed_run_id = run_grouping_workflow(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        state_store=OCRWorkflowStateStore(scratch / "ocr-state.sqlite"),
        tree=tree,
        document_id=tree.document_id,
        grouping_policy_version="test-v1",
        grouping_proposals=(GroupProposal(kind="section", members=("leaf:item-a", "leaf:item-b")),),
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
        "persist_grouping_result",
    }


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
    workflow_engine, conversation_engine, _knowledge_engine = build_workflow_engine_triplet(
        _scratch("ocr_grouping_failed_nested_workflow") / "engines",
        "in_memory",
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
        ocr_runtime_module,
        "ensure_ocr_grouping_workflow_design",
        lambda _engine: None,
    )
    monkeypatch.setattr(
        ocr_runtime_module,
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


@pytest.mark.ci
def test_grouping_resume_rejects_frontier_that_disagrees_with_tree() -> None:
    scratch = _scratch("ocr_grouping_frontier_mismatch")
    workflow_engine, conversation_engine, _knowledge_engine = build_workflow_engine_triplet(
        scratch / "engines",
        "in_memory",
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
                children=(ChildRef(node_id="leaf:item-a"), ChildRef(node_id="leaf:item-b")),
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
                        parts=(SourceSlice(item_id=f"item-{name}", start=0, end=len(name)),),
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

    ref_engines = build_workflow_engine_triplet(reference_scratch / "engines", "in_memory")
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

    original_generate_tree_summaries = ocr_runtime_module._generate_tree_summaries
    interrupted = True

    def _interrupt_after_grouping(*args: object, **kwargs: object) -> DocumentTree:
        nonlocal interrupted
        if interrupted:
            interrupted = False
            raise RuntimeError("simulated interruption after grouping checkpoint")
        return original_generate_tree_summaries(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        ocr_runtime_module,
        "_generate_tree_summaries",
        _interrupt_after_grouping,
    )

    int_engines = build_workflow_engine_triplet(interrupted_scratch / "engines", "in_memory")
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
            ),
            generator=_summary_generator,
            grouping_policy_version="summary-v1",
            expected_tree_revision=initial_revision,
        )


@pytest.mark.ci
def test_ocr_pages_run_as_persisted_steps_and_terminal_resume_does_not_recall_provider() -> None:
    scratch = _scratch("ocr_page_runtime")
    image_paths = []
    for page_number, text in ((1, "page one"), (2, "page two")):
        image_path = scratch / "source" / f"page-{page_number}.png"
        _draw_test_image(image_path, lines=[text])
        image_paths.append((page_number, image_path))
    workflow_engine, conversation_engine, _knowledge_engine = build_workflow_engine_triplet(
        scratch / "engines",
        "in_memory",
    )
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        embedding=EmbeddingProviderConfig(provider="fake", model="fake-embed", dimension=1),
    )
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    calls = {"count": 0}

    def _runner(image_path: Path, page_number: int, provider_settings: WorkflowProviderSettings):
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

    def _unexpected_runner(image_path: Path, page_number: int, provider_settings: WorkflowProviderSettings):
        raise AssertionError(f"provider called during terminal resume for page {page_number}: {image_path}")

    resumed_context = replace(context, ocr_page_runner=_unexpected_runner)
    resumed_pages, resumed_completed, resumed_reused, resumed_run_id = run_ocr_page_workflow(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        context=resumed_context,
        source_plan=source_plan,
        progress_path=scratch / "ocr-progress.json",
        run_id=run_id,
        resume_from_checkpoint=True,
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

    page_two = source_plan.legacy_dir / "page_2.json"
    page_two.write_text(page_two.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    repair = _reconcile_page_checkpoint(
        state={
            "page_sources": [
                {"page_number": number, "image_path": str(path)} for number, path in image_paths
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


def test_new_page_checkpoint_rehydrates_pages_from_artifacts_without_raw_payload() -> None:
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
    state.update({"page_cursor": 2, "completed_pages": [1, 2], "reused_pages": []})

    assert "raw_pages" not in state
    pages = _json_pages(state)
    assert [page["printed_page_number"] for page in pages] == ["1", "2"]


@pytest.mark.ci
def test_ocr_page_resume_reopens_state_in_a_fresh_process() -> None:
    """A fresh process reuses accepted pages and processes only the gap."""
    scratch = _scratch("ocr_page_fresh_process_resume")
    for page_number, text in ((1, "fresh page one"), (2, "fresh page two")):
        _draw_test_image(
            scratch / "source" / f"page-{page_number}.png",
            lines=[text],
        )
    worker = Path(__file__).with_name("_ocr_fresh_process_worker.py")
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
    interrupted = subprocess.run(
        [sys.executable, str(worker), str(scratch), "interrupt"],
        capture_output=True,
        text=True,
        check=False,
        env=child_env,
    )
    assert interrupted.returncode != 0
    assert "fresh-process interruption" in interrupted.stderr

    resumed = subprocess.run(
        [sys.executable, str(worker), str(scratch), "resume"],
        capture_output=True,
        text=True,
        check=False,
        env=child_env,
    )
    assert resumed.returncode == 0, resumed.stderr
    result = json.loads(
        (scratch / "resume-result.json").read_text(encoding="utf-8")
    )
    assert result == {
        "pages": 2,
        "completed": [1, 2],
        "reused": [1],
        "run_id": "ocr|fresh-process-document",
        "calls": [2],
    }


@pytest.mark.ci
def test_ocr_root_resume_reopens_same_lineage_in_a_fresh_process() -> None:
    """A fresh process retries only the failed page and accepts one root result."""
    scratch = _scratch("ocr_root_fresh_process_resume")
    for page_number in (1, 2):
        _draw_test_image(
            scratch / "source" / f"page-{page_number}.png",
            lines=[f"root page {page_number}"],
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

    interrupted = subprocess.run(
        [sys.executable, str(worker), str(scratch), "interrupt"],
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
    assert interruption["calls"] == [1, 2], interruption

    resumed = subprocess.run(
        [sys.executable, str(worker), str(scratch), "resume"],
        capture_output=True,
        text=True,
        check=False,
        env=child_env,
        timeout=120,
    )
    assert resumed.returncode == 0, resumed.stderr
    result = json.loads(
        (scratch / "resume-result.json").read_text(encoding="utf-8")
    )
    assert result == {
        "status": "succeeded",
        "run_id": "ocr|fresh-root-document",
        "calls": [2],
        "has_grouping_tree": True,
    }


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

    resumed = subprocess.run(
        [sys.executable, str(worker), str(scratch), "resume"],
        capture_output=True,
        text=True,
        check=False,
        env=child_env,
        timeout=120,
    )
    assert resumed.returncode == 0, resumed.stderr
    result = json.loads(
        (scratch / "resume-result.json").read_text(encoding="utf-8")
    )
    assert result == {
        "status": "succeeded",
        "run_id": "ocr|fresh-root-document",
        "calls": [2],
        "has_grouping_tree": True,
    }


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
            assert (
                "WorkflowRunOwnershipError" in error_text
                or "Nested workflow 'kg_doc_parser.ocr_grouping.v1' returned status 'failure'"
                in error_text
            )
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
    result = json.loads(
        (scratch / "resume-result.json").read_text(encoding="utf-8")
    )
    assert result["run_id"] == "ocr-grouping|fresh-grouping-document"
    assert result["section_count"] == 1
    assert result["summary_count"] == 1


@pytest.mark.ci
@pytest.mark.slow
def test_ocr_root_resumes_after_page_63_and_matches_uninterrupted_reference() -> None:
    def _make_case(name: str) -> tuple[Path, list[OCRImagePayload]]:
        scratch = _scratch(name)
        payloads: list[OCRImagePayload] = []
        for page_number in range(1, 101):
            image_path = scratch / "source" / f"page-{page_number}.png"
            _draw_test_image(image_path, lines=[f"Acceptance page {page_number}"])
            payloads.append(OCRImagePayload(page_number=page_number, image_path=str(image_path)))
        return scratch, payloads

    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        parser=ProviderEndpointConfig(provider="fake", model="fake-parser"),
        embedding=EmbeddingProviderConfig(provider="fake", model="fake-embed", dimension=1),
        triage_enabled=False,
    )
    interrupted_scratch, interrupted_payloads = _make_case("ocr_root_100_interrupted")
    reference_scratch, reference_payloads = _make_case("ocr_root_100_reference")
    interrupted_engines = build_workflow_engine_triplet(
        interrupted_scratch / "engines", "in_memory"
    )
    reference_engines = build_workflow_engine_triplet(reference_scratch / "engines", "in_memory")
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
        return _fake_ocr_response(page_number, f"Acceptance page {page_number}")

    def _reference_runner(
        image_path: Path,
        page_number: int,
        provider_settings: WorkflowProviderSettings,
    ) -> OCRClusterResponse:
        del image_path, provider_settings
        reference_calls.append(page_number)
        return _fake_ocr_response(page_number, f"Acceptance page {page_number}")

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
        image_payloads=reference_payloads,
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
            image_payloads=interrupted_payloads,
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
        image_payloads=interrupted_payloads,
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
    assert interrupted_calls.count(63) == 2
    assert all(interrupted_calls.count(page_number) == 1 for page_number in range(1, 63))
    assert len(reference_calls) == 100


@pytest.mark.ci
def test_ocr_root_resume_rejects_changed_provider_before_replaying_pages() -> None:
    scratch = _scratch("ocr_root_resume_identity")
    payloads: list[OCRImagePayload] = []
    for page_number in (1, 2):
        image_path = scratch / "source" / f"page-{page_number}.png"
        _draw_test_image(image_path, lines=[f"Identity page {page_number}"])
        payloads.append(OCRImagePayload(page_number=page_number, image_path=str(image_path)))
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="identity-ocr-v1"),
        parser=ProviderEndpointConfig(provider="fake", model="identity-parser"),
        embedding=EmbeddingProviderConfig(provider="fake", model="identity-embed", dimension=1),
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
            run_id="ocr|ocr-root-identity",
        )
    assert calls == ["v1:1", "v1:2"]

    def _unexpected_runner(
        image_path: Path,
        page_number: int,
        provider_settings: WorkflowProviderSettings,
    ) -> OCRClusterResponse:
        raise AssertionError(f"provider replayed before identity validation: {image_path}:{page_number}")

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
            run_id="ocr|ocr-root-identity",
            resume_from_checkpoint=True,
        )


def _fake_ocr_response(page_number: int, text: str) -> OCRClusterResponse:
    return OCRClusterResponse(
        OCR_text_clusters=[
            TextCluster(
                text=text,
                bb_x_min=10.0,
                bb_x_max=500.0,
                bb_y_min=10.0,
                bb_y_max=50.0,
                cluster_number=0,
            )
        ],
        non_text_objects=[],
        is_empty_page=False,
        printed_page_number=str(page_number),
        meaningful_ordering=[0],
        page_x_min=0.0,
        page_x_max=1200.0,
        page_y_min=0.0,
        page_y_max=800.0,
        estimated_rotation_degrees=0.0,
        incomplete_words_on_edge=False,
        incomplete_text=False,
        data_loss_likelihood=0.0,
        scan_quality="high",
        contains_table=False,
    )


@pytest.mark.ci
@pytest.mark.parametrize(
    "update",
    [
        {"is_empty_page": True},
        {"meaningful_ordering": []},
        {"meaningful_ordering": [99]},
        {"meaningful_ordering": [0, 0]},
    ],
)
def test_ocr_cluster_response_rejects_inconsistent_empty_or_ordering_state(update: dict[str, object]) -> None:
    payload = _fake_ocr_response(1, "Alpha").model_dump()
    payload.update(update)

    with pytest.raises(ValueError, match="(is_empty_page|meaningful_ordering)"):
        OCRClusterResponse.model_validate(payload)


@pytest.mark.ci
def test_ocr_quality_policy_flags_and_can_reject_provider_loss_signals() -> None:
    response = _fake_ocr_response(1, "quality signal").model_copy(
        update={
            "incomplete_text": True,
            "incomplete_words_on_edge": True,
            "data_loss_likelihood": 0.75,
        }
    )
    flagged = _evaluate_ocr_quality(response, OCRQualityPolicy())
    assert {item.code for item in flagged} == {
        "incomplete_text",
        "incomplete_words_on_edge",
    }
    assert all(item.severity == "warning" for item in flagged)

    rejected = _evaluate_ocr_quality(
        response,
        OCRQualityPolicy(
            max_data_loss_likelihood=0.5,
            reject_incomplete_text=True,
            reject_incomplete_words_on_edge=True,
        ),
    )
    assert {item.code for item in rejected} == {
        "incomplete_text",
        "incomplete_words_on_edge",
        "data_loss_likelihood",
    }
    assert all(item.severity == "error" for item in rejected)


@pytest.mark.ci
def test_ocr_quality_policy_can_reject_empty_provider_output() -> None:
    payload = _fake_ocr_response(1, "quality signal").model_dump()
    payload.update({"OCR_text_clusters": [], "is_empty_page": True, "meaningful_ordering": []})
    response = OCRClusterResponse.model_validate(payload)

    flagged = _evaluate_ocr_quality(response, OCRQualityPolicy())
    assert [(item.code, item.severity) for item in flagged] == [("empty_ocr_output", "warning")]

    rejected = _evaluate_ocr_quality(response, OCRQualityPolicy(reject_empty_page=True))
    assert [(item.code, item.severity) for item in rejected] == [("empty_ocr_output", "error")]


@pytest.mark.ci
def test_ocr_quality_diagnostics_are_persisted_and_strict_policy_fails_closed() -> None:
    scratch = _scratch("ocr_quality_persistence")
    image_path = scratch / "images" / "page_1.png"
    _draw_test_image(image_path, lines=["quality persistence"])
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        embedding=EmbeddingProviderConfig(provider="fake", model="fake-embed", dimension=1),
    )

    def _incomplete_runner(
        image_path: Path,
        page_number: int,
        provider_settings: WorkflowProviderSettings,
    ) -> OCRClusterResponse:
        return _fake_ocr_response(page_number, "quality persistence").model_copy(
            update={"incomplete_text": True, "data_loss_likelihood": 0.25}
        )

    accepted = prepare_ocr_workflow_input(
        document_id="ocr-quality-warning-doc",
        title="OCR Quality Warning",
        output_dir=scratch / "warning",
        image_payloads=[OCRImagePayload(page_number=1, image_path=str(image_path))],
        provider_settings=settings,
        ocr_runner=_incomplete_runner,
    )
    persisted = json.loads((accepted.legacy_dir / "page_1.json").read_text(encoding="utf-8"))
    metadata = persisted["metadata"]
    assert metadata["ocr_quality_status"] == "flagged"
    assert metadata["ocr_quality_diagnostics"] == [
        {
            "code": "incomplete_text",
            "severity": "warning",
            "message": "provider marked the OCR text as incomplete",
        }
    ]

    with pytest.raises(RuntimeError, match="OCR preparation incomplete"):
        prepare_ocr_workflow_input(
            document_id="ocr-quality-strict-doc",
            title="OCR Quality Strict",
            output_dir=scratch / "strict",
            image_payloads=[OCRImagePayload(page_number=1, image_path=str(image_path))],
            provider_settings=settings,
            ocr_runner=_incomplete_runner,
            quality_policy=OCRQualityPolicy(reject_incomplete_text=True),
        )
    strict_store = OCRWorkflowStateStore(scratch / "strict" / "ocr-state.sqlite")
    strict_state = strict_store.get_page_state(
        document_id="ocr-quality-strict-doc", page_number=1, stage="ocr"
    )
    assert strict_state is not None
    assert strict_state.status == "failed"
    assert not (scratch / "strict" / "legacy" / "page_1.json").exists()


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


def test_workflow_ocr_structured_output_prefers_function_calling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []
    image_path = _scratch("ocr_function_calling") / "page_1.png"
    _draw_test_image(image_path, lines=["OCR structured output"])

    class _FakeStructured:
        def invoke(self, messages):
            return {"parsed": _fake_ocr_response(1, "OCR structured output")}

    class _FakeChat:
        def with_structured_output(self, schema, include_raw=True, **kwargs):
            captured.append({"schema": schema.__name__, "include_raw": include_raw, **kwargs})
            return _FakeStructured()

    monkeypatch.setattr(ocr_pipeline_module, "build_chat_model_for_role", lambda *args, **kwargs: _FakeChat())

    response = ocr_pipeline_module._run_live_ocr_page(
        image_path,
        1,
        WorkflowProviderSettings(
            ocr=ProviderEndpointConfig(provider="azure", model="gpt-5-nano", base_url="https://example.openai.azure.com/")
        ),
    )

    assert response.OCR_text_clusters[0].text == "OCR structured output"
    assert captured
    assert captured[0]["method"] == "json_schema"


@pytest.mark.ci
def test_live_ocr_provider_diagnostics_include_attempt_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = _scratch("ocr_provider_diagnostics") / "page_1.png"
    _draw_test_image(image_path, lines=["OCR diagnostics"])

    class _FakeStructured:
        def invoke(self, messages):
            return {"parsed": _fake_ocr_response(1, "OCR diagnostics")}

    class _FakeChat:
        def with_structured_output(self, schema, include_raw=True, **kwargs):
            return _FakeStructured()

    monkeypatch.setattr(ocr_pipeline_module, "build_chat_model_for_role", lambda *args, **kwargs: _FakeChat())
    diagnostics: list[dict[str, object]] = []
    response = ocr_pipeline_module._run_live_ocr_page(
        image_path,
        1,
        WorkflowProviderSettings(
            ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        ),
        provider_diagnostics_sink=diagnostics.append,
    )

    assert response.OCR_text_clusters[0].text == "OCR diagnostics"
    assert len(diagnostics) == 1
    assert diagnostics[0]["operation"] == "ocr_page"
    assert diagnostics[0]["call_role"] == "ocr"
    assert diagnostics[0]["strategy"] == "ocr"
    assert diagnostics[0]["attempt_index"] == 1
    assert diagnostics[0]["success"] is True
    assert isinstance(diagnostics[0]["elapsed_ms"], int)


@pytest.mark.ci
def test_prepare_ocr_workflow_input_resumes_completed_pages() -> None:
    scratch = _scratch("ocr_resume")
    images_dir = scratch / "images"
    image_paths = [images_dir / "page_1.png", images_dir / "page_2.png"]
    _draw_test_image(image_paths[0], lines=["Resume test page one"])
    _draw_test_image(image_paths[1], lines=["Resume test page two"])
    payloads = [
        OCRImagePayload(page_number=1, image_path=str(image_paths[0])),
        OCRImagePayload(page_number=2, image_path=str(image_paths[1])),
    ]
    calls = {"count": 0}

    def _runner(image_path: Path, page_number: int, provider_settings: WorkflowProviderSettings):
        calls["count"] += 1
        return _fake_ocr_response(page_number, f"Resume page {page_number}")

    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        embedding=EmbeddingProviderConfig(provider="fake", model="fake-embed", dimension=1),
    )
    first = prepare_ocr_workflow_input(
        document_id="ocr-resume-doc",
        title="OCR Resume Doc",
        output_dir=scratch / "artifacts",
        image_payloads=payloads,
        provider_settings=settings,
        ocr_runner=_runner,
    )
    second = prepare_ocr_workflow_input(
        document_id="ocr-resume-doc",
        title="OCR Resume Doc",
        output_dir=scratch / "artifacts",
        image_payloads=payloads,
        provider_settings=settings,
        ocr_runner=_runner,
    )

    assert calls["count"] == 2
    assert first.completed_pages == [1, 2]
    assert second.completed_pages == [1, 2]
    assert second.reused_pages == [1, 2]
    assert second.state_db_path.exists()
    progress = json.loads(second.progress_path.read_text(encoding="utf-8"))
    assert sorted(progress["pages"].keys()) == ["1", "2"]


@pytest.mark.ci
def test_prepare_ocr_workflow_retries_checkpoint_without_artifact() -> None:
    """A database-first crash cannot make a missing OCR artifact reusable."""
    scratch = _scratch("ocr_checkpoint_without_artifact")
    image_path = scratch / "images" / "page_1.png"
    _draw_test_image(image_path, lines=["checkpoint recovery"])
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        embedding=EmbeddingProviderConfig(provider="fake", model="fake-embed", dimension=1),
    )
    calls: list[int] = []

    def _runner(
        _image_path: Path,
        page_number: int,
        _provider_settings: WorkflowProviderSettings,
    ) -> OCRClusterResponse:
        calls.append(page_number)
        return _fake_ocr_response(page_number, "checkpoint recovery")

    first = prepare_ocr_workflow_input(
        document_id="ocr-checkpoint-without-artifact",
        title="OCR Checkpoint Without Artifact",
        output_dir=scratch / "artifacts",
        image_payloads=[OCRImagePayload(page_number=1, image_path=str(image_path))],
        provider_settings=settings,
        ocr_runner=_runner,
    )
    artifact = first.legacy_dir / "page_1.json"
    assert artifact.is_file()
    artifact.unlink()

    reopened = prepare_ocr_workflow_input(
        document_id="ocr-checkpoint-without-artifact",
        title="OCR Checkpoint Without Artifact",
        output_dir=scratch / "artifacts",
        image_payloads=[OCRImagePayload(page_number=1, image_path=str(image_path))],
        provider_settings=settings,
        ocr_runner=_runner,
    )

    assert calls == [1, 1]
    assert reopened.completed_pages == [1]
    assert reopened.reused_pages == []
    assert artifact.is_file()
    assert OCRWorkflowStateStore(reopened.state_db_path).should_skip_page(
        document_id="ocr-checkpoint-without-artifact",
        page_number=1,
        stage="ocr",
        content_hash=hashlib.sha256(image_path.read_bytes()).hexdigest(),
        artifact_path=artifact,
        config_fingerprint=ocr_pipeline_module._ocr_configuration_fingerprint(
            settings,
            candidate_models=["fake-ocr"],
        ),
    )


@pytest.mark.ci
def test_prepare_ocr_workflow_input_resumes_after_mid_document_interruption() -> None:
    """A failed page does not discard accepted pages from a long document."""

    scratch = _scratch("ocr_mid_document_resume")
    image_paths = [scratch / "images" / f"page_{page}.png" for page in range(1, 7)]
    for page, image_path in enumerate(image_paths, start=1):
        _draw_test_image(image_path, lines=[f"Long document page {page}"])
    payloads = [
        OCRImagePayload(page_number=page, image_path=str(image_path))
        for page, image_path in enumerate(image_paths, start=1)
    ]
    calls: list[int] = []
    interrupted = True

    def _runner(
        image_path: Path,
        page_number: int,
        provider_settings: WorkflowProviderSettings,
    ) -> OCRClusterResponse:
        nonlocal interrupted
        calls.append(page_number)
        if page_number == 4 and interrupted:
            interrupted = False
            raise RuntimeError("simulated process interruption")
        return _fake_ocr_response(page_number, f"Long document page {page_number}")

    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        embedding=EmbeddingProviderConfig(provider="fake", model="fake-embed", dimension=1),
    )
    with pytest.raises(RuntimeError, match=r"pages \[4\]"):
        prepare_ocr_workflow_input(
            document_id="ocr-mid-document-resume",
            title="OCR Mid Document Resume",
            output_dir=scratch / "artifacts",
            image_payloads=payloads,
            provider_settings=settings,
            ocr_runner=_runner,
        )

    resumed = prepare_ocr_workflow_input(
        document_id="ocr-mid-document-resume",
        title="OCR Mid Document Resume",
        output_dir=scratch / "artifacts",
        image_payloads=payloads,
        provider_settings=settings,
        ocr_runner=_runner,
    )

    assert resumed.completed_pages == list(range(1, 7))
    assert resumed.reused_pages == [1, 2, 3, 5, 6]
    assert calls == [1, 2, 3, 4, 5, 6, 4]
    assert resumed.document_tree is not None


@pytest.mark.ci
def test_ocr_ingest_forwards_runtime_checkpoint_resume_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    artifacts = SimpleNamespace(
        workflow_input=WorkflowIngestInput.from_text(
            document_id="ocr-runtime-resume",
            text="A persisted OCR page",
            title="OCR Runtime Resume",
        ),
        document_tree=None,
    )

    def _fake_build_preparation(**kwargs: object) -> object:
        captured["preparation_kwargs"] = kwargs
        return SimpleNamespace()

    def _fake_run_ocr_root_workflow(**kwargs: object):
        captured["runtime_kwargs"] = kwargs
        return (
            SimpleNamespace(
                run_id="run|ocr-runtime-resume",
                status="suspended",
                final_state={},
            ),
            None,
            artifacts,
        )

    monkeypatch.setattr(ocr_pipeline_module, "_build_ocr_page_preparation", _fake_build_preparation)
    monkeypatch.setattr(
        "kg_doc_parser.workflow_ingest.ocr_runtime.run_ocr_root_workflow",
        _fake_run_ocr_root_workflow,
    )

    run, bundle, returned_artifacts = run_ocr_ingest_workflow(
        document_id="ocr-runtime-resume",
        title="OCR Runtime Resume",
        output_dir=_scratch("ocr_runtime_resume") / "artifacts",
        workflow_engine=object(),
        conversation_engine=object(),
        image_payloads=[OCRImagePayload(page_number=1, image_path="ignored.png")],
        run_id="run|ocr-runtime-resume",
        resume_from_checkpoint=True,
    )

    runtime_kwargs = captured["runtime_kwargs"]
    assert isinstance(runtime_kwargs, dict)
    assert runtime_kwargs["run_id"] == "run|ocr-runtime-resume"
    assert runtime_kwargs["resume_from_checkpoint"] is True
    assert run.status == "suspended"
    assert bundle is None
    assert returned_artifacts is artifacts


@pytest.mark.ci
def test_prepare_ocr_workflow_input_checkpoints_cross_page_grouping() -> None:
    scratch = _scratch("ocr_live_cross_page_grouping")
    image_paths = [scratch / "images" / f"page_{page}.png" for page in (1, 2, 3)]
    for page, image_path in enumerate(image_paths, start=1):
        _draw_test_image(image_path, lines=[f"Continuation page {page}"])
    payloads = [
        OCRImagePayload(page_number=page, image_path=str(image_path))
        for page, image_path in enumerate(image_paths, start=1)
    ]

    def _runner(image_path: Path, page_number: int, provider_settings: WorkflowProviderSettings):
        response = _fake_ocr_response(page_number, f"Continuation page {page_number}")
        if page_number == 1:
            response = response.model_copy(update={"incomplete_text": True})
        return response

    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        embedding=EmbeddingProviderConfig(provider="fake", model="fake-embed", dimension=1),
    )
    artifacts = prepare_ocr_workflow_input(
        document_id="ocr-live-cross-page-doc",
        title="OCR Live Cross Page Doc",
        output_dir=scratch / "artifacts",
        image_payloads=payloads,
        provider_settings=settings,
        ocr_runner=_runner,
    )

    assert artifacts.document_tree is not None
    continuation_nodes = [
        node
        for node in artifacts.document_tree.nodes
        if node.kind == "layout:cross-page-continuation"
    ]
    assert len(continuation_nodes) == 1
    continuation = continuation_nodes[0]
    assert len(continuation.children) == 2
    assert {
        artifacts.document_tree.source_items[
            next(
                index
                for index, item in enumerate(artifacts.document_tree.source_items)
                if item.item_id == child.node_id.removeprefix("leaf:")
            )
        ].location.page_index
        for child in continuation.children
    } == {0, 1}

    persisted = OCRWorkflowStateStore(artifacts.state_db_path).load_document_tree(
        document_id="ocr-live-cross-page-doc",
        source_fingerprint=artifacts.document_tree.source_fingerprint(),
        grouping_policy_version="baseline-v1",
    )
    assert persisted is not None
    assert any(node.kind == "layout:cross-page-continuation" for node in persisted.nodes)


@pytest.mark.ci
def test_prepare_ocr_workflow_input_generates_grounded_summary_with_journal() -> None:
    scratch = _scratch("ocr_summary_journal")
    image_paths = [scratch / "images" / f"page_{page}.png" for page in (1, 2)]
    for page, image_path in enumerate(image_paths, start=1):
        _draw_test_image(image_path, lines=[f"Summary page {page}"])
    payloads = [
        OCRImagePayload(page_number=page, image_path=str(image_path))
        for page, image_path in enumerate(image_paths, start=1)
    ]
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        embedding=EmbeddingProviderConfig(provider="fake", model="fake-embed", dimension=1),
    )

    def _generator(
        *,
        tree: DocumentTree,
        node_id: str,
        source_items: tuple[SourceItem, ...],
        input_fingerprint: str,
    ) -> Summary:
        assert node_id == tree.root_id
        return Summary(
            status="generated",
            text="A grounded summary",
            evidence_item_ids=tuple(item.item_id for item in source_items),
            provider_model="fake-summary",
            prompt_revision="summary-v1",
            input_fingerprint=input_fingerprint,
        )

    artifacts = prepare_ocr_workflow_input(
        document_id="ocr-summary-journal-doc",
        title="OCR Summary Journal Doc",
        output_dir=scratch / "artifacts",
        image_payloads=payloads,
        provider_settings=settings,
        ocr_runner=lambda image_path, page_number, provider_settings: _fake_ocr_response(
            page_number, f"Summary page {page_number}"
        ),
        summary_policy=SummaryPolicy(
            eligible_kinds=("document",),
            minimum_source_items=2,
            minimum_source_chars=1,
        ),
        summary_generator=_generator,
    )

    assert artifacts.document_tree is not None
    root = next(node for node in artifacts.document_tree.nodes if node.node_id == "root")
    assert root.summary is not None
    assert root.summary.status == "generated"
    assert root.summary.input_fingerprint == artifacts.document_tree.summary_input_fingerprint("root")
    attempts = OCRWorkflowStateStore(artifacts.state_db_path).list_summary_attempts(
        document_id="ocr-summary-journal-doc"
    )
    assert [attempt["status"] for attempt in attempts] == ["pending", "generated"]


@pytest.mark.ci
def test_grounded_summary_generator_receives_only_owned_source_slices() -> None:
    scratch = _scratch("grounded_summary_slices")
    tree = DocumentTree(
        document_id="grounded-summary-doc",
        source_revision="ocr-v1:grounded-summary",
        tree_revision=1,
        root_id="root",
        source_items=(
            SourceItem(
                item_id="page-1",
                kind="text",
                text="alphaonebetatwo",
                location=SourceLocation(page_index=0, accuracy="page"),
            ),
        ),
        nodes=(
            DocumentNode(
                node_id="section-alpha",
                kind="section",
                children=(ChildRef(node_id="leaf-alpha"), ChildRef(node_id="leaf-one")),
            ),
            DocumentNode(
                node_id="section-beta",
                kind="section",
                children=(ChildRef(node_id="leaf-beta"), ChildRef(node_id="leaf-two")),
            ),
            DocumentNode(
                node_id="leaf-alpha",
                kind="text",
                parts=(SourceSlice(item_id="page-1", start=0, end=5),),
            ),
            DocumentNode(
                node_id="leaf-one",
                kind="text",
                parts=(SourceSlice(item_id="page-1", start=5, end=8, separator_before=" "),),
            ),
            DocumentNode(
                node_id="leaf-beta",
                kind="text",
                parts=(SourceSlice(item_id="page-1", start=8, end=12),),
            ),
            DocumentNode(
                node_id="leaf-two",
                kind="text",
                parts=(SourceSlice(item_id="page-1", start=12, end=15, separator_before=" "),),
            ),
            DocumentNode(
                node_id="root",
                kind="document",
                children=(
                    ChildRef(node_id="section-alpha"),
                    ChildRef(node_id="section-beta", separator_before=" "),
                ),
            ),
        ),
    )
    seen: dict[str, tuple[EvidenceSlice, ...]] = {}

    def grounded_generator(
        *,
        tree: DocumentTree,
        node_id: str,
        evidence_slices: tuple[EvidenceSlice, ...],
        input_fingerprint: str,
    ) -> Summary:
        seen[node_id] = evidence_slices
        return Summary(
            status="generated",
            text=f"summary:{node_id}",
            evidence_item_ids=(evidence_slices[0].item_id,),
            provider_model="fake-grounded-summary",
            prompt_revision="summary-v1",
            input_fingerprint=input_fingerprint,
        )

    policy = SummaryPolicy(
        eligible_kinds=("section",),
        minimum_source_items=1,
        minimum_source_chars=1,
        minimum_information_gain_chars=0,
    )
    updated = _generate_tree_summaries(
        tree,
        document_id=tree.document_id,
        state_store=OCRWorkflowStateStore(scratch / "ocr-state.sqlite"),
        policy=policy,
        generator=None,
        grounded_generator=grounded_generator,
        grouping_policy_version="grounded-summary-v1",
    )

    assert [slice_.text for slice_ in seen["section-alpha"]] == ["alpha", "one"]
    assert [(slice_.start, slice_.end) for slice_ in seen["section-alpha"]] == [(0, 5), (5, 8)]
    assert [slice_.text for slice_ in seen["section-beta"]] == ["beta", "two"]
    assert [(slice_.start, slice_.end) for slice_ in seen["section-beta"]] == [(8, 12), (12, 15)]
    assert all(node.summary is not None for node in updated.nodes if node.kind == "section")


def test_summary_cache_reuses_exact_generation_and_slice_identity() -> None:
    scratch = _scratch("ocr_summary_cache_identity")
    tree = DocumentTree(
        document_id="summary-cache-doc",
        source_revision="ocr-v1:summary-cache",
        tree_revision=1,
        root_id="root",
        source_items=(
            SourceItem(
                item_id="shared",
                kind="text",
                text="alpha beta",
                location=SourceLocation(page_index=0, accuracy="page"),
            ),
        ),
        nodes=(
            DocumentNode(
                node_id="leaf:left",
                kind="text",
                parts=(SourceSlice(item_id="shared", start=0, end=6),),
            ),
            DocumentNode(
                node_id="leaf:right",
                kind="text",
                parts=(SourceSlice(item_id="shared", start=6, end=10),),
            ),
            DocumentNode(
                node_id="root",
                kind="document",
                children=(ChildRef(node_id="leaf:left"), ChildRef(node_id="leaf:right")),
            ),
        ),
    )
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    initial_revision = store.save_document_tree(
        document_id=tree.document_id,
        tree=tree,
        grouping_policy_version="summary-cache-v1",
        require_no_existing_tree=True,
    )
    policy = SummaryPolicy(
        eligible_kinds=("document",),
        minimum_source_items=1,
        minimum_source_chars=1,
        minimum_information_gain_chars=0,
    )
    calls = {"count": 0}

    def _generator(
        *,
        tree: DocumentTree,
        node_id: str,
        source_items: tuple[SourceItem, ...],
        input_fingerprint: str,
    ) -> Summary:
        del tree
        calls["count"] += 1
        return Summary(
            status="generated",
            text=f"summary:{node_id}",
            evidence_item_ids=tuple(item.item_id for item in source_items),
            provider_model="fake-summary",
            prompt_revision="summary-v1",
            input_fingerprint=input_fingerprint,
        )

    generated = _generate_tree_summaries(
        tree,
        document_id=tree.document_id,
        state_store=store,
        policy=policy,
        generator=_generator,
        grouping_policy_version="summary-cache-v1",
        expected_tree_revision=initial_revision,
    )
    root_summary = next(node.summary for node in generated.nodes if node.node_id == "root")
    assert root_summary is not None
    assert root_summary.evidence_slice_fingerprint == generated.summary_slice_fingerprint("root")

    # Reopen the persisted tree's structure without its derived summary. The
    # same generation key must reuse the accepted journal row, not call the
    # provider again.
    without_summary = generated.model_copy(
        update={
            "tree_revision": tree.tree_revision,
            "nodes": tuple(node.model_copy(update={"summary": None}) for node in generated.nodes),
        }
    )
    current_revision = store.document_tree_revision(document_id=tree.document_id)
    assert current_revision is not None
    reused = _generate_tree_summaries(
        without_summary,
        document_id=tree.document_id,
        state_store=store,
        policy=policy,
        generator=_generator,
        grouping_policy_version="summary-cache-v1",
        expected_tree_revision=current_revision,
    )
    assert calls["count"] == 1
    assert next(node.summary for node in reused.nodes if node.node_id == "root") == root_summary
    attempts = OCRWorkflowStateStore(scratch / "ocr-state.sqlite").list_summary_attempts(
        document_id=tree.document_id
    )
    assert [attempt["status"] for attempt in attempts] == ["pending", "generated"]

    # A retry with the already restored tree must be a true no-op: it must not
    # advance the tree revision or append another accepted journal result.
    persisted_revision = store.document_tree_revision(document_id=tree.document_id)
    assert persisted_revision is not None
    rerun = _generate_tree_summaries(
        reused,
        document_id=tree.document_id,
        state_store=store,
        policy=policy,
        generator=_generator,
        grouping_policy_version="summary-cache-v1",
        expected_tree_revision=persisted_revision,
    )
    assert rerun.model_dump(mode="json") == reused.model_dump(mode="json")
    assert store.document_tree_revision(document_id=tree.document_id) == persisted_revision
    attempts = store.list_summary_attempts(document_id=tree.document_id)
    assert [attempt["status"] for attempt in attempts] == ["pending", "generated"]


@pytest.mark.ci
def test_summary_generation_fingerprint_invalidates_persisted_cache() -> None:
    scratch = _scratch("ocr_summary_generation_fingerprint")
    tree = DocumentTree(
        document_id="summary-generation-doc",
        source_revision="ocr-v1:summary-generation",
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
                node_id="leaf-a",
                kind="text",
                parts=(SourceSlice(item_id="item-a", start=0, end=5),),
            ),
            DocumentNode(
                node_id="leaf-b",
                kind="text",
                parts=(SourceSlice(item_id="item-b", start=0, end=4),),
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
        grouping_policy_version="summary-cache-v1",
        require_no_existing_tree=True,
    )
    calls = {"count": 0}

    def _generator(
        *,
        tree: DocumentTree,
        node_id: str,
        source_items: tuple[SourceItem, ...],
        input_fingerprint: str,
    ) -> Summary:
        del tree
        calls["count"] += 1
        return Summary(
            status="generated",
            text=f"summary:{calls['count']}",
            evidence_item_ids=tuple(item.item_id for item in source_items),
            provider_model="fake-summary",
            prompt_revision="summary-v1",
            input_fingerprint=input_fingerprint,
        )

    first = _generate_tree_summaries(
        tree,
        document_id=tree.document_id,
        state_store=store,
        policy=SummaryPolicy(
            eligible_kinds=("document",),
            minimum_source_items=1,
            minimum_source_chars=1,
            minimum_information_gain_chars=0,
            generation_fingerprint="provider-a:model-a:prompt-a",
        ),
        generator=_generator,
        grouping_policy_version="summary-cache-v1",
        expected_tree_revision=initial_revision,
    )
    current_revision = store.document_tree_revision(document_id=tree.document_id)
    assert current_revision is not None
    without_summary = first.model_copy(
        update={"nodes": tuple(node.model_copy(update={"summary": None}) for node in first.nodes)}
    )
    second = _generate_tree_summaries(
        without_summary,
        document_id=tree.document_id,
        state_store=store,
        policy=SummaryPolicy(
            eligible_kinds=("document",),
            minimum_source_items=1,
            minimum_source_chars=1,
            minimum_information_gain_chars=0,
            generation_fingerprint="provider-b:model-b:prompt-b",
        ),
        generator=_generator,
        grouping_policy_version="summary-cache-v1",
        expected_tree_revision=current_revision,
    )

    assert calls["count"] == 2
    assert next(node.summary for node in second.nodes if node.node_id == "root").text == "summary:2"


@pytest.mark.ci
def test_summary_failure_is_journaled_without_failing_ocr_ingestion() -> None:
    scratch = _scratch("ocr_summary_failure")
    image_paths = [scratch / "images" / f"page_{page}.png" for page in (1, 2)]
    for page, image_path in enumerate(image_paths, start=1):
        _draw_test_image(image_path, lines=[f"Summary failure page {page}"])
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        embedding=EmbeddingProviderConfig(provider="fake", model="fake-embed", dimension=1),
    )

    def _generator(
        *,
        tree: DocumentTree,
        node_id: str,
        source_items: tuple[SourceItem, ...],
        input_fingerprint: str,
    ) -> Summary:
        raise RuntimeError("summary provider unavailable")

    artifacts = prepare_ocr_workflow_input(
        document_id="ocr-summary-failure-doc",
        title="OCR Summary Failure Doc",
        output_dir=scratch / "artifacts",
        image_payloads=[
            OCRImagePayload(page_number=page, image_path=str(image_path))
            for page, image_path in enumerate(image_paths, start=1)
        ],
        provider_settings=settings,
        ocr_runner=lambda image_path, page_number, provider_settings: _fake_ocr_response(
            page_number, f"Summary failure page {page_number}"
        ),
        summary_policy=SummaryPolicy(
            eligible_kinds=("document",),
            minimum_source_items=1,
            minimum_source_chars=1,
        ),
        summary_generator=_generator,
    )

    assert artifacts.document_tree is not None
    assert next(node for node in artifacts.document_tree.nodes if node.node_id == "root").summary is None
    attempts = OCRWorkflowStateStore(artifacts.state_db_path).list_summary_attempts(
        document_id="ocr-summary-failure-doc"
    )
    assert [attempt["status"] for attempt in attempts] == ["pending", "failed"]
    assert "summary provider unavailable" in str(attempts[-1]["error_message"])


@pytest.mark.ci
def test_failed_summary_is_retryable_without_replaying_ocr() -> None:
    scratch = _scratch("ocr_summary_retry")
    image_paths = [scratch / "images" / f"page_{page}.png" for page in (1, 2)]
    for page, image_path in enumerate(image_paths, start=1):
        _draw_test_image(image_path, lines=[f"Summary retry page {page}"])
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        embedding=EmbeddingProviderConfig(provider="fake", model="fake-embed", dimension=1),
    )
    ocr_calls = {"count": 0}

    def _runner(
        image_path: Path,
        page_number: int,
        provider_settings: WorkflowProviderSettings,
    ) -> OCRClusterResponse:
        del image_path, provider_settings
        ocr_calls["count"] += 1
        return _fake_ocr_response(page_number, f"Summary retry page {page_number}")

    policy = SummaryPolicy(
        eligible_kinds=("document",),
        minimum_source_items=1,
        minimum_source_chars=1,
    )

    def _failed_generator(
        *,
        tree: DocumentTree,
        node_id: str,
        source_items: tuple[SourceItem, ...],
        input_fingerprint: str,
    ) -> Summary:
        del tree, node_id, source_items, input_fingerprint
        raise RuntimeError("transient summary outage")

    artifacts = prepare_ocr_workflow_input(
        document_id="ocr-summary-retry-doc",
        title="OCR Summary Retry Doc",
        output_dir=scratch / "artifacts",
        image_payloads=[
            OCRImagePayload(page_number=page, image_path=str(image_path))
            for page, image_path in enumerate(image_paths, start=1)
        ],
        provider_settings=settings,
        ocr_runner=_runner,
        summary_policy=policy,
        summary_generator=_failed_generator,
    )
    assert artifacts.document_tree is not None
    assert ocr_calls["count"] == 2
    def _successful_generator(
        *,
        tree: DocumentTree,
        node_id: str,
        source_items: tuple[SourceItem, ...],
        input_fingerprint: str,
    ) -> Summary:
        return Summary(
            status="generated",
            text=f"retry:{node_id}",
            evidence_item_ids=tuple(item.item_id for item in source_items),
            provider_model="fake-summary",
            prompt_revision="summary-v1",
            input_fingerprint=input_fingerprint,
        )

    retried = prepare_ocr_workflow_input(
        document_id="ocr-summary-retry-doc",
        title="OCR Summary Retry Doc",
        output_dir=scratch / "artifacts",
        image_payloads=[
            OCRImagePayload(page_number=page, image_path=str(image_path))
            for page, image_path in enumerate(image_paths, start=1)
        ],
        provider_settings=settings,
        ocr_runner=_runner,
        summary_policy=policy,
        summary_generator=_successful_generator,
    )
    assert retried.document_tree is not None
    root = next(node for node in retried.document_tree.nodes if node.node_id == retried.document_tree.root_id)
    assert root.summary is not None
    assert root.summary.text == "retry:root"
    assert ocr_calls["count"] == 2
    attempts = [
        attempt
        for attempt in OCRWorkflowStateStore(retried.state_db_path).list_summary_attempts(
            document_id="ocr-summary-retry-doc"
        )
        if attempt["node_id"] == "root"
    ]
    statuses = [attempt["status"] for attempt in attempts]
    assert statuses.count("pending") == 2
    assert statuses.count("failed") == 1
    assert statuses.count("generated") == 1


@pytest.mark.ci
def test_prepare_ocr_workflow_input_rebuilds_state_db_from_existing_artifacts() -> None:
    scratch = _scratch("ocr_rebuild")
    image_path = scratch / "images" / "page_1.png"
    _draw_test_image(image_path, lines=["Rebuild page"])
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        embedding=EmbeddingProviderConfig(provider="fake", model="fake-embed", dimension=1),
    )
    calls = {"count": 0}

    def _runner(image_path: Path, page_number: int, provider_settings: WorkflowProviderSettings):
        calls["count"] += 1
        return _fake_ocr_response(page_number, "Rebuild page")

    first = prepare_ocr_workflow_input(
        document_id="ocr-rebuild-doc",
        title="OCR Rebuild Doc",
        output_dir=scratch / "artifacts",
        image_payloads=[OCRImagePayload(page_number=1, image_path=str(image_path))],
        provider_settings=settings,
        ocr_runner=_runner,
    )
    first.state_db_path.unlink()
    second = prepare_ocr_workflow_input(
        document_id="ocr-rebuild-doc",
        title="OCR Rebuild Doc",
        output_dir=scratch / "artifacts",
        image_payloads=[OCRImagePayload(page_number=1, image_path=str(image_path))],
        provider_settings=settings,
        ocr_runner=_runner,
    )

    assert calls["count"] == 1
    assert second.state_db_path.exists()
    assert second.reused_pages == [1]


@pytest.mark.ci
def test_ocr_state_rebuild_ignores_corrupt_progress_metadata() -> None:
    scratch = _scratch("ocr_corrupt_progress")
    rendered_dir = scratch / "rendered"
    legacy_dir = scratch / "legacy"
    rendered_dir.mkdir()
    legacy_dir.mkdir()
    rendered = rendered_dir / "page_1.png"
    rendered.write_bytes(b"authoritative rendered page")
    (legacy_dir / "page_1.json").write_text("{}", encoding="utf-8")
    progress_path = scratch / "ocr-progress.json"
    progress_path.write_text("{truncated", encoding="utf-8")

    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    assert store.rebuild_from_artifacts(
        document_id="corrupt-progress-doc",
        title="Corrupt Progress",
        source_kind="image",
        total_pages=1,
        input_fingerprint="fingerprint",
        rendered_dir=rendered_dir,
        legacy_dir=legacy_dir,
        progress_path=progress_path,
    )

    state = store.get_page_state(
        document_id="corrupt-progress-doc", page_number=1, stage="ocr"
    )
    assert state is not None
    assert state.content_hash == hashlib.sha256(rendered.read_bytes()).hexdigest()
    assert state.status == "pending"


@pytest.mark.ci
def test_ocr_state_rebuild_rejects_non_digest_progress_hash() -> None:
    scratch = _scratch("ocr_invalid_progress_hash")
    rendered_dir = scratch / "rendered"
    legacy_dir = scratch / "legacy"
    rendered_dir.mkdir()
    legacy_dir.mkdir()
    rendered = rendered_dir / "page_1.png"
    rendered.write_bytes(b"authoritative rendered page")
    (legacy_dir / "page_1.json").write_text("{}", encoding="utf-8")
    progress_path = scratch / "ocr-progress.json"
    progress_path.write_text(
        json.dumps({"pages": {"1": {"image_sha256": "pretend-hash"}}}),
        encoding="utf-8",
    )

    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.rebuild_from_artifacts(
        document_id="invalid-progress-hash-doc",
        title="Invalid Progress Hash",
        source_kind="image",
        total_pages=1,
        input_fingerprint="fingerprint",
        rendered_dir=rendered_dir,
        legacy_dir=legacy_dir,
        progress_path=progress_path,
    )

    state = store.get_page_state(
        document_id="invalid-progress-hash-doc", page_number=1, stage="ocr"
    )
    assert state is not None
    assert state.content_hash == hashlib.sha256(rendered.read_bytes()).hexdigest()
    assert state.status == "pending"


@pytest.mark.ci
def test_ocr_state_store_round_trips_grouped_tree_and_invalidates_changed_source() -> None:
    scratch = _scratch("ocr_grouping_checkpoint")
    tree = DocumentTree(
        document_id="grouped-doc",
        source_revision="ocr-v1",
        tree_revision=1,
        root_id="root",
        source_items=(
            SourceItem(
                item_id="a",
                kind="text",
                location=SourceLocation(page_index=0, accuracy="page"),
                text="alpha",
            ),
            SourceItem(
                item_id="b",
                kind="text",
                location=SourceLocation(page_index=0, accuracy="page"),
                text="beta",
            ),
            SourceItem(
                item_id="c",
                kind="text",
                location=SourceLocation(page_index=0, accuracy="page"),
                text="gamma",
            ),
        ),
        nodes=(
            DocumentNode(
                node_id="root",
                kind="document",
                children=(ChildRef(node_id="group"), ChildRef(node_id="leaf")),
            ),
            DocumentNode(
                node_id="group",
                kind="section",
                parts=(SourceSlice(item_id="a", start=0, end=5), SourceSlice(item_id="b", start=0, end=4, separator_before="\n")),
            ),
            DocumentNode(
                node_id="leaf",
                kind="text",
                parts=(SourceSlice(item_id="c", start=0, end=5),),
            ),
        ),
    )
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="grouped-doc",
        title="Grouped",
        source_kind="image",
        total_pages=1,
        input_fingerprint="source-v1",
    )
    revision = store.save_document_tree(document_id="grouped-doc", tree=tree, grouping_policy_version="group-v1")
    assert len(revision) == 64
    assert store.document_tree_revision(document_id="grouped-doc") == revision
    with pytest.raises(RuntimeError, match="create conflict"):
        store.save_document_tree(
            document_id="grouped-doc",
            tree=tree,
            grouping_policy_version="group-v1",
            require_no_existing_tree=True,
        )
    with pytest.raises(RuntimeError, match="revision conflict"):
        store.save_document_tree(
            document_id="grouped-doc",
            tree=tree,
            grouping_policy_version="group-v1",
            expected_tree_revision="stale-revision",
        )
    with pytest.raises(RuntimeError, match="requires an expected tree revision"):
        store.save_document_tree(
            document_id="grouped-doc",
            tree=tree,
            grouping_policy_version="group-v1",
        )
    assert store.load_grouping_frontier(
        document_id="grouped-doc",
        source_fingerprint=tree.source_fingerprint(),
        grouping_policy_version="group-v1",
    ) == ("group", "leaf")
    assert store.load_grouping_frontier(
        document_id="grouped-doc",
        source_fingerprint=tree.source_fingerprint(),
        grouping_policy_version="different-policy",
    ) is None
    restored = store.load_document_tree(document_id="grouped-doc", source_fingerprint=tree.source_fingerprint())
    assert restored is not None
    assert restored.model_dump(mode="json") == tree.model_dump(mode="json")
    assert (
        store.load_document_tree(
            document_id="grouped-doc",
            source_fingerprint=tree.source_fingerprint(),
            grouping_policy_version="different-policy",
        )
        is None
    )

    with sqlite3.connect(store.db_path) as conn:
        conn.execute("DELETE FROM document_tree_state WHERE document_id = ?", ("grouped-doc",))
    with pytest.raises(RuntimeError, match="create conflict.*frontier"):
        store.save_document_tree(
            document_id="grouped-doc",
            tree=tree,
            grouping_policy_version="group-v1",
            require_no_existing_tree=True,
        )

    with pytest.raises(RuntimeError, match="orphaned frontier"):
        store.save_document_tree(
            document_id="grouped-doc",
            tree=tree,
            grouping_policy_version="group-v1",
        )

    inconsistent_store = OCRWorkflowStateStore(scratch / "inconsistent.sqlite")
    inconsistent_store.ensure_document(
        document_id="grouped-doc",
        title="Grouped",
        source_kind="image",
        total_pages=1,
        input_fingerprint="source-v1",
    )
    inconsistent_revision = inconsistent_store.save_document_tree(
        document_id="grouped-doc",
        tree=tree,
        grouping_policy_version="group-v1",
    )
    with sqlite3.connect(inconsistent_store.db_path) as conn:
        conn.execute("DELETE FROM grouping_frontier_state WHERE document_id = ?", ("grouped-doc",))
    with pytest.raises(RuntimeError, match="frontier is missing"):
        inconsistent_store.save_document_tree(
            document_id="grouped-doc",
            tree=tree,
            grouping_policy_version="group-v1",
            expected_tree_revision=inconsistent_revision,
        )

    store.ensure_document(
        document_id="grouped-doc",
        title="Grouped",
        source_kind="image",
        total_pages=1,
        input_fingerprint="source-v2",
    )
    assert store.load_document_tree(document_id="grouped-doc") is None
    assert store.load_grouping_frontier(document_id="grouped-doc") is None


@pytest.mark.ci
def test_ocr_completion_requires_exact_expected_page_numbers() -> None:
    scratch = _scratch("ocr_completion_page_set")
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="page-set-doc",
        title="Page Set",
        source_kind="image",
        total_pages=1,
        input_fingerprint="fingerprint",
    )
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            """
            INSERT INTO page_state (
                document_id, page_number, stage, attempt_count, status,
                content_hash, artifact_path, last_error, last_model, last_attempted_ts
            ) VALUES (?, ?, 'ocr', 1, 'completed', ?, NULL, NULL, NULL, ?)
            """,
            ("page-set-doc", 999, "hash", 0.0),
        )

    assert not store.refresh_document_completion(document_id="page-set-doc")
    assert not store.read_document_completed(document_id="page-set-doc")


@pytest.mark.ci
def test_prepare_ocr_workflow_input_writes_legacy_compatible_artifacts() -> None:
    scratch = _scratch("ocr_legacy_compat")
    image_path = scratch / "images" / "page_1.png"
    _draw_test_image(image_path, lines=["Compatibility page"])
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        embedding=EmbeddingProviderConfig(provider="fake", model="fake-embed", dimension=1),
    )
    artifacts = prepare_ocr_workflow_input(
        document_id="ocr-compat-doc",
        title="OCR Compatibility Doc",
        output_dir=scratch / "artifacts",
        image_payloads=[OCRImagePayload(page_number=1, image_path=str(image_path))],
        provider_settings=settings,
        ocr_runner=lambda image_path, page_number, provider_settings: _fake_ocr_response(page_number, "Compatibility page"),
    )

    regenerated = regen_doc(str(artifacts.legacy_dir), use_raw=True)

    assert artifacts.legacy_dir.exists()
    assert len(regenerated) == 1
    assert regenerated[0]["OCR_text_clusters"][0]["text"] == "Compatibility page"
    assert artifacts.workflow_input.collections[0].pages[0].units[0].text == "Compatibility page"
    persisted_page = json.loads((artifacts.legacy_dir / "page_1.json").read_text(encoding="utf-8"))
    assert persisted_page["metadata"]["ocr_provider"] == "fake"
    assert len(persisted_page["metadata"]["ocr_configuration_fingerprint"]) == 64
    assert artifacts.document_tree is not None
    assert artifacts.document_tree.source_items[0].ocr_metadata is not None


@pytest.mark.ci
def test_prepare_ocr_workflow_input_accepts_pdf_with_injected_rasterizer() -> None:
    scratch = _scratch("ocr_pdf")
    pdf_path = scratch / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
    rendered_image = scratch / "rendered" / "page_1.png"
    _draw_test_image(rendered_image, lines=["Injected raster page"])
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        embedding=EmbeddingProviderConfig(provider="fake", model="fake-embed", dimension=1),
    )

    def _rasterizer(pdf_path: Path, rendered_dir: Path) -> list[Path]:
        rendered_dir.mkdir(parents=True, exist_ok=True)
        target = rendered_dir / "page_1.png"
        shutil_path = rendered_image
        if shutil_path.resolve() != target.resolve():
            import shutil

            shutil.copy2(shutil_path, target)
        return [target]

    artifacts = prepare_ocr_workflow_input(
        document_id="ocr-pdf-doc",
        title="OCR PDF Doc",
        output_dir=scratch / "artifacts",
        pdf_path=pdf_path,
        provider_settings=settings,
        ocr_runner=lambda image_path, page_number, provider_settings: _fake_ocr_response(page_number, "Injected raster page"),
        pdf_rasterizer=_rasterizer,
    )

    assert len(artifacts.ocr_pages) == 1
    assert artifacts.workflow_input.collections[0].pages[0].units[0].text == "Injected raster page"


@pytest.mark.ci
def test_prepare_ocr_workflow_input_retries_failed_pages_and_tracks_attempts() -> None:
    scratch = _scratch("ocr_retry")
    images_dir = scratch / "images"
    image_paths = [images_dir / "page_1.png", images_dir / "page_2.png"]
    _draw_test_image(image_paths[0], lines=["Retry test page one"])
    _draw_test_image(image_paths[1], lines=["Retry test page two"])
    payloads = [
        OCRImagePayload(page_number=1, image_path=str(image_paths[0])),
        OCRImagePayload(page_number=2, image_path=str(image_paths[1])),
    ]
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        embedding=EmbeddingProviderConfig(provider="fake", model="fake-embed", dimension=1),
    )
    calls = {"page1": 0, "page2": 0}

    def _runner(image_path: Path, page_number: int, provider_settings: WorkflowProviderSettings):
        if page_number == 1:
            calls["page1"] += 1
            if calls["page1"] == 1:
                raise RuntimeError("temporary ocr failure")
        else:
            calls["page2"] += 1
        return _fake_ocr_response(page_number, f"Retry page {page_number}")

    with pytest.raises(RuntimeError, match="OCR preparation incomplete"):
        prepare_ocr_workflow_input(
            document_id="ocr-retry-doc",
            title="OCR Retry Doc",
            output_dir=scratch / "artifacts",
            image_payloads=payloads,
            provider_settings=settings,
            ocr_runner=_runner,
        )

    second = prepare_ocr_workflow_input(
        document_id="ocr-retry-doc",
        title="OCR Retry Doc",
        output_dir=scratch / "artifacts",
        image_payloads=payloads,
        provider_settings=settings,
        ocr_runner=_runner,
    )

    with sqlite3.connect(second.state_db_path) as conn:
        row = conn.execute(
            """
            SELECT attempt_count, status
            FROM page_state
            WHERE document_id = ? AND page_number = ? AND stage = 'ocr'
            """,
            ("ocr-retry-doc", 1),
        ).fetchone()
    assert row is not None
    assert int(row[0]) == 2
    assert row[1] == "completed"
    assert second.reused_pages == [2]


@pytest.mark.ci
def test_prepare_ocr_workflow_input_exhausts_candidate_models_until_success() -> None:
    scratch = _scratch("ocr_candidates")
    image_path = scratch / "images" / "page_1.png"
    _draw_test_image(image_path, lines=["Candidate page"])
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="primary-model"),
        embedding=EmbeddingProviderConfig(provider="fake", model="fake-embed", dimension=1),
    )

    def _runner(image_path: Path, page_number: int, provider_settings: WorkflowProviderSettings):
        if provider_settings.ocr.model == "primary-model":
            raise RuntimeError("primary failed")
        return _fake_ocr_response(page_number, f"winner:{provider_settings.ocr.model}")

    artifacts = prepare_ocr_workflow_input(
        document_id="ocr-candidates-doc",
        title="OCR Candidate Doc",
        output_dir=scratch / "artifacts",
        image_payloads=[OCRImagePayload(page_number=1, image_path=str(image_path))],
        provider_settings=settings,
        ocr_runner=_runner,
        ocr_candidate_models=["primary-model", "backup-model"],
    )

    with sqlite3.connect(artifacts.state_db_path) as conn:
        rows = conn.execute(
            """
            SELECT model_name, status
            FROM model_attempts
            WHERE document_id = ? AND page_number = ? AND stage = 'ocr'
            ORDER BY attempt_index
            """,
            ("ocr-candidates-doc", 1),
        ).fetchall()
    assert [tuple(row) for row in rows] == [
        ("primary-model", "failed"),
        ("backup-model", "completed"),
    ]
    assert artifacts.workflow_input.collections[0].pages[0].units[0].text == "winner:backup-model"


@pytest.mark.ci
def test_resolve_ocr_source_plan_uses_document_fingerprint_for_pdf_source() -> None:
    scratch = _scratch("ocr_fingerprint_plan")
    pdf_path = scratch / "sample.pdf"
    page_1 = scratch / "page_1.png"
    page_2 = scratch / "page_2.png"
    _draw_test_image(page_1, lines=["Fingerprint page one"])
    _draw_test_image(page_2, lines=["Fingerprint page two"])
    _write_two_page_pdf([page_1, page_2], pdf_path)

    output_dir = scratch / "artifacts"
    plan = _resolve_ocr_source_plan(
        document_id="ocr-fingerprint-doc",
        output_dir=output_dir,
        image_payloads=None,
        pdf_path=pdf_path,
        pdf_rasterizer=lambda source_pdf, rendered_dir: [page_1, page_2],
    )

    assert plan.source_kind == "pdf"
    assert plan.pdf_source == pdf_path
    assert plan.input_fingerprint == _compute_input_fingerprint(pdf_path=pdf_path)
    assert plan.input_fingerprint != _compute_input_fingerprint(page_sources=[(1, page_1)])


@pytest.mark.ci
def test_pdf_artifact_directories_do_not_collide_for_same_basename() -> None:
    scratch = _scratch("ocr_same_basename")
    first_pdf = scratch / "first" / "report.pdf"
    second_pdf = scratch / "second" / "report.pdf"
    first_pdf.parent.mkdir(parents=True)
    second_pdf.parent.mkdir(parents=True)
    first_pdf.write_bytes(b"%PDF-1.4\nfirst source\n")
    second_pdf.write_bytes(b"%PDF-1.4\nsecond source\n")

    def _rasterizer(source_pdf: Path, rendered_dir: Path) -> list[Path]:
        rendered_dir.mkdir(parents=True, exist_ok=True)
        page = rendered_dir / "page_1.png"
        color = "red" if b"first source" in source_pdf.read_bytes() else "blue"
        Image.new("RGB", (11, 7), color).save(page, format="PNG")
        return [page]

    output_dir = scratch / "artifacts"
    first = _resolve_ocr_source_plan(
        document_id="same-document",
        output_dir=output_dir,
        image_payloads=None,
        pdf_path=first_pdf,
        pdf_rasterizer=_rasterizer,
    )
    second = _resolve_ocr_source_plan(
        document_id="same-document",
        output_dir=output_dir,
        image_payloads=None,
        pdf_path=second_pdf,
        pdf_rasterizer=_rasterizer,
    )

    assert first.rendered_dir != second.rendered_dir
    assert first.legacy_dir != second.legacy_dir
    assert first.rendered_dir.name.startswith("same-document-report-")
    assert second.rendered_dir.name.startswith("same-document-report-")
    assert first.page_sources[0][1].read_bytes() != second.page_sources[0][1].read_bytes()


@pytest.mark.ci
def test_prepare_ocr_workflow_input_keeps_document_fingerprint_stable_in_state_db() -> None:
    scratch = _scratch("ocr_fingerprint_state")
    page_1 = scratch / "images" / "page_1.png"
    page_2 = scratch / "images" / "page_2.png"
    _draw_test_image(page_1, lines=["Fingerprint state page one"])
    _draw_test_image(page_2, lines=["Fingerprint state page two"])
    payloads = [
        OCRImagePayload(page_number=1, image_path=str(page_1)),
        OCRImagePayload(page_number=2, image_path=str(page_2)),
    ]
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        embedding=EmbeddingProviderConfig(provider="fake", model="fake-embed", dimension=1),
    )

    def _runner(image_path: Path, page_number: int, provider_settings: WorkflowProviderSettings):
        return _fake_ocr_response(page_number, f"Fingerprint page {page_number}")

    artifacts = prepare_ocr_workflow_input(
        document_id="ocr-fingerprint-doc",
        title="OCR Fingerprint Doc",
        output_dir=scratch / "artifacts",
        image_payloads=payloads,
        provider_settings=settings,
        ocr_runner=_runner,
    )

    expected_fingerprint = _compute_input_fingerprint(page_sources=[(1, page_1), (2, page_2)])
    with sqlite3.connect(artifacts.state_db_path) as conn:
        row = conn.execute(
            """
            SELECT input_fingerprint, total_pages, source_kind, is_completed
            FROM document_state
            WHERE document_id = ?
            """,
            ("ocr-fingerprint-doc",),
        ).fetchone()
    assert row is not None
    assert row[0] == expected_fingerprint
    assert int(row[1]) == 2
    assert row[2] == "image"
    assert int(row[3]) == 1

    second = prepare_ocr_workflow_input(
        document_id="ocr-fingerprint-doc",
        title="OCR Fingerprint Doc",
        output_dir=scratch / "artifacts",
        image_payloads=payloads,
        provider_settings=settings,
        ocr_runner=_runner,
    )
    assert second.reused_pages == [1, 2]
    with sqlite3.connect(second.state_db_path) as conn:
        row2 = conn.execute(
            """
            SELECT input_fingerprint
            FROM document_state
            WHERE document_id = ?
            """,
            ("ocr-fingerprint-doc",),
        ).fetchone()
    assert row2 is not None
    assert row2[0] == expected_fingerprint


@pytest.mark.ci
def test_ocr_state_fingerprint_change_discards_stale_page_and_model_state() -> None:
    scratch = _scratch("ocr_fingerprint_invalidation")
    artifact = scratch / "page-1.json"
    artifact.write_text("{}", encoding="utf-8")
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="fingerprint-invalidation-doc",
        title="Fingerprint Invalidation",
        source_kind="image",
        total_pages=1,
        input_fingerprint="fingerprint-a",
    )
    attempt = store.record_attempt(
        document_id="fingerprint-invalidation-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-a",
        model_name="model-a",
        artifact_path=artifact,
    )
    store.record_page_completed(
        document_id="fingerprint-invalidation-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-a",
        artifact_path=artifact,
        model_name="model-a",
        attempt_index=attempt,
    )
    assert store.get_page_state(
        document_id="fingerprint-invalidation-doc", page_number=1, stage="ocr"
    ) is not None
    assert store.list_model_attempts(
        document_id="fingerprint-invalidation-doc", page_number=1
    )

    store.ensure_document(
        document_id="fingerprint-invalidation-doc",
        title="Fingerprint Invalidation",
        source_kind="image",
        total_pages=1,
        input_fingerprint="fingerprint-b",
    )

    assert store.get_page_state(
        document_id="fingerprint-invalidation-doc", page_number=1, stage="ocr"
    ) is None
    assert store.list_model_attempts(
        document_id="fingerprint-invalidation-doc", page_number=1
    ) == []
    assert not store.read_document_completed(document_id="fingerprint-invalidation-doc")


@pytest.mark.ci
def test_ocr_page_completion_rejects_a_late_writer_after_acceptance() -> None:
    scratch = _scratch("ocr_page_completion_cas")
    artifact = scratch / "page-1.json"
    artifact.write_text('{"page": 1}', encoding="utf-8")
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="page-completion-cas-doc",
        title="Page Completion CAS",
        source_kind="image",
        total_pages=1,
        input_fingerprint="fingerprint",
    )
    attempt = store.record_attempt(
        document_id="page-completion-cas-doc",
        page_number=1,
        stage="render",
        content_hash="image-a",
        model_name="renderer-a",
        artifact_path=artifact,
        config_fingerprint="render-a",
    )
    assert store.record_page_completed(
        document_id="page-completion-cas-doc",
        page_number=1,
        stage="render",
        content_hash="image-a",
        artifact_path=artifact,
        model_name="renderer-a",
        attempt_index=attempt,
        config_fingerprint="render-a",
    )

    assert not store.record_page_completed(
        document_id="page-completion-cas-doc",
        page_number=1,
        stage="render",
        content_hash="image-b",
        artifact_path=artifact,
        model_name="renderer-b",
        config_fingerprint="render-b",
    )
    state = store.get_page_state(
        document_id="page-completion-cas-doc", page_number=1, stage="render"
    )
    assert state is not None
    assert state.content_hash == "image-a"
    assert state.config_fingerprint == "render-a"


@pytest.mark.ci
def test_ocr_page_failure_cannot_downgrade_an_accepted_result() -> None:
    scratch = _scratch("ocr_page_failure_cas")
    artifact = scratch / "page-1.json"
    artifact.write_text('{"page": 1}', encoding="utf-8")
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="page-failure-cas-doc",
        title="Page Failure CAS",
        source_kind="image",
        total_pages=1,
        input_fingerprint="fingerprint",
    )
    attempt = store.record_attempt(
        document_id="page-failure-cas-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-a",
        model_name="model-a",
        artifact_path=artifact,
        config_fingerprint="provider-a/model-a",
    )
    assert store.record_page_completed(
        document_id="page-failure-cas-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-a",
        artifact_path=artifact,
        model_name="model-a",
        attempt_index=attempt,
        config_fingerprint="provider-a/model-a",
    )

    assert not store.record_page_failed(
        document_id="page-failure-cas-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-a",
        config_fingerprint="provider-a/model-a",
        error_message="late provider failure",
        model_name="model-a",
        attempt_index=attempt,
    )
    state = store.get_page_state(
        document_id="page-failure-cas-doc", page_number=1, stage="ocr"
    )
    assert state is not None
    assert state.status == "completed"
    assert state.last_error is None


@pytest.mark.ci
def test_ocr_page_failure_cannot_replace_a_newer_pending_attempt() -> None:
    scratch = _scratch("ocr_page_failure_attempt_cas")
    artifact = scratch / "page-1.json"
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="page-failure-attempt-cas-doc",
        title="Page Failure Attempt CAS",
        source_kind="image",
        total_pages=1,
        input_fingerprint="fingerprint",
    )
    first = store.record_attempt(
        document_id="page-failure-attempt-cas-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-a",
        model_name="model-a",
        artifact_path=artifact,
        config_fingerprint="provider-a/model-a",
    )
    second = store.record_attempt(
        document_id="page-failure-attempt-cas-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-a",
        model_name="model-b",
        artifact_path=artifact,
        config_fingerprint="provider-b/model-b",
    )
    assert second == first + 1
    assert not store.record_page_failed(
        document_id="page-failure-attempt-cas-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-a",
        config_fingerprint="provider-a/model-a",
        error_message="late first-attempt failure",
        model_name="model-a",
        attempt_index=first,
    )
    state = store.get_page_state(
        document_id="page-failure-attempt-cas-doc", page_number=1, stage="ocr"
    )
    assert state is not None
    assert state.status == "pending"
    assert state.attempt_count == second
    assert state.config_fingerprint == "provider-b/model-b"


@pytest.mark.ci
def test_ocr_page_completion_cannot_replace_a_newer_pending_attempt() -> None:
    scratch = _scratch("ocr_page_completion_attempt_cas")
    artifact = scratch / "page-1.json"
    artifact.write_text('{"page": 1}', encoding="utf-8")
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="page-completion-attempt-cas-doc",
        title="Page Completion Attempt CAS",
        source_kind="image",
        total_pages=1,
        input_fingerprint="fingerprint",
    )
    first = store.record_attempt(
        document_id="page-completion-attempt-cas-doc",
        page_number=1,
        stage="render",
        content_hash="source-a",
        model_name="renderer",
        artifact_path=artifact,
        config_fingerprint="renderer-v1",
    )
    second = store.record_attempt(
        document_id="page-completion-attempt-cas-doc",
        page_number=1,
        stage="render",
        content_hash="source-a",
        model_name="renderer",
        artifact_path=artifact,
        config_fingerprint="renderer-v1",
    )
    assert second == first + 1
    assert not store.record_page_completed(
        document_id="page-completion-attempt-cas-doc",
        page_number=1,
        stage="render",
        content_hash="source-a",
        artifact_path=artifact,
        model_name="renderer",
        attempt_index=first,
        config_fingerprint="renderer-v1",
    )
    state = store.get_page_state(
        document_id="page-completion-attempt-cas-doc", page_number=1, stage="render"
    )
    assert state is not None
    assert state.status == "pending"
    assert state.attempt_count == second


@pytest.mark.ci
def test_ocr_state_does_not_skip_when_matching_artifact_was_removed() -> None:
    scratch = _scratch("ocr_missing_artifact")
    artifact = scratch / "page-1.json"
    artifact.write_text("{}", encoding="utf-8")
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="missing-artifact-doc",
        title="Missing Artifact",
        source_kind="image",
        total_pages=1,
        input_fingerprint="fingerprint",
    )
    attempt = store.record_attempt(
        document_id="missing-artifact-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-hash",
        model_name="model",
        artifact_path=artifact,
    )
    store.record_page_completed(
        document_id="missing-artifact-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-hash",
        artifact_path=artifact,
        model_name="model",
        attempt_index=attempt,
    )
    assert store.should_skip_page(
        document_id="missing-artifact-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-hash",
        artifact_path=artifact,
    )

    artifact.unlink()

    assert not store.should_skip_page(
        document_id="missing-artifact-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-hash",
        artifact_path=artifact,
    )


@pytest.mark.ci
def test_ocr_state_rejects_stale_provider_config_or_tampered_artifact() -> None:
    scratch = _scratch("ocr_config_identity")
    artifact = scratch / "page-1.json"
    artifact.write_text('{"page": 1}', encoding="utf-8")
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="config-identity-doc",
        title="Config Identity",
        source_kind="image",
        total_pages=1,
        input_fingerprint="fingerprint",
    )
    attempt = store.record_attempt(
        document_id="config-identity-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-hash",
        model_name="model-a",
        artifact_path=artifact,
        config_fingerprint="provider-a/model-a",
    )
    store.record_page_completed(
        document_id="config-identity-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-hash",
        artifact_path=artifact,
        model_name="model-a",
        attempt_index=attempt,
        config_fingerprint="provider-a/model-a",
    )

    assert store.should_skip_page(
        document_id="config-identity-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-hash",
        artifact_path=artifact,
        config_fingerprint=("provider-a/model-a", "provider-a/model-b"),
    )
    assert not store.should_skip_page(
        document_id="config-identity-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-hash",
        artifact_path=artifact,
        config_fingerprint="provider-b/model-a",
    )

    child_script = f"""
from pathlib import Path
from kg_doc_parser.workflow_ingest.ocr_pipeline import OCRWorkflowStateStore
store = OCRWorkflowStateStore(Path({str(scratch / 'ocr-state.sqlite')!r}))
assert not store.should_skip_page(
    document_id='config-identity-doc', page_number=1, stage='ocr',
    content_hash='image-hash', artifact_path=Path({str(artifact)!r}),
    config_fingerprint='provider-b/model-a'
)
assert store.should_skip_page(
    document_id='config-identity-doc', page_number=1, stage='ocr',
    content_hash='image-hash', artifact_path=Path({str(artifact)!r}),
    config_fingerprint='provider-a/model-a'
)
"""
    child = subprocess.run(
        [sys.executable, "-c", child_script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0

    artifact.write_text('{"page": 1, "tampered": true}', encoding="utf-8")
    assert not store.should_skip_page(
        document_id="config-identity-doc",
        page_number=1,
        stage="ocr",
        content_hash="image-hash",
        artifact_path=artifact,
        config_fingerprint="provider-a/model-a",
    )


@pytest.mark.ci
def test_ocr_render_reuse_requires_render_configuration_identity() -> None:
    scratch = _scratch("ocr_render_config_identity")
    artifact = scratch / "page-1.png"
    artifact.write_bytes(b"rendered-page")
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    store.ensure_document(
        document_id="render-config-doc",
        title="Render Config",
        source_kind="pdf",
        total_pages=1,
        input_fingerprint="pdf-fingerprint",
    )
    attempt = store.record_attempt(
        document_id="render-config-doc",
        page_number=1,
        stage="render",
        content_hash="source-image",
        model_name="rasterizer-a",
        artifact_path=artifact,
        config_fingerprint="renderer-a",
    )
    store.record_page_completed(
        document_id="render-config-doc",
        page_number=1,
        stage="render",
        content_hash="source-image",
        artifact_path=artifact,
        model_name="rasterizer-a",
        attempt_index=attempt,
        config_fingerprint="renderer-a",
    )

    assert store.should_skip_page(
        document_id="render-config-doc",
        page_number=1,
        stage="render",
        content_hash="source-image",
        artifact_path=artifact,
        config_fingerprint="renderer-a",
    )
    assert not store.should_skip_page(
        document_id="render-config-doc",
        page_number=1,
        stage="render",
        content_hash="source-image",
        artifact_path=artifact,
        config_fingerprint="renderer-b",
    )


@pytest.mark.ci
def test_pdf_render_manifest_rejects_tampered_page_and_records_identity(tmp_path: Path) -> None:
    from kg_doc_parser.workflow_ingest.ocr_pipeline import _render_pdf_with_identity

    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.4\nsource\n")
    rendered = tmp_path / "rendered"
    calls: list[int] = []

    class _Rasterizer:
        configuration_fingerprint = "dpi=144;rotation=0"

        def __call__(self, pdf_path: Path, rendered_dir: Path) -> list[Path]:
            del pdf_path
            calls.append(1)
            rendered_dir.mkdir(parents=True, exist_ok=True)
            path = rendered_dir / "page_1.png"
            Image.new("RGB", (11, 7), "white").save(path, "PNG")
            return [path]

    rasterizer = _Rasterizer()
    first = _render_pdf_with_identity(pdf_path=source, rendered_dir=rendered, rasterizer=rasterizer)
    manifest = json.loads((rendered / "render-manifest.json").read_text(encoding="utf-8"))
    assert manifest["rasterizer_configuration"] == "dpi=144;rotation=0"
    assert manifest["pages"][0]["dimensions"] == [11, 7]
    second = _render_pdf_with_identity(pdf_path=source, rendered_dir=rendered, rasterizer=rasterizer)
    assert second == first
    assert calls == [1]

    first[0].write_bytes(b"truncated")
    _render_pdf_with_identity(pdf_path=source, rendered_dir=rendered, rasterizer=rasterizer)
    assert calls == [1, 1]


@pytest.mark.ci
def test_pdf_render_manifest_rejects_a_missing_page_from_a_valid_pdf(tmp_path: Path) -> None:
    from pypdf import PdfWriter

    from kg_doc_parser.workflow_ingest.ocr_pipeline import _render_pdf_with_identity

    source = tmp_path / "two-pages.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_blank_page(width=72, height=72)
    with source.open("wb") as handle:
        writer.write(handle)

    rendered = tmp_path / "rendered"
    calls: list[int] = []

    class _Rasterizer:
        def __call__(self, pdf_path: Path, rendered_dir: Path) -> list[Path]:
            del pdf_path
            calls.append(1)
            rendered_dir.mkdir(parents=True, exist_ok=True)
            paths = [rendered_dir / "page_1.png", rendered_dir / "page_2.png"]
            for path in paths:
                Image.new("RGB", (11, 7), "white").save(path, "PNG")
            return paths

    rasterizer = _Rasterizer()
    _render_pdf_with_identity(pdf_path=source, rendered_dir=rendered, rasterizer=rasterizer)
    manifest_path = rendered / "render-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pages"] = manifest["pages"][:1]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _render_pdf_with_identity(pdf_path=source, rendered_dir=rendered, rasterizer=rasterizer)
    assert calls == [1, 1]


@pytest.mark.ci
def test_pdf_render_manifest_rejects_same_path_replacement(tmp_path: Path) -> None:
    from pypdf import PdfWriter

    from kg_doc_parser.workflow_ingest.ocr_pipeline import _render_pdf_with_identity

    source = tmp_path / "same-path.pdf"
    rendered = tmp_path / "rendered"
    calls: list[bytes] = []

    def _write_pdf(title: str) -> None:
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.add_blank_page(width=72, height=72)
        writer.add_metadata({"/Title": title})
        with source.open("wb") as handle:
            writer.write(handle)

    class _Rasterizer:
        def __call__(self, pdf_path: Path, rendered_dir: Path) -> list[Path]:
            source_bytes = pdf_path.read_bytes()
            calls.append(source_bytes)
            rendered_dir.mkdir(parents=True, exist_ok=True)
            paths = [rendered_dir / "page_1.png", rendered_dir / "page_2.png"]
            color = "white" if len(calls) == 1 else "black"
            for path in paths:
                Image.new("RGB", (11, 7), color).save(path, "PNG")
            return paths

    _write_pdf("revision-a")
    rasterizer = _Rasterizer()
    first = _render_pdf_with_identity(
        pdf_path=source, rendered_dir=rendered, rasterizer=rasterizer
    )
    _write_pdf("revision-b")
    second = _render_pdf_with_identity(
        pdf_path=source, rendered_dir=rendered, rasterizer=rasterizer
    )

    assert first == second
    assert len(calls) == 2
    with Image.open(second[1]) as image:
        assert image.getpixel((0, 0)) == (0, 0, 0)


@pytest.mark.ci
def test_builtin_pdf_render_identity_includes_poppler_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"placeholder")
    first = _pdf_render_manifest_payload(
        pdf_path=source,
        rasterizer=_render_pdf_to_images,
    )

    monkeypatch.setenv("POPPLER_PATH", str(tmp_path / "other-poppler"))
    second = _pdf_render_manifest_payload(
        pdf_path=source,
        rasterizer=_render_pdf_to_images,
    )

    assert first["rasterizer_configuration"] != second["rasterizer_configuration"]
    assert first["render_identity"] != second["render_identity"]


@pytest.mark.ci
def test_png_publication_failure_keeps_previous_artifact_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "page-1.png"
    target.write_bytes(b"previous")
    image = Image.new("RGB", (4, 4), "white")

    def _fail_save(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("injected rasterizer failure")

    monkeypatch.setattr(image, "save", _fail_save)
    with pytest.raises(OSError, match="injected rasterizer failure"):
        _write_png_atomic(image, target)

    assert target.read_bytes() == b"previous"
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


@pytest.mark.ci
def test_atomic_json_publication_uses_distinct_temporary_files_for_concurrent_writers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = _scratch("ocr_atomic_publication_race")
    target = scratch / "page.json"
    sources: list[str] = []
    replace_lock = threading.Lock()
    original_replace = os.replace

    def _replace(source: str, destination: str, /) -> None:
        with replace_lock:
            sources.append(source)
            original_replace(source, destination)

    monkeypatch.setattr(ocr_pipeline_module.os, "replace", _replace)

    def _publish(value: str) -> None:
        _write_json_atomic(target, {"value": value, "padding": "x" * 100_000})

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_publish, value) for value in ("a", "b")]
        for future in futures:
            future.result()

    assert len(sources) == 2
    assert len(set(sources)) == 2
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["value"] in {"a", "b"}
    assert not list(target.parent.glob(f".{target.name}.*.tmp"))


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
    store.record_page_completed(
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
        embedding=EmbeddingProviderConfig(provider="fake", model="fake-embed", dimension=1),
    )
    store_path = scratch / "ocr-state.sqlite"
    store = OCRWorkflowStateStore(store_path)
    config_fingerprint = ocr_pipeline_module._ocr_configuration_fingerprint(
        settings,
        candidate_models=["fake-ocr"],
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
    )
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    original_write = ocr_pipeline_module._write_json_atomic
    interrupted = True

    def _publish_then_interrupt(path: Path, payload: object) -> None:
        nonlocal interrupted
        original_write(path, payload)
        if interrupted and path == page_json_path:
            interrupted = False
            raise KeyboardInterrupt("simulated process termination after artifact publish")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ocr_pipeline_module, "_write_json_atomic", _publish_then_interrupt)
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
                raw_pages=[],
                completed_pages=[],
                reused_pages=[],
                page_failures=[],
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
        embedding=EmbeddingProviderConfig(provider="fake", model="fake-embed", dimension=1),
    )
    store_path = scratch / "ocr-state.sqlite"
    config_fingerprint = ocr_pipeline_module._ocr_configuration_fingerprint(
        settings,
        candidate_models=["fake-ocr"],
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
            )
        )
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()

    def _process(context: OCRPageProcessingContext) -> tuple[list[OCRPageJSON], list[int], list[int]]:
        raw_pages: list[OCRPageJSON] = []
        completed_pages: list[int] = []
        reused_pages: list[int] = []
        _process_ocr_page(
            context=context,
            done_count=0,
            page_number=1,
            image_path=image_path,
            image_sha256=image_sha256,
            page_json_path=page_json_path,
            page_image_path=page_image_path,
            raw_pages=raw_pages,
            completed_pages=completed_pages,
            reused_pages=reused_pages,
            page_failures=[],
        )
        return raw_pages, completed_pages, reused_pages

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(_process, contexts))

    assert all(result[1] == [1] for result in results)
    assert sum(bool(result[2]) for result in results) == 1
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
def test_workflow_first_ocr_manual_matrix(provider: str, model: str, input_kind: str) -> None:
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

    workflow_engine, conversation_engine, knowledge_engine = build_workflow_engine_triplet(
        case_dir / "engines",
        "in_memory",
    )
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(
            provider=provider,
            model=model,
            base_url=os.getenv("KG_DOC_OCR_BASE_URL", "http://127.0.0.1:11434") if provider == "ollama" else None,
            api_key_env="GOOGLE_API_KEY" if provider == "gemini" else None,
        ),
        embedding=EmbeddingProviderConfig(provider="fake", model="ocr-manual-embed", dimension=1),
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
        drain_phase1_indexes_until_idle(workflow_engine, conversation_engine, knowledge_engine)
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
    _LOGGER.info("  completed_pages=%s reused_pages=%s", artifacts.completed_pages, artifacts.reused_pages)


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

    def _runner(image_path: Path, page_number: int, provider_settings: WorkflowProviderSettings):
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
        embedding=EmbeddingProviderConfig(provider="fake", model="fake-embed", dimension=1),
    )
    workflow_engine, conversation_engine, knowledge_engine = build_workflow_engine_triplet(
        scratch / "engines",
        "in_memory",
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
    assert calls["ocr"] == 2
    assert calls["parse"] == 1
    assert _first_artifacts.document_tree is not None
    assert captured_grouping_trees == [_first_artifacts.document_tree.model_dump(mode="json")]
    assert second_artifacts.reused_pages == [1, 2]
    assert second_artifacts.state_db_path.exists()
    assert SemanticNode.model_validate(second_run.final_state["semantic_tree"]).child_nodes
    assert second_run.final_state["semantic_tree"]["child_nodes"][0]["total_content_pointers"]
    assert second_run.final_state["semantic_tree"]["child_nodes"][0]["title"] == "grouped:root"
    assert first_bundle.grouping_tree == _first_artifacts.document_tree.model_dump(mode="json")
    assert second_bundle.grouping_tree == second_artifacts.document_tree.model_dump(mode="json")
    # The semantic child must publish the exact accepted grouping snapshot,
    # rather than merely parsing the same source pages independently.
    assert first_bundle.grouping_tree == first_run.final_state["export_bundle"]["grouping_tree"]
    assert second_bundle.grouping_tree == second_run.final_state["export_bundle"]["grouping_tree"]
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
            where={"$and": [{"entity_type": "workflow_step_exec"}, {"run_id": first_run.run_id}]}
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
