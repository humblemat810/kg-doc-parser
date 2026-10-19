"""OCR page preparation, resume, and artifact-led state recovery tests."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from ocr_workflow_support import (
    draw_test_image as _draw_test_image,
)
from ocr_workflow_support import (
    fake_ocr_response as _fake_ocr_response,
)
from ocr_workflow_support import (
    scratch_path as _scratch,
)

import kg_doc_parser.workflow_ingest.ocr_pipeline as ocr_pipeline_module
from kg_doc_parser.models import OCRClusterResponse
from kg_doc_parser.workflow_ingest import (
    EmbeddingProviderConfig,
    OCRImagePayload,
    ProviderEndpointConfig,
    WorkflowIngestInput,
    WorkflowProviderSettings,
    prepare_ocr_workflow_input,
    run_ocr_ingest_workflow,
)
from kg_doc_parser.workflow_ingest.ocr_pipeline import OCRQualityPolicy
from kg_doc_parser.workflow_ingest.ocr_state_store import OCRWorkflowStateStore

pytestmark = [pytest.mark.workflow]

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

    def _runner(
        image_path: Path, page_number: int, provider_settings: WorkflowProviderSettings
    ):
        calls["count"] += 1
        return _fake_ocr_response(page_number, f"Resume page {page_number}")

    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        embedding=EmbeddingProviderConfig(
            provider="fake", model="fake-embed", dimension=1
        ),
    )
    first = prepare_ocr_workflow_input(
        document_id="ocr-resume-doc",
        title="OCR Resume Doc",
        output_dir=scratch / "artifacts",
        image_payloads=payloads,
        provider_settings=settings,
        ocr_runner=_runner,
        quality_policy=OCRQualityPolicy(mode="permissive"),
    )
    second = prepare_ocr_workflow_input(
        document_id="ocr-resume-doc",
        title="OCR Resume Doc",
        output_dir=scratch / "artifacts",
        image_payloads=payloads,
        provider_settings=settings,
        ocr_runner=_runner,
        quality_policy=OCRQualityPolicy(mode="permissive"),
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
        embedding=EmbeddingProviderConfig(
            provider="fake", model="fake-embed", dimension=1
        ),
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
        quality_policy=OCRQualityPolicy(mode="permissive"),
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
        quality_policy=OCRQualityPolicy(mode="permissive"),
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
            quality_policy=OCRQualityPolicy(mode="permissive"),
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
        embedding=EmbeddingProviderConfig(
            provider="fake", model="fake-embed", dimension=1
        ),
    )
    with pytest.raises(RuntimeError, match=r"pages \[4\]"):
        prepare_ocr_workflow_input(
            document_id="ocr-mid-document-resume",
            title="OCR Mid Document Resume",
            output_dir=scratch / "artifacts",
            image_payloads=payloads,
            provider_settings=settings,
            ocr_runner=_runner,
            quality_policy=OCRQualityPolicy(mode="permissive"),
        )

    resumed = prepare_ocr_workflow_input(
        document_id="ocr-mid-document-resume",
        title="OCR Mid Document Resume",
        output_dir=scratch / "artifacts",
        image_payloads=payloads,
        provider_settings=settings,
        ocr_runner=_runner,
        quality_policy=OCRQualityPolicy(mode="permissive"),
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

    monkeypatch.setattr(
        ocr_pipeline_module, "_build_ocr_page_preparation", _fake_build_preparation
    )
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

    def _runner(
        image_path: Path, page_number: int, provider_settings: WorkflowProviderSettings
    ):
        page_text = (
            "this sentence continues"
            if page_number == 1
            else "on the next page"
            if page_number == 2
            else "independent final page"
        )
        response = _fake_ocr_response(page_number, page_text)
        response = response.model_copy(
            update={
                "OCR_text_clusters": [
                    cluster.model_copy(
                        update={
                            "continuation_id": "paragraph-1",
                            "bb_y_min": 740 if page_number == 1 else 10,
                            "bb_y_max": 790 if page_number == 1 else 60,
                        }
                    )
                    for cluster in response.OCR_text_clusters
                ]
            }
        )
        if page_number == 1:
            response = response.model_copy(update={"incomplete_text": True})
        return response

    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        embedding=EmbeddingProviderConfig(
            provider="fake", model="fake-embed", dimension=1
        ),
    )
    artifacts = prepare_ocr_workflow_input(
        document_id="ocr-live-cross-page-doc",
        title="OCR Live Cross Page Doc",
        output_dir=scratch / "artifacts",
        image_payloads=payloads,
        provider_settings=settings,
        ocr_runner=_runner,
        quality_policy=OCRQualityPolicy(mode="permissive"),
    )

    assert artifacts.ocr_pages[0].get("incomplete_text") is True
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
    assert any(
        node.kind == "layout:cross-page-continuation" for node in persisted.nodes
    )


@pytest.mark.ci
def test_prepare_ocr_workflow_input_rebuilds_state_db_from_existing_artifacts() -> None:
    scratch = _scratch("ocr_rebuild")
    image_path = scratch / "images" / "page_1.png"
    _draw_test_image(image_path, lines=["Rebuild page"])
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        embedding=EmbeddingProviderConfig(
            provider="fake", model="fake-embed", dimension=1
        ),
    )
    calls = {"count": 0}

    def _runner(
        image_path: Path, page_number: int, provider_settings: WorkflowProviderSettings
    ):
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
def test_grouping_checkpoint_migrates_legacy_schema_with_generation_column() -> None:
    db_path = _scratch("legacy_ocr_state_schema") / "legacy-ocr-state.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE grouping_checkpoint_metadata (
                document_id TEXT PRIMARY KEY,
                source_fingerprint TEXT NOT NULL,
                grouping_policy_version TEXT NOT NULL,
                accepted_page_frontier_json TEXT NOT NULL,
                tree_revision TEXT NOT NULL,
                last_runtime_step TEXT NOT NULL,
                last_updated_ts REAL NOT NULL
            )
            """
        )

    OCRWorkflowStateStore(db_path)
    with sqlite3.connect(db_path) as connection:
        columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(grouping_checkpoint_metadata)"
            ).fetchall()
        }
    assert "merge_revision" in columns


@pytest.mark.ci
def test_ocr_attempts_migrate_legacy_schema_with_retry_feedback_column() -> None:
    db_path = _scratch("legacy_ocr_attempt_schema") / "legacy-ocr-attempts.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE model_attempts (
                document_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                stage TEXT NOT NULL,
                attempt_index INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                status TEXT NOT NULL,
                error_message TEXT,
                attempted_ts REAL NOT NULL,
                PRIMARY KEY (document_id, page_number, stage, attempt_index)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO model_attempts (
                document_id, page_number, stage, attempt_index, model_name,
                status, error_message, attempted_ts
            ) VALUES ('legacy-doc', 1, 'ocr', 1, 'legacy-model', 'failed',
                     'legacy error', 1.0)
            """
        )

    store = OCRWorkflowStateStore(db_path)
    attempt = store.list_model_attempts(
        document_id="legacy-doc", page_number=1, stage="ocr"
    )[0]
    with sqlite3.connect(db_path) as connection:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(model_attempts)")
        }

    assert "retry_feedback" in columns
    assert attempt["error_message"] == "legacy error"
    assert attempt["retry_feedback"] is None
