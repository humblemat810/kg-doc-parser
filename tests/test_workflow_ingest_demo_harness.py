from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import pytest

from _kogwistar_test_helpers import load_kogwistar_fake_backend
from src.workflow_ingest import DemoHarnessConfig, WorkflowLLMCallCache, run_demo_harness
from src.workflow_ingest.demo_harness import _start_subprocess_server


def _scratch(name: str) -> Path:
    root = Path("tests") / ".tmp_workflow_ingest_demo"
    path = root / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _probe_events(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_workflow_llm_call_cache_hits_on_repeated_fingerprint():
    scratch = _scratch("llm_cache")
    calls = {"count": 0}
    cache = WorkflowLLMCallCache(scratch / "cache")

    def _fn():
        calls["count"] += 1
        return {"answer": 42}

    first = cache.cached_call(
        operation="review",
        fingerprint={"a": 1, "b": ["x"]},
        fn=_fn,
    )
    second = cache.cached_call(
        operation="review",
        fingerprint={"a": 1, "b": ["x"]},
        fn=_fn,
    )

    assert first == {"answer": 42}
    assert second == {"answer": 42}
    assert calls["count"] == 1


@pytest.mark.workflow
@pytest.mark.integration
def test_demo_harness_writes_probe_summary_and_cache():
    pytest.importorskip("chromadb")
    pytest.importorskip("fastapi")
    pytest.importorskip("fastmcp")

    scratch = _scratch("demo_harness")
    artifacts = run_demo_harness(
        DemoHarnessConfig(
            output_dir=scratch,
            document_id="demo-harness-doc",
            text="Alpha clause\nBeta clause\nGamma clause",
            parser_mode="fake_layered",
            server_mode="testclient",
            enable_sys_monitoring=False,
            backend_factory=load_kogwistar_fake_backend(),
        )
    )

    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    events = _probe_events(artifacts.probe_path)
    kinds = [event["kind"] for event in events]

    assert artifacts.status == "succeeded"
    assert artifacts.canonical_write_confirmed is True
    assert summary["canonical_write_confirmed"] is True
    assert summary["status"] == "succeeded"
    assert artifacts.cache_dir.exists()
    assert "demo.started" in kinds
    assert "demo.server_started" in kinds
    assert "workflow.run_started" in kinds
    assert "workflow.step_started" in kinds
    assert "workflow.step_finished" in kinds
    assert "workflow.persistence_result" in kinds
    assert "workflow.llm_cache_miss" in kinds
    assert "demo.finished" in kinds
    assert "demo.server_stopped" in kinds
    assert "probe.closed" in kinds


@pytest.mark.workflow
@pytest.mark.integration
def test_demo_harness_second_run_hits_llm_cache():
    pytest.importorskip("chromadb")
    pytest.importorskip("fastapi")
    pytest.importorskip("fastmcp")

    scratch = _scratch("demo_harness_cache")
    config = DemoHarnessConfig(
        output_dir=scratch,
        document_id="demo-harness-cache-doc",
        text="Alpha clause\nBeta clause",
        parser_mode="fake_layered",
        server_mode="testclient",
        enable_sys_monitoring=False,
        backend_factory=load_kogwistar_fake_backend(),
    )

    first = run_demo_harness(config)
    second = run_demo_harness(config)

    events = _probe_events(second.probe_path)
    kinds = [event["kind"] for event in events]

    assert first.status == "succeeded"
    assert second.status == "succeeded"
    assert "workflow.llm_cache_hit" in kinds


@pytest.mark.workflow
@pytest.mark.ci_full
def test_demo_harness_subprocess_server_mode_ci_full():
    if str(os.getenv("KG_DOC_ENABLE_SERVER_E2E_CI_FULL") or "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        pytest.skip("set KG_DOC_ENABLE_SERVER_E2E_CI_FULL=1 to run subprocess demo harness")
    pytest.importorskip("chromadb")
    pytest.importorskip("fastapi")
    pytest.importorskip("fastmcp")
    pytest.importorskip("requests")
    pytest.importorskip("uvicorn")

    scratch = _scratch("demo_harness_subprocess")
    artifacts = run_demo_harness(
        DemoHarnessConfig(
            output_dir=scratch,
            document_id="demo-harness-subprocess-doc",
            text="Alpha clause\nBeta clause",
            parser_mode="fake_layered",
            server_mode="subprocess_http",
            enable_sys_monitoring=False,
            backend_factory=load_kogwistar_fake_backend(),
        )
    )

    assert artifacts.status == "succeeded"
    assert artifacts.canonical_write_confirmed is True


@pytest.mark.workflow
@pytest.mark.ci_full
def test_demo_harness_external_http_mode_ci_full():
    if str(os.getenv("KG_DOC_ENABLE_SERVER_E2E_CI_FULL") or "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        pytest.skip("set KG_DOC_ENABLE_SERVER_E2E_CI_FULL=1 to run external-http demo harness")
    pytest.importorskip("chromadb")
    pytest.importorskip("fastapi")
    pytest.importorskip("fastmcp")
    pytest.importorskip("requests")
    pytest.importorskip("uvicorn")

    scratch = _scratch("demo_harness_external")
    server_data_dir = scratch / "server-data"
    with _start_subprocess_server(server_data_dir) as server_ctx:
        artifacts = run_demo_harness(
            DemoHarnessConfig(
                output_dir=scratch / "run",
                document_id="demo-harness-external-doc",
                text="Alpha clause\nBeta clause",
                parser_mode="fake_layered",
                server_mode="external_http",
                external_base_url=server_ctx.base_url,
                enable_sys_monitoring=False,
                backend_factory=load_kogwistar_fake_backend(),
            )
        )

    assert artifacts.status == "succeeded"
    assert artifacts.canonical_write_confirmed is True
