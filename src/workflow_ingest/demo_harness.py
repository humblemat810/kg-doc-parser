from __future__ import annotations

import importlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .cache import WorkflowLLMCallCache
from .clients import DocumentTreeApiPersistenceClient, ServerCanonicalKgClient
from .models import CurrentLayerResult, CurrentLayerReview, LayerChildCandidate, WorkflowIngestInput
from .providers import WorkflowProviderSettings
from .probe import WorkflowProbe, emit_probe_event
from .semantics import HydratedTextPointer
from .service import build_default_engines


@dataclass
class DemoHarnessConfig:
    output_dir: Path
    document_id: str = "demo-doc"
    title: str = "Workflow Ingest Demo"
    text: str = "Alpha clause\nBeta clause\nGamma clause"
    workflow_id: str = "kg_doc_parser.ingest.v1"
    parser_mode: Literal["fake_layered", "legacy_cached"] = "fake_layered"
    server_mode: Literal["testclient", "subprocess_http", "external_http"] = "testclient"
    external_base_url: str | None = None
    backend_factory: Any | None = None
    provider_settings: WorkflowProviderSettings | None = None
    enable_sys_monitoring: bool = True
    probe_filename: str = "probe-events.jsonl"
    summary_filename: str = "demo-summary.json"
    cache_dirname: str = "llm-cache"
    deps: dict[str, Any] = field(default_factory=dict)


@dataclass
class DemoHarnessArtifacts:
    output_dir: Path
    probe_path: Path
    summary_path: Path
    cache_dir: Path
    engine_dir: Path
    server_data_dir: Path
    run_id: str | None = None
    status: str | None = None
    canonical_write_confirmed: bool = False
    persistence_mode: str | None = None
    kg_authority: str | None = None


class _ServerContext(AbstractContextManager):
    def __init__(self, *, client: Any, transport: str, base_url: str = "", cleanup=None) -> None:
        self.client = client
        self.transport = transport
        self.base_url = base_url
        self._cleanup = cleanup

    def __exit__(self, exc_type, exc, tb):
        if self._cleanup is not None:
            self._cleanup(exc_type, exc, tb)
        return False


def _load_isolated_server_app(server_data_dir: Path) -> _ServerContext:
    from fastapi.testclient import TestClient

    os.environ["GKE_BACKEND"] = "chroma"
    os.environ["GKE_PERSIST_DIRECTORY"] = str(server_data_dir)
    os.environ["AUTH_MODE"] = "dev"
    os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"
    os.environ["KOGWISTAR_LOG_LEVEL"] = "WARNING"
    for module_name in (
        "kogwistar.server.resources",
        "kogwistar.server.bootstrap",
        "kogwistar.server_mcp_with_admin",
    ):
        sys.modules.pop(module_name, None)
    server_module = importlib.import_module("kogwistar.server_mcp_with_admin")
    client = TestClient(server_module.app)
    client.__enter__()
    return _ServerContext(
        client=client,
        transport="fastapi_testclient",
        base_url="",
        cleanup=client.__exit__,
    )


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _start_subprocess_server(server_data_dir: Path) -> _ServerContext:
    import requests

    host = "127.0.0.1"
    port = _pick_free_port()
    env = os.environ.copy()
    env["GKE_BACKEND"] = "chroma"
    env["GKE_PERSIST_DIRECTORY"] = str(server_data_dir)
    env["AUTH_MODE"] = "dev"
    env["ANONYMIZED_TELEMETRY"] = "FALSE"
    env["KOGWISTAR_LOG_LEVEL"] = "WARNING"
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "kogwistar.server_mcp_with_admin:app",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(Path(__file__).resolve().parents[2]),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines: list[str] = []

    def _reader() -> None:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            lines.append(line.rstrip())

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()

    base_url = f"http://{host}:{port}"
    deadline = time.time() + 60.0
    last_error: Exception | None = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                "demo server exited before healthy:\n" + "\n".join(lines[-50:])
            )
        try:
            response = requests.get(f"{base_url}/health", timeout=1.5)
            if response.ok:
                session = requests.Session()
                return _ServerContext(
                    client=session,
                    transport="subprocess_http",
                    base_url=base_url,
                    cleanup=lambda *_args: _shutdown_subprocess_server(proc, session),
                )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(0.2)
    _shutdown_subprocess_server(proc, None)
    raise RuntimeError(f"demo server did not become healthy: {last_error}")


def _connect_external_server(base_url: str) -> _ServerContext:
    import requests

    base_url = str(base_url).rstrip("/")
    response = requests.get(f"{base_url}/health", timeout=5.0)
    response.raise_for_status()
    session = requests.Session()
    return _ServerContext(
        client=session,
        transport="external_http",
        base_url=base_url,
        cleanup=lambda *_args: session.close(),
    )


def _shutdown_subprocess_server(proc: subprocess.Popen[str], session: Any | None) -> None:
    if session is not None:
        session.close()
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            proc.kill()
            proc.wait(timeout=5)


def _fake_layered_deps(inp: WorkflowIngestInput) -> dict[str, Any]:
    text = inp.collections[0].pages[0].units[0].text or ""
    unit_id = f"{inp.request_id}|p1_t0"
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    def _pointer(fragment: str) -> HydratedTextPointer:
        start = text.index(fragment)
        end = start + len(fragment) - 1
        return HydratedTextPointer(
            source_cluster_id=unit_id,
            start_char=start,
            end_char=end,
            verbatim_text=fragment,
        )

    def _propose_layer_fn(*, current_layer_context, **kwargs):
        if current_layer_context.depth == 0:
            return CurrentLayerResult(
                children=[
                    LayerChildCandidate(
                        node_id=f"{inp.request_id}|section|overview",
                        parent_node_id=current_layer_context.parent_node_ids[0],
                        title="Overview",
                        node_type="TEXT_FLOW",
                        total_content_pointers=[_pointer(text)],
                        expandable=len(lines) > 1,
                    )
                ],
                satisfied=True,
                reasoning_history=[{"stage": "proposal", "depth": 0, "lines": len(lines)}],
            )
        return CurrentLayerResult(
            children=[
                LayerChildCandidate(
                    node_id=f"{inp.request_id}|clause|{idx}",
                    parent_node_id=current_layer_context.parent_node_ids[0],
                    title=line,
                    node_type="TEXT_FLOW",
                    total_content_pointers=[_pointer(line)],
                    expandable=False,
                )
                for idx, line in enumerate(lines)
            ],
            satisfied=True,
            reasoning_history=[{"stage": "proposal", "depth": current_layer_context.depth}],
        )

    def _review_layer_fn(*, current_layer_result, **kwargs):
        return CurrentLayerReview(
            updated_result=current_layer_result.model_copy(update={"satisfied": True}),
            coverage_ok=True,
            satisfied=True,
            review_notes=["fake_review_ok"],
        )

    return {
        "propose_layer_fn": _propose_layer_fn,
        "review_layer_fn": _review_layer_fn,
    }


def _configure_legacy_cache(cache_dir: Path, probe: WorkflowProbe | None) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KG_DOC_PARSER_JOBLIB_CACHE_DIR"] = str(cache_dir)
    module_name = "src.semantic_document_splitting_layerwise_edits"
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
        emit_probe_event(probe, "demo.cache_reloaded", module=module_name, cache_dir=str(cache_dir))
    else:
        emit_probe_event(probe, "demo.cache_configured", module=module_name, cache_dir=str(cache_dir))


def run_demo_harness(config: DemoHarnessConfig) -> DemoHarnessArtifacts:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = WorkflowProbe(output_dir / config.probe_filename)
    if config.enable_sys_monitoring:
        probe.enable_sys_monitoring()

    artifacts = DemoHarnessArtifacts(
        output_dir=output_dir,
        probe_path=output_dir / config.probe_filename,
        summary_path=output_dir / config.summary_filename,
        cache_dir=output_dir / config.cache_dirname,
        engine_dir=output_dir / "engines",
        server_data_dir=output_dir / "server-data",
    )
    inp = WorkflowIngestInput.from_text(
        document_id=config.document_id,
        text=config.text,
        title=config.title,
    )

    deps = dict(config.deps)
    deps["llm_cache"] = WorkflowLLMCallCache(artifacts.cache_dir, probe=probe)
    if config.parser_mode == "fake_layered":
        deps.update(_fake_layered_deps(inp))
    else:
        _configure_legacy_cache(artifacts.cache_dir, probe)

    deps["probe"] = probe

    emit_probe_event(
        probe,
        "demo.started",
        output_dir=str(output_dir),
        parser_mode=config.parser_mode,
        server_mode=config.server_mode,
        document_id=config.document_id,
        external_base_url=config.external_base_url,
    )

    if config.server_mode == "testclient":
        server_ctx = _load_isolated_server_app(artifacts.server_data_dir)
    elif config.server_mode == "subprocess_http":
        server_ctx = _start_subprocess_server(artifacts.server_data_dir)
    else:
        if not config.external_base_url:
            raise ValueError("external_base_url is required when server_mode='external_http'")
        server_ctx = _connect_external_server(config.external_base_url)
    emit_probe_event(
        probe,
        "demo.server_started",
        server_mode=config.server_mode,
        transport=server_ctx.transport,
        server_data_dir=str(artifacts.server_data_dir),
        base_url=server_ctx.base_url,
    )
    try:
        workflow_engine, conversation_engine, _knowledge_engine = build_default_engines(
            artifacts.engine_dir,
            backend_factory=config.backend_factory,
            provider_settings=config.provider_settings,
        )
        persistence_client = DocumentTreeApiPersistenceClient(
            client=server_ctx.client,
            base_url=server_ctx.base_url,
            transport=server_ctx.transport,
        )
        client = ServerCanonicalKgClient(
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            persistence_client=persistence_client,
        )
        result = client.run_ingest(
            inp=inp,
            workflow_id=config.workflow_id,
            deps=deps,
        )
        artifacts.run_id = result.handle.run_id
        artifacts.status = result.status
        if result.bundle is not None:
            artifacts.canonical_write_confirmed = result.bundle.canonical_write_confirmed
            artifacts.persistence_mode = result.bundle.persistence_mode
            artifacts.kg_authority = result.bundle.kg_authority
        summary = {
            "run_id": artifacts.run_id,
            "status": artifacts.status,
            "canonical_write_confirmed": artifacts.canonical_write_confirmed,
            "persistence_mode": artifacts.persistence_mode,
            "kg_authority": artifacts.kg_authority,
            "probe_path": str(artifacts.probe_path),
            "cache_dir": str(artifacts.cache_dir),
            "engine_dir": str(artifacts.engine_dir),
            "server_data_dir": str(artifacts.server_data_dir),
            "parser_mode": config.parser_mode,
            "server_mode": config.server_mode,
        }
        artifacts.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        emit_probe_event(
            probe,
            "demo.finished",
            run_id=artifacts.run_id,
            status=artifacts.status,
            canonical_write_confirmed=artifacts.canonical_write_confirmed,
        )
        return artifacts
    finally:
        emit_probe_event(probe, "demo.server_stopped", server_mode=config.server_mode)
        server_ctx.__exit__(None, None, None)
        probe.close()
