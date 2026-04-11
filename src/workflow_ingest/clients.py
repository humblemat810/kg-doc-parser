from __future__ import annotations

"""Execution clients for the workflow ingest pipeline.

This module keeps the execution surface explicit and testable:
- `DirectRuntimeIngestClient` runs the local workflow engine end-to-end.
- `ServerCanonicalKgClient` runs the workflow but hands canonical graph
  persistence to an external server client.
- `DocumentTreeApiPersistenceClient` adapts the export bundle into the server
  tree-upsert API payload.

The classes here are intentionally thin wrappers around the workflow runtime so
tests can swap transport and persistence behavior without changing workflow
logic.
"""

from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4

from kogwistar.engine_core.models import Edge, Node

from .design import DEFAULT_WORKFLOW_ID, ensure_ingest_workflow_design
from .models import (
    CanonicalGraphWriteResult,
    IngestRunHandle,
    IngestRunResult,
    WorkflowExportBundle,
    WorkflowIngestInput,
)
from .probe import emit_probe_event


class UnsupportedClientOperation(RuntimeError):
    """Raised when a client path is intentionally not implemented."""

    pass


class CanonicalGraphPersistenceClient(ABC):
    """Protocol for persisting an exported workflow graph bundle."""

    @abstractmethod
    def persist_graph_payload(self, bundle: WorkflowExportBundle) -> CanonicalGraphWriteResult:
        raise NotImplementedError


def _jsonable_payload(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(field_mode="backend", dump_format="json")
        except TypeError:
            return value.model_dump()
    if isinstance(value, dict):
        return {str(k): _jsonable_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable_payload(item) for item in value]
    return value


def _to_temp_id_graph_payload(graph_payload: dict[str, Any]) -> dict[str, Any]:
    """Adapt a canonical export bundle into the server's batch-temp-id contract."""

    nodes = [_jsonable_payload(node) for node in graph_payload.get("nodes", [])]
    edges = [_jsonable_payload(edge) for edge in graph_payload.get("edges", [])]

    node_id_map: dict[str, str] = {}
    for idx, node in enumerate(nodes, start=1):
        original_id = str(node.get("id") or "")
        temp_id = f"nn:{idx}"
        if original_id:
            node_id_map[original_id] = temp_id
        node["id"] = temp_id

    edge_id_map: dict[str, str] = {}
    for idx, edge in enumerate(edges, start=1):
        original_id = str(edge.get("id") or "")
        temp_id = f"ne:{idx}"
        if original_id:
            edge_id_map[original_id] = temp_id
        edge["id"] = temp_id

    for edge in edges:
        edge["source_ids"] = [node_id_map.get(str(x), str(x)) for x in edge.get("source_ids", [])]
        edge["target_ids"] = [node_id_map.get(str(x), str(x)) for x in edge.get("target_ids", [])]
        edge["source_edge_ids"] = [
            edge_id_map.get(str(x), str(x)) for x in edge.get("source_edge_ids", []) or []
        ]
        edge["target_edge_ids"] = [
            edge_id_map.get(str(x), str(x)) for x in edge.get("target_edge_ids", []) or []
        ]

    return {
        "doc_id": str(graph_payload.get("doc_id") or "workflow-ingest-doc"),
        "insertion_method": str(graph_payload.get("insertion_method") or "workflow_ingest"),
        "nodes": nodes,
        "edges": edges,
    }


class DocumentTreeApiPersistenceClient(CanonicalGraphPersistenceClient):
    """Bridge an export bundle to the server document-tree upsert endpoint."""

    def __init__(
        self,
        *,
        client: Any,
        endpoint: str = "/api/document.upsert_tree",
        base_url: str = "",
        transport: str = "server_http_document_tree",
        server_parser_used: bool = False,
    ) -> None:
        self.client = client
        self.endpoint = endpoint
        self.base_url = base_url.rstrip("/")
        self.transport = transport
        self.server_parser_used = server_parser_used

    def persist_graph_payload(self, bundle: WorkflowExportBundle) -> CanonicalGraphWriteResult:
        payload = _to_temp_id_graph_payload(bundle.graph_payload)
        endpoint = self.endpoint
        if self.base_url and not endpoint.startswith("http://") and not endpoint.startswith("https://"):
            endpoint = f"{self.base_url}{endpoint}"
        response = self.client.post(endpoint, json=payload)
        status_code = int(getattr(response, "status_code", 500))
        if status_code >= 400:
            body = getattr(response, "text", "")
            raise RuntimeError(
                f"canonical server persistence failed: status={status_code} body={body}"
            )
        response_json = response.json()
        engine_result = response_json.get("engine_result") or {}
        return CanonicalGraphWriteResult(
            persistence_mode="server_canonical",
            kg_authority="server",
            canonical_write_confirmed=str(response_json.get("status") or "").lower() == "ok",
            nodes_written=int(
                engine_result.get("nodes_added")
                or response_json.get("inserted_nodes")
                or len(payload["nodes"])
            ),
            edges_written=int(
                engine_result.get("edges_added")
                or response_json.get("inserted_edges")
                or len(payload["edges"])
            ),
            transport=self.transport,
            server_parser_used=self.server_parser_used,
        )


class IngestExecutionClient(ABC):
    """Shared ingest client contract used by direct and server-backed flows."""

    @abstractmethod
    def run_ingest(
        self,
        *,
        inp: WorkflowIngestInput,
        workflow_id: str = DEFAULT_WORKFLOW_ID,
        deps: dict[str, Any] | None = None,
    ) -> IngestRunResult:
        raise NotImplementedError

    @abstractmethod
    def resume_ingest(self, **kwargs) -> IngestRunResult:
        raise NotImplementedError

    @abstractmethod
    def persist_graph_payload(self, bundle: WorkflowExportBundle) -> CanonicalGraphWriteResult:
        raise NotImplementedError

    @abstractmethod
    def get_run_trace(self, *, run_id: str) -> list[Any]:
        raise NotImplementedError

    @abstractmethod
    def get_latest_checkpoint(self, *, run_id: str) -> Any:
        raise NotImplementedError


class DirectRuntimeIngestClient(IngestExecutionClient):
    """Run ingest entirely against the local workflow and knowledge engines."""

    def __init__(
        self,
        *,
        workflow_engine,
        conversation_engine,
        knowledge_engine=None,
    ) -> None:
        self.workflow_engine = workflow_engine
        self.conversation_engine = conversation_engine
        self.knowledge_engine = knowledge_engine

    def run_ingest(
        self,
        *,
        inp: WorkflowIngestInput,
        workflow_id: str = DEFAULT_WORKFLOW_ID,
        deps: dict[str, Any] | None = None,
    ) -> IngestRunResult:
        ensure_ingest_workflow_design(self.workflow_engine, workflow_id=workflow_id)
        from .service import build_runtime

        probe = (deps or {}).get("probe")
        runtime = build_runtime(
            workflow_engine=self.workflow_engine,
            conversation_engine=self.conversation_engine,
            deps={
                "knowledge_engine": self.knowledge_engine,
                "persistence_mode": "local_debug",
                "kg_authority": "local",
                "graph_persistence_client": self,
                **(deps or {}),
            },
        )
        run_id = f"run|{inp.request_id}|{uuid4()}"
        emit_probe_event(
            probe,
            "workflow.run_started",
            request_id=inp.request_id,
            workflow_id=workflow_id,
            execution_mode="direct_runtime",
            run_id=run_id,
        )
        run = runtime.run(
            workflow_id=workflow_id,
            conversation_id=f"ingest:{inp.request_id}",
            turn_node_id=f"ingest:{inp.request_id}:turn:{uuid4()}",
            initial_state={"input": inp.model_dump(field_mode="backend", dump_format="json")},
            run_id=run_id,
        )
        bundle = None
        if "export_bundle" in run.final_state:
            bundle = WorkflowExportBundle.model_validate(run.final_state["export_bundle"])
        emit_probe_event(
            probe,
            "workflow.run_finished",
            request_id=inp.request_id,
            workflow_id=workflow_id,
            execution_mode="direct_runtime",
            run_id=run.run_id,
            status=run.status,
        )
        return IngestRunResult(
            handle=IngestRunHandle(
                run_id=run.run_id,
                workflow_id=workflow_id,
                execution_mode="direct_runtime",
            ),
            status=run.status,
            bundle=bundle,
            final_state=dict(run.final_state),
        )

    def resume_ingest(self, **kwargs) -> IngestRunResult:
        from .service import build_runtime

        deps = dict(kwargs.pop("deps", {}) or {})
        probe = deps.get("probe")
        runtime = build_runtime(
            workflow_engine=self.workflow_engine,
            conversation_engine=self.conversation_engine,
            deps={
                "knowledge_engine": self.knowledge_engine,
                "persistence_mode": "local_debug",
                "kg_authority": "local",
                "graph_persistence_client": self,
                **deps,
            },
        )
        emit_probe_event(
            probe,
            "workflow.resume_started",
            execution_mode="direct_runtime",
            run_id=kwargs.get("run_id"),
        )
        resumed = runtime.resume_run(**kwargs)
        bundle = None
        if "export_bundle" in resumed.final_state:
            bundle = WorkflowExportBundle.model_validate(resumed.final_state["export_bundle"])
        workflow_id = kwargs.get("workflow_id", DEFAULT_WORKFLOW_ID)
        emit_probe_event(
            probe,
            "workflow.resume_finished",
            execution_mode="direct_runtime",
            run_id=resumed.run_id,
            status=resumed.status,
        )
        return IngestRunResult(
            handle=IngestRunHandle(
                run_id=resumed.run_id,
                workflow_id=workflow_id,
                execution_mode="direct_runtime",
            ),
            status=resumed.status,
            bundle=bundle,
            final_state=dict(resumed.final_state),
        )

    def persist_graph_payload(self, bundle: WorkflowExportBundle) -> CanonicalGraphWriteResult:
        if self.knowledge_engine is None:
            return CanonicalGraphWriteResult(
                persistence_mode="local_debug",
                kg_authority="local",
                canonical_write_confirmed=False,
                transport="direct_runtime",
                server_parser_used=False,
            )
        nodes_written = 0
        edges_written = 0
        for node in bundle.graph_payload.get("nodes", []):
            node_obj = node if isinstance(node, Node) else Node.model_validate(node)
            if not self.knowledge_engine.persist.exists_node(str(node_obj.safe_get_id())):
                self.knowledge_engine.write.add_node(node_obj)
                nodes_written += 1
        for edge in bundle.graph_payload.get("edges", []):
            edge_obj = edge if isinstance(edge, Edge) else Edge.model_validate(edge)
            if not self.knowledge_engine.persist.exists_edge(str(edge_obj.safe_get_id())):
                self.knowledge_engine.write.add_edge(edge_obj)
                edges_written += 1
        return CanonicalGraphWriteResult(
            persistence_mode="local_debug",
            kg_authority="local",
            canonical_write_confirmed=False,
            nodes_written=nodes_written,
            edges_written=edges_written,
            transport="direct_runtime",
            server_parser_used=False,
        )

    def get_run_trace(self, *, run_id: str) -> list[Any]:
        return list(
            self.conversation_engine.read.get_nodes(
                where={"$and": [{"entity_type": "workflow_step_exec"}, {"run_id": str(run_id)}]}
            )
        )

    def get_latest_checkpoint(self, *, run_id: str) -> Any:
        checkpoints = list(
            self.conversation_engine.read.get_nodes(
                where={"$and": [{"entity_type": "workflow_checkpoint"}, {"run_id": str(run_id)}]}
            )
        )
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda node: int(node.metadata["step_seq"]))


class ServerCanonicalKgClient(IngestExecutionClient):
    """Run ingest locally but delegate canonical graph persistence to a server."""

    def __init__(
        self,
        *,
        workflow_engine,
        conversation_engine,
        persistence_client: CanonicalGraphPersistenceClient,
    ) -> None:
        self.workflow_engine = workflow_engine
        self.conversation_engine = conversation_engine
        self.persistence_client = persistence_client

    def run_ingest(
        self,
        *,
        inp: WorkflowIngestInput,
        workflow_id: str = DEFAULT_WORKFLOW_ID,
        deps: dict[str, Any] | None = None,
    ) -> IngestRunResult:
        ensure_ingest_workflow_design(self.workflow_engine, workflow_id=workflow_id)
        from .service import build_runtime

        probe = (deps or {}).get("probe")
        runtime = build_runtime(
            workflow_engine=self.workflow_engine,
            conversation_engine=self.conversation_engine,
            deps={
                "knowledge_engine": None,
                "persistence_mode": "server_canonical",
                "kg_authority": "server",
                "graph_persistence_client": self.persistence_client,
                **(deps or {}),
            },
        )
        run_id = f"run|{inp.request_id}|{uuid4()}"
        emit_probe_event(
            probe,
            "workflow.run_started",
            request_id=inp.request_id,
            workflow_id=workflow_id,
            execution_mode="server_canonical_client",
            run_id=run_id,
        )
        run = runtime.run(
            workflow_id=workflow_id,
            conversation_id=f"ingest:{inp.request_id}",
            turn_node_id=f"ingest:{inp.request_id}:turn:{uuid4()}",
            initial_state={"input": inp.model_dump(field_mode="backend", dump_format="json")},
            run_id=run_id,
        )
        bundle = None
        if "export_bundle" in run.final_state:
            bundle = WorkflowExportBundle.model_validate(run.final_state["export_bundle"])
        emit_probe_event(
            probe,
            "workflow.run_finished",
            request_id=inp.request_id,
            workflow_id=workflow_id,
            execution_mode="server_canonical_client",
            run_id=run.run_id,
            status=run.status,
        )
        return IngestRunResult(
            handle=IngestRunHandle(
                run_id=run.run_id,
                workflow_id=workflow_id,
                execution_mode="server_canonical_client",
            ),
            status=run.status,
            bundle=bundle,
            final_state=dict(run.final_state),
        )

    def resume_ingest(self, **kwargs) -> IngestRunResult:
        raise UnsupportedClientOperation(
            "remote/server-backed runtime resume is not implemented in this repo"
        )

    def persist_graph_payload(self, bundle: WorkflowExportBundle) -> CanonicalGraphWriteResult:
        return self.persistence_client.persist_graph_payload(bundle)

    def get_run_trace(self, *, run_id: str) -> list[Any]:
        raise UnsupportedClientOperation(
            "server-backed trace retrieval is not implemented in this repo"
        )

    def get_latest_checkpoint(self, *, run_id: str) -> Any:
        raise UnsupportedClientOperation(
            "server-backed checkpoint retrieval is not implemented in this repo"
        )
