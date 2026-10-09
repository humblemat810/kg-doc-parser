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

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Literal, Protocol, cast
from uuid import uuid4

from kogwistar.engine_core.models import Edge, Node
from kogwistar.json_types import JsonValue
from kogwistar.runtime.models import StepRunResult

from .design import DEFAULT_WORKFLOW_ID, ensure_ingest_workflow_design
from .models import (
    CanonicalGraphWriteResult,
    IngestRunHandle,
    IngestRunResult,
    WorkflowExportBundle,
    WorkflowIngestInput,
)
from .probe import WorkflowProbe, emit_probe_event


class UnsupportedClientOperation(RuntimeError):
    """Raised when a client path is intentionally not implemented."""


IngestStatus = Literal["succeeded", "failed", "failure", "suspended"]
JsonObject = dict[str, JsonValue]


def _workflow_probe(deps: Mapping[str, object] | None) -> WorkflowProbe | None:
    value = (deps or {}).get("probe")
    return value if isinstance(value, WorkflowProbe) else None


def _state_json(value: Mapping[str, object]) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], dict(value))


class HttpResponseLike(Protocol):
    """Small response surface required from an HTTP transport adapter."""

    status_code: int

    def json(self) -> object: ...


class HttpClientLike(Protocol):
    """HTTP client boundary used by server-backed graph persistence.

    The keyword payload remains opaque because transports such as ``httpx``
    and ``requests`` expose different concrete request types.
    """

    def post(self, endpoint: str, **kwargs: object) -> HttpResponseLike: ...


def _ingest_status(value: str) -> IngestStatus:
    if value not in {"succeeded", "failed", "failure", "suspended"}:
        raise ValueError(f"unsupported runtime status: {value}")
    return cast(IngestStatus, value)


class CanonicalGraphPersistenceClient(ABC):
    """Protocol for persisting an exported workflow graph bundle."""

    @abstractmethod
    def persist_graph_payload(self, bundle: WorkflowExportBundle) -> CanonicalGraphWriteResult:
        raise NotImplementedError


def _jsonable_payload(value: object) -> object:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(field_mode="backend", dump_format="json")
        except TypeError:
            return model_dump()
    if isinstance(value, Mapping):
        return {str(k): _jsonable_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable_payload(item) for item in value]
    return value


def _record_list(value: object, *, field_name: str) -> list[JsonObject]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"graph payload field {field_name!r} must be a list")
    records: list[JsonObject] = []
    for item in value:
        converted = _jsonable_payload(item)
        if not isinstance(converted, dict):
            raise TypeError(f"graph payload {field_name!r} items must be objects")
        records.append(cast(JsonObject, {str(key): payload for key, payload in converted.items()}))
    return records


def _string_list(value: object, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"graph payload field {field_name!r} must be a list")
    return [str(item) for item in value]


def _json_int(value: object, default: int = 0) -> int:
    """Read an integer counter from an untrusted JSON response."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _to_temp_id_graph_payload(graph_payload: Mapping[str, object]) -> JsonObject:
    """Adapt a canonical export bundle into the server's batch-temp-id contract."""

    nodes = _record_list(graph_payload.get("nodes", []), field_name="nodes")
    edges = _record_list(graph_payload.get("edges", []), field_name="edges")

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
        edge["source_ids"] = [
            node_id_map.get(value, value)
            for value in _string_list(edge.get("source_ids"), field_name="source_ids")
        ]
        edge["target_ids"] = [
            node_id_map.get(value, value)
            for value in _string_list(edge.get("target_ids"), field_name="target_ids")
        ]
        edge["source_edge_ids"] = [
            edge_id_map.get(value, value)
            for value in _string_list(edge.get("source_edge_ids"), field_name="source_edge_ids")
        ]
        edge["target_edge_ids"] = [
            edge_id_map.get(value, value)
            for value in _string_list(edge.get("target_edge_ids"), field_name="target_edge_ids")
        ]

    return {
        "doc_id": str(graph_payload.get("doc_id") or "workflow-ingest-doc"),
        "insertion_method": str(graph_payload.get("insertion_method") or "workflow_ingest"),
        "nodes": cast(JsonValue, nodes),
        "edges": cast(JsonValue, edges),
    }


class DocumentTreeApiPersistenceClient(CanonicalGraphPersistenceClient):
    """Bridge an export bundle to the server document-tree upsert endpoint."""

    def __init__(
        self,
        *,
        client: HttpClientLike,
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
        node_count = len(_record_list(bundle.graph_payload.get("nodes", []), field_name="nodes"))
        edge_count = len(_record_list(bundle.graph_payload.get("edges", []), field_name="edges"))
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
        response_payload = response.json()
        if not isinstance(response_payload, Mapping):
            raise TypeError("canonical server persistence returned a non-object JSON body")
        response_json = cast(Mapping[str, JsonValue], response_payload)
        raw_engine_result = response_json.get("engine_result")
        engine_result = raw_engine_result if isinstance(raw_engine_result, Mapping) else {}
        return CanonicalGraphWriteResult(
            persistence_mode="server_canonical",
            kg_authority="server",
            canonical_write_confirmed=str(response_json.get("status") or "").lower() == "ok",
            nodes_written=_json_int(
                engine_result.get("nodes_added")
                or response_json.get("inserted_nodes")
                or node_count
            ),
            edges_written=_json_int(
                engine_result.get("edges_added")
                or response_json.get("inserted_edges")
                or edge_count
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
        deps: dict[str, object] | None = None,
        run_id: str | None = None,
        resume_from_checkpoint: bool = False,
    ) -> IngestRunResult:
        raise NotImplementedError

    @abstractmethod
    def resume_ingest(self, **kwargs: object) -> IngestRunResult:
        raise NotImplementedError

    @abstractmethod
    def persist_graph_payload(self, bundle: WorkflowExportBundle) -> CanonicalGraphWriteResult:
        raise NotImplementedError

    @abstractmethod
    def get_run_trace(self, *, run_id: str) -> list[Node]:
        raise NotImplementedError

    @abstractmethod
    def get_latest_checkpoint(self, *, run_id: str) -> Node | None:
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
        deps: dict[str, object] | None = None,
        run_id: str | None = None,
        resume_from_checkpoint: bool = False,
    ) -> IngestRunResult:
        ensure_ingest_workflow_design(self.workflow_engine, workflow_id=workflow_id)
        from .service import build_runtime

        probe = _workflow_probe(deps)
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
        effective_run_id = str(run_id or f"run|{inp.request_id}|{uuid4()}")
        emit_probe_event(
            probe,
            "workflow.run_started",
            request_id=inp.request_id,
            workflow_id=workflow_id,
            execution_mode="direct_runtime",
            run_id=effective_run_id,
        )
        conversation_id = f"ingest:{inp.request_id}"
        turn_node_id = f"ingest:{inp.request_id}:turn"
        if resume_from_checkpoint:
            try:
                run = runtime.resume_from_latest_checkpoint(
                    run_id=effective_run_id,
                    workflow_id=workflow_id,
                    conversation_id=conversation_id,
                    turn_node_id=turn_node_id,
                )
            except ValueError as exc:
                # A first attempt can die before the first checkpoint. Treat
                # that as an ordinary fresh execution, not as a reason to
                # discard the stable parser run identity.
                if "no checkpoints found" not in str(exc).lower():
                    raise
                run = runtime.run(
                    workflow_id=workflow_id,
                    conversation_id=conversation_id,
                    turn_node_id=turn_node_id,
                    initial_state={"input": inp.model_dump(field_mode="backend", dump_format="json")},
                    run_id=effective_run_id,
                )
        else:
            run = runtime.run(
                workflow_id=workflow_id,
                conversation_id=conversation_id,
                turn_node_id=turn_node_id,
                initial_state={"input": inp.model_dump(field_mode="backend", dump_format="json")},
                run_id=effective_run_id,
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
            status=_ingest_status(run.status),
        )
        return IngestRunResult(
            handle=IngestRunHandle(
                run_id=run.run_id,
                workflow_id=workflow_id,
                execution_mode="direct_runtime",
            ),
            status=_ingest_status(run.status),
            bundle=bundle,
            final_state=_state_json(run.final_state),
        )

    def resume_ingest(self, **kwargs: object) -> IngestRunResult:
        from .service import build_runtime

        raw_deps = kwargs.pop("deps", None)
        deps = dict(raw_deps) if isinstance(raw_deps, Mapping) else {}
        probe = _workflow_probe(deps)
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
        resumed = runtime.resume_run(
            run_id=str(kwargs["run_id"]),
            suspended_node_id=str(kwargs["suspended_node_id"]),
            suspended_token_id=str(kwargs["suspended_token_id"]),
            client_result=cast(StepRunResult, kwargs["client_result"]),
            workflow_id=str(kwargs["workflow_id"]),
            conversation_id=str(kwargs["conversation_id"]),
            turn_node_id=str(kwargs["turn_node_id"]),
        )
        bundle = None
        if "export_bundle" in resumed.final_state:
            bundle = WorkflowExportBundle.model_validate(resumed.final_state["export_bundle"])
        workflow_id = str(kwargs.get("workflow_id", DEFAULT_WORKFLOW_ID))
        emit_probe_event(
            probe,
            "workflow.resume_finished",
            execution_mode="direct_runtime",
            run_id=resumed.run_id,
            status=_ingest_status(resumed.status),
        )
        return IngestRunResult(
            handle=IngestRunHandle(
                run_id=resumed.run_id,
                workflow_id=workflow_id,
                execution_mode="direct_runtime",
            ),
            status=_ingest_status(resumed.status),
            bundle=bundle,
            final_state=_state_json(resumed.final_state),
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
        for node in _record_list(bundle.graph_payload.get("nodes", []), field_name="nodes"):
            node_obj = node if isinstance(node, Node) else Node.model_validate(node)
            if not self.knowledge_engine.persist.exists_node(str(node_obj.safe_get_id())):
                self.knowledge_engine.write.add_node(node_obj)
                nodes_written += 1
        for edge in _record_list(bundle.graph_payload.get("edges", []), field_name="edges"):
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

    def get_run_trace(self, *, run_id: str) -> list[Node]:
        return list(
            self.conversation_engine.read.get_nodes(
                where={"$and": [{"entity_type": "workflow_step_exec"}, {"run_id": str(run_id)}]}
            )
        )

    def get_latest_checkpoint(self, *, run_id: str) -> Node | None:
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
        deps: dict[str, object] | None = None,
        run_id: str | None = None,
        resume_from_checkpoint: bool = False,
    ) -> IngestRunResult:
        if resume_from_checkpoint:
            raise UnsupportedClientOperation(
                "server-backed ingest does not support local checkpoint resume"
            )
        ensure_ingest_workflow_design(self.workflow_engine, workflow_id=workflow_id)
        from .service import build_runtime

        probe = _workflow_probe(deps)
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
        effective_run_id = str(run_id or f"run|{inp.request_id}|{uuid4()}")
        emit_probe_event(
            probe,
            "workflow.run_started",
            request_id=inp.request_id,
            workflow_id=workflow_id,
            execution_mode="server_canonical_client",
            run_id=effective_run_id,
        )
        run = runtime.run(
            workflow_id=workflow_id,
            conversation_id=f"ingest:{inp.request_id}",
            turn_node_id=f"ingest:{inp.request_id}:turn:{uuid4()}",
            initial_state={"input": inp.model_dump(field_mode="backend", dump_format="json")},
            run_id=effective_run_id,
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
            status=_ingest_status(run.status),
        )
        return IngestRunResult(
            handle=IngestRunHandle(
                run_id=run.run_id,
                workflow_id=workflow_id,
                execution_mode="server_canonical_client",
            ),
            status=_ingest_status(run.status),
            bundle=bundle,
            final_state=_state_json(run.final_state),
        )

    def resume_ingest(self, **kwargs: object) -> IngestRunResult:
        raise UnsupportedClientOperation(
            "remote/server-backed runtime resume is not implemented in this repo"
        )

    def persist_graph_payload(self, bundle: WorkflowExportBundle) -> CanonicalGraphWriteResult:
        return self.persistence_client.persist_graph_payload(bundle)

    def get_run_trace(self, *, run_id: str) -> list[Node]:
        raise UnsupportedClientOperation(
            "server-backed trace retrieval is not implemented in this repo"
        )

    def get_latest_checkpoint(self, *, run_id: str) -> Node | None:
        raise UnsupportedClientOperation(
            "server-backed checkpoint retrieval is not implemented in this repo"
        )
