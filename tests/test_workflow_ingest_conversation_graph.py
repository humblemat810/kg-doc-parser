from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from kogwistar.runtime.models import RunSuccess, RunSuspended
from kogwistar.runtime.runtime import WorkflowRuntime

from _kogwistar_test_helpers import load_kogwistar_fake_backend
from src.workflow_ingest.design import DEFAULT_WORKFLOW_ID, ensure_ingest_workflow_design
from src.workflow_ingest.handlers import build_ingest_step_resolver
from src.workflow_ingest.models import WorkflowIngestInput
from src.workflow_ingest.semantics import HydratedTextPointer, SemanticNode
from src.workflow_ingest.service import build_default_engines, run_ingest_workflow


def _scratch(name: str) -> Path:
    root = Path("tests") / ".tmp_workflow_ingest_conversation"
    path = root / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fake_semantic_tree(*, collection, parser_input_dict, parser_source_map):
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


def _partial_semantic_tree(*, collection, parser_input_dict, parser_source_map):
    first_id = next(iter(parser_source_map))
    text = parser_source_map[first_id]["text"]
    partial_end = max(0, (len(text) // 2) - 1)
    return SemanticNode(
        title=collection.title,
        node_type="DOCUMENT_ROOT",
        total_content_pointers=[],
        child_nodes=[
            SemanticNode(
                title="partial",
                node_type="TEXT_FLOW",
                total_content_pointers=[
                    HydratedTextPointer(
                        source_cluster_id=first_id,
                        start_char=0,
                        end_char=partial_end,
                        verbatim_text=text[: partial_end + 1],
                    )
                ],
                child_nodes=[],
                level_from_root=1,
            )
        ],
        level_from_root=0,
    )


def _conversation_nodes(conversation_engine, *, entity_type: str, run_id: str):
    return conversation_engine.read.get_nodes(
        where={
            "$and": [
                {"entity_type": entity_type},
                {"run_id": str(run_id)},
            ]
        }
    )


class _LocalDebugPersistenceClient:
    def __init__(self, knowledge_engine) -> None:
        self.knowledge_engine = knowledge_engine

    def persist_graph_payload(self, bundle):
        from kogwistar.engine_core.models import Edge, Node

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
        return {
            "persistence_mode": "local_debug",
            "kg_authority": "local",
            "canonical_write_confirmed": False,
            "nodes_written": nodes_written,
            "edges_written": edges_written,
            "transport": "direct_runtime",
            "server_parser_used": False,
        }


def test_conversation_graph_persists_trace_and_checkpoints_for_success():
    scratch_dir = _scratch("success")
    fake_backend = load_kogwistar_fake_backend()
    workflow_engine, conversation_engine, knowledge_engine = build_default_engines(
        scratch_dir / "engines",
        backend_factory=fake_backend,
    )
    inp = WorkflowIngestInput.from_text(
        document_id="conv-success",
        text="Alpha clause\nBeta clause",
        title="Conversation Success",
    )

    run, bundle = run_ingest_workflow(
        inp=inp,
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        deps={"parse_semantic_fn": _fake_semantic_tree},
    )

    assert run.status == "succeeded"
    assert bundle is not None

    checkpoints = _conversation_nodes(
        conversation_engine,
        entity_type="workflow_checkpoint",
        run_id=run.run_id,
    )
    step_execs = _conversation_nodes(
        conversation_engine,
        entity_type="workflow_step_exec",
        run_id=run.run_id,
    )
    workflow_runs = _conversation_nodes(
        conversation_engine,
        entity_type="workflow_run",
        run_id=run.run_id,
    )

    assert checkpoints
    assert step_execs
    assert workflow_runs

    latest_ckpt = max(checkpoints, key=lambda node: int(node.metadata["step_seq"]))
    state = json.loads(latest_ckpt.metadata["state_json"])
    ops = {node.metadata["op"] for node in step_execs}

    assert "normalized_input" in state
    assert "export_bundle" in state
    assert "propose_layer_breakdown" in ops
    assert "export_graph" in ops


def test_conversation_graph_keeps_checkpoint_snapshot_for_failure():
    scratch_dir = _scratch("failure")
    fake_backend = load_kogwistar_fake_backend()
    workflow_engine, conversation_engine, knowledge_engine = build_default_engines(
        scratch_dir / "engines",
        backend_factory=fake_backend,
    )
    inp = WorkflowIngestInput.from_text(
        document_id="conv-failure",
        text="First section\nSecond section",
        title="Conversation Failure",
    )

    run, bundle = run_ingest_workflow(
        inp=inp,
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        knowledge_engine=knowledge_engine,
        deps={
            "parse_semantic_fn": _partial_semantic_tree,
            "coverage_threshold": 0.99,
        },
    )

    assert run.status in {"failed", "failure"}
    assert bundle is None

    checkpoints = _conversation_nodes(
        conversation_engine,
        entity_type="workflow_checkpoint",
        run_id=run.run_id,
    )
    step_execs = _conversation_nodes(
        conversation_engine,
        entity_type="workflow_step_exec",
        run_id=run.run_id,
    )

    assert checkpoints
    assert step_execs

    latest_ckpt = max(checkpoints, key=lambda node: int(node.metadata["step_seq"]))
    state = json.loads(latest_ckpt.metadata["state_json"])
    failed_steps = [
        node for node in step_execs if str(node.metadata.get("status")) == "failure"
    ]

    assert "semantic_tree" in state
    assert "workflow_errors" in state
    assert "export_bundle" not in state
    assert failed_steps
    assert failed_steps[-1].metadata["op"] == "validate_tree"


def test_conversation_graph_resume_from_suspended_checkpoint():
    scratch_dir = _scratch("resume")
    fake_backend = load_kogwistar_fake_backend()
    workflow_engine, conversation_engine, knowledge_engine = build_default_engines(
        scratch_dir / "engines",
        backend_factory=fake_backend,
    )
    ensure_ingest_workflow_design(workflow_engine, workflow_id=DEFAULT_WORKFLOW_ID)

    resolver = build_ingest_step_resolver(
        deps={
            "knowledge_engine": knowledge_engine,
            "graph_persistence_client": _LocalDebugPersistenceClient(knowledge_engine),
            "persistence_mode": "local_debug",
            "kg_authority": "local",
            "propose_layer_fn": lambda **kwargs: {
                "children": [],
                "satisfied": True,
                "reasoning_history": [],
            },
        }
    )

    @resolver.register("propose_layer_breakdown")
    def _suspend_propose_layer_breakdown(ctx):
        return RunSuspended(
            conversation_node_id=None,
            state_update=[],
            resume_payload={"kind": "manual_review"},
        )

    runtime = WorkflowRuntime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=resolver.resolve,
        predicate_registry={},
        trace=False,
    )

    inp = WorkflowIngestInput.from_text(
        document_id="conv-resume",
        text="Alpha clause\nBeta clause",
        title="Conversation Resume",
    )
    conversation_id = f"ingest:{inp.request_id}"
    turn_node_id = f"ingest:{inp.request_id}:turn:{uuid4()}"

    run = runtime.run(
        workflow_id=DEFAULT_WORKFLOW_ID,
        conversation_id=conversation_id,
        turn_node_id=turn_node_id,
        initial_state={
            "input": inp.model_dump(field_mode="backend", dump_format="json"),
        },
        run_id=f"run|{inp.request_id}|{uuid4()}",
    )

    assert run.status == "suspended"

    suspended_steps = _conversation_nodes(
        conversation_engine,
        entity_type="workflow_step_exec",
        run_id=run.run_id,
    )
    suspended_step = next(
        node for node in suspended_steps if str(node.metadata.get("status")) == "suspended"
    )

    normalized = WorkflowIngestInput.model_validate(run.final_state["normalized_input"])
    resumed_tree = _fake_semantic_tree(
        collection=normalized.collections[0],
        parser_input_dict=run.final_state["parser_input_dict"],
        parser_source_map=run.final_state["parser_source_map"],
    )

    resumed = runtime.resume_run(
        run_id=run.run_id,
        suspended_node_id=suspended_step.metadata["workflow_node_id"],
        suspended_token_id=suspended_step.metadata["token_id"],
        client_result=RunSuccess(
            conversation_node_id=None,
            state_update=[
                (
                    "u",
                    {
                        "current_layer_result": {
                            "children": [
                                {
                                    "node_id": child.node_id,
                                    "parent_node_id": child.parent_id or resumed_tree.node_id,
                                    "title": child.title,
                                    "node_type": child.node_type,
                                    "total_content_pointers": [
                                        pointer.model_dump(mode="json")
                                        for pointer in child.total_content_pointers
                                    ],
                                    "expandable": True,
                                    "metadata": {},
                                }
                                for child in resumed_tree.child_nodes
                            ],
                            "satisfied": True,
                            "reasoning_history": [],
                            "review_rounds": 0,
                            "metadata": {},
                        }
                    },
                )
            ],
            _route_next=["review_cud_proposal"],
        ),
        workflow_id=DEFAULT_WORKFLOW_ID,
        conversation_id=conversation_id,
        turn_node_id=turn_node_id,
    )

    assert resumed.status == "succeeded"
    assert "export_bundle" in resumed.final_state

    checkpoints = _conversation_nodes(
        conversation_engine,
        entity_type="workflow_checkpoint",
        run_id=run.run_id,
    )
    step_execs = _conversation_nodes(
        conversation_engine,
        entity_type="workflow_step_exec",
        run_id=run.run_id,
    )

    assert len(checkpoints) >= 2
    assert any(str(node.metadata.get("status")) == "suspended" for node in step_execs)
    assert any(str(node.metadata.get("op")) == "export_graph" for node in step_execs)
