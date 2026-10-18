from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

from _kogwistar_test_helpers import build_workflow_engine_triplet
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.rust_meta_sqlite import build_sqlite_meta_store

from kg_doc_parser.models import OCRClusterResponse, TextCluster
from kg_doc_parser.workflow_ingest import (
    EmbeddingProviderConfig,
    OCRImagePayload,
    ProviderEndpointConfig,
    Summary,
    SummaryPolicy,
    WorkflowProviderSettings,
    run_ocr_ingest_workflow,
)
from kg_doc_parser.workflow_ingest.document_grouping import (
    GroupingPassContext,
    GroupProposal,
    GroupProposalBatch,
)
from kg_doc_parser.workflow_ingest.document_models import (
    AdaptiveGroupingPolicy,
    DocumentTree,
    EvidenceSlice,
)
from kg_doc_parser.workflow_ingest.semantics import HydratedTextPointer, SemanticNode
from kg_doc_parser.workflow_ingest.service import _TinyEmbeddingFunction


def _response(page_number: int) -> OCRClusterResponse:
    return OCRClusterResponse(
        OCR_text_clusters=[
            TextCluster(
                text=f"root page {page_number}",
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


def _semantic_tree(
    *,
    collection: object,
    parser_input_dict: dict[str, object],
    parser_source_map: dict[str, dict[str, object]],
) -> SemanticNode:
    del parser_input_dict
    root = SemanticNode(
        title=str(getattr(collection, "title", "Fresh Root")),
        node_type="DOCUMENT_ROOT",
        total_content_pointers=[],
        child_nodes=[],
        level_from_root=0,
    )
    for cluster_id, record in parser_source_map.items():
        text = record.get("text")
        if not isinstance(text, str) or not text:
            continue
        root.child_nodes.append(
            SemanticNode(
                title=f"section:{cluster_id}",
                node_type="TEXT_FLOW",
                total_content_pointers=[
                    HydratedTextPointer(
                        source_cluster_id=cluster_id,
                        start_char=0,
                        end_char=len(text) - 1,
                        verbatim_text=text,
                    )
                ],
                child_nodes=[],
                level_from_root=1,
                parent_id=root.node_id,
            )
        )
    return root


class _FreshProcessGroupProposer:
    configuration_fingerprint = "fresh-process-grounded-summary-grouping-v1"

    def __call__(self, context: GroupingPassContext, /) -> GroupProposalBatch:
        if context.pass_index != 0 or len(context.frontier_ids) < 2:
            return GroupProposalBatch()
        return GroupProposalBatch(
            proposals=(
                GroupProposal(
                    kind="section",
                    members=context.frontier_ids[:2],
                ),
            )
        )


class _FreshProcessGroundedSummaryGenerator:
    configuration_fingerprint = "fresh-process-grounded-summary-v1"

    def __init__(
        self,
        calls_path: Path,
        identity_calls_path: Path | None = None,
    ) -> None:
        self.calls_path = calls_path
        self.identity_calls_path = identity_calls_path

    def __call__(
        self,
        *,
        tree: DocumentTree,
        node_id: str,
        evidence_slices: tuple[EvidenceSlice, ...],
        input_fingerprint: str,
    ) -> Summary:
        del tree
        with self.calls_path.open("a", encoding="utf-8") as stream:
            stream.write("called\n")
        if self.identity_calls_path is not None:
            with self.identity_calls_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {"node_id": node_id, "input_fingerprint": input_fingerprint},
                        sort_keys=True,
                    )
                    + "\n"
                )
        return Summary(
            status="generated",
            text="Grounded summary of the accepted section.",
            evidence_item_ids=tuple(sorted({item.item_id for item in evidence_slices})),
            provider_model="fresh-process-test-summary",
            prompt_revision="fresh-process-summary-v1",
            input_fingerprint=input_fingerprint,
        )


def main() -> int:
    os.environ.setdefault("KOGWISTAR_IMPL_META_STORE", "python")
    scratch = Path(sys.argv[1])
    mode = sys.argv[2]
    page_count_path = scratch / "page-count.txt"
    page_count = int(page_count_path.read_text(encoding="utf-8")) if page_count_path.exists() else 2
    page_numbers = tuple(range(1, page_count + 1))
    image_payloads = [
        OCRImagePayload(
            page_number=page,
            image_path=str(scratch / "source" / f"page-{page}.png"),
        )
        for page in page_numbers
    ]
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fresh-root-ocr"),
        parser=ProviderEndpointConfig(provider="fake", model="fresh-root-parser"),
        embedding=EmbeddingProviderConfig(
            provider="fake", model="fresh-root-embed", dimension=1
        ),
        triage_enabled=False,
    )
    calls: list[int] = []

    def runner(
        image_path: Path,
        page_number: int,
        provider_settings: WorkflowProviderSettings,
    ) -> OCRClusterResponse:
        del image_path, provider_settings
        calls.append(page_number)
        if mode == "interrupt" and page_number == 2:
            raise RuntimeError("fresh root interruption")
        hold_page = 64 if mode.endswith("long-hold") else 2
        if mode in {
            "hold",
            "long-hold",
            "durable-long-hold",
            "durable-summary-long-hold",
            "durable-hold",
            "durable-summary-hold",
        } and page_number == hold_page:
            (scratch / f"page-{hold_page}-started").write_text("ready", encoding="utf-8")
            time.sleep(120.0)
        if mode.startswith("race-") and page_number == 2:
            (scratch / f"{mode}-page-2-started").write_text(
                "ready", encoding="utf-8"
            )
            time.sleep(3.0)
        if mode in {
            "resume",
            "durable-resume",
            "durable-summary-publish-hold",
            "durable-summary-resume",
            "race-a",
            "race-b",
        } and page_number == 1:
            raise AssertionError("accepted page was replayed")
        if (
            mode.endswith("long-resume")
            or mode
            in {
                "durable-summary-long-publish-hold",
                "durable-summary-long-resume",
            }
        ) and page_number <= 63:
            raise AssertionError("page before the termination boundary was replayed")
        return _response(page_number)

    workflow_engine, conversation_engine, knowledge_engine = build_workflow_engine_triplet(
        scratch / "engines", "in_memory"
    )
    if mode.startswith("durable-"):
        knowledge_engine.close()
        knowledge_engine = GraphKnowledgeEngine(
            persist_directory=str(scratch / "engines" / "durable-knowledge"),
            kg_graph_type="knowledge",
            embedding_function=_TinyEmbeddingFunction(),
        )
    for name, engine in (
        ("workflow", workflow_engine),
        ("conversation", conversation_engine),
        ("knowledge", knowledge_engine),
    ):
        metadata = build_sqlite_meta_store(scratch / "engines" / name, "meta.sqlite")
        metadata.ensure_initialized()
        engine.metadata = metadata
        engine._phase1_enable_index_jobs = False
    try:
        lease_ms = int(os.environ.get("OCR_TEST_LEASE_MS", "30000"))
        if mode in {
            "resume",
            "long-resume",
            "durable-long-resume",
            "durable-summary-long-publish-hold",
            "durable-summary-long-resume",
            "durable-resume",
            "durable-summary-publish-hold",
            "durable-summary-resume",
            "race-a",
            "race-b",
        }:
            # Simulate a fresh process after the crash-recovery lease expires.
            # The production default remains long-lived; this test-only lease
            # keeps the subprocess test bounded without bypassing ownership.
            time.sleep((lease_ms + 200) / 1000)

        summary_mode = mode.startswith("durable-summary-")
        long_summary_mode = mode in {
            "durable-summary-long-hold",
            "durable-summary-long-publish-hold",
            "durable-summary-long-resume",
            "durable-summary-reference",
        }

        def parse_semantic_fn(
            *,
            collection: object,
            parser_input_dict: dict[str, object],
            parser_source_map: dict[str, dict[str, object]],
        ) -> SemanticNode:
            if summary_mode:
                grouping_tree = parser_input_dict.get("document_tree")
                if not isinstance(grouping_tree, dict):
                    raise AssertionError("semantic parser did not receive the grouping tree")
                nodes = grouping_tree.get("nodes")
                if not isinstance(nodes, list):
                    raise AssertionError("semantic parser received malformed grouping nodes")
                generated_summaries = [
                    node["summary"]
                    for node in nodes
                    if isinstance(node, dict)
                    and isinstance(node.get("summary"), dict)
                    and node["summary"].get("status") == "generated"
                ]
                if not generated_summaries:
                    raise AssertionError(
                        "semantic parser must receive an accepted grounded summary"
                    )
                if mode in {
                    "durable-summary-publish-hold",
                    "durable-summary-long-publish-hold",
                }:
                    grouping_tree_digest = hashlib.sha256(
                        json.dumps(
                            grouping_tree,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                        ).encode("utf-8")
                    ).hexdigest()
                    (scratch / "semantic-input-grouping-tree-sha256.txt").write_text(
                        grouping_tree_digest,
                        encoding="utf-8",
                    )
                    (scratch / "grounded-summary-ready").write_text(
                        "ready", encoding="utf-8"
                    )
                    time.sleep(120.0)
            return _semantic_tree(
                collection=collection,
                parser_input_dict=parser_input_dict,
                parser_source_map=parser_source_map,
            )

        result, bundle, _ = run_ocr_ingest_workflow(
            document_id="fresh-root-document",
            title="Fresh Root",
            output_dir=scratch / "artifacts",
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            knowledge_engine=knowledge_engine,
            image_payloads=image_payloads,
            provider_settings=settings,
            ocr_runner=runner,
            deps={"parse_semantic_fn": parse_semantic_fn},
            adaptive_grouping_policy=(
                AdaptiveGroupingPolicy(
                    maximum_rounds=2,
                    window_size=64 if long_summary_mode else 3,
                    window_overlap=31 if long_summary_mode else 1,
                )
                if summary_mode
                else None
            ),
            grouping_proposer=_FreshProcessGroupProposer() if summary_mode else None,
            summary_policy=(
                SummaryPolicy(
                    eligible_kinds=("section",),
                    minimum_source_items=2,
                    minimum_source_chars=1,
                    maximum_targets=1,
                )
                if summary_mode
                else None
            ),
            grounded_summary_generator=(
                _FreshProcessGroundedSummaryGenerator(
                    scratch / "summary-calls.txt",
                    scratch / "summary-call-identities.jsonl"
                    if long_summary_mode
                    else None,
                )
                if summary_mode
                else None
            ),
            run_id="ocr|fresh-root-document",
            resume_from_checkpoint=mode
            in {
                "resume",
                "long-resume",
                "durable-long-resume",
                "durable-summary-publish-hold",
                "durable-summary-long-publish-hold",
                "durable-summary-long-resume",
                "durable-summary-resume",
                "race-a",
                "race-b",
            },
            run_ownership_lease_ms=lease_ms,
        )
    except Exception as error:
        if mode != "interrupt":
            error_name = (
                f"{mode}-error.txt" if mode.startswith("race-") else "resume-error.txt"
            )
            (scratch / error_name).write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            raise
        (scratch / "interrupt-result.json").write_text(
            json.dumps({"error": str(error), "calls": calls}), encoding="utf-8"
        )
        return 17

    if mode not in {
        "durable-reference",
        "durable-summary-reference",
        "resume",
        "long-resume",
        "durable-long-resume",
        "durable-resume",
        "durable-summary-resume",
        "durable-summary-long-resume",
        "resume-missing-artifact",
        "race-a",
        "race-b",
    } or result.status != "succeeded" or bundle is None:
        raise AssertionError("fresh root resume did not produce one accepted result")
    result_name = (
        f"{mode}-result.json"
        if mode.startswith("race-") or mode == "resume-missing-artifact"
        else "resume-result.json"
    )
    (scratch / result_name).write_text(
        json.dumps(
            {
                "status": result.status,
                "run_id": result.run_id,
                "calls": calls,
                "has_grouping_tree": bundle.grouping_tree is not None,
                **(
                    {
                        "grouping_summary_count": sum(
                            1
                            for node in bundle.grouping_tree.get("nodes", [])
                            if isinstance(node, dict)
                            and isinstance(node.get("summary"), dict)
                            and node["summary"].get("status") == "generated"
                        )
                    }
                    if summary_mode and isinstance(bundle.grouping_tree, dict)
                    else {}
                ),
            }
        ),
        encoding="utf-8",
    )
    lineage = [
        {
            "id": str(edge.id),
            "parent_run_id": str((edge.metadata or {}).get("parent_run_id")),
            "workflow_id": str((edge.metadata or {}).get("workflow_id")),
            "child_run_id": str((edge.metadata or {}).get("child_run_id")),
        }
        for edge in conversation_engine.read.get_edges(limit=10_000)
        if (edge.metadata or {}).get("relation") == "wf_invoked"
    ]
    graph_node_ids = [
        str(node["id"])
        for node in bundle.graph_payload.get("nodes", [])
        if isinstance(node, dict) and isinstance(node.get("id"), str)
    ]
    graph_edge_ids = [
        str(edge["id"])
        for edge in bundle.graph_payload.get("edges", [])
        if isinstance(edge, dict) and isinstance(edge.get("id"), str)
    ]
    if len(graph_node_ids) != len(set(graph_node_ids)):
        raise AssertionError("accepted OCR graph contains duplicate node IDs")
    if len(graph_edge_ids) != len(set(graph_edge_ids)):
        raise AssertionError("accepted OCR graph contains duplicate edge IDs")
    graph_payload_digest = hashlib.sha256(
        json.dumps(
            bundle.graph_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    if not isinstance(bundle.grouping_tree, dict):
        raise TypeError("accepted OCR result is missing its grouping tree")
    grouping_tree_digest = hashlib.sha256(
        json.dumps(
            bundle.grouping_tree,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    persisted_graph_node_ids = [
        node_id
        for node_id in graph_node_ids
        if knowledge_engine.persist.exists_node(node_id)
    ]
    (scratch / f"{mode}-lineage.json").write_text(
        json.dumps(
            {
                "lineage": lineage,
                "graph_node_ids": graph_node_ids,
                "graph_edge_ids": graph_edge_ids,
                "graph_payload_sha256": graph_payload_digest,
                "grouping_tree_sha256": grouping_tree_digest,
                "persisted_graph_node_ids": persisted_graph_node_ids,
                "persisted_graph_edge_ids": [
                    edge_id
                    for edge_id in graph_edge_ids
                    if knowledge_engine.persist.exists_edge(edge_id)
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
