"""Construction of the persisted OCR root workflow's step resolver."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.runtime import MappingStepResolver
from kogwistar.runtime.models import (
    RunSuccess,
    StepRunResult,
    WorkflowInvocationRequest,
)
from kogwistar.runtime.runtime import StepContext

from .design import (
    DEFAULT_OCR_GROUPING_WORKFLOW_ID,
    DEFAULT_OCR_PAGE_WORKFLOW_ID,
    DEFAULT_OCR_WORKFLOW_ID,
    DEFAULT_WORKFLOW_ID,
)
from .document_adapters import (
    attach_document_tree_metadata,
    suggest_cross_page_continuations,
)
from .document_grouping import GroupProposal
from .models import WorkflowIngestInput
from .ocr_pipeline import (
    OCRPageJSON,
    OCRPagePreparation,
    _finalize_ocr_workflow_artifacts,
)
from .ocr_runtime_common import document_tree_from_state as _tree
from .ocr_runtime_common import workflow_success as _success
from .ocr_runtime_lineage import recovery_generation as _recovery_generation
from .ocr_runtime_lineage import semantic_parse_generation as _semantic_parse_generation
from .ocr_runtime_page_state import (
    page_result_needs_recovery as _page_result_needs_recovery,
)
from .ocr_runtime_page_state import (
    page_workflow_initial_state as _page_workflow_initial_state,
)
from .ocr_runtime_page_state import reused_pages_from_trace as _reused_pages_from_trace
from .ocr_runtime_state import int_list as _int_list
from .ocr_runtime_state import json_pages as _json_pages
from .ocr_runtime_state import typed_child_state as _typed_child_state
from .ocr_runtime_state import typed_state as _typed_state
from .ocr_runtime_state import (
    validate_root_resume_identity as _validate_root_resume_identity,
)

if TYPE_CHECKING:
    from .providers import WorkflowProviderSettings


@dataclass
class OCRRootResolverArtifacts:
    prepared_page_payloads: list[OCRPageJSON] | None = None
    prepared_reused_pages: list[int] | None = None


@dataclass(frozen=True)
class OCRRootResolverBuild:
    step_resolver: MappingStepResolver
    artifacts: OCRRootResolverArtifacts


def build_ocr_root_step_resolver(
    *,
    workflow_engine: GraphKnowledgeEngine,
    conversation_engine: GraphKnowledgeEngine,
    knowledge_engine: GraphKnowledgeEngine | None,
    preparation: OCRPagePreparation,
    provider_settings: WorkflowProviderSettings,
    document_id: str,
    title: str,
    grouping_policy_version: str,
    source_namespace: str | None,
    source_namespace_identity: str,
    explicit_passes: Sequence[Sequence[GroupProposal]],
    resume_from_checkpoint: bool,
    page_resolver: MappingStepResolver,
    grouping_resolver: MappingStepResolver,
    semantic_resolver: MappingStepResolver,
) -> OCRRootResolverBuild:
    """Build root, child, and dispatch handlers for one OCR execution."""
    root_resolver = MappingStepResolver()
    prepared_state = OCRRootResolverArtifacts()
    root_resolver.set_state_schema(
        {
            "ocr_page_result": "u",
            "ocr_page_recovery_generation": "u",
            "workflow_input": "u",
            "tree": "u",
            "grouping_result": "u",
            "semantic_result": "u",
            "semantic_tree": "u",
            "export_bundle": "u",
            "ocr_root_state_version": "u",
            "ocr_root_lineage_id": "u",
            "source_fingerprint": "u",
            "source_namespace": "u",
            "ocr_configuration_fingerprints": "u",
            "parse_generation": "u",
            "checkpoint_revision": "u",
            "checkpoint_absent": "u",
            "grouping_passes": "u",
            "adaptive_grouping_round": "u",
            "adaptive_grouping_window": "u",
            "adaptive_grouping_history": "u",
            "adaptive_grouping_window_proposals": "u",
            "adaptive_grouping_complete": "u",
            **page_resolver._state_schema,
            **semantic_resolver._state_schema,
        }
    )

    @root_resolver.register("ocr_root_start")
    def _root_start(ctx: StepContext) -> StepRunResult:
        del ctx
        return _success("ocr_root_invoke_pages")

    @root_resolver.register("ocr_root_invoke_pages", is_nested=True)
    def _root_invoke_pages(ctx: StepContext) -> StepRunResult:
        state = _typed_state(ctx.state_view)
        recovery_generation = _recovery_generation(state)
        recovery_suffix = (
            f"|recovery-{recovery_generation}" if recovery_generation else ""
        )
        child_run_id = f"{ctx.run_id}|ocr-pages{recovery_suffix}"
        return RunSuccess(
            conversation_node_id=None,
            state_update=[],
            workflow_invocations=[
                WorkflowInvocationRequest(
                    workflow_id=DEFAULT_OCR_PAGE_WORKFLOW_ID,
                    initial_state={
                        **_page_workflow_initial_state(preparation.source_plan),
                        "document_id": preparation.page_context.document_id,
                        "ocr_configuration_fingerprints": list(
                            preparation.page_context.configuration_fingerprints
                        ),
                    },
                    result_state_key="ocr_page_result",
                    run_id=child_run_id,
                    invocation_key=(
                        "ocr-pages-v1"
                        if not recovery_generation
                        else f"ocr-pages-recovery-v{recovery_generation}"
                    ),
                    conversation_id=f"ocr:{document_id}",
                    turn_node_id=f"ocr:{document_id}:pages",
                )
            ],
            _route_next=["ocr_root_prepare_grouping_input"],
        )

    @root_resolver.register("ocr_root_prepare_grouping_input")
    def _root_prepare_grouping(ctx: StepContext) -> StepRunResult:
        state = _typed_state(ctx.state_view)
        page_result = _typed_child_state(
            state.get("ocr_page_result"),
            label="OCR page child",
        )
        if _page_result_needs_recovery(
            page_result=page_result,
            context=preparation.page_context,
            source_plan=preparation.source_plan,
        ):
            generation = _recovery_generation(state) + 1
            return _success(
                "ocr_root_invoke_pages",
                state_update=[("u", {"ocr_page_recovery_generation": generation})],
            )
        raw_pages = _json_pages(page_result, source_plan=preparation.source_plan)
        prepared_state.prepared_page_payloads = raw_pages
        completed_pages = [
            page_number
            for page_number, _image_path in preparation.source_plan.page_sources
        ]
        if "completed_pages" in page_result:
            legacy_completed = _int_list(page_result, "completed_pages")
            if legacy_completed:
                completed_pages = legacy_completed
        recovery_generation = _recovery_generation(state)
        recovery_suffix = (
            f"|recovery-{recovery_generation}" if recovery_generation else ""
        )
        page_run_id = f"{ctx.run_id}|ocr-pages{recovery_suffix}"
        reused_pages = _reused_pages_from_trace(conversation_engine, page_run_id)
        prepared_state.prepared_reused_pages = reused_pages
        if "reused_pages" in page_result:
            legacy_reused = _int_list(page_result, "reused_pages")
            if legacy_reused:
                reused_pages = legacy_reused
        artifacts = _finalize_ocr_workflow_artifacts(
            document_id=document_id,
            title=title,
            raw_pages=raw_pages,
            completed_pages=completed_pages,
            reused_pages=reused_pages,
            rendered_dir=preparation.source_plan.rendered_dir,
            legacy_dir=preparation.source_plan.legacy_dir,
            state_db_path=preparation.state_db_path,
            progress_path=preparation.progress_path,
            summary_path=preparation.summary_path,
            provider_settings=provider_settings,
            state_store=preparation.state_store,
            restore_grouping_checkpoint=False,
            persist_grouping_checkpoint=False,
            publish_artifacts=False,
            grouping_policy_version=grouping_policy_version,
        )
        if artifacts.document_tree is None:
            raise ValueError("OCR normalization produced no document tree")
        if not explicit_passes:
            passes = [suggest_cross_page_continuations(artifacts.document_tree)]
        else:
            passes = explicit_passes
        return _success(
            "ocr_root_invoke_grouping",
            state_update=[
                (
                    "u",
                    {
                        "workflow_input": artifacts.workflow_input.model_dump(
                            field_mode="backend", dump_format="json"
                        ),
                        "tree": artifacts.document_tree.model_dump(mode="json"),
                        "grouping_passes": [
                            [
                                proposal.model_dump(mode="json")
                                for proposal in current_pass
                            ]
                            for current_pass in passes
                        ],
                        "completed_pages": completed_pages,
                        "reused_pages": reused_pages,
                    },
                )
            ],
        )

    @root_resolver.register("ocr_root_invoke_grouping", is_nested=True)
    def _root_invoke_grouping(ctx: StepContext) -> StepRunResult:
        state = _typed_state(ctx.state_view)
        tree = _tree(state)
        existing_revision = preparation.state_store.document_tree_revision(
            document_id=document_id
        )
        recovery_generation = _recovery_generation(state)
        recovery_suffix = (
            f"|recovery-{recovery_generation}" if recovery_generation else ""
        )
        return RunSuccess(
            conversation_node_id=None,
            state_update=[],
            workflow_invocations=[
                WorkflowInvocationRequest(
                    workflow_id=DEFAULT_OCR_GROUPING_WORKFLOW_ID,
                    initial_state={
                        "document_id": document_id,
                        "tree": tree.model_dump(mode="json"),
                        "grouping_policy_version": grouping_policy_version,
                        "grouping_passes": state.get("grouping_passes", []),
                        "pass_index": 0,
                        "adaptive_grouping_round": 0,
                        "adaptive_grouping_window": 0,
                        "adaptive_grouping_history": [],
                        "adaptive_grouping_window_proposals": [],
                        "adaptive_grouping_complete": False,
                        "checkpoint_revision": existing_revision,
                        "checkpoint_absent": existing_revision is None,
                    },
                    result_state_key="grouping_result",
                    run_id=f"{ctx.run_id}|grouping{recovery_suffix}",
                    invocation_key=(
                        "grouping-v1"
                        if not recovery_generation
                        else f"grouping-recovery-v{recovery_generation}"
                    ),
                    conversation_id=f"ocr:{document_id}",
                    turn_node_id=f"ocr:{document_id}:grouping",
                )
            ],
            _route_next=["ocr_root_prepare_semantic_input"],
        )

    @root_resolver.register("ocr_root_prepare_semantic_input")
    def _root_prepare_semantic(ctx: StepContext) -> StepRunResult:
        state = _typed_state(ctx.state_view)
        grouping_result = _typed_child_state(
            state.get("grouping_result"),
            label="grouping child",
        )
        grouped_tree = _tree(grouping_result)
        workflow_input_payload = state.get("workflow_input")
        if not isinstance(workflow_input_payload, dict):
            raise TypeError("OCR root has no persisted semantic workflow input")
        workflow_input = WorkflowIngestInput.model_validate(workflow_input_payload)
        primary_collection = workflow_input.collections[0]
        workflow_input = workflow_input.model_copy(
            update={
                "collections": [
                    collection.model_copy(update={"source_namespace": source_namespace})
                    if collection.collection_id == primary_collection.collection_id
                    else collection
                    for collection in workflow_input.collections
                ]
            }
        )
        attach_document_tree_metadata(workflow_input, grouped_tree)
        return _success(
            "ocr_root_invoke_semantic",
            state_update=[
                (
                    "u",
                    {
                        "tree": grouped_tree.model_dump(mode="json"),
                        "workflow_input": workflow_input.model_dump(
                            field_mode="backend", dump_format="json"
                        ),
                    },
                )
            ],
        )

    @root_resolver.register("ocr_root_invoke_semantic", is_nested=True)
    def _root_invoke_semantic(ctx: StepContext) -> StepRunResult:
        state = _typed_state(ctx.state_view)
        input_payload = state.get("workflow_input")
        if not isinstance(input_payload, dict):
            raise TypeError("OCR root has no semantic workflow input")
        recovery_generation = _recovery_generation(state)
        parse_generation = state.get("parse_generation")
        expected_parse_generation = _semantic_parse_generation(recovery_generation)
        if parse_generation != expected_parse_generation:
            raise RuntimeError(
                "OCR root parse-generation identity mismatch; refusing to invoke "
                f"semantic parser with {parse_generation!r}, expected "
                f"{expected_parse_generation!r}"
            )
        recovery_suffix = (
            f"|recovery-{recovery_generation}" if recovery_generation else ""
        )
        return RunSuccess(
            conversation_node_id=None,
            state_update=[],
            workflow_invocations=[
                WorkflowInvocationRequest(
                    workflow_id=DEFAULT_WORKFLOW_ID,
                    initial_state={"input": input_payload},
                    result_state_key="semantic_result",
                    run_id=f"{ctx.run_id}|semantic{recovery_suffix}",
                    invocation_key=parse_generation,
                    conversation_id=f"ingest:{document_id}",
                    turn_node_id=f"ingest:{document_id}:turn",
                )
            ],
            _route_next=["ocr_root_finalize"],
        )

    @root_resolver.register("ocr_root_finalize")
    def _root_finalize(ctx: StepContext) -> StepRunResult:
        state = _typed_state(ctx.state_view)
        semantic_result = _typed_child_state(
            state.get("semantic_result"),
            label="semantic child",
        )
        bundle = semantic_result.get("export_bundle")
        if not isinstance(bundle, dict):
            raise TypeError("semantic child did not publish an export bundle")
        grouped_tree = _tree(state)
        exported_grouping_tree = bundle.get("grouping_tree")
        if not isinstance(exported_grouping_tree, dict):
            raise TypeError(
                "semantic child did not preserve the accepted grouping tree"
            )
        if exported_grouping_tree != grouped_tree.model_dump(mode="json"):
            raise ValueError(
                "semantic child grouping tree differs from the accepted OCR grouping"
            )
        semantic_tree = semantic_result.get("semantic_tree")
        if not isinstance(semantic_tree, dict):
            raise TypeError("semantic child did not publish a semantic tree")
        return _success(
            "ocr_root_end",
            state_update=[
                (
                    "u",
                    {
                        "export_bundle": bundle,
                        "semantic_tree": semantic_tree,
                    },
                )
            ],
        )

    @root_resolver.register("ocr_root_end")
    def _root_end(ctx: StepContext) -> StepRunResult:
        del ctx
        return _success(None)

    # Child workflow designs reuse operation names such as ``start`` and
    # ``end``. Dispatch by persisted workflow ID, not by a mutable global mode.
    root_ops = set(root_resolver.ops)
    child_ops = page_resolver.ops | grouping_resolver.ops | semantic_resolver.ops
    dispatch = MappingStepResolver()
    dispatch.set_state_schema(root_resolver._state_schema)

    for op in root_ops | child_ops:
        is_nested = op in {
            "ocr_root_invoke_pages",
            "ocr_root_invoke_grouping",
            "ocr_root_invoke_semantic",
        }

        @dispatch.register(op, is_nested=is_nested)
        def _dispatch(ctx: StepContext, operation: str = op) -> StepRunResult:
            if ctx.workflow_id == DEFAULT_OCR_WORKFLOW_ID:
                _validate_root_resume_identity(
                    _typed_state(ctx.state_view),
                    document_id=document_id,
                    source_fingerprint=preparation.source_plan.input_fingerprint,
                    source_namespace=source_namespace_identity,
                    configuration_fingerprints=preparation.page_context.configuration_fingerprints,
                    grouping_policy_version=grouping_policy_version,
                    parse_generation=_semantic_parse_generation(
                        _recovery_generation(_typed_state(ctx.state_view))
                    ),
                    require_match=resume_from_checkpoint,
                )
                return root_resolver.resolve(operation)(ctx)
            if ctx.workflow_id == DEFAULT_OCR_PAGE_WORKFLOW_ID:
                return page_resolver.resolve(operation)(ctx)
            if ctx.workflow_id == DEFAULT_OCR_GROUPING_WORKFLOW_ID:
                return grouping_resolver.resolve(operation)(ctx)
            if ctx.workflow_id == DEFAULT_WORKFLOW_ID:
                return semantic_resolver.resolve(operation)(ctx)
            raise ValueError(f"unsupported OCR workflow child {ctx.workflow_id!r}")

    return OCRRootResolverBuild(
        step_resolver=dispatch,
        artifacts=prepared_state,
    )

