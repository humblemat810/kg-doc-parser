"""Grounded summaries, summary caching, journaling, and recovery tests."""

from __future__ import annotations

from pathlib import Path

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
from kg_doc_parser.models import (
    OCRClusterResponse,
)
from kg_doc_parser.workflow_ingest import (
    EmbeddingProviderConfig,
    OCRImagePayload,
    ProviderEndpointConfig,
    WorkflowProviderSettings,
    prepare_ocr_workflow_input,
)
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
    GroundedChatSummaryGenerator,
    OCRQualityPolicy,
    _generate_tree_summaries,
    _summary_generator_identity,
)
from kg_doc_parser.workflow_ingest.ocr_state_store import (
    OCRWorkflowStateStore,
)
from kg_doc_parser.workflow_ingest.providers import SupportsStructuredOutput

pytestmark = [pytest.mark.workflow]

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
        embedding=EmbeddingProviderConfig(
            provider="fake", model="fake-embed", dimension=1
        ),
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
        ocr_runner=lambda image_path, page_number, provider_settings: (
            _fake_ocr_response(page_number, f"Summary page {page_number}")
        ),
        summary_policy=SummaryPolicy(
            eligible_kinds=("document",),
            minimum_source_items=2,
            minimum_source_chars=1,
            generation_fingerprint="summary-journal-test-v1",
        ),
        summary_generator=_generator,
    )

    assert artifacts.document_tree is not None
    root = next(
        node for node in artifacts.document_tree.nodes if node.node_id == "root"
    )
    assert root.summary is not None
    assert root.summary.status == "generated"
    assert (
        root.summary.input_fingerprint
        == artifacts.document_tree.summary_input_fingerprint("root")
    )
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
                parts=(
                    SourceSlice(item_id="page-1", start=5, end=8, separator_before=" "),
                ),
            ),
            DocumentNode(
                node_id="leaf-beta",
                kind="text",
                parts=(SourceSlice(item_id="page-1", start=8, end=12),),
            ),
            DocumentNode(
                node_id="leaf-two",
                kind="text",
                parts=(
                    SourceSlice(
                        item_id="page-1", start=12, end=15, separator_before=" "
                    ),
                ),
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
        generation_fingerprint="grounded-summary-slices-v1",
    )
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    updated = _generate_tree_summaries(
        tree,
        document_id=tree.document_id,
        state_store=store,
        policy=policy,
        generator=None,
        grounded_generator=grounded_generator,
        grouping_policy_version="grounded-summary-v1",
    )

    assert [slice_.text for slice_ in seen["section-alpha"]] == ["alpha", "one"]
    assert [(slice_.start, slice_.end) for slice_ in seen["section-alpha"]] == [
        (0, 5),
        (5, 8),
    ]
    assert [slice_.text for slice_ in seen["section-beta"]] == ["beta", "two"]
    assert [(slice_.start, slice_.end) for slice_ in seen["section-beta"]] == [
        (8, 12),
        (12, 15),
    ]
    assert all(
        node.summary is not None for node in updated.nodes if node.kind == "section"
    )

    second_seen: set[str] = set()

    def second_grounded_generator(
        *,
        tree: DocumentTree,
        node_id: str,
        evidence_slices: tuple[EvidenceSlice, ...],
        input_fingerprint: str,
    ) -> Summary:
        del tree, evidence_slices
        second_seen.add(node_id)
        return Summary(
            status="generated",
            text=f"second-summary:{node_id}",
            evidence_item_ids=("page-1",),
            provider_model="second-grounded-summary",
            prompt_revision="summary-v1",
            input_fingerprint=input_fingerprint,
        )

    without_summary = updated.model_copy(
        update={
            "tree_revision": tree.tree_revision,
            "nodes": tuple(
                node.model_copy(update={"summary": None}) for node in updated.nodes
            ),
        }
    )
    current_revision = store.document_tree_revision(document_id=tree.document_id)
    assert current_revision is not None
    second_updated = _generate_tree_summaries(
        without_summary,
        document_id=tree.document_id,
        state_store=store,
        policy=policy,
        generator=None,
        grounded_generator=second_grounded_generator,
        grouping_policy_version="grounded-summary-v1",
        expected_tree_revision=current_revision,
    )
    assert second_seen == {"section-alpha", "section-beta"}
    assert all(
        node.summary is not None
        and node.summary.provider_model == "second-grounded-summary"
        for node in second_updated.nodes
        if node.kind == "section"
    )


@pytest.mark.ci
def test_legacy_summary_generator_rejects_partial_source_ownership() -> None:
    scratch = _scratch("legacy_summary_partial_ownership")
    tree = DocumentTree(
        document_id="legacy-summary-partial-doc",
        source_revision="ocr-v1:legacy-summary-partial",
        tree_revision=1,
        root_id="root",
        source_items=(
            SourceItem(
                item_id="shared-text",
                kind="text",
                text="first section; second section",
                location=SourceLocation(page_index=0, accuracy="page"),
            ),
        ),
        nodes=(
            DocumentNode(
                node_id="section:first",
                kind="section",
                children=(
                    ChildRef(node_id="leaf:first-a"),
                    ChildRef(node_id="leaf:first-b", separator_before=" "),
                ),
            ),
            DocumentNode(
                node_id="section:second",
                kind="section",
                children=(
                    ChildRef(node_id="leaf:second-a"),
                    ChildRef(node_id="leaf:second-b", separator_before=" "),
                ),
            ),
            DocumentNode(
                node_id="leaf:first-a",
                kind="text",
                parts=(SourceSlice(item_id="shared-text", start=0, end=5),),
            ),
            DocumentNode(
                node_id="leaf:first-b",
                kind="text",
                parts=(SourceSlice(item_id="shared-text", start=5, end=14),),
            ),
            DocumentNode(
                node_id="leaf:second-a",
                kind="text",
                parts=(SourceSlice(item_id="shared-text", start=14, end=20),),
            ),
            DocumentNode(
                node_id="leaf:second-b",
                kind="text",
                parts=(SourceSlice(item_id="shared-text", start=20, end=29),),
            ),
            DocumentNode(
                node_id="root",
                kind="document",
                children=(
                    ChildRef(node_id="section:first"),
                    ChildRef(node_id="section:second", separator_before=" "),
                ),
            ),
        ),
    )
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    calls = 0

    def legacy_generator(
        *,
        tree: DocumentTree,
        node_id: str,
        source_items: tuple[SourceItem, ...],
        input_fingerprint: str,
    ) -> Summary:
        nonlocal calls
        del tree, node_id, source_items, input_fingerprint
        calls += 1
        raise AssertionError("partial text must not reach the legacy callback")

    with pytest.raises(ValueError, match="migrate to GroundedSummaryGenerator"):
        _generate_tree_summaries(
            tree,
            document_id=tree.document_id,
            state_store=store,
            policy=SummaryPolicy(
                eligible_kinds=("section",),
                minimum_source_items=1,
                minimum_source_chars=1,
                minimum_information_gain_chars=0,
                generation_fingerprint="legacy-summary-partial-v1",
            ),
            generator=legacy_generator,
            grouping_policy_version="legacy-summary-partial-v1",
        )

    assert calls == 0
    assert store.list_summary_attempts(document_id=tree.document_id) == []


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
                children=(
                    ChildRef(node_id="leaf:left"),
                    ChildRef(node_id="leaf:right"),
                ),
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
        generation_fingerprint="summary-cache-test-v1",
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
    root_summary = next(
        node.summary for node in generated.nodes if node.node_id == "root"
    )
    assert root_summary is not None
    assert (
        root_summary.evidence_slice_fingerprint
        == generated.summary_slice_fingerprint("root")
    )

    # Reopen the persisted tree's structure without its derived summary. The
    # same generation key must reuse the accepted journal row, not call the
    # provider again.
    without_summary = generated.model_copy(
        update={
            "tree_revision": tree.tree_revision,
            "nodes": tuple(
                node.model_copy(update={"summary": None}) for node in generated.nodes
            ),
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
    assert (
        next(node.summary for node in reused.nodes if node.node_id == "root")
        == root_summary
    )
    attempts = OCRWorkflowStateStore(
        scratch / "ocr-state.sqlite"
    ).list_summary_attempts(document_id=tree.document_id)
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
    assert (
        store.document_tree_revision(document_id=tree.document_id) == persisted_revision
    )
    attempts = store.list_summary_attempts(document_id=tree.document_id)
    assert [attempt["status"] for attempt in attempts] == ["pending", "generated"]


def test_grounded_summary_generator_identity_is_not_none_type() -> None:
    def grounded_generator(
        *,
        tree: DocumentTree,
        node_id: str,
        evidence_slices: tuple[EvidenceSlice, ...],
        input_fingerprint: str,
    ) -> Summary:
        del tree, node_id, evidence_slices, input_fingerprint
        return Summary(
            status="generated",
            text="unused",
            evidence_item_ids=(),
            provider_model="test",
            prompt_revision="test",
            input_fingerprint="test",
        )

    identity = _summary_generator_identity(grounded_generator)

    assert identity.endswith(".grounded_generator")
    assert identity != "builtins.NoneType"


def test_custom_summary_generator_requires_stable_cache_identity() -> None:
    def generator(
        *,
        tree: DocumentTree,
        node_id: str,
        source_items: tuple[SourceItem, ...],
        input_fingerprint: str,
    ) -> Summary:
        del tree, node_id, source_items, input_fingerprint
        raise AssertionError("identity validation must run before generation")

    with pytest.raises(ValueError, match="configuration_fingerprint"):
        ocr_pipeline_module._effective_summary_policy(SummaryPolicy(), generator)

    explicit_policy = SummaryPolicy(generation_fingerprint="custom-v1")
    effective = ocr_pipeline_module._effective_summary_policy(
        explicit_policy, generator
    )
    assert effective.generation_fingerprint is not None
    assert len(effective.generation_fingerprint) == 64
    assert effective.generation_fingerprint != "custom-v1"


class _FakeGroundedSummaryChat:
    def __init__(self) -> None:
        self.messages: list[object] | None = None

    def with_structured_output(
        self,
        schema: type[Summary],
        *,
        include_raw: bool,
        **kwargs: object,
    ) -> SupportsStructuredOutput:
        assert schema is Summary
        assert include_raw is True
        del kwargs

        class _Response:
            def __init__(self, chat: _FakeGroundedSummaryChat) -> None:
                self.chat = chat

            def invoke(self, messages: list[object]) -> dict[str, object]:
                self.chat.messages = messages
                return {
                    "parsed": Summary(
                        status="generated",
                        text="alpha beta",
                        evidence_item_ids=("page-1",),
                        provider_model="provider-claimed",
                        prompt_revision="provider-claimed",
                        input_fingerprint="provider-claimed",
                    )
                }

        return _Response(self)


@pytest.mark.ci
def test_production_grounded_summary_uses_exact_evidence_and_parser_identity() -> None:
    settings = WorkflowProviderSettings(
        parser=ProviderEndpointConfig(provider="openai", model="test-model")
    )
    chat = _FakeGroundedSummaryChat()
    generator = GroundedChatSummaryGenerator(
        provider_settings=settings, chat_model=chat
    )
    evidence = (
        EvidenceSlice(
            item_id="page-1",
            kind="text",
            location=SourceLocation(page_index=0, accuracy="page"),
            text="alpha",
            start=0,
            end=5,
        ),
        EvidenceSlice(
            item_id="page-1",
            kind="text",
            location=SourceLocation(page_index=0, accuracy="page"),
            text="beta",
            start=5,
            end=9,
        ),
    )
    tree = DocumentTree(
        document_id="production-summary-doc",
        source_revision="ocr-v1:production-summary",
        tree_revision=1,
        root_id="root",
        source_items=(
            SourceItem(
                item_id="page-1",
                kind="text",
                text="alphabeta",
                location=SourceLocation(page_index=0, accuracy="page"),
            ),
        ),
        nodes=(
            DocumentNode(
                node_id="root",
                kind="document",
                parts=(SourceSlice(item_id="page-1", start=0, end=9),),
            ),
        ),
    )

    summary = generator(
        tree=tree,
        node_id="root",
        evidence_slices=evidence,
        input_fingerprint="evidence-fingerprint",
    )

    assert chat.messages is not None
    human_message = chat.messages[1]
    assert "alpha" in human_message.content
    assert "beta" in human_message.content
    assert summary.provider_model == "openai:test-model"
    assert summary.prompt_revision == "grounded-summary-v1"
    assert summary.input_fingerprint == "evidence-fingerprint"


@pytest.mark.ci
def test_production_grounded_summary_identity_includes_parser_config() -> None:
    first = GroundedChatSummaryGenerator(
        provider_settings=WorkflowProviderSettings(
            parser=ProviderEndpointConfig(
                provider="openai",
                model="model-a",
                base_url="https://router-a.example/v1",
                reasoning_effort="none",
            )
        ),
        chat_model=_FakeGroundedSummaryChat(),
    )
    second = GroundedChatSummaryGenerator(
        provider_settings=WorkflowProviderSettings(
            parser=ProviderEndpointConfig(
                provider="openai",
                model="model-a",
                base_url="https://router-b.example/v1",
                reasoning_effort="none",
            )
        ),
        chat_model=_FakeGroundedSummaryChat(),
    )
    third = GroundedChatSummaryGenerator(
        provider_settings=WorkflowProviderSettings(
            parser=ProviderEndpointConfig(
                provider="openai",
                model="model-a",
                base_url="https://router-a.example/v1",
                reasoning_effort="high",
            )
        ),
        chat_model=_FakeGroundedSummaryChat(),
    )

    first_identity = _summary_generator_identity(first)
    second_identity = _summary_generator_identity(second)
    third_identity = _summary_generator_identity(third)

    assert first_identity != second_identity
    assert first_identity != third_identity
    assert first_identity.startswith(
        "kg_doc_parser.workflow_ingest.ocr_generators.GroundedChatSummaryGenerator:"
    )


def test_duplicate_accepted_summary_delivery_is_idempotent() -> None:
    scratch = _scratch("ocr_summary_duplicate_delivery")
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    summary = Summary(
        status="generated",
        text="stable summary",
        evidence_item_ids=("item-a",),
        provider_model="fake-summary",
        prompt_revision="summary-v1",
        input_fingerprint="evidence-fingerprint",
    )

    first_index = store.record_summary_attempt(
        document_id="duplicate-summary-doc",
        node_id="root",
        input_fingerprint="generation-fingerprint",
        status="generated",
        summary=summary,
    )
    second_index = store.record_summary_attempt(
        document_id="duplicate-summary-doc",
        node_id="root",
        input_fingerprint="generation-fingerprint",
        status="generated",
        summary=summary,
    )

    assert second_index == first_index
    attempts = store.list_summary_attempts(document_id="duplicate-summary-doc")
    assert len(attempts) == 1
    assert attempts[0]["status"] == "generated"


@pytest.mark.ci
def test_summary_tree_uses_existing_accepted_journal_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A concurrent duplicate cannot make the tree disagree with the journal."""
    scratch = _scratch("ocr_summary_accepted_journal_wins")
    tree = DocumentTree(
        document_id="summary-accepted-journal-doc",
        source_revision="ocr-v1:summary-accepted-journal",
        tree_revision=1,
        root_id="root",
        source_items=(
            SourceItem(
                item_id="item-a",
                kind="text",
                text="summary source",
                location=SourceLocation(page_index=0, accuracy="page"),
            ),
        ),
        nodes=(
            DocumentNode(
                node_id="leaf:item-a",
                kind="text",
                parts=(SourceSlice(item_id="item-a", start=0, end=14),),
            ),
            DocumentNode(
                node_id="root",
                kind="document",
                children=(ChildRef(node_id="leaf:item-a"),),
            ),
        ),
    )
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    accepted_tree_revision = store.save_document_tree(
        document_id=tree.document_id,
        tree=tree,
        grouping_policy_version="summary-journal-v1",
        require_no_existing_tree=True,
    )
    canonical = Summary(
        status="generated",
        text="first accepted result",
        evidence_item_ids=("item-a",),
        provider_model="fake-summary",
        prompt_revision="summary-v1",
        input_fingerprint=tree.summary_input_fingerprint("root"),
        generation_fingerprint="journal-summary-v1",
        evidence_slice_fingerprint=tree.summary_slice_fingerprint("root", ("item-a",)),
    )
    original_record = store.record_summary_attempt
    inserted = False

    def _record_with_competing_acceptance(*args: object, **kwargs: object) -> int:
        nonlocal inserted
        if kwargs.get("status") == "generated" and not inserted:
            inserted = True
            original_record(
                document_id=str(kwargs["document_id"]),
                node_id=str(kwargs["node_id"]),
                input_fingerprint=str(kwargs["input_fingerprint"]),
                status="generated",
                summary=canonical,
            )
        return original_record(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        store, "record_summary_attempt", _record_with_competing_acceptance
    )

    def _generator(
        *,
        tree: DocumentTree,
        node_id: str,
        source_items: tuple[SourceItem, ...],
        input_fingerprint: str,
    ) -> Summary:
        del tree, node_id, source_items
        return Summary(
            status="generated",
            text="late duplicate result",
            evidence_item_ids=("item-a",),
            provider_model="fake-summary",
            prompt_revision="summary-v1",
            input_fingerprint=input_fingerprint,
        )

    result = _generate_tree_summaries(
        tree,
        document_id=tree.document_id,
        state_store=store,
        policy=SummaryPolicy(
            eligible_kinds=("document",),
            minimum_source_items=1,
            minimum_source_chars=1,
            minimum_information_gain_chars=0,
            generation_fingerprint="journal-summary-v1",
        ),
        generator=_generator,
        grouping_policy_version="summary-journal-v1",
        expected_tree_revision=accepted_tree_revision,
    )

    root = next(node for node in result.nodes if node.node_id == result.root_id)
    accepted_attempt = next(
        attempt
        for attempt in store.list_summary_attempts(document_id=tree.document_id)
        if attempt["status"] == "generated"
    )
    journal_summary = store.load_completed_summary(
        document_id=tree.document_id,
        node_id="root",
        input_fingerprint=str(accepted_attempt["input_fingerprint"]),
    )
    assert journal_summary == canonical
    assert root.summary == journal_summary, [
        (attempt["status"], attempt["error_message"])
        for attempt in store.list_summary_attempts(document_id=tree.document_id)
    ]


@pytest.mark.ci
def test_summary_journal_commit_is_recoverable_before_runtime_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An accepted journal entry repairs the tree after process interruption."""
    scratch = _scratch("ocr_summary_journal_commit_crash")
    tree = DocumentTree(
        document_id="summary-journal-commit-crash-doc",
        source_revision="ocr-v1:summary-journal-commit-crash",
        tree_revision=1,
        root_id="root",
        source_items=(
            SourceItem(
                item_id="item-a",
                kind="text",
                text="summary source",
                location=SourceLocation(page_index=0, accuracy="page"),
            ),
        ),
        nodes=(
            DocumentNode(
                node_id="leaf:item-a",
                kind="text",
                parts=(SourceSlice(item_id="item-a", start=0, end=14),),
            ),
            DocumentNode(
                node_id="root",
                kind="document",
                children=(ChildRef(node_id="leaf:item-a"),),
            ),
        ),
    )
    store = OCRWorkflowStateStore(scratch / "ocr-state.sqlite")
    initial_revision = store.save_document_tree(
        document_id=tree.document_id,
        tree=tree,
        grouping_policy_version="summary-journal-v1",
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
        del tree, node_id, source_items
        calls["count"] += 1
        return Summary(
            status="generated",
            text="durable summary",
            evidence_item_ids=("item-a",),
            provider_model="fake-summary",
            prompt_revision="summary-v1",
            input_fingerprint=input_fingerprint,
        )

    class _SimulatedProcessCrash(BaseException):
        pass

    original_save = store.save_document_tree
    crashed = False

    def _save_then_crash(*args: object, **kwargs: object) -> str:
        nonlocal crashed
        candidate = kwargs.get("tree")
        if (
            isinstance(candidate, DocumentTree)
            and any(node.summary is not None for node in candidate.nodes)
            and not crashed
        ):
            crashed = True
            raise _SimulatedProcessCrash("crash before derived tree checkpoint")
        return original_save(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(store, "save_document_tree", _save_then_crash)
    with pytest.raises(_SimulatedProcessCrash, match="before derived tree checkpoint"):
        _generate_tree_summaries(
            tree,
            document_id=tree.document_id,
            state_store=store,
            policy=SummaryPolicy(
                eligible_kinds=("document",),
                minimum_source_items=1,
                minimum_source_chars=1,
                minimum_information_gain_chars=0,
                generation_fingerprint="journal-summary-v1",
            ),
            generator=_generator,
            grouping_policy_version="summary-journal-v1",
            expected_tree_revision=initial_revision,
        )

    assert calls["count"] == 1
    accepted_attempt = next(
        attempt
        for attempt in store.list_summary_attempts(document_id=tree.document_id)
        if attempt["status"] == "generated"
    )
    accepted = store.load_completed_summary(
        document_id=tree.document_id,
        node_id="root",
        input_fingerprint=str(accepted_attempt["input_fingerprint"]),
    )
    assert accepted is not None
    assert (
        store.document_tree_revision(document_id=tree.document_id) == initial_revision
    )

    monkeypatch.setattr(store, "save_document_tree", original_save)
    resumed = _generate_tree_summaries(
        tree,
        document_id=tree.document_id,
        state_store=store,
        policy=SummaryPolicy(
            eligible_kinds=("document",),
            minimum_source_items=1,
            minimum_source_chars=1,
            minimum_information_gain_chars=0,
            generation_fingerprint="journal-summary-v1",
        ),
        generator=_generator,
        grouping_policy_version="summary-journal-v1",
        expected_tree_revision=initial_revision,
    )

    root = next(node for node in resumed.nodes if node.node_id == resumed.root_id)
    assert root.summary == accepted
    assert calls["count"] == 1


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

    class _Generator:
        def __init__(self, configuration_fingerprint: str) -> None:
            self.configuration_fingerprint = configuration_fingerprint

        def __call__(
            self,
            *,
            tree: DocumentTree,
            node_id: str,
            source_items: tuple[SourceItem, ...],
            input_fingerprint: str,
        ) -> Summary:
            del tree, node_id
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
        ),
        generator=_Generator("provider-a:model-a:prompt-a"),
        grouping_policy_version="summary-cache-v1",
        expected_tree_revision=initial_revision,
    )
    first_summary = next(node.summary for node in first.nodes if node.node_id == "root")
    assert first_summary is not None
    assert first_summary.generation_fingerprint is not None
    current_revision = store.document_tree_revision(document_id=tree.document_id)
    assert current_revision is not None
    second = _generate_tree_summaries(
        first,
        document_id=tree.document_id,
        state_store=store,
        policy=SummaryPolicy(
            eligible_kinds=("document",),
            minimum_source_items=1,
            minimum_source_chars=1,
            minimum_information_gain_chars=0,
        ),
        generator=_Generator("provider-b:model-b:prompt-b"),
        grouping_policy_version="summary-cache-v1",
        expected_tree_revision=current_revision,
    )

    assert calls["count"] == 2
    second_summary = next(
        node.summary for node in second.nodes if node.node_id == "root"
    )
    assert second_summary is not None
    assert second_summary.text == "summary:2"
    assert second_summary.generation_fingerprint is not None
    assert second_summary.generation_fingerprint != first_summary.generation_fingerprint


@pytest.mark.ci
def test_summary_failure_is_journaled_without_failing_ocr_ingestion() -> None:
    scratch = _scratch("ocr_summary_failure")
    image_paths = [scratch / "images" / f"page_{page}.png" for page in (1, 2)]
    for page, image_path in enumerate(image_paths, start=1):
        _draw_test_image(image_path, lines=[f"Summary failure page {page}"])
    settings = WorkflowProviderSettings(
        ocr=ProviderEndpointConfig(provider="fake", model="fake-ocr"),
        embedding=EmbeddingProviderConfig(
            provider="fake", model="fake-embed", dimension=1
        ),
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
        ocr_runner=lambda image_path, page_number, provider_settings: (
            _fake_ocr_response(page_number, f"Summary failure page {page_number}")
        ),
        quality_policy=OCRQualityPolicy(mode="permissive"),
        summary_policy=SummaryPolicy(
            eligible_kinds=("document",),
            minimum_source_items=1,
            minimum_source_chars=1,
            generation_fingerprint="summary-failure-test-v1",
        ),
        summary_generator=_generator,
    )

    assert artifacts.document_tree is not None
    assert (
        next(
            node for node in artifacts.document_tree.nodes if node.node_id == "root"
        ).summary
        is None
    )
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
        embedding=EmbeddingProviderConfig(
            provider="fake", model="fake-embed", dimension=1
        ),
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
        generation_fingerprint="summary-retry-test-v1",
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
        quality_policy=OCRQualityPolicy(mode="permissive"),
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
        quality_policy=OCRQualityPolicy(mode="permissive"),
        summary_policy=policy,
        summary_generator=_successful_generator,
    )
    assert retried.document_tree is not None
    root = next(
        node
        for node in retried.document_tree.nodes
        if node.node_id == retried.document_tree.root_id
    )
    assert root.summary is not None
    assert root.summary.text == "retry:root"
    assert ocr_calls["count"] == 2
    attempts = [
        attempt
        for attempt in OCRWorkflowStateStore(
            retried.state_db_path
        ).list_summary_attempts(document_id="ocr-summary-retry-doc")
        if attempt["node_id"] == "root"
    ]
    statuses = [attempt["status"] for attempt in attempts]
    assert statuses.count("pending") == 2
    assert statuses.count("failed") == 1
    assert statuses.count("generated") == 1

