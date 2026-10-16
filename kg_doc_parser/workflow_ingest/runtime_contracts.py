"""Shared typed contracts for workflow runtime collaborators."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from .cache import WorkflowLLMCallCache
    from .clients import CanonicalGraphPersistenceClient
    from .parser_core import ParseSemanticFn, ProposeLayerFn, ReviewLayerFn, SplitStrategy
    from .probe import WorkflowProbe
    from .providers import ProviderDiagnosticsSink, WorkflowProviderSettings
    from .strategy import StrategyTriageFn


class WorkflowRuntimeDeps(TypedDict, total=False):
    """Optional typed collaborators and policy for persisted workflow steps.

    The checkpoint state remains JSON-like and is decoded separately. This
    contract is only for executable collaborators and bounded policy values.
    """

    probe: WorkflowProbe | None
    parse_semantic_fn: ParseSemanticFn
    propose_layer_fn: ProposeLayerFn
    review_layer_fn: ReviewLayerFn
    llm_cache: WorkflowLLMCallCache
    graph_persistence_client: CanonicalGraphPersistenceClient
    persistence_mode: str
    kg_authority: str
    max_depth: int
    allow_review: bool
    split_strategy: SplitStrategy
    fallback_split_strategy: SplitStrategy
    max_review_retries: int
    layer_frontier_batch_size: int
    coverage_threshold: float
    provider_settings: WorkflowProviderSettings
    triage_strategy_fn: StrategyTriageFn
    provider_diagnostics_sink: ProviderDiagnosticsSink
    require_grouping_tree: bool
