from __future__ import annotations

from types import SimpleNamespace

import kg_doc_parser.workflow_ingest.strategy as strategy_module
import pytest
from kg_doc_parser.workflow_ingest.cli import _provider_settings_from_args, build_parser
from kg_doc_parser.workflow_ingest.models import WorkflowIngestInput
from kg_doc_parser.workflow_ingest.providers import (
    ProviderEndpointConfig,
    WorkflowProviderSettings,
)
from kg_doc_parser.workflow_ingest.service import _workflow_predicates
from kg_doc_parser.workflow_ingest.strategy import (
    HARD_CODED_STRATEGY_PRIORITY,
    ParseStrategyAssessment,
    ParseStrategyTriage,
    build_llm_strategy_triage,
    select_parse_strategy,
)


def _assessments() -> list[ParseStrategyAssessment]:
    return [
        ParseStrategyAssessment(
            strategy=strategy,
            advantages=[f"advantage of {strategy}"],
            limitations=[f"limitation of {strategy}"],
        )
        for strategy in HARD_CODED_STRATEGY_PRIORITY
    ]


@pytest.mark.ci
def test_auto_strategy_without_triage_uses_deterministic_cascade_head() -> None:
    decision = select_parse_strategy(
        requested="auto",
        context={"layer_depth": 1},
        triage_enabled=False,
    )

    assert decision.selected_strategy == "layer_excerpt"
    assert decision.source == "hardcoded_fallback"
    assert decision.fallback_order == HARD_CODED_STRATEGY_PRIORITY


@pytest.mark.ci
def test_explicit_strategy_bypasses_triage_for_the_current_layer() -> None:
    decision = select_parse_strategy(
        requested="page_index",
        context={"layer_depth": 2},
        triage_enabled=True,
        triage_fn=lambda _context: (_ for _ in ()).throw(AssertionError("must not be called")),
    )

    assert decision.selected_strategy == "page_index"
    assert decision.source == "config"


@pytest.mark.ci
def test_triage_choice_is_layer_local_and_low_confidence_falls_back() -> None:
    selected = select_parse_strategy(
        requested="auto",
        context={"layer_depth": 3},
        triage_enabled=True,
        triage_fn=lambda _context: ParseStrategyTriage(
            selected_strategy="layer_boundary",
            confidence=0.91,
            rationale="this parent has clear boundaries",
            assessments=_assessments(),
        ),
    )
    fallback = select_parse_strategy(
        requested="auto",
        context={"layer_depth": 4},
        triage_enabled=True,
        triage_fn=lambda _context: ParseStrategyTriage(
            selected_strategy="page_index",
            confidence=0.20,
            assessments=_assessments(),
        ),
    )

    assert selected.selected_strategy == "layer_boundary"
    assert selected.source == "llm_triage"
    assert fallback.selected_strategy == "layer_excerpt"
    assert fallback.source == "llm_triage_fallback"
    assert [item.strategy for item in selected.assessments] == list(HARD_CODED_STRATEGY_PRIORITY)


@pytest.mark.ci
def test_triage_provider_failure_falls_back_without_retrying_forever() -> None:
    decision = select_parse_strategy(
        requested="auto",
        context={"layer_depth": 1},
        triage_enabled=True,
        triage_fn=lambda _context: (_ for _ in ()).throw(TimeoutError("provider stalled")),
    )

    assert decision.selected_strategy == "layer_excerpt"
    assert decision.source == "llm_triage_fallback"
    assert "TimeoutError" in decision.rationale


@pytest.mark.ci
def test_provider_triage_emits_complete_success_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Structured:
        def invoke(self, messages):
            return {
                "parsed": ParseStrategyTriage(
                    selected_strategy="layer_excerpt",
                    confidence=0.9,
                    rationale="bounded evidence is available",
                    assessments=_assessments(),
                )
            }

    class _Chat:
        def with_structured_output(self, schema, include_raw=True):
            assert schema is ParseStrategyTriage
            return _Structured()

    monkeypatch.setattr(strategy_module, "build_chat_model_for_role", lambda *args, **kwargs: _Chat())
    diagnostics: list[dict[str, object]] = []
    triage = build_llm_strategy_triage(
        WorkflowProviderSettings(
            parser=ProviderEndpointConfig(provider="fake", model="test-model"),
        ),
        diagnostics_sink=diagnostics.append,
    )

    result = triage({"layer_depth": 2})

    assert result.selected_strategy == "layer_excerpt"
    assert len(diagnostics) == 1
    assert diagnostics[0]["operation"] == "parse_strategy_triage"
    assert diagnostics[0]["call_role"] == "triage"
    assert diagnostics[0]["strategy"] == "triage"
    assert diagnostics[0]["attempt_index"] == 1
    assert diagnostics[0]["success"] is True
    assert isinstance(diagnostics[0]["elapsed_ms"], int)


@pytest.mark.ci
def test_provider_triage_emits_parse_failure_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Structured:
        def invoke(self, messages):
            return {"parsed": None, "parsing_error": "missing structured triage"}

    class _Chat:
        def with_structured_output(self, schema, include_raw=True):
            return _Structured()

    monkeypatch.setattr(strategy_module, "build_chat_model_for_role", lambda *args, **kwargs: _Chat())
    diagnostics: list[dict[str, object]] = []
    triage = build_llm_strategy_triage(
        WorkflowProviderSettings(
            parser=ProviderEndpointConfig(provider="fake", model="test-model"),
        ),
        diagnostics_sink=diagnostics.append,
    )

    with pytest.raises(ValueError, match="strategy triage parsing failed"):
        triage({"layer_depth": 2})

    assert len(diagnostics) == 1
    assert diagnostics[0]["success"] is False
    assert diagnostics[0]["failure_type"] == "structured_output_parse_failure"
    assert diagnostics[0]["error_type"] == "ValueError"


@pytest.mark.ci
def test_explicit_strategy_order_controls_the_fallback_cascade() -> None:
    order = ("layer_boundary", "layer_excerpt", "page_index")
    decision = select_parse_strategy(
        requested="auto",
        context={"layer_depth": 1},
        triage_enabled=False,
        strategy_order=order,
    )
    assert decision.selected_strategy == "layer_boundary"
    assert decision.fallback_order == order

    fallback = select_parse_strategy(
        requested="auto",
        context={"layer_depth": 1},
        triage_enabled=False,
        strategy_order=order,
        disabled_strategies={"layer_boundary"},
    )
    assert fallback.selected_strategy == "layer_excerpt"
    assert fallback.fallback_order == order


@pytest.mark.ci
def test_failed_strategy_is_disabled_and_exhaustion_is_explicit() -> None:
    order = ("layer_boundary", "layer_excerpt", "page_index")
    for disabled, expected in [
        ({"layer_boundary"}, "layer_excerpt"),
        ({"layer_boundary", "layer_excerpt"}, "page_index"),
    ]:
        decision = select_parse_strategy(
            requested="auto",
            context={},
            triage_enabled=False,
            strategy_order=order,
            disabled_strategies=disabled,
        )
        assert decision.selected_strategy == expected

    with pytest.raises(ValueError, match="all parser strategies are disabled"):
        select_parse_strategy(
            requested="auto",
            context={},
            triage_enabled=False,
            strategy_order=order,
            disabled_strategies=set(order),
        )


@pytest.mark.ci
def test_fake_layer_payloads_select_success_fallback_and_terminal_routes() -> None:
    predicates = _workflow_predicates()

    def edge(target: str) -> SimpleNamespace:
        return SimpleNamespace(dst=f"wf|parser|{target}")

    successful = {
        "current_layer_context": {
            "metadata": {
                "parse_strategy": "layer_boundary",
                "disabled_strategies": [],
            }
        },
        "current_layer_result": {"satisfied": True},
        "current_layer_review": {
            "coverage_ok": True,
            "metadata": {},
            "overlap_conflicts": [],
            "coverage_gap_notes": [],
            "duplicate_child_notes": [],
        },
    }
    assert predicates["parse_strategy_layer_boundary"](edge("layer_boundary_method"), successful, None)
    assert predicates["layer_satisfied"](edge("repair_layer_pointers"), successful, None)

    failed_layer = {
        "current_layer_context": {
            "metadata": {
                "parse_strategy": "layer_boundary",
                "disabled_strategies": ["layer_boundary"],
            }
        },
        "current_layer_result": {"satisfied": False},
        "current_layer_review": {
            "coverage_ok": False,
            "metadata": {},
            "overlap_conflicts": [],
            "coverage_gap_notes": ["missing source interval"],
            "duplicate_child_notes": [],
        },
    }
    assert predicates["strategy_failed_with_remaining"](edge("triage_parse_strategy"), failed_layer, None)

    failed_once = {
        "current_layer_context": {
            "metadata": {
                "parse_strategy": "layer_boundary",
                "disabled_strategies": ["layer_boundary"],
            }
        }
    }
    assert predicates["strategy_failed_with_remaining"](edge("triage_parse_strategy"), failed_once, None)
    assert not predicates["all_strategies_exhausted"](edge("parse_failure"), failed_once, None)

    exhausted = {
        "current_layer_context": {
            "metadata": {
                "parse_strategy": "page_index",
                "disabled_strategies": ["layer_boundary", "layer_excerpt", "page_index"],
            }
        }
    }
    assert predicates["all_strategies_exhausted"](edge("parse_failure"), exhausted, None)
    assert not predicates["strategy_failed_with_remaining"](edge("triage_parse_strategy"), exhausted, None)


@pytest.mark.ci
def test_triage_requires_bounded_pros_and_cons_for_every_strategy() -> None:
    with pytest.raises(ValueError, match="assess all allowed strategies"):
        ParseStrategyTriage(
            selected_strategy="layer_excerpt",
            confidence=0.9,
            assessments=[
                ParseStrategyAssessment(
                    strategy="layer_excerpt",
                    advantages=["keeps exact source text"],
                    limitations=["may miss structural boundaries"],
                )
            ],
        )

    with pytest.raises(ValueError, match="advantages and limitations"):
        ParseStrategyTriage(
            selected_strategy="layer_excerpt",
            confidence=0.9,
            assessments=[
                ParseStrategyAssessment(
                    strategy=strategy,
                    advantages=["bounded"],
                    limitations=["bounded" if strategy != "layer_boundary" else ""],
                )
                for strategy in HARD_CODED_STRATEGY_PRIORITY
            ],
        )


@pytest.mark.ci
def test_workflow_input_supports_request_level_strategy_overrides() -> None:
    default_input = WorkflowIngestInput.from_text(document_id="doc", text="content")
    assert default_input.parse_strategy is None
    assert default_input.triage_enabled is None

    request_input = default_input.model_copy(
        update={
            "parse_strategy": "page_index",
            "triage_enabled": False,
            "page_index_summary_enabled": False,
            "page_index_hierarchical_summary_enabled": True,
            "parse_strategy_order": ["layer_boundary", "layer_excerpt", "page_index"],
        }
    )
    assert request_input.parse_strategy == "page_index"
    assert request_input.triage_enabled is False
    assert request_input.page_index_summary_enabled is False
    assert request_input.page_index_hierarchical_summary_enabled is True
    assert request_input.parse_strategy_order == ["layer_boundary", "layer_excerpt", "page_index"]
    backend_payload = request_input.model_dump(field_mode="backend", dump_format="json")
    assert backend_payload["parse_strategy"] == "page_index"
    assert backend_payload["page_index_summary_enabled"] is False
    assert backend_payload["page_index_hierarchical_summary_enabled"] is True
    assert backend_payload["parse_strategy_order"] == ["layer_boundary", "layer_excerpt", "page_index"]


@pytest.mark.ci
def test_workflow_input_rejects_incomplete_strategy_order() -> None:
    with pytest.raises(ValueError, match="parse_strategy_order"):
        WorkflowIngestInput.model_validate(
            WorkflowIngestInput.from_text(document_id="doc", text="content").model_dump()
            | {"parse_strategy_order": ["layer_boundary", "layer_excerpt"]}
        )


@pytest.mark.ci
def test_provider_settings_rejects_invalid_strategy_order() -> None:
    with pytest.raises(ValueError, match="parse_strategy_order"):
        WorkflowProviderSettings(parse_strategy_order=("page_index", "page_index", "layer_excerpt"))


@pytest.mark.ci
def test_strategy_selection_failure_predicate_routes_to_failure() -> None:
    predicates = _workflow_predicates()
    assert predicates["strategy_selection_failed"](
        SimpleNamespace(dst="wf|parser|parse_failure"),
        {"strategy_selection_error": "invalid strategy order"},
        None,
    )


@pytest.mark.ci
def test_cli_provider_overrides_support_strategy_and_triage_per_parse_call() -> None:
    args = build_parser().parse_args(
        [
            "page-index",
            "document.md",
            "--output-dir",
            "out",
            "--parse-strategy",
            "page_index",
            "--no-triage-enabled",
            "--no-page-index-summary-enabled",
            "--page-index-hierarchical-summary-enabled",
        ]
    )
    settings = _provider_settings_from_args(args)
    assert settings is not None
    assert settings.parse_strategy == "page_index"
    assert settings.triage_enabled is False
    assert settings.page_index_summary_enabled is False
    assert settings.page_index_hierarchical_summary_enabled is True


@pytest.mark.ci
def test_page_index_summary_defaults_on_and_reads_process_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KG_DOC_PARSER_PAGE_INDEX_SUMMARY_ENABLED", raising=False)
    assert WorkflowProviderSettings.from_env().page_index_summary_enabled is True

    monkeypatch.setenv("KG_DOC_PARSER_PAGE_INDEX_SUMMARY_ENABLED", "0")
    assert WorkflowProviderSettings.from_env().page_index_summary_enabled is False

    monkeypatch.delenv("KG_DOC_PARSER_PAGE_INDEX_HIERARCHICAL_SUMMARY_ENABLED", raising=False)
    assert WorkflowProviderSettings.from_env().page_index_hierarchical_summary_enabled is False

    monkeypatch.setenv("KG_DOC_PARSER_PAGE_INDEX_HIERARCHICAL_SUMMARY_ENABLED", "1")
    assert WorkflowProviderSettings.from_env().page_index_hierarchical_summary_enabled is True
