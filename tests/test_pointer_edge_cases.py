from __future__ import annotations

import pytest
from pydantic import ValidationError

import kg_doc_parser.semantic_document_splitting_layerwise_edits as legacy_layerwise
from kg_doc_parser.workflow_ingest.handlers import _correct_parser_pointer
from kg_doc_parser.workflow_ingest.layerwise_llm import _hydrate_llm_pointer_payload
from kg_doc_parser.workflow_ingest.page_index import parse_page_index_layer
from kg_doc_parser.workflow_ingest.semantics import (
    HydratedTextPointer,
    SemanticNode,
    compute_pointer_coverage,
    compute_terminal_content_coverage,
    correct_and_validate_pointer,
    hydrate_pointer_from_offsets,
    pointer_source_validation_error,
)


def _source(text: str) -> dict[str, dict[str, str]]:
    return {"doc|p1_t0": {"text": text}}


def test_hydration_uses_python_code_point_offsets_and_preserves_unicode() -> None:
    text = "prefix \U0001f430 ’quoted\nend"
    start = text.index("\U0001f430")
    pointer = HydratedTextPointer(
        source_cluster_id="doc|p1_t0",
        start_char=start,
        end_char=start + len("\U0001f430 ’quoted") - 1,
        verbatim_text="model transcription",
    )

    hydrated = hydrate_pointer_from_offsets(pointer, _source(text))

    assert hydrated is not None
    assert hydrated.verbatim_text == "\U0001f430 ’quoted"


@pytest.mark.parametrize(
    ("start_char", "end_char"),
    [(-1, 0), (0, -2), (3, 2), (0, 100), (len("abc"), len("abc"))],
)
def test_invalid_bounds_never_become_a_valid_empty_or_clamped_span(
    start_char: int,
    end_char: int,
) -> None:
    pointer = HydratedTextPointer(
        source_cluster_id="doc|p1_t0",
        start_char=start_char,
        end_char=end_char,
        verbatim_text="not present",
    )

    assert hydrate_pointer_from_offsets(pointer, _source("abc")) is None
    assert correct_and_validate_pointer(pointer, _source("abc")) is None


def test_full_source_sentinel_is_resolved_without_changing_its_start() -> None:
    text = "prefix\nbody"
    pointer = HydratedTextPointer(
        source_cluster_id="doc|p1_t0",
        start_char=7,
        end_char=-1,
        verbatim_text="model transcription",
    )

    hydrated = hydrate_pointer_from_offsets(pointer, _source(text))

    assert hydrated is not None
    assert hydrated.start_char == 7
    assert hydrated.end_char == len(text) - 1
    assert hydrated.verbatim_text == "body"


def test_final_validation_accepts_a_nonzero_start_with_resolved_sentinel() -> None:
    text = "prefix\nbody"
    pointer = HydratedTextPointer(
        source_cluster_id="doc|p1_t0",
        start_char=7,
        end_char=-1,
        verbatim_text="body",
    )

    assert pointer_source_validation_error(pointer, _source(text)) is None


def test_exact_relocation_rejects_ambiguous_occurrences() -> None:
    source = "Alpha. repeated. middle. repeated."
    pointer = HydratedTextPointer(
        source_cluster_id="doc|p1_t0",
        start_char=999,
        end_char=1000,
        verbatim_text="repeated.",
    )

    assert correct_and_validate_pointer(pointer, _source(source)) is None


def test_legacy_valid_offsets_rehydrate_provider_text() -> None:
    source = "Alpha clause."
    pointer = legacy_layerwise.HydratedTextPointer(
        source_cluster_id="doc|p1_t0",
        start_char=0,
        end_char=len(source) - 1,
        verbatim_text="Alpha clause",
    )

    repaired = legacy_layerwise.correct_and_validate_pointer(
        pointer,
        {"doc|p1_t0": {"text": source, "id": "doc|p1_t0"}},
    )

    assert repaired is not None
    assert repaired.verbatim_text == source


def test_legacy_delimiters_cannot_create_a_reversed_span() -> None:
    pointer = legacy_layerwise.HydratedTextPointer(
        source_cluster_id="doc|p1_t0",
        start_char=0,
        end_char=-1,
        verbatim_text=None,
        start_delimiter="end",
        end_delimiter="start",
    )

    with pytest.raises(ValueError, match="before Start"):
        legacy_layerwise.resolve_delimiter_pointer(
            pointer,
            {"doc|p1_t0": {"text": "start body end", "id": "doc|p1_t0"}},
        )


def test_coverage_fails_closed_for_malformed_source_records() -> None:
    pointer = HydratedTextPointer(
        source_cluster_id="doc|p1_t0",
        start_char=0,
        end_char=4,
        verbatim_text="Alpha",
    )
    root = SemanticNode(
        node_id="root",
        title="Document",
        node_type="DOCUMENT_ROOT",
        child_nodes=[
            SemanticNode(
                node_id="child",
                parent_id="root",
                title="Alpha",
                total_content_pointers=[pointer],
            )
        ],
    )

    compatibility = compute_pointer_coverage(root, {"doc|p1_t0": None})  # type: ignore[arg-type]
    terminal = compute_terminal_content_coverage(root, {"doc|p1_t0": None})  # type: ignore[arg-type]

    assert compatibility["overall"] == 1.0
    assert terminal["valid"] is False
    assert terminal["overall"] == 1.0
    assert terminal["invalid_pointers"]


def test_compatibility_coverage_does_not_count_an_invalid_pointer() -> None:
    pointer = HydratedTextPointer(
        source_cluster_id="doc|p1_t0",
        start_char=0,
        end_char=99,
        verbatim_text="Alpha",
    )
    root = SemanticNode(
        node_id="root",
        title="Document",
        node_type="DOCUMENT_ROOT",
        child_nodes=[
            SemanticNode(
                node_id="child",
                parent_id="root",
                title="Alpha",
                total_content_pointers=[pointer],
            )
        ],
    )

    coverage = compute_pointer_coverage(root, {"doc|p1_t0": {"text": "Alpha"}})

    assert coverage["per_cluster"]["doc|p1_t0"] == 0.0
    assert coverage["overall"] == 0.0


def test_repair_rehydrates_valid_offsets_despite_transcription_drift() -> None:
    text = "A  markdown \\[link\\]"
    start = 0
    pointer = HydratedTextPointer(
        source_cluster_id="doc|p1_t0",
        start_char=start,
        end_char=len(text) - 1,
        verbatim_text="A markdown [link]",
    )

    repaired = correct_and_validate_pointer(pointer, _source(text))
    assert repaired is not None
    assert repaired.verbatim_text == text
    assert pointer_source_validation_error(repaired, _source(text)) is None


def test_persisted_pointer_validation_rejects_unrehydrated_transcription() -> None:
    text = "A  markdown \\[link\\]"
    pointer = HydratedTextPointer(
        source_cluster_id="doc|p1_t0",
        start_char=0,
        end_char=len(text) - 1,
        verbatim_text="A markdown [link]",
    )

    assert pointer_source_validation_error(pointer, _source(text)) == (
        "pointer excerpt does not match authoritative source slice"
    )


def test_malformed_source_text_is_not_stringified_into_authoritative_evidence() -> None:
    pointer = HydratedTextPointer(
        source_cluster_id="doc|p1_t0",
        start_char=0,
        end_char=3,
        verbatim_text="None",
    )

    assert hydrate_pointer_from_offsets(pointer, {"doc|p1_t0": {"text": None}}) is None
    assert pointer_source_validation_error(
        pointer, {"doc|p1_t0": {"text": None}}
    ) == "pointer exceeds source bounds"


def test_handler_repair_rejects_non_text_source_instead_of_stringifying_it() -> None:
    pointer = HydratedTextPointer(
        source_cluster_id="doc|p1_t0",
        start_char=0,
        end_char=3,
        verbatim_text="None",
    )

    assert _correct_parser_pointer(pointer, {"doc|p1_t0": {"text": None}}) is None


def test_page_index_layer_skips_non_text_source_records() -> None:
    pointer = HydratedTextPointer(
        source_cluster_id="doc|p1_t0",
        start_char=0,
        end_char=-1,
        verbatim_text="provider text",
    )

    assert parse_page_index_layer(
        parent_id="root",
        parent_title="Document",
        parent_pointers=[pointer],
        parser_source_map={"doc|p1_t0": {"text": 123}},
    ) == []


def test_pointer_offsets_are_strict_integers() -> None:
    with pytest.raises(ValidationError):
        HydratedTextPointer(
            source_cluster_id="doc|p1_t0",
            start_char=True,
            end_char=0,
            verbatim_text="a",
        )


def test_malformed_provider_boolean_offsets_fail_closed() -> None:
    # The provider adapter checks raw values before constructing the strict
    # pointer model; boolean values must not be treated as character offsets.
    payload = _hydrate_llm_pointer_payload(
        {"source_cluster_id": "doc|p1_t0", "start_char": True, "end_char": 0},
        parser_source_map=_source("a"),
    )

    assert payload["source_cluster_id"] == "doc|p1_t0"
    assert payload["verbatim_text"] == ""


@pytest.mark.parametrize("field", ["start_char", "end_char"])
def test_pointer_text_does_not_treat_boolean_offsets_as_positions(field: str) -> None:
    from kg_doc_parser.workflow_ingest.layerwise_llm import _pointer_text

    pointer = {
        "source_cluster_id": "doc|p1_t0",
        "start_char": 0,
        "end_char": 0,
    }
    pointer[field] = True

    assert _pointer_text(pointer, parser_source_map=_source("secret")) == ""


def test_non_string_provider_cluster_id_cannot_match_string_source_key() -> None:
    from kg_doc_parser.workflow_ingest.layerwise_llm import _pointer_text

    assert _pointer_text(
        {
            "source_cluster_id": 123,
            "start_char": 0,
            "end_char": 0,
        },
        parser_source_map={"123": {"text": "secret"}},
    ) == ""


@pytest.mark.parametrize("verbatim_text", [True, 123, ["a"], {"text": "a"}])
def test_malformed_provider_verbatim_text_is_not_stringified(verbatim_text: object) -> None:
    payload = _hydrate_llm_pointer_payload(
        {
            "source_cluster_id": "doc|p1_t0",
            "start_char": True,
            "end_char": 0,
            "verbatim_text": verbatim_text,
        },
        parser_source_map=_source("a"),
    )

    assert payload["verbatim_text"] == ""


@pytest.mark.parametrize("offset", [True, 1.5])
def test_pointer_bookkeeping_does_not_truncate_non_integer_offsets(offset: object) -> None:
    from kg_doc_parser.workflow_ingest.layerwise_llm import _offset_value

    assert _offset_value(offset, 99) == 99


def test_legacy_sentinel_is_not_shifted_by_a_nonzero_start() -> None:
    pointer = legacy_layerwise.HydratedTextPointer(
        source_cluster_id="doc|p1_t0",
        start_char=2,
        end_char=-1,
        verbatim_text="cdef",
    )

    assert pointer.end_char == -1


def test_legacy_repair_does_not_move_text_to_another_cluster() -> None:
    pointer = legacy_layerwise.HydratedTextPointer(
        source_cluster_id="doc|p1_t0",
        start_char=999,
        end_char=1000,
        verbatim_text="needle",
    )

    assert legacy_layerwise.correct_and_validate_pointer(
        pointer,
        {
            "doc|p1_t0": {"text": "unrelated"},
            "doc|p1_t1": {"text": "needle"},
        },
    ) is None
