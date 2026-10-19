"""Focused tests for OCR persisted-run identity and invocation lineage."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kg_doc_parser.workflow_ingest import ocr_runtime_state as ocr_runtime_state_module
from kg_doc_parser.workflow_ingest.ocr_runtime import _semantic_parse_generation
from kg_doc_parser.workflow_ingest.ocr_runtime_lineage import (
    validate_root_invocation_lineage,
)

pytestmark = [pytest.mark.workflow]


@pytest.mark.ci
def test_ocr_root_resume_rejects_missing_or_changed_immutable_identity() -> None:
    expected = {
        "ocr_root_state_version": "ocr-root-v3",
        "document_id": "doc",
        "source_fingerprint": "source-a",
        "source_namespace": "",
        "ocr_configuration_fingerprints": ["ocr-a"],
        "grouping_policy_version": "group-a",
        "parse_generation": "semantic-v1",
    }

    ocr_runtime_state_module.validate_root_resume_identity(
        expected,
        document_id="doc",
        source_fingerprint="source-a",
        source_namespace="",
        configuration_fingerprints=("ocr-a",),
        grouping_policy_version="group-a",
        parse_generation="semantic-v1",
        require_match=True,
    )
    with pytest.raises(RuntimeError, match="immutable identity fields"):
        ocr_runtime_state_module.validate_root_resume_identity(
            {"document_id": "doc"},
            document_id="doc",
            source_fingerprint="source-a",
            source_namespace="",
            configuration_fingerprints=("ocr-a",),
            grouping_policy_version="group-a",
            parse_generation="semantic-v1",
            require_match=True,
        )
    changed = dict(expected, source_fingerprint="source-b")
    with pytest.raises(RuntimeError, match="identity mismatch"):
        ocr_runtime_state_module.validate_root_resume_identity(
            changed,
            document_id="doc",
            source_fingerprint="source-a",
            source_namespace="",
            configuration_fingerprints=("ocr-a",),
            grouping_policy_version="group-a",
            parse_generation="semantic-v1",
            require_match=True,
        )
    changed_namespace = dict(expected, source_namespace="tenant-b")
    with pytest.raises(RuntimeError, match="identity mismatch"):
        ocr_runtime_state_module.validate_root_resume_identity(
            changed_namespace,
            document_id="doc",
            source_fingerprint="source-a",
            source_namespace="",
            configuration_fingerprints=("ocr-a",),
            grouping_policy_version="group-a",
            parse_generation="semantic-v1",
            require_match=True,
        )


@pytest.mark.ci
def test_ocr_root_resume_rejects_a_different_parse_generation() -> None:
    state = {
        "ocr_root_state_version": "ocr-root-v3",
        "document_id": "doc",
        "source_fingerprint": "source-a",
        "source_namespace": "",
        "ocr_configuration_fingerprints": ["ocr-a"],
        "grouping_policy_version": "group-a",
        "parse_generation": "semantic-v1",
    }

    with pytest.raises(RuntimeError, match="identity mismatch"):
        ocr_runtime_state_module.validate_root_resume_identity(
            state,
            document_id="doc",
            source_fingerprint="source-a",
            source_namespace="",
            configuration_fingerprints=("ocr-a",),
            grouping_policy_version="group-a",
            parse_generation="semantic-recovery-v1",
            require_match=True,
        )
    assert _semantic_parse_generation(0) == "semantic-v1"
    assert _semantic_parse_generation(1) == "semantic-recovery-v1"


@pytest.mark.ci
def test_ocr_root_lineage_rejects_missing_or_duplicate_child_edges() -> None:
    root_run_id = "ocr|lineage-contract"
    edges = [
        SimpleNamespace(
            metadata={
                "relation": "wf_invoked",
                "parent_run_id": root_run_id,
                "workflow_id": "kg_doc_parser.ocr_pages.v1",
                "invocation_key": "ocr-pages-v1",
                "child_run_id": f"{root_run_id}|ocr-pages",
            }
        ),
        SimpleNamespace(
            metadata={
                "relation": "wf_invoked",
                "parent_run_id": root_run_id,
                "workflow_id": "kg_doc_parser.ocr_pages.v1",
                "invocation_key": "ocr-pages-v1",
                "child_run_id": f"{root_run_id}|ocr-pages",
            }
        ),
    ]
    engine = SimpleNamespace(read=SimpleNamespace(get_edges=lambda **_: edges))

    with pytest.raises(RuntimeError, match="persisted invocation lineage is invalid"):
        validate_root_invocation_lineage(
            engine,
            root_run_id=root_run_id,
            recovery_generation=0,
        )
