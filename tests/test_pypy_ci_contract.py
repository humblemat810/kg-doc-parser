from __future__ import annotations

import tomllib
from pathlib import Path

import pytest


pytestmark = pytest.mark.ci
ROOT = Path(__file__).resolve().parents[1]


def test_pypy_311_parser_workflow_is_pinned_and_nonblocking() -> None:
    workflow = (ROOT / ".github" / "workflows" / "pypy-311-experimental.yml").read_text(
        encoding="utf-8"
    )

    assert "continue-on-error: true" in workflow
    assert "uses: actions/setup-python@v7" in workflow
    assert "python-version: pypy-3.11-v7.3.20" in workflow
    assert "cache: pip" in workflow
    assert "Install official PyPy 3.11 release" not in workflow
    assert "pypy_url" not in workflow
    assert "pypy_sha256" not in workflow


def test_parser_ci_uses_the_declared_kogwistar_revision() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    revision = metadata["tool"]["poetry"]["dependencies"]["kogwistar"]["rev"]
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert f"ref: {revision}" in workflow

    exported_requirements = (ROOT / "req.txt").read_text(encoding="utf-8")
    assert f"kogwistar.git@{revision}" in exported_requirements


def test_parser_main_ci_matrix_uses_hosted_cpython_and_pypy_runtimes() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "label: cpython312" in workflow
    assert "label: cpython313" in workflow
    assert "label: cpython314" in workflow
    assert "python-version: pypy-3.11-v7.3.20" in workflow
    assert "uses: actions/setup-python@v7" in workflow
    assert "Install PyPy 3.11 Python-authority dependencies" in workflow
    assert "Run deterministic PyPy 3.11 CI tests" in workflow


def test_parser_mcp_dependency_is_official_sdk_only() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = metadata["tool"]["poetry"]["dependencies"]
    assert dependencies["mcp"] == "^1.27.0"
    assert "fastmcp" not in metadata["tool"]["poetry"].get("dependencies", {})

    exported_requirements = (ROOT / "req.txt").read_text(encoding="utf-8").lower()
    assert "fastmcp" not in exported_requirements
    assert "mcp==1.30.0" in exported_requirements


def test_pypy_311_parser_profile_excludes_native_optional_dependencies() -> None:
    requirements_path = ROOT / "requirements-pypy-3.11-experimental.txt"
    requirements = "\n".join(
        line.split("#", 1)[0].strip()
        for line in requirements_path
        .read_text(encoding="utf-8")
        .lower()
        .splitlines()
    )

    assert "diskcache>=5.6,<6" in requirements
    assert "joblib" not in requirements

    for forbidden in (
        "numpy",
        "chromadb",
        "pgvector",
        "torch",
        "pikepdf",
    ):
        assert forbidden not in requirements


def test_parser_cache_selects_a_provider_without_hard_coding_joblib() -> None:
    source = (ROOT / "kg_doc_parser" / "semantic_document_splitting_layerwise_edits.py").read_text(
        encoding="utf-8"
    )

    assert "KG_DOC_PARSER_CACHE_DIR" in source
    assert "KG_DOC_PARSER_CACHE_BACKEND" in source
    assert "KG_DOC_PARSER_JOBLIB_CACHE_DIR" in source
    assert "Memory(location=_PARSER_CACHE_DIR, backend=_PARSER_CACHE_BACKEND)" in source
    assert "def memory_cached(" in source
