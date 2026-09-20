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
    assert "pypy3.11-v7.3.20-linux64.tar.bz2" in workflow
    assert "1410db3a7ae47603e2b7cbfd7ff6390b891b2e041c9eb4f1599f333677bccb3e" in workflow


def test_parser_ci_uses_the_declared_kogwistar_revision() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    revision = metadata["tool"]["poetry"]["dependencies"]["kogwistar"]["rev"]
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert f"ref: {revision}" in workflow

    pypy_workflow = (ROOT / ".github" / "workflows" / "pypy-311-experimental.yml").read_text(
        encoding="utf-8"
    )
    assert f"ref: {revision}" in pypy_workflow

    exported_requirements = (ROOT / "req.txt").read_text(encoding="utf-8")
    assert f"kogwistar.git@{revision}" in exported_requirements


def test_parser_mcp_dependency_is_official_sdk_only() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = metadata["tool"]["poetry"]["dependencies"]
    assert dependencies["mcp"] == "^1.27.0"
    assert "fastmcp" not in metadata["tool"]["poetry"].get("dependencies", {})

    exported_requirements = (ROOT / "req.txt").read_text(encoding="utf-8").lower()
    assert "fastmcp" not in exported_requirements
    assert "mcp==1.30.0" in exported_requirements


def test_pypy_311_parser_profile_excludes_native_optional_dependencies() -> None:
    requirements = "\n".join(
        line.split("#", 1)[0]
        for line in (ROOT / "requirements-pypy-3.11-experimental.txt")
        .read_text(encoding="utf-8")
        .lower()
        .splitlines()
    )

    for forbidden in (
        "numpy",
        "chromadb",
        "pgvector",
        "torch",
        "pikepdf",
    ):
        assert forbidden not in requirements
