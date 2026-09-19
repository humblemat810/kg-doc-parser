from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

pytestmark = [pytest.mark.ci]


def test_package_supports_python_312_and_guards_pikepdf_on_pypy() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["tool"]["poetry"]["dependencies"]

    assert dependencies["python"] == "^3.12"
    assert "platform_python_implementation != 'PyPy'" in dependencies["pikepdf"]["markers"]
