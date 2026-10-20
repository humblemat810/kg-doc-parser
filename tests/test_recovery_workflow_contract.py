from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.ci
ROOT = Path(__file__).resolve().parents[1]


def test_ci_workflow_pytest_node_ids_resolve_to_test_functions() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    selectors = re.findall(
        r"(tests/[A-Za-z0-9_./]+\.py)::(test_[A-Za-z0-9_]+)", workflow
    )

    assert selectors
    for relative_path, test_name in selectors:
        test_file = ROOT / relative_path
        assert test_file.is_file(), f"workflow test file does not exist: {relative_path}"
        module = ast.parse(test_file.read_text(encoding="utf-8"))
        test_functions = {
            node.name
            for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert test_name in test_functions, (
            f"workflow node id does not resolve: {relative_path}::{test_name}"
        )
