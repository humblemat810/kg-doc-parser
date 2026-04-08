from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_kogwistar_fake_backend():
    helper_path = (
        Path(__file__).resolve().parents[1]
        / "kogwistar"
        / "tests"
        / "_helpers"
        / "fake_backend.py"
    )
    if not helper_path.exists():
        raise FileNotFoundError(f"kogwistar fake backend helper not found: {helper_path}")

    module_name = "kogwistar_tests_fake_backend"
    spec = importlib.util.spec_from_file_location(module_name, helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load spec for {helper_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.build_fake_backend
