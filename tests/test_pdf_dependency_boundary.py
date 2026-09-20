from __future__ import annotations

import importlib
import sys


def test_pdf_native_dependency_is_lazy(monkeypatch) -> None:
    module_name = "kg_doc_parser.pdf2png"
    previous = sys.modules.pop(module_name, None)
    monkeypatch.setitem(sys.modules, "pikepdf", None)
    try:
        module = importlib.import_module(module_name)
        assert hasattr(module, "split_pdf_with_pikepdf")
    finally:
        sys.modules.pop(module_name, None)
        if previous is not None:
            sys.modules[module_name] = previous
