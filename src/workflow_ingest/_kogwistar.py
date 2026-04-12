from __future__ import annotations

from importlib import import_module


def ensure_kogwistar_on_path() -> None:
    """Compatibility shim: require a normal kogwistar installation/import.

    We intentionally do not mutate ``sys.path`` here. The repo may contain a sibling
    checkout for inspection, but runtime imports should resolve through the installed
    package environment, including editable installs.
    """
    import_module("kogwistar")
