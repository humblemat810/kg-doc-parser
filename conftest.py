from pathlib import Path
import sys

_VENDORED_KOGWISTAR = Path(__file__).resolve().parent / "kogwistar"
if str(_VENDORED_KOGWISTAR) not in sys.path:
    sys.path.insert(0, str(_VENDORED_KOGWISTAR))

import pytest


def pytest_collection_modifyitems(items):
    """Keep model/provider quality checks out of deterministic contract CI."""
    for item in items:
        marker_names = {marker.name for marker in item.iter_markers()}
        if marker_names.intersection({"ci", "ci_full"}) and marker_names.intersection(
            {"manual", "llm_real", "requires_ollama"}
        ):
            raise pytest.UsageError(
                f"{item.nodeid} mixes deterministic CI and real-provider markers; "
                "remove ci/ci_full or replace the provider with a fake/injected model"
            )


@pytest.fixture
def gemini_key():
    import dotenv
    import os

    dotenv.load_dotenv()
    return os.environ.get("GOOGLE_API_KEY")
