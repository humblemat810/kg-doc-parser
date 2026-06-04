from pathlib import Path
import sys

_VENDORED_KOGWISTAR = Path(__file__).resolve().parent / "kogwistar"
if str(_VENDORED_KOGWISTAR) not in sys.path:
    sys.path.insert(0, str(_VENDORED_KOGWISTAR))

import pytest


@pytest.fixture
def gemini_key():
    import dotenv
    import os

    dotenv.load_dotenv()
    return os.environ.get("GOOGLE_API_KEY")
