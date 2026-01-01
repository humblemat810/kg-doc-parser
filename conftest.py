import pytest
@pytest.fixture
def gemini_key():
    import dotenv
    dotenv.load_dotenv()
    import os
    return os.environ.get('GOOGLE_API_KEY')