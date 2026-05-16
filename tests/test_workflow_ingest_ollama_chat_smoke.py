from __future__ import annotations

import os

import pytest
import requests


def _ollama_base_url() -> str:
    return (
        os.getenv("KG_DOC_PARSER_BASE_URL")
        or os.getenv("OLLAMA_BASE_URL")
        or os.getenv("OLLAMA_HOST")
        or "http://127.0.0.1:11434"
    )


def _ollama_available(base_url: str) -> tuple[bool, str | None]:
    try:
        response = requests.get(f"{base_url}/api/version", timeout=2.0)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    if response.status_code >= 400:
        return False, f"unexpected status {response.status_code}"
    return True, None


@pytest.mark.ci_full
def test_chat_ollama_simple_invoke_smoke() -> None:
    pytest.importorskip("langchain_ollama")

    from langchain_core.messages import HumanMessage
    from langchain_ollama import ChatOllama

    model_name = os.getenv("KG_DOC_PARSER_MODEL", "gemma4:e2b")
    base_url = _ollama_base_url()
    ok, reason = _ollama_available(base_url)
    if not ok:
        pytest.skip(f"ollama unavailable at {base_url}: {reason}")

    model = ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=0.0,
        num_predict=280,
        sync_client_kwargs={"timeout": 15.0},
    )

    response = model.invoke(
        [
            HumanMessage(
                content="Reply with exactly one word: pong."
            )
        ]
    )

    content = str(getattr(response, "content", "")).strip()
    if not content and isinstance(response, dict):
        content = str(response.get("content", "")).strip()

    assert content, "expected a non-empty response from ChatOllama"
