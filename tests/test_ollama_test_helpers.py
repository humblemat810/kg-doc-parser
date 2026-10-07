from __future__ import annotations

from types import SimpleNamespace

import requests
from _ollama_test_helpers import ollama_available


def test_ollama_available_requires_requested_model(monkeypatch) -> None:
    def fake_get(url: str, *, timeout: float):
        del timeout
        if url.endswith("/api/version"):
            return SimpleNamespace(status_code=200)
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"models": [{"name": "other:latest"}]},
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(requests, "get", fake_get)

    available, reason = ollama_available("http://ollama", "required:latest")

    assert available is False
    assert reason == "model 'required:latest' is not present in /api/tags"


def test_ollama_available_accepts_requested_model(monkeypatch) -> None:
    def fake_get(url: str, *, timeout: float):
        del timeout
        if url.endswith("/api/version"):
            return SimpleNamespace(status_code=200)
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"models": [{"name": "required:latest"}]},
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(requests, "get", fake_get)

    assert ollama_available("http://ollama", "required:latest") == (True, None)


def test_ollama_available_reports_transport_failure(monkeypatch) -> None:
    def fake_get(url: str, *, timeout: float):
        del url, timeout
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "get", fake_get)

    available, reason = ollama_available("http://ollama")

    assert available is False
    assert reason == "offline"
