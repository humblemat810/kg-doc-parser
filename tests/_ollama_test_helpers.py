from __future__ import annotations

from typing import Any

import requests


def ollama_available(base_url: str, model_name: str | None = None) -> tuple[bool, str | None]:
    """Check endpoint reachability and, when requested, model availability."""

    try:
        response = requests.get(f"{base_url}/api/version", timeout=2.0)
    except Exception as exc:  # noqa: BLE001 - unavailable test service is skippable.
        return False, str(exc)
    if response.status_code >= 400:
        return False, f"unexpected status {response.status_code}"
    if model_name is None:
        return True, None

    try:
        tags_response = requests.get(f"{base_url}/api/tags", timeout=2.0)
        tags_response.raise_for_status()
        payload: Any = tags_response.json()
        tags = payload.get("models", []) if isinstance(payload, dict) else []
    except Exception as exc:  # noqa: BLE001 - unavailable test service is skippable.
        return False, f"model inventory unavailable: {exc}"
    names = {
        str(item.get("name"))
        for item in tags
        if isinstance(item, dict) and item.get("name")
    }
    requested_names = {model_name, f"{model_name}:latest"}
    if not names.intersection(requested_names):
        return False, f"model {model_name!r} is not present in /api/tags"
    return True, None
