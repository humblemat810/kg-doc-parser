from __future__ import annotations

from typing import Protocol


class StructuredOutputRunnable(Protocol):
    """Runnable returned by a structured-output model adapter."""

    steps: list[object]

    def invoke(self, *args: object, **kwargs: object) -> object: ...


class StructuredOutputModel(Protocol):
    """Minimum model surface required by the parser's structured-output path."""

    def with_structured_output(
        self,
        schema: object,
        **kwargs: object,
    ) -> StructuredOutputRunnable: ...


def build_structured_output_runnable(
    model: StructuredOutputModel,
    schema: object,
    *,
    include_raw: bool = True,
    prefer_json_schema: bool = True,
) -> StructuredOutputRunnable:
    """Build a structured-output runnable with strict-schema-first fallback."""
    attempts: list[dict[str, object]] = []
    if prefer_json_schema:
        attempts.append({"include_raw": include_raw, "method": "json_schema"})
    attempts.append({"include_raw": include_raw, "method": "function_calling"})
    attempts.append({"include_raw": include_raw})

    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            return model.with_structured_output(schema, **kwargs)
        except (TypeError, ValueError) as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise TypeError("with_structured_output is unavailable on this model")
