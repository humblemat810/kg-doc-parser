from __future__ import annotations

from typing import Literal, Protocol, TypeAlias, TypedDict, cast

from kogwistar.llm_tasks.providers import (
    StructuredModelLike,
    SupportsStructuredOutput,
)

# Keep the historical parser names as aliases, but use the core contracts as
# the single source of truth for provider/schema compatibility.
StructuredSchema: TypeAlias = StructuredModelLike
StructuredOutputModel: TypeAlias = SupportsStructuredOutput


class _StructuredOutputOptions(TypedDict, total=False):
    include_raw: bool
    method: Literal["function_calling", "json_mode", "json_schema"]


class StructuredOutputRunnable(Protocol):
    """Runnable returned by a structured-output model adapter."""

    steps: list[object]

    def invoke(
        self,
        messages: object,
        config: object | None = None,
        **kwargs: object,
    ) -> dict[str, object]: ...


def build_structured_output_runnable(
    model: StructuredOutputModel,
    schema: type[StructuredSchema],
    *,
    include_raw: bool = True,
    prefer_json_schema: bool = True,
) -> StructuredOutputRunnable:
    """Build a structured-output runnable with strict-schema-first fallback."""
    attempts: list[_StructuredOutputOptions] = []
    if prefer_json_schema:
        attempts.append({"include_raw": include_raw, "method": "json_schema"})
    attempts.append({"include_raw": include_raw, "method": "function_calling"})
    attempts.append({"include_raw": include_raw})

    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            # LangChain returns a runnable with ``steps``; the dependency-light
            # core protocol intentionally does not require that implementation
            # detail. Keep the cast at this parser-only compatibility boundary.
            return cast(
                StructuredOutputRunnable,
                model.with_structured_output(schema, **kwargs),
            )
        except (TypeError, ValueError) as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise TypeError("with_structured_output is unavailable on this model")
