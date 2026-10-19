"""Typed structured-provider invocation used by layerwise parsing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel

from kg_doc_parser.llm_structured_output import build_structured_output_runnable

from .providers import (
    ProviderDiagnostics,
    ProviderDiagnosticsSink,
    SupportsStructuredOutput,
    invoke_with_timeout,
)
from .serialization import ModelDumpLike

TStructuredModel = TypeVar("TStructuredModel", bound=BaseModel)


def structured_invoke(
    model: SupportsStructuredOutput,
    schema: type[TStructuredModel],
    messages: Sequence[tuple[str, str]],
    *,
    timeout_seconds: float = 120.0,
    diagnostics: ProviderDiagnostics | None = None,
    diagnostics_sink: ProviderDiagnosticsSink | None = None,
    operation: str = "parser_structured_call",
    max_in_flight: int = 1,
    attempt_index: int = 1,
    call_role: str | None = None,
    strategy: str | None = None,
) -> TStructuredModel:
    structured = build_structured_output_runnable(model, schema, include_raw=True)
    call_diagnostics = diagnostics if diagnostics is not None else {}
    call_diagnostics.setdefault("operation", operation)
    call_diagnostics.setdefault("timeout_seconds", timeout_seconds)
    try:
        try:
            response = invoke_with_timeout(
                lambda: structured.invoke(list(messages)),
                timeout_seconds=timeout_seconds,
                diagnostics=call_diagnostics,
                operation=operation,
                max_in_flight=max_in_flight,
                attempt_index=attempt_index,
                call_role=call_role,
                strategy=strategy,
            )
        except Exception as exc:
            call_diagnostics.setdefault(
                "failure_type",
                "timeout"
                if isinstance(exc, TimeoutError)
                else "in_flight_limit"
                if isinstance(exc, RuntimeError) and "in-flight call limit" in str(exc)
                else "transport/provider_exception",
            )
            call_diagnostics["error_type"] = type(exc).__name__
            call_diagnostics["success"] = False
            raise
        try:
            if isinstance(response, dict):
                parsed = response.get("parsed")
                if parsed is not None:
                    if isinstance(parsed, schema):
                        result = parsed
                    elif isinstance(parsed, ModelDumpLike):
                        # Trusted internal DTOs may be narrowed to the requested schema;
                        # raw provider mappings remain strictly validated as-is.
                        payload = parsed.model_dump()
                        if isinstance(payload, dict):
                            payload = {
                                field_name: payload[field_name]
                                for field_name in schema.model_fields
                                if field_name in payload
                            }
                        result = schema.model_validate(payload)
                    else:
                        result = schema.model_validate(parsed)
                else:
                    parsing_error = response.get("parsing_error")
                    if parsing_error is not None:
                        raise ValueError(str(parsing_error))
                    result = schema.model_validate(response)
            elif isinstance(response, schema):
                result = response
            elif isinstance(response, ModelDumpLike):
                result = schema.model_validate(response.model_dump())
            else:
                result = schema.model_validate(response)
        except Exception as exc:
            call_diagnostics.update(
                {
                    "success": False,
                    "failure_type": "structured_output_parse_failure",
                    "error_type": type(exc).__name__,
                }
            )
            raise
        call_diagnostics["success"] = True
        return result
    finally:
        if diagnostics_sink is not None:
            diagnostics_sink(dict(call_diagnostics))
