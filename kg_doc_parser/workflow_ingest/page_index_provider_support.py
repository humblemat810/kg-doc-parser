"""Provider call accounting shared by PageIndex orchestration stages."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .page_index_contracts import PageIndexDiagnostics


@runtime_checkable
class ProviderCallCounter(Protocol):
    """Optional callback accounting exposed by provider telemetry sinks."""

    recorded_call_count: int | None


@runtime_checkable
class ProviderCallRecorder(ProviderCallCounter, Protocol):
    """Optional callback hook for adapters bypassing LangChain callbacks."""

    def record_untracked_call(self, call_key: str, /) -> None: ...


def increment_diagnostic(
    diagnostics: PageIndexDiagnostics, key: str, amount: int = 1
) -> None:
    current = diagnostics.get(key)
    current_count = current if type(current) is int and current >= 0 else 0
    diagnostics[key] = current_count + amount


def callback_call_counts(callbacks: list[object] | None) -> dict[int, int | None]:
    """Snapshot callback accounting before a provider invocation."""

    counts: dict[int, int | None] = {}
    for callback in callbacks or []:
        counter = callback if isinstance(callback, ProviderCallCounter) else None
        counts[id(callback)] = counter.recorded_call_count if counter is not None else None
    return counts


def notify_untracked_provider_call(
    callbacks: list[object] | None,
    before_counts: dict[int, int | None],
    call_key: str,
) -> None:
    """Account for successful adapters that do not emit LangChain callbacks."""

    for callback in callbacks or []:
        recorder = callback if isinstance(callback, ProviderCallRecorder) else None
        if recorder is None:
            continue
        before = before_counts.get(id(callback))
        after = recorder.recorded_call_count
        if before is not None and after == before:
            recorder.record_untracked_call(call_key)
